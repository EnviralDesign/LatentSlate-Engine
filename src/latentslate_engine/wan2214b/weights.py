from __future__ import annotations

import ctypes
import json
import os
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import comfy_kitchen as ck
import torch
import torch.nn.functional as F
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout
from safetensors import safe_open

FP8_LAYOUT = "TensorCoreFP8Layout"
LIVE_ATTENTION_BLOCKS = frozenset({0, 1, 2, *range(10, 24)})
MATERIALIZED_PATCH_COUNT = 263
MATERIALIZED_REMAP_INTERVAL = 16
LIVE_PATCH_ORDER = tuple(
    f"blocks.{block}.{attention}.{projection}"
    for block in range(40)
    for attention in ("self_attn", "cross_attn")
    for projection in ("q", "k", "v", "o")
    if block in LIVE_ATTENTION_BLOCKS
) + ("blocks.24.cross_attn.k",)
NEXT_LIVE_PATCH = dict(pairwise(LIVE_PATCH_ORDER))


def _trim_process_working_set() -> None:
    if os.name == "nt":
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.EmptyWorkingSet(process)


@dataclass(frozen=True)
class ArtifactIdentity:
    path: str
    size: int
    mtime_ns: int

    @classmethod
    def from_path(cls, path: str | Path) -> ArtifactIdentity:
        resolved = Path(path).resolve(strict=True)
        stat = resolved.stat()
        return cls(str(resolved), stat.st_size, stat.st_mtime_ns)


class TensorStore:
    """Open safetensors mapping whose tensors stay backed by the source file."""

    def __init__(self, path: str | Path):
        self.identity = ArtifactIdentity.from_path(path)
        self._mapping = safe_open(self.identity.path, framework="pt", device="cpu")
        self.keys = frozenset(self._mapping.keys())

    def tensor(self, key: str) -> torch.Tensor:
        if self._mapping is None:
            raise RuntimeError(f"tensor store is closed: {self.identity.path}")
        return self._mapping.get_tensor(key)

    def close(self) -> None:
        self._mapping = None

    def reopen(self) -> None:
        self.close()
        self._mapping = safe_open(self.identity.path, framework="pt", device="cpu")


class WanWeights:
    def __init__(
        self,
        checkpoint: str | Path,
        lora: str | Path | None = None,
        *,
        lora_strength: float = 1.0,
        native_fp8: bool = True,
    ):
        self.base = TensorStore(checkpoint)
        self.lora = TensorStore(lora) if lora is not None else None
        self.lora_strength = lora_strength
        self.native_fp8 = native_fp8
        self._patched_weights: dict[str, QuantizedTensor] = {}
        self._active_weights: dict[str, QuantizedTensor] = {}
        self._active_qk_norms: dict[str, torch.Tensor] = {}
        self._active_device: torch.device | None = None
        self._base_reopened = False
        self._materialized_since_reopen = 0
        self._reopen_before_next_access = False
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._prefetched_live: (
            tuple[
                str,
                QuantizedTensor,
                torch.Tensor,
                torch.Tensor,
                float,
                tuple[torch.Tensor, ...],
            ]
            | None
        ) = None
        self._validate_lora()

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            self.base.identity,
            self.lora.identity if self.lora else None,
            self.lora_strength,
            self.native_fp8,
        )

    def _validate_lora(self) -> None:
        if self.lora is None:
            return
        targets = [
            key[: -len(".lora_up.weight")]
            for key in self.lora.keys
            if key.endswith(".lora_up.weight")
        ]
        if len(targets) != 400:
            raise ValueError(
                f"canonical Wan LoRA must contain 400 targets, found {len(targets)}"
            )
        for target in targets:
            down = f"{target}.lora_down.weight"
            alpha = f"{target}.alpha"
            base = f"{target.removeprefix('diffusion_model.')}.weight"
            if (
                down not in self.lora.keys
                or alpha not in self.lora.keys
                or base not in self.base.keys
            ):
                raise ValueError(
                    f"incomplete or unmapped canonical Wan LoRA target: {target}"
                )
            if int(self.lora.tensor(alpha).item()) != 8:
                raise ValueError(f"unexpected canonical Wan LoRA alpha at {target}")

    def _plain(
        self, key: str, device: torch.device, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        value = self._active_qk_norms.get(key)
        if value is None:
            value = self.base.tensor(key).to(device=device, non_blocking=True)
        return value.to(dtype=dtype) if dtype is not None else value

    def activate(self, device: torch.device) -> None:
        if self._active_device == device:
            return
        active: dict[str, QuantizedTensor] = {}
        for key, value in self._patched_weights.items():
            qdata = value._qdata.to(device=device, non_blocking=True)
            params = TensorCoreFP8Layout.Params(
                scale=value._params.scale.to(device=device, non_blocking=True),
                orig_dtype=value._params.orig_dtype,
                orig_shape=value._params.orig_shape,
            )
            active[key] = QuantizedTensor(qdata, FP8_LAYOUT, params)
        qk_norms = {
            key: self.base.tensor(key).to(device=device, non_blocking=True)
            for key in self.base.keys
            if key.endswith((".norm_q.weight", ".norm_k.weight"))
        }
        torch.cuda.current_stream(device).synchronize()
        _trim_process_working_set()
        self._active_weights = active
        self._active_qk_norms = qk_norms
        self._active_device = device
        self._prefetch_stream = torch.cuda.Stream(device=device)

    def deactivate(self) -> None:
        if self._active_device is None:
            return
        if self._prefetch_stream is not None:
            self._prefetch_stream.synchronize()
        self._prefetched_live = None
        self._prefetch_stream = None
        torch.cuda.current_stream(self._active_device).synchronize()
        self._active_weights.clear()
        self._active_qk_norms.clear()
        self._active_device = None
        if (
            not self._base_reopened
            and len(self._patched_weights) == MATERIALIZED_PATCH_COUNT
        ):
            self.base.reopen()
            self._base_reopened = True
            _trim_process_working_set()

    def _prepare_base_access(self) -> None:
        if not self._reopen_before_next_access:
            return
        self.base.reopen()
        self._materialized_since_reopen = 0
        self._reopen_before_next_access = False
        _trim_process_working_set()

    @staticmethod
    def _cpu_copy(value: QuantizedTensor) -> QuantizedTensor:
        qdata = torch.empty_like(value._qdata, device="cpu")
        qdata.copy_(value._qdata, non_blocking=True)
        scale = torch.empty_like(value._params.scale, device="cpu")
        scale.copy_(value._params.scale, non_blocking=True)
        params = TensorCoreFP8Layout.Params(
            scale=scale,
            orig_dtype=value._params.orig_dtype,
            orig_shape=value._params.orig_shape,
        )
        return QuantizedTensor(qdata, FP8_LAYOUT, params)

    def _quantized_weight(
        self,
        prefix: str,
        device: torch.device,
        compute_dtype: torch.dtype,
    ) -> QuantizedTensor:
        weight_key = f"{prefix}.weight"
        scale_key = f"{prefix}.scale_weight"
        qdata = self._plain(weight_key, device)
        scale = self._plain(scale_key, device, torch.float32)
        params = TensorCoreFP8Layout.Params(
            scale=scale,
            orig_dtype=compute_dtype,
            orig_shape=tuple(qdata.shape),
        )
        quantized = QuantizedTensor(qdata, FP8_LAYOUT, params)

        return quantized

    @staticmethod
    def _is_live_patch(prefix: str) -> bool:
        parts = prefix.split(".")
        block = int(parts[1])
        return (
            parts[2] in {"self_attn", "cross_attn"} and block in LIVE_ATTENTION_BLOCKS
        ) or prefix == "blocks.24.cross_attn.k"

    def _load_live_patch(
        self,
        prefix: str,
        device: torch.device,
        compute_dtype: torch.dtype,
    ) -> tuple[
        QuantizedTensor,
        torch.Tensor,
        torch.Tensor,
        float,
        tuple[torch.Tensor, ...],
    ]:
        lora_prefix = f"diffusion_model.{prefix}"
        qdata_cpu = self.base.tensor(f"{prefix}.weight")
        scale_cpu = self.base.tensor(f"{prefix}.scale_weight")
        up_cpu = self.lora.tensor(f"{lora_prefix}.lora_up.weight")
        down_cpu = self.lora.tensor(f"{lora_prefix}.lora_down.weight")
        qdata = qdata_cpu.to(device=device, non_blocking=True)
        scale = scale_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        params = TensorCoreFP8Layout.Params(
            scale=scale,
            orig_dtype=compute_dtype,
            orig_shape=tuple(qdata.shape),
        )
        base = QuantizedTensor(qdata, FP8_LAYOUT, params)
        up = up_cpu.to(device=device, dtype=compute_dtype, non_blocking=True)
        down = down_cpu.to(device=device, dtype=compute_dtype, non_blocking=True)
        alpha = float(self.lora.tensor(f"{lora_prefix}.alpha").item()) / down.shape[0]
        return base, up, down, alpha, (qdata_cpu, scale_cpu, up_cpu, down_cpu)

    def _consume_live_patch(
        self,
        prefix: str,
        device: torch.device,
        compute_dtype: torch.dtype,
    ) -> tuple[
        QuantizedTensor,
        torch.Tensor,
        torch.Tensor,
        float,
        tuple[torch.Tensor, ...],
    ]:
        if self._prefetched_live is not None and self._prefetched_live[0] == prefix:
            current = torch.cuda.current_stream(device)
            current.wait_stream(self._prefetch_stream)
            _, base, up, down, alpha, sources = self._prefetched_live
            base._qdata.record_stream(current)
            base._params.scale.record_stream(current)
            up.record_stream(current)
            down.record_stream(current)
            self._prefetched_live = None
        else:
            base, up, down, alpha, sources = self._load_live_patch(
                prefix, device, compute_dtype
            )
        next_prefix = NEXT_LIVE_PATCH.get(prefix)
        if next_prefix is not None and self._prefetch_stream is not None:
            with torch.cuda.stream(self._prefetch_stream):
                values = self._load_live_patch(next_prefix, device, compute_dtype)
            self._prefetched_live = (next_prefix, *values)
        return base, up, down, alpha, sources

    @staticmethod
    def _patch_seed(prefix: str) -> int:
        crc = 0xFFFFFFFF
        for byte in f"diffusion_model.{prefix}.weight".encode():
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ 0xEDB88320 if crc & 1 else crc >> 1
        return crc ^ 0xFFFFFFFF

    def _patched_weight(
        self,
        prefix: str,
        device: torch.device,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor | QuantizedTensor | None:
        lora_prefix = f"diffusion_model.{prefix}"
        up_key = f"{lora_prefix}.lora_up.weight"
        if self.lora is None or up_key not in self.lora.keys:
            return None
        keep_live = self._is_live_patch(prefix)
        if not keep_live:
            cached = self._active_weights.get(prefix)
            if cached is not None:
                return cached

        if keep_live:
            base, up, down, alpha, _sources = self._consume_live_patch(
                prefix, device, compute_dtype
            )
        else:
            down_key = f"{lora_prefix}.lora_down.weight"
            alpha_key = f"{lora_prefix}.alpha"
            base = self._quantized_weight(prefix, device, compute_dtype)
            up = self.lora.tensor(up_key).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            down = self.lora.tensor(down_key).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            alpha = float(self.lora.tensor(alpha_key).item()) / down.shape[0]
        weight = base.dequantize()
        delta = torch.mm(up.flatten(start_dim=1), down.flatten(start_dim=1)).reshape(
            weight.shape
        )
        weight.add_(((self.lora_strength * alpha) * delta).to(weight.dtype))
        if keep_live:
            return weight
        scale = (
            torch.amax(weight.abs()).to(torch.float32)
            / torch.finfo(base._qdata.dtype).max
        )
        dtype_info = torch.finfo(weight.dtype)
        scale = 1.0 / torch.clamp(1.0 / scale, min=dtype_info.min, max=dtype_info.max)
        weight *= (1.0 / scale).to(weight.dtype)
        generator = torch.Generator(device=device)
        generator.manual_seed(self._patch_seed(prefix))
        rng = torch.randint(
            0,
            256,
            weight.size(),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        qdata = ck.stochastic_rounding_fp8(weight, rng, base._qdata.dtype)
        params = TensorCoreFP8Layout.Params(
            scale=scale,
            orig_dtype=compute_dtype,
            orig_shape=tuple(qdata.shape),
        )
        patched = QuantizedTensor(qdata, FP8_LAYOUT, params)
        self._patched_weights[prefix] = self._cpu_copy(patched)
        self._materialized_since_reopen += 1
        if self._materialized_since_reopen >= MATERIALIZED_REMAP_INTERVAL:
            self._reopen_before_next_access = True
        if self._active_device is not None:
            self._active_weights[prefix] = patched
            return patched
        return self._patched_weights[prefix].to(device=device)

    def linear(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        self._prepare_base_access()
        weight_key = f"{prefix}.weight"
        scale_key = f"{prefix}.scale_weight"
        bias_key = f"{prefix}.bias"
        bias = (
            self._plain(bias_key, x.device, x.dtype)
            if bias_key in self.base.keys
            else None
        )

        patched = self._patched_weight(prefix, x.device, x.dtype)
        if patched is not None:
            weight = (
                patched.dequantize()
                if isinstance(patched, QuantizedTensor)
                else patched
            )
            out = F.linear(x, weight, bias)
        elif scale_key not in self.base.keys:
            weight = self._plain(weight_key, x.device, x.dtype)
            out = F.linear(x, weight, bias)
        elif not self.native_fp8:
            weight = self._quantized_weight(prefix, x.device, x.dtype).dequantize()
            out = F.linear(x, weight, bias)
        else:
            original_shape = x.shape
            x2 = x.reshape(-1, original_shape[-1])
            weight = self._quantized_weight(prefix, x.device, x.dtype)
            input_scale_key = f"{prefix}.scale_input"
            input_scale = (
                self._plain(input_scale_key, x.device, torch.float32)
                if input_scale_key in self.base.keys
                else None
            )
            qinput = QuantizedTensor.from_float(x2, FP8_LAYOUT, scale=input_scale)
            out = F.linear(qinput, weight, bias)
            out = out.reshape(*original_shape[:-1], out.shape[-1])
        return out

    def conv3d(
        self,
        x: torch.Tensor,
        prefix: str,
        *,
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (0, 0, 0),
    ) -> torch.Tensor:
        weight = self._plain(f"{prefix}.weight", x.device, x.dtype)
        bias_key = f"{prefix}.bias"
        bias = (
            self._plain(bias_key, x.device, x.dtype)
            if bias_key in self.base.keys
            else None
        )
        return F.conv3d(x, weight, bias, stride=stride, padding=padding)

    def affine(
        self, key: str, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return self._plain(key, device, dtype)


def metadata_tensor(payload: dict[str, object]) -> torch.Tensor:
    return torch.tensor(list(json.dumps(payload).encode("utf-8")), dtype=torch.uint8)
