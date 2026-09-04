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
from comfy_kitchen.tensor import (
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorCoreNVFP4Layout,
    TensorWiseINT8Layout,
)
from safetensors import safe_open

FP8_LAYOUT = "TensorCoreFP8Layout"
NVFP4_LAYOUT = "TensorCoreNVFP4Layout"
INT8_LAYOUT = "TensorWiseINT8Layout"
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


def _nvfp4_blocked_scales(input_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = input_matrix.shape
    padded_rows = ((rows + 127) // 128) * 128
    padded_cols = ((cols + 3) // 4) * 4
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix
        input_matrix = padded
    row_blocks = padded_rows // 128
    col_blocks = padded_cols // 4
    blocks = input_matrix.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    return (
        blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(padded_rows, padded_cols)
    )


def _stochastic_fp4_e2m1(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    shape = x.shape
    sign = torch.signbit(x).to(torch.uint8)
    exponent = torch.floor(torch.log2(x.abs()) + 1.0).clamp(0, 3)
    x = (
        x
        + (
            torch.rand(
                x.size(),
                dtype=x.dtype,
                layout=x.layout,
                device=x.device,
                generator=generator,
            )
            - 0.5
        )
        * (2 ** (exponent - 2.0))
        * 1.25
    )
    x = x.abs()
    exponent = torch.floor(torch.log2(x) + 1.1925).clamp(0, 3)
    mantissa = (
        torch.where(
            exponent > 0,
            (x / (2.0 ** (exponent - 1)) - 1.0) * 2.0,
            x * 2.0,
            out=x,
        )
        .round()
        .to(torch.uint8)
    )
    fp4 = (sign << 3) | (exponent.to(torch.uint8) << 1) | mantissa
    flat = fp4.view(-1)
    return ((flat[0::2] << 4) | flat[1::2]).reshape(*shape[:-1], -1)


def _stochastic_quantize_nvfp4(
    value: torch.Tensor, scale: torch.Tensor, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = value.shape
    padded_rows = ((rows + 15) // 16) * 16
    padded_cols = ((cols + 15) // 16) * 16
    if (rows, cols) != (padded_rows, padded_cols):
        value = F.pad(value, (0, padded_cols - cols, 0, padded_rows - rows))
    shape = value.shape
    qdata = torch.empty(
        (shape[0], shape[1] // 2), dtype=torch.uint8, device=value.device
    )
    block_scales = torch.empty(
        (shape[0], shape[1] // 16),
        dtype=torch.float8_e4m3fn,
        device=value.device,
    )
    generator = torch.Generator(device=value.device)
    generator.manual_seed(seed)
    slice_count = max(1, value.numel() / (4096 * 4096))
    slice_size = max(1, round(shape[0] / slice_count))
    for start in range(0, shape[0], slice_size):
        current = value[start : start + slice_size]
        current_shape = current.shape
        blocks = current.reshape(current_shape[0], -1, 16)
        current_scales = torch.clamp(
            (torch.amax(torch.abs(blocks), dim=-1) / 6.0) / scale.to(current.dtype),
            max=448.0,
        ).to(torch.float8_e4m3fn)
        blocks = blocks / (
            scale.to(current.dtype) * current_scales.to(current.dtype)
        ).unsqueeze(-1)
        qdata[start : start + slice_size].copy_(
            _stochastic_fp4_e2m1(blocks.view(current_shape).nan_to_num(), generator)
        )
        block_scales[start : start + slice_size].copy_(current_scales)
    return qdata, _nvfp4_blocked_scales(block_scales)


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
        physical_keys = tuple(self._mapping.keys())
        self._key_prefix = (
            "model.diffusion_model."
            if "model.diffusion_model.patch_embedding.weight" in physical_keys
            else ""
        )
        self.keys = frozenset(
            key.removeprefix(self._key_prefix) for key in physical_keys
        )

    def tensor(self, key: str) -> torch.Tensor:
        if self._mapping is None:
            raise RuntimeError(f"tensor store is closed: {self.identity.path}")
        return self._mapping.get_tensor(f"{self._key_prefix}{key}")

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
        secondary_lora: str | Path | None = None,
        secondary_lora_strength: float = 1.0,
        native_fp8: bool = True,
    ):
        self.base = TensorStore(checkpoint)
        self.lora = TensorStore(lora) if lora is not None else None
        self.lora_strength = lora_strength
        self.secondary_lora = (
            TensorStore(secondary_lora) if secondary_lora is not None else None
        )
        self.secondary_lora_strength = secondary_lora_strength
        self.native_fp8 = native_fp8
        self._patched_weights: dict[str, QuantizedTensor] = {}
        self._active_weights: dict[str, QuantizedTensor] = {}
        self._active_qk_norms: dict[str, torch.Tensor] = {}
        self._active_device: torch.device | None = None
        self._materialized_device_bytes = 0
        self._retain_materialized_on_device = False
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
        if self.secondary_lora is not None:
            self._validate_lora(self.secondary_lora, account_materialized=False)

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            self.base.identity,
            self.lora.identity if self.lora else None,
            self.lora_strength,
            self.secondary_lora.identity if self.secondary_lora else None,
            self.secondary_lora_strength,
            self.native_fp8,
        )

    def _validate_lora(
        self,
        store: TensorStore | None = None,
        *,
        account_materialized: bool = True,
    ) -> None:
        store = self.lora if store is None else store
        if store is None:
            return
        targets = {
            key[: -len(suffix)]
            for key in store.keys
            for suffix in (".lora_up.weight", ".lora_B.weight")
            if key.endswith(suffix)
        }
        if len(targets) != 400:
            raise ValueError(
                f"canonical Wan LoRA must contain 400 targets, found {len(targets)}"
            )
        for target in targets:
            parts = self._lora_parts(target.removeprefix("diffusion_model."), store)
            base = f"{target.removeprefix('diffusion_model.')}.weight"
            if parts is None or base not in self.base.keys:
                raise ValueError(
                    f"incomplete or unmapped canonical Wan LoRA target: {target}"
                )
            _up, _down, alpha = parts
            if alpha is not None and int(store.tensor(alpha).item()) != 8:
                raise ValueError(f"unexpected canonical Wan LoRA alpha at {target}")
            prefix = target.removeprefix("diffusion_model.")
            if account_materialized and not self._is_live_patch(prefix):
                self._materialized_device_bytes += self.base.tensor(
                    f"{prefix}.weight"
                ).nbytes
                for suffix in ("scale_weight", "weight_scale", "weight_scale_2"):
                    key = f"{prefix}.{suffix}"
                    if key in self.base.keys:
                        self._materialized_device_bytes += self.base.tensor(key).nbytes

    def _lora_parts(
        self, prefix: str, store: TensorStore | None = None
    ) -> tuple[str, str, str | None] | None:
        store = self.lora if store is None else store
        if store is None:
            return None
        target = f"diffusion_model.{prefix}"
        legacy_up = f"{target}.lora_up.weight"
        if legacy_up in store.keys:
            return legacy_up, f"{target}.lora_down.weight", f"{target}.alpha"
        diffusers_up = f"{target}.lora_B.weight"
        if diffusers_up in store.keys:
            return diffusers_up, f"{target}.lora_A.weight", None
        return None

    def _lora_values(
        self,
        prefix: str,
        device: torch.device,
        dtype: torch.dtype,
        store: TensorStore | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float, tuple[torch.Tensor, ...]]:
        store = self.lora if store is None else store
        parts = self._lora_parts(prefix, store)
        if parts is None or store is None:
            raise ValueError(f"missing Wan LoRA target: {prefix}")
        up_key, down_key, alpha_key = parts
        up_cpu = store.tensor(up_key)
        down_cpu = store.tensor(down_key)
        up = up_cpu.to(device=device, dtype=dtype, non_blocking=True)
        down = down_cpu.to(device=device, dtype=dtype, non_blocking=True)
        alpha = (
            float(store.tensor(alpha_key).item()) / down.shape[0]
            if alpha_key is not None
            else 1.0
        )
        return up, down, alpha, (up_cpu, down_cpu)

    def _plain(
        self, key: str, device: torch.device, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        value = self._active_qk_norms.get(key)
        if value is None:
            value = self.base.tensor(key).to(device=device, non_blocking=True)
        return value.to(dtype=dtype) if dtype is not None else value

    def activate(self, device: torch.device, *, workspace_bytes: int = 0) -> None:
        if self._active_device == device:
            return
        torch.cuda.empty_cache()
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        self._retain_materialized_on_device = (
            len(self._patched_weights) == MATERIALIZED_PATCH_COUNT
            and self._materialized_device_bytes + workspace_bytes <= free_bytes
        )
        active: dict[str, QuantizedTensor] = {}
        if self._retain_materialized_on_device:
            for key, value in self._patched_weights.items():
                qdata = value._qdata.to(device=device, non_blocking=True)
                scale = value._params.scale.to(device=device, non_blocking=True)
                if isinstance(value._params, TensorCoreNVFP4Layout.Params):
                    params = TensorCoreNVFP4Layout.Params(
                        scale=scale,
                        block_scale=value._params.block_scale.to(
                            device=device, non_blocking=True
                        ),
                        orig_dtype=value._params.orig_dtype,
                        orig_shape=value._params.orig_shape,
                    )
                    active[key] = QuantizedTensor(qdata, NVFP4_LAYOUT, params)
                elif isinstance(value._params, TensorWiseINT8Layout.Params):
                    params = TensorWiseINT8Layout.Params(
                        scale=scale,
                        orig_dtype=value._params.orig_dtype,
                        orig_shape=value._params.orig_shape,
                        is_weight=value._params.is_weight,
                        convrot=value._params.convrot,
                        convrot_groupsize=value._params.convrot_groupsize,
                    )
                    active[key] = QuantizedTensor(qdata, INT8_LAYOUT, params)
                else:
                    params = TensorCoreFP8Layout.Params(
                        scale=scale,
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
        self._retain_materialized_on_device = False
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
        if isinstance(value._params, TensorCoreNVFP4Layout.Params):
            block_scale = torch.empty_like(value._params.block_scale, device="cpu")
            block_scale.copy_(value._params.block_scale, non_blocking=True)
            params = TensorCoreNVFP4Layout.Params(
                scale=scale,
                block_scale=block_scale,
                orig_dtype=value._params.orig_dtype,
                orig_shape=value._params.orig_shape,
            )
            return QuantizedTensor(qdata, NVFP4_LAYOUT, params)
        if isinstance(value._params, TensorWiseINT8Layout.Params):
            params = TensorWiseINT8Layout.Params(
                scale=scale,
                orig_dtype=value._params.orig_dtype,
                orig_shape=value._params.orig_shape,
                is_weight=value._params.is_weight,
                convrot=value._params.convrot,
                convrot_groupsize=value._params.convrot_groupsize,
            )
            return QuantizedTensor(qdata, INT8_LAYOUT, params)
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
        qdata = self._plain(weight_key, device)
        comfy_scale_key = f"{prefix}.weight_scale"
        if qdata.dtype == torch.int8 and comfy_scale_key in self.base.keys:
            config_key = f"{prefix}.comfy_quant"
            config = (
                json.loads(bytes(self.base.tensor(config_key).tolist()))
                if config_key in self.base.keys
                else {}
            )
            params = TensorWiseINT8Layout.Params(
                scale=self._plain(comfy_scale_key, device, torch.float32),
                orig_dtype=compute_dtype,
                orig_shape=tuple(qdata.shape),
                is_weight=True,
                convrot=bool(config.get("convrot", False)),
                convrot_groupsize=int(config.get("convrot_groupsize", 256)),
            )
            return QuantizedTensor(qdata, INT8_LAYOUT, params)
        tensor_scale_key = f"{prefix}.weight_scale_2"
        if tensor_scale_key in self.base.keys:
            block_scale = self._plain(f"{prefix}.weight_scale", device)
            tensor_scale = self._plain(tensor_scale_key, device, torch.float32)
            params = TensorCoreNVFP4Layout.Params(
                scale=tensor_scale,
                block_scale=block_scale,
                orig_dtype=compute_dtype,
                orig_shape=(qdata.shape[0], qdata.shape[1] * 2),
            )
            return QuantizedTensor(qdata, NVFP4_LAYOUT, params)

        legacy_scale_key = f"{prefix}.scale_weight"
        scale_key = (
            legacy_scale_key if legacy_scale_key in self.base.keys else comfy_scale_key
        )
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
        qdata_cpu = self.base.tensor(f"{prefix}.weight")
        base = self._quantized_weight(prefix, device, compute_dtype)
        up, down, alpha, lora_sources = self._lora_values(prefix, device, compute_dtype)
        scale_sources = tuple(
            self.base.tensor(key)
            for key in (
                f"{prefix}.scale_weight",
                f"{prefix}.weight_scale",
                f"{prefix}.weight_scale_2",
            )
            if key in self.base.keys
        )
        return base, up, down, alpha, (qdata_cpu, *scale_sources, *lora_sources)

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
            if isinstance(base._params, TensorCoreNVFP4Layout.Params):
                base._params.block_scale.record_stream(current)
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
        for byte in f"diffusion_model.{prefix}".encode():
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ 0xEDB88320 if crc & 1 else crc >> 1
        return crc ^ 0xFFFFFFFF

    def _requantize_patched(
        self,
        prefix: str,
        base: QuantizedTensor,
        weight: torch.Tensor,
        device: torch.device,
        compute_dtype: torch.dtype,
    ) -> QuantizedTensor:
        seed = self._patch_seed(prefix)
        if isinstance(base._params, TensorCoreNVFP4Layout.Params):
            scale = torch.amax(weight.abs()).to(torch.float32) / (448.0 * 6.0)
            qdata, block_scale = _stochastic_quantize_nvfp4(weight, scale, seed)
            params = TensorCoreNVFP4Layout.Params(
                scale=scale,
                block_scale=block_scale,
                orig_dtype=compute_dtype,
                orig_shape=tuple(weight.shape),
            )
            return QuantizedTensor(qdata, NVFP4_LAYOUT, params)
        if isinstance(base._params, TensorWiseINT8Layout.Params):
            qdata, params = TensorWiseINT8Layout.quantize(
                weight,
                scale="recalculate",
                stochastic_rounding=seed,
                **TensorWiseINT8Layout.requantize_kwargs(base),
            )
            return QuantizedTensor(qdata, INT8_LAYOUT, params)

        scale = (
            torch.amax(weight.abs()).to(torch.float32)
            / torch.finfo(base._qdata.dtype).max
        )
        dtype_info = torch.finfo(weight.dtype)
        scale = 1.0 / torch.clamp(1.0 / scale, min=dtype_info.min, max=dtype_info.max)
        weight *= (1.0 / scale).to(weight.dtype)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
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
        return QuantizedTensor(qdata, FP8_LAYOUT, params)

    def _stacked_patched_weight(
        self,
        prefix: str,
        device: torch.device,
        compute_dtype: torch.dtype,
    ) -> QuantizedTensor | None:
        stores = tuple(
            (store, strength)
            for store, strength in (
                (self.lora, self.lora_strength),
                (
                    getattr(self, "secondary_lora", None),
                    getattr(self, "secondary_lora_strength", 1.0),
                ),
            )
            if store is not None and self._lora_parts(prefix, store) is not None
        )
        if not stores:
            return None
        keep_live = self._is_live_patch(prefix)
        if not keep_live:
            active = self._active_weights.get(prefix)
            if active is not None:
                return active
            cached = self._patched_weights.get(prefix)
            if cached is not None:
                return cached.to(device=device)

        base = self._quantized_weight(prefix, device, compute_dtype)
        weight = base.dequantize()
        for store, strength in stores:
            up, down, alpha, _sources = self._lora_values(
                prefix, device, torch.float32, store
            )
            delta = torch.mm(
                up.flatten(start_dim=1), down.flatten(start_dim=1)
            ).reshape(weight.shape)
            weight.add_(((strength * alpha) * delta).to(weight.dtype))
        patched = self._requantize_patched(prefix, base, weight, device, compute_dtype)
        if keep_live:
            return patched
        self._patched_weights[prefix] = self._cpu_copy(patched)
        self._materialized_since_reopen += 1
        if self._materialized_since_reopen >= MATERIALIZED_REMAP_INTERVAL:
            self._reopen_before_next_access = True
        if self._active_device is not None and self._retain_materialized_on_device:
            self._active_weights[prefix] = patched
            return patched
        return self._patched_weights[prefix].to(device=device)

    def _patched_weight(
        self,
        prefix: str,
        device: torch.device,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor | QuantizedTensor | None:
        if getattr(self, "secondary_lora", None) is not None:
            return self._stacked_patched_weight(prefix, device, compute_dtype)
        if self._lora_parts(prefix) is None:
            return None
        keep_live = self._is_live_patch(prefix)
        if not keep_live:
            active = self._active_weights.get(prefix)
            if active is not None:
                return active
            cached = self._patched_weights.get(prefix)
            if cached is not None:
                return cached.to(device=device)

        if keep_live:
            base, up, down, alpha, _sources = self._consume_live_patch(
                prefix, device, compute_dtype
            )
        else:
            base = self._quantized_weight(prefix, device, compute_dtype)
            up, down, alpha, _sources = self._lora_values(prefix, device, torch.float32)
        weight = base.dequantize()
        delta = torch.mm(up.flatten(start_dim=1), down.flatten(start_dim=1)).reshape(
            weight.shape
        )
        weight.add_(((self.lora_strength * alpha) * delta).to(weight.dtype))
        patched = self._requantize_patched(prefix, base, weight, device, compute_dtype)
        if keep_live:
            return patched
        self._patched_weights[prefix] = self._cpu_copy(patched)
        self._materialized_since_reopen += 1
        if self._materialized_since_reopen >= MATERIALIZED_REMAP_INTERVAL:
            self._reopen_before_next_access = True
        if self._active_device is not None and self._retain_materialized_on_device:
            self._active_weights[prefix] = patched
            return patched
        return self._patched_weights[prefix].to(device=device)

    def linear(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        self._prepare_base_access()
        weight_key = f"{prefix}.weight"
        scale_key = f"{prefix}.scale_weight"
        comfy_scale_key = f"{prefix}.weight_scale"
        nvfp4_scale_key = f"{prefix}.weight_scale_2"
        bias_key = f"{prefix}.bias"
        bias = (
            self._plain(bias_key, x.device, x.dtype)
            if bias_key in self.base.keys
            else None
        )

        comfy_fp8 = comfy_scale_key in self.base.keys and self.base.tensor(
            weight_key
        ).dtype in {torch.float8_e4m3fn, torch.float8_e5m2}
        comfy_int8 = (
            comfy_scale_key in self.base.keys
            and self.base.tensor(weight_key).dtype == torch.int8
        )
        patched = self._patched_weight(prefix, x.device, x.dtype)
        if patched is not None:
            if isinstance(patched, QuantizedTensor):
                original_shape = x.shape
                x2 = x.reshape(-1, original_shape[-1])
                if isinstance(patched._params, TensorWiseINT8Layout.Params):
                    out = F.linear(x2, patched, bias)
                else:
                    input_layout = (
                        patched._layout_cls
                        if isinstance(patched._params, TensorCoreNVFP4Layout.Params)
                        else FP8_LAYOUT
                    )
                    input_scale_key = f"{prefix}.input_scale"
                    input_scale = (
                        self._plain(input_scale_key, x.device, torch.float32)
                        if input_scale_key in self.base.keys
                        else torch.ones((), device=x.device, dtype=torch.float32)
                    )
                    qinput = QuantizedTensor.from_float(
                        x2,
                        input_layout,
                        scale=input_scale,
                    )
                    out = F.linear(qinput, patched, bias)
                out = out.reshape(*original_shape[:-1], out.shape[-1])
            else:
                weight = (
                    patched.dequantize()
                    if isinstance(patched, QuantizedTensor)
                    else patched
                )
                out = F.linear(x, weight, bias)
        elif (
            scale_key not in self.base.keys
            and nvfp4_scale_key not in self.base.keys
            and not comfy_fp8
            and not comfy_int8
        ):
            weight = self._plain(weight_key, x.device, x.dtype)
            out = F.linear(x, weight, bias)
        elif not self.native_fp8:
            weight = self._quantized_weight(prefix, x.device, x.dtype).dequantize()
            out = F.linear(x, weight, bias)
        else:
            original_shape = x.shape
            x2 = x.reshape(-1, original_shape[-1])
            weight = self._quantized_weight(prefix, x.device, x.dtype)
            if isinstance(weight._params, TensorWiseINT8Layout.Params):
                out = F.linear(x2, weight, bias)
            else:
                input_scale_key = (
                    f"{prefix}.input_scale"
                    if nvfp4_scale_key in self.base.keys
                    else f"{prefix}.scale_input"
                )
                input_scale = (
                    self._plain(input_scale_key, x.device, torch.float32)
                    if input_scale_key in self.base.keys
                    else torch.ones((), device=x.device, dtype=torch.float32)
                )
                qinput = QuantizedTensor.from_float(
                    x2, weight._layout_cls, scale=input_scale
                )
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
