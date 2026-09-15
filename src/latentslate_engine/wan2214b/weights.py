from __future__ import annotations

import json
from contextlib import nullcontext
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

from .contracts import ArtifactIdentity

FP8_LAYOUT = "TensorCoreFP8Layout"
NVFP4_LAYOUT = "TensorCoreNVFP4Layout"
INT8_LAYOUT = "TensorWiseINT8Layout"


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


class TensorStore:
    """Read checkpoint tensors without Torch's mmap-backed storage slicing."""

    def __init__(self, path: str | Path):
        self.identity = ArtifactIdentity.from_path(path)
        self._mapping = safe_open(
            self.identity.path, framework="pt", device="cpu", backend="pread"
        )
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
        self._mapping = safe_open(
            self.identity.path, framework="pt", device="cpu", backend="pread"
        )


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
        self._active_device: torch.device | None = None
        self._residency = None
        self._current_values = {}
        self._current_updates = {}
        self._current_binding = None
        self._validate_lora()
        if self.secondary_lora is not None:
            self._validate_lora(self.secondary_lora)

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
        updates = getattr(self, "_current_updates", {}).get(id(store), {})
        up_cpu = updates[up_key] if up_key in updates else store.tensor(up_key)
        down_cpu = updates[down_key] if down_key in updates else store.tensor(down_key)
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
        value = getattr(self, "_current_values", {}).get(key)
        residency = getattr(self, "_residency", None)
        if value is None and residency is not None:
            value = residency.direct.get(key)
        if value is None:
            value = self.base.tensor(key).to(device=device, non_blocking=True)
        return value.to(dtype=dtype) if dtype is not None else value

    def activate(self, device: torch.device) -> None:
        if self._active_device == device:
            return
        from .residency import WanResidency

        if self._residency is None:
            self._residency = WanResidency(self, device)
        self._residency.activate()
        self._active_device = device

    def deactivate(self) -> None:
        if self._residency is not None:
            self._residency.deactivate()
        self._active_device = None

    def close(self) -> None:
        if self._residency is not None:
            self._residency.close()
            self._residency = None
        self.base.close()
        for store in (self.lora, self.secondary_lora):
            if store is not None:
                store.close()

    def _use(self, prefix):
        residency = getattr(self, "_residency", None)
        return residency.use(prefix) if residency is not None else nullcontext()

    def _quantized_weight(
        self,
        prefix: str,
        device: torch.device,
        compute_dtype: torch.dtype,
        *,
        source_weight: torch.Tensor | None = None,
    ) -> QuantizedTensor:
        weight_key = f"{prefix}.weight"
        qdata = (
            self._plain(weight_key, device)
            if source_weight is None
            else source_weight.to(device=device, non_blocking=True)
        )
        comfy_scale_key = f"{prefix}.weight_scale"
        if qdata.dtype == torch.int8 and comfy_scale_key in self.base.keys:
            config_key = f"{prefix}.comfy_quant"
            config = (
                json.loads(bytes(self.base.tensor(config_key).tolist()))
                if config_key in self.base.keys
                else {}
            )
            quantization_format = config.get("format")
            if quantization_format != "int8_tensorwise":
                raise ValueError(
                    f"unsupported Wan INT8 quantization format: {quantization_format!r}"
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

    def _patched_weight(
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
        binding = getattr(self, "_current_binding", None)
        if binding is not None and binding.resident:
            return binding.patched
        base = self._quantized_weight(prefix, device, compute_dtype)
        weight = base.dequantize()
        for store, strength in stores:
            up, down, alpha, _sources = self._lora_values(
                prefix, device, compute_dtype, store
            )
            delta = torch.mm(
                up.flatten(start_dim=1), down.flatten(start_dim=1)
            ).reshape(weight.shape)
            del up, down
            weight.add_(((strength * alpha) * delta).to(weight.dtype))
            # Comfy's adapter call returns before requantization; its full delta
            # allocation must be released at that same boundary here.
            del delta, _sources
        patched = self._requantize_patched(prefix, base, weight, device, compute_dtype)
        if binding is not None and binding.signature is not None:
            # Kitchen copies the complete logical payload, including sidecars.
            # Only the fault signature permits reusing these patched GPU bytes.
            base.copy_(patched)
            binding.patched = base
        return patched

    def linear(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        with self._use(prefix):
            return self._linear(x, prefix)

    def _linear(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
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

        source_weight = None
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
            and (
                comfy_scale_key not in self.base.keys
                or (source_weight := self._plain(weight_key, x.device)).dtype
                not in {torch.float8_e4m3fn, torch.float8_e5m2, torch.int8}
            )
        ):
            weight = (
                self._plain(weight_key, x.device, x.dtype)
                if source_weight is None
                else source_weight.to(device=x.device, dtype=x.dtype)
            )
            out = F.linear(x, weight, bias)
        elif not self.native_fp8:
            weight = self._quantized_weight(
                prefix, x.device, x.dtype, source_weight=source_weight
            ).dequantize()
            out = F.linear(x, weight, bias)
        else:
            original_shape = x.shape
            x2 = x.reshape(-1, original_shape[-1])
            weight = self._quantized_weight(
                prefix, x.device, x.dtype, source_weight=source_weight
            )
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
        with self._use(prefix):
            weight = self._plain(f"{prefix}.weight", x.device, x.dtype)
            bias_key = f"{prefix}.bias"
            bias = (
                self._plain(bias_key, x.device, x.dtype)
                if bias_key in self.base.keys
                else None
            )
            return F.conv3d(x, weight, bias, stride=stride, padding=padding)

    def layer_norm(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        with self._use(prefix):
            weight = self._plain(f"{prefix}.weight", x.device, x.dtype)
            bias = self._plain(f"{prefix}.bias", x.device, x.dtype)
            return F.layer_norm(x, (x.shape[-1],), weight, bias, eps=1e-6)

    def affine(
        self, key: str, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return self._plain(key, device, dtype)
