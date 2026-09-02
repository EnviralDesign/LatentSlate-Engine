from __future__ import annotations

import math

import comfy_kitchen as ck
import torch
from comfy_kitchen.tensor import (
    QuantizedTensor,
    TensorCoreFP8Layout,
)
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, *, device: str = "meta"
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device), requires_grad=False
        )
        self.register_buffer("weight_scale", None)
        self.register_buffer("weight_scale_2", None)
        self.register_buffer("input_scale", None)
        self.weight_updates: list[tuple[str, Tensor, Tensor]] = []
        self._klein_dynamic_weight = None
        self._klein_dynamic_device_index: int | None = None

    def add_weight_update(self, kind: str, first: Tensor, second: Tensor) -> None:
        self.weight_updates.append((kind, first, second))

    def bind_dynamic_weight(self, weight, device_index: int) -> None:
        self._klein_dynamic_weight = weight
        self._klein_dynamic_device_index = device_index

    def clear_dynamic_weight(self) -> None:
        self._klein_dynamic_weight = None
        self._klein_dynamic_device_index = None

    def _forward_weight(self, value: Tensor, weight: Tensor) -> Tensor:
        if self.weight_updates:
            if isinstance(weight, QuantizedTensor):
                weight = weight.dequantize().to(value.dtype)
            else:
                weight = weight.to(value.dtype)
            for kind, first, second in self.weight_updates:
                if kind == "lora":
                    update = first.to(value.dtype) @ second.to(value.dtype)
                elif kind == "lokr":
                    update = torch.kron(first.to(value.dtype), second.to(value.dtype))
                else:
                    raise RuntimeError(f"Unknown Klein weight update: {kind}")
                weight = weight + update.reshape(weight.shape).to(weight.dtype)
            return F.linear(value, weight)
        if isinstance(weight, QuantizedTensor):
            original_shape = value.shape[:-1]
            value = value.reshape(-1, value.shape[-1])
            quantized = QuantizedTensor.from_float(value, "TensorCoreNVFP4Layout")
            result = F.linear(quantized, weight)
            return result.reshape(*original_shape, self.out_features)
        if weight.dtype != torch.float8_e4m3fn:
            return F.linear(value, weight)
        if self.weight_scale is None:
            return F.linear(value, weight.to(value.dtype))
        original_shape = value.shape[:-1]
        value = value.reshape(-1, value.shape[-1])
        weight = QuantizedTensor(
            weight,
            "TensorCoreFP8Layout",
            TensorCoreFP8Layout.Params(
                scale=self.weight_scale,
                orig_dtype=value.dtype,
                orig_shape=tuple(weight.shape),
            ),
        )
        quantized = QuantizedTensor.from_float(
            value, "TensorCoreFP8Layout", scale=self.input_scale
        )
        result = F.linear(quantized, weight)
        return result.reshape(*original_shape, self.out_features)

    def forward(self, value: Tensor) -> Tensor:
        dynamic_weight = self._klein_dynamic_weight
        if dynamic_weight is None:
            return self._forward_weight(value, self.weight)
        prepared = getattr(self, "_klein_prepared_weight", None)
        weight = (
            prepared
            if prepared is not None
            else dynamic_weight.materialize(self._klein_dynamic_device_index)
        )
        try:
            return self._forward_weight(value, weight)
        finally:
            if prepared is None:
                dynamic_weight.unpin(self._klein_dynamic_device_index)


class RMSNorm(nn.Module):
    def __init__(self, size: int, *, device: str = "meta") -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.empty(size, device=device), requires_grad=False)

    def forward(self, value: Tensor) -> Tensor:
        normalized = F.rms_norm(value.float(), (value.shape[-1],))
        return (normalized * self.scale.float()).to(value.dtype)


def _linear(in_features: int, out_features: int) -> Linear:
    return Linear(in_features, out_features)


class MLPEmbedder(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.in_layer = _linear(in_features, 4096)
        self.out_layer = _linear(4096, 4096)

    def forward(self, value: Tensor) -> Tensor:
        return self.out_layer(F.silu(self.in_layer(value)))


class QKNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_norm = RMSNorm(128)
        self.key_norm = RMSNorm(128)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self.query_norm(query).to(value), self.key_norm(key).to(value)


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = _linear(4096, 12288)
        self.norm = QKNorm()
        self.proj = _linear(4096, 4096)


class Modulation(nn.Module):
    def __init__(self, multiplier: int) -> None:
        super().__init__()
        self.lin = _linear(4096, multiplier * 4096)

    def forward(self, vector: Tensor) -> tuple[tuple[Tensor, Tensor, Tensor], ...]:
        if vector.ndim == 2:
            vector = vector[:, None, :]
        parts = self.lin(F.silu(vector)).chunk(self.lin.out_features // 4096, dim=-1)
        return tuple(
            tuple(parts[index : index + 3]) for index in range(0, len(parts), 3)
        )


def _modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return torch.addcmul(shift, value, 1 + scale)


def _attention(
    query: Tensor, key: Tensor, value: Tensor, rope: Tensor, mask: Tensor | None
) -> Tensor:
    query, key = ck.apply_rope(query, key, rope)
    backends = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
    with sdpa_kernel(backends, set_priority=True):
        result = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    return result.transpose(1, 2).reshape(result.shape[0], result.shape[2], -1)


class DoubleBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.img_attn = Attention()
        self.txt_attn = Attention()
        self.img_mlp = nn.ModuleDict(
            {"0": _linear(4096, 24576), "2": _linear(12288, 4096)}
        )
        self.txt_mlp = nn.ModuleDict(
            {"0": _linear(4096, 24576), "2": _linear(12288, 4096)}
        )

    @staticmethod
    def _mlp(layers: nn.ModuleDict, value: Tensor) -> Tensor:
        gate, linear = layers["0"](value).chunk(2, dim=-1)
        return layers["2"](F.silu(gate) * linear)

    def forward(
        self,
        image: Tensor,
        text: Tensor,
        image_modulation: tuple[tuple[Tensor, Tensor, Tensor], ...],
        text_modulation: tuple[tuple[Tensor, Tensor, Tensor], ...],
        rope: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        (img_shift1, img_scale1, img_gate1), (img_shift2, img_scale2, img_gate2) = (
            image_modulation
        )
        (txt_shift1, txt_scale1, txt_gate1), (txt_shift2, txt_scale2, txt_gate2) = (
            text_modulation
        )
        img_norm = _modulate(
            F.layer_norm(image, (4096,), eps=1e-6), img_shift1, img_scale1
        )
        txt_norm = _modulate(
            F.layer_norm(text, (4096,), eps=1e-6), txt_shift1, txt_scale1
        )
        img_q, img_k, img_v = (
            self.img_attn.qkv(img_norm)
            .reshape(image.shape[0], image.shape[1], 3, 32, 128)
            .permute(2, 0, 3, 1, 4)
        )
        txt_q, txt_k, txt_v = (
            self.txt_attn.qkv(txt_norm)
            .reshape(text.shape[0], text.shape[1], 3, 32, 128)
            .permute(2, 0, 3, 1, 4)
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        attended = _attention(
            torch.cat((txt_q, img_q), dim=2),
            torch.cat((txt_k, img_k), dim=2),
            torch.cat((txt_v, img_v), dim=2),
            rope,
            mask,
        )
        text_attended, image_attended = attended.split(
            (text.shape[1], image.shape[1]), dim=1
        )
        image = image + img_gate1 * self.img_attn.proj(image_attended)
        text = text + txt_gate1 * self.txt_attn.proj(text_attended)
        image = image + img_gate2 * self._mlp(
            self.img_mlp,
            _modulate(F.layer_norm(image, (4096,), eps=1e-6), img_shift2, img_scale2),
        )
        text = text + txt_gate2 * self._mlp(
            self.txt_mlp,
            _modulate(F.layer_norm(text, (4096,), eps=1e-6), txt_shift2, txt_scale2),
        )
        return image, text


class SingleBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = _linear(4096, 36864)
        self.linear2 = _linear(16384, 4096)
        self.norm = QKNorm()

    def forward(
        self,
        value: Tensor,
        modulation: tuple[tuple[Tensor, Tensor, Tensor], ...],
        rope: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        ((shift, scale, gate),) = modulation
        projected = self.linear1(
            _modulate(F.layer_norm(value, (4096,), eps=1e-6), shift, scale)
        )
        qkv, mlp = projected.split((12288, 24576), dim=-1)
        query, key, attention_value = qkv.reshape(
            value.shape[0], value.shape[1], 3, 32, 128
        ).permute(2, 0, 3, 1, 4)
        query, key = self.norm(query, key, attention_value)
        attended = _attention(query, key, attention_value, rope, mask)
        mlp_gate, mlp_linear = mlp.chunk(2, dim=-1)
        return value + gate * self.linear2(
            torch.cat((attended, F.silu(mlp_gate) * mlp_linear), dim=-1)
        )


def _timestep_embedding(timestep: Tensor) -> Tensor:
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(128, device=timestep.device, dtype=torch.float32)
        / 128
    )
    angles = (1000 * timestep.float())[:, None] * frequencies[None]
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1).to(timestep.dtype)


def _rope(ids: Tensor) -> Tensor:
    pieces = []
    for axis in range(4):
        positions = ids[..., axis].float()
        scale = torch.linspace(0, 30 / 32, 16, dtype=torch.float64, device=ids.device)
        omega = 1.0 / (2000**scale)
        angles = torch.einsum("bn,d->bnd", positions, omega)
        matrix = torch.stack(
            (
                torch.cos(angles),
                -torch.sin(angles),
                torch.sin(angles),
                torch.cos(angles),
            ),
            dim=-1,
        ).reshape(*angles.shape, 2, 2)
        pieces.append(matrix.float())
    return torch.cat(pieces, dim=-3).unsqueeze(1)


class KleinTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.img_in = _linear(128, 4096)
        self.txt_in = _linear(12288, 4096)
        self.time_in = MLPEmbedder(256)
        self.double_blocks = nn.ModuleList(DoubleBlock() for _ in range(8))
        self.single_blocks = nn.ModuleList(SingleBlock() for _ in range(24))
        self.double_stream_modulation_img = Modulation(6)
        self.double_stream_modulation_txt = Modulation(6)
        self.single_stream_modulation = Modulation(3)
        self.final_layer = nn.Module()
        self.final_layer.adaLN_modulation = nn.Sequential(
            nn.SiLU(), _linear(4096, 8192)
        )
        self.final_layer.linear = _linear(4096, 128)

    def forward(
        self,
        latent: Tensor,
        timestep: Tensor,
        context: Tensor,
        attention_mask: Tensor | None,
        reference_latents: tuple[Tensor, ...] | None = None,
    ) -> Tensor:
        batch, _, height, width = latent.shape
        image = latent.permute(0, 2, 3, 1).reshape(batch, height * width, 128)
        text_ids = torch.zeros(batch, context.shape[1], 4, device=latent.device)
        text_ids[:, :, 3] = torch.arange(context.shape[1], device=latent.device)
        rows = torch.arange(height, device=latent.device)[:, None].expand(height, width)
        columns = torch.arange(width, device=latent.device)[None, :].expand(
            height, width
        )
        image_ids = torch.zeros(height, width, 4, device=latent.device)
        image_ids[:, :, 1] = rows
        image_ids[:, :, 2] = columns
        image_ids = image_ids.reshape(1, height * width, 4).expand(batch, -1, -1)

        target_tokens = height * width
        if reference_latents:
            images = [image]
            ids = [image_ids]
            for reference_index, reference in enumerate(reference_latents, start=1):
                ref_batch, ref_channels, ref_height, ref_width = reference.shape
                if ref_batch != batch or ref_channels != 128:
                    raise ValueError(
                        "Reference latent must match the target batch and have 128 channels"
                    )
                images.append(
                    reference.permute(0, 2, 3, 1).reshape(
                        batch, ref_height * ref_width, 128
                    )
                )
                ref_ids = torch.zeros(ref_height, ref_width, 4, device=latent.device)
                ref_ids[:, :, 0] = reference_index * 10
                ref_ids[:, :, 1] = torch.arange(ref_height, device=latent.device)[
                    :, None
                ]
                ref_ids[:, :, 2] = torch.arange(ref_width, device=latent.device)[
                    None, :
                ]
                ids.append(
                    ref_ids.reshape(1, ref_height * ref_width, 4).expand(batch, -1, -1)
                )
            image = torch.cat(images, dim=1)
            image_ids = torch.cat(ids, dim=1)

        image = self.img_in(image)
        text = self.txt_in(context)
        vector = self.time_in(_timestep_embedding(timestep).to(image.dtype))
        rotary = _rope(torch.cat((text_ids, image_ids), dim=1))

        if attention_mask is not None:
            image_mask = torch.ones(
                batch, image.shape[1], device=latent.device, dtype=torch.bool
            )
            attention_mask = torch.cat((attention_mask.bool(), image_mask), dim=1)[
                :, None, None, :
            ]

        image_modulation = self.double_stream_modulation_img(vector)
        text_modulation = self.double_stream_modulation_txt(vector)
        for block in self.double_blocks:
            image, text = block(
                image, text, image_modulation, text_modulation, rotary, attention_mask
            )

        combined = torch.cat((text, image), dim=1)
        single_modulation = self.single_stream_modulation(vector)
        for block in self.single_blocks:
            combined = block(combined, single_modulation, rotary, attention_mask)
        image = combined[:, text.shape[1] : text.shape[1] + target_tokens]
        shift, scale = self.final_layer.adaLN_modulation(vector).chunk(2, dim=-1)
        image = _modulate(
            F.layer_norm(image, (4096,), eps=1e-6), shift[:, None], scale[:, None]
        )
        result = self.final_layer.linear(image)
        return result.reshape(batch, height, width, 128).permute(0, 3, 1, 2)
