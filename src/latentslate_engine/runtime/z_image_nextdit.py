"""Engine-native fused NextDiT used by the pinned Z-Image Turbo checkpoint.

The implementation is derived from the Apache-2.0 first-party Z-Image
transformer and keeps the checkpoint's fused ``attention.qkv`` spelling.
That is important: the 202 stored ConvRot tensors are never split or
dequantized into dense projections.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ..artifacts import revalidate_artifact
from ..stored_quant import restore_stored_quantized_tensor
from ..z_image_turbo_recipe import (
    _Z_TRANSFORMER_STORED_CATEGORY_COUNTS,
    ZImageTransformerPlan,
    _read_z_safetensors_header,
    _z_image_transformer_stored_category,
)
from .z_image_stored_adapter import ZImageStoredConvRotLinear

_SEQ_MULTIPLE = 32


@dataclass(frozen=True, slots=True)
class ZImageNextDiTConfig:
    in_channels: int = 16
    dim: int = 3840
    layers: int = 30
    refiner_layers: int = 2
    heads: int = 30
    kv_heads: int = 30
    norm_eps: float = 1e-5
    cap_dim: int = 2560
    patch_size: int = 2
    temporal_patch_size: int = 1
    rope_theta: float = 256.0
    time_scale: float = 1000.0
    axes_dims: tuple[int, ...] = (32, 48, 48)
    axes_lens: tuple[int, ...] = (1536, 512, 512)


class ZImageRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(value.dtype) * self.weight


class ZImageTimestepEmbedder(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int = 1024, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim)
        )

    @staticmethod
    def sinusoidal(value: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=value.device)
            / half
        )
        angles = value[:, None].float() * frequencies[None]
        result = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        if dim % 2:
            result = torch.cat((result, torch.zeros_like(result[:, :1])), dim=-1)
        return result

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        encoded = self.sinusoidal(value, self.frequency_dim)
        return self.mlp(encoded.to(self.mlp[0].weight.dtype))


class ZImageRope:
    def __init__(
        self, theta: float, axes_dims: tuple[int, ...], axes_lens: tuple[int, ...]
    ) -> None:
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self._cache: tuple[torch.Tensor, ...] | None = None

    def _frequencies(self) -> tuple[torch.Tensor, ...]:
        if self._cache is None:
            values = []
            for dim, length in zip(self.axes_dims, self.axes_lens, strict=True):
                inverse = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
                angles = torch.outer(torch.arange(length, dtype=torch.float64), inverse).float()
                values.append(torch.polar(torch.ones_like(angles), angles).to(torch.complex64))
            self._cache = tuple(values)
        return self._cache

    def __call__(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim != 2 or coordinates.shape[-1] != len(self.axes_dims):
            raise ValueError("Z-Image RoPE coordinates must be [tokens, axes]")
        tables = self._frequencies()
        return torch.cat(
            tuple(
                table.to(coordinates.device)[coordinates[:, axis]]
                for axis, table in enumerate(tables)
            ),
            dim=-1,
        )


def _apply_rope(value: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    as_complex = torch.view_as_complex(value.float().reshape(*value.shape[:-1], -1, 2))
    rotated = torch.view_as_real(as_complex * frequencies.unsqueeze(2)).flatten(3)
    return rotated.to(value.dtype)


class ZImageFusedAttention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, eps: float) -> None:
        super().__init__()
        if heads != kv_heads:
            raise ValueError("Pinned Z-Image Turbo requires equal query and KV head counts")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, (heads + 2 * kv_heads) * self.head_dim, bias=False)
        self.out = nn.Linear(heads * self.head_dim, dim, bias=False)
        self.q_norm = ZImageRMSNorm(self.head_dim, eps)
        self.k_norm = ZImageRMSNorm(self.head_dim, eps)

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        query = self.q_norm(query.unflatten(-1, (self.heads, self.head_dim)))
        key = self.k_norm(key.unflatten(-1, (self.heads, self.head_dim)))
        value = value.unflatten(-1, (self.heads, self.head_dim))
        query = _apply_rope(query, frequencies).transpose(1, 2)
        key = _apply_rope(key, frequencies).transpose(1, 2)
        value = value.transpose(1, 2)
        attention_mask = None if mask is None else mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        return self.out(attended.transpose(1, 2).flatten(2))


class ZImageFeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = int(dim / 3 * 8)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(value)) * self.w3(value))


class ZImageTransformerBlock(nn.Module):
    def __init__(self, config: ZImageNextDiTConfig, *, modulation: bool) -> None:
        super().__init__()
        self.modulation = modulation
        self.attention = ZImageFusedAttention(
            config.dim, config.heads, config.kv_heads, config.norm_eps
        )
        self.feed_forward = ZImageFeedForward(config.dim)
        self.attention_norm1 = ZImageRMSNorm(config.dim, config.norm_eps)
        self.ffn_norm1 = ZImageRMSNorm(config.dim, config.norm_eps)
        self.attention_norm2 = ZImageRMSNorm(config.dim, config.norm_eps)
        self.ffn_norm2 = ZImageRMSNorm(config.dim, config.norm_eps)
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(config.dim, 256), 4 * config.dim, bias=True)
            )

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        frequencies: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation:
            if condition is None:
                raise ValueError("Z-Image modulated block requires timestep conditioning")
            scale_attn, gate_attn, scale_ffn, gate_ffn = (
                self.adaLN_modulation(condition).unsqueeze(1).chunk(4, dim=2)
            )
            attention = self.attention(
                self.attention_norm1(value) * (1.0 + scale_attn), mask, frequencies
            )
            value = value + gate_attn.tanh() * self.attention_norm2(attention)
            feed_forward = self.feed_forward(self.ffn_norm1(value) * (1.0 + scale_ffn))
            return value + gate_ffn.tanh() * self.ffn_norm2(feed_forward)
        attention = self.attention(self.attention_norm1(value), mask, frequencies)
        value = value + self.attention_norm2(attention)
        return value + self.ffn_norm2(self.feed_forward(self.ffn_norm1(value)))


class ZImageFinalLayer(nn.Module):
    def __init__(self, dim: int, output_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, output_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(min(dim, 256), dim))

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaLN_modulation(condition)
        return self.linear(self.norm_final(value) * scale.unsqueeze(1))


class ZImageNextDiTShell(nn.Module):
    """Source-key-compatible fused NextDiT with an executable forward."""

    def __init__(self, config: ZImageNextDiTConfig | None = None) -> None:
        super().__init__()
        config = ZImageNextDiTConfig() if config is None else config
        if config.dim % config.heads or config.dim // config.heads != sum(config.axes_dims):
            raise ValueError("Z-Image head and RoPE dimensions are inconsistent")
        self.config = config
        patch_dim = (
            config.temporal_patch_size * config.patch_size * config.patch_size * config.in_channels
        )
        self.x_embedder = nn.Linear(patch_dim, config.dim)
        self.final_layer = ZImageFinalLayer(config.dim, patch_dim)
        self.noise_refiner = nn.ModuleList(
            ZImageTransformerBlock(config, modulation=True) for _ in range(config.refiner_layers)
        )
        self.context_refiner = nn.ModuleList(
            ZImageTransformerBlock(config, modulation=False) for _ in range(config.refiner_layers)
        )
        self.t_embedder = ZImageTimestepEmbedder(min(config.dim, 256))
        self.cap_embedder = nn.Sequential(
            ZImageRMSNorm(config.cap_dim, config.norm_eps),
            nn.Linear(config.cap_dim, config.dim),
        )
        self.x_pad_token = nn.Parameter(torch.empty((1, config.dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, config.dim)))
        self.layers = nn.ModuleList(
            ZImageTransformerBlock(config, modulation=True) for _ in range(config.layers)
        )
        self.rope = ZImageRope(config.rope_theta, config.axes_dims, config.axes_lens)

    @staticmethod
    def coordinate_grid(
        size: tuple[int, int, int], start: tuple[int, int, int], device: torch.device
    ) -> torch.Tensor:
        axes = [
            torch.arange(offset, offset + span, dtype=torch.int32, device=device)
            for offset, span in zip(start, size, strict=True)
        ]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flatten(0, 2)

    def _prepare_item(
        self, image: torch.Tensor, caption: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[int, int, int],
    ]:
        config = self.config
        channels, frames, height, width = image.shape
        if channels != config.in_channels:
            raise ValueError("Z-Image latent channel count differs from the checkpoint")
        if (
            frames % config.temporal_patch_size
            or height % config.patch_size
            or width % config.patch_size
        ):
            raise ValueError("Z-Image latent dimensions must align to the patch size")
        if caption.ndim != 2 or caption.shape[-1] != config.cap_dim or not len(caption):
            raise ValueError("Z-Image caption embedding shape differs from the checkpoint")

        caption_length = len(caption)
        caption_padding = (-caption_length) % _SEQ_MULTIPLE
        if caption_padding:
            caption = torch.cat((caption, caption[-1:].repeat(caption_padding, 1)))
        caption_mask = torch.zeros(len(caption), dtype=torch.bool, device=image.device)
        caption_mask[caption_length:] = True
        caption_positions = self.coordinate_grid((len(caption), 1, 1), (1, 0, 0), image.device)

        pf, patch = config.temporal_patch_size, config.patch_size
        ft, ht, wt = frames // pf, height // patch, width // patch
        patches = (
            image.view(channels, ft, pf, ht, patch, wt, patch)
            .permute(1, 3, 5, 2, 4, 6, 0)
            .reshape(ft * ht * wt, pf * patch * patch * channels)
        )
        image_length = len(patches)
        image_padding = (-image_length) % _SEQ_MULTIPLE
        positions = self.coordinate_grid((ft, ht, wt), (len(caption) + 1, 0, 0), image.device)
        if image_padding:
            patches = torch.cat((patches, patches[-1:].repeat(image_padding, 1)))
            positions = torch.cat(
                (
                    positions,
                    torch.zeros((image_padding, 3), dtype=torch.int32, device=image.device),
                )
            )
        image_mask = torch.zeros(len(patches), dtype=torch.bool, device=image.device)
        image_mask[image_length:] = True
        return (
            patches,
            caption,
            positions,
            caption_positions,
            image_mask,
            caption_mask,
            (frames, height, width),
        )

    @staticmethod
    def _stage_sequence(
        values: list[torch.Tensor],
        positions: list[torch.Tensor],
        inner_masks: list[torch.Tensor],
        pad_token: torch.Tensor,
        rope: ZImageRope,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        lengths = [len(item) for item in values]
        replaced = [
            torch.where(mask[:, None], pad_token, item)
            for item, mask in zip(values, inner_masks, strict=True)
        ]
        frequencies = [rope(item) for item in positions]
        padded = pad_sequence(replaced, batch_first=True, padding_value=0.0)
        padded_frequencies = pad_sequence(frequencies, batch_first=True, padding_value=0.0)
        mask = torch.zeros((len(values), max(lengths)), dtype=torch.bool, device=padded.device)
        for index, length in enumerate(lengths):
            mask[index, :length] = True
        return padded, padded_frequencies, mask, lengths

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        caption_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the saved model output for ``[B,C,H,W]`` latents.

        ``timestep`` is the shifted flow sigma in ``[0, 1]``.  The saved model
        conditions on ``1 - sigma`` while the outer flow wrapper converts its
        raw output to a denoised estimate.
        """

        if latents.ndim != 4 or caption_embeddings.ndim != 3:
            raise ValueError("Z-Image forward requires BCHW latents and BSC captions")
        if latents.shape[0] != caption_embeddings.shape[0]:
            raise ValueError("Z-Image latent and caption batches differ")
        if timestep.ndim == 0:
            timestep = timestep.expand(latents.shape[0])
        if tuple(timestep.shape) != (latents.shape[0],):
            raise ValueError("Z-Image timestep must have one value per batch item")
        prepared = [
            self._prepare_item(image.unsqueeze(1), caption)
            for image, caption in zip(latents, caption_embeddings, strict=True)
        ]
        image_values = self.x_embedder(torch.cat([item[0] for item in prepared]))
        image_lengths = [len(item[0]) for item in prepared]
        image_values = list(image_values.split(image_lengths))
        image_masks = [item[4] for item in prepared]
        image, image_freqs, image_mask, image_lengths = self._stage_sequence(
            image_values,
            [item[2] for item in prepared],
            image_masks,
            self.x_pad_token,
            self.rope,
        )
        model_timestep = (1.0 - timestep.float()) * self.config.time_scale
        condition = self.t_embedder(model_timestep).to(image.dtype)
        for layer in self.noise_refiner:
            image = layer(image, image_mask, image_freqs, condition)

        caption_values = self.cap_embedder(torch.cat([item[1] for item in prepared]))
        caption_lengths = [len(item[1]) for item in prepared]
        caption_values = list(caption_values.split(caption_lengths))
        caption_masks = [item[5] for item in prepared]
        caption, caption_freqs, caption_mask, caption_lengths = self._stage_sequence(
            caption_values,
            [item[3] for item in prepared],
            caption_masks,
            self.cap_pad_token,
            self.rope,
        )
        for layer in self.context_refiner:
            caption = layer(caption, caption_mask, caption_freqs)

        unified_values, unified_freqs = [], []
        for index, (image_length, caption_length) in enumerate(
            zip(image_lengths, caption_lengths, strict=True)
        ):
            unified_values.append(
                torch.cat((caption[index, :caption_length], image[index, :image_length]))
            )
            unified_freqs.append(
                torch.cat(
                    (caption_freqs[index, :caption_length], image_freqs[index, :image_length])
                )
            )
        unified = pad_sequence(unified_values, batch_first=True, padding_value=0.0)
        frequencies = pad_sequence(unified_freqs, batch_first=True, padding_value=0.0)
        unified_mask = torch.zeros(unified.shape[:2], dtype=torch.bool, device=unified.device)
        for index, value in enumerate(unified_values):
            unified_mask[index, : len(value)] = True
        for layer in self.layers:
            unified = layer(unified, unified_mask, frequencies, condition)
        output = self.final_layer(unified, condition)

        results = []
        patch = self.config.patch_size
        for index, (_, height, width) in enumerate(item[6] for item in prepared):
            token_count = (height // patch) * (width // patch)
            cap_size = caption_lengths[index]
            item = (
                output[index, cap_size : cap_size + token_count]
                .view(height // patch, width // patch, patch, patch, self.config.in_channels)
                .permute(4, 0, 2, 1, 3)
                .reshape(self.config.in_channels, height, width)
            )
            results.append(item)
        return -torch.stack(results)

    def forward_contract(self) -> dict[str, object]:
        convrot = tuple(
            name
            for name, module in self.named_modules()
            if isinstance(module, ZImageStoredConvRotLinear)
        )
        return {
            "executable": True,
            "stored_category_counts": dict(
                Counter(_z_image_transformer_stored_category(name + ".weight") for name in convrot)
            ),
            "stored_convrot_modules": len(convrot),
            "dense_parameters_are_meta": any(value.is_meta for value in self.state_dict().values()),
            "source_sequence_order": "caption_then_image",
            "image_output_offset": "cap_size",
        }


def build_z_image_nextdit_shell(plan: ZImageTransformerPlan) -> ZImageNextDiTShell:
    """Build and prove the exact pinned architecture from the bounded header."""

    from accelerate import init_empty_weights

    raw, header = _read_z_safetensors_header(plan.identity.path, plan.identity.size_bytes)
    del raw
    bias_keys = _z_image_nextdit_stored_bias_keys(plan, header)
    with init_empty_weights():
        shell = ZImageNextDiTShell()
    sidecars = {layer.scale_key for layer in plan.stored_layers.values()} | {
        layer.marker_key for layer in plan.stored_layers.values()
    }
    expected = {key for key in header if key != "__metadata__" and key not in sidecars}
    if set(shell.state_dict()) != expected:
        missing = sorted(expected - set(shell.state_dict()))
        extra = sorted(set(shell.state_dict()) - expected)
        raise ValueError(
            f"Z-Image fused shell differs from checkpoint: missing={missing[:2]}, extra={extra[:2]}"
        )
    shell._latentslate_z_image_convrot_bias_keys = tuple(sorted(bias_keys))
    return shell


def materialize_z_image_nextdit(
    plan: ZImageTransformerPlan,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> ZImageNextDiTShell:
    """Restore exact dense and stored tensors into the executable fused shell."""

    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    plan.require_stored_layout()
    if not revalidate_artifact(plan.identity):
        raise ValueError("Z-Image transformer changed after planning")
    shell = build_z_image_nextdit_shell(plan)
    bias_keys = frozenset(shell._latentslate_z_image_convrot_bias_keys)
    stored_weights = set(plan.stored_layers)
    with safe_open(str(plan.identity.path), framework="pt", device="cpu") as handle:
        if not revalidate_artifact(plan.identity):
            raise ValueError("Z-Image transformer changed before materialization")
        dense_keys = [
            key for key in shell.state_dict() if key not in stored_weights and key not in bias_keys
        ]
        total = len(dense_keys) + len(plan.stored_layers)
        completed = 0
        for key in dense_keys:
            if cancelled():
                raise RuntimeError("Z-Image NextDiT materialization canceled")
            value = handle.get_tensor(key)
            set_module_tensor_to_device(shell, key, "cpu", value=value, dtype=torch.bfloat16)
            completed += 1
            if progress is not None and completed % 8 == 0:
                progress(completed, total)
        for source, layer in plan.stored_layers.items():
            if cancelled():
                raise RuntimeError("Z-Image NextDiT materialization canceled")
            stem = source.removesuffix(".weight")
            parent, leaf = _parent_for(shell, stem)
            bias = handle.get_tensor(stem + ".bias") if stem + ".bias" in bias_keys else None
            if bias is not None:
                _validate_z_image_nextdit_bias(source, bias, handle.get_tensor(source).shape[0])
            setattr(
                parent,
                leaf,
                ZImageStoredConvRotLinear(
                    restore_stored_quantized_tensor(handle, layer, torch.bfloat16), bias
                ),
            )
            completed += 1
            if progress is not None and (completed == total or completed % 8 == 0):
                progress(completed, total)
    if completed != total:
        raise RuntimeError("Z-Image NextDiT materialization count is incomplete")
    unresolved = [key for key, value in shell.state_dict().items() if value.is_meta]
    if unresolved:
        raise ValueError(f"Z-Image NextDiT retains meta parameters: {unresolved[:2]}")
    names = tuple(
        name
        for name, module in shell.named_modules()
        if isinstance(module, ZImageStoredConvRotLinear)
    )
    if len(names) != 202:
        raise RuntimeError("Z-Image NextDiT did not materialize exactly 202 ConvRot modules")
    categories = Counter(_z_image_transformer_stored_category(name + ".weight") for name in names)
    if dict(categories) != dict(_Z_TRANSFORMER_STORED_CATEGORY_COUNTS):
        raise RuntimeError("Z-Image NextDiT ConvRot categories differ from the exact pin")
    shell._latentslate_z_image_convrot_modules = names
    shell._latentslate_z_image_identity = plan.identity
    shell.eval()
    return shell


def _expected_z_image_nextdit_stored_bias_keys(plan: ZImageTransformerPlan) -> frozenset[str]:
    return frozenset(
        source.removesuffix(".weight") + ".bias"
        for source in plan.stored_layers
        if _z_image_transformer_stored_category(source) == "adaLN_modulation"
    )


def _z_image_nextdit_stored_bias_keys(
    plan: ZImageTransformerPlan, header: Mapping[str, object]
) -> frozenset[str]:
    expected = _expected_z_image_nextdit_stored_bias_keys(plan)
    actual = {
        key
        for key in header
        if key.endswith(".bias") and key.removesuffix(".bias") + ".weight" in plan.stored_layers
    }
    if actual != expected:
        raise ValueError("Z-Image NextDiT stored ConvRot bias stems differ from the exact pin")
    return expected


def _validate_z_image_nextdit_bias(source: str, bias: torch.Tensor, rows: int) -> None:
    if (
        _z_image_transformer_stored_category(source) != "adaLN_modulation"
        or bias.dtype is not torch.float32
        or tuple(bias.shape) != (rows,)
    ):
        raise ValueError(f"Z-Image NextDiT stored ConvRot bias differs from pin: {source}")


def _parent_for(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parent_path, _, leaf = dotted.rpartition(".")
    return root.get_submodule(parent_path), leaf
