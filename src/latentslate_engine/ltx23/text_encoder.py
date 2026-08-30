"""Standalone Gemma 3 text conditioning for the canonical LTX 2.3 T2V fixture."""

from __future__ import annotations

import json
import math

import sentencepiece as spm
import torch
from comfy_kitchen.tensor import (
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorCoreNVFP4Layout,
)
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Gemma3TextConfig, Gemma3TextModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention

from .checkpoint import Ltx23Checkpoint

_SDPA_BACKEND_PRIORITY = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def _comfy_gemma_attention(
    _module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    **_: object,
) -> tuple[torch.Tensor, None]:
    """Pinned Comfy PyTorch attention, including its masked GQA fallback."""
    enable_gqa = query.shape[-3] != key.shape[-3]
    if enable_gqa and attention_mask is not None:
        repeats = query.shape[-3] // key.shape[-3]
        key = key.repeat_interleave(repeats, dim=-3)
        value = value.repeat_interleave(repeats, dim=-3)
        enable_gqa = False
    with sdpa_kernel(_SDPA_BACKEND_PRIORITY, set_priority=True):
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            is_causal=False,
            enable_gqa=enable_gqa,
        )
    return output.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS.register("ltx_comfy_sdpa", _comfy_gemma_attention)


def _replace_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


class _ComfyGemmaEmbedding(nn.Embedding):
    """Gemma embedding with the pinned Comfy float32 lookup and scale."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input_ids,
            self.weight.float(),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        ) * math.sqrt(self.embedding_dim)


class _ComfyGemmaRmsNorm(nn.Module):
    """Gemma RMSNorm preserving Comfy's BF16 `(weight + 1)` rounding."""

    def __init__(self, hidden_size: int, eps: float, *, device: str) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device))
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            input,
            (input.shape[-1],),
            weight=(self.weight + 1.0).to(input),
            eps=self.eps,
        )


class _ComfyGemmaAttention(Gemma3Attention):
    """Gemma attention with pinned Comfy's in-place RoPE arithmetic."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query = self.q_norm(query)
        key = self.k_norm(key)

        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        half = query.shape[-1] // 2
        query_rotated = query * cos
        query_rotated[..., :half].addcmul_(query[..., half:], -sin[..., :half])
        query_rotated[..., half:].addcmul_(query[..., :half], sin[..., half:])
        key_rotated = key * cos
        key_rotated[..., :half].addcmul_(key[..., half:], -sin[..., :half])
        key_rotated[..., half:].addcmul_(key[..., :half], sin[..., half:])

        if past_key_values is not None:
            key_rotated, value = past_key_values.update(
                key_rotated,
                value,
                self.layer_idx,
            )
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            _comfy_gemma_attention,
        )
        output, weights = attention_interface(
            self,
            query_rotated,
            key_rotated,
            value,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        output = output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(output), weights


class _MappedNvfp4Linear(nn.Module):
    """One Gemma NVFP4 linear backed directly by the mapped text checkpoint."""

    def __init__(
        self, checkpoint: Ltx23Checkpoint, prefix: str, shape: tuple[int, ...]
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.shape = shape
        self.qdata = checkpoint.tensor(f"{prefix}.weight")
        self.scale = checkpoint.tensor(f"{prefix}.weight_scale_2")
        self.block_scale = checkpoint.tensor(f"{prefix}.weight_scale")
        if self.qdata.dtype is not torch.uint8 or self.scale.dtype is not torch.float32:
            raise ValueError(f"{prefix} is not an NVFP4 linear")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # AIMDO's direct file-reader rejects this mixed-FP4 checkpoint's slices.
        # Keep its mapped source tensor and use PyTorch's working CUDA copy path.
        qdata = self.qdata.to(device=input.device)
        scale = self.scale.to(device=input.device)
        block_scale = self.block_scale.to(device=input.device).view(torch.float8_e4m3fn)
        weight = QuantizedTensor(
            qdata,
            "TensorCoreNVFP4Layout",
            TensorCoreNVFP4Layout.Params(
                scale=scale,
                block_scale=block_scale,
                orig_dtype=torch.bfloat16,
                orig_shape=self.shape,
            ),
        )
        return F.linear(input, weight.to(dtype=input.dtype).dequantize())


class _MappedFp8Linear(nn.Module):
    """One mixed-checkpoint FP8 Gemma linear backed by the mapped checkpoint."""

    def __init__(
        self, checkpoint: Ltx23Checkpoint, prefix: str, shape: tuple[int, ...]
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.shape = shape
        self.qdata = checkpoint.tensor(f"{prefix}.weight")
        self.scale = checkpoint.tensor(f"{prefix}.weight_scale")
        if (
            self.qdata.dtype is not torch.float8_e4m3fn
            or self.scale.dtype is not torch.float32
        ):
            raise ValueError(f"{prefix} is not an E4M3 FP8 linear")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qdata = self.qdata.to(device=input.device)
        scale = self.scale.to(device=input.device)
        weight = QuantizedTensor(
            qdata,
            "TensorCoreFP8Layout",
            TensorCoreFP8Layout.Params(
                scale=scale,
                orig_dtype=torch.bfloat16,
                orig_shape=self.shape,
            ),
        )
        return F.linear(input, weight.to(dtype=input.dtype).dequantize())


class Ltx23TextEncoder:
    """Own the canonical T2V text encoder for one LTX 2.3 model identity."""

    def __init__(
        self,
        text_checkpoint_path: str,
        model_checkpoint_path: str,
        device_index: int = 0,
    ) -> None:
        self.device = torch.device("cuda", device_index)
        self.text_checkpoint = Ltx23Checkpoint(text_checkpoint_path)
        self.model_checkpoint = Ltx23Checkpoint(model_checkpoint_path)
        self.tokenizer = spm.SentencePieceProcessor()
        if not self.tokenizer.LoadFromSerializedProto(
            self.text_checkpoint.tensor("spiece_model").numpy().tobytes()
        ):
            raise RuntimeError("unable to load the canonical Gemma sentencepiece model")

        config = Gemma3TextConfig(
            vocab_size=262208,
            hidden_size=3840,
            intermediate_size=15360,
            num_hidden_layers=48,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=256,
            rms_norm_eps=1e-6,
            sliding_window=1024,
            rope_parameters={
                "full_attention": {"rope_theta": 1_000_000.0, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"},
            },
        )
        config._attn_implementation = "ltx_comfy_sdpa"
        with torch.device("meta"):
            self.model = Gemma3TextModel(config)
            for index, layer in enumerate(self.model.layers):
                layer.self_attn = _ComfyGemmaAttention(config, index)
        self.model.embed_tokens = _ComfyGemmaEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            device="meta",
        )
        rms_norm_modules = [
            (name, module)
            for name, module in self.model.named_modules()
            if module.__class__.__name__ == "Gemma3RMSNorm"
        ]
        for name, module in rms_norm_modules:
            _replace_module(
                self.model,
                name,
                _ComfyGemmaRmsNorm(config.hidden_size, module.eps, device="meta"),
            )

        dynamic_modules = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        for name, module in dynamic_modules:
            prefix = f"model.{name}"
            quant_name = f"{prefix}.comfy_quant"
            if quant_name not in self.text_checkpoint.tensor_names:
                raise ValueError(f"missing NVFP4 metadata for {prefix}")
            quant_format = json.loads(
                self.text_checkpoint.tensor(quant_name).numpy().tobytes()
            )["format"]
            if quant_format == "nvfp4":
                replacement = _MappedNvfp4Linear(
                    self.text_checkpoint, prefix, tuple(module.weight.shape)
                )
            elif quant_format == "float8_e4m3fn":
                replacement = _MappedFp8Linear(
                    self.text_checkpoint, prefix, tuple(module.weight.shape)
                )
            else:
                raise ValueError(
                    f"unsupported canonical Gemma format {quant_format!r} for {prefix}"
                )
            _replace_module(
                self.model,
                name,
                replacement,
            )

        for name, parameter in list(self.model.named_parameters()):
            source_name = f"model.{name}"
            if source_name not in self.text_checkpoint.tensor_names:
                raise ValueError(f"missing text parameter {source_name}")
            parent = self.model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            source = self.text_checkpoint.tensor(source_name)
            setattr(
                parent,
                parts[-1],
                nn.Parameter(
                    source.to(device=self.device, dtype=torch.bfloat16),
                    requires_grad=False,
                ),
            )
        for attention_type, theta, scale in (
            ("sliding_attention", 10_000.0, 1.0),
            ("full_attention", 1_000_000.0, 8.0),
        ):
            inv_freq = 1.0 / (
                theta
                ** (
                    torch.arange(0, 256, 2, device=self.device, dtype=torch.float32)
                    / 256
                )
            )
            inv_freq.div_(scale)
            setattr(self.model.rotary_emb, f"{attention_type}_inv_freq", inv_freq)
            setattr(
                self.model.rotary_emb,
                f"{attention_type}_original_inv_freq",
                inv_freq.clone(),
            )
        self.model.eval()

    def _tokens(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer.encode(text, out_type=int, add_bos=True, add_eos=False)
        if len(tokens) > 1024:
            raise ValueError("canonical LTX T2V text input exceeds 1024 Gemma tokens")
        padding = 1024 - len(tokens)
        input_ids = torch.tensor(
            [[0] * padding + tokens], device=self.device, dtype=torch.long
        )
        attention_mask = torch.tensor(
            [[0] * padding + [1] * len(tokens)], device=self.device, dtype=torch.long
        )
        return input_ids, attention_mask

    @staticmethod
    def _attention_masks(attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build the pinned Comfy Gemma causal and sliding additive masks."""
        batch, sequence_length = attention_mask.shape
        dtype = torch.float32
        mask = 1.0 - attention_mask.to(dtype).reshape(
            batch, 1, 1, sequence_length
        ).expand(batch, 1, sequence_length, sequence_length)
        mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(dtype).min / 4)
        causal = torch.empty(
            sequence_length,
            sequence_length,
            device=attention_mask.device,
            dtype=dtype,
        ).fill_(torch.finfo(dtype).min / 4)
        mask = mask + causal.triu_(1)
        sliding = torch.full(
            (sequence_length, sequence_length),
            torch.finfo(dtype).min,
            device=attention_mask.device,
            dtype=dtype,
        ).tril_(diagonal=-1024)
        return {
            "full_attention": mask,
            "sliding_attention": mask + sliding,
        }

    @torch.inference_mode()
    def encode(self, text: str) -> torch.Tensor:
        """Return the unprocessed LTX AV text embedding used by the transformer."""
        input_ids, attention_mask = self._tokens(text)
        with sdpa_kernel(_SDPA_BACKEND_PRIORITY, set_priority=True):
            output = self.model(
                input_ids=input_ids,
                attention_mask=self._attention_masks(attention_mask),
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        active_tokens = int(attention_mask.sum().item())
        hidden_states = torch.stack(output.hidden_states, dim=1)[
            :, :, -active_tokens:
        ].to(torch.bfloat16)
        x = hidden_states.movedim(1, -1)
        x = (x * torch.rsqrt(torch.mean(x**2, dim=2, keepdim=True) + 1e-6)).flatten(
            start_dim=2
        )
        source_dim = hidden_states.shape[-1]
        video = self._project(
            "text_embedding_projection.video_aggregate_embed", x, 4096, source_dim
        )
        audio = self._project(
            "text_embedding_projection.audio_aggregate_embed", x, 2048, source_dim
        )
        return torch.cat((video, audio), dim=-1)

    def _project(
        self, prefix: str, x: torch.Tensor, out_features: int, source_dim: int
    ) -> torch.Tensor:
        weight = self.model_checkpoint.tensor(f"{prefix}.weight").to(
            device=self.device, dtype=x.dtype
        )
        bias = self.model_checkpoint.tensor(f"{prefix}.bias").to(
            device=self.device, dtype=x.dtype
        )
        return torch.nn.functional.linear(
            x * math.sqrt(out_features / source_dim), weight, bias
        )

    def close(self) -> None:
        """Drop all text-side state belonging to this LTX 2.3 identity."""
        self.model = None
        self.tokenizer = None
        self.text_checkpoint = None
        self.model_checkpoint = None
        torch.cuda.empty_cache()
