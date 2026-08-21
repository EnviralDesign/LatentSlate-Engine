"""Pure Qwen3-4B shell and conditioning mathematics for Z-Image."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F

QWEN_WEIGHT_COUNT = 398
QWEN_BLOCK_COUNT = 36
QWEN_LINEAR_COUNT = 7 * QWEN_BLOCK_COUNT
QWEN_CAPTURE_BLOCK = 34
QWEN_HIDDEN_SIZE = 2560
QWEN_INTERMEDIATE_SIZE = 9728
QWEN_QUERY_HEADS = 32
QWEN_KV_HEADS = 8
QWEN_HEAD_DIM = 128
QWEN_ROPE_THETA = 1_000_000.0
QWEN_NORM_EPS = 1e-6
QWEN_VOCAB_SIZE = 151936

Cancel = Callable[[], bool]
Diagnostic = Callable[[str], None]


def checkpoint(cancelled: Cancel, diagnostic: Diagnostic, stage: str) -> None:
    diagnostic(stage)
    if cancelled():
        raise RuntimeError("Z-Image Qwen conditioning canceled")


class ZImageQwenModuleFactory(Protocol):
    """Construct storage/execution modules without coupling the shell to residency."""

    def linear(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | str,
        ordinal: int,
    ) -> nn.Module: ...

    def embedding(self, *, device: torch.device | str) -> nn.Module: ...

    def norm(self, width: int, *, device: torch.device | str) -> nn.Module: ...


class _ShellLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | str,
        ordinal: int,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.ordinal = ordinal

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight)


class _ShellEmbedding(nn.Module):
    def __init__(self, *, device: torch.device | str) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                (QWEN_VOCAB_SIZE, QWEN_HIDDEN_SIZE),
                device=device,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )

    def forward(self, input_ids: torch.Tensor, *, out_dtype: torch.dtype) -> torch.Tensor:
        return F.embedding(input_ids, self.weight).to(dtype=out_dtype)


class _ShellRMSNorm(nn.Module):
    def __init__(self, width: int, *, device: torch.device | str) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(width, device=device, dtype=torch.bfloat16), requires_grad=False
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            value,
            (value.shape[-1],),
            weight=self.weight,
            eps=QWEN_NORM_EPS,
        )


class _ShellFactory:
    def linear(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | str,
        ordinal: int,
    ) -> nn.Module:
        return _ShellLinear(
            in_features, out_features, device=device, ordinal=ordinal
        )

    def embedding(self, *, device: torch.device | str) -> nn.Module:
        return _ShellEmbedding(device=device)

    def norm(self, width: int, *, device: torch.device | str) -> nn.Module:
        return _ShellRMSNorm(width, device=device)


def qwen_rope(
    position_ids: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, ...]:
    numerator = torch.arange(0, QWEN_HEAD_DIM, 2, device=device).float()
    inv_freq = 1.0 / (QWEN_ROPE_THETA ** (numerator / QWEN_HEAD_DIM))
    expanded_freq = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    expanded_position = position_ids[:, None, :].float()
    frequencies = (expanded_freq.float() @ expanded_position.float()).transpose(1, 2)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cosine = embedding.cos().unsqueeze(1)
    sine = embedding.sin().unsqueeze(1)
    split = sine.shape[-1] // 2
    return cosine, sine[..., :split], -sine[..., split:]


def apply_qwen_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    frequencies: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    original_dtype = query.dtype
    cosine, sine, negative_sine = frequencies
    query_out = query * cosine
    query_split = query_out.shape[-1] // 2
    query_out[..., :query_split].addcmul_(query[..., query_split:], negative_sine)
    query_out[..., query_split:].addcmul_(query[..., :query_split], sine)
    key_out = key * cosine
    key_split = key_out.shape[-1] // 2
    key_out[..., :key_split].addcmul_(key[..., key_split:], negative_sine)
    key_out[..., key_split:].addcmul_(key[..., :key_split], sine)
    return query_out.to(original_dtype), key_out.to(original_dtype)


def qwen_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    additive_mask: torch.Tensor,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=additive_mask,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=True,
    )


class ZImageQwenAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device | str,
        first_ordinal: int,
        factory: ZImageQwenModuleFactory,
    ) -> None:
        super().__init__()
        self.q_proj = factory.linear(
            QWEN_HIDDEN_SIZE,
            QWEN_QUERY_HEADS * QWEN_HEAD_DIM,
            device=device,
            ordinal=first_ordinal,
        )
        self.k_proj = factory.linear(
            QWEN_HIDDEN_SIZE,
            QWEN_KV_HEADS * QWEN_HEAD_DIM,
            device=device,
            ordinal=first_ordinal + 1,
        )
        self.v_proj = factory.linear(
            QWEN_HIDDEN_SIZE,
            QWEN_KV_HEADS * QWEN_HEAD_DIM,
            device=device,
            ordinal=first_ordinal + 2,
        )
        self.o_proj = factory.linear(
            QWEN_QUERY_HEADS * QWEN_HEAD_DIM,
            QWEN_HIDDEN_SIZE,
            device=device,
            ordinal=first_ordinal + 3,
        )
        self.q_norm = factory.norm(QWEN_HEAD_DIM, device=device)
        self.k_norm = factory.norm(QWEN_HEAD_DIM, device=device)

    def forward(
        self,
        hidden: torch.Tensor,
        additive_mask: torch.Tensor,
        frequencies: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        query = self.q_proj(hidden).view(
            batch, sequence, QWEN_QUERY_HEADS, QWEN_HEAD_DIM
        ).transpose(1, 2)
        key = self.k_proj(hidden).view(
            batch, sequence, QWEN_KV_HEADS, QWEN_HEAD_DIM
        ).transpose(1, 2)
        value = self.v_proj(hidden).view(
            batch, sequence, QWEN_KV_HEADS, QWEN_HEAD_DIM
        ).transpose(1, 2)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = apply_qwen_rope(query, key, frequencies)
        attended = qwen_gqa_attention(query, key, value, additive_mask)
        attended = attended.transpose(1, 2).reshape(
            batch, sequence, QWEN_QUERY_HEADS * QWEN_HEAD_DIM
        )
        return self.o_proj(attended)


class ZImageQwenMLP(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device | str,
        first_ordinal: int,
        factory: ZImageQwenModuleFactory,
    ) -> None:
        super().__init__()
        self.gate_proj = factory.linear(
            QWEN_HIDDEN_SIZE,
            QWEN_INTERMEDIATE_SIZE,
            device=device,
            ordinal=first_ordinal,
        )
        self.up_proj = factory.linear(
            QWEN_HIDDEN_SIZE,
            QWEN_INTERMEDIATE_SIZE,
            device=device,
            ordinal=first_ordinal + 1,
        )
        self.down_proj = factory.linear(
            QWEN_INTERMEDIATE_SIZE,
            QWEN_HIDDEN_SIZE,
            device=device,
            ordinal=first_ordinal + 2,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class ZImageQwenBlock(nn.Module):
    def __init__(
        self,
        index: int,
        *,
        device: torch.device | str,
        factory: ZImageQwenModuleFactory,
    ) -> None:
        super().__init__()
        first = index * 7
        self.self_attn = ZImageQwenAttention(
            device=device, first_ordinal=first, factory=factory
        )
        self.mlp = ZImageQwenMLP(
            device=device, first_ordinal=first + 4, factory=factory
        )
        self.input_layernorm = factory.norm(QWEN_HIDDEN_SIZE, device=device)
        self.post_attention_layernorm = factory.norm(QWEN_HIDDEN_SIZE, device=device)

    def forward(
        self,
        hidden: torch.Tensor,
        additive_mask: torch.Tensor,
        frequencies: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        output = hidden
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = residual + self.self_attn(hidden, additive_mask, frequencies)
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return torch.add(residual, hidden, out=output)


class ZImageQwenBackbone(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device | str = "meta",
        factory: ZImageQwenModuleFactory,
    ) -> None:
        super().__init__()
        self.embed_tokens = factory.embedding(device=device)
        self.layers = nn.ModuleList(
            ZImageQwenBlock(index, device=device, factory=factory)
            for index in range(QWEN_BLOCK_COUNT)
        )
        self.norm = factory.norm(QWEN_HIDDEN_SIZE, device=device)


class ZImageQwenTextEncoder(nn.Module):
    """Raw Qwen shell with exact ``model.*`` state keys and block-34 capture."""

    def __init__(
        self,
        *,
        device: torch.device | str = "meta",
        factory: ZImageQwenModuleFactory | None = None,
    ) -> None:
        super().__init__()
        self.model = ZImageQwenBackbone(
            device=device, factory=_ShellFactory() if factory is None else factory
        )
        self.final_norm_execution_count = 0

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    @torch.inference_mode()
    def forward_conditioning(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        cancelled: Cancel = lambda: False,
        diagnostic: Diagnostic = lambda _stage: None,
    ) -> torch.Tensor:
        if input_ids.dtype is not torch.long or input_ids.ndim != 2:
            raise TypeError("Z-Image Qwen input IDs must be int64 [B,L]")
        if attention_mask.dtype is not torch.long or attention_mask.shape != input_ids.shape:
            raise TypeError("Z-Image Qwen attention mask must be int64 [B,L]")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("Z-Image Qwen attention mask must be binary")
        device = input_ids.device
        context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with context:
            for module in self.modules():
                if hasattr(module, "set_runtime_callbacks"):
                    module.set_runtime_callbacks(cancelled, diagnostic)
            checkpoint(cancelled, diagnostic, "conditioning.edge_12")
            hidden = self.model.embed_tokens(input_ids, out_dtype=torch.float32)
            if hidden.dtype is not torch.float32:
                raise RuntimeError("Z-Image Qwen embedding output is not F32")

            checkpoint(cancelled, diagnostic, "conditioning.edge_15")
            batch, sequence = attention_mask.shape
            additive_mask = 1.0 - attention_mask.to(hidden.dtype).reshape(
                batch, 1, -1, sequence
            ).expand(batch, 1, sequence, sequence)
            additive_mask = additive_mask.masked_fill(
                additive_mask.to(torch.bool), torch.finfo(hidden.dtype).min / 4
            )
            causal = torch.empty(
                sequence, sequence, device=device, dtype=hidden.dtype
            ).fill_(torch.finfo(hidden.dtype).min / 4).triu_(1)
            additive_mask += causal

            checkpoint(cancelled, diagnostic, "conditioning.edge_16")
            position_ids = torch.arange(sequence, device=device).unsqueeze(0)
            checkpoint(cancelled, diagnostic, "conditioning.edge_17")
            frequencies = qwen_rope(position_ids, device)
            intermediate: torch.Tensor | None = None
            for index, layer in enumerate(self.model.layers):
                checkpoint(cancelled, diagnostic, f"conditioning.block_{index:02d}")
                hidden = layer(hidden, additive_mask, frequencies)
                if index == QWEN_CAPTURE_BLOCK:
                    diagnostic("conditioning.edge_18")
                    intermediate = hidden.clone()
            checkpoint(cancelled, diagnostic, "conditioning.edge_19")
            hidden = self.model.norm(hidden)
            self.final_norm_execution_count += 1
            del hidden
        checkpoint(cancelled, diagnostic, "conditioning.edge_20")
        if intermediate is None:
            raise RuntimeError("Z-Image Qwen did not capture block 34")
        return intermediate.float()
