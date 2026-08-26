"""Narrow Engine-owned adaptation of ComfyUI's Gemma-3 12B text substrate.

The math and autoregressive loop in this file follow ComfyUI commit
``12d5279438bfefc058a269eae805ceab6047777f``
(``comfy/text_encoders/llama.py``: ``Gemma3_12B_Config``, ``RMSNorm``, RoPE,
``Attention``, ``MLP``, ``TransformerBlockGemma2``, ``Llama2_``, and
``BaseGenerate``).  The adaptation deliberately excludes ComfyUI's graph
executor, model manager, operations wrappers, prefetch policy, and vision
shell. Engine checkpoint planning, Kitchen stored linears, AIMDO hooks, and
outer lifecycle continue to own those concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class LTX23GemmaOutput:
    """Small text-only output carrying the Diffusers-consumed contract."""

    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    past_key_values: list[tuple[torch.Tensor, torch.Tensor, int]] | None = None
    logits: torch.Tensor | None = None
    next_token: torch.Tensor | None = None
    next_embeddings: torch.Tensor | None = None
    attentions: None = None
    image_hidden_states: None = None

    def __getitem__(self, index: int) -> torch.Tensor:
        if index == 0:
            return self.last_hidden_state
        raise IndexError(index)


class LTX23ComfyRMSNorm(nn.Module):
    """Comfy's fused RMSNorm contract, including Gemma's ``weight + 1``."""

    def __init__(self, dim: int, *, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = (self.weight + 1.0).to(device=value.device, dtype=value.dtype)
        return F.rms_norm(value, weight.shape, weight=weight, eps=self.eps)


class LTX23ComfyScaledEmbedding(nn.Embedding):
    """Gemma embedding with Comfy's optional output cast before scaling."""

    def __init__(self, vocab_size: int, hidden_size: int, *, padding_idx: int) -> None:
        super().__init__(vocab_size, hidden_size, padding_idx=padding_idx)
        self.embed_scale = hidden_size**0.5

    def forward(
        self,
        input_ids: torch.Tensor,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        output = F.embedding(
            input_ids,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        if out_dtype is not None:
            output = output.to(dtype=out_dtype)
        return output * self.embed_scale


def _precompute_freqs_cis(
    head_dim: int,
    position_ids: torch.Tensor,
    *,
    theta: tuple[float, float],
    rope_scale: tuple[float, float],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    output: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for base, scale in zip(theta, rope_scale, strict=True):
        numerator = torch.arange(0, head_dim, 2, device=device).float()
        inv_freq = 1.0 / (base ** (numerator / head_dim))
        inv_freq /= scale
        expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        positions = position_ids[:, None, :].float()
        freqs = (expanded.float() @ positions.float()).transpose(1, 2)
        embedding = torch.cat((freqs, freqs), dim=-1)
        cosine = embedding.cos().unsqueeze(1)
        sine = embedding.sin().unsqueeze(1)
        split = sine.shape[-1] // 2
        output.append((cosine, sine[..., :split], -sine[..., split:]))
    return output


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    original_dtype = query.dtype
    cosine, sine, negative_sine = frequencies

    query_output = query * cosine
    query_split = query_output.shape[-1] // 2
    query_output[..., :query_split].addcmul_(query[..., query_split:], negative_sine)
    query_output[..., query_split:].addcmul_(query[..., :query_split], sine)

    key_output = key * cosine
    key_split = key_output.shape[-1] // 2
    key_output[..., :key_split].addcmul_(key[..., key_split:], negative_sine)
    key_output[..., key_split:].addcmul_(key[..., :key_split], sine)
    return query_output.to(original_dtype), key_output.to(original_dtype)


def _repeat_kv_for_gqa(
    key: torch.Tensor,
    value: torch.Tensor,
    query_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_heads = key.shape[-3]
    if key_heads != value.shape[-3] or query_heads % key_heads:
        raise ValueError("LTX Gemma grouped-query attention geometry changed")
    repeat = query_heads // key_heads
    if repeat > 1:
        key = key.repeat_interleave(repeat, dim=-3)
        value = value.repeat_interleave(repeat, dim=-3)
    return key, value


def _scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Pinned Comfy PyTorch-attention selection without global CLI policy."""

    enable_gqa = query.shape[-3] != key.shape[-3]
    if enable_gqa and mask is not None:
        key, value = _repeat_kv_for_gqa(key, value, query.shape[-3])
        enable_gqa = False
    kwargs: dict[str, Any] = {"enable_gqa": enable_gqa}
    if query.numel() < 1024 * 128 or query.device.type != "cuda":
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            **kwargs,
        )

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        priority = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        with sdpa_kernel(priority, set_priority=True):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                **kwargs,
            )
    except (AttributeError, TypeError):
        # Older compatible PyTorch builds lack priority ordering. Comfy falls
        # back to the same public SDPA primitive in that case.
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            **kwargs,
        )


class LTX23ComfyAttention(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.inner_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        hidden_size = int(config.hidden_size)
        self.q_proj = nn.Linear(hidden_size, self.inner_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.inner_size, hidden_size, bias=False)
        self.q_norm = LTX23ComfyRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LTX23ComfyRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor, int] | tuple[()],
        sliding_window: int | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, int]]:
        batch, sequence, _ = hidden_states.shape
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = query.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = _apply_rope(query, key, freqs_cis)

        index = 0
        token_count = key.shape[2]
        if past_key_value:
            past_key, past_value, index = past_key_value
            if past_key.shape[2] >= index + token_count:
                past_key[:, :, index : index + token_count] = key
                past_value[:, :, index : index + token_count] = value
                key = past_key[:, :, : index + token_count]
                value = past_value[:, :, : index + token_count]
                present = (past_key, past_value, index + token_count)
            else:
                key = torch.cat((past_key[:, :, :index], key), dim=2)
                value = torch.cat((past_value[:, :, :index], value), dim=2)
                present = (key, value, index + token_count)
        else:
            present = (key, value, token_count)

        if sliding_window is not None and key.shape[2] > sliding_window and sequence == 1:
            key = key[:, :, -sliding_window:]
            value = value[:, :, -sliding_window:]
            if attention_mask is not None:
                attention_mask = attention_mask[..., -sliding_window:]

        output = _scaled_dot_product_attention(query, key, value, attention_mask)
        output = output.transpose(1, 2).reshape(batch, sequence, self.inner_size)
        return self.o_proj(output), present


class LTX23ComfyMLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden = int(config.hidden_size)
        intermediate = int(config.intermediate_size)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(value), approximate="tanh")
        return self.down_proj(gate * self.up_proj(value))


class LTX23ComfyDecoderLayer(nn.Module):
    def __init__(self, config: Any, index: int) -> None:
        super().__init__()
        hidden = int(config.hidden_size)
        eps = float(config.rms_norm_eps)
        self.self_attn = LTX23ComfyAttention(config)
        self.mlp = LTX23ComfyMLP(config)
        self.input_layernorm = LTX23ComfyRMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = LTX23ComfyRMSNorm(hidden, eps=eps)
        self.pre_feedforward_layernorm = LTX23ComfyRMSNorm(hidden, eps=eps)
        self.post_feedforward_layernorm = LTX23ComfyRMSNorm(hidden, eps=eps)
        layer_types = tuple(config.layer_types)
        self.sliding_attention = (
            int(config.sliding_window)
            if layer_types[index] == "sliding_attention"
            else 0
        )

    def forward(
        self,
        value: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        freqs_cis: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        past_key_value: tuple[torch.Tensor, torch.Tensor, int] | tuple[()],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, int]]:
        output = value
        sliding_window: int | None = None
        if self.sliding_attention:
            sliding_window = self.sliding_attention
            if value.shape[1] > sliding_window:
                sliding_mask = torch.full(
                    (value.shape[1], value.shape[1]),
                    torch.finfo(value.dtype).min,
                    device=value.device,
                    dtype=value.dtype,
                )
                sliding_mask.tril_(diagonal=-sliding_window)
                attention_mask = (
                    sliding_mask
                    if attention_mask is None
                    else attention_mask + sliding_mask
                )
            frequencies = freqs_cis[1]
        else:
            frequencies = freqs_cis[0]

        residual = value
        value = self.input_layernorm(value)
        value, present = self.self_attn(
            value,
            attention_mask=attention_mask,
            freqs_cis=frequencies,
            past_key_value=past_key_value,
            sliding_window=sliding_window,
        )
        value = self.post_attention_layernorm(value)
        value = residual + value

        residual = value
        value = self.pre_feedforward_layernorm(value)
        value = self.mlp(value)
        value = self.post_feedforward_layernorm(value)
        value = torch.add(residual, value, out=output)
        return value, present


class LTX23ComfyLanguageModel(nn.Module):
    """Text-only pinned Comfy Gemma stack with the existing module namespace."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.embed_tokens = LTX23ComfyScaledEmbedding(
            self.vocab_size,
            int(config.hidden_size),
            padding_idx=int(getattr(config, "pad_token_id", 0) or 0),
        )
        self.layers = nn.ModuleList(
            LTX23ComfyDecoderLayer(config, index)
            for index in range(int(config.num_hidden_layers))
        )
        self.norm = LTX23ComfyRMSNorm(
            int(config.hidden_size), eps=float(config.rms_norm_eps)
        )
        rope_parameters = config.rope_parameters
        full_rope = rope_parameters["full_attention"]
        local_rope = rope_parameters["sliding_attention"]
        self.rope_theta = (
            float(full_rope["rope_theta"]),
            float(local_rope["rope_theta"]),
        )
        self.rope_scale = (float(full_rope.get("factor", 1.0)), 1.0)
        self.head_dim = int(config.head_dim)
        self.num_kv_heads = int(config.num_key_value_heads)

    def init_kv_cache(
        self,
        batch: int,
        capacity: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        caches = []
        for _ in self.layers:
            key = torch.empty(
                (batch, self.num_kv_heads, capacity, self.head_dim),
                device=device,
                dtype=dtype,
            )
            caches.append((key, torch.empty_like(key), 0))
        return caches

    @staticmethod
    def _past_length(
        past_key_values: list[tuple[torch.Tensor, torch.Tensor, int]] | None,
    ) -> int:
        return 0 if not past_key_values else int(past_key_values[0][2])

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor, int]] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_kwargs: Any,
    ) -> LTX23GemmaOutput:
        del use_cache, return_dict
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("LTX Gemma requires exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids, out_dtype=torch.float32)
        value = inputs_embeds
        sequence = value.shape[1]
        past_length = self._past_length(past_key_values)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + sequence,
                device=value.device,
            ).unsqueeze(0)
        freqs_cis = _precompute_freqs_cis(
            self.head_dim,
            position_ids,
            theta=self.rope_theta,
            rope_scale=self.rope_scale,
            device=value.device,
        )

        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(value.dtype).reshape(
                (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
            ).expand(attention_mask.shape[0], 1, sequence, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(value.dtype).min / 4)
        if sequence > 1:
            causal = torch.empty(
                past_length + sequence,
                past_length + sequence,
                dtype=value.dtype,
                device=value.device,
            ).fill_(torch.finfo(value.dtype).min / 4).triu_(1)
            mask = causal if mask is None else mask + causal

        hidden_states: list[torch.Tensor] | None = [] if output_hidden_states else None
        next_key_values = list(past_key_values) if past_key_values is not None else []
        for index, layer in enumerate(self.layers):
            if hidden_states is not None:
                hidden_states.append(value.clone())
            past = () if past_key_values is None else past_key_values[index]
            value, present = layer(
                value,
                attention_mask=mask,
                freqs_cis=freqs_cis,
                past_key_value=past,
            )
            if next_key_values:
                next_key_values[index] = present
        value = self.norm(value)
        if hidden_states is not None:
            hidden_states.append(value.clone())
        return LTX23GemmaOutput(
            last_hidden_state=value,
            hidden_states=None if hidden_states is None else tuple(hidden_states),
            past_key_values=next_key_values or None,
        )


class _LTX23ComfyTextContainer(nn.Module):
    def __init__(self, language_model: LTX23ComfyLanguageModel) -> None:
        super().__init__()
        self.language_model = language_model


class LTX23ComfyGemmaForConditionalGeneration(nn.Module):
    """Text-only public shell retaining Gemma's established Engine namespace."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        text_config = config.text_config
        self.model = _LTX23ComfyTextContainer(LTX23ComfyLanguageModel(text_config))
        self.lm_head = nn.Linear(
            int(text_config.hidden_size), int(text_config.vocab_size), bias=False
        )
        self.tie_weights()
        self._latentslate_comfy_gemma_direct = True

    @property
    def dtype(self) -> torch.dtype:
        return self.model.language_model.embed_tokens.weight.dtype

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Module:
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        self.model.language_model.embed_tokens = embeddings
        self.tie_weights()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor, int]] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool = False,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> LTX23GemmaOutput:
        if pixel_values is not None or labels is not None:
            raise ValueError("LTX Gemma direct substrate is text-only")
        generation_step = kwargs.pop("_latentslate_generation_step", None)
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("LTX Gemma requires input_ids or inputs_embeds")
            inputs_embeds = self.model.language_model.embed_tokens(
                input_ids, out_dtype=torch.float32
            )
            if generation_step is not None:
                inputs_embeds = inputs_embeds.to(generation_step["execution_dtype"])
        output = self.model.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        if generation_step is not None:
            output.logits = self._logits(output.last_hidden_state)[:, -1]
            output.next_token = self._sample_token(
                output.logits,
                temperature=generation_step["temperature"],
                top_k=generation_step["top_k"],
                top_p=generation_step["top_p"],
                min_p=generation_step["min_p"],
                repetition_penalty=generation_step["repetition_penalty"],
                token_history=generation_step["token_history"],
                generator=generation_step["generator"],
                do_sample=generation_step["do_sample"],
            )
            output.next_embeddings = self.model.language_model.embed_tokens(
                output.next_token
            ).to(generation_step["execution_dtype"])
        return output

    def _logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        value = hidden_states[:, -1:]
        embedding = self.model.language_model.embed_tokens
        tied_logits = getattr(embedding, "tied_logits", None)
        if callable(tied_logits):
            return tied_logits(value)
        return F.linear(value, embedding.weight)

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        token_history: list[int],
        generator: torch.Generator | None,
        do_sample: bool,
    ) -> torch.Tensor:
        if not do_sample or temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        if token_history and repetition_penalty != 1.0:
            token_ids = torch.tensor(list(set(token_history)), device=logits.device)
            token_logits = logits[:, token_ids]
            token_logits = torch.where(
                token_logits < 0,
                token_logits * repetition_penalty,
                token_logits / repetition_penalty,
            )
            logits[:, token_ids] = token_logits
        if temperature != 1.0:
            logits = logits / temperature
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            logits, top_indices = torch.topk(logits, top_k)
            if min_p > 0.0:
                probabilities = F.softmax(logits, dim=-1)
                threshold = min_p * probabilities.max(dim=-1, keepdim=True).values
                logits[probabilities < threshold] = torch.finfo(logits.dtype).min
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_remove = cumulative > top_p
                sorted_remove[..., 0] = False
                remove = torch.zeros_like(logits, dtype=torch.bool)
                remove.scatter_(1, sorted_indices, sorted_remove)
                logits[remove] = torch.finfo(logits.dtype).min
            probabilities = F.softmax(logits, dim=-1)
            selected = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            )
            return top_indices.gather(1, selected)
        if min_p > 0.0:
            probabilities = F.softmax(logits, dim=-1)
            threshold = min_p * probabilities.max(dim=-1, keepdim=True).values
            logits[probabilities < threshold] = torch.finfo(logits.dtype).min
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cumulative > top_p
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(logits, dtype=torch.bool)
            remove.scatter_(1, sorted_indices, sorted_remove)
            logits[remove] = torch.finfo(logits.dtype).min
        return torch.multinomial(
            F.softmax(logits, dim=-1), num_samples=1, generator=generator
        )

    @torch.inference_mode()
    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int,
        eos_token_id: int | list[int],
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        seed: int,
        execution_dtype: torch.dtype,
        check_cancelled: Any | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        del attention_mask
        language = self.model.language_model
        device = input_ids.device
        caches = language.init_kv_cache(
            input_ids.shape[0], input_ids.shape[1] + max_new_tokens, device, execution_dtype
        )
        generator = torch.Generator(device=device).manual_seed(seed) if do_sample else None
        stop_tokens = (
            {int(eos_token_id)}
            if isinstance(eos_token_id, int)
            else {int(token) for token in eos_token_id}
        )
        generated: list[int] = []
        embeddings = None
        for step in range(max_new_tokens):
            if check_cancelled is not None:
                check_cancelled()
            output = self(
                input_ids=input_ids if step == 0 else None,
                inputs_embeds=embeddings,
                attention_mask=None,
                past_key_values=caches,
                use_cache=True,
                output_hidden_states=False,
                _latentslate_generation_step={
                    "execution_dtype": execution_dtype,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "min_p": min_p,
                    "repetition_penalty": repetition_penalty,
                    "token_history": generated,
                    "generator": generator,
                    "do_sample": do_sample,
                },
            )
            if output.past_key_values is None:
                raise RuntimeError("LTX Gemma generation cache was not advanced")
            caches = output.past_key_values
            next_token = output.next_token
            embeddings = output.next_embeddings
            if next_token is None or embeddings is None:
                raise RuntimeError("LTX Gemma generation step did not publish its token")
            token_id = int(next_token[0].item())
            generated.append(token_id)
            if token_id in stop_tokens:
                break
        suffix = torch.tensor([generated], device=device, dtype=input_ids.dtype)
        return torch.cat((input_ids, suffix), dim=1)


def build_ltx23_comfy_gemma(config: Any) -> LTX23ComfyGemmaForConditionalGeneration:
    """Build the production text-only shell from a Gemma3 outer config."""

    text = getattr(config, "text_config", None)
    required = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "layer_types",
        "sliding_window",
        "rms_norm_eps",
        "rope_parameters",
    )
    if text is None or any(not hasattr(text, name) for name in required):
        raise TypeError("LTX Gemma config does not expose the pinned text contract")
    return LTX23ComfyGemmaForConditionalGeneration(config)
