"""E2B language blocks adapted from ComfyUI 1a14b82e gemma4.py."""

from contextlib import nullcontext
from dataclasses import dataclass

import comfy_kitchen as ck
import torch
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .enhancer_vision import RMSNorm, rms_norm


class E2BEmbedding(nn.Embedding):
    """Use Kitchen's selected-row dequantization, as in the reference enhancer."""

    def forward(self, input, out_dtype=None):
        weight = self.weight
        if isinstance(weight, QuantizedTensor):
            output = TensorWiseINT8Layout.dequantize_embedding(
                weight._qdata, weight._params, input
            )
        else:
            output = super().forward(input)
        return output.to(dtype=out_dtype) if out_dtype is not None else output


@dataclass
class E2BConfig:
    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_hidden_layers: int = 35
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    num_global_key_value_heads: int | None = None
    head_dim: int = 256
    global_head_dim: int = 512
    qkv_bias: bool = False
    q_norm: str = "gemma3"
    k_norm: str = "gemma3"
    rms_norm_eps: float = 1e-6
    attention_k_eq_v: bool = False
    num_kv_shared_layers: int = 20
    hidden_size_per_layer_input: int = 256
    use_double_wide_mlp: bool = True
    sliding_attention: tuple = (512, 512, 512, 512, False)


@dataclass
class FixedKV:
    key: torch.Tensor
    value: torch.Tensor
    index: int
    position: torch.Tensor
    seqlen: torch.Tensor

    def prepare(self, num_tokens):
        self.position.copy_(self.seqlen)
        self.seqlen.add_(num_tokens)

    def advance(self, num_tokens):
        self.index += num_tokens


class RingKV(FixedKV):
    def prepare(self, num_tokens):
        capacity = self.key.shape[2]
        self.position.fill_(self.index % capacity)
        self.seqlen.fill_(min(self.index + num_tokens, capacity))


def _attention(
    q, k, v, heads, mask=None, skip_reshape=True, scale=1.0, enable_gqa=False
):
    backend = (
        sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ],
            set_priority=True,
        )
        if q.nelement() >= 1024 * 128
        else nullcontext()
    )
    with backend:
        if (
            enable_gqa
            and mask is not None
            and q.shape[1] != k.shape[1]
            and q.nelement() >= 1024 * 128
        ):
            params = torch.backends.cuda.SDPAParams(q, k, v, mask, 0.0, False, True)
            supported = (
                torch.backends.cuda.can_use_flash_attention(params)
                or torch.backends.cuda.can_use_cudnn_attention(params)
                or torch.backends.cuda.can_use_efficient_attention(params)
            )
            if not supported:
                k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
                v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
                enable_gqa = False
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=enable_gqa
        )
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


class MLP(nn.Module):
    def __init__(
        self, config, device=None, dtype=None, ops=None, intermediate_size=None
    ):
        super().__init__()
        size = intermediate_size or config.intermediate_size
        self.gate_proj = ops.Linear(
            config.hidden_size, size, bias=False, device=device, dtype=dtype
        )
        self.up_proj = ops.Linear(
            config.hidden_size, size, bias=False, device=device, dtype=dtype
        )
        self.down_proj = ops.Linear(
            size, config.hidden_size, bias=False, device=device, dtype=dtype
        )

    def forward(self, x):
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


class Gemma4Attention(nn.Module):
    def __init__(
        self,
        config,
        head_dim,
        num_kv_heads=None,
        k_eq_v=False,
        device=None,
        dtype=None,
        ops=None,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = (
            num_kv_heads if num_kv_heads is not None else config.num_key_value_heads
        )
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.inner_size = self.num_heads * head_dim
        self.q_proj = ops.Linear(
            config.hidden_size,
            self.inner_size,
            bias=config.qkv_bias,
            device=device,
            dtype=dtype,
        )
        self.k_proj = ops.Linear(
            config.hidden_size,
            self.num_kv_heads * head_dim,
            bias=config.qkv_bias,
            device=device,
            dtype=dtype,
        )
        self.v_proj = (
            None
            if k_eq_v
            else ops.Linear(
                config.hidden_size,
                self.num_kv_heads * head_dim,
                bias=config.qkv_bias,
                device=device,
                dtype=dtype,
            )
        )
        self.o_proj = ops.Linear(
            self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype
        )
        self.q_norm = None
        self.k_norm = None
        if config.q_norm == "gemma3":
            self.q_norm = RMSNorm(
                head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype
            )
        if config.k_norm == "gemma3":
            self.k_norm = RMSNorm(
                head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype
            )

    def _decode_attention(self, xq, cache, bias):
        if bias is None:
            n = min(cache.index + 1, cache.key.shape[2])
            gqa_kwargs = (
                {"enable_gqa": True} if self.num_heads != self.num_kv_heads else {}
            )
            attention = _attention
            return attention(
                xq,
                cache.key[:, :, :n],
                cache.value[:, :, :n],
                self.num_heads,
                skip_reshape=True,
                scale=1.0,
                **gqa_kwargs,
            )
        batch_size = xq.shape[0]
        groups = self.num_heads // self.num_kv_heads
        q = xq.reshape(batch_size, self.num_kv_heads, groups, self.head_dim)
        scores = q @ cache.key.transpose(-1, -2) + bias
        probs = torch.softmax(scores.float(), dim=-1).to(xq.dtype)
        return (probs @ cache.value).reshape(batch_size, 1, self.inner_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        freqs_cis=None,
        past_key_value=None,
        sliding_window=None,
        shared_kv=None,
    ):
        batch_size, seq_length, _ = hidden_states.shape
        xq = self.q_proj(hidden_states)
        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if isinstance(shared_kv, FixedKV):
            xq = ck.apply_rope_split_half1(xq, freqs_cis)
            output = self._decode_attention(xq, shared_kv, attention_mask)
            return (self.o_proj(output), None, None)
        if shared_kv is not None:
            xk, xv = shared_kv
            xq = ck.apply_rope_split_half1(xq, freqs_cis)
            present_key_value = None
            shareable_kv = None
        else:
            xk = self.k_proj(hidden_states).view(
                batch_size, seq_length, self.num_kv_heads, self.head_dim
            )
            if self.v_proj is not None:
                xv = self.v_proj(hidden_states).view(
                    batch_size, seq_length, self.num_kv_heads, self.head_dim
                )
            else:
                xv = xk
            if self.k_norm is not None:
                xk = self.k_norm(xk)
            xv = rms_norm(xv)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)
            xq = ck.apply_rope_split_half1(xq, freqs_cis)
            xk = ck.apply_rope_split_half1(xk, freqs_cis)
            present_key_value = None
            fixed_cache = (
                past_key_value if isinstance(past_key_value, FixedKV) else None
            )
            if fixed_cache is not None:
                if seq_length == 1 and fixed_cache.index > 0:
                    fixed_cache.key.index_copy_(2, fixed_cache.position, xk)
                    fixed_cache.value.index_copy_(2, fixed_cache.position, xv)
                    output = self._decode_attention(xq, fixed_cache, attention_mask)
                    return (self.o_proj(output), fixed_cache, None)
                capacity = fixed_cache.key.shape[2]
                index = fixed_cache.index
                if index + seq_length <= capacity:
                    fixed_cache.key[:, :, index : index + seq_length] = xk
                    fixed_cache.value[:, :, index : index + seq_length] = xv
                    if index > 0:
                        xk = fixed_cache.key[:, :, : index + seq_length]
                        xv = fixed_cache.value[:, :, : index + seq_length]
                elif index == 0:
                    slots = (
                        torch.arange(
                            seq_length - capacity, seq_length, device=xk.device
                        )
                        % capacity
                    )
                    fixed_cache.key.index_copy_(2, slots, xk[:, :, -capacity:])
                    fixed_cache.value.index_copy_(2, slots, xv[:, :, -capacity:])
                else:
                    raise RuntimeError(
                        "gemma4: chunked prefill past the sliding window is not supported"
                    )
                present_key_value = fixed_cache
            elif past_key_value is not None:
                cumulative_len = 0
                if len(past_key_value) > 0:
                    past_key, past_value, cumulative_len = past_key_value
                    xk = torch.cat((past_key, xk), dim=2)
                    xv = torch.cat((past_value, xv), dim=2)
                new_cumulative = cumulative_len + seq_length
                if sliding_window is not None and xk.shape[2] > sliding_window - 1:
                    cache_k = xk[:, :, -(sliding_window - 1) :]
                    cache_v = xv[:, :, -(sliding_window - 1) :]
                else:
                    cache_k = xk
                    cache_v = xv
                present_key_value = (cache_k, cache_v, new_cumulative)
            shareable_kv = (xk, xv)
        expand_kv = (
            self.num_heads != self.num_kv_heads
            and sliding_window is not None
            and (xk.shape[2] >= sliding_window)
        )
        if expand_kv:
            xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        gqa_kwargs = (
            {}
            if expand_kv
            else {"enable_gqa": True}
            if self.num_heads != self.num_kv_heads
            else {}
        )
        output = _attention(
            xq,
            xk,
            xv,
            self.num_heads,
            mask=attention_mask,
            skip_reshape=True,
            scale=1.0,
            **gqa_kwargs,
        )
        return (self.o_proj(output), present_key_value, shareable_kv)


class TransformerBlockGemma4(nn.Module):
    def __init__(self, config, index, device=None, dtype=None, ops=None):
        super().__init__()
        if config.sliding_attention is not None:
            self.sliding_attention = config.sliding_attention[
                index % len(config.sliding_attention)
            ]
        else:
            self.sliding_attention = False
        head_dim = config.head_dim if self.sliding_attention else config.global_head_dim
        k_eq_v = config.attention_k_eq_v and (not self.sliding_attention)
        num_kv_heads = (
            config.num_global_key_value_heads if k_eq_v else config.num_key_value_heads
        )
        self.self_attn = Gemma4Attention(
            config,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            k_eq_v=k_eq_v,
            device=device,
            dtype=dtype,
            ops=ops,
        )
        num_kv_shared = config.num_kv_shared_layers
        first_kv_shared = config.num_hidden_layers - num_kv_shared
        mlp_size = (
            config.intermediate_size * 2
            if config.use_double_wide_mlp and index >= first_kv_shared
            else None
        )
        self.mlp = MLP(
            config, device=device, dtype=dtype, ops=ops, intermediate_size=mlp_size
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype
        )
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = ops.Linear(
                config.hidden_size,
                self.hidden_size_per_layer_input,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.per_layer_projection = ops.Linear(
                self.hidden_size_per_layer_input,
                config.hidden_size,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.post_per_layer_input_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype
            )
        self.register_buffer("layer_scalar", torch.empty(1, device=device, dtype=dtype))

    def forward(
        self,
        x,
        attention_mask=None,
        freqs_cis=None,
        past_key_value=None,
        per_layer_input=None,
        shared_kv=None,
    ):
        output = x
        sliding_window = None
        if self.sliding_attention:
            sliding_window = self.sliding_attention
            if x.shape[1] > self.sliding_attention:
                sw_mask = torch.zeros(
                    x.shape[1], x.shape[1], dtype=x.dtype, device=x.device
                )
                sw_mask.masked_fill_(
                    torch.ones_like(sw_mask, dtype=torch.bool).tril_(
                        -self.sliding_attention
                    ),
                    torch.finfo(x.dtype).min,
                )
                attention_mask = (
                    attention_mask + sw_mask if attention_mask is not None else sw_mask
                )
            freqs_cis = freqs_cis[1]
        else:
            freqs_cis = freqs_cis[0]
        residual = x
        x = self.input_layernorm(x)
        x, present_key_value, shareable_kv = self.self_attn(
            hidden_states=x,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            past_key_value=past_key_value,
            sliding_window=sliding_window,
            shared_kv=shared_kv,
        )
        x = self.post_attention_layernorm(x)
        x = residual + x
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x
        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = x
            x = self.per_layer_input_gate(x)
            x = torch.nn.functional.gelu(x, approximate="tanh")
            x = x * per_layer_input
            x = self.per_layer_projection(x)
            x = self.post_per_layer_input_norm(x)
            x = residual + x
        x = torch.mul(x, self.layer_scalar.to(x), out=output)
        return (x, present_key_value, shareable_kv)


class E2BTransformer(nn.Module):
    """E2B prefill and fixed-cache decode over the source-defined 35 layers."""

    def __init__(self, device=None, dtype=None, ops=nn):
        super().__init__()
        config = self.config = E2BConfig()
        self.embed_tokens = E2BEmbedding(262144, 1536, device=device, dtype=dtype)
        self.embed_tokens_per_layer = E2BEmbedding(
            262144, 8960, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            TransformerBlockGemma4(config, i, device=device, dtype=dtype, ops=ops)
            for i in range(35)
        )
        self.norm = RMSNorm(1536, eps=1e-6, device=device, dtype=dtype)
        self.per_layer_model_projection = ops.Linear(
            1536, 8960, bias=False, device=device, dtype=dtype
        )
        self.per_layer_projection_norm = RMSNorm(
            256, eps=1e-6, device=device, dtype=dtype
        )
        self._global_inv = torch.cat(
            (
                1.0 / (1000000.0 ** (torch.arange(0, 128, 2).float() / 512)),
                torch.zeros(192),
            )
        )
        self._sliding_inv = 1.0 / (10000.0 ** (torch.arange(0, 256, 2).float() / 256))

    def embed(self, ids, out_dtype=None):
        return self.embed_tokens(ids, out_dtype=out_dtype) * (1536**0.5)

    def logits(self, hidden):
        logits = F.linear(hidden[:, -1:], self.embed_tokens.weight)
        return 30.0 * torch.tanh(logits / 30.0)

    @torch.inference_mode()
    def generate(self, embeds, input_ids, *, progress=None):
        """Run the curated 600-token, seed-zero enhancer sampling settings."""
        embeds = embeds.to(torch.bfloat16)
        caches = self.init_cache(
            embeds.shape[0], embeds.shape[1] + 600, embeds.device, embeds.dtype
        )
        generator = torch.Generator(device=embeds.device).manual_seed(0)
        history = input_ids[0].tolist()
        generated = []
        ids = input_ids
        for step in range(600):
            if step:
                embeds = self.embed(ids).to(torch.bfloat16)
            hidden = self(embeds, ids, caches)
            logits = self.logits(hidden)[:, -1]
            seen = torch.tensor(list(set(history)), device=logits.device)
            token_logits = logits[:, seen]
            logits[:, seen] = torch.where(
                token_logits < 0, token_logits * 1.15, token_logits / 1.15
            )
            logits, indices = torch.topk(logits / 0.7, 64)
            probabilities = F.softmax(logits, dim=-1)
            threshold = 0.05 * probabilities.max(dim=-1, keepdim=True).values
            logits[probabilities < threshold] = torch.finfo(logits.dtype).min
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            remove = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) > 0.95
            remove[..., 0] = False
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, sorted_indices, remove)
            logits[mask] = torch.finfo(logits.dtype).min
            selected = torch.multinomial(
                F.softmax(logits, dim=-1), 1, generator=generator
            )
            ids = indices.gather(1, selected)
            token = ids[0].item()
            generated.append(token)
            history.append(token)
            if progress is not None:
                progress(step + 1, 600)
            if token in (1, 50, 106):
                break
        return generated

    def init_cache(self, batch, max_length, device, dtype):
        caches, trackers = [], {}
        for i in range(35):
            if i >= 15:
                caches.append(())
                continue
            sliding = bool(self.layers[i].sliding_attention)
            length, head_dim = (
                (min(512, max_length), 256) if sliding else (max_length, 512)
            )
            cls = RingKV if sliding else FixedKV
            if (cls, length) not in trackers:
                trackers[cls, length] = (
                    torch.empty(1, device=device, dtype=torch.int64),
                    torch.zeros(batch, device=device, dtype=torch.int32),
                )
            position, seqlen = trackers[cls, length]
            key = torch.zeros(batch, 1, length, head_dim, device=device, dtype=dtype)
            caches.append(cls(key, torch.zeros_like(key), 0, position, seqlen))
        return caches

    def forward(self, embeds, input_ids, caches):
        x = embeds
        length = x.shape[1]
        past = caches[0].index
        positions = torch.arange(past, past + length, device=x.device).unsqueeze(0)
        freqs = []
        for inv in (self._global_inv, self._sliding_inv):
            angles = (
                inv[None, :, None].float().expand(x.shape[0], -1, 1).to(x.device)
                @ positions[:, None, :].float()
            ).transpose(1, 2)
            cos, sin = angles.cos(), angles.sin()
            freqs.append(
                torch.stack(
                    (torch.stack((cos, -sin), -1), torch.stack((sin, cos), -1)), -2
                )
                .unsqueeze(1)
                .to(x.dtype)
            )
        mask = None
        if length > 1:
            mask = torch.zeros(
                past + length, past + length, device=x.device, dtype=x.dtype
            )
            mask.masked_fill_(
                torch.ones_like(mask, dtype=torch.bool).triu_(1),
                torch.finfo(x.dtype).min,
            )
        projection = self.per_layer_model_projection(x) * (1.0 / (1536**0.5))
        projection = self.per_layer_projection_norm(
            projection.reshape(*x.shape[:-1], 35, 256)
        )
        per_layer = (
            projection
            + (self.embed_tokens_per_layer(input_ids) * 16).reshape(
                *input_ids.shape, 35, 256
            )
        ) * (0.5**0.5)
        prepared = set()
        for cache in caches:
            if isinstance(cache, FixedKV) and id(cache.position) not in prepared:
                cache.prepare(length)
                prepared.add(id(cache.position))
        shared = {}
        decode = length == 1 and past > 0
        bias = {}
        if decode:
            for cache in caches:
                if isinstance(cache, FixedKV) and cache.key.shape[2] not in bias:
                    capacity = cache.key.shape[2]
                    b = torch.full(
                        (1, 1, 1, capacity),
                        torch.finfo(x.dtype).min,
                        device=x.device,
                        dtype=x.dtype,
                    )
                    b[..., : min(past + 1, capacity)] = 0
                    bias[capacity] = b
            x = x.clone()
        for i, layer in enumerate(self.layers):
            sliding = bool(layer.sliding_attention)
            source = caches[13 if sliding else 14] if decode else shared.get(sliding)
            layer_shared = source if i >= 15 else None
            cache = layer_shared if decode and i >= 15 else caches[i]
            layer_mask = bias[cache.key.shape[2]] if decode else mask
            x, _, share = layer(
                x,
                attention_mask=layer_mask,
                freqs_cis=freqs,
                past_key_value=caches[i],
                per_layer_input=per_layer[:, :, i, :],
                shared_kv=layer_shared,
            )
            if i < 15 and share is not None:
                shared[sliding] = share
        for cache in caches:
            if isinstance(cache, FixedKV):
                cache.advance(length)
        return self.norm(x)
