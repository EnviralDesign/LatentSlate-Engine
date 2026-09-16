"""Packed Gemma 4 12B conditioning for LTX 2.5.

Text-only adaptation of ComfyUI 1a14b82e ``text_encoders/gemma4.py``,
``llama.py`` and ``lt.py``; no multimodal or graph runtime is required.
"""

from __future__ import annotations

import math

import comfy_kitchen as ck
import torch
from tokenizers import Tokenizer
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from latentslate_engine.ltx23.checkpoint import Ltx23Checkpoint
from latentslate_engine.ltx23.fp8_linear import Ltx23Int8Linear, _aimdo_modules


class Ltx25TextEncoder:
    """Own one packed text checkpoint and its INT8 projection residency."""

    def __init__(self, checkpoint_path: str, device_index: int = 0) -> None:
        self.device_index = device_index
        self.device = torch.device("cuda", device_index)
        self.checkpoint = Ltx23Checkpoint(checkpoint_path)
        self.tokenizer = Tokenizer.from_str(
            self.checkpoint.tensor("tokenizer_json").numpy().tobytes().decode("utf-8")
        )
        self._linears = {
            name.removesuffix(".weight"): Ltx23Int8Linear(
                self.checkpoint, name.removesuffix(".weight")
            )
            for name in self.checkpoint.tensor_names
            if name.startswith("model.layers.")
            and name.endswith(".weight")
            and self.checkpoint.tensor(name).dtype is torch.int8
        }
        model_vbar, _ = _aimdo_modules(device_index)
        self._vbar = model_vbar.ModelVBAR(
            10 * sum(binding.source_size for binding in self._linears.values()),
            device_index,
        )
        for binding in self._linears.values():
            binding.allocate(self._vbar)
        self._small = {
            name: self.checkpoint.tensor(name).to(self.device)
            for name in self.checkpoint.tensor_names
            if name.startswith("model.layers.")
            and name.endswith(("norm.weight", "layer_scalar"))
        }
        self._small["model.norm.weight"] = self.checkpoint.tensor(
            "model.norm.weight"
        ).to(self.device)
        # These frequencies are computed on CPU in the reference before transfer.
        self._global_inv = torch.cat(
            (
                1.0 / (1000000.0 ** (torch.arange(0, 128, 2).float() / 512)),
                torch.zeros(192),
            )
        )
        self._sliding_inv = 1.0 / (10000.0 ** (torch.arange(0, 256, 2).float() / 256))

    def tokenize(self, prompt: str) -> list[int]:
        """Match the plain, unweighted LTX tokenizer with BOS and left padding."""
        ids = [2] + self.tokenizer.encode(prompt, add_special_tokens=False).ids
        return [0] * max(0, 1024 - len(ids)) + ids

    def _linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        binding = self._linears[name]
        weight, bias, _ = binding.materialize(self.device_index)
        try:
            # Comfy's SDClipModel selects full_precision_mm for text encoders:
            # storage is INT8, but this path does not quantize activations.
            return F.linear(
                x,
                weight.to(dtype=x.dtype).dequantize(),
                None if bias is None else bias.to(x),
            )
        finally:
            binding.unpin(self.device_index)

    def _norm(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self._small[name].to(x), 1e-6)

    def _freqs(self, inv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        expanded = (
            inv[None, :, None].float().expand(positions.shape[0], -1, 1).to(self.device)
        )
        angles = (expanded @ positions[:, None, :].float()).transpose(1, 2)
        cos, sin = angles.cos(), angles.sin()
        return torch.stack(
            (torch.stack((cos, -sin), -1), torch.stack((sin, cos), -1)), -2
        ).unsqueeze(1)

    def _block(
        self, index: int, x: torch.Tensor, mask: torch.Tensor, freqs: tuple
    ) -> torch.Tensor:
        prefix = f"model.layers.{index}"
        local = index % 6 != 5
        heads, head_dim, kv_heads = 16, 256 if local else 512, 8 if local else 1
        batch, length, _ = x.shape
        attention_mask = mask
        if local and length > 1024:
            sliding = torch.zeros(length, length, device=x.device, dtype=x.dtype)
            sliding.masked_fill_(
                torch.ones_like(sliding, dtype=torch.bool).tril_(-1024),
                torch.finfo(x.dtype).min,
            )
            attention_mask = mask + sliding
        normalized = self._norm(f"{prefix}.input_layernorm.weight", x)
        attn = f"{prefix}.self_attn"
        q = (
            self._linear(f"{attn}.q_proj", normalized)
            .view(batch, length, heads, head_dim)
            .transpose(1, 2)
        )
        q = self._norm(f"{attn}.q_norm.weight", q)
        k = self._linear(f"{attn}.k_proj", normalized).view(
            batch, length, kv_heads, head_dim
        )
        v = (
            self._linear(f"{attn}.v_proj", normalized).view(
                batch, length, kv_heads, head_dim
            )
            if local
            else k
        )
        k = self._norm(f"{attn}.k_norm.weight", k).transpose(1, 2)
        v = F.rms_norm(v, (head_dim,), eps=1e-6).transpose(1, 2)
        rotation = freqs[1 if local else 0]
        q = ck.apply_rope_split_half1(q, rotation)
        k = ck.apply_rope_split_half1(k, rotation)
        expand = local and length >= 1024
        if expand:
            k = k.repeat_interleave(heads // kv_heads, dim=1)
            v = v.repeat_interleave(heads // kv_heads, dim=1)
        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ],
            set_priority=True,
        ):
            if not expand:
                params = torch.backends.cuda.SDPAParams(
                    q, k, v, attention_mask, 0.0, False, True
                )
                native_gqa = (
                    torch.backends.cuda.can_use_flash_attention(params)
                    or torch.backends.cuda.can_use_cudnn_attention(params)
                    or torch.backends.cuda.can_use_efficient_attention(params)
                )
                if not native_gqa:
                    k = k.repeat_interleave(heads // kv_heads, dim=1)
                    v = v.repeat_interleave(heads // kv_heads, dim=1)
                    expand = True
            attention = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                scale=1.0,
                enable_gqa=not expand,
                dropout_p=0.0,
            )
        attention = attention.transpose(1, 2).reshape(batch, length, heads * head_dim)
        result = x + self._norm(
            f"{prefix}.post_attention_layernorm.weight",
            self._linear(f"{attn}.o_proj", attention),
        )
        normalized = self._norm(f"{prefix}.pre_feedforward_layernorm.weight", result)
        gate = F.gelu(
            self._linear(f"{prefix}.mlp.gate_proj", normalized), approximate="tanh"
        )
        mlp = self._linear(
            f"{prefix}.mlp.down_proj",
            gate * self._linear(f"{prefix}.mlp.up_proj", normalized),
        )
        result = result + self._norm(f"{prefix}.post_feedforward_layernorm.weight", mlp)
        return torch.mul(result, self._small[f"{prefix}.layer_scalar"].to(x), out=x)

    @torch.inference_mode()
    def encode_tokens(self, ids: list[int]) -> torch.Tensor:
        """Return the dual video/audio projection before AV-model connectors."""
        count = len(ids) - next(
            (i for i, value in enumerate(ids) if value != 0), len(ids)
        )
        if count == 0:
            raise ValueError("text tokens must contain at least BOS")
        tokens = torch.tensor([ids], dtype=torch.long)
        # Index mapped BF16 rows first; scaling follows the float32 lookup.
        embedding = self.checkpoint.tensor("model.embed_tokens.weight")
        x = F.embedding(tokens, embedding).float().to(self.device) * math.sqrt(3840)
        positions = torch.arange(len(ids), device=self.device).unsqueeze(0)
        freqs = (
            self._freqs(self._global_inv, positions),
            self._freqs(self._sliding_inv, positions),
        )
        valid = torch.zeros(1, len(ids), device=self.device, dtype=torch.long)
        valid[:, -count:] = 1
        mask = (1.0 - valid.to(x.dtype).reshape(1, 1, 1, -1)).expand(
            1, 1, len(ids), len(ids)
        )
        mask = mask.masked_fill(mask.bool(), torch.finfo(x.dtype).min)
        causal = torch.zeros(len(ids), len(ids), dtype=x.dtype, device=self.device)
        causal.masked_fill_(
            torch.ones_like(causal, dtype=torch.bool).triu_(1), torch.finfo(x.dtype).min
        )
        mask = mask + causal
        states = []
        for index in range(48):
            states.append(x[:, -count:].clone())
            x = self._block(index, x, mask, freqs)
        states.append(self._norm("model.norm.weight", x)[:, -count:].clone())
        features = torch.stack(states, dim=1).to(torch.bfloat16).movedim(1, -1)
        del states, x
        features = (
            features * torch.rsqrt(torch.mean(features**2, dim=2, keepdim=True) + 1e-6)
        ).flatten(start_dim=2)
        outputs = []
        for role, width in (("video", 4096), ("audio", 2048)):
            prefix = f"text_embedding_projection.{role}_aggregate_embed"
            weight = self.checkpoint.tensor(f"{prefix}.weight").to(self.device)
            bias = self.checkpoint.tensor(f"{prefix}.bias").to(self.device)
            outputs.append(F.linear(features * math.sqrt(width / 3840), weight, bias))
            del weight, bias
        return torch.cat(outputs, dim=-1).float().cpu()

    def encode(self, prompt: str) -> torch.Tensor:
        return self.encode_tokens(self.tokenize(prompt))

    def close(self) -> None:
        self._linears.clear()
        self._small.clear()
        self._vbar = None
        self.checkpoint = None
        torch.cuda.empty_cache()
