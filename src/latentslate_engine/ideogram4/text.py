"""Ideogram v4 Qwen3-VL text conditioning, following ComfyUI 1a14b82e (GPL-3.0).

Adapted from text_encoders/ideogram4.py and qwen3vl.py and the text-only llama.py path.
Conditioning uses FP32 execution, twelve raw taps and the normalized final tap.
"""

import json
from pathlib import Path

import comfy_kitchen as ck
import torch
from comfy_kitchen.tensor import (
    QuantizedTensor,
    TensorCoreFP8Layout,
)
from safetensors import safe_open
from torch.nn import functional as F
from transformers import Qwen2Tokenizer

from latentslate_engine.torch_attention import attention

TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class Ideogram4TextEncoder:
    """Own the official mixed-FP8 Qwen3 weights for one conditioning phase."""

    def __init__(self, checkpoint: Path, tokenizer: Path, device: torch.device):
        self.device = device
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            str(tokenizer), local_files_only=True
        )
        with safe_open(checkpoint, framework="pt") as source:
            keys = source.keys()
            self.quantization = {
                key.removeprefix("model.").removesuffix(".comfy_quant"): json.loads(
                    source.get_tensor(key).numpy().tobytes()
                )["format"]
                for key in keys
                if key.endswith(".comfy_quant")
            }
            unsupported = set(self.quantization.values()) - {"float8_e4m3fn"}
            if unsupported:
                raise ValueError(
                    f"Unsupported Ideogram v4 text quantization: {sorted(unsupported)}"
                )
            self.weights = {
                key.removeprefix("model."): source.get_tensor(key).to(device)
                for key in keys
                if key.startswith("model.") and not key.endswith("comfy_quant")
            }

    def close(self):
        """Release text weights after conditioning or identity invalidation."""
        self.weights.clear()

    def tokenize(self, prompt: str) -> list[int]:
        text = prompt if prompt.startswith("<|im_start|>") else TEMPLATE.format(prompt)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _linear(self, x, name):
        weight = self.weights[name + ".weight"]
        quantization = self.quantization.get(name)
        if quantization == "float8_e4m3fn":
            weight = QuantizedTensor(
                weight,
                "TensorCoreFP8Layout",
                TensorCoreFP8Layout.Params(
                    scale=self.weights[name + ".weight_scale"],
                    orig_dtype=x.dtype,
                    orig_shape=tuple(weight.shape),
                ),
            ).dequantize()
        return F.linear(x, weight.to(x.dtype))

    def _norm(self, x, name):
        return F.rms_norm(
            x, (x.shape[-1],), self.weights[name + ".weight"].to(x.dtype), 1e-6
        )

    @torch.inference_mode()
    def encode_tokens(self, tokens: list[int]):
        ids = torch.tensor([tokens], device=self.device, dtype=torch.long)
        x = F.embedding(ids, self.weights["embed_tokens.weight"]).float()
        length = x.shape[1]
        inv_freq = 1.0 / (
            5_000_000.0 ** (torch.arange(0, 128, 2, device=self.device).float() / 128)
        )
        positions = torch.arange(length, device=self.device).unsqueeze(0)
        freqs = (inv_freq[None, :, None] @ positions[:, None, :].float()).transpose(
            1, 2
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)
        matrix = torch.stack(
            (cos[..., :64], -sin[..., 64:], sin[..., :64], cos[..., 64:]), dim=-1
        )
        matrix = matrix.reshape(*matrix.shape[:-1], 2, 2)
        mask = torch.empty(length, length, device=self.device, dtype=x.dtype)
        mask.fill_(torch.finfo(x.dtype).min / 4).triu_(1)
        taps = []
        for index in range(36):
            name = f"layers.{index}"
            norm = self._norm(x, name + ".input_layernorm")
            attn = name + ".self_attn"
            q = (
                self._linear(norm, attn + ".q_proj")
                .view(1, length, 32, 128)
                .transpose(1, 2)
            )
            k = (
                self._linear(norm, attn + ".k_proj")
                .view(1, length, 8, 128)
                .transpose(1, 2)
            )
            v = (
                self._linear(norm, attn + ".v_proj")
                .view(1, length, 8, 128)
                .transpose(1, 2)
            )
            q, k = ck.apply_rope_split_half(
                self._norm(q, attn + ".q_norm"), self._norm(k, attn + ".k_norm"), matrix
            )
            attended = attention(q, k, v, mask, enable_gqa=True)
            attended = attended.transpose(1, 2).reshape(1, length, 4096)
            x = x + self._linear(attended, attn + ".o_proj")
            norm = self._norm(x, name + ".post_attention_layernorm")
            gate = F.silu(self._linear(norm, name + ".mlp.gate_proj"))
            up = self._linear(norm, name + ".mlp.up_proj")
            x = x + self._linear(gate * up, name + ".mlp.down_proj")
            if index == 35:
                x = self._norm(x, "norm")
            if index in (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35):
                taps.append(x.clone())
        return torch.stack(taps, dim=-1).reshape(1, length, 53248).cpu()

    def encode(self, prompt: str):
        """Return thirteen taps in the reference's interleaved feature order."""
        return self.encode_tokens(self.tokenize(prompt))
