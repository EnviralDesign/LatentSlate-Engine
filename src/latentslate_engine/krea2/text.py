"""Krea's text-only Qwen3-VL path, following the pinned Comfy oracle.

The 12 conditioning taps execute in FP32; autoregressive enhancement uses BF16.
Vision conditioning is deliberately outside this family mode.
"""

from __future__ import annotations

from pathlib import Path

import torch
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout
from safetensors import safe_open
from torch import Tensor
from torch.nn import functional as F
from .attention import attention as scaled_attention
from transformers import Qwen2Tokenizer

TAP_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
CONDITIONING_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


# Exact curated enhancer instruction, including its published punctuation.
ENHANCEMENT_INSTRUCTIONS = 'You are an expert prompt engineer for text-to-image models. Your task is to expand the user\'s prompt into a highly effective image-generation prompt.\n\nThink step by step about the request before writing the answer:\n- What is the subject and mood?\n- What visual styles, mediums, and lighting options would fit? Consider two or three alternatives and pick the one that best serves the caption.\n- What composition, framing, and grounded details will help the text-to-image model?\n\nThen output a single expanded prompt paragraph.\n\nFollow these rules strictly:\n1. **Faithfulness First:** Preserve all original subjects, actions, colors, and spatial relationships. Do not add new objects, props, characters, or animals unless the user clearly implies them.\n2. **Practical T2I Structure:** Write a prompt that a text-to-image model can parse cleanly. Group subjects with their own attributes and actions. Use grounded phrasing for poses, interactions, and spatial layout.\n3. **Style Planning Stays Internal:** Use your internal reasoning to choose style, medium, framing, and lighting. Do not emit planning tags or wrappers in the visible answer body.\n4. **Text Rendering:** If the user requests visible text, quotes, labels, or typography, specify the exact text clearly and wrap requested words in quotes.\n5. **Avoid Over-Specification:** Do not invent highly specific clothing, colors, materials, or scene details unless the input supports them.\n6. **Structure:** Write one cohesive paragraph after the thinking block. No bullets, JSON, or markdown.\n7. **Respect Existing Detail:** If the user\'s prompt is already detailed, lightly polish and finalize rather than heavily expanding \xe2\u20ac\u201d preserve their phrasing and direction.\n8. **Respect the Human Form:** Treat depictions of people with dignity. Assume clothing covers genitals and intimate anatomy.\n9. **Preserve User Medium:** When the user explicitly requests a medium (e.g. "photo of", "photograph of", "illustration of", "painting of", "sketch of", "3D render of"), honor it. Do not pivot to a different medium to avoid difficulty \xe2\u20ac\u201d match the user\'s stated intent.\n\nUser\'s Input:\n\n'


class KreaTextEncoder:
    """Own the concrete text weights and tokenizer for Krea Turbo."""

    def __init__(self, checkpoint: Path, tokenizer: Path, device: torch.device):
        self.device = device
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            str(tokenizer), local_files_only=True
        )
        with safe_open(checkpoint, framework="pt") as source:
            self.weights = {
                key.removeprefix("model."): source.get_tensor(key).to(device)
                for key in source.keys()
                if key.startswith("model.") and not key.endswith("comfy_quant")
            }

    def tokenize(self, prompt: str) -> list[int]:
        """Apply the oracle's conditioning template without an empty think block."""
        return self.tokenizer.encode(
            CONDITIONING_TEMPLATE.format(prompt), add_special_tokens=False
        )

    def _linear(self, x: Tensor, name: str) -> Tensor:
        weight = self.weights[name + ".weight"]
        if weight.dtype != torch.float8_e4m3fn:
            return F.linear(x, weight.to(x.dtype))
        weight = QuantizedTensor(
            weight,
            "TensorCoreFP8Layout",
            TensorCoreFP8Layout.Params(
                scale=self.weights[name + ".weight_scale"],
                orig_dtype=x.dtype,
                orig_shape=tuple(weight.shape),
            ),
        )
        return F.linear(x, weight.dequantize().to(x.dtype))

    def _norm(self, x: Tensor, name: str) -> Tensor:
        return F.rms_norm(
            x, (x.shape[-1],), self.weights[name + ".weight"].to(x.dtype), 1e-6
        )

    @staticmethod
    def _rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        out = x * cos
        half = x.shape[-1] // 2
        out[..., :half].addcmul_(x[..., half:], -sin[..., half:])
        out[..., half:].addcmul_(x[..., :half], sin[..., :half])
        return out.to(x.dtype)

    @torch.inference_mode()
    def _forward(self, tokens, dtype, cache=None, offset=0, collect=False):
        ids = torch.tensor([tokens], device=self.device, dtype=torch.long)
        x = F.embedding(ids, self.weights["embed_tokens.weight"]).to(dtype)
        length = x.shape[1]
        inv_freq = 1.0 / (
            5_000_000.0 ** (torch.arange(0, 128, 2, device=self.device).float() / 128)
        )
        positions = torch.arange(offset, offset + length, device=self.device).unsqueeze(
            0
        )
        freqs = (inv_freq[None, :, None] @ positions[:, None, :].float()).transpose(
            1, 2
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)
        mask = None
        if length > 1:
            mask = torch.empty(length, length, device=self.device, dtype=x.dtype)
            mask.fill_(torch.finfo(x.dtype).min / 4).triu_(1)
            mask = mask[None, None]
        taps = []
        for i in range(36):
            if collect and i in TAP_LAYERS:
                taps.append(x.clone())
            name = f"layers.{i}"
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
            q = self._rope(self._norm(q, attn + ".q_norm"), cos, sin)
            k = self._rope(self._norm(k, attn + ".k_norm"), cos, sin)
            if cache is not None:
                past_k, past_v = cache[i]
                past_k[:, :, offset : offset + length] = k
                past_v[:, :, offset : offset + length] = v
                k = past_k[:, :, : offset + length]
                v = past_v[:, :, : offset + length]
            attention = scaled_attention(q, k, v, mask, enable_gqa=True)
            attention = attention.transpose(1, 2).reshape(1, length, 4096)
            x = x + self._linear(attention, attn + ".o_proj")
            norm = self._norm(x, name + ".post_attention_layernorm")
            gate = F.silu(self._linear(norm, name + ".mlp.gate_proj"))
            up = self._linear(norm, name + ".mlp.up_proj")
            x = x + self._linear(gate * up, name + ".mlp.down_proj")
        if not collect:
            return self._norm(x, "norm")
        start = tokens.index(151644, tokens.index(151644) + 1) + 3
        stacked = torch.stack(taps, dim=1)[:, :, start:]
        return stacked.permute(0, 2, 1, 3).reshape(1, length - start, 30720).cpu()

    def encode_tokens(self, tokens: list[int]) -> Tensor:
        """Return the flattened, prefix-stripped twelve-layer conditioning."""
        return self._forward(tokens, torch.float32, collect=True)

    def encode(self, prompt: str) -> Tensor:
        """Encode a plain text prompt into Krea's conditioning tensor."""
        return self.encode_tokens(self.tokenize(prompt))

    @torch.inference_mode()
    def enhance(self, prompt: str) -> str:
        """Run the curated no-thinking enhancer with its independent seed zero."""
        text = CONDITIONING_TEMPLATE.format(ENHANCEMENT_INSTRUCTIONS + prompt)
        tokens = self.tokenizer.encode(
            text + "<think>\n\n</think>\n\n", add_special_tokens=False
        )
        capacity = len(tokens) + 512
        cache = [
            tuple(
                torch.empty(
                    1, 8, capacity, 128, device=self.device, dtype=torch.bfloat16
                )
                for _ in range(2)
            )
            for _ in range(36)
        ]
        generator = torch.Generator(device=self.device).manual_seed(0)
        history = []
        offset = 0
        for _ in range(512):
            hidden = self._forward(tokens, torch.bfloat16, cache=cache, offset=offset)
            logits = F.linear(
                hidden[:, -1:], self.weights["embed_tokens.weight"].to(hidden.dtype)
            )[:, -1]
            if history:
                ids = torch.tensor(list(set(history)), device=self.device)
                previous = logits[:, ids]
                logits[:, ids] = torch.where(
                    previous < 0, previous * 1.05, previous / 1.05
                )
            logits, indices = torch.topk(logits / 0.7, 64)
            probabilities = F.softmax(logits, dim=-1)
            cutoff = 0.05 * probabilities.max(dim=-1, keepdim=True).values
            logits[probabilities < cutoff] = torch.finfo(logits.dtype).min
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            remove = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1) > 0.95
            remove[..., 0] = False
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                1, sorted_indices, remove
            )
            logits[mask] = torch.finfo(logits.dtype).min
            sampled = torch.multinomial(
                F.softmax(logits, dim=-1), 1, generator=generator
            )
            token = indices.gather(1, sampled).item()
            history.append(token)
            if token in (151643, 151645):
                break
            offset += len(tokens)
            tokens = [token]
        return self.tokenizer.decode(history, skip_special_tokens=True).strip()

    def close(self) -> None:
        """Release this encoder's device weights."""
        self.weights.clear()
