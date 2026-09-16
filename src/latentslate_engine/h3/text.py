"""H3's raw Qwen3-VL-32B layer-50 text conditioning.

Adapted from ComfyUI 1a14b82e text_encoders/minimax.py, qwen3vl.py and the
prefill path in llama.py (GPL-3.0). No chat template or final norm is applied.
"""

from pathlib import Path

import comfy_kitchen as ck
import torch
from comfy_kitchen.tensor import TensorWiseINT8Layout
from torch import nn
from torch.nn import functional as F
from transformers import Qwen2Tokenizer

from latentslate_engine.torch_attention import attention

from .vision import (
    CONFIG,
    H3VisionModel,
    VisionOperations,
    image_patches,
    image_positions,
    video_patches,
)
from .weights import H3Weights, Linear, RMSNorm

EXTRA_TOKENS = (
    "<d>",
    "</d>",
    "<|cutoff|>",
    "<|lyrics_start|>",
    "<|lyrics_end|>",
    "<|caption_start|>",
    "<|caption_end|>",
)


class TextAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = Linear(5120, 8192, bias=False)
        self.k_proj = Linear(5120, 1024, bias=False)
        self.v_proj = Linear(5120, 1024, bias=False)
        self.o_proj = Linear(8192, 5120, bias=False)
        self.q_norm = RMSNorm(128, eps=1e-6, device="meta")
        self.k_norm = RMSNorm(128, eps=1e-6, device="meta")

    def forward(self, x, mask, rotary):
        b, length, _ = x.shape
        q = self.q_proj(x).view(b, length, 64, 128).transpose(1, 2)
        k = self.k_proj(x).view(b, length, 8, 128).transpose(1, 2)
        v = self.v_proj(x).view(b, length, 8, 128).transpose(1, 2)
        q, k = ck.apply_rope_split_half(self.q_norm(q), self.k_norm(k), rotary)
        x = attention(q, k, v, mask, enable_gqa=True)
        return self.o_proj(x.transpose(1, 2).reshape(b, length, 8192))


class TextMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = Linear(5120, 25600, bias=False)
        self.up_proj = Linear(5120, 25600, bias=False)
        self.down_proj = Linear(25600, 5120, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TextLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(5120, eps=1e-6, device="meta")
        self.self_attn = TextAttention()
        self.post_attention_layernorm = RMSNorm(5120, eps=1e-6, device="meta")
        self.mlp = TextMLP()

    def forward(self, x, mask, rotary):
        x = x + self.self_attn(self.input_layernorm(x), mask, rotary)
        return x + self.mlp(self.post_attention_layernorm(x))


class TextModel(nn.Module):
    def __init__(self, with_vision=False):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TextLayer() for _ in range(50)])
        if with_vision:
            self.visual = H3VisionModel(CONFIG, device="meta", ops=VisionOperations)

    def forward(self, x, positions=None, visual_mask=None, deepstack=None):
        length = x.shape[1]
        if positions is None:
            positions = torch.arange(length, device=x.device).unsqueeze(0)
        inv = 1.0 / (
            5_000_000.0 ** (torch.arange(0, 128, 2, device=x.device).float() / 128)
        )
        freqs = (inv[None, :, None] @ positions[:, None, :].float()).transpose(1, 2)
        if positions.shape[0] == 3:
            interleaved = freqs[0].clone()
            for axis in (1, 2):
                interleaved[..., axis:60:3] = freqs[axis, ..., axis:60:3]
            freqs = interleaved.unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)
        rotary = torch.stack(
            (cos[..., :64], -sin[..., 64:], sin[..., :64], cos[..., 64:]), dim=-1
        )
        rotary = rotary.reshape(*rotary.shape[:-1], 2, 2)
        mask = None
        if length > 1:
            mask = torch.empty(length, length, device=x.device, dtype=x.dtype)
            mask.fill_(torch.finfo(x.dtype).min / 4).triu_(1)
        for index, layer in enumerate(self.model.layers):
            x = layer(x, mask, rotary)
            if deepstack is not None and index < len(deepstack):
                x[visual_mask] = x[visual_mask] + deepstack[index].to(x)
        return x


class H3TextEncoder:
    """Own the text stage's mapped weights and raw prompt tokenizer."""

    def __init__(
        self,
        checkpoint: Path,
        tokenizer: Path,
        device: torch.device,
        *,
        with_vision=False,
    ):
        self.device = device
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            str(tokenizer), local_files_only=True
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(EXTRA_TOKENS)}
        )
        self.model = TextModel(with_vision=with_vision).eval()
        self.weights = H3Weights(checkpoint, self.model, device)
        source = self.weights.checkpoint
        config = source.quantization_config("model.embed_tokens.comfy_quant")
        if config != {"format": "int8_tensorwise"}:
            raise ValueError("Unsupported H3 text embedding representation")
        self.embedding = source.tensor("model.embed_tokens.weight").to(device)
        self.embedding_params = TensorWiseINT8Layout.Params(
            scale=source.tensor("model.embed_tokens.weight_scale").to(device),
            orig_dtype=torch.bfloat16,
            orig_shape=tuple(self.embedding.shape),
        )

    def tokenize(self, prompt):
        return self.tokenizer.encode(prompt, add_special_tokens=False) or [151643]

    def _embed(self, tokens):
        ids = torch.tensor([tokens], device=self.device, dtype=torch.long)
        return TensorWiseINT8Layout.dequantize_embedding(
            self.embedding, self.embedding_params, ids
        ).float()

    @torch.inference_mode()
    def encode(self, prompt, images=(), references=()):
        self.weights.activate()
        if not images and not references:
            result = self.model(self._embed(self.tokenize(prompt)))
            return result.cpu(), torch.ones(result.shape[1], dtype=torch.long)
        if images and references:
            raise ValueError(
                "H3 keyframe images and multimodal references are separate"
            )
        pieces, spans, features = [], [], []
        length = 0

        def add_text(text):
            nonlocal length
            if text:
                value = self._embed(self.tokenize(text))
                pieces.append(value)
                length += value.shape[1]

        def add_vision(data, video=False):
            nonlocal length
            if not hasattr(self.model, "visual"):
                raise ValueError(
                    "H3 image/video conditioning requires the vision encoder"
                )
            pieces.append(self._embed([151652]))
            length += 1
            patches, grid = video_patches(data) if video else image_patches(data)
            merged, deepstack = self.model.visual(
                patches.to(self.device, dtype=torch.float32), grid
            )
            spans.append((length, merged.shape[0], grid))
            features.append(deepstack)
            pieces.extend((merged.unsqueeze(0).float(), self._embed([151653])))
            length += merged.shape[0] + 1

        counters = {"image": 0, "video": 0, "audio": 0}
        for item in references or [
            {"type": "image", "data": image} for image in images
        ]:
            kind = item["type"]
            if kind not in counters:
                raise ValueError(f"Unsupported H3 reference kind: {kind}")
            counters[kind] += 1
            if kind == "image":
                add_text(f"<Picture {counters[kind]}>: ")
                add_vision(item["data"])
            elif kind == "audio":
                add_text(f"<Audio {counters[kind]}>: ")
            else:
                frames = item["data"]
                timestamps = item.get(
                    "timestamps", [i / 2.0 for i in range(frames.shape[0])]
                )
                if frames.shape[0] % 2:
                    frames = torch.cat([frames, frames[-1:]])
                    timestamps = list(timestamps) + [timestamps[-1]]
                add_text(f"<Video {counters[kind]}>: ")
                for index in range(0, frames.shape[0], 2):
                    timestamp = (timestamps[index] + timestamps[index + 1]) / 2.0
                    add_text(f"<{timestamp:.1f} seconds>")
                    add_vision(frames[index : index + 2], video=True)
        add_text(prompt)
        x = torch.cat(pieces, dim=1)
        positions = image_positions(spans, x.shape[1], self.device) if spans else None
        visual_mask = torch.zeros((1, x.shape[1]), device=self.device, dtype=torch.bool)
        tags = torch.ones(x.shape[1], dtype=torch.long)
        for start, size, _ in spans:
            visual_mask[:, start : start + size] = True
            tags[start - 1 : start + size + 1] = 0
        deepstack = (
            [torch.cat([item[index] for item in features]) for index in range(3)]
            if features
            else None
        )
        result = self.model(x, positions, visual_mask, deepstack)
        return result.cpu(), tags

    def close(self):
        self.embedding = self.embedding_params = None
        self.weights.close()
        self.model = None
