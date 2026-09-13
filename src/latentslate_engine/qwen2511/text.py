"""The curated Qwen 2.5 VL encoder's FP32 compute over scaled FP8 weights.

Narrowly adapted from ComfyUI 12d5279438bfefc058a269eae805ceab6047777f
text_encoders/llama.py, qwen_image.py, sd1_clip.py and ops.py (GPL-3.0).
"""

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout
from transformers import Qwen2Tokenizer

from latentslate_engine.mapped_checkpoint import MappedCheckpoint
from latentslate_engine.torch_attention import attention
from .preprocessing import picture_prompt
from .vision import Qwen2VLVisionTransformer, process_qwen2vl_images, qwen2vl_mrope_position_ids

EDIT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, "
    "size, texture, objects, background), then explain how the user's text instruction "
    "should alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


class TextLinear(nn.Linear):
    """Keep scaled FP8 storage and use the reference's full-precision matmul."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("scale_weight", None)

    def forward(self, x):
        weight = self.weight
        if weight.dtype == torch.float8_e4m3fn:
            weight = QuantizedTensor(
                weight, "TensorCoreFP8Layout",
                TensorCoreFP8Layout.Params(
                    scale=self.scale_weight, orig_dtype=x.dtype,
                    orig_shape=tuple(weight.shape),
                ),
            ).dequantize()
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, weight.to(x.dtype), bias)


def load_vision(checkpoint: Path, device: torch.device):
    """Load the concrete visual submodel without the unused language head."""
    ops = SimpleNamespace(Linear=TextLinear, RMSNorm=nn.RMSNorm, Conv3d=nn.Conv3d)
    model = Qwen2VLVisionTransformer(
        hidden_size=1280, output_hidden_size=3584,
        device="meta", dtype=torch.float32, ops=ops,
    )
    source = MappedCheckpoint(checkpoint)
    for name, module in model.named_modules():
        for param_name, parameter in list(module.named_parameters(recurse=False)):
            key = f"visual.{name}.{param_name}"
            value = source.tensor(key).to(device)
            if value.dtype != torch.float8_e4m3fn:
                value = value.float()
            setattr(module, param_name, nn.Parameter(value, requires_grad=False))
        if isinstance(module, TextLinear):
            module.scale_weight = source.tensor(f"visual.{name}.scale_weight").to(device)
    return model.eval()


@torch.inference_mode()
def encode_image(model, pixels, device):
    """Return visual embeddings and the corresponding image grid."""
    patches, grid = process_qwen2vl_images(pixels)
    return model(patches.to(device, dtype=torch.float32), grid), grid


class QwenTextEncoder:
    """Own the visual and language weights for image-edit conditioning."""

    def __init__(self, checkpoint: Path, tokenizer: Path, device: torch.device):
        self.device = device
        self.tokenizer = Qwen2Tokenizer.from_pretrained(str(tokenizer), local_files_only=True)
        self.visual = load_vision(checkpoint, device)
        source = MappedCheckpoint(checkpoint)
        self.weights = {
            name.removeprefix("model."): source.tensor(name).to(device)
            for name in source.tensor_names if name.startswith("model.")
        }

    def _linear(self, x, name):
        weight = self.weights[name + ".weight"]
        if weight.dtype == torch.float8_e4m3fn:
            weight = QuantizedTensor(
                weight, "TensorCoreFP8Layout",
                TensorCoreFP8Layout.Params(
                    scale=self.weights[name + ".scale_weight"],
                    orig_dtype=x.dtype, orig_shape=tuple(weight.shape),
                ),
            ).dequantize()
        bias = self.weights.get(name + ".bias")
        return F.linear(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))

    def _norm(self, x, name):
        return F.rms_norm(x, (x.shape[-1],), self.weights[name + ".weight"].to(x.dtype), 1e-6)

    @staticmethod
    def _rope(x, cos, sin):
        out = x * cos
        out[..., :64].addcmul_(x[..., 64:], -sin[..., 64:])
        out[..., 64:].addcmul_(x[..., :64], sin[..., :64])
        return out.to(x.dtype)

    @torch.inference_mode()
    def encode(self, prompt: str, images: tuple, slots: tuple[int, ...]):
        """Encode ordered image references plus the edit text, stripping the system prefix."""
        tokens = self.tokenizer.encode(
            EDIT_TEMPLATE.format(picture_prompt(prompt, slots)), add_special_tokens=False,
        )
        ids = torch.tensor([[token for token in tokens if token != 151655]], device=self.device)
        x = F.embedding(ids, self.weights["embed_tokens.weight"]).float()
        info = []
        offset = 0
        images_iter = iter(images)
        for index, token in enumerate(tokens):
            if token != 151655:
                continue
            embedding, grid = encode_image(self.visual, next(images_iter), self.device)
            start = index + offset
            x = torch.cat([x[:, :start], embedding[None], x[:, start:]], dim=1)
            info.append(dict(type="image", index=start, size=embedding.shape[0], extra=grid))
            offset += embedding.shape[0] - 1
        length = x.shape[1]
        positions = qwen2vl_mrope_position_ids(info, length, self.device)
        inv_freq = 1.0 / (1_000_000.0 ** (torch.arange(0, 128, 2, device=self.device).float() / 128))
        freqs = (inv_freq[None, :, None].expand(3, -1, 1) @ positions[:, None, :].float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = [
            torch.cat([chunk[i % 3] for i, chunk in enumerate(t.split([16, 24, 24] * 2, dim=-1))], dim=-1)[None]
            for t in (emb.cos(), emb.sin())
        ]
        mask = torch.empty(length, length, device=self.device, dtype=x.dtype)
        mask.fill_(torch.finfo(x.dtype).min / 4).triu_(1)
        mask = mask[None, None]
        for index in range(28):
            name = f"layers.{index}"
            norm = self._norm(x, name + ".input_layernorm")
            attn = name + ".self_attn"
            q = self._linear(norm, attn + ".q_proj").view(1, length, 28, 128).transpose(1, 2)
            k = self._linear(norm, attn + ".k_proj").view(1, length, 4, 128).transpose(1, 2)
            v = self._linear(norm, attn + ".v_proj").view(1, length, 4, 128).transpose(1, 2)
            q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
            attended = attention(q, k, v, mask, enable_gqa=True).transpose(1, 2).reshape(1, length, 3584)
            x = x + self._linear(attended, attn + ".o_proj")
            norm = self._norm(x, name + ".post_attention_layernorm")
            gate = F.silu(self._linear(norm, name + ".mlp.gate_proj"))
            up = self._linear(norm, name + ".mlp.up_proj")
            x = x + self._linear(gate * up, name + ".mlp.down_proj")
        start = tokens.index(151644, tokens.index(151644) + 1) + 3
        return self._norm(x, "norm")[:, start:].cpu()

    def close(self):
        """Release the loaded device weights."""
        self.weights.clear()
        self.visual = None
