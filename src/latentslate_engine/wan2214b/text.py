from __future__ import annotations

import math

import sentencepiece
import torch
import torch.nn.functional as F

from .attention import scaled_dot_product_attention
from .weights import WanWeights

MODEL_DIM = 4096
FF_DIM = 10240
NUM_HEADS = 64
NUM_LAYERS = 24
MIN_LENGTH = 512


def tokenize(weights: WanWeights, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    model = weights.base.tensor("spiece_model").numpy().tobytes()
    processor = sentencepiece.SentencePieceProcessor(
        model_proto=model, add_bos=False, add_eos=True
    )
    ids = list(processor.encode(text))
    if ids[-1] != 1:
        ids.append(1)
    mask = [1] * len(ids)
    if len(ids) < MIN_LENGTH:
        padding = MIN_LENGTH - len(ids)
        ids.extend([0] * padding)
        mask.extend([0] * padding)
    return torch.tensor([ids], dtype=torch.long), torch.tensor([mask], dtype=torch.long)


def _layer_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def _relative_position_bucket(relative_position: torch.Tensor) -> torch.Tensor:
    num_buckets = 16
    relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
    relative_position = torch.abs(relative_position)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(128 / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    large = torch.min(large, torch.full_like(large, num_buckets - 1))
    return relative_buckets + torch.where(is_small, relative_position, large)


class Umt5Encoder:
    def __init__(self, weights: WanWeights):
        self.weights = weights

    def _attention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        prefix = f"encoder.block.{layer}.layer.0.SelfAttention"
        q = self.weights.linear(x, f"{prefix}.q")
        k = self.weights.linear(x, f"{prefix}.k")
        v = self.weights.linear(x, f"{prefix}.v")
        q = q.view(q.shape[0], q.shape[1], NUM_HEADS, -1).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], NUM_HEADS, -1).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], NUM_HEADS, -1).transpose(1, 2)

        length = x.shape[1]
        positions = torch.arange(length, dtype=torch.long, device=x.device)
        buckets = _relative_position_bucket(positions[None, :] - positions[:, None])
        bias_weight = self.weights.affine(
            f"{prefix}.relative_attention_bias.weight", x.device, x.dtype
        )
        bias = F.embedding(buckets, bias_weight).permute(2, 0, 1).unsqueeze(0)
        padding = (1.0 - mask.to(x.dtype)).reshape(mask.shape[0], 1, 1, length)
        attention_mask = (
            padding.masked_fill(padding.to(torch.bool), -torch.finfo(x.dtype).max)
            + bias
        )
        out = scaled_dot_product_attention(
            q,
            k * math.sqrt(k.shape[-1]),
            v,
            attn_mask=attention_mask,
        )
        out = out.transpose(1, 2).reshape(x.shape)
        return self.weights.linear(out, f"{prefix}.o")

    @torch.inference_mode()
    def encode(
        self, text: str, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids, mask = tokenize(self.weights, text)
        shared = self.weights.base.tensor("shared.weight")
        x = shared[ids].to(device=device, dtype=torch.float32, non_blocking=True)
        mask_device = mask.to(device)

        for layer in range(NUM_LAYERS):
            prefix = f"encoder.block.{layer}"
            norm = self.weights.affine(
                f"{prefix}.layer.0.layer_norm.weight", device, x.dtype
            )
            x = x + self._attention(_layer_norm(x, norm), mask_device, layer)
            norm = self.weights.affine(
                f"{prefix}.layer.1.layer_norm.weight", device, x.dtype
            )
            y = _layer_norm(x, norm)
            y0 = F.gelu(
                self.weights.linear(y, f"{prefix}.layer.1.DenseReluDense.wi_0"),
                approximate="tanh",
            )
            y1 = self.weights.linear(y, f"{prefix}.layer.1.DenseReluDense.wi_1")
            y = self.weights.linear(y0 * y1, f"{prefix}.layer.1.DenseReluDense.wo")
            x = x + y

        final_norm = self.weights.affine(
            "encoder.final_layer_norm.weight", device, x.dtype
        )
        x = _layer_norm(x, final_norm)
        x = x.float() * mask_device.unsqueeze(-1).float()
        return ids, mask, x.cpu()
