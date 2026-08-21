"""Pinned Z-Image prompt/token boundary feeding the Engine-owned Qwen shell."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..z_image_turbo_recipe import (
    ZImagePipelineSupportPlan,
    revalidate_z_image_pipeline_support,
)

# This is the literal template in Comfy's pinned ``z_image.py`` support code.
# Do not route through ``tokenizer.chat_template``: it is mutable model data and
# several otherwise compatible Qwen releases inject a system message or spacing.
Z_IMAGE_PROMPT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
Z_IMAGE_PAD_TOKEN_ID = 151643
Z_IMAGE_BOS_TOKEN_ID = 151643
Z_IMAGE_EOS_TOKEN_ID = 151645
Z_IMAGE_HIDDEN_STATE_INDEX = -2


def build_z_image_qwen_tokenizer(support: ZImagePipelineSupportPlan):
    """Build the exact slow Qwen BPE tokenizer from the pinned support closure."""

    from transformers import Qwen2Tokenizer

    if not revalidate_z_image_pipeline_support(support):
        raise ValueError("Z-Image pipeline support changed before tokenizer construction")
    tokenizer = Qwen2Tokenizer.from_pretrained(
        support.root / "tokenizer", local_files_only=True
    )
    if (
        tokenizer.pad_token_id != 151643
        or tokenizer.eos_token_id != 151645
        or tokenizer.added_tokens_encoder.get("<|im_start|>") != 151644
        or tokenizer.added_tokens_encoder.get("<|im_end|>") != 151645
    ):
        raise ValueError("Z-Image tokenizer special-token facts differ from the exact pin")
    return tokenizer


@dataclass(frozen=True, slots=True)
class ZImagePromptConditioning:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_template_sha256: str
    hidden_state_index: int = Z_IMAGE_HIDDEN_STATE_INDEX


@dataclass(frozen=True, slots=True)
class ZImageEncodedConditioning:
    positive: torch.Tensor
    attention_mask: torch.Tensor
    token_count: int


def z_image_prompt_envelope(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError("Z-Image prompt must be text")
    return Z_IMAGE_PROMPT_TEMPLATE.format(prompt=prompt)


def tokenize_z_image_prompt(tokenizer: object, prompt: str) -> ZImagePromptConditioning:
    """Tokenize Comfy's literal envelope without weights, BOS/EOS, or fixed padding."""

    encoded = tokenizer(
        z_image_prompt_envelope(prompt),
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_tensors="pt",
    )
    ids = encoded["input_ids"]
    if ids.dtype is not torch.long:
        ids = ids.to(dtype=torch.long)
    if ids.ndim != 2:
        raise ValueError("Z-Image tokenizer did not return a canonical token batch")
    # This is SDClipModel.process_tokens lines 184-207: pad 151643 is the
    # comparison token because Z does not configure a separate end token.
    masks: list[list[int]] = []
    for row in ids.tolist():
        row_mask: list[int] = []
        eos = False
        left_pad = False
        for index, token in enumerate(row):
            if index == 0 and token == Z_IMAGE_PAD_TOKEN_ID:
                left_pad = True
            if eos or (left_pad and token == Z_IMAGE_PAD_TOKEN_ID):
                row_mask.append(0)
            else:
                row_mask.append(1)
                left_pad = False
            if not eos and token == Z_IMAGE_PAD_TOKEN_ID and not left_pad:
                row_mask[-1] = 0
                eos = True
        masks.append(row_mask)
    mask = torch.tensor(masks, dtype=torch.long)
    return ZImagePromptConditioning(
        ids,
        mask,
        hashlib.sha256(Z_IMAGE_PROMPT_TEMPLATE.encode()).hexdigest(),
    )


@torch.inference_mode()
def encode_z_image_prompt(
    model: torch.nn.Module,
    tokenizer: object,
    prompt: str,
    *,
    device: torch.device | str,
    cancelled: Callable[[], bool] = lambda: False,
    diagnostic: Callable[[str], None] = lambda _stage: None,
) -> ZImageEncodedConditioning:
    """Run the raw Qwen path and return its post-block-34 FP32 conditioning."""

    diagnostic("conditioning.edge_10")
    if cancelled():
        raise RuntimeError("Z-Image Qwen conditioning canceled")
    tokenized = tokenize_z_image_prompt(tokenizer, prompt)
    diagnostic("conditioning.edge_11")
    if cancelled():
        raise RuntimeError("Z-Image Qwen conditioning canceled")
    input_ids = tokenized.input_ids.to(device=device, dtype=torch.long)
    mask = tokenized.attention_mask.to(device=device, dtype=torch.long)
    if mask.ndim != 2 or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("Z-Image tokenizer attention mask is not binary [B,L] int64")
    forward = getattr(model, "forward_conditioning", None)
    if not callable(forward):
        raise TypeError("Z-Image Qwen is not the exact Engine-owned conditioning shell")
    hidden = forward(
        input_ids,
        mask,
        cancelled=cancelled,
        diagnostic=diagnostic,
    )
    if hidden.ndim != 3 or hidden.shape[:2] != mask.shape or hidden.shape[-1] != 2560:
        raise ValueError("Z-Image Qwen penultimate hidden state has an invalid shape")
    if not torch.isfinite(hidden).all():
        raise ValueError("Z-Image Qwen returned non-finite conditioning")
    # Runtime jobs are one prompt today; keeping the tensor batched makes the
    # core ready for future batch wiring without inventing padding semantics.
    return ZImageEncodedConditioning(
        positive=hidden,
        attention_mask=mask,
        token_count=int(mask.sum().item()),
    )
