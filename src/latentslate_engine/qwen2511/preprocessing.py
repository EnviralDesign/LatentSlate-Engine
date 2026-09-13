"""Ordered image preparation from the pinned Comfy Qwen 2511 edit path.

Adapted from ComfyUI 12d5279438bfefc058a269eae805ceab6047777f,
comfy_extras/nodes_flux.py, nodes_qwen.py and comfy/utils.py (GPL-3.0).
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F


CANVAS_RESOLUTIONS = (
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672),
)


def load_image(path: Path) -> torch.Tensor:
    """Decode one still image as unpremultiplied RGB in a CPU BHWC tensor."""
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("Qwen image edit requires a still image")
        image = ImageOps.exif_transpose(source).convert("RGB")
        pixels = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(pixels)[None]


def canvas_image(image: torch.Tensor) -> torch.Tensor:
    """Center crop and Lanczos-resize slot 1 to its nearest curated canvas."""
    old_height, old_width = image.shape[1:3]
    old_aspect = old_width / old_height
    _, width, height = min(
        (abs(old_aspect - w / h), w, h) for w, h in CANVAS_RESOLUTIONS
    )
    x = y = 0
    new_aspect = width / height
    if old_aspect > new_aspect:
        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
    cropped = image[:, y:old_height - y, x:old_width - x]
    pixels = np.clip(255.0 * cropped[0].numpy(), 0, 255).astype(np.uint8)
    resized = Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)
    # Preserve the oracle's CHW stack followed by a BHWC view.
    tensor = torch.from_numpy(np.array(resized).astype(np.float32) / 255.0)
    return torch.stack([tensor.movedim(-1, 0)]).movedim(1, -1)


def reference_images(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce the distinct visual-language and reference-VAE representations."""
    height, width = image.shape[1:3]
    samples = image.movedim(-1, 1)
    vl_scale = math.sqrt(384 * 384 / (width * height))
    vl = F.interpolate(
        samples, size=(round(height * vl_scale), round(width * vl_scale)), mode="area"
    ).movedim(1, -1)
    vae_scale = math.sqrt(1024 * 1024 / (width * height))
    vae = F.interpolate(
        samples,
        size=(round(height * vae_scale / 8) * 8, round(width * vae_scale / 8) * 8),
        mode="area",
    ).movedim(1, -1)
    return vl, vae


def picture_prompt(prompt: str, slots: tuple[int, ...]) -> str:
    """Keep logical picture numbers even when an optional middle slot is absent."""
    return "".join(
        f"Picture {slot}: <|vision_start|><|image_pad|><|vision_end|>" for slot in slots
    ) + prompt
