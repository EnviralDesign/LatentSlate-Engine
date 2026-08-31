from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Self

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.nn import functional as F

from latentslate_engine.identity import FileContentIdentity as SourceImageIdentity

from .runtime import (
    KLEIN_ALIGNMENT,
    Klein9BIdentity,
    Klein9BRuntime,
    _encode_prompt,
    _load_transformer,
    _load_vae,
    _sigmas_for_dimensions,
    _unpack_latent,
    validate_klein_dimensions,
    validate_klein_seed,
)


@dataclass
class ReferenceCacheEntry:
    identity: SourceImageIdentity
    latent: Tensor
    scaled_width: int
    scaled_height: int


@dataclass(frozen=True)
class TwoImageGenerationResult:
    output: Path
    elapsed_seconds: float
    conditioning_reused: bool
    reference_reused: tuple[bool, bool]
    models_reused: bool


def _load_rgb(path: Path) -> Tensor:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels.copy()).unsqueeze(0)


def _scale_to_one_megapixel(image: Tensor, method: str) -> Tensor:
    height, width = image.shape[1:3]
    scaled_width, scaled_height = _one_megapixel_dimensions(width, height)
    samples = image.movedim(-1, 1)
    if method == "lanczos":
        resized = []
        for sample in samples:
            array = np.clip(255.0 * sample.movedim(0, -1).numpy(), 0, 255).astype(
                np.uint8
            )
            pil = Image.fromarray(array).resize(
                (scaled_width, scaled_height), resample=Image.Resampling.LANCZOS
            )
            resized.append(
                torch.from_numpy(
                    np.asarray(pil, dtype=np.float32).copy() / 255.0
                ).movedim(-1, 0)
            )
        samples = torch.stack(resized)
    else:
        samples = F.interpolate(
            samples, size=(scaled_height, scaled_width), mode="nearest-exact"
        )
    return samples.movedim(1, -1)


def _one_megapixel_dimensions(width: int, height: int) -> tuple[int, int]:
    scale = math.sqrt((1024 * 1024) / (width * height))
    scaled_width = round(width * scale)
    scaled_height = round(height * scale)
    if scaled_width < KLEIN_ALIGNMENT or scaled_height < KLEIN_ALIGNMENT:
        raise ValueError(
            "source image aspect ratio leaves a reference side below 16 pixels"
        )
    return scaled_width, scaled_height


def _source_scaled_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as source:
        width, height = ImageOps.exif_transpose(source).size
    return _one_megapixel_dimensions(width, height)


def _encode_reference(
    vae: torch.nn.Module, pixels: Tensor, device: torch.device
) -> Tensor:
    height, width = pixels.shape[1:3]
    cropped_height = (height // 16) * 16
    cropped_width = (width // 16) * 16
    y = (height % 16) // 2
    x = (width % 16) // 2
    pixels = pixels[:, y : y + cropped_height, x : x + cropped_width]
    pixels = (pixels.movedim(-1, 1).to(device=device) * 2.0 - 1.0).to(torch.bfloat16)
    with torch.inference_mode():
        latent = vae.encode(pixels).latent_dist.mode()
        batch, channels, latent_height, latent_width = latent.shape
        latent = (
            latent.reshape(batch, channels, latent_height // 2, 2, latent_width // 2, 2)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(batch, channels * 4, latent_height // 2, latent_width // 2)
        )
        running_mean = vae.bn.running_mean.to(device=device, dtype=latent.dtype)
        running_var = vae.bn.running_var.to(device=device, dtype=latent.dtype)
        latent = F.batch_norm(
            latent,
            running_mean,
            running_var,
            training=False,
            momentum=0.1,
            eps=1e-4,
        )
    return latent


def _target_geometry(
    first_scaled_width: int,
    first_scaled_height: int,
    width: int | None,
    height: int | None,
) -> tuple[int, int, int, int]:
    if (width is None) != (height is None):
        raise ValueError("width and height must either both be provided or both omitted")
    if width is not None and height is not None:
        validate_klein_dimensions(width, height)
        return width, height, width, height

    target_width = (first_scaled_width // KLEIN_ALIGNMENT) * KLEIN_ALIGNMENT
    target_height = (first_scaled_height // KLEIN_ALIGNMENT) * KLEIN_ALIGNMENT
    validate_klein_dimensions(target_width, target_height)
    return target_width, target_height, first_scaled_width, first_scaled_height


class Klein9BTwoImageRuntime(Klein9BRuntime):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device)
        self.references: list[ReferenceCacheEntry | None] = [None, None]

    def close(self) -> None:
        self.references = [None, None]
        super().close()

    def _reference(
        self, slot: int, path: Path, method: str
    ) -> tuple[ReferenceCacheEntry, bool]:
        identity = SourceImageIdentity.from_path(path)
        cached = self.references[slot]
        if cached is not None and cached.identity == identity:
            return cached, True
        assert self.vae is not None
        scaled = _scale_to_one_megapixel(_load_rgb(identity.path), method)
        entry = ReferenceCacheEntry(
            identity=identity,
            latent=_encode_reference(self.vae, scaled, self.device),
            scaled_width=scaled.shape[2],
            scaled_height=scaled.shape[1],
        )
        self.references[slot] = entry
        return entry, False

    def generate_two_image(
        self,
        identity: Klein9BIdentity,
        prompt: str,
        first_image: Path,
        second_image: Path,
        seed: int,
        output: Path,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> TwoImageGenerationResult:
        validate_klein_seed(seed)
        first_scaled_width, first_scaled_height = _source_scaled_dimensions(first_image)
        _source_scaled_dimensions(second_image)
        target_width, target_height, schedule_width, schedule_height = (
            _target_geometry(first_scaled_width, first_scaled_height, width, height)
        )
        started = time.perf_counter()
        models_reused = self.ensure_identity(identity) and self.transformer is not None
        conditioning_reused = (
            self.conditioning is not None and self.conditioning[0] == prompt
        )
        if not conditioning_reused:
            self.conditioning = (
                prompt,
                _encode_prompt(
                    prompt,
                    identity.text_encoder.path,
                    identity.tokenizer,
                    self.device,
                ),
            )
        if self.vae is None:
            self.vae = _load_vae(identity.vae.path, self.device)
        first, first_reused = self._reference(0, first_image, "nearest-exact")
        second, second_reused = self._reference(1, second_image, "lanczos")
        if self.transformer is None:
            self.transformer = _load_transformer(identity.diffusion.path, self.device)

        assert self.conditioning is not None
        context = self.conditioning[1]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        schedule = _sigmas_for_dimensions(
            4, schedule_width, schedule_height, self.device
        )
        noise = torch.randn(
            (
                1,
                128,
                target_height // KLEIN_ALIGNMENT,
                target_width // KLEIN_ALIGNMENT,
            ),
            generator=generator,
        )
        latent = noise.to(self.device) * schedule[0]
        reference_latents = (first.latent, second.latent)
        with torch.inference_mode():
            for current, following in pairwise(schedule):
                prediction = self.transformer(
                    latent.to(torch.bfloat16),
                    current.expand(1),
                    context.to(torch.bfloat16),
                    None,
                    reference_latents,
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                denoised = latent - prediction.float() * current
                derivative = (latent - denoised) / current
                latent = latent + derivative * (following - current)
            running_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
                device=latent.device, dtype=latent.dtype
            )
            running_var = self.vae.bn.running_var.view(1, -1, 1, 1).to(
                device=latent.device, dtype=latent.dtype
            )
            latent = latent * torch.sqrt(running_var + 1e-4) + running_mean
            decoded = self.vae.decode(
                _unpack_latent(latent).to(torch.bfloat16), return_dict=False
            )[0]
        pixels = ((decoded.float().clamp(-1, 1) + 1) * 127.5).byte()
        pixels = pixels[0].permute(1, 2, 0).cpu().numpy()
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(output, format="PNG")
        return TwoImageGenerationResult(
            output=output,
            elapsed_seconds=time.perf_counter() - started,
            conditioning_reused=conditioning_reused,
            reference_reused=(first_reused, second_reused),
            models_reused=models_reused,
        )

    def __enter__(self) -> Self:
        return self


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonical FLUX.2 Klein 9B distilled two-image operation"
    )
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--first-image", type=Path, required=True)
    parser.add_argument("--second-image", type=Path, required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = Klein9BIdentity.from_paths(
        args.diffusion, args.text_encoder, args.vae, args.tokenizer
    )
    with Klein9BTwoImageRuntime() as runtime:
        for index, seed in enumerate(args.seed):
            output = args.output.with_stem(f"{args.output.stem}-{index:02d}-{seed}")
            result = runtime.generate_two_image(
                identity,
                args.prompt,
                args.first_image,
                args.second_image,
                seed,
                output,
                width=args.width,
                height=args.height,
            )
            print(
                json.dumps(
                    {
                        "output": str(result.output),
                        "seconds": result.elapsed_seconds,
                        "conditioning_reused": result.conditioning_reused,
                        "reference_reused": result.reference_reused,
                        "models_reused": result.models_reused,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
