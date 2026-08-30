from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .i2v import (
    NEGATIVE_PROMPT,
    SourceImageIdentity,
    WanI2VRecipe,
    _load_source_image,
    _load_vae_encoder,
    _resize_source,
)
from .model import WanT2VTransformer
from .pipeline import (
    LATENT_MEAN,
    LATENT_STD,
    GenerationResult,
    WanSession,
    canonical_sigmas,
    process_latent_out,
    save_video,
)

POSITIVE_PROMPT = (
    "the woman is posing in a photograph studio, then turns around and faces "
    "away from the camera."
)


@dataclass(frozen=True)
class WanFLFRecipe(WanI2VRecipe):
    high_lora_strength: float = 1.0
    low_lora_strength: float = 1.0
    positive: str = POSITIVE_PROMPT
    negative: str = NEGATIVE_PROMPT

    def validate(self) -> None:
        expected = WanFLFRecipe()
        fixed = (
            "shift",
            "steps",
            "split_step",
            "cfg",
            "width",
            "height",
            "duration",
            "fps",
        )
        mismatches = [
            name for name in fixed if getattr(self, name) != getattr(expected, name)
        ]
        if mismatches:
            raise ValueError(
                f"canonical Wan FLF runtime does not support changed settings: {mismatches}"
            )
        if self.frame_count != 81:
            raise ValueError("canonical Wan FLF frame count must be 81")


@dataclass(frozen=True)
class OrderedSourceIdentity:
    first: SourceImageIdentity
    last: SourceImageIdentity


@dataclass(frozen=True)
class FLFConditioning:
    identity: OrderedSourceIdentity
    latent: torch.Tensor
    mask: torch.Tensor


@torch.inference_mode()
def _build_flf_conditioning(
    first_path: str | Path,
    last_path: str | Path,
    identity: OrderedSourceIdentity,
    recipe: WanFLFRecipe,
    device: torch.device,
) -> FLFConditioning:
    first = _resize_source(_load_source_image(first_path), recipe.width, recipe.height)
    last = _resize_source(_load_source_image(last_path), recipe.width, recipe.height)
    video = torch.full(
        (recipe.frame_count, recipe.height, recipe.width, 3),
        0.5,
        dtype=first.dtype,
    )
    video[: first.shape[0]] = first
    video[-last.shape[0] :] = last
    encoder_input = (video.movedim(-1, 1).movedim(1, 0).unsqueeze(0).mul(2).sub(1)).to(
        device=device, dtype=torch.bfloat16
    )
    encoder = _load_vae_encoder(recipe.vae, device)
    latent = encoder.encode(encoder_input).float().cpu()
    del encoder, encoder_input, first, last, video
    gc.collect()
    torch.cuda.empty_cache()

    latent_frames = (recipe.frame_count - 1) // 4 + 1
    mask = torch.ones(
        (
            1,
            1,
            latent_frames * 4,
            recipe.height // 8,
            recipe.width // 8,
        ),
        dtype=torch.float32,
    )
    mask[:, :, :4] = 0
    mask[:, :, -1:] = 0
    mask = mask.view(
        1,
        latent_frames,
        4,
        recipe.height // 8,
        recipe.width // 8,
    ).transpose(1, 2)
    return FLFConditioning(identity=identity, latent=latent, mask=mask)


def _model_conditioning(
    conditioning: FLFConditioning, device: torch.device
) -> torch.Tensor:
    latent = conditioning.latent.to(device=device)
    mean = latent.new_tensor(LATENT_MEAN).view(1, 16, 1, 1, 1)
    std = latent.new_tensor(LATENT_STD).view(1, 16, 1, 1, 1)
    normalized = (latent - mean) / std
    mask = 1 - conditioning.mask.to(device=device)
    return torch.cat((mask, normalized), dim=1).to(torch.float16)


class WanFLFSession(WanSession):
    recipe: WanFLFRecipe

    def __init__(
        self,
        recipe: WanFLFRecipe | None = None,
        device: str | torch.device = "cuda:0",
    ):
        super().__init__(recipe or WanFLFRecipe(), device)
        self._flf_conditioning: FLFConditioning | None = None

    def destroy(self) -> None:
        self._flf_conditioning = None
        super().destroy()

    def replaced(self, recipe: WanFLFRecipe) -> WanFLFSession:
        self.destroy()
        return WanFLFSession(recipe, self.device)

    def _ensure_flf_conditioning(
        self, first_path: str | Path, last_path: str | Path
    ) -> FLFConditioning:
        identity = OrderedSourceIdentity(
            first=SourceImageIdentity.from_path(first_path),
            last=SourceImageIdentity.from_path(last_path),
        )
        if (
            self._flf_conditioning is not None
            and self._flf_conditioning.identity == identity
        ):
            return self._flf_conditioning
        self._flf_conditioning = _build_flf_conditioning(
            first_path, last_path, identity, self.recipe, self.device
        )
        return self._flf_conditioning

    @torch.inference_mode()
    def generate(
        self,
        first_path: str | Path,
        last_path: str | Path,
        output_path: str | Path,
        *,
        seed: int = 984937593540091,
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
    ) -> GenerationResult:
        self._require_alive()
        if self.recipe.identity != self._identity:
            raise RuntimeError(
                "Wan FLF recipe identity changed after session construction"
            )
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        started = time.perf_counter()
        image = self._ensure_flf_conditioning(first_path, last_path)
        timings["image_conditioning"] = time.perf_counter() - started

        started = time.perf_counter()
        positive, _negative = self._ensure_conditioning(
            self.recipe.positive if positive_prompt is None else positive_prompt,
            self.recipe.negative if negative_prompt is None else negative_prompt,
        )
        timings["conditioning"] = time.perf_counter() - started

        latent_shape = (
            1,
            16,
            (self.recipe.frame_count - 1) // 4 + 1,
            self.recipe.height // 8,
            self.recipe.width // 8,
        )
        generator = torch.manual_seed(seed)
        noise = torch.randn(
            latent_shape, dtype=torch.float32, generator=generator, device="cpu"
        )
        x = noise.to(self.device)
        sigmas = canonical_sigmas(self.recipe.shift, self.recipe.steps).to(self.device)
        image_conditioning = _model_conditioning(image, self.device)
        context = positive.to(self.device, dtype=torch.float16)

        high = WanT2VTransformer(self.high_weights)
        self.high_weights.activate(self.device)
        started = time.perf_counter()
        for index in range(self.recipe.split_step):
            timestep = (sigmas[index] * 1000).reshape(1)
            model_input = torch.cat((x.to(torch.float16), image_conditioning), dim=1)
            flow = high(model_input, timestep, context).float()
            x = x + flow * (sigmas[index + 1] - sigmas[index])
        timings["high_noise"] = time.perf_counter() - started
        del high
        self.high_weights.deactivate()
        torch.cuda.empty_cache()

        low = WanT2VTransformer(self.low_weights)
        self.low_weights.activate(self.device)
        started = time.perf_counter()
        for index in range(self.recipe.split_step, self.recipe.steps):
            timestep = (sigmas[index] * 1000).reshape(1)
            model_input = torch.cat((x.to(torch.float16), image_conditioning), dim=1)
            flow = low(model_input, timestep, context).float()
            x = x + flow * (sigmas[index + 1] - sigmas[index])
        timings["low_noise"] = time.perf_counter() - started
        x = process_latent_out(x)
        del low, context, image_conditioning
        self.low_weights.deactivate()
        torch.cuda.empty_cache()

        started = time.perf_counter()
        if self._vae is None:
            from .vae import load_vae

            self._vae = load_vae(self.recipe.vae, self.device)
        images = self._vae.decode(x.to(torch.bfloat16)).float()
        images = images.add_(1.0).div_(2.0).clamp_(0.0, 1.0).movedim(1, -1)[0].cpu()
        timings["decode"] = time.perf_counter() - started

        started = time.perf_counter()
        save_video(images, output_path, self.recipe.fps)
        timings["save"] = time.perf_counter() - started
        timings["total"] = time.perf_counter() - total_start
        return GenerationResult(
            output_path=str(Path(output_path).resolve()),
            seed=seed,
            frame_count=images.shape[0],
            fps=self.recipe.fps,
            duration=images.shape[0] / self.recipe.fps,
            width=images.shape[2],
            height=images.shape[1],
            container="mp4",
            codec="h264",
            pixel_format="yuv420p",
            timings=timings,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical Wan 2.2 14B FLF")
    parser.add_argument("first")
    parser.add_argument("last")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=984937593540091)
    args = parser.parse_args()
    session = WanFLFSession()
    result = session.generate(args.first, args.last, args.output, seed=args.seed)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
