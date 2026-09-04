from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from latentslate_engine.identity import FileContentIdentity as SourceImageIdentity
from latentslate_engine.progress import ProgressCallback, report_progress

from .model import WanT2VTransformer
from .pipeline import (
    FRAME_RATE,
    LATENT_MEAN,
    LATENT_STD,
    GenerationResult,
    WanRecipe,
    WanSession,
    canonical_sigmas,
    cpu_noise,
    process_latent_out,
    save_half_open_video,
    transformer_workspace_bytes,
    validate_request,
)
from .vae import AttentionBlock, CausalConv3d, ResidualBlock, RMSNorm
from .weights import TensorStore

POSITIVE_PROMPT = (
    "the woman poses in a white photography studio, smiling and waving at the camera."
)
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


@dataclass(frozen=True)
class WanI2VRecipe(WanRecipe):
    high_checkpoint: str = r"M:\ComfyUI\models\diffusion_models\wan22\wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    high_lora: str | None = (
        r"M:\ComfyUI\models\loras\wan\wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
    )
    low_checkpoint: str = r"M:\ComfyUI\models\diffusion_models\wan22\wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
    low_lora: str | None = (
        r"M:\ComfyUI\models\loras\wan\wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
    )
    positive: str = POSITIVE_PROMPT
    negative: str = NEGATIVE_PROMPT

    def validate(self) -> None:
        expected = WanI2VRecipe()
        fixed = (
            "shift",
            "steps",
            "split_step",
            "cfg",
        )
        mismatches = [
            name for name in fixed if getattr(self, name) != getattr(expected, name)
        ]
        if mismatches:
            raise ValueError(
                f"Wan I2V turbo runtime does not support changed settings: {mismatches}"
            )
        validate_request(self.width, self.height, self.frame_count, 0)


@dataclass(frozen=True)
class ImageConditioningIdentity:
    source: SourceImageIdentity
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True)
class ImageConditioning:
    identity: ImageConditioningIdentity
    latent: torch.Tensor
    mask: torch.Tensor


class _EncoderResample(nn.Module):
    def __init__(self, dim: int, temporal: bool):
        super().__init__()
        self.temporal = temporal
        self.resample = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(dim, dim, 3, stride=(2, 2)),
        )
        if temporal:
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )

    def forward(self, x, cache, index, *, final: bool):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.reshape(b, t, c, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)
        if not self.temporal or cache is None:
            return x

        slot = index[0]
        if cache[slot] is None:
            cache[slot] = x
        else:
            current = x[:, :, -1:]
            x = self.time_conv(torch.cat([cache[slot][:, :, -1:], x], 2))
            cache[slot] = current
            deferred = cache[slot + 1]
            if deferred is not None:
                x = torch.cat([deferred, x], 2)
                cache[slot + 1] = None
            if x.shape[2] == 1 and not final:
                cache[slot + 1] = x
                x = None
        index[0] += 2
        return x


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [96, 96, 192, 384, 384]
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)
        layers: list[nn.Module] = []
        temporal = [False, True, True]
        for stage, (in_dim, out_dim) in enumerate(pairwise(dims)):
            for _ in range(2):
                layers.append(ResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if stage != 3:
                layers.append(_EncoderResample(out_dim, temporal[stage]))
        self.downsamples = nn.ModuleList(layers)
        self.middle = nn.ModuleList(
            [
                ResidualBlock(dims[-1], dims[-1]),
                AttentionBlock(dims[-1]),
                ResidualBlock(dims[-1], dims[-1]),
            ]
        )
        self.head = nn.ModuleList(
            [
                RMSNorm(dims[-1], images=False),
                nn.SiLU(),
                CausalConv3d(dims[-1], 32, 3, padding=1),
            ]
        )

    def forward(self, x, cache, index, *, final: bool):
        slot = index[0]
        current = x[:, :, -2:]
        x = self.conv1(x, cache[slot])
        cache[slot] = current
        index[0] += 1
        for layer in self.downsamples:
            if isinstance(layer, _EncoderResample):
                x = layer(x, cache, index, final=final)
            else:
                x = layer(x, cache, index)
            if x is None:
                return None
        for layer in self.middle:
            x = layer(x, cache, index)
        for layer in self.head:
            if isinstance(layer, CausalConv3d):
                slot = index[0]
                current = x[:, :, -2:]
                x = layer(x, cache[slot])
                cache[slot] = current
                index[0] += 1
            else:
                x = layer(x)
        return x


def _encoder_cache_layers(model: nn.Module) -> int:
    causal = sum(isinstance(module, CausalConv3d) for module in model.modules())
    temporal = sum(
        isinstance(module, _EncoderResample) and module.temporal
        for module in model.modules()
    )
    return causal + temporal


class _WanVaeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.conv1 = CausalConv3d(32, 32, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        frames = 1 + ((x.shape[2] - 1) // 4) * 4
        iterations = 1 + (frames - 1) // 2
        cache = [None] * _encoder_cache_layers(self.encoder)
        output = None
        for iteration in range(iterations):
            index = [0]
            if iteration == 0:
                chunk = x[:, :, :1]
            else:
                chunk = x[:, :, 1 + 2 * (iteration - 1) : 1 + 2 * iteration]
            encoded = self.encoder(
                chunk, cache, index, final=iteration == iterations - 1
            )
            if encoded is not None:
                output = encoded if output is None else torch.cat([output, encoded], 2)
        if output is None:
            raise RuntimeError("Wan VAE encoder produced no latent")
        mean, _log_variance = self.conv1(output).chunk(2, dim=1)
        return mean


def _load_vae_encoder(path: str, device: torch.device) -> _WanVaeEncoder:
    store = TensorStore(path)
    with torch.device("meta"):
        model = _WanVaeEncoder()
    state = {
        key: store.tensor(key)
        for key in store.keys
        if key.startswith(("encoder.", "conv1."))
    }
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    relevant_missing = [
        key for key in missing if key.startswith(("encoder.", "conv1."))
    ]
    if unexpected or relevant_missing:
        raise ValueError(
            f"Wan VAE encoder state mismatch: missing={relevant_missing}, "
            f"unexpected={unexpected}"
        )
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    store.close()
    return model


def _load_source_image(path: str | Path) -> torch.Tensor:
    with av.open(str(Path(path).resolve(strict=True)), mode="r") as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            image = frame.to_ndarray(format="rgb24")
            if frame.rotation != 0:
                image = np.rot90(
                    image, k=round(frame.rotation // 90), axes=(0, 1)
                ).copy()
            return torch.from_numpy(image).float().div_(255.0).unsqueeze(0)
    raise ValueError(f"source image has no decodable frame: {path}")


def _resize_source(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    samples = image.movedim(-1, 1)
    old_width = samples.shape[-1]
    old_height = samples.shape[-2]
    old_aspect = old_width / old_height
    new_aspect = width / height
    x = 0
    y = 0
    if old_aspect > new_aspect:
        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
    samples = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    return F.interpolate(samples, size=(height, width), mode="bilinear").movedim(1, -1)


@torch.inference_mode()
def _build_image_conditioning(
    source_path: str | Path,
    identity: ImageConditioningIdentity,
    recipe: WanI2VRecipe,
    device: torch.device,
) -> ImageConditioning:
    source = _resize_source(
        _load_source_image(source_path), identity.width, identity.height
    )
    video = torch.full(
        (identity.frame_count, identity.height, identity.width, 3),
        0.5,
        dtype=source.dtype,
    )
    video[: source.shape[0]] = source
    encoder_input = (video.movedim(-1, 1).movedim(1, 0).unsqueeze(0).mul(2).sub(1)).to(
        device=device, dtype=torch.bfloat16
    )
    encoder = _load_vae_encoder(recipe.vae, device)
    latent = encoder.encode(encoder_input).float().cpu()
    del encoder, encoder_input, source, video
    gc.collect()
    torch.cuda.empty_cache()
    mask = torch.ones(
        (
            1,
            1,
            (identity.frame_count - 1) // 4 + 1,
            identity.height // 8,
            identity.width // 8,
        ),
        dtype=torch.float32,
    )
    mask[:, :, 0] = 0
    return ImageConditioning(identity=identity, latent=latent, mask=mask)


def _model_conditioning(image: ImageConditioning, device: torch.device) -> torch.Tensor:
    latent = image.latent.to(device=device)
    mean = latent.new_tensor(LATENT_MEAN).view(1, 16, 1, 1, 1)
    std = latent.new_tensor(LATENT_STD).view(1, 16, 1, 1, 1)
    normalized = (latent - mean) / std
    mask = (1 - image.mask.to(device=device)).repeat(1, 4, 1, 1, 1)
    return torch.cat((mask, normalized), dim=1).to(torch.float16)


class WanI2VSession(WanSession):
    recipe: WanI2VRecipe

    def __init__(
        self,
        recipe: WanI2VRecipe | None = None,
        device: str | torch.device = "cuda:0",
    ):
        super().__init__(recipe or WanI2VRecipe(), device)
        self._image_conditioning: ImageConditioning | None = None

    def destroy(self) -> None:
        self._image_conditioning = None
        super().destroy()

    def replaced(self, recipe: WanI2VRecipe) -> WanI2VSession:
        recipe.validate()
        if recipe.identity == self.identity:
            self.recipe = recipe
            return self
        self.destroy()
        return WanI2VSession(recipe, self.device)

    def _ensure_image_conditioning(
        self, source_path: str | Path, width: int, height: int, frame_count: int
    ) -> ImageConditioning:
        identity = ImageConditioningIdentity(
            source=SourceImageIdentity.from_path(source_path),
            width=width,
            height=height,
            frame_count=frame_count,
        )
        if (
            self._image_conditioning is not None
            and self._image_conditioning.identity == identity
        ):
            return self._image_conditioning
        self._image_conditioning = _build_image_conditioning(
            source_path, identity, self.recipe, self.device
        )
        return self._image_conditioning

    @torch.inference_mode()
    def generate(
        self,
        source_path: str | Path,
        output_path: str | Path,
        *,
        seed: int = 264244520398999,
        width: int | None = None,
        height: int | None = None,
        frame_count: int | None = None,
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        self._require_alive()
        if self.recipe.identity != self._identity:
            raise RuntimeError(
                "Wan I2V recipe identity changed after session construction"
            )
        width = self.recipe.width if width is None else width
        height = self.recipe.height if height is None else height
        frame_count = self.recipe.frame_count if frame_count is None else frame_count
        validate_request(width, height, frame_count, seed)
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        report_progress(progress, 0.02, "Source image conditioning")
        started = time.perf_counter()
        image = self._ensure_image_conditioning(source_path, width, height, frame_count)
        timings["image_conditioning"] = time.perf_counter() - started

        report_progress(progress, 0.12, "Text conditioning")
        started = time.perf_counter()
        positive, _negative = self._ensure_conditioning(
            self.recipe.positive if positive_prompt is None else positive_prompt,
            self.recipe.negative if negative_prompt is None else negative_prompt,
        )
        timings["conditioning"] = time.perf_counter() - started

        noise = cpu_noise(seed, width, height, frame_count)
        x = noise.to(self.device)
        sigmas = canonical_sigmas(self.recipe.shift, self.recipe.steps).to(self.device)
        image_conditioning = _model_conditioning(image, self.device)
        context = positive.to(self.device, dtype=torch.float16)
        workspace_bytes = transformer_workspace_bytes(width, height, frame_count)

        high = WanT2VTransformer(self.high_weights)
        report_progress(progress, 0.22, "High-noise sampling", stage_progress=0.0)
        self.high_weights.activate(self.device, workspace_bytes=workspace_bytes)
        started = time.perf_counter()
        for index in range(self.recipe.split_step):
            timestep = (sigmas[index] * 1000).reshape(1)
            model_input = torch.cat((x.to(torch.float16), image_conditioning), dim=1)
            flow = high(model_input, timestep, context).float()
            x = x + flow * (sigmas[index + 1] - sigmas[index])
            completed = index + 1
            report_progress(
                progress,
                0.22 + 0.2 * completed / self.recipe.split_step,
                "High-noise sampling",
                stage_progress=completed / self.recipe.split_step,
                detail=f"Step {completed} of {self.recipe.split_step}",
            )
        timings["high_noise"] = time.perf_counter() - started
        del high
        self.high_weights.deactivate()
        torch.cuda.empty_cache()

        low = WanT2VTransformer(self.low_weights)
        low_steps = self.recipe.steps - self.recipe.split_step
        report_progress(progress, 0.42, "Low-noise sampling", stage_progress=0.0)
        self.low_weights.activate(self.device, workspace_bytes=workspace_bytes)
        started = time.perf_counter()
        for index in range(self.recipe.split_step, self.recipe.steps):
            timestep = (sigmas[index] * 1000).reshape(1)
            model_input = torch.cat((x.to(torch.float16), image_conditioning), dim=1)
            flow = low(model_input, timestep, context).float()
            x = x + flow * (sigmas[index + 1] - sigmas[index])
            completed = index - self.recipe.split_step + 1
            report_progress(
                progress,
                0.42 + 0.2 * completed / low_steps,
                "Low-noise sampling",
                stage_progress=completed / low_steps,
                detail=f"Step {completed} of {low_steps}",
            )
        timings["low_noise"] = time.perf_counter() - started
        x = process_latent_out(x)
        del low, context, image_conditioning
        self.low_weights.deactivate()
        torch.cuda.empty_cache()

        report_progress(progress, 0.62, "VAE decode")
        started = time.perf_counter()
        if self._vae is None:
            from .vae import load_vae

            self._vae = load_vae(self.recipe.vae, self.device)
        images = self._vae.decode(x.to(torch.bfloat16)).float()
        images = images.add_(1.0).div_(2.0).clamp_(0.0, 1.0).movedim(1, -1)[0].cpu()
        timings["decode"] = time.perf_counter() - started

        report_progress(progress, 0.9, "Artifact encoding")
        started = time.perf_counter()
        delivered_images = save_half_open_video(images, output_path, FRAME_RATE)
        timings["save"] = time.perf_counter() - started
        report_progress(progress, 1.0, "Artifact encoding", stage_progress=1.0)
        timings["total"] = time.perf_counter() - total_start
        return GenerationResult(
            output_path=str(Path(output_path).resolve()),
            seed=seed,
            frame_count=delivered_images.shape[0],
            fps=float(FRAME_RATE),
            duration=delivered_images.shape[0] / FRAME_RATE,
            width=images.shape[2],
            height=images.shape[1],
            container="mp4",
            codec="h264",
            pixel_format="yuv420p",
            timings=timings,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical Wan 2.2 14B LightX2V I2V")
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=264244520398999)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=81)
    args = parser.parse_args()
    session = WanI2VSession()
    result = session.generate(
        args.source,
        args.output,
        seed=args.seed,
        width=args.width,
        height=args.height,
        frame_count=args.frames,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
