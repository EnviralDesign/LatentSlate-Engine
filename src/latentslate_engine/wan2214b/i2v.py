from __future__ import annotations

import argparse
import gc
import hashlib
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

from .model import WanT2VTransformer
from .pipeline import (
    LATENT_MEAN,
    LATENT_STD,
    GenerationResult,
    WanRecipe,
    WanSession,
    canonical_sigmas,
    process_latent_out,
    save_video,
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
    high_lora: str = r"M:\ComfyUI\models\loras\wan\wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
    low_checkpoint: str = r"M:\ComfyUI\models\diffusion_models\wan22\wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
    low_lora: str = r"M:\ComfyUI\models\loras\wan\wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
    positive: str = POSITIVE_PROMPT
    negative: str = NEGATIVE_PROMPT

    def validate(self) -> None:
        expected = WanI2VRecipe()
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
                f"canonical Wan I2V runtime does not support changed settings: {mismatches}"
            )
        if self.frame_count != 81:
            raise ValueError("canonical Wan I2V frame count must be 81")


@dataclass(frozen=True)
class SourceImageIdentity:
    size: int
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> SourceImageIdentity:
        digest = hashlib.sha256()
        size = 0
        with Path(path).resolve(strict=True).open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        return cls(size=size, sha256=digest.hexdigest())


@dataclass(frozen=True)
class ImageConditioning:
    identity: SourceImageIdentity
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
    identity: SourceImageIdentity,
    recipe: WanI2VRecipe,
    device: torch.device,
) -> ImageConditioning:
    source = _resize_source(
        _load_source_image(source_path), recipe.width, recipe.height
    )
    video = torch.full(
        (recipe.frame_count, recipe.height, recipe.width, 3),
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
            (recipe.frame_count - 1) // 4 + 1,
            recipe.height // 8,
            recipe.width // 8,
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
        self.destroy()
        return WanI2VSession(recipe, self.device)

    def _ensure_image_conditioning(self, source_path: str | Path) -> ImageConditioning:
        identity = SourceImageIdentity.from_path(source_path)
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
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
    ) -> GenerationResult:
        self._require_alive()
        if self.recipe.identity != self._identity:
            raise RuntimeError(
                "Wan I2V recipe identity changed after session construction"
            )
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        started = time.perf_counter()
        image = self._ensure_image_conditioning(source_path)
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
    parser = argparse.ArgumentParser(description="Canonical Wan 2.2 14B LightX2V I2V")
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=264244520398999)
    args = parser.parse_args()
    session = WanI2VSession()
    result = session.generate(args.source, args.output, seed=args.seed)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
