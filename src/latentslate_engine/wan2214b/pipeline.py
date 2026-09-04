from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import av
import torch

from latentslate_engine.progress import ProgressCallback, report_progress

from . import contracts as _contracts
from .model import DIM, FFN_DIM, WanT2VTransformer
from .text import Umt5Encoder
from .vae import WanVaeDecoder, load_vae
from .weights import WanWeights

FRAME_RATE = _contracts.FRAME_RATE
MAX_ASPECT_DENOMINATOR = _contracts.MAX_ASPECT_DENOMINATOR
MAX_ASPECT_NUMERATOR = _contracts.MAX_ASPECT_NUMERATOR
MAX_FRAME_COUNT = _contracts.MAX_FRAME_COUNT
MAX_PIXELS = _contracts.MAX_PIXELS
MAX_SEED = _contracts.MAX_SEED
MIN_FRAME_COUNT = _contracts.MIN_FRAME_COUNT
MIN_SIDE = _contracts.MIN_SIDE
NEGATIVE_PROMPT = _contracts.NEGATIVE_PROMPT
POSITIVE_PROMPT = _contracts.POSITIVE_PROMPT
WanRecipe = _contracts.WanRecipe
validate_request = _contracts.validate_request

LATENT_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
LATENT_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)


@dataclass(frozen=True)
class GenerationResult:
    output_path: str
    seed: int
    frame_count: int
    fps: float
    duration: float
    width: int
    height: int
    container: str
    codec: str
    pixel_format: str
    timings: dict[str, float]


def latent_shape(width: int, height: int, frame_count: int) -> tuple[int, ...]:
    return (1, 16, (frame_count - 1) // 4 + 1, height // 8, width // 8)


def transformer_token_count(width: int, height: int, frame_count: int) -> int:
    return ((frame_count - 1) // 4 + 1) * (height // 16) * (width // 16)


def transformer_workspace_bytes(width: int, height: int, frame_count: int) -> int:
    tokens = transformer_token_count(width, height, frame_count)
    return tokens * (8 * DIM + 2 * FFN_DIM) * torch.float16.itemsize


def cpu_noise(seed: int, width: int, height: int, frame_count: int) -> torch.Tensor:
    generator = torch.manual_seed(seed)
    return torch.randn(
        latent_shape(width, height, frame_count),
        dtype=torch.float32,
        generator=generator,
        device="cpu",
    )


def canonical_sigmas(shift: float, steps: int) -> torch.Tensor:
    timesteps = torch.arange(1, 1001, dtype=torch.float32) / 1000
    schedule = shift * timesteps / (1 + (shift - 1) * timesteps)
    stride = len(schedule) / steps
    return torch.tensor(
        [float(schedule[-(1 + int(index * stride))]) for index in range(steps)] + [0.0],
        dtype=torch.float32,
    )


def process_latent_out(latent: torch.Tensor) -> torch.Tensor:
    mean = latent.new_tensor(LATENT_MEAN).view(1, 16, 1, 1, 1)
    std = latent.new_tensor(LATENT_STD).view(1, 16, 1, 1, 1)
    return latent * std + mean


def save_video(images: torch.Tensor, path: str | Path, fps: float) -> None:
    path = str(Path(path).resolve())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with av.open(
        path,
        mode="w",
        format="mp4",
        options={"movflags": "use_metadata_tags+faststart"},
    ) as output:
        stream = output.add_stream("h264", rate=Fraction(round(fps * 1000), 1000))
        stream.width = images.shape[2]
        stream.height = images.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.codec_context.color_primaries = 1
        stream.codec_context.color_trc = 13
        stream.codec_context.colorspace = 1
        stream.codec_context.color_range = 1
        arrays = images.mul(255).clamp(0, 255).byte().contiguous().numpy()
        for array in arrays:
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame = frame.reformat(format="yuv420p", dst_colorspace=1)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def half_open_delivery(images: torch.Tensor) -> torch.Tensor:
    if images.shape[0] < 2:
        raise ValueError("Wan delivery requires a terminal boundary sample")
    return images[:-1]


def save_half_open_video(
    images: torch.Tensor, path: str | Path, fps: float
) -> torch.Tensor:
    delivered_images = half_open_delivery(images)
    save_video(delivered_images, path, fps)
    return delivered_images


class WanSession:
    def __init__(
        self, recipe: WanRecipe | None = None, device: str | torch.device = "cuda:0"
    ):
        self.recipe = recipe or WanRecipe()
        self.recipe.validate()
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("canonical Wan 14B runtime requires CUDA")
        self._identity = self.recipe.identity
        self._alive = True
        self._conditioning: tuple[torch.Tensor, torch.Tensor] | None = None
        self._conditioning_key: tuple[str, str] | None = None
        self._vae: WanVaeDecoder | None = None
        self.high_weights = WanWeights(
            self.recipe.high_checkpoint,
            self.recipe.high_lora,
            lora_strength=self.recipe.high_lora_strength,
            secondary_lora=self.recipe.high_secondary_lora,
            secondary_lora_strength=self.recipe.high_secondary_lora_strength,
        )
        self.low_weights = WanWeights(
            self.recipe.low_checkpoint,
            self.recipe.low_lora,
            lora_strength=self.recipe.low_lora_strength,
            secondary_lora=self.recipe.low_secondary_lora,
            secondary_lora_strength=self.recipe.low_secondary_lora_strength,
        )
        self.text_weights = WanWeights(self.recipe.text_encoder, native_fp8=False)
        if self.high_weights.identity == self.low_weights.identity:
            raise ValueError(
                "canonical high- and low-noise model/LoRA identities must be distinct"
            )

    @property
    def identity(self) -> tuple[object, ...]:
        self._require_alive()
        return self._identity

    def _require_alive(self) -> None:
        if not self._alive:
            raise RuntimeError("Wan session was destructively replaced")

    def destroy(self) -> None:
        self._alive = False
        self._conditioning = None
        self._conditioning_key = None
        self._vae = None
        self.high_weights = None
        self.low_weights = None
        self.text_weights = None
        gc.collect()
        torch.cuda.empty_cache()

    def replaced(self, recipe: WanRecipe) -> WanSession:
        recipe.validate()
        if recipe.identity == self.identity:
            self.recipe = recipe
            return self
        self.destroy()
        return WanSession(recipe, self.device)

    def _ensure_conditioning(
        self, positive_prompt: str, negative_prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (positive_prompt, negative_prompt)
        if self._conditioning is not None and self._conditioning_key == key:
            return self._conditioning
        self._conditioning = None
        self._conditioning_key = None
        if self.text_weights is None:
            self.text_weights = WanWeights(self.recipe.text_encoder, native_fp8=False)
        encoder = Umt5Encoder(self.text_weights)
        _pos_ids, _pos_mask, positive = encoder.encode(positive_prompt, self.device)
        _neg_ids, _neg_mask, negative = encoder.encode(negative_prompt, self.device)
        self.text_weights.base.close()
        self.text_weights = None
        self._conditioning = (positive, negative)
        self._conditioning_key = key
        torch.cuda.empty_cache()
        return self._conditioning

    @torch.inference_mode()
    def generate(
        self,
        output_path: str | Path,
        *,
        seed: int = 923510416338945,
        width: int | None = None,
        height: int | None = None,
        frame_count: int | None = None,
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        self._require_alive()
        if self.recipe.identity != self._identity:
            raise RuntimeError("Wan recipe identity changed after session construction")
        width = self.recipe.width if width is None else width
        height = self.recipe.height if height is None else height
        frame_count = self.recipe.frame_count if frame_count is None else frame_count
        validate_request(width, height, frame_count, seed)
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        report_progress(progress, 0.02, "Text conditioning")
        started = time.perf_counter()
        positive, _negative = self._ensure_conditioning(
            self.recipe.positive if positive_prompt is None else positive_prompt,
            self.recipe.negative if negative_prompt is None else negative_prompt,
        )
        timings["conditioning"] = time.perf_counter() - started

        noise = cpu_noise(seed, width, height, frame_count)
        x = noise.to(self.device)
        sigmas = canonical_sigmas(self.recipe.shift, self.recipe.steps).to(self.device)

        context = positive.to(self.device, dtype=torch.float16)
        high = WanT2VTransformer(self.high_weights)
        workspace_bytes = transformer_workspace_bytes(width, height, frame_count)
        report_progress(progress, 0.15, "High-noise sampling", stage_progress=0.0)
        self.high_weights.activate(self.device, workspace_bytes=workspace_bytes)
        started = time.perf_counter()
        for index in range(self.recipe.split_step):
            timestep = (sigmas[index] * 1000).reshape(1)
            flow = high(x.to(torch.float16), timestep, context).float()
            x = x + flow * (sigmas[index + 1] - sigmas[index])
            completed = index + 1
            report_progress(
                progress,
                0.15 + 0.2 * completed / self.recipe.split_step,
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
        report_progress(progress, 0.35, "Low-noise sampling", stage_progress=0.0)
        self.low_weights.activate(self.device, workspace_bytes=workspace_bytes)
        started = time.perf_counter()
        for index in range(self.recipe.split_step, self.recipe.steps):
            timestep = (sigmas[index] * 1000).reshape(1)
            flow = low(x.to(torch.float16), timestep, context).float()
            x = x + flow * (sigmas[index + 1] - sigmas[index])
            completed = index - self.recipe.split_step + 1
            report_progress(
                progress,
                0.35 + 0.2 * completed / low_steps,
                "Low-noise sampling",
                stage_progress=completed / low_steps,
                detail=f"Step {completed} of {low_steps}",
            )
        timings["low_noise"] = time.perf_counter() - started
        x = process_latent_out(x)
        del low, context
        self.low_weights.deactivate()
        torch.cuda.empty_cache()

        report_progress(progress, 0.55, "VAE decode")
        started = time.perf_counter()
        if self._vae is None:
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


def result_json(result: GenerationResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False, indent=2)
