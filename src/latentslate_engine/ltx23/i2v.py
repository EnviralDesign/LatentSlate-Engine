"""The standalone LTX 2.3 I2V operation proved by the canonical fixture."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from latentslate_engine.identity import FileContentIdentity
from latentslate_engine.progress import ProgressCallback, report_progress

from .audio_vae import Ltx23AudioMelDecoder
from .sampling import (
    FRAME_RATE,
    empty_av_latents,
    euler_sample_masked,
    ltx_temporal_shapes,
    nested_noise,
    validate_ltx_request,
)
from .spatial_upsampler import Ltx23SpatialUpsampler
from .t2v import Ltx23T2VOutput
from .text_encoder import Ltx23TextEncoder
from .transformer_context import Ltx23TransformerContext
from .video_vae import Ltx23VideoDecoder, Ltx23VideoEncoder
from .vocoder import Ltx23AudioVocoder

_FIRST_PASS_SIGMAS = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
)
_SECOND_PASS_SIGMAS = (0.85, 0.725, 0.4219, 0.0)
_SECOND_PASS_SEED = 42
_CANONICAL_FIRST_PASS_SEED = 60540193790228

Ltx23I2VOutput = Ltx23T2VOutput


@dataclass(frozen=True)
class Ltx23I2VIdentity:
    """The complete, concrete model identity of the LTX I2V operation."""

    checkpoint_path: str
    text_checkpoint_path: str
    transformer_lora_path: str
    upsampler_path: str
    lora_strength: float = 0.5
    device_index: int = 0


def _preprocess_source_image(path: str | Path, width: int, height: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.size != (width, height):
        raise ValueError(f"normalized LTX I2V source image must be {width}x{height}")

    image = image.resize((width, height), Image.Resampling.LANCZOS)
    if width > height:
        resized_width = 1536
        resized_height = int(height * (1536 / width))
    else:
        resized_height = 1536
        resized_width = int(width * (1536 / height))
    image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    pixels = np.asarray(image).astype(np.float32) / 255.0
    pixels = pixels[: (pixels.shape[0] // 2) * 2, : (pixels.shape[1] // 2) * 2]
    image_tensor = torch.from_numpy(pixels)
    image_array = (image_tensor * 255.0).byte().numpy()

    with BytesIO() as output_file:
        container = av.open(output_file, "w", format="mp4")
        try:
            stream = container.add_stream(
                "libx264", rate=1, options={"crf": "18", "preset": "veryfast"}
            )
            stream.height = image_array.shape[0]
            stream.width = image_array.shape[1]
            frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(
                format="yuv420p"
            )
            container.mux(stream.encode(frame))
            container.mux(stream.encode())
        finally:
            container.close()
        video_bytes = output_file.getvalue()

    with BytesIO(video_bytes) as video_file:
        container = av.open(video_file)
        try:
            stream = next(item for item in container.streams if item.type == "video")
            decoded = next(container.decode(stream)).to_ndarray(format="rgb24")
        finally:
            container.close()
    return (
        torch.from_numpy(decoded.astype(np.float32) / 255.0).movedim(-1, 0).unsqueeze(0)
    )


def _resize_center_bilinear(
    image: torch.Tensor, width: int, height: int
) -> torch.Tensor:
    old_height, old_width = image.shape[-2:]
    old_aspect = old_width / old_height
    new_aspect = width / height
    x = 0
    y = 0
    if old_aspect > new_aspect:
        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
    cropped = image.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    return F.interpolate(cropped, size=(height, width), mode="bilinear")


def _conditioned_video_latent(
    encoded_frame: torch.Tensor,
    width: int,
    height: int,
    video_frames: int,
    strength: float,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_height = height // 64
    latent_width = width // 64
    expected = (1, 128, 1, latent_height, latent_width)
    if tuple(encoded_frame.shape) != expected:
        raise ValueError(f"encoded I2V frame must have shape {expected}")
    samples = torch.zeros(
        (1, 128, video_frames, latent_height, latent_width),
        device=device,
        dtype=torch.float32,
    )
    samples[:, :, :1] = encoded_frame.to(device=device, dtype=torch.float32)
    mask = torch.ones_like(samples)
    mask[:, :, :1] = 1.0 - strength
    return samples, mask


class Ltx23I2VRuntime:
    """Keep exactly one I2V transformer identity warm between requests."""

    def __init__(self, identity: Ltx23I2VIdentity) -> None:
        self.identity = identity
        self._transformer: Ltx23TransformerContext | None = None
        self._text_encoder: Ltx23TextEncoder | None = None
        self._vocoder: Ltx23AudioVocoder | None = None
        self._prompt_cache: tuple[str, torch.Tensor] | None = None
        self._source_cache: (
            tuple[
                FileContentIdentity,
                int,
                int,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
            | None
        ) = None

    def replace_identity(self, identity: Ltx23I2VIdentity) -> Ltx23I2VRuntime:
        if identity == self.identity:
            return self
        self.close()
        return Ltx23I2VRuntime(identity)

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        if self._prompt_cache is not None and self._prompt_cache[0] == prompt:
            return self._prompt_cache[1]
        if self._text_encoder is None:
            self._text_encoder = Ltx23TextEncoder(
                self.identity.text_checkpoint_path,
                self.identity.checkpoint_path,
                self.identity.device_index,
            )
        condition = self._text_encoder.encode(prompt)
        self._prompt_cache = (prompt, condition)
        return condition

    def _transformer_context(self) -> Ltx23TransformerContext:
        if self._transformer is None:
            self._transformer = Ltx23TransformerContext(
                self.identity.checkpoint_path,
                self.identity.device_index,
                self.identity.transformer_lora_path,
                self.identity.lora_strength,
            )
        return self._transformer

    def _encode_source(
        self, image_path: str | Path, width: int, height: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_key = FileContentIdentity.from_path(image_path)
        cache_key = (source_key, width, height)
        if self._source_cache is None or self._source_cache[:3] != cache_key:
            source = _preprocess_source_image(image_path, width, height)
            low = None
            full = None
        else:
            _, _, _, source, low, full = self._source_cache
            return low, full
        encoder = Ltx23VideoEncoder(self.identity.checkpoint_path)
        try:
            if low is None:
                low = encoder.encode(
                    _resize_center_bilinear(source, width // 2, height // 2)
                )
            if full is None:
                full = encoder.encode(_resize_center_bilinear(source, width, height))
            self._source_cache = (source_key, width, height, source, low, full)
            return low, full
        finally:
            encoder.close()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        image_path: str | Path,
        width: int = 1280,
        height: int = 704,
        duration_seconds: float = 5.0,
        seed: int = _CANONICAL_FIRST_PASS_SEED,
        progress: ProgressCallback | None = None,
    ) -> Ltx23I2VOutput:
        """Execute the concrete two-pass, CFG=1 LTX 2.3 I2V operation."""
        validate_ltx_request(width, height, duration_seconds, seed, alignment=64)
        report_progress(progress, 0.02, "Source image conditioning")
        low_frame, full_frame = self._encode_source(image_path, width, height)
        report_progress(progress, 0.1, "Text conditioning")
        condition = self._encode_prompt(prompt)
        report_progress(progress, 0.15, "Loading transformer")
        transformer = self._transformer_context()
        device = transformer.device_index
        _, video_frames, _, _ = ltx_temporal_shapes(duration_seconds)

        first_video, first_video_mask = _conditioned_video_latent(
            low_frame, width, height, video_frames, 0.7, device
        )
        first_audio = empty_av_latents(
            width,
            height,
            duration_seconds,
            spatial_divisor=64,
            device=device,
        )[1]
        first_latents = [first_video, first_audio]
        first_masks = [first_video_mask, torch.ones_like(first_audio)]
        report_progress(progress, 0.2, "First-pass sampling", stage_progress=0.0)
        first_pass = euler_sample_masked(
            transformer,
            condition,
            first_latents,
            nested_noise(seed, first_latents),
            first_masks,
            _FIRST_PASS_SIGMAS,
            frame_rate=FRAME_RATE,
            step_callback=lambda index, count: report_progress(
                progress,
                0.2 + 0.3 * index / count,
                "First-pass sampling",
                stage_progress=index / count,
                detail=f"Step {index} of {count}",
            ),
        )
        del first_latents, first_masks, first_video, first_audio, first_video_mask

        report_progress(progress, 0.52, "Spatial refinement")
        upsampler = Ltx23SpatialUpsampler(
            self.identity.upsampler_path,
            self.identity.checkpoint_path,
        )
        try:
            second_video = upsampler.upsample(first_pass[0])
        finally:
            upsampler.close()
        second_video[:, :, :1] = full_frame.to(device=device, dtype=torch.float32)
        second_video_mask = torch.ones_like(second_video)
        second_video_mask[:, :, :1] = 0.0
        second_latents = [second_video, first_pass[1]]
        second_masks = [second_video_mask, torch.ones_like(first_pass[1])]
        report_progress(progress, 0.6, "Second-pass sampling", stage_progress=0.0)
        second_pass = euler_sample_masked(
            transformer,
            condition,
            second_latents,
            nested_noise(_SECOND_PASS_SEED, second_latents),
            second_masks,
            _SECOND_PASS_SIGMAS,
            frame_rate=FRAME_RATE,
            step_callback=lambda index, count: report_progress(
                progress,
                0.6 + 0.15 * index / count,
                "Second-pass sampling",
                stage_progress=index / count,
                detail=f"Step {index} of {count}",
            ),
        )
        del first_pass, second_latents, second_masks, condition, low_frame, full_frame

        report_progress(progress, 0.78, "Video decode")
        video_decoder = Ltx23VideoDecoder(self.identity.checkpoint_path)
        try:
            frames = video_decoder.decode(second_pass[0]).movedim(1, -1).cpu()
        finally:
            video_decoder.close()

        report_progress(progress, 0.85, "Audio decode")
        audio_decoder = Ltx23AudioMelDecoder(self.identity.checkpoint_path)
        try:
            mel = audio_decoder.decode(second_pass[1]).transpose(2, 3)
        finally:
            audio_decoder.close()
        del second_pass

        report_progress(progress, 0.9, "Audio synthesis")
        if self._vocoder is None:
            self._vocoder = Ltx23AudioVocoder(self.identity.checkpoint_path)
        waveform = self._vocoder.decode(mel).cpu()
        return Ltx23I2VOutput(frames=frames, waveform=waveform)

    def close(self) -> None:
        self._prompt_cache = None
        self._source_cache = None
        if self._text_encoder is not None:
            self._text_encoder.close()
            self._text_encoder = None
        if self._transformer is not None:
            self._transformer.close()
            self._transformer = None
        if self._vocoder is not None:
            self._vocoder.close()
            self._vocoder = None
