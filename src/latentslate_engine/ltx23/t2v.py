"""The one standalone LTX 2.3 T2V operation proved by the canonical fixture."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import av
import torch

from latentslate_engine.progress import ProgressCallback, report_progress

from .audio_vae import Ltx23AudioMelDecoder
from .contracts import Ltx23T2VIdentity
from .sampling import (
    FRAME_RATE,
    empty_av_latents,
    euler_sample,
    nested_noise,
    validate_ltx_request,
)
from .spatial_upsampler import Ltx23SpatialUpsampler
from .text_encoder import Ltx23TextEncoder
from .transformer_context import Ltx23TransformerContext
from .video_vae import Ltx23VideoDecoder
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
_CANONICAL_FIRST_PASS_SEED = 810138461690240


def _trim_windows_working_set() -> None:
    """Release inactive mapped checkpoint pages after the full media boundary."""
    if os.name != "nt":
        return
    import ctypes

    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
    ctypes.windll.psapi.EmptyWorkingSet.restype = ctypes.c_int
    ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())


@dataclass
class Ltx23T2VOutput:
    """Decoded LTX media, ready for the operation-local MP4 writer."""

    frames: torch.Tensor
    waveform: torch.Tensor
    frame_rate: int = FRAME_RATE
    sample_rate: int = 48_000

    def save_mp4(self, path: str | Path) -> None:
        """Write H.264/AAC, 30 fps, stereo 48 kHz LTX media."""
        if (
            self.frames.ndim != 5
            or self.frames.shape[0] != 1
            or self.frames.shape[-1] != 3
        ):
            raise ValueError("LTX media requires one batch of RGB video frames")
        height = self.frames.shape[2]
        width = self.frames.shape[3]
        if height < 64 or width < 64 or height % 2 or width % 2:
            raise ValueError("LTX media requires even width and height of at least 64")
        if tuple(self.waveform.shape[:2]) != (1, 2):
            raise ValueError("LTX media requires one stereo waveform")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with av.open(str(destination), mode="w") as container:
                video = container.add_stream("h264", rate=self.frame_rate)
                video.width = width
                video.height = height
                video.pix_fmt = "yuv420p"
                video.codec_context.color_primaries = 1
                video.codec_context.color_trc = 13
                video.codec_context.colorspace = 1
                video.codec_context.color_range = 1
                audio = container.add_stream(
                    "aac", rate=self.sample_rate, layout="stereo"
                )
                for image in self.frames[0]:
                    pixels = (
                        torch.clamp(image.float() * 255, 0, 255).to(torch.uint8).numpy()
                    )
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    for packet in video.encode(frame):
                        container.mux(packet)
                for packet in video.encode():
                    container.mux(packet)

                sample_count = math.ceil(
                    self.sample_rate / self.frame_rate * self.frames.shape[1]
                )
                samples = (
                    self.waveform[0, :, :sample_count].float().contiguous().numpy()
                )
                frame = av.AudioFrame.from_ndarray(
                    samples, format="fltp", layout="stereo"
                )
                frame.sample_rate = self.sample_rate
                frame.pts = 0
                for packet in audio.encode(frame):
                    container.mux(packet)
                for packet in audio.encode():
                    container.mux(packet)
        finally:
            _trim_windows_working_set()


class Ltx23T2VRuntime:
    """Keep exactly one T2V transformer identity warm between requests."""

    def __init__(self, identity: Ltx23T2VIdentity) -> None:
        self.identity = identity
        self._transformer: Ltx23TransformerContext | None = None
        self._text_encoder: Ltx23TextEncoder | None = None
        self._prompt_cache: tuple[str, torch.Tensor] | None = None

    def replace_identity(self, identity: Ltx23T2VIdentity) -> Ltx23T2VRuntime:
        """Return this warm runtime or destroy it before constructing a new identity."""
        if identity == self.identity:
            return self
        self.close()
        return Ltx23T2VRuntime(identity)

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
                lora_paths=self.identity.transformer_loras,
            )
        return self._transformer

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        width: int = 1280,
        height: int = 704,
        duration_seconds: float = 5.0,
        seed: int = _CANONICAL_FIRST_PASS_SEED,
        progress: ProgressCallback | None = None,
    ) -> Ltx23T2VOutput:
        """Execute the concrete two-pass, CFG=1 LTX 2.3 T2V operation."""
        validate_ltx_request(width, height, duration_seconds, seed, alignment=64)
        report_progress(progress, 0.03, "Text conditioning")
        condition = self._encode_prompt(prompt)
        report_progress(progress, 0.12, "Loading transformer")
        transformer = self._transformer_context()

        first_latents = empty_av_latents(
            width,
            height,
            duration_seconds,
            spatial_divisor=64,
            device=transformer.device_index,
        )
        report_progress(progress, 0.2, "First-pass sampling", stage_progress=0.0)
        first_pass = euler_sample(
            transformer,
            condition,
            first_latents,
            nested_noise(seed, first_latents),
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
        del first_latents

        report_progress(progress, 0.52, "Spatial refinement")
        upsampler = Ltx23SpatialUpsampler(
            self.identity.upsampler_path,
            self.identity.checkpoint_path,
        )
        try:
            second_video_latent = upsampler.upsample(first_pass[0])
        finally:
            upsampler.close()

        second_noise = nested_noise(
            _SECOND_PASS_SEED, [second_video_latent, first_pass[1]]
        )
        report_progress(progress, 0.6, "Second-pass sampling", stage_progress=0.0)
        second_pass = euler_sample(
            transformer,
            condition,
            [second_video_latent, first_pass[1]],
            second_noise,
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
        del first_pass, second_video_latent, second_noise, condition

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
        vocoder = Ltx23AudioVocoder(self.identity.checkpoint_path)
        try:
            waveform = vocoder.decode(mel).cpu()
        finally:
            vocoder.close()
        return Ltx23T2VOutput(frames=frames, waveform=waveform)

    def close(self) -> None:
        self._prompt_cache = None
        if self._text_encoder is not None:
            self._text_encoder.close()
            self._text_encoder = None
        if self._transformer is not None:
            self._transformer.close()
            self._transformer = None
