"""Native LTX 2.5 operations from the curated Comfy workflows."""

from __future__ import annotations

from pathlib import Path

import torch

from latentslate_engine.identity import FileContentIdentity
from latentslate_engine.ltx23.audio_vae import Ltx23AudioMelDecoder
from latentslate_engine.ltx23.flf import _guided_video_latent
from latentslate_engine.ltx23.i2v import _resize_center_bilinear
from latentslate_engine.ltx23.spatial_upsampler import Ltx23SpatialUpsampler
from latentslate_engine.ltx23.t2v import Ltx23T2VOutput
from latentslate_engine.ltx23.transformer_context import Ltx23TransformerContext
from latentslate_engine.ltx23.video_vae import Ltx23VideoEncoder
from latentslate_engine.ltx23.vocoder import Ltx23AudioVocoder
from latentslate_engine.progress import ProgressCallback, report_progress

from .contracts import Ltx25Identity
from .enhancer import Ltx25Enhancer
from .image_conditioning import preprocess_guide_image, preprocess_source_image
from .sampling import FIRST_PASS_SIGMAS, SECOND_PASS_SIGMAS, ancestral_sample
from .text_encoder import Ltx25TextEncoder
from .video_vae import Ltx25VideoDecoder


class Ltx25Runtime:
    """Retain one model identity and its most recent prompt between requests."""

    def __init__(self, identity: Ltx25Identity) -> None:
        self.identity = identity
        self._text: Ltx25TextEncoder | None = None
        self._transformer: Ltx23TransformerContext | None = None
        self._prompt_cache: tuple[str, torch.Tensor] | None = None
        self._enhancement_cache = None
        self.conditioning_reused = False
        self._source_cache: (
            tuple[FileContentIdentity, int, int, torch.Tensor, torch.Tensor] | None
        ) = None

        self._guide_cache: (
            tuple[
                FileContentIdentity,
                FileContentIdentity,
                int,
                int,
                torch.Tensor,
                torch.Tensor,
            ]
            | None
        ) = None

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        frame_count: int,
        *,
        image_path: str | Path | None = None,
        last_image_path: str | Path | None = None,
        fps: int = 24,
        seed: int = 42,
        prompt_enhancement: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Ltx23T2VOutput:
        """Generate an aligned canvas and 8n+1 frames with audio."""
        if last_image_path is not None and image_path is None:
            raise ValueError("First/last-frame generation requires both images")
        alignment = 32 if last_image_path is not None else 64
        if (
            width < alignment
            or height < alignment
            or width % alignment
            or height % alignment
        ):
            raise ValueError(
                f"LTX output dimensions must be positive multiples of {alignment}"
            )
        if frame_count < 1 or frame_count % 8 != 1:
            raise ValueError("LTX output frame count must be 8n+1")
        if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
            raise ValueError("LTX frame rate must be a positive integer")
        identity = self.identity
        device = torch.device("cuda", identity.device_index)
        if prompt_enhancement:
            if identity.prompt_enhancer_path is None:
                raise ValueError("Prompt enhancement requires its model artifact")
            image_identity = (
                FileContentIdentity.from_path(image_path)
                if image_path is not None
                else None
            )
            key = (
                prompt,
                image_identity,
                (width, height) if last_image_path is not None else None,
            )
            if self._enhancement_cache is None or self._enhancement_cache[0] != key:
                report_progress(progress, 0.01, "Loading prompt enhancer")
                image = None
                if image_path is not None:
                    image = (
                        preprocess_guide_image(image_path, width, height)
                        if last_image_path is not None
                        else preprocess_source_image(image_path)
                    ).movedim(1, -1)
                enhancer = Ltx25Enhancer(
                    identity.prompt_enhancer_path, identity.device_index
                )
                try:
                    enhanced = enhancer.enhance(
                        prompt,
                        image,
                        progress=lambda i, n: report_progress(
                            progress,
                            0.02,
                            "Enhancing prompt",
                            stage_progress=i / n,
                            detail=f"Generated {i} tokens (up to {n})",
                        ),
                    )
                finally:
                    enhancer.close()
                self._enhancement_cache = (key, enhanced)
            prompt = self._enhancement_cache[1]
        report_progress(progress, 0.03, "Text conditioning")
        self.conditioning_reused = (
            self._prompt_cache is not None and self._prompt_cache[0] == prompt
        )
        if self.conditioning_reused:
            condition = self._prompt_cache[1]
        else:
            if self._text is None:
                self._text = Ltx25TextEncoder(
                    identity.text_encoder_path, identity.device_index
                )
            condition = self._text.encode(prompt)
            self._prompt_cache = (prompt, condition)
        report_progress(progress, 0.12, "Loading transformer")
        if self._transformer is None:
            self._transformer = Ltx23TransformerContext(
                identity.diffusion_path,
                identity.device_index,
                lora_paths=identity.transformer_adapters,
            )
        if last_image_path is not None:
            samples = self._sample_guides(
                condition,
                image_path,
                last_image_path,
                width,
                height,
                frame_count,
                fps,
                seed,
                progress,
            )
            return self._decode(samples, fps, progress)
        if identity.upsampler_path is None:
            raise ValueError("Two-stage LTX generation requires a spatial upsampler")
        latents = [
            torch.zeros(
                1,
                128,
                (frame_count - 1) // 8 + 1,
                height // 64,
                width // 64,
                device=device,
            ),
            torch.zeros(1, 8, round(frame_count / fps * 25.0), 16, device=device),
        ]
        source_frames = None
        masks = None
        if image_path is not None:
            key = (FileContentIdentity.from_path(image_path), width, height)
            if self._source_cache is None or self._source_cache[:3] != key:
                source = preprocess_source_image(image_path)
                encoder = Ltx23VideoEncoder(
                    identity.video_vae_path, device=str(device), component_prefix=""
                )
                try:
                    low = encoder.encode(
                        _resize_center_bilinear(source, width // 2, height // 2)
                    ).cpu()
                    full = encoder.encode(
                        _resize_center_bilinear(source, width, height)
                    ).cpu()
                finally:
                    encoder.close()
                self._source_cache = (*key, low, full)
            source_frames = self._source_cache[3:]
            latents[0][:, :, :1] = source_frames[0].to(device)
            masks = [torch.ones_like(item) for item in latents]
            masks[0][:, :, :1] = 0.3
        first = ancestral_sample(
            self._transformer,
            condition,
            latents,
            FIRST_PASS_SIGMAS,
            masks=masks,
            seed=seed,
            frame_rate=fps,
            step_callback=lambda i, n: report_progress(
                progress, 0.2 + 0.3 * i / n, "First-pass sampling", stage_progress=i / n
            ),
        )
        del latents
        report_progress(progress, 0.52, "Spatial upscale")
        upsampler = Ltx23SpatialUpsampler(
            identity.upsampler_path,
            identity.video_vae_path,
            device=str(device),
            statistics_prefix="",
        )
        try:
            first[0] = upsampler.upsample(first[0])
        finally:
            upsampler.close()
        masks = None
        if source_frames is not None:
            first[0][:, :, :1] = source_frames[1].to(device)
            masks = [torch.ones_like(item) for item in first]
            masks[0][:, :, :1] = 0.0
        second = ancestral_sample(
            self._transformer,
            condition,
            first,
            SECOND_PASS_SIGMAS,
            masks=masks,
            seed=42,
            frame_rate=fps,
            step_callback=lambda i, n: report_progress(
                progress,
                0.6 + 0.15 * i / n,
                "Second-pass sampling",
                stage_progress=i / n,
            ),
        )
        del first, condition
        return self._decode(second, fps, progress)

    def _decode(self, samples, fps, progress):
        identity = self.identity
        device = torch.device("cuda", identity.device_index)
        report_progress(progress, 0.78, "Video decode")
        video = Ltx25VideoDecoder(identity.video_vae_path, device=str(device))
        try:
            frames = video.decode(samples[0]).movedim(1, -1)
        finally:
            video.close()
        report_progress(progress, 0.85, "Audio decode")
        audio = Ltx23AudioMelDecoder(identity.audio_vae_path, device=str(device))
        try:
            mel = audio.decode(samples[1]).transpose(2, 3)
        finally:
            audio.close()
        del samples
        report_progress(progress, 0.9, "Audio synthesis")
        vocoder = Ltx23AudioVocoder(identity.audio_vae_path, device=str(device))
        try:
            waveform = vocoder.decode(mel).cpu()
        finally:
            vocoder.close()
        return Ltx23T2VOutput(frames=frames, waveform=waveform, frame_rate=fps)

    def _sample_guides(
        self,
        condition,
        first_path,
        last_path,
        width,
        height,
        frame_count,
        fps,
        seed,
        progress,
    ):
        device = torch.device("cuda", self.identity.device_index)
        key = (
            FileContentIdentity.from_path(first_path),
            FileContentIdentity.from_path(last_path),
            width,
            height,
        )
        if self._guide_cache is None or self._guide_cache[:4] != key:
            encoder = Ltx23VideoEncoder(
                self.identity.video_vae_path, device=str(device), component_prefix=""
            )
            try:
                first = encoder.encode(
                    preprocess_guide_image(first_path, width, height)
                ).cpu()
                last = encoder.encode(
                    preprocess_guide_image(last_path, width, height)
                ).cpu()
            finally:
                encoder.close()
            self._guide_cache = (*key, first, last)
        first, last = self._guide_cache[4:]
        video, mask, coords, entries = _guided_video_latent(
            first,
            last,
            width,
            height,
            (frame_count - 1) // 8 + 1,
            device,
        )
        audio = torch.zeros(1, 8, round(frame_count / fps * 25.0), 16, device=device)
        sampled = ancestral_sample(
            self._transformer,
            condition,
            [video, audio],
            FIRST_PASS_SIGMAS,
            seed=seed,
            frame_rate=fps,
            eta=0,
            masks=[mask, torch.ones_like(audio)],
            keyframe_idxs=coords,
            guide_attention_entries=entries,
            step_callback=lambda i, n: report_progress(
                progress, 0.2 + 0.55 * i / n, "Sampling", stage_progress=i / n
            ),
        )
        sampled[0] = sampled[0][:, :, :-2]
        return sampled

    def close(self) -> None:
        self._enhancement_cache = None
        self._prompt_cache = None
        self._source_cache = None
        self._guide_cache = None
        if self._text is not None:
            self._text.close()
            self._text = None
        if self._transformer is not None:
            self._transformer.close()
            self._transformer = None
