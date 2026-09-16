"""H3 joint audio/video generation with family-local retained state."""

import gc
import math
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors import safe_open
from safetensors.torch import load_file

from latentslate_engine.identity import FileContentIdentity
from latentslate_engine.progress import report_progress

from .audio_vae import MiniMaxH3AudioVAE
from .contracts import H3Identity, validate_request
from .media import load_audio, load_video
from .model import MiniMaxH3Model, PackedLayout
from .sampling import (
    noise,
    pack,
    res_multistep,
    simple_schedule,
    temporal_shape,
    unpack,
)
from .text import H3TextEncoder
from .video_vae import MiniMaxH3VideoVAE
from .weights import H3Weights


@dataclass
class H3Output:
    frames: torch.Tensor
    waveform: torch.Tensor
    frame_rate: int = 24
    sample_rate: int = 32000

    def save_mp4(self, path: str | Path) -> None:
        """Encode the curated H.264/sRGB and stereo AAC output contract.

        Adapted from ComfyUI 1a14b82e VideoFromComponents.save_to (GPL-3.0).
        """
        if (
            self.frames.ndim != 5
            or self.frames.shape[0] != 1
            or self.frames.shape[-1] != 3
        ):
            raise ValueError("H3 output requires one batch of RGB video")
        validate_request(
            self.frames.shape[3],
            self.frames.shape[2],
            self.frames.shape[1],
            0,
            self.frame_rate,
        )
        if self.waveform.ndim != 3 or tuple(self.waveform.shape[:2]) != (1, 2):
            raise ValueError("H3 output requires one stereo waveform")
        if self.sample_rate != 32000:
            raise ValueError("H3 audio requires 32000 Hz")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with av.open(
            str(destination),
            mode="w",
            format="mp4",
            options={"movflags": "use_metadata_tags+faststart"},
        ) as container:
            video = container.add_stream("h264", rate=self.frame_rate)
            video.width, video.height = self.frames.shape[3], self.frames.shape[2]
            video.pix_fmt = "yuv420p"
            video.codec_context.color_primaries = 1
            video.codec_context.color_trc = 13
            video.codec_context.colorspace = 1
            video.codec_context.color_range = 1
            audio = container.add_stream("aac", rate=self.sample_rate, layout="stereo")
            for image in self.frames[0]:
                pixels = (image * 255).clamp(0, 255).byte().cpu().numpy()
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame = frame.reformat(format="yuv420p", dst_colorspace=1)
                frame.color_primaries = 1
                frame.color_trc = 13
                frame.colorspace = 1
                frame.color_range = 1
                container.mux(video.encode(frame))
            container.mux(video.encode(None))
            count = math.ceil(self.sample_rate / self.frame_rate * self.frames.shape[1])
            samples = self.waveform[0, :, :count].float().cpu().contiguous().numpy()
            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
            frame.sample_rate = self.sample_rate
            frame.pts = 0
            container.mux(audio.encode(frame))
            container.mux(audio.encode(None))


class H3Runtime:
    """Own one fixed artifact identity and the latest prompt conditioning."""

    def __init__(self, identity: H3Identity):
        self.identity = identity
        self.device = torch.device("cuda", identity.device_index)
        self.model = self.weights = self.video_vae = self.audio_vae = None
        self.conditioning = None
        self.conditioning_reused = False

    def close(self):
        if self.weights is not None:
            self.weights.close()
        self.weights = self.model = self.video_vae = self.audio_vae = None
        self.conditioning = None
        gc.collect()
        torch.cuda.empty_cache()

    def _release_scratch(self):
        from comfy_aimdo import model_vbar

        torch.cuda.synchronize(self.device)
        if self.weights is not None:
            self.weights.copy_buffers.clear()
        torch.cuda.empty_cache()
        if model_vbar.lib is not None:
            model_vbar.vbars_reset_watermark_limits()

    def _load_transformer(self):
        if self.model is not None:
            return
        with safe_open(self.identity.diffusion, framework="pt", device="cpu") as source:
            names = source.keys()
            if "adaln_t_table" in names:
                grid, dim = source.get_slice("adaln_t_table").get_shape()
                config = {"adaln_curve_grid": grid, "time_embed_dim": dim}
            else:
                config = {}
        self.model = MiniMaxH3Model(**config).eval().requires_grad_(False)
        self.weights = H3Weights(
            Path(self.identity.diffusion),
            self.model,
            self.device,
            self.identity.adapters,
        )

    def _load_video_vae(self):
        if self.video_vae is None:
            self.video_vae = MiniMaxH3VideoVAE().eval().requires_grad_(False)
            self.video_vae.load_state_dict(
                load_file(self.identity.video_vae), assign=True
            )
            self.video_vae.half()

    def _encode_images(self, images, *, reference):
        self._load_video_vae()
        self.video_vae.to(self.device)
        conditions = []
        try:
            for image in images:
                pixels = (image.movedim(-1, 1).unsqueeze(2) * 2.0 - 1.0).to(
                    self.device, dtype=torch.float16
                )
                latent = self.video_vae.encode(pixels).float().cpu()
                conditions.append(
                    {
                        "kind": "image",
                        "latent_h": image.shape[1] // 16,
                        "latent_w": image.shape[2] // 16,
                        "latent": latent,
                    }
                    if reference
                    else {"resolved_frame_index": 0, "latent": latent}
                )
        finally:
            self.video_vae.cpu()
        return conditions

    def _load_audio_vae(self):
        if self.audio_vae is None:
            self.audio_vae = MiniMaxH3AudioVAE().eval().requires_grad_(False)
            self.audio_vae.load_state_dict(
                load_file(self.identity.audio_vae), assign=True
            )

    def _encode_audio(self, waveform):
        # Comfy VAE.vae_encode_crop_pixels center-crops before the H3 codec.
        length = waveform.shape[-1] // 800 * 800
        if not length:
            raise ValueError(
                "H3 reference audio requires at least 800 samples at 32000 Hz"
            )
        offset = waveform.shape[-1] % 800 // 2
        waveform = waveform[..., offset : offset + length]
        self._load_audio_vae()
        self.audio_vae.to(self.device)
        try:
            return self.audio_vae.encode(waveform.to(self.device)).float().cpu()
        finally:
            self.audio_vae.cpu()

    def _encode_video(self, frames):
        self._load_video_vae()
        self.video_vae.to(self.device)
        try:
            pixels = (frames.movedim(-1, 0)[None] * 2.0 - 1.0).half()
            return self.video_vae.encode(pixels, device=self.device).float().cpu()
        finally:
            self.video_vae.cpu()

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        width,
        height,
        frame_count,
        *,
        seed=42,
        fps=24,
        image_path=None,
        reference_image_paths=(),
        reference_image_size="match",
        reference_video_paths=(),
        reference_video_audio_paths=(),
        reference_audio_paths=(),
        progress=None,
    ):
        """Generate the explicitly requested canvas and temporal grid with audio."""
        validate_request(width, height, frame_count, seed, fps)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("H3 prompt must be nonempty text")
        if image_path is not None and (
            reference_image_paths or reference_video_paths or reference_audio_paths
        ):
            raise ValueError(
                "H3 start-image and reference-media operations are separate"
            )
        if len(reference_image_paths) > 9:
            raise ValueError("H3 supports at most nine reference images")
        if reference_image_size not in ("match", "max"):
            raise ValueError("H3 reference image size must be match or max")
        if len(reference_video_paths) > 3 or len(reference_audio_paths) > 3:
            raise ValueError(
                "H3 supports at most three reference videos and three audios"
            )
        if reference_video_audio_paths and len(reference_video_audio_paths) != len(
            reference_video_paths
        ):
            raise ValueError(
                "H3 video soundtracks must pair by index with reference videos"
            )
        soundtracks = reference_video_audio_paths or (None,) * len(
            reference_video_paths
        )
        images = []
        image_identity = None
        if image_path is not None:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                if image.size != (width, height):
                    raise ValueError(
                        f"H3 start image is {image.width}x{image.height}; it must match the {width}x{height} output canvas"
                    )
                images.append(torch.from_numpy(np.array(image)).float()[None] / 255.0)
            image_identity = FileContentIdentity.from_path(image_path)
        reference_identities = []
        for path in reference_image_paths:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                scale = (
                    min(1.0, math.sqrt(width * height / (image.width * image.height)))
                    if reference_image_size == "match"
                    else min(1.0, 2048 / min(image.size))
                )
                reference_size = tuple(
                    max(32, round(side * scale / 32) * 32) for side in image.size
                )
                image = image.resize(reference_size, Image.Resampling.LANCZOS)
                images.append(torch.from_numpy(np.array(image)).float()[None] / 255.0)
            reference_identities.append(FileContentIdentity.from_path(path))
        condition_key = (
            prompt,
            image_identity,
            tuple(reference_identities),
            reference_image_size,
            width,
            height,
            tuple(FileContentIdentity.from_path(p) for p in reference_video_paths),
            tuple(
                FileContentIdentity.from_path(p) if p is not None else None
                for p in soundtracks
            ),
            tuple(FileContentIdentity.from_path(p) for p in reference_audio_paths),
            frame_count if reference_video_paths else None,
        )
        self.conditioning_reused = (
            self.conditioning is not None and self.conditioning[0] == condition_key
        )
        if not self.conditioning_reused:
            refs = []
            presentation = []
            if reference_image_paths:
                report_progress(progress, 0.02, "Encoding reference images")
                refs = self._encode_images(images, reference=True)
                presentation.extend(
                    {"type": "image", "data": image} for image in images
                )
                self._release_scratch()
            for path, audio_path in zip(reference_video_paths, soundtracks):
                report_progress(progress, 0.04, "Encoding reference video")
                frames = load_video(path, frame_count)
                waveform = (
                    load_audio(audio_path, soundtrack=True)
                    if audio_path is not None
                    else None
                )
                if waveform is not None:
                    presentation.append({"type": "audio"})
                sampled = frames[::12]
                presentation.append(
                    {
                        "type": "video",
                        "data": sampled,
                        "timestamps": [i / 2.0 for i in range(len(sampled))],
                    }
                )
                latent = self._encode_video(frames)
                self._release_scratch()
                audio_latent = (
                    self._encode_audio(waveform) if waveform is not None else None
                )
                audio_time = audio_latent.shape[-1] if audio_latent is not None else 0
                refs.append(
                    {
                        "kind": "video_audio" if audio_time else "video",
                        "latent_t": latent.shape[2],
                        "latent_h": frames.shape[1] // 16,
                        "latent_w": frames.shape[2] // 16,
                        "ref_audio_t": audio_time,
                        "latent": latent,
                        "audio_latent": audio_latent,
                    }
                )
                del frames, waveform
                self._release_scratch()
            for path in reference_audio_paths:
                report_progress(progress, 0.06, "Encoding reference audio")
                audio_latent = self._encode_audio(load_audio(path))
                refs.append(
                    {
                        "kind": "audio",
                        "ref_audio_t": audio_latent.shape[-1],
                        "audio_latent": audio_latent,
                    }
                )
                presentation.append({"type": "audio"})
                self._release_scratch()
            report_progress(progress, 0.08 if refs else 0.02, "Text conditioning")
            encoder = H3TextEncoder(
                Path(self.identity.text_encoder),
                Path(self.identity.tokenizer),
                self.device,
                with_vision=bool(images or reference_video_paths),
            )
            try:
                condition, tags = encoder.encode(
                    prompt,
                    images=images if image_path is not None else (),
                    references=presentation,
                )
            finally:
                encoder.close()
            self._release_scratch()
            keyframes = []
            if image_path is not None:
                report_progress(progress, 0.10, "Encoding start image")
                keyframes = self._encode_images(images, reference=False)
                self._release_scratch()
            self.conditioning = (condition_key, condition, tags, keyframes, refs)
        _, condition, tags, keyframes, refs = self.conditioning
        report_progress(progress, 0.12, "Loading transformer")
        self._load_transformer()
        self.weights.activate()
        context = self.model.preprocess_text_embeds(
            condition.to(self.device, dtype=torch.bfloat16)
        )
        _, video_time, audio_time = temporal_shape(frame_count)
        shapes = [
            (1, 24, video_time, height // 16, width // 16),
            (1, 32, 2, audio_time),
        ]
        payload = {
            "seed": seed,
            "audio_scale": 4.0,
            "text_token_tags": tags,
            "keyframes": keyframes,
            "refs": refs,
            "cond_video_latents": [
                frame["latent"] for frame in (*keyframes, *refs) if "latent" in frame
            ],
            "cond_audio_latents": [
                frame["audio_latent"]
                for frame in (*keyframes, *refs)
                if frame.get("audio_latent") is not None
            ],
            "layout": PackedLayout(
                context.shape[1],
                video_time,
                height // 16,
                width // 16,
                audio_time,
                keyframes=keyframes,
                refs=refs,
            ),
        }

        def denoise(latent, sigma, context=context, payload=payload):
            streams = unpack(latent.to(torch.bfloat16), shapes)
            velocity = self.model(
                streams, sigma * 1000.0, context, minimax_payload=payload
            )
            return latent - sigma.reshape(-1, 1, 1) * pack(velocity).float()

        report_progress(progress, 0.15, "Sampling", stage_progress=0.0)
        result = res_multistep(
            denoise,
            noise(shapes, seed).to(self.device),
            simple_schedule(20).to(self.device),
            progress=lambda i, n: report_progress(
                progress, 0.15 + 0.65 * i / n, "Sampling", stage_progress=i / n
            ),
        )
        video, audio = [stream.cpu() for stream in unpack(result, shapes)]
        audio = audio * 0.25
        del result, denoise, context, payload
        self._release_scratch()
        report_progress(progress, 0.82, "Decoding audio")
        self._load_audio_vae()
        self.audio_vae.to(self.device)
        try:
            waveform = self.audio_vae.decode(audio.to(self.device)).cpu()
        finally:
            self.audio_vae.cpu()
        self._release_scratch()
        report_progress(progress, 0.88, "Decoding video")
        self._load_video_vae()
        self.video_vae.to(self.device)
        try:
            frames = self.video_vae.decode(
                video.to(self.device, dtype=torch.float16)
            ).cpu()
        finally:
            self.video_vae.cpu()
        self._release_scratch()
        return H3Output(frames.movedim(1, -1), waveform)
