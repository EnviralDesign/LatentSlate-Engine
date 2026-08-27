"""The standalone LTX 2.3 first/last-frame operation proved by its fixture."""

from __future__ import annotations

import hashlib
import importlib
import math
from contextlib import nullcontext
from dataclasses import dataclass
from io import BytesIO
from itertools import pairwise
from pathlib import Path

import av
import numpy as np
import torch
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout
from PIL import Image
from torch.nn import functional as F

from .audio_vae import Ltx23AudioMelDecoder
from .fp8_linear import Ltx23PlainLinear
from .ops import Ltx23Linear
from .sampling import CANONICAL_AUDIO_SHAPE, nested_noise
from .symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords
from .t2v import Ltx23T2VOutput
from .text_encoder import Ltx23TextEncoder
from .transformer_context import Ltx23TransformerContext
from .video_vae import Ltx23VideoDecoder, Ltx23VideoEncoder
from .vocoder import Ltx23AudioVocoder

_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
_SEED = 315253765879496
_FRAME_RATE = 30
_GUIDE_STRENGTH = 0.7
_VIDEO_SHAPE = (1, 128, 19, 16, 16)
_VAE_SCALE_FACTORS = (8, 32, 32)
_GUIDE_PATCHIFIER = SymmetricPatchifier(1, start_end=True)

class Ltx23FlfOutput(Ltx23T2VOutput):
    """Decoded canonical FLF media with its measured direct-RGB writer."""

    def save_mp4(self, path: str | Path) -> None:
        if tuple(self.frames.shape) != (1, 145, 512, 512, 3):
            raise ValueError("canonical FLF media requires 512x512 RGB frames")
        if tuple(self.waveform.shape[:2]) != (1, 2):
            raise ValueError("canonical FLF media requires one stereo waveform")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(destination), mode="w") as container:
            video = container.add_stream("h264", rate=self.frame_rate)
            video.width = 512
            video.height = 512
            video.pix_fmt = "yuv420p"
            audio = container.add_stream(
                "aac", rate=self.sample_rate, layout="stereo"
            )
            for image in self.frames[0]:
                pixels = (
                    torch.clamp(image.float() * 255, 0, 255)
                    .to(torch.uint8)
                    .numpy()
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


@dataclass(frozen=True)
class Ltx23FlfIdentity:
    """The complete model/recipe identity of the canonical FLF fixture."""

    checkpoint_path: str
    text_checkpoint_path: str
    device_index: int = 0


def _packed_binding(binding, packed, offset):
    def take(name, source):
        start = offset + binding._offsets[name]
        return packed[start : start + source.nbytes].view(source.dtype).view(
            source.shape
        )

    weight = take("weight", binding._weight)
    if isinstance(binding, Ltx23PlainLinear):
        return weight, take("bias", binding._bias)
    scale = take("scale", binding._scale)
    bias = take("bias", binding._bias)
    return (
        QuantizedTensor(
            weight,
            "TensorCoreFP8Layout",
            TensorCoreFP8Layout.Params(
                scale=scale,
                orig_dtype=torch.bfloat16,
                orig_shape=tuple(binding._weight.shape),
            ),
        ),
        bias,
    )


class _Ltx23FlfTransformerContext(Ltx23TransformerContext):
    """Gather each warm FLF block through one packed host-to-device copy."""

    def __init__(self, checkpoint_path: str, device_index: int) -> None:
        super().__init__(checkpoint_path, device_index, block_contiguous=True)
        blocks = [
            [module for module in block.modules() if isinstance(module, Ltx23Linear)]
            for block in self.model.transformer_blocks
        ]
        self._flf_stage = None
        self._flf_pipeline_ready = False
        aimdo_torch = importlib.import_module("comfy_aimdo.torch")
        host_cache = aimdo_torch.hostbuf_to_tensor(self._host_cache)

        for block, modules in zip(self.model.transformer_blocks, blocks, strict=True):
            block_start = modules[0]._latentslate_weight._host_cache_offset
            block_size = sum(
                module._latentslate_weight.allocation_size for module in modules
            )

            def prepare(
                stream=None,
                _buffer=None,
                linears=modules,
                source_start=block_start,
                source_size=block_size,
            ):
                if self._flf_stage is None and all(
                    module._latentslate_weight._host_cache_loaded
                    for group in blocks
                    for module in group
                ):
                    self._vbar.free_memory(1e32)
                    stage_size = max(
                        sum(
                            module._latentslate_weight.allocation_size
                            for module in group
                        )
                        for group in blocks
                    )
                    self._flf_stage = tuple(
                        torch.empty(
                            stage_size,
                            dtype=torch.uint8,
                            device=torch.device("cuda", device_index),
                        )
                        for _ in range(2)
                    )
                    for transformer_block in self.model.transformer_blocks:
                        transformer_block._latentslate_host_buffers = self._flf_stage
                    self._flf_pipeline_ready = True
                if self._flf_stage is None:
                    for module in linears:
                        module._latentslate_prepared = (
                            module._latentslate_weight.materialize(device_index, stream)
                        )
                    return

                with torch.cuda.stream(stream) if stream is not None else nullcontext():
                    stage = _buffer if _buffer is not None else self._flf_stage[0]
                    packed = stage[:source_size]
                    packed.copy_(
                        host_cache[source_start : source_start + source_size],
                        non_blocking=True,
                    )
                    offset = 0
                    for module in linears:
                        binding = module._latentslate_weight
                        binding_size = binding.allocation_size
                        module._latentslate_prepared = _packed_binding(
                            binding, packed, offset
                        )
                        offset += binding_size

            def release(linears=modules):
                for module in linears:
                    module._latentslate_prepared = None
                    module._latentslate_weight.unpin(device_index)

            block._latentslate_prepare = prepare
            block._latentslate_release = release

    def close(self) -> None:
        self._flf_stage = None
        self._flf_pipeline_ready = False
        super().close()


def _resize_center_nearest(image: torch.Tensor, size: int = 512) -> torch.Tensor:
    old_height, old_width = image.shape[-2:]
    old_aspect = old_width / old_height
    new_aspect = 1.0
    x = 0
    y = 0
    if old_aspect > new_aspect:
        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
    cropped = image.narrow(-2, y, old_height - y * 2).narrow(
        -1, x, old_width - x * 2
    )
    return F.interpolate(cropped, size=(size, size), mode="nearest-exact")


def _preprocess_guide(path: str | Path) -> torch.Tensor:
    with Image.open(path) as source:
        pixels = np.asarray(source.convert("RGB")).astype(np.float32) / 255.0
    image = torch.from_numpy(pixels).movedim(-1, 0).unsqueeze(0)
    image = _resize_center_nearest(image).movedim(1, -1)[0]

    image_array = (image * 255.0).byte().numpy()
    with BytesIO() as output_file:
        container = av.open(output_file, "w", format="mp4")
        try:
            stream = container.add_stream(
                "libx264", rate=1, options={"crf": "25", "preset": "veryfast"}
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
    return torch.from_numpy(decoded.astype(np.float32) / 255.0).movedim(-1, 0).unsqueeze(0)


def _guided_video_latent(
    first_frame: torch.Tensor,
    last_frame: torch.Tensor,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    expected = (1, 128, 1, 16, 16)
    if tuple(first_frame.shape) != expected or tuple(last_frame.shape) != expected:
        raise ValueError(f"encoded FLF guides must each have shape {expected}")

    base = torch.zeros(_VIDEO_SHAPE, device=device, dtype=torch.float32)
    first = first_frame.to(device=device, dtype=torch.float32)
    last = last_frame.to(device=device, dtype=torch.float32)
    latent = torch.cat((base, first, last), dim=2)
    mask = torch.ones_like(latent)
    mask[:, :, -2:] = 1.0 - _GUIDE_STRENGTH

    _, first_coords = _GUIDE_PATCHIFIER.patchify(first)
    first_coords = latent_to_pixel_coords(
        first_coords, _VAE_SCALE_FACTORS, causal_fix=True
    )
    last_coords = first_coords.clone()
    last_coords[:, 0] += 144
    keyframe_idxs = torch.cat((first_coords, last_coords), dim=2)
    entries: list[dict[str, object]] = [
        {
            "pre_filter_count": 256,
            "strength": _GUIDE_STRENGTH,
            "pixel_mask": None,
            "latent_shape": (1, 16, 16),
        },
        {
            "pre_filter_count": 256,
            "strength": _GUIDE_STRENGTH,
            "pixel_mask": None,
            "latent_shape": (1, 16, 16),
        },
    ]
    return latent, mask, keyframe_idxs, entries


@torch.inference_mode()
def _sample_guided(
    model: Ltx23TransformerContext,
    condition: torch.Tensor,
    latents: list[torch.Tensor],
    noise: list[torch.Tensor],
    masks: list[torch.Tensor],
    keyframe_idxs: torch.Tensor,
    guide_attention_entries: list[dict[str, object]],
) -> list[torch.Tensor]:
    condition = model.model.preprocess_text_embeds(
        condition.to(dtype=torch.bfloat16), unprocessed=True
    )
    sigma0 = _SIGMAS[0]
    x = [
        sample * sigma0 + latent * (1.0 - sigma0)
        for latent, sample in zip(latents, noise, strict=True)
    ]
    for sigma, sigma_next in pairwise(_SIGMAS):
        sigma_value = float(sigma)
        model_input = [
            stream * mask + latent * (1.0 - mask)
            for stream, mask, latent in zip(x, masks, latents, strict=True)
        ]
        video_timestep = model.model.patchifier.patchify(
            masks[0][:, :1] * sigma_value
        )[0]
        audio_timestep = model.model.a_patchifier.patchify(
            masks[1][:, :1, :, :1] * sigma_value
        )[0]
        flow = model.model(
            [stream.to(dtype=torch.bfloat16) for stream in model_input],
            [video_timestep, audio_timestep],
            condition,
            frame_rate=_FRAME_RATE,
            denoise_mask=masks[0],
            keyframe_idxs=keyframe_idxs,
            guide_attention_entries=guide_attention_entries,
            transformer_options={
                "latentslate_pipeline_prefetch": model._flf_pipeline_ready
            },
        )
        denoised = [
            (source - predicted.float() * sigma_value) * mask
            + latent * (1.0 - mask)
            for source, predicted, mask, latent in zip(
                model_input, flow, masks, latents, strict=True
            )
        ]
        if float(sigma_next) == 0.0:
            x = denoised
        else:
            ratio = float(sigma_next) / sigma_value
            x = [
                ratio * stream + (1.0 - ratio) * clean
                for stream, clean in zip(x, denoised, strict=True)
            ]
    return x


class Ltx23FlfRuntime:
    """Keep one canonical FLF model/recipe identity warm between requests."""

    def __init__(self, identity: Ltx23FlfIdentity) -> None:
        self.identity = identity
        self._transformer: Ltx23TransformerContext | None = None
        self._text_encoder: Ltx23TextEncoder | None = None
        self._video_decoder: Ltx23VideoDecoder | None = None
        self._audio_decoder: Ltx23AudioMelDecoder | None = None
        self._vocoder: Ltx23AudioVocoder | None = None
        self._prompt_cache: tuple[str, torch.Tensor] | None = None
        self._guide_cache: tuple[bytes, bytes, torch.Tensor, torch.Tensor] | None = None

    def replace_identity(self, identity: Ltx23FlfIdentity) -> Ltx23FlfRuntime:
        if identity == self.identity:
            return self
        self.close()
        return Ltx23FlfRuntime(identity)

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
            self._transformer = _Ltx23FlfTransformerContext(
                self.identity.checkpoint_path,
                self.identity.device_index,
            )
        return self._transformer

    def _encode_guides(
        self, first_path: str | Path, last_path: str | Path
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_key = hashlib.sha256(Path(first_path).read_bytes()).digest()
        last_key = hashlib.sha256(Path(last_path).read_bytes()).digest()
        if (
            self._guide_cache is not None
            and self._guide_cache[:2] == (first_key, last_key)
        ):
            return self._guide_cache[2], self._guide_cache[3]

        first = _preprocess_guide(first_path)
        last = _preprocess_guide(last_path)
        encoder = Ltx23VideoEncoder(self.identity.checkpoint_path)
        try:
            first_latent = encoder.encode(first)
            last_latent = encoder.encode(last)
        finally:
            encoder.close()
        self._guide_cache = (first_key, last_key, first_latent, last_latent)
        return first_latent, last_latent

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        first_image_path: str | Path,
        last_image_path: str | Path,
    ) -> Ltx23FlfOutput:
        """Execute the canonical 512px, CFG=1, single-stage FLF gate."""
        condition = self._encode_prompt(prompt)
        first, last = self._encode_guides(first_image_path, last_image_path)
        transformer = self._transformer_context()
        device = transformer.device_index

        video, video_mask, keyframe_idxs, entries = _guided_video_latent(
            first, last, device
        )
        audio = torch.zeros(CANONICAL_AUDIO_SHAPE, device=device, dtype=torch.float32)
        latents = [video, audio]
        masks = [video_mask, torch.ones_like(audio)]
        sampled = _sample_guided(
            transformer,
            condition,
            latents,
            nested_noise(_SEED, latents),
            masks,
            keyframe_idxs,
            entries,
        )
        sampled[0] = sampled[0][:, :, :-2]
        del latents, masks, video, audio, video_mask, condition, first, last

        if self._video_decoder is None:
            self._video_decoder = Ltx23VideoDecoder(self.identity.checkpoint_path)
        frames = self._video_decoder.decode(sampled[0]).movedim(1, -1).cpu()
        if self._audio_decoder is None:
            self._audio_decoder = Ltx23AudioMelDecoder(self.identity.checkpoint_path)
        mel = self._audio_decoder.decode(sampled[1]).transpose(2, 3)
        del sampled
        if self._vocoder is None:
            self._vocoder = Ltx23AudioVocoder(self.identity.checkpoint_path)
        waveform = self._vocoder.decode(mel).cpu()
        return Ltx23FlfOutput(frames=frames, waveform=waveform)

    def close(self) -> None:
        self._prompt_cache = None
        self._guide_cache = None
        if self._text_encoder is not None:
            self._text_encoder.close()
            self._text_encoder = None
        if self._transformer is not None:
            self._transformer.close()
            self._transformer = None
        if self._video_decoder is not None:
            self._video_decoder.close()
            self._video_decoder = None
        if self._audio_decoder is not None:
            self._audio_decoder.close()
            self._audio_decoder = None
        if self._vocoder is not None:
            self._vocoder.close()
            self._vocoder = None
