"""The standalone LTX 2.3 first/last-frame operation proved by its fixture."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from contextlib import nullcontext
from io import BytesIO
from itertools import pairwise
from pathlib import Path

import av
import numpy as np
import torch
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout
from PIL import Image

from latentslate_engine.identity import FileContentIdentity
from latentslate_engine.progress import ProgressCallback, report_progress

from .audio_vae import Ltx23AudioMelDecoder
from .contracts import Ltx23FlfIdentity
from .fp8_linear import Ltx23PlainLinear
from .ops import Ltx23Linear
from .sampling import (
    FRAME_RATE,
    empty_av_latents,
    ltx_temporal_shapes,
    nested_noise,
    validate_ltx_request,
)
from .symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords
from .t2v import Ltx23T2VOutput
from .text_encoder import Ltx23TextEncoder
from .transformer_context import Ltx23TransformerContext
from .video_vae import Ltx23VideoDecoder, Ltx23VideoEncoder
from .vocoder import Ltx23AudioVocoder

_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
_CANONICAL_SEED = 315253765879496
_GUIDE_STRENGTH = 0.7
_VAE_SCALE_FACTORS = (8, 32, 32)
_GUIDE_PATCHIFIER = SymmetricPatchifier(1, start_end=True)


class Ltx23FlfOutput(Ltx23T2VOutput):
    """Decoded FLF media using the measured direct-RGB writer."""

    def save_mp4(self, path: str | Path) -> None:
        super().save_mp4(path)


def _packed_binding(binding, packed, offset):
    def take(name, source):
        start = offset + binding._offsets[name]
        return (
            packed[start : start + source.nbytes].view(source.dtype).view(source.shape)
        )

    weight = take("weight", binding._weight)
    if isinstance(binding, Ltx23PlainLinear):
        return weight, take("bias", binding._bias), None
    scale = take("scale", binding._scale)
    bias = take("bias", binding._bias)
    input_scale = (
        take("input_scale", binding._input_scale)
        if binding._input_scale is not None
        else None
    )
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
        input_scale,
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


def _preprocess_guide(path: str | Path, width: int, height: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.size != (width, height):
        raise ValueError(f"normalized LTX FLF guide image must be {width}x{height}")
    image = image.resize((width, height), Image.Resampling.NEAREST)
    image = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0)

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
    return (
        torch.from_numpy(decoded.astype(np.float32) / 255.0).movedim(-1, 0).unsqueeze(0)
    )


def _guided_video_latent(
    first_frame: torch.Tensor,
    last_frame: torch.Tensor,
    width: int,
    height: int,
    video_frames: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    latent_height = height // 32
    latent_width = width // 32
    expected = (1, 128, 1, latent_height, latent_width)
    if tuple(first_frame.shape) != expected or tuple(last_frame.shape) != expected:
        raise ValueError(f"encoded FLF guides must each have shape {expected}")

    base = torch.zeros(
        (1, 128, video_frames, latent_height, latent_width),
        device=device,
        dtype=torch.float32,
    )
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
    last_coords[:, 0] += video_frames * 8 - 8
    keyframe_idxs = torch.cat((first_coords, last_coords), dim=2)
    guide_tokens = latent_height * latent_width
    entries: list[dict[str, object]] = [
        {
            "pre_filter_count": guide_tokens,
            "strength": _GUIDE_STRENGTH,
            "pixel_mask": None,
            "latent_shape": (1, latent_height, latent_width),
        },
        {
            "pre_filter_count": guide_tokens,
            "strength": _GUIDE_STRENGTH,
            "pixel_mask": None,
            "latent_shape": (1, latent_height, latent_width),
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
    step_callback: Callable[[int, int], None] | None = None,
) -> list[torch.Tensor]:
    condition = model.model.preprocess_text_embeds(
        condition.to(dtype=torch.bfloat16), unprocessed=True
    )
    sigma_values = torch.as_tensor(
        _SIGMAS, device=latents[0].device, dtype=torch.float32
    )
    shapes = [stream.shape for stream in latents]
    sizes = [stream[0].numel() for stream in latents]
    packed_latents = torch.cat(
        [stream.reshape(stream.shape[0], 1, -1) for stream in latents], dim=-1
    )
    packed_noise = torch.cat(
        [stream.reshape(stream.shape[0], 1, -1) for stream in noise], dim=-1
    )
    packed_masks = torch.cat(
        [stream.reshape(stream.shape[0], 1, -1) for stream in masks], dim=-1
    )
    sigma0 = sigma_values[0]
    packed_x = packed_noise * sigma0 + packed_latents * (1.0 - sigma0)
    step_count = len(sigma_values) - 1
    for index, (sigma_value, sigma_next) in enumerate(pairwise(sigma_values), start=1):
        packed_model_input = packed_x * packed_masks + packed_latents * (
            1.0 - packed_masks
        )
        model_input = [
            stream.view(shape)
            for stream, shape in zip(
                packed_model_input.split(sizes, dim=-1), shapes, strict=True
            )
        ]
        video_timestep = model.model.patchifier.patchify(
            masks[0][:, :1].to(torch.bfloat16).float() * sigma_value
        )[0]
        audio_timestep = model.model.a_patchifier.patchify(
            masks[1][:, :1, :, :1].to(torch.bfloat16).float() * sigma_value
        )[0]
        flow = model.model(
            [stream.to(dtype=torch.bfloat16) for stream in model_input],
            [video_timestep, audio_timestep],
            condition,
            frame_rate=FRAME_RATE,
            denoise_mask=masks[0],
            keyframe_idxs=keyframe_idxs,
            guide_attention_entries=guide_attention_entries,
            transformer_options={
                "latentslate_pipeline_prefetch": model._flf_pipeline_ready
            },
        )
        packed_flow = torch.cat(
            [
                predicted.float().reshape(predicted.shape[0], 1, -1)
                for predicted in flow
            ],
            dim=-1,
        )
        packed_denoised = (
            packed_model_input - packed_flow * sigma_value
        ) * packed_masks + packed_latents * (1.0 - packed_masks)
        if sigma_next == 0:
            packed_x = packed_denoised
        else:
            sigma_down_ratio = sigma_next / sigma_value
            packed_x = (
                sigma_down_ratio * packed_x + (1.0 - sigma_down_ratio) * packed_denoised
            )
        if step_callback is not None:
            step_callback(index, step_count)
    return [
        stream.view(shape)
        for stream, shape in zip(packed_x.split(sizes, dim=-1), shapes, strict=True)
    ]


class Ltx23FlfRuntime:
    """Keep one FLF model/recipe identity warm between requests."""

    def __init__(self, identity: Ltx23FlfIdentity) -> None:
        self.identity = identity
        self._transformer: Ltx23TransformerContext | None = None
        self._text_encoder: Ltx23TextEncoder | None = None
        self._video_decoder: Ltx23VideoDecoder | None = None
        self._audio_decoder: Ltx23AudioMelDecoder | None = None
        self._vocoder: Ltx23AudioVocoder | None = None
        self._prompt_cache: tuple[str, torch.Tensor] | None = None
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
        self,
        first_path: str | Path,
        last_path: str | Path,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_key = FileContentIdentity.from_path(first_path)
        last_key = FileContentIdentity.from_path(last_path)
        if self._guide_cache is not None and self._guide_cache[:4] == (
            first_key,
            last_key,
            width,
            height,
        ):
            return self._guide_cache[4], self._guide_cache[5]

        first = _preprocess_guide(first_path, width, height)
        last = _preprocess_guide(last_path, width, height)
        encoder = Ltx23VideoEncoder(self.identity.checkpoint_path)
        try:
            first_latent = encoder.encode(first)
            last_latent = encoder.encode(last)
        finally:
            encoder.close()
        self._guide_cache = (
            first_key,
            last_key,
            width,
            height,
            first_latent,
            last_latent,
        )
        return first_latent, last_latent

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        first_image_path: str | Path,
        last_image_path: str | Path,
        width: int = 1280,
        height: int = 704,
        duration_seconds: float = 5.0,
        seed: int = _CANONICAL_SEED,
        progress: ProgressCallback | None = None,
    ) -> Ltx23FlfOutput:
        """Execute the concrete CFG=1, single-stage LTX 2.3 FLF operation."""
        validate_ltx_request(width, height, duration_seconds, seed, alignment=32)
        report_progress(progress, 0.03, "Endpoint conditioning")
        first, last = self._encode_guides(
            first_image_path, last_image_path, width, height
        )
        report_progress(progress, 0.12, "Text conditioning")
        condition = self._encode_prompt(prompt)
        report_progress(progress, 0.2, "Loading transformer")
        transformer = self._transformer_context()
        device = transformer.device_index
        _, video_frames, _, _ = ltx_temporal_shapes(duration_seconds)

        video, video_mask, keyframe_idxs, entries = _guided_video_latent(
            first, last, width, height, video_frames, device
        )
        audio = empty_av_latents(
            width,
            height,
            duration_seconds,
            spatial_divisor=32,
            device=device,
        )[1]
        latents = [video, audio]
        masks = [video_mask, torch.ones_like(audio)]
        report_progress(progress, 0.3, "Guided sampling", stage_progress=0.0)
        sampled = _sample_guided(
            transformer,
            condition,
            latents,
            nested_noise(seed, latents),
            masks,
            keyframe_idxs,
            entries,
            lambda index, count: report_progress(
                progress,
                0.3 + 0.45 * index / count,
                "Guided sampling",
                stage_progress=index / count,
                detail=f"Step {index} of {count}",
            ),
        )
        sampled[0] = sampled[0][:, :, :-2]
        del latents, masks, video, audio, video_mask, condition, first, last

        report_progress(progress, 0.78, "Video decode")
        if self._video_decoder is None:
            self._video_decoder = Ltx23VideoDecoder(self.identity.checkpoint_path)
        frames = self._video_decoder.decode(sampled[0]).movedim(1, -1).cpu()
        report_progress(progress, 0.85, "Audio decode")
        if self._audio_decoder is None:
            self._audio_decoder = Ltx23AudioMelDecoder(self.identity.checkpoint_path)
        mel = self._audio_decoder.decode(sampled[1]).transpose(2, 3)
        del sampled
        report_progress(progress, 0.9, "Audio synthesis")
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
