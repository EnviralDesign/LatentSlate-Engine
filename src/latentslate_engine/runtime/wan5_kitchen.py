"""Engine-owned Wan 2.2 TI2V 5B execution with direct Kitchen text kernels."""

from __future__ import annotations

import gc
import hashlib
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..wan5_kitchen_recipe import (
    WAN5_FPS,
    WAN5_GUIDANCE_SCALE,
    WAN5_MAX_SEQUENCE_LENGTH,
    WAN5_NEGATIVE_PROMPT,
    WAN5_STEPS,
    Wan5KitchenRuntimeRequest,
    revalidate_wan5_kitchen_runtime_request,
)
from .umt5_stored_adapter import (
    UMT5_XXL_CONFIG,
    UMT5EncoderResidencySession,
    materialize_umt5_encoder,
)
from .wan5_kitchen_contracts import materialize_wan5_stored_artifact
from .wan22_prompt import WanSentencePieceTokenizer, encode_wan_prompt_pair
from .wan22_stored_adapter import NativeStoredLinear

WAN5_MIN_FRAMES = 25
WAN5_MAX_FRAMES = 121
WAN5_MAX_PIXELS = 901_120
WAN5_ALIGNMENT = 32

Wan5Progress = Callable[[float, str | None], None]
Wan5Cancellation = Callable[[], None]


@dataclass(frozen=True, slots=True)
class Wan5KitchenGeneration:
    operation: str
    prompt: str
    width: int
    height: int
    num_frames: int
    seed: int
    output_path: Path
    staging_output_path: Path
    start_image_path: Path | None = None
    start_image_identity: dict[str, int | str] | None = None


@dataclass(frozen=True, slots=True)
class Wan5KitchenResult:
    output_path: Path
    metadata: dict[str, Any]


class Wan5SimpleUniPCScheduler:
    """Construct the workflow-derived 30-step simple-sigma UniPC/bh1 scheduler."""

    @staticmethod
    def build() -> Any:
        from diffusers import UniPCMultistepScheduler

        class _PinnedUniPC(UniPCMultistepScheduler):
            def set_timesteps(
                self,
                num_inference_steps: int | None = None,
                device: str | torch.device | None = None,
                **kwargs: Any,
            ) -> None:
                if kwargs or num_inference_steps != WAN5_STEPS:
                    raise ValueError("Wan 5B requires the pinned 30-step simple sigma schedule")
                import numpy as np

                # The pinned source asks simple_scheduler for 31 steps because
                # UniPC discards its penultimate sigma.  Select the first 30
                # nonzero values from that 31-step grid; Diffusers applies the
                # fixed flow shift and appends terminal zero itself.
                sigmas = np.asarray(
                    [
                        (1000 - int(index * 1000 / (WAN5_STEPS + 1))) / 1000
                        for index in range(WAN5_STEPS)
                    ],
                    dtype=np.float32,
                )
                super().set_timesteps(sigmas=sigmas, device=device)

        return _PinnedUniPC(
            num_train_timesteps=1000,
            solver_order=3,
            prediction_type="flow_prediction",
            solver_type="bh1",
            lower_order_final=True,
            final_sigmas_type="zero",
            flow_shift=8.0,
            use_flow_sigmas=True,
        )


class Wan5KitchenRuntime:
    """One-shot runtime instantiated only inside an Engine disposable worker."""

    def __init__(self, request: Wan5KitchenRuntimeRequest, *, device: str = "cuda") -> None:
        if not revalidate_wan5_kitchen_runtime_request(request):
            raise ValueError("Wan 5B Kitchen request failed exact revalidation")
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Wan 5B Kitchen execution requires CUDA")
        self.request = request
        self.device = torch.device(device)

    def generate(
        self,
        generation: Wan5KitchenGeneration,
        *,
        progress: Wan5Progress,
        check_cancelled: Wan5Cancellation,
    ) -> Wan5KitchenResult:
        validate_wan5_kitchen_generation(generation, self.request.operation)
        check_cancelled()
        output = generation.output_path.resolve(strict=False)
        temporary = generation.staging_output_path.resolve(strict=False)
        transformer = vae = text_encoder = pipe = None
        try:
            progress(0.01, "Materializing Wan 2.2 prompt encoder")
            text_encoder = materialize_umt5_encoder(
                self.request.plans["text_encoder"],
                UMT5_XXL_CONFIG,
                compute_dtype=torch.float16,
            )
            tokenizer = WanSentencePieceTokenizer.from_file(
                self.request.plans["pipeline_support"].root / "tokenizer" / "spiece.model"
            )
            tokens = tokenizer.tokenize_pair(
                generation.prompt,
                WAN5_NEGATIVE_PROMPT,
                sequence_length=WAN5_MAX_SEQUENCE_LENGTH,
            )
            _reset_text_dispatch(text_encoder)
            check_cancelled()
            with UMT5EncoderResidencySession(text_encoder, onload_device=self.device) as session:
                conditioning = encode_wan_prompt_pair(session, tokens)
            text_dispatch = _prove_text_dispatch(text_encoder, self.request)
            check_cancelled()
            _empty_cuda()

            progress(0.08, "Materializing Wan 2.2 transformer and VAE")
            transformer = materialize_wan5_stored_artifact(
                self.request.plans["transformer"], compute_dtype=torch.float16
            )
            vae = materialize_wan5_stored_artifact(
                self.request.plans["vae"], compute_dtype=torch.float16
            )
            check_cancelled()
            transformer.to(device=self.device, dtype=torch.float16)
            if generation.operation == "wan5_i2v":
                vae.to(device=self.device, dtype=torch.float16)

            scheduler = Wan5SimpleUniPCScheduler.build()
            pipe = _build_pipeline(generation.operation, transformer, vae, scheduler)
            prompt_embeds = conditioning.prompt_embeds.to(self.device, dtype=torch.float16)
            negative_embeds = conditioning.negative_prompt_embeds.to(
                self.device, dtype=torch.float16
            )
            image = _load_image(generation.start_image_path, generation.start_image_identity)
            transition = _Wan5ResidencyTransition(
                transformer,
                vae,
                operation=generation.operation,
                device=self.device,
            )
            transition.attach()
            generator = torch.Generator(device="cpu").manual_seed(generation.seed)

            def callback_on_step_end(
                _pipe: Any,
                step_index: int,
                _timestep: Any,
                callback_kwargs: dict[str, Any],
            ) -> dict[str, Any]:
                check_cancelled()
                progress(
                    0.12 + 0.76 * ((step_index + 1) / WAN5_STEPS),
                    f"Generating video ({step_index + 1}/{WAN5_STEPS})",
                )
                if step_index == WAN5_STEPS - 1:
                    transition.prepare_decode()
                return callback_kwargs

            progress(0.10, "Generating Wan 2.2 video")
            kwargs = {
                "prompt": None,
                "negative_prompt": None,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_embeds,
                "width": generation.width,
                "height": generation.height,
                "num_frames": generation.num_frames,
                "num_inference_steps": WAN5_STEPS,
                "guidance_scale": WAN5_GUIDANCE_SCALE,
                "generator": generator,
                "output_type": "np",
                "callback_on_step_end": callback_on_step_end,
            }
            if image is None:
                frames = pipe(**kwargs).frames[0]
            else:
                frames = pipe(image=image, **kwargs).frames[0]
            transition.assert_decode_complete()
            check_cancelled()

            progress(0.93, "Encoding Wan 2.2 MP4")
            from diffusers.utils import export_to_video

            export_to_video(frames, str(temporary), fps=WAN5_FPS)
            check_cancelled()
            observed = _validate_mp4(temporary, generation)
            check_cancelled()
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise FileExistsError(f"Wan 5B output already exists: {output}")
            os.replace(temporary, output)
            metadata = {
                "family": "wan22",
                "runtime": "engine-native/wan5-kitchen",
                "operation": generation.operation,
                "request_fingerprint": self.request.fingerprint,
                "component_fingerprint": self.request.component_fingerprint,
                "components": self.request.public_component_manifest(),
                "width": generation.width,
                "height": generation.height,
                "frame_count": generation.num_frames,
                "fps": WAN5_FPS,
                "duration_seconds": generation.num_frames / WAN5_FPS,
                "seed": generation.seed,
                "sampling": {
                    "steps": WAN5_STEPS,
                    "source_schedule_steps": WAN5_STEPS + 1,
                    "discard_penultimate_sigma": True,
                    "guidance_scale": WAN5_GUIDANCE_SCALE,
                    "sampler": "uni_pc/bh1",
                    "sampler_runtime": "diffusers/UniPCMultistepScheduler",
                    "scheduler": "simple",
                    "flow_shift": 8.0,
                },
                "conditioning": {
                    "mode": "text" if image is None else "first_frame",
                    "negative_prompt_sha256": hashlib.sha256(
                        WAN5_NEGATIVE_PROMPT.encode()
                    ).hexdigest(),
                    "tokenizer_sha256": conditioning.tokenizer_sha256,
                },
                "kitchen_dispatch": text_dispatch,
                "residency": transition.provenance(),
                "output": observed,
                "output_sha256": _sha256_file(output),
            }
            progress(1.0, "Complete")
            return Wan5KitchenResult(output.resolve(strict=True), metadata)
        except BaseException:
            temporary.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
            raise
        finally:
            pipe = transformer = vae = text_encoder = None
            gc.collect()
            try:
                _empty_cuda()
            except BaseException:  # noqa: BLE001, S110 - worker exit owns final reclamation
                pass


class _Wan5ResidencyTransition:
    def __init__(self, transformer: Any, vae: Any, *, operation: str, device: torch.device) -> None:
        self.transformer = transformer
        self.vae = vae
        self.operation = operation
        self.device = device
        self._hook: Any | None = None
        self._guide_encoded = operation == "wan5_t2v"
        self._decode_prepared = False

    def attach(self) -> None:
        if self.operation == "wan5_i2v":
            self._hook = self.transformer.register_forward_pre_hook(
                self._before_first_transformer, prepend=True
            )

    def _before_first_transformer(self, _module: Any, _args: Any) -> None:
        if self._guide_encoded:
            return
        self.vae.to("cpu")
        _empty_cuda()
        self._guide_encoded = True
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def prepare_decode(self) -> None:
        if not self._guide_encoded:
            raise RuntimeError("Wan 5B guide VAE stage did not complete before denoising")
        self.transformer.to("cpu")
        _empty_cuda()
        self.vae.to(device=self.device, dtype=torch.float16)
        self._decode_prepared = True

    def assert_decode_complete(self) -> None:
        if not self._decode_prepared:
            raise RuntimeError("Wan 5B final callback did not establish decode residency")

    def provenance(self) -> dict[str, object]:
        return {
            "text_then_transformer_then_vae": True,
            "guide_vae_precedes_transformer": self.operation == "wan5_i2v",
            "transformer_offloaded_before_decode": self._decode_prepared,
            "parent_tensor_residency": False,
        }


def validate_wan5_kitchen_generation(
    generation: Wan5KitchenGeneration, expected_operation: str
) -> None:
    if generation.operation != expected_operation or generation.operation not in {
        "wan5_t2v",
        "wan5_i2v",
    }:
        raise ValueError("Wan 5B generation operation does not match its recipe")
    if not isinstance(generation.prompt, str) or not generation.prompt.strip():
        raise ValueError("Wan 5B prompt must be nonempty")
    integers = (generation.width, generation.height, generation.num_frames, generation.seed)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise TypeError("Wan 5B dimensions, frames, and seed must be integers")
    if (
        generation.width <= 0
        or generation.height <= 0
        or generation.width % WAN5_ALIGNMENT
        or generation.height % WAN5_ALIGNMENT
        or generation.width * generation.height > WAN5_MAX_PIXELS
    ):
        raise ValueError("Wan 5B dimensions must be /32 and within the accepted pixel area")
    if (
        not WAN5_MIN_FRAMES <= generation.num_frames <= WAN5_MAX_FRAMES
        or generation.num_frames % 4 != 1
    ):
        raise ValueError("Wan 5B frame count must be 25..121 and satisfy 4k+1")
    if generation.seed < 0:
        raise ValueError("Wan 5B seed must be nonnegative")
    if generation.output_path.suffix.lower() != ".mp4" or generation.output_path.exists():
        raise ValueError("Wan 5B output must be a fresh MP4 path")
    output = generation.output_path.resolve(strict=False)
    staging = generation.staging_output_path.resolve(strict=False)
    if (
        staging.suffix.lower() != ".mp4"
        or staging.exists()
        or staging.parent != output.parent
        or staging == output
    ):
        raise ValueError("Wan 5B staging output must be a fresh sibling MP4 path")
    if generation.operation == "wan5_t2v":
        if generation.start_image_path is not None or generation.start_image_identity is not None:
            raise ValueError("Wan 5B T2V does not accept a guide image")
    elif (
        generation.start_image_path is None
        or not generation.start_image_path.is_file()
        or not isinstance(generation.start_image_identity, dict)
    ):
        raise ValueError("Wan 5B I2V requires an existing first-frame image")


def _build_pipeline(operation: str, transformer: Any, vae: Any, scheduler: Any) -> Any:
    if operation == "wan5_t2v":
        from diffusers import WanPipeline

        return WanPipeline(
            tokenizer=None,
            text_encoder=None,
            vae=vae,
            scheduler=scheduler,
            transformer=transformer,
            transformer_2=None,
            boundary_ratio=None,
            expand_timesteps=True,
        )
    from diffusers import WanImageToVideoPipeline

    return WanImageToVideoPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        scheduler=scheduler,
        image_processor=None,
        image_encoder=None,
        transformer=transformer,
        transformer_2=None,
        boundary_ratio=None,
        expand_timesteps=True,
    )


def _load_image(path: Path | None, expected_identity: dict[str, int | str] | None) -> Any | None:
    if path is None:
        if expected_identity is not None:
            raise ValueError("Wan 5B image identity exists without an image")
        return None
    if expected_identity is None or _endpoint_identity(path) != expected_identity:
        raise ValueError("Wan 5B guide image changed immediately before decode")
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _endpoint_identity(path: Path) -> dict[str, int | str]:
    candidate = path.resolve(strict=True)
    if not candidate.is_file():
        raise ValueError("Wan 5B guide image is not a file")
    before = candidate.stat()
    digest = _sha256_file(candidate)
    after = candidate.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Wan 5B guide image changed while measuring identity")
    return {
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
    }


def _reset_text_dispatch(model: Any) -> None:
    for module in model.modules():
        if isinstance(module, NativeStoredLinear):
            module.native_dispatch_count = 0
            module.fallback_dispatch_count = 0


def _prove_text_dispatch(model: Any, request: Wan5KitchenRuntimeRequest) -> dict[str, object]:
    modules = [module for module in model.modules() if isinstance(module, NativeStoredLinear)]
    expected = len(request.plans["text_encoder"].quant_sources)
    native_modules = sum(module.native_dispatch_count > 0 for module in modules)
    native_calls = sum(module.native_dispatch_count for module in modules)
    fallback_calls = sum(module.fallback_dispatch_count for module in modules)
    if len(modules) != expected or native_modules != expected or fallback_calls != 0:
        raise RuntimeError("Wan 5B text encoder did not prove complete native Kitchen dispatch")
    return {
        "backend": "comfy_kitchen/cuda",
        "expected_modules": expected,
        "native_modules": native_modules,
        "native_calls": native_calls,
        "fallback_calls": fallback_calls,
        "proven": True,
    }


def _validate_mp4(path: Path, generation: Wan5KitchenGeneration) -> dict[str, object]:
    import av

    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("Wan 5B encoder produced no MP4")
    with av.open(str(path)) as container:
        format_name = str(container.format.name)
        streams = list(container.streams.video)
        if len(streams) != 1 or container.streams.audio:
            raise RuntimeError("Wan 5B output must contain exactly one video stream and no audio")
        stream = streams[0]
        decoded = sum(1 for _frame in container.decode(stream))
        rate = float(stream.average_rate) if stream.average_rate is not None else 0.0
        codec = stream.codec_context.name
        width, height = stream.codec_context.width, stream.codec_context.height
    if (
        "mp4" not in format_name.split(",")
        or decoded != generation.num_frames
        or not math.isclose(rate, WAN5_FPS, abs_tol=0.01)
        or codec != "h264"
        or (width, height) != (generation.width, generation.height)
    ):
        raise RuntimeError("Wan 5B MP4 stream facts differ from the requested contract")
    return {
        "container": format_name,
        "codec": codec,
        "width": width,
        "height": height,
        "fps": rate,
        "frame_count": decoded,
        "has_audio": False,
        "size_bytes": path.stat().st_size,
    }


def _empty_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
