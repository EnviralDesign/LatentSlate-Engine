"""Engine-native LTX 2.3 Kitchen runtime.

Diffusers owns the LTX pipeline and model forwards.  This module owns the
operation topology, explicit component residency, cancellation, provenance,
and final A/V mux. It converts no base model at runtime; stored quantized
linears dispatch directly through Comfy Kitchen as materialized by
:mod:`ltx23_av_stored_adapter`.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch import nn

from ..ltx23_kitchen_recipe import (
    LTX23_FPS,
    LTX23_GUIDANCE_SCALE,
    LTX23_GUIDE_STRENGTH,
    LTX23_MAIN_SIGMAS,
    LTX23_MODEL_LORA_STRENGTH,
    LTX23_REFINE_SIGMAS,
    LTX23_TEXT_LORA_STRENGTH,
    LTX23KitchenRuntimeRequest,
    revalidate_ltx23_kitchen_runtime_request,
)
from .ltx23_av_stored_adapter import (
    LTX23ModuleBinding,
    LTX23StoredFP8Linear,
    build_ltx23_av_meta_shell,
    build_ltx23_connector_meta_shell,
    capture_ltx23_module_storage,
    inspect_ltx23_av_artifact,
    inspect_ltx23_model_lora,
    install_ltx23_model_lora,
    ltx23_model_lora_dispatch_evidence,
    ltx23_module_physical_bytes,
    materialize_ltx23_av,
    materialize_ltx23_connectors,
    plan_ltx23_av_materialization,
    plan_ltx23_connector_materialization,
)
from .ltx23_kitchen_media import (
    build_ltx23_media_shell,
    materialize_ltx23_media_component,
    plan_ltx23_media_component,
)
from .ltx23_kitchen_text import (
    LTX23GemmaMixedTextStage,
    install_ltx23_gemma_text_lora,
    load_ltx23_gemma_mixed_text_encoder,
    plan_ltx23_gemma_mixed_text_encoder,
    plan_ltx23_gemma_text_lora,
)
from .residency_policy import ResidencyDecision, choose_cuda_residency

LTX23_AUDIO_SAMPLE_RATE = 48_000
LTX23_AUDIO_CHANNELS = 2
LTX23_DEV_NEGATIVE_PROMPT = "pc game, console game, video game, cartoon, childish, ugly"
LTX23_FLF_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted "
    "proportions, unnatural skin tones, deformed facial features, asymmetrical face, "
    "missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts "
    "around text, unreadable text on shirt or hat, incorrect lettering on cap (\u201cPNTR\u201d), "
    "incorrect t-shirt slogan (\u201cJUST DO IT\u201d), missing microphone, misplaced microphone, "
    "inconsistent perspective, camera shake, incorrect depth of field, background too sharp, "
    "background clutter, distracting reflections, harsh shadows, inconsistent lighting "
    "direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, "
    "uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, "
    "smiling, laughing, exaggerated sadness, wrong gaze direction, eyes looking at camera, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, "
    "background noise, off-sync audio, missing sniff sounds, incorrect dialogue, added "
    "dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, "
    "unnatural transitions, inconsistent framing, tilted camera, missing door or shelves, "
    "missing shallow depth of field, flat lighting, inconsistent tone, cinematic "
    "oversaturation, stylized filters, or AI artifacts."
)
LTX23_PROMPT_ENHANCEMENT_SEED = 0
LTX23_PROMPT_MAX_NEW_TOKENS = 2_048
LTX23_REFINE_SEED = 42
_LTX23_STAGE_MINIMUM_HEADROOM_BYTES = 2 * 1024**3
LTX23_PROMPT_GENERATION_SETTINGS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 64,
    "top_p": 0.95,
    "min_p": 0.05,
    "repetition_penalty": 1.05,
}

LTX23KitchenProgress = Callable[[float, str | None], None]
LTX23KitchenCancellation = Callable[[], None]
_PROCESS_OWNERSHIP = threading.Lock()


@dataclass(frozen=True, slots=True)
class LTX23KitchenGeneration:
    """One exact, already-resolved LTX generation invocation."""

    prompt: str
    output_path: Path
    width: int
    height: int
    num_frames: int
    seed: int
    start_image_path: Path | None = None
    end_image_path: Path | None = None
    start_image_identity: Mapping[str, object] | None = None
    end_image_identity: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class LTX23KitchenResult:
    """Published output plus identity-bound native execution evidence."""

    output_path: Path
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LTX23KitchenOperationSpec:
    operation: str
    stages: tuple[str, ...]
    main_sigmas: tuple[float, ...]
    refine_sigmas: tuple[float, ...] | None
    prompt_enhancement: bool
    model_lora_strength: float | None
    text_lora_strength: float | None
    guide_strengths: tuple[float, ...]
    fps: int = LTX23_FPS
    audio_sample_rate: int = LTX23_AUDIO_SAMPLE_RATE
    audio_channels: int = LTX23_AUDIO_CHANNELS


def ltx23_kitchen_operation_spec(operation: str) -> LTX23KitchenOperationSpec:
    """Return the fixed operation topology without loading a tensor payload."""

    if operation == "ltx23_dev_t2v":
        return LTX23KitchenOperationSpec(
            operation,
            ("prompt_enhance", "text", "main", "x2", "refine", "decode", "mux"),
            LTX23_MAIN_SIGMAS,
            LTX23_REFINE_SIGMAS,
            True,
            LTX23_MODEL_LORA_STRENGTH,
            LTX23_TEXT_LORA_STRENGTH,
            (),
        )
    if operation == "ltx23_dev_i2v":
        return LTX23KitchenOperationSpec(
            operation,
            ("text", "guide_half", "main", "x2", "guide_full", "refine", "decode", "mux"),
            LTX23_MAIN_SIGMAS,
            LTX23_REFINE_SIGMAS,
            False,
            LTX23_MODEL_LORA_STRENGTH,
            LTX23_TEXT_LORA_STRENGTH,
            (LTX23_GUIDE_STRENGTH, 1.0),
        )
    if operation == "ltx23_distilled_flf":
        return LTX23KitchenOperationSpec(
            operation,
            ("text", "guide_first", "guide_last", "main", "decode", "mux"),
            LTX23_MAIN_SIGMAS,
            None,
            False,
            None,
            None,
            (LTX23_GUIDE_STRENGTH, LTX23_GUIDE_STRENGTH),
        )
    raise ValueError(f"unsupported LTX 2.3 Kitchen operation {operation!r}")


def validate_ltx23_kitchen_generation(operation: str, generation: LTX23KitchenGeneration) -> None:
    """Fail before loading models if invocation geometry or guides are invalid."""

    ltx23_kitchen_operation_spec(operation)
    if not isinstance(generation.prompt, str) or not generation.prompt.strip():
        raise ValueError("LTX 2.3 prompt must be nonempty")
    if generation.width <= 0 or generation.height <= 0:
        raise ValueError("LTX 2.3 dimensions must be positive")
    divisor = 64 if operation.startswith("ltx23_dev_") else 32
    if generation.width % divisor or generation.height % divisor:
        raise ValueError(f"LTX 2.3 {operation} dimensions must be divisible by {divisor}")
    if generation.num_frames <= 0 or generation.num_frames % 8 != 1:
        raise ValueError("LTX 2.3 frame count must be positive and of the form 8k+1")
    if isinstance(generation.seed, bool) or not isinstance(generation.seed, int):
        raise TypeError("LTX 2.3 seed must be an integer")
    expected = {
        "ltx23_dev_t2v": (False, False),
        "ltx23_dev_i2v": (True, False),
        "ltx23_distilled_flf": (True, True),
    }[operation]
    actual = (generation.start_image_path is not None, generation.end_image_path is not None)
    if actual != expected:
        raise ValueError(f"LTX 2.3 {operation} endpoint-image contract differs")
    identities = (
        generation.start_image_identity is not None,
        generation.end_image_identity is not None,
    )
    if identities != expected:
        raise ValueError(f"LTX 2.3 {operation} endpoint-image identity contract differs")
    for label, path, identity in (
        ("start", generation.start_image_path, generation.start_image_identity),
        ("end", generation.end_image_path, generation.end_image_identity),
    ):
        if path is not None and not Path(path).resolve(strict=True).is_file():
            raise ValueError(f"LTX 2.3 {label} guide is not a file")
        if identity is not None and set(identity) != {"size_bytes", "mtime_ns", "sha256"}:
            raise ValueError(f"LTX 2.3 {label} guide identity is not canonical")
    output = Path(generation.output_path).resolve(strict=False)
    if output.suffix.lower() != ".mp4":
        raise ValueError("LTX 2.3 output must be an MP4 path")


class LTX23KitchenRuntime:
    """Persistent request-bound runtime with explicit 16 GB-class staging."""

    def __init__(
        self,
        request: LTX23KitchenRuntimeRequest,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        self.request = request
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("LTX 2.3 Kitchen runtime requires direct CUDA execution")
        self._components: dict[str, Any] | None = None
        self._transformer_residency: _LTX23TransformerResidency | None = None

    def generate(
        self,
        generation: LTX23KitchenGeneration,
        *,
        progress: LTX23KitchenProgress,
        check_cancelled: LTX23KitchenCancellation,
    ) -> LTX23KitchenResult:
        """Materialize once, then execute compatible jobs against warmed components."""

        if not _PROCESS_OWNERSHIP.acquire(blocking=False):
            raise RuntimeError("an LTX 2.3 Kitchen runtime is already active in this process")
        try:
            check_cancelled()
            if not torch.cuda.is_available():
                raise RuntimeError("LTX 2.3 Kitchen runtime requires an available CUDA device")
            if not revalidate_ltx23_kitchen_runtime_request(self.request):
                raise RuntimeError("LTX 2.3 Kitchen request changed after resolution")
            validate_ltx23_kitchen_generation(self.request.operation, generation)
            output = Path(generation.output_path).resolve(strict=False)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise FileExistsError(f"LTX 2.3 output already exists: {output}")
            pipeline_warm = self._components is not None
            if self._components is None:
                progress(0.0, "Loading LTX 2.3 components")
                self._components = self._materialize(check_cancelled, progress)
                self._transformer_residency = _LTX23TransformerResidency(
                    self._components["transformer"], self.device
                )
            else:
                progress(0.0, "Reusing warmed LTX components")
            result = self._execute(
                self._components, generation, progress=progress, check_cancelled=check_cancelled
            )
            result.metadata["cache"] = {
                "pipeline_warm": pipeline_warm,
                "policy": "components",
                "prompt_hit": False,
                "media_hit": False,
            }
            progress(1.0, "LTX 2.3 output ready")
            return result
        except BaseException:
            Path(generation.output_path).unlink(missing_ok=True)
            # A failed execution can leave staged modules in an unknown state.
            # Do not attempt reuse until this process has rebuilt its exact recipe.
            self.unload()
            raise
        finally:
            _PROCESS_OWNERSHIP.release()

    def clear_cache(self) -> None:
        """LTX currently retains only model components; prompt/media caches are absent."""

    def unload(self) -> None:
        """Release warmed components and establish a CUDA cleanup barrier."""

        components, self._components = self._components, None
        residency = getattr(self, "_transformer_residency", None)
        self._transformer_residency = None
        residency_error: BaseException | None = None
        if residency is not None:
            try:
                residency.close()
            except BaseException as exc:  # noqa: BLE001 - release remaining components too
                residency_error = exc
        if components is not None:
            try:
                _release_components(components, self.device)
            except BaseException as exc:  # noqa: BLE001 - preserve residency failure first
                if residency_error is None:
                    residency_error = exc
        if residency_error is not None:
            raise RuntimeError(f"LTX 2.3 runtime unload failed: {residency_error}") from residency_error

    def _materialize(
        self,
        check_cancelled: LTX23KitchenCancellation,
        progress: LTX23KitchenProgress,
    ) -> dict[str, Any]:
        plans = self.request.plans
        support = plans["pipeline_support"].root
        checkpoint_path = plans["checkpoint"].identity.path
        variant: Literal["dev", "distilled"] = (
            "distilled" if self.request.operation == "ltx23_distilled_flf" else "dev"
        )

        progress(0.0, "Materializing LTX transformer")
        av_contract = inspect_ltx23_av_artifact(checkpoint_path, expected_variant=variant)
        transformer = build_ltx23_av_meta_shell(av_contract)
        transformer = materialize_ltx23_av(
            transformer,
            plan_ltx23_av_materialization(transformer, checkpoint_path, expected_variant=variant),
        )
        transformer.eval()
        check_cancelled()
        progress(0.01, "Materializing LTX connectors")
        connector = build_ltx23_connector_meta_shell(av_contract)
        connector = materialize_ltx23_connectors(
            connector,
            plan_ltx23_connector_materialization(
                connector, checkpoint_path, expected_variant=variant
            ),
        )
        connector.eval()

        media: dict[str, nn.Module] = {}
        materialization_progress = {
            "video_vae": (0.02, "Materializing LTX video VAE"),
            "audio_vae": (0.03, "Materializing LTX audio VAE"),
            "vocoder": (0.04, "Materializing LTX vocoder"),
        }
        for component in ("video_vae", "audio_vae", "vocoder"):
            progress(*materialization_progress[component])
            shell = build_ltx23_media_shell(component)  # type: ignore[arg-type]
            plan = plan_ltx23_media_component(plans["checkpoint"], component, shell)  # type: ignore[arg-type]
            media[component] = materialize_ltx23_media_component(shell, plan)
            media[component].eval()
            check_cancelled()
        if variant == "dev":
            progress(0.05, "Materializing LTX latent upsampler")
            shell = build_ltx23_media_shell("latent_upsampler")
            up_plan = plan_ltx23_media_component(
                plans["latent_upscaler"], "latent_upsampler", shell
            )
            media["latent_upsampler"] = materialize_ltx23_media_component(shell, up_plan)
            media["latent_upsampler"].eval()

        progress(0.06, "Materializing LTX text encoder")
        text_plan = plan_ltx23_gemma_mixed_text_encoder(plans["text_encoder"].identity.path)
        text = load_ltx23_gemma_mixed_text_encoder(text_plan, support / "text_encoder")
        text.eval()
        check_cancelled()

        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import Gemma3Processor

        processor = Gemma3Processor.from_pretrained(support / "processor", local_files_only=True)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            support / "scheduler", local_files_only=True
        )

        model_lora = None
        text_lora = None
        if variant == "dev":
            progress(0.07, "Installing LTX model LoRA")
            model_contract = inspect_ltx23_model_lora(
                av_contract, plans["model_lora"].identity.path
            )
            model_lora = install_ltx23_model_lora(
                transformer,
                model_contract,
                adapter_name="latentslate_ltx23_distilled",
                strength=LTX23_MODEL_LORA_STRENGTH,
            )
            progress(0.075, "Installing LTX text LoRA")
            text_lora_plan = plan_ltx23_gemma_text_lora(plans["text_lora"].identity.path)
            text_lora = install_ltx23_gemma_text_lora(
                text,
                text_lora_plan,
                adapter_name="latentslate_ltx23_abliterated",
                strength=LTX23_TEXT_LORA_STRENGTH,
            )
        components = {
            "support": support,
            "transformer": transformer,
            "connector": connector,
            "text": text,
            "processor": processor,
            "scheduler": scheduler,
            "model_lora": model_lora,
            "text_lora": text_lora,
            **media,
        }
        components["_stage_bytes"] = {
            name: ltx23_module_physical_bytes(components[name])
            for name in (
                "connector",
                "text",
                "video_vae",
                "audio_vae",
                "vocoder",
            )
        }
        if "latent_upsampler" in components:
            components["_stage_bytes"]["latent_upsampler"] = ltx23_module_physical_bytes(
                components["latent_upsampler"]
            )
        return components

    def _execute(
        self,
        c: dict[str, Any],
        g: LTX23KitchenGeneration,
        *,
        progress: LTX23KitchenProgress,
        check_cancelled: LTX23KitchenCancellation,
    ) -> LTX23KitchenResult:
        base, conditioned, upsample = _build_pipelines(c, self.device)
        c["_video_processor"] = base.video_processor
        residency = self._transformer_residency
        if residency is None:
            raise RuntimeError("LTX transformer residency was not initialized")
        residency.prepare_stage(
            c["_stage_bytes"]["text"] + _LTX23_STAGE_MINIMUM_HEADROOM_BYTES
        )
        text_stage = LTX23GemmaMixedTextStage(c["text"], self.device)
        text_stage.onload()
        try:
            text_native_before = _text_native_dispatch_snapshot(c["text"])
            text_before = c["text_lora"].dispatch_snapshot() if c["text_lora"] else None
            prompt = g.prompt.strip()
            if self.request.operation == "ltx23_dev_t2v":
                progress(0.08, "Enhancing prompt")
                prompt = _enhance_prompt(
                    c["processor"],
                    c["text"],
                    prompt,
                    LTX23_PROMPT_ENHANCEMENT_SEED,
                    self.device,
                    check_cancelled,
                )
            progress(0.13, "Encoding prompt")
            prompt_embeds, prompt_mask, _, _ = base.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=False,
                max_sequence_length=1024,
                device=self.device,
                dtype=torch.bfloat16,
            )
            check_cancelled()
            text_proof = (
                c["text_lora"].verify_dispatch(text_before) if text_before is not None else None
            )
            text_native_proof = _verify_text_native_dispatch(c["text"], text_native_before)
        finally:
            text_stage.offload()

        generator = torch.Generator(device=self.device).manual_seed(g.seed)
        fp8_before = _fp8_dispatch_snapshot(c["transformer"])
        _reset_model_lora_dispatch(c)

        negative_prompt = (
            LTX23_FLF_NEGATIVE_PROMPT
            if self.request.operation == "ltx23_distilled_flf"
            else LTX23_DEV_NEGATIVE_PROMPT
        )
        if self.request.operation == "ltx23_distilled_flf":
            first = _load_rgb(g.start_image_path, g.start_image_identity)
            last = _load_rgb(g.end_image_path, g.end_image_identity)
            conditions = _conditions(
                ((first, 0, LTX23_GUIDE_STRENGTH), (last, -1, LTX23_GUIDE_STRENGTH))
            )
            output = _run_denoise(
                conditioned,
                c,
                residency=residency,
                conditions=conditions,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                generator=generator,
                width=g.width,
                height=g.height,
                num_frames=g.num_frames,
                sigmas=LTX23_MAIN_SIGMAS,
                noise_scale=None,
                progress_base=0.18,
                progress_span=0.58,
                progress=progress,
                check_cancelled=check_cancelled,
            )
            video_latents, audio_latents = output.frames, output.audio
        else:
            half_width, half_height = g.width // 2, g.height // 2
            conditions = None
            pipe = base
            if self.request.operation == "ltx23_dev_i2v":
                conditions = _conditions(
                    (
                        (
                            _load_rgb(g.start_image_path, g.start_image_identity),
                            0,
                            LTX23_GUIDE_STRENGTH,
                        ),
                    )
                )
                pipe = conditioned
            stage1 = _run_denoise(
                pipe,
                c,
                residency=residency,
                conditions=conditions,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                generator=generator,
                width=half_width,
                height=half_height,
                num_frames=g.num_frames,
                sigmas=LTX23_MAIN_SIGMAS,
                noise_scale=None,
                progress_base=0.18,
                progress_span=0.34,
                progress=progress,
                check_cancelled=check_cancelled,
            )
            check_cancelled()
            progress(0.54, "Upscaling LTX video latents")
            residency.prepare_stage(
                c["_stage_bytes"]["latent_upsampler"]
                + _LTX23_STAGE_MINIMUM_HEADROOM_BYTES
            )
            _move_module(c["latent_upsampler"], self.device)
            try:
                upscaled = upsample(
                    latents=stage1.frames,
                    latents_normalized=False,
                    height=half_height,
                    width=half_width,
                    num_frames=g.num_frames,
                    output_type="latent",
                ).frames
            finally:
                _move_module(c["latent_upsampler"], "cpu")
            check_cancelled()
            refine_conditions = None
            refine_pipe = base
            if self.request.operation == "ltx23_dev_i2v":
                refine_conditions = _conditions(
                    ((_load_rgb(g.start_image_path, g.start_image_identity), 0, 1.0),)
                )
                refine_pipe = conditioned
            stage2 = _run_denoise(
                refine_pipe,
                c,
                residency=residency,
                conditions=refine_conditions,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                generator=torch.Generator(device=self.device).manual_seed(LTX23_REFINE_SEED),
                width=g.width,
                height=g.height,
                num_frames=g.num_frames,
                sigmas=LTX23_REFINE_SIGMAS,
                noise_scale=LTX23_REFINE_SIGMAS[0],
                latents=upscaled,
                audio_latents=stage1.audio,
                progress_base=0.58,
                progress_span=0.18,
                progress=progress,
                check_cancelled=check_cancelled,
            )
            # The official two-stage topology keeps stage-one audio. Stage two
            # uses a noised copy for cross-modal refinement but its audio output
            # is deliberately discarded.
            video_latents, audio_latents = stage2.frames, stage1.audio

        fp8_proof = _native_dispatch_proof(c["transformer"], fp8_before)
        model_lora_proof = (
            ltx23_model_lora_dispatch_evidence(c["transformer"], c["model_lora"])
            if c["model_lora"] is not None
            else None
        )
        if model_lora_proof is not None and not model_lora_proof["complete"]:
            raise RuntimeError("LTX 2.3 model LoRA did not dispatch on every selected target")

        progress(0.79, "Decoding LTX video and audio")
        residency.prepare_stage(
            c["_stage_bytes"]["video_vae"] + residency.activation_headroom_bytes
        )
        frames, audio = _decode_media(c, video_latents, audio_latents, self.device, check_cancelled)
        progress(0.91, "Muxing 24 fps video and 48 kHz stereo audio")
        output = Path(g.output_path).resolve(strict=False)
        _mux_mp4(frames, audio, output, check_cancelled=check_cancelled)
        observed = _probe_mp4(output, check_cancelled)
        if (
            observed["width"] != g.width
            or observed["height"] != g.height
            or observed["num_frames"] != g.num_frames
            or observed["fps"] != LTX23_FPS
            or observed["audio_sample_rate"] != LTX23_AUDIO_SAMPLE_RATE
            or observed["audio_channels"] != LTX23_AUDIO_CHANNELS
        ):
            raise RuntimeError("LTX 2.3 published MP4 does not match its requested A/V contract")
        output_size = output.stat().st_size
        output_sha256 = _sha256_file(output, check_cancelled)
        operation_spec = ltx23_kitchen_operation_spec(self.request.operation)
        metadata = {
            "family": "ltx23",
            "runtime": "engine-native/ltx23-kitchen",
            "operation": self.request.operation,
            "request_fingerprint": self.request.fingerprint,
            "component_fingerprint": self.request.component_fingerprint,
            "seed": g.seed,
            **observed,
            "output_size_bytes": output_size,
            "output_sha256": output_sha256,
            "components": self.request.public_component_manifest(),
            "main_sigmas": list(LTX23_MAIN_SIGMAS),
            "refine_sigmas": list(LTX23_REFINE_SIGMAS)
            if self.request.operation != "ltx23_distilled_flf"
            else None,
            "prompt_enhanced": self.request.operation == "ltx23_dev_t2v",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_enhancement_system_sha256": _prompt_system_sha256()
            if operation_spec.prompt_enhancement
            else None,
            "prompt_enhancement_seed": LTX23_PROMPT_ENHANCEMENT_SEED
            if operation_spec.prompt_enhancement
            else None,
            "prompt_enhancement_max_new_tokens": LTX23_PROMPT_MAX_NEW_TOKENS
            if operation_spec.prompt_enhancement
            else None,
            "refine_seed": LTX23_REFINE_SEED if operation_spec.refine_sigmas else None,
            "negative_prompt": negative_prompt,
            "guide_strengths": list(operation_spec.guide_strengths),
            "model_lora_strength": operation_spec.model_lora_strength,
            "text_lora_strength": operation_spec.text_lora_strength,
            "native_fp8": fp8_proof,
            "native_text": text_native_proof,
            "model_lora": model_lora_proof,
            "text_lora": text_proof,
            "dense_base_dequantizations": 0,
            "residency_policy": residency.policy,
            "pipeline": "diffusers/LTX2Pipeline+LTX2ConditionPipeline+LTX2LatentUpsamplePipeline",
        }
        return LTX23KitchenResult(output, metadata)


class _LTX23TransformerResidency:
    """Persistent, budgeted transformer residency with CPU-authoritative state.

    Roots and a stable largest-first block subset remain on the execution
    device. Every other block receives a temporary synchronous GPU copy. The
    original CPU objects are rebound after a CUDA barrier, so streamed weights
    never make a device-to-host trip and compatible jobs reuse materialized RAM.
    """

    def __init__(
        self,
        transformer: nn.Module,
        device: torch.device,
        *,
        resident_weight_budget_bytes: int | None = None,
    ) -> None:
        poisoned = getattr(transformer, "_latentslate_ltx23_residency_poisoned", None)
        if poisoned:
            raise RuntimeError(f"LTX transformer residency is poisoned: {poisoned}")
        self.transformer = transformer
        self.device = _canonical_device(device)
        self.blocks = OrderedDict(
            (f"transformer_blocks.{index}", block)
            for index, block in enumerate(transformer.transformer_blocks)
        )
        if len(self.blocks) != 48:
            raise RuntimeError("LTX 2.3 transformer block topology changed")
        self.root_storage = capture_ltx23_module_storage(
            transformer, exclude_children=frozenset({"transformer_blocks"})
        )
        self.block_storage = {
            name: capture_ltx23_module_storage(block) for name, block in self.blocks.items()
        }
        self.stored_bytes = self.root_storage.physical_bytes + sum(
            storage.physical_bytes for storage in self.block_storage.values()
        )
        self.largest_group_bytes = max(
            storage.physical_bytes for storage in self.block_storage.values()
        )
        self._explicit_budget = resident_weight_budget_bytes
        self._decision: ResidencyDecision | None = None
        self._desired_resident: tuple[str, ...] = ()
        self._retention_priority: tuple[str, ...] = ()
        self._resident: dict[str, LTX23ModuleBinding] = {}
        self._root_binding: LTX23ModuleBinding | None = None
        self._streamed_binding: tuple[str, LTX23ModuleBinding] | None = None
        self._handles: list[Any] = []
        self._before_first: Callable[[], None] | None = None
        self._scope_started = False
        self._executing = False
        self._owner_thread: int | None = None
        self._closed = False
        self._barrier_failed = False
        self._streamed_transitions = 0
        self._resident_refills = 0
        self._attach()

    @property
    def handles(self) -> list[Any]:
        return self._handles

    @property
    def active(self) -> str | None:
        return None if self._streamed_binding is None else self._streamed_binding[0]

    @property
    def activation_headroom_bytes(self) -> int:
        if self._decision is not None:
            return self._decision.reserved_headroom_bytes
        if self.device.type == "cuda":
            _, total = self._cuda_capacity()
            return max(_LTX23_STAGE_MINIMUM_HEADROOM_BYTES, int(total * 0.60))
        return _LTX23_STAGE_MINIMUM_HEADROOM_BYTES

    @property
    def policy(self) -> dict[str, Any]:
        decision = (
            {"mode": "unplanned", "reason": "transformer has not executed"}
            if self._decision is None
            else self._decision.provenance()
        )
        resident = tuple(name for name in self.blocks if name in self._resident)
        streamed_count = len(self.blocks) - len(resident)
        return {
            **decision,
            "root_bytes": self.root_storage.physical_bytes,
            "resident_block_count": len(resident),
            "resident_block_bytes": sum(
                self.block_storage[name].physical_bytes for name in resident
            ),
            "streamed_block_count": streamed_count,
            "streamed_block_bytes": sum(
                storage.physical_bytes
                for name, storage in self.block_storage.items()
                if name not in self._resident
            ),
            "stream_buffer_count": int(streamed_count > 0),
            "streaming": "synchronous_cpu_master",
            "streamed_transitions": self._streamed_transitions,
            "resident_refills": self._resident_refills,
        }

    @contextmanager
    def forward_scope(self, before_first: Callable[[], None]):
        self._require_owner()
        if self._closed or self._before_first is not None:
            raise RuntimeError("LTX transformer residency forward scope is unavailable")
        self._before_first = before_first
        self._scope_started = False
        try:
            yield self
        finally:
            self._before_first = None
            self._scope_started = False

    def prepare_stage(self, required_free_bytes: int) -> None:
        """Trim optional warm transformer state until the next stage can fit."""

        self._require_owner()
        if required_free_bytes < 0:
            raise ValueError("LTX stage free-memory requirement cannot be negative")
        if self._executing or self._streamed_binding is not None:
            raise RuntimeError("cannot trim LTX transformer residency during a forward")
        if self.device.type != "cuda" or not self._resident and self._root_binding is None:
            return
        self._barrier("stage trim")
        for name in reversed(self._retention_priority):
            if self._effective_free_bytes() >= required_free_bytes:
                break
            binding = self._resident.pop(name, None)
            if binding is not None:
                binding.restore_cpu()
        if self._effective_free_bytes() < required_free_bytes and self._root_binding is not None:
            self._root_binding.restore_cpu()
            self._root_binding = None
        if self._effective_free_bytes() < required_free_bytes:
            raise RuntimeError("LTX stage cannot establish its conservative CUDA memory budget")

    def close(self) -> None:
        if self._closed:
            return
        self._require_owner()
        if self._executing:
            raise RuntimeError("cannot close LTX transformer residency during a forward")
        barrier_error: BaseException | None = None
        try:
            if self._barrier_failed:
                raise RuntimeError("an earlier CUDA residency barrier failed")
            self._barrier("teardown")
        except BaseException as exc:  # noqa: BLE001 - preserve unsafe CUDA bindings
            barrier_error = exc
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._closed = True
        if barrier_error is not None:
            reason = f"LTX CUDA residency teardown barrier failed: {barrier_error}"
            self.transformer._latentslate_ltx23_residency_poisoned = reason
            raise RuntimeError(reason) from barrier_error
        self._restore_streamed_after_barrier()
        for binding in self._resident.values():
            binding.restore_cpu()
        self._resident.clear()
        if self._root_binding is not None:
            self._root_binding.restore_cpu()
            self._root_binding = None

    def _attach(self) -> None:
        try:
            self._handles.append(self.transformer.register_forward_pre_hook(self._root_pre))
            self._handles.append(
                self.transformer.register_forward_hook(self._root_post, always_call=True)
            )
            for name, block in self.blocks.items():
                self._handles.append(block.register_forward_pre_hook(self._block_pre(name)))
        except BaseException:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            raise

    def _root_pre(self, _module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        self._require_owner()
        if self._executing:
            raise RuntimeError("LTX transformer residency is non-reentrant")
        if self._before_first is None:
            raise RuntimeError("LTX transformer forward lacks its owning residency scope")
        if not self._scope_started:
            self._scope_started = True
            self._before_first()
        self._refill()
        self._executing = True

    def _root_post(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        try:
            self._retire_streamed()
        finally:
            self._executing = False
        return output

    def _block_pre(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            self._require_owner()
            if not self._executing:
                raise RuntimeError("LTX block forward escaped transformer residency")
            self._retire_streamed()
            if name in self._resident:
                return
            try:
                binding = self.block_storage[name].copy_to(self.device)
                binding.activate()
            except BaseException as exc:
                self._poison(f"streamed onload failed for {name}: {exc}")
                raise
            self._streamed_binding = (name, binding)
            self._streamed_transitions += 1

        return hook

    def _refill(self) -> None:
        if self._decision is None:
            self._plan()
        target_budget = self._resident_budget()
        if self.root_storage.physical_bytes > target_budget:
            raise RuntimeError("LTX residency budget cannot retain required transformer roots")
        newly_bound: list[tuple[str, LTX23ModuleBinding]] = []
        root_new = False
        try:
            if self._root_binding is None:
                self._root_binding = self.root_storage.copy_to(self.device)
                self._root_binding.activate()
                root_new = True
            resident_bytes = self.root_storage.physical_bytes + sum(
                self.block_storage[name].physical_bytes for name in self._resident
            )
            for name in self._retention_priority:
                size = self.block_storage[name].physical_bytes
                if name in self._resident or resident_bytes + size > target_budget:
                    continue
                binding = self.block_storage[name].copy_to(self.device)
                binding.activate()
                self._resident[name] = binding
                newly_bound.append((name, binding))
                resident_bytes += size
            self._resident_refills += bool(root_new or newly_bound)
        except BaseException as exc:
            for name, binding in reversed(newly_bound):
                binding.restore_cpu()
                self._resident.pop(name, None)
            if root_new and self._root_binding is not None:
                self._root_binding.restore_cpu()
                self._root_binding = None
            self._poison(f"residency refill failed: {exc}")
            raise

    def _plan(self) -> None:
        if self._explicit_budget is not None:
            if self._explicit_budget < 0:
                raise ValueError("LTX explicit residency budget cannot be negative")
            self._decision = ResidencyDecision(
                mode="grouped",
                free_bytes=self._explicit_budget + self.largest_group_bytes,
                total_bytes=self._explicit_budget + self.largest_group_bytes,
                stored_bytes=self.stored_bytes,
                reserved_headroom_bytes=0,
                stream_buffer_bytes=self.largest_group_bytes,
                resident_weight_budget_bytes=min(self.stored_bytes, self._explicit_budget),
                reason="explicit test residency budget",
            )
        elif self.device.type == "cuda":
            free, total = self._cuda_capacity()
            self._decision = choose_cuda_residency(
                free_bytes=free,
                total_bytes=total,
                stored_bytes=self.stored_bytes,
                largest_group_bytes=self.largest_group_bytes,
            )
        else:
            self._decision = ResidencyDecision(
                mode="grouped",
                free_bytes=self.stored_bytes,
                total_bytes=self.stored_bytes,
                stored_bytes=self.stored_bytes,
                reserved_headroom_bytes=0,
                stream_buffer_bytes=self.largest_group_bytes,
                resident_weight_budget_bytes=(
                    self.root_storage.physical_bytes + self.largest_group_bytes
                ),
                reason="non-CUDA residency test",
            )
        budget = self._decision.resident_weight_budget_bytes
        used = self.root_storage.physical_bytes
        resident: set[str] = set()
        for name in sorted(
            self.blocks,
            key=lambda item: (-self.block_storage[item].physical_bytes, item),
        ):
            size = self.block_storage[name].physical_bytes
            if used + size <= budget:
                resident.add(name)
                used += size
        self._desired_resident = tuple(name for name in self.blocks if name in resident)
        self._retention_priority = tuple(
            name
            for name in sorted(
                self.blocks,
                key=lambda item: (-self.block_storage[item].physical_bytes, item),
            )
            if name in resident
        )

    def _resident_budget(self) -> int:
        assert self._decision is not None
        if self.device.type != "cuda" or self._explicit_budget is not None:
            return self._decision.resident_weight_budget_bytes
        free, total = self._cuda_capacity()
        owned = (
            (self.root_storage.physical_bytes if self._root_binding is not None else 0)
            + sum(self.block_storage[name].physical_bytes for name in self._resident)
        )
        decision = choose_cuda_residency(
            free_bytes=min(total, free + owned),
            total_bytes=total,
            stored_bytes=self.stored_bytes,
            largest_group_bytes=self.largest_group_bytes,
        )
        return decision.resident_weight_budget_bytes

    def _cuda_capacity(self) -> tuple[int, int]:
        driver_free, total = torch.cuda.mem_get_info(self.device)
        reusable = max(
            0,
            int(torch.cuda.memory_reserved(self.device))
            - int(torch.cuda.memory_allocated(self.device)),
        )
        return min(int(total), int(driver_free) + reusable), int(total)

    def _effective_free_bytes(self) -> int:
        return self._cuda_capacity()[0] if self.device.type == "cuda" else self.stored_bytes

    def _retire_streamed(self) -> None:
        if self._streamed_binding is None:
            return
        self._barrier("streamed block retirement")
        self._restore_streamed_after_barrier()

    def _restore_streamed_after_barrier(self) -> None:
        if self._streamed_binding is None:
            return
        _, binding = self._streamed_binding
        binding.restore_cpu()
        self._streamed_binding = None

    def _barrier(self, label: str) -> None:
        if self.device.type != "cuda":
            return
        try:
            torch.cuda.synchronize(self.device)
        except BaseException as exc:
            self._barrier_failed = True
            self._poison(f"CUDA {label} barrier failed: {exc}")
            raise

    def _poison(self, reason: str) -> None:
        self.transformer._latentslate_ltx23_residency_poisoned = reason

    def _require_owner(self) -> None:
        current = threading.get_ident()
        if self._owner_thread is None:
            self._owner_thread = current
        elif self._owner_thread != current:
            raise RuntimeError("LTX transformer residency crossed execution threads")


def _build_pipelines(c: Mapping[str, Any], device: torch.device) -> tuple[Any, Any, Any]:
    from diffusers import LTX2ConditionPipeline, LTX2LatentUpsamplePipeline, LTX2Pipeline

    class BoundPipeline(LTX2Pipeline):
        @property
        def _execution_device(self) -> torch.device:
            return device

    class BoundConditionPipeline(LTX2ConditionPipeline):
        @property
        def _execution_device(self) -> torch.device:
            return device

    class BoundUpsamplePipeline(LTX2LatentUpsamplePipeline):
        @property
        def _execution_device(self) -> torch.device:
            return device

    common = {
        "scheduler": c["scheduler"],
        "vae": c["video_vae"],
        "audio_vae": c["audio_vae"],
        "text_encoder": c["text"],
        "tokenizer": c["processor"].tokenizer,
        "connectors": c["connector"],
        "transformer": c["transformer"],
        "vocoder": c["vocoder"],
    }
    base_common = {**common, "processor": c["processor"]}
    base = BoundPipeline(**base_common)
    conditioned = BoundConditionPipeline(**common)
    upsample = (
        BoundUpsamplePipeline(vae=c["video_vae"], latent_upsampler=c.get("latent_upsampler"))
        if c.get("latent_upsampler") is not None
        else None
    )
    return base, conditioned, upsample


def _run_denoise(
    pipeline: Any,
    c: Mapping[str, Any],
    *,
    residency: _LTX23TransformerResidency,
    conditions: Any,
    negative_prompt: str,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    generator: torch.Generator,
    width: int,
    height: int,
    num_frames: int,
    sigmas: tuple[float, ...],
    noise_scale: float | None,
    progress_base: float,
    progress_span: float,
    progress: LTX23KitchenProgress,
    check_cancelled: LTX23KitchenCancellation,
    latents: torch.Tensor | None = None,
    audio_latents: torch.Tensor | None = None,
) -> Any:
    stage_bytes = c["_stage_bytes"]["connector"]
    if conditions is not None:
        stage_bytes += c["_stage_bytes"]["video_vae"]
    residency.prepare_stage(stage_bytes + _LTX23_STAGE_MINIMUM_HEADROOM_BYTES)
    connector_handles = [
        c["connector"].register_forward_pre_hook(
            lambda module, _inputs: _move_module(module, pipeline._execution_device)
        ),
        c["connector"].register_forward_hook(
            lambda module, _inputs, output: (_move_module(module, "cpu"), output)[1],
            always_call=True,
        ),
    ]
    guide_vae_loaded = conditions is not None
    if guide_vae_loaded:
        _move_module(c["video_vae"], pipeline._execution_device)

    def before_transformer() -> None:
        if guide_vae_loaded:
            _move_module(c["video_vae"], "cpu")
        check_cancelled()

    step_count = len(sigmas) - 1

    def callback(_pipe: Any, index: int, _timestep: Any, values: dict[str, Any]) -> dict[str, Any]:
        check_cancelled()
        progress(
            progress_base + progress_span * ((index + 1) / step_count),
            f"LTX denoise step {index + 1}/{step_count}",
        )
        return values

    kwargs = {
        "prompt": None,
        "negative_prompt": negative_prompt,
        "prompt_embeds": prompt_embeds,
        "prompt_attention_mask": prompt_mask,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frame_rate": float(LTX23_FPS),
        "num_inference_steps": step_count,
        "sigmas": _diffusers_sigmas(sigmas),
        "guidance_scale": LTX23_GUIDANCE_SCALE,
        "generator": generator,
        "latents": latents,
        "audio_latents": audio_latents,
        "noise_scale": noise_scale,
        "use_cross_timestep": True,
        "output_type": "latent",
        "callback_on_step_end": callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }
    if conditions is not None:
        kwargs["conditions"] = conditions
    try:
        with residency.forward_scope(before_transformer):
            return pipeline(**kwargs)
    finally:
        for handle in connector_handles:
            handle.remove()
        _move_module(c["connector"], "cpu")
        _move_module(c["video_vae"], "cpu")


def _enhance_prompt(
    processor: Any,
    model: Any,
    prompt: str,
    seed: int,
    device: torch.device,
    check_cancelled: LTX23KitchenCancellation,
) -> str:
    """Pinned public Gemma generation path without moving its meta-only vision shell."""

    check_cancelled()
    messages = [
        {"role": "system", "content": _prompt_system_text()},
        {"role": "user", "content": f"user prompt: {prompt}"},
    ]
    template = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=template, images=None, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    from transformers import StoppingCriteria, StoppingCriteriaList

    class CancellationCriteria(StoppingCriteria):
        def __call__(self, *_args: Any, **_kwargs: Any) -> bool:
            check_cancelled()
            return False

    generated = model.generate(
        **inputs,
        max_new_tokens=LTX23_PROMPT_MAX_NEW_TOKENS,
        **LTX23_PROMPT_GENERATION_SETTINGS,
        stopping_criteria=StoppingCriteriaList([CancellationCriteria()]),
    )
    check_cancelled()
    suffixes = [row[len(inputs.input_ids[index]) :] for index, row in enumerate(generated)]
    values = processor.tokenizer.batch_decode(suffixes, skip_special_tokens=True)
    if len(values) != 1 or not values[0].strip():
        raise RuntimeError("LTX 2.3 prompt enhancement returned no prompt")
    return values[0].strip()


def _decode_media(
    c: Mapping[str, Any],
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    device: torch.device,
    check_cancelled: LTX23KitchenCancellation,
) -> tuple[np.ndarray, np.ndarray]:
    vae = c["video_vae"]
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    _move_module(vae, device)
    try:
        check_cancelled()
        video = vae.decode(
            video_latents.to(device=device, dtype=vae.dtype), None, return_dict=False
        )[0]
        frames = c["_video_processor"].postprocess_video(video, output_type="np")
    finally:
        _move_module(vae, "cpu")
    check_cancelled()
    audio_vae = c["audio_vae"]
    _move_module(audio_vae, device)
    try:
        mel = audio_vae.decode(
            audio_latents.to(device=device, dtype=audio_vae.dtype), return_dict=False
        )[0]
    finally:
        _move_module(audio_vae, "cpu")
    check_cancelled()
    vocoder = c["vocoder"]
    _move_module(vocoder, device)
    try:
        audio = vocoder(mel.to(device=device, dtype=vocoder.dtype))
        audio = audio[0].float().cpu().numpy()
    finally:
        _move_module(vocoder, "cpu")
    if isinstance(frames, list) or isinstance(frames, np.ndarray) and frames.ndim == 5:
        frames = frames[0]
    return np.asarray(frames), np.asarray(audio)


def _mux_mp4(
    frames: np.ndarray,
    audio: np.ndarray,
    output: Path,
    *,
    check_cancelled: LTX23KitchenCancellation,
) -> None:
    import av

    frames = _uint8_frames(frames)
    audio = _stereo_audio(audio)
    required_audio_samples = round(frames.shape[0] / LTX23_FPS * LTX23_AUDIO_SAMPLE_RATE)
    if audio.shape[1] < required_audio_samples:
        raise ValueError(
            "LTX 2.3 audio is shorter than its generated video: "
            f"{audio.shape[1]} < {required_audio_samples} samples"
        )
    audio = audio[:, :required_audio_samples]
    staging = output.with_name(f".{output.name}.{os.getpid()}.tmp.mp4")
    staging.unlink(missing_ok=True)
    try:
        with av.open(str(staging), "w") as container:
            video_stream = container.add_stream("libx264", rate=LTX23_FPS)
            video_stream.width = int(frames.shape[2])
            video_stream.height = int(frames.shape[1])
            video_stream.pix_fmt = "yuv420p"
            audio_stream = container.add_stream("aac", rate=LTX23_AUDIO_SAMPLE_RATE)
            audio_stream.layout = "stereo"
            for index, pixels in enumerate(frames):
                check_cancelled()
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts = index
                frame.time_base = Fraction(1, LTX23_FPS)
                for packet in video_stream.encode(frame):
                    container.mux(packet)
            for packet in video_stream.encode():
                container.mux(packet)
            for offset in range(0, audio.shape[1], 1024):
                check_cancelled()
                chunk = np.ascontiguousarray(audio[:, offset : offset + 1024])
                frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo")
                frame.sample_rate = LTX23_AUDIO_SAMPLE_RATE
                frame.pts = offset
                frame.time_base = Fraction(1, LTX23_AUDIO_SAMPLE_RATE)
                for packet in audio_stream.encode(frame):
                    container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)
        check_cancelled()
        os.replace(staging, output)
    finally:
        staging.unlink(missing_ok=True)


def _probe_mp4(
    output: Path, check_cancelled: LTX23KitchenCancellation
) -> dict[str, int | float | str]:
    """Observe the published container instead of echoing request-side media facts."""

    import av

    check_cancelled()
    with av.open(str(output)) as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) != 1:
            raise ValueError("LTX 2.3 output must contain exactly one video and one audio stream")
        video = container.streams.video[0]
        audio = container.streams.audio[0]
        rate = video.average_rate
        if rate is None or rate.denominator == 0:
            raise ValueError("LTX 2.3 output video frame rate is unavailable")
        observed: dict[str, int | float | str] = {
            "container_format": container.format.name,
            "video_codec": video.codec_context.name,
            "audio_codec": audio.codec_context.name,
            "width": int(video.codec_context.width),
            "height": int(video.codec_context.height),
            "fps": int(rate) if rate.denominator == 1 else float(rate),
            "audio_sample_rate": int(audio.codec_context.sample_rate),
            "audio_channels": len(audio.codec_context.layout.channels),
        }
        frame_count = 0
        for _frame in container.decode(video=0):
            check_cancelled()
            frame_count += 1
    with av.open(str(output)) as container:
        audio_samples = 0
        for frame in container.decode(audio=0):
            check_cancelled()
            audio_samples += int(frame.samples)
    observed["num_frames"] = frame_count
    observed["audio_samples"] = audio_samples
    observed["video_duration_seconds"] = frame_count / float(observed["fps"])
    observed["audio_duration_seconds"] = audio_samples / int(observed["audio_sample_rate"])
    if "mp4" not in str(observed["container_format"]).split(","):
        raise ValueError("LTX 2.3 output container is not MP4")
    if observed["video_codec"] != "h264" or observed["audio_codec"] != "aac":
        raise ValueError("LTX 2.3 output codecs are not H.264/AAC")
    if abs(
        float(observed["video_duration_seconds"]) - float(observed["audio_duration_seconds"])
    ) > (1 / LTX23_FPS + 1024 / LTX23_AUDIO_SAMPLE_RATE):
        raise ValueError("LTX 2.3 output audio/video durations drift beyond tolerance")
    return observed


def _uint8_frames(value: np.ndarray) -> np.ndarray:
    frames = np.asarray(value)
    if frames.ndim != 4 or frames.shape[-1] != 3 or not frames.shape[0]:
        raise ValueError("LTX 2.3 video must be nonempty FHWC RGB")
    if frames.dtype == np.uint8:
        return np.ascontiguousarray(frames)
    if not np.issubdtype(frames.dtype, np.floating) or not np.isfinite(frames).all():
        raise ValueError("LTX 2.3 video samples must be finite floats or uint8")
    return np.ascontiguousarray((np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8))


def _stereo_audio(value: np.ndarray) -> np.ndarray:
    audio = np.asarray(value, dtype=np.float32)
    if audio.ndim != 2:
        raise ValueError("LTX 2.3 audio must be a two-dimensional stereo waveform")
    if audio.shape[0] != LTX23_AUDIO_CHANNELS and audio.shape[1] == LTX23_AUDIO_CHANNELS:
        audio = audio.T
    if audio.shape[0] != LTX23_AUDIO_CHANNELS or not audio.shape[1]:
        raise ValueError("LTX 2.3 audio must contain exactly two nonempty channels")
    if not np.isfinite(audio).all():
        raise ValueError("LTX 2.3 audio samples must be finite")
    return np.ascontiguousarray(np.clip(audio, -1.0, 1.0))


def _conditions(items: tuple[tuple[Image.Image, int, float], ...]) -> list[Any]:
    from diffusers.pipelines.ltx2 import LTX2VideoCondition

    return [
        LTX2VideoCondition(frames=image, index=index, strength=strength)
        for image, index, strength in items
    ]


def ltx23_guide_identity(path: Path) -> dict[str, int | str]:
    """Hash one guide image with a stable stat envelope for request binding."""

    candidate = Path(path).resolve(strict=True)
    before = candidate.stat()
    digest = _sha256_file(candidate, lambda: None)
    after = candidate.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("LTX 2.3 guide changed while its identity was measured")
    return {
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
    }


def _load_rgb(path: Path | None, identity: Mapping[str, object] | None) -> Image.Image:
    if path is None or identity is None:
        raise ValueError("LTX 2.3 guide path or identity is missing")
    if ltx23_guide_identity(path) != dict(identity):
        raise ValueError("LTX 2.3 guide image changed after request binding")
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _move_module(module: nn.Module, device: torch.device | str) -> None:
    module.to(device=device)
    for nested in module.modules():
        if isinstance(nested, LTX23StoredFP8Linear):
            nested.move_stored_storage(device)


def _canonical_device(device: torch.device | str) -> torch.device:
    target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return target


def _reset_model_lora_dispatch(c: Mapping[str, Any]) -> None:
    installation = c.get("model_lora")
    if installation is not None:
        ltx23_model_lora_dispatch_evidence(c["transformer"], installation, reset=True)


def _text_native_dispatch_snapshot(model: nn.Module) -> dict[str, int]:
    expected = getattr(model, "_latentslate_ltx23_gemma_quant_modules", None)
    if not isinstance(expected, Mapping) or not expected:
        raise RuntimeError("LTX 2.3 mixed text native-dispatch contract is missing")
    snapshot = {
        name: int(getattr(model.get_submodule(name), "native_dispatch_count", -1))
        for name in expected
    }
    if any(value < 0 for value in snapshot.values()):
        raise RuntimeError("LTX 2.3 mixed text native-dispatch counters are missing")
    return snapshot


def _verify_text_native_dispatch(
    model: nn.Module, before: Mapping[str, int]
) -> dict[str, int | str]:
    after = _text_native_dispatch_snapshot(model)
    if set(after) != set(before):
        raise RuntimeError("LTX 2.3 mixed text quantized module identity changed")
    deltas = {name: after[name] - int(before[name]) for name in after}
    if not deltas or any(value <= 0 for value in deltas.values()):
        missed = sorted(name for name, value in deltas.items() if value <= 0)
        raise RuntimeError(
            f"LTX 2.3 mixed text did not use every native quantized layer: {missed[:3]}"
        )
    return {
        "backend": "comfy_kitchen/cuda/mixed-fp8-nvfp4",
        "module_count": len(deltas),
        "total_dispatches": sum(deltas.values()),
        "minimum_module_dispatches": min(deltas.values()),
        "maximum_module_dispatches": max(deltas.values()),
    }


def _fp8_dispatch_snapshot(transformer: nn.Module) -> dict[str, tuple[int, int, int]]:
    return {
        name: (
            module.native_dispatch_count,
            module.rejected_dispatch_count,
            module.dense_fallback_count,
        )
        for name, module in transformer.named_modules()
        if isinstance(module, LTX23StoredFP8Linear)
    }


def _native_dispatch_proof(
    transformer: nn.Module, before: Mapping[str, tuple[int, int, int]]
) -> dict[str, Any]:
    after = _fp8_dispatch_snapshot(transformer)
    if set(after) != set(before):
        raise RuntimeError("LTX 2.3 native FP8 module topology changed during execution")
    native = sum(after[name][0] - before[name][0] for name in after)
    rejected = sum(after[name][1] - before[name][1] for name in after)
    fallback = sum(after[name][2] - before[name][2] for name in after)
    dispatched_modules = sum(after[name][0] > before[name][0] for name in after)
    if dispatched_modules != len(after) or rejected or fallback:
        raise RuntimeError(
            "LTX 2.3 native FP8 proof failed: "
            f"modules={dispatched_modules}/{len(after)}, native={native}, "
            f"rejected={rejected}, dense_fallback={fallback}"
        )
    return {
        "backend": "comfy_kitchen.tensorcore_fp8",
        "modules": len(after),
        "dispatched_modules": dispatched_modules,
        "complete": dispatched_modules == len(after),
        "native_dispatch_count": native,
        "rejected_dispatch_count": rejected,
        "dense_fallback_count": fallback,
    }


def _prompt_system_text() -> str:
    """Return the pinned first-party LTX 2.3 T2V enhancement instruction."""

    from diffusers.pipelines.ltx2.utils import T2V_DEFAULT_SYSTEM_PROMPT

    return T2V_DEFAULT_SYSTEM_PROMPT.strip()


def _prompt_system_sha256() -> str:
    return hashlib.sha256(_prompt_system_text().encode()).hexdigest()


def _diffusers_sigmas(saved_sigmas: tuple[float, ...]) -> list[float]:
    """Remove the saved terminal zero that Diffusers appends internally."""

    if (
        len(saved_sigmas) < 2
        or saved_sigmas[-1] != 0.0
        or any(left <= right for left, right in pairwise(saved_sigmas))
    ):
        raise ValueError("LTX 2.3 saved sigma schedule must strictly descend to zero")
    return list(saved_sigmas[:-1])


def _release_components(c: dict[str, Any], device: torch.device) -> None:
    errors: list[Exception] = []
    for name, value in c.items():
        if isinstance(value, nn.Module):
            if getattr(value, "_latentslate_ltx23_residency_poisoned", None):
                continue
            try:
                if name == "text":
                    # The Gemma text component deliberately retains its unused
                    # vision/projector hierarchy on meta.  Its marker is an
                    # ownership aid, not permission to fall back to a generic
                    # whole-model move during failed-prompt cleanup.
                    LTX23GemmaMixedTextStage(value, device).offload()
                else:
                    _move_module(value, "cpu")
            except Exception as exc:  # noqa: BLE001 - attempt every safety transition
                errors.append(exc)
    c.clear()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    if errors:
        raise RuntimeError(f"LTX 2.3 component release failed: {errors[0]}") from errors[0]


def _sha256_file(path: Path, check_cancelled: LTX23KitchenCancellation) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            check_cancelled()
            digest.update(chunk)
    return digest.hexdigest()
