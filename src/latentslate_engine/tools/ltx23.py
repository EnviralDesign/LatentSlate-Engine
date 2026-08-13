from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ..config import Settings
from ..ltx23_comfy_recipe import LTX23ComfyRuntimeRequest
from ..protocol import (
    AssetInput,
    InputRole,
    InputType,
    InputUi,
    MediaType,
    ToolDescriptor,
    ToolInput,
    ToolOutput,
    ToolRequirement,
    WorkflowKind,
)
from ..resources import ResourceDescriptor
from ..runtime.diffusers_repository import (
    LTX23_REPOSITORY_CONTRACT,
    validate_diffusers_repository,
)
from ..runtime.kit import ResolvedRuntimePlan
from ..runtime.ltx23 import (
    LTX23_MAX_DURATION_SECONDS,
    LTX23_MIN_DURATION_SECONDS,
    resolve_ltx23_runtime_plan,
)
from ..runtime.ltx23_managed import ManagedLTX23Runtime
from ..runtime.manager import RUNTIME_MANAGER
from ..storage import StoredArtifact
from .base import ExecutionCapabilities, Tool, ToolCancelled, ToolContext

TEXT_TO_VIDEO_ID = UUID("46bdb57c-3b19-5397-8949-4e20ffe757c9")
FIRST_FRAME_TO_VIDEO_ID = UUID("5d6e2d6f-216c-5f35-a4ec-1565d6e56ee7")
FIRST_LAST_FRAME_TO_VIDEO_ID = UUID("1a8f9c0b-410e-56e4-90de-23bcb9d644ca")


def _comfy_artifact_metadata(
    audio_video: dict[str, Any],
    *,
    seed: int,
    operation: str,
    provenance: dict[str, Any],
    conditioning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate observed ffprobe facts into a stored-artifact metadata view."""

    metadata = {
        "width": audio_video["width"],
        "height": audio_video["height"],
        "frame_count": audio_video["frame_count"],
        "fps": audio_video["fps"],
        "duration_seconds": audio_video["video_duration_seconds"],
        "has_audio": True,
        "seed": seed,
        "operation": operation,
        "runtime_provenance": provenance,
    }
    if conditioning is not None:
        metadata["conditioning"] = conditioning
    return metadata


def _runtime_availability() -> tuple[bool, str | None]:
    # This family hosts both native-Diffusers reference recipes and official
    # Comfy recipes.  The latter must remain available without importing or
    # requiring the native reference extras; its checkout/node check occurs in
    # the isolated Comfy runtime before any artifact staging.
    return True, None


def _native_runtime_availability() -> tuple[bool, str | None]:
    missing = [
        module
        for module in (
            "torch",
            "diffusers",
            "transformers",
            "accelerate",
            "sentencepiece",
            "av",
        )
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return False, f"Install the ltx23 extra; missing: {', '.join(missing)}"
    return True, None


def _comfy_runtime_availability(root: Path) -> tuple[bool, str | None]:
    from ..runtime.ltx23_comfy import _validate_comfy_checkout

    if os.name != "nt":
        return False, "LTX 2.3 optimized Comfy workers require Windows Job Object support"
    try:
        _validate_comfy_checkout(root)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return False, str(exc)
    return (
        True,
        "Comfy checkout is structurally ready; GPU/CUDA capability is verified at execution time.",
    )


def _inputs() -> list[ToolInput]:
    return [
        ToolInput(
            key="prompt",
            label="Prompt",
            type=InputType.TEXT,
            role=InputRole.PROMPT,
            required=True,
            ui=InputUi(
                group="Prompt",
                multiline=True,
                placeholder="Describe the shot, motion, camera, and sound.",
            ),
        ),
        ToolInput(
            key="width",
            label="Width",
            type=InputType.INTEGER,
            role=InputRole.WIDTH,
            required=True,
            default=768,
            ui=InputUi(group="Output", min=64, step=1, unit="pixels"),
        ),
        ToolInput(
            key="height",
            label="Height",
            type=InputType.INTEGER,
            role=InputRole.HEIGHT,
            required=True,
            default=512,
            ui=InputUi(group="Output", min=64, step=1, unit="pixels"),
        ),
        ToolInput(
            key="duration_seconds",
            label="Duration",
            type=InputType.NUMBER,
            role=InputRole.DURATION_SECONDS,
            required=True,
            default=5.0,
            ui=InputUi(
                group="Output",
                min=LTX23_MIN_DURATION_SECONDS,
                max=LTX23_MAX_DURATION_SECONDS,
                step=0.5,
                unit="seconds",
            ),
        ),
        ToolInput(
            key="seed",
            label="Seed",
            type=InputType.INTEGER,
            role=InputRole.SEED,
            required=True,
            default=0,
            ui=InputUi(group="Advanced", advanced=True, min=0, step=1),
        ),
    ]


class LTX23TextToVideoTool(Tool):
    _worker_operation = "t2v"
    def model_family(self) -> str:
        return "ltx23"

    def variant_base_availability(self) -> tuple[bool, str | None]:
        return _runtime_availability()

    def variant_recipe_availability(
        self,
        recipe_type: str | None,
        settings: Settings,
    ) -> tuple[bool, str | None]:
        if recipe_type in {
            "ltx23_comfy_dev_t2v",
            "ltx23_comfy_dev_i2v",
            "ltx23_comfy_distilled_flf",
        }:
            return _comfy_runtime_availability(settings.comfyui_root)
        return _native_runtime_availability()

    def variant_recipe_provenance(self, recipe_type: str | None) -> dict[str, Any]:
        operations = {
            "ltx23_comfy_dev_t2v": "comfy_dev_t2v",
            "ltx23_comfy_dev_i2v": "comfy_dev_i2v",
            "ltx23_comfy_distilled_flf": "comfy_distilled_flf",
        }
        if recipe_type in operations:
            return {
                "runtime": "comfyui_disposable_worker",
                "pipeline": "official_comfy_ltx23_graph",
                "model_family": "ltx_2_3",
                "artifact_contract": "pinned_comfy_fp8_operation_closure",
                "operation": operations[recipe_type],
            }
        return self.provenance()

    def execution_capabilities(self) -> ExecutionCapabilities:
        available, _reason = _runtime_availability()
        if not available:
            return ExecutionCapabilities()
        return ExecutionCapabilities(
            recipe_types=frozenset({
                "ltx23_comfy_dev_t2v",
                "ltx23_comfy_dev_i2v",
                "ltx23_comfy_distilled_flf",
            }),
            # LTX2Pipeline.from_pretrained consumes a complete repository.  A
            # standalone SafeTensors or GGUF resource is not a substitutable
            # pipeline, even when its name contains LTX.
            model_formats=frozenset({"diffusers", "safetensors"}),
            lora_formats=frozenset(),
            attention_modes=frozenset({"native"}),
            offload_modes=frozenset({"sequential", "model", "none"}),
            quantization_modes=frozenset({"bf16", "fp8"}),
            compile_modes=frozenset(),
            vae_tiling_modes=frozenset({"on"}),
            vae_slicing_modes=frozenset(),
            # A fresh process owns every LTX job, so tensor prompt caches cannot
            # safely cross jobs and must not be advertised as warmable.
            cache_modes=frozenset({"none"}),
            load_policy=False,
            residency_policy=True,
            runtime_parameters=False,
        )

    def validate_model_resource(
        self,
        _resource: ResourceDescriptor,
        path: Path,
    ) -> list[str]:
        try:
            validate_diffusers_repository(path, LTX23_REPOSITORY_CONTRACT)
        except (OSError, TypeError, ValueError) as exc:
            return [str(exc)]
        return []

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability()
        return ToolDescriptor(
            id=TEXT_TO_VIDEO_ID,
            key="ltx23.text_to_video",
            schema_revision=2,
            name="Text to Video",
            description="Generate synchronized video and audio with LTX 2.3.",
            workflow_kind=WorkflowKind.TEXT_TO_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=_inputs(),
            requirements=[ToolRequirement(bundle_id="ltx23-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def _resolve_plan(self, context: ToolContext) -> ResolvedRuntimePlan:
        return resolve_ltx23_runtime_plan(context.settings, context.execution)

    def _runtime(
        self,
        context: ToolContext,
        plan: ResolvedRuntimePlan,
    ) -> ManagedLTX23Runtime:
        key = ("ltx23", self._worker_operation, plan.pipeline_fingerprint)
        return RUNTIME_MANAGER.activate(
            key,
            lambda: ManagedLTX23Runtime(
                context.settings, plan, operation=self._worker_operation
            ),
        )

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        if isinstance(context.execution.recipe if context.execution else None, LTX23ComfyRuntimeRequest):
            return self._run_comfy(context, inputs)
        plan = self._resolve_plan(context)
        context.record_provenance(runtime_plan=plan.provenance())
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")
        runtime = self._runtime(context, plan)
        try:
            metadata = runtime.generate(
                plan=plan,
                prompt=str(inputs["prompt"]),
                output_path=output_path,
                width=int(inputs["width"]),
                height=int(inputs["height"]),
                duration_seconds=float(inputs["duration_seconds"]),
                seed=int(inputs["seed"]),
                progress=context.progress,
                check_cancelled=context.check_cancelled,
            )
            context.record_provenance(
                runtime_result={
                    "pipeline_fingerprint": metadata["pipeline_fingerprint"],
                    "pipeline_warm": metadata["cache"]["pipeline_warm"],
                    "pipeline_kit": metadata["pipeline_kit"],
                    "cache": metadata["cache"],
                    "worker": runtime.status()["last_worker"],
                    "sampling": {
                        "steps": metadata["steps"],
                        "guidance_scale": metadata["guidance_scale"],
                        "sigmas": metadata["sigmas"],
                    },
                    "audio_video": {
                        "fps": metadata["fps"],
                        "frame_count": metadata["frame_count"],
                        "duration_seconds": metadata["duration_seconds"],
                        "has_audio": metadata["has_audio"],
                    },
                }
            )
        finally:
            # The supervisor retains no model tensors in the Engine process.
            pass
        return [
            StoredArtifact(
                id=uuid4(),
                filename=output_path.name,
                content_type="video/mp4",
                path=output_path,
                role="primary",
                media_type="video",
                metadata=metadata,
            )
        ]

    def _run_comfy(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        return self._run_comfy_operation(context, inputs)

    def _run_comfy_condition(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        return self._run_comfy_operation(context, inputs)

    def _run_comfy_operation(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        """One operation-bound entry point for all isolated optimized graphs."""

        from ..runtime.ltx23_comfy import LTX23ComfyRequest, ManagedLTX23ComfyRuntime

        recipe = context.execution.recipe if context.execution else None
        expected = {
            "t2v": "comfy_dev_t2v",
            "first_frame": "comfy_dev_i2v",
            "first_last": "comfy_distilled_flf",
        }[self._worker_operation]
        if not isinstance(recipe, LTX23ComfyRuntimeRequest) or recipe.operation != expected:
            raise TypeError("LTX 2.3 Comfy operation does not match its public tool")
        start = AssetInput.model_validate(inputs["start_image"]) if expected != "comfy_dev_t2v" else None
        end = AssetInput.model_validate(inputs["end_image"]) if expected == "comfy_distilled_flf" else None
        runtime = RUNTIME_MANAGER.activate(
            ("ltx23_comfy", recipe.operation, recipe.component_fingerprint),
            lambda: ManagedLTX23ComfyRuntime(recipe, comfy_root=context.settings.comfyui_root),
        )
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")
        context.record_provenance(
            ltx23_comfy_recipe={
                "operation": recipe.operation,
                "fingerprint": recipe.fingerprint,
                "component_fingerprint": recipe.component_fingerprint,
                "components": recipe.public_component_manifest(),
            }
        )
        try:
            result = runtime.generate(
                LTX23ComfyRequest(
                    prompt=str(inputs["prompt"]),
                    width=int(inputs["width"]),
                    height=int(inputs["height"]),
                    duration_seconds=float(inputs["duration_seconds"]),
                    seed=int(inputs["seed"]),
                    start_image=context.resolve_asset(start.asset_id) if start else None,
                    end_image=context.resolve_asset(end.asset_id) if end else None,
                ),
                output_path=output_path,
                progress=context.progress,
                check_cancelled=context.check_cancelled,
            )
        except (asyncio.CancelledError, ToolCancelled) as exc:
            # Retain the zero-residency wrapper for public cancellation status.
            raise ToolCancelled("Generation canceled") from exc
        except BaseException:
            # The worker itself is already terminal/zero-residency. Keep this
            # wrapper so public status retains failure/tree/cleanup evidence.
            raise
        context.record_provenance(runtime_result=result.provenance)
        audio_video = result.provenance["audio_video"]
        return [
            StoredArtifact(
                id=uuid4(), filename=output_path.name, content_type="video/mp4",
                path=output_path, role="primary", media_type="video",
                metadata=_comfy_artifact_metadata(
                    audio_video, seed=int(inputs["seed"]), operation=recipe.operation,
                    provenance=result.provenance,
                    conditioning=result.provenance.get("conditioning"),
                ),
            )
        ]

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime": "diffusers_disposable_worker",
            "pipeline": "LTX2Pipeline",
            "model_family": "ltx_2_3",
            "artifact_contract": "complete_diffusers_bf16_native",
        }


class LTX23ImageToVideoTool(LTX23TextToVideoTool):
    """LTX 2.3 video with explicitly required first and last frames."""

    _worker_operation = "first_last"

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability()
        media_inputs = [
            ToolInput(
                key="start_image",
                label="First Frame",
                type=InputType.IMAGE,
                role=InputRole.START_IMAGE,
                required=True,
                ui=InputUi(group="Keyframes"),
            ),
            ToolInput(
                key="end_image",
                label="Last Frame",
                type=InputType.IMAGE,
                role=InputRole.END_IMAGE,
                required=False,
                ui=InputUi(group="Keyframes"),
            ),
        ]
        return ToolDescriptor(
            id=FIRST_LAST_FRAME_TO_VIDEO_ID,
            key="ltx23.first_last_frame_to_video",
            schema_revision=1,
            name="First and Last Frame to Video",
            description=(
                "Generate synchronized video and audio with LTX 2.3 from required first "
                "and last-frame anchors."
            ),
            workflow_kind=WorkflowKind.FIRST_FRAME_LAST_FRAME_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=[
                _inputs()[0],
                media_inputs[0],
                media_inputs[1].model_copy(update={"required": True}),
                *_inputs()[1:],
            ],
            requirements=[ToolRequirement(bundle_id="ltx23-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def _runtime(
        self,
        context: ToolContext,
        plan: ResolvedRuntimePlan,
    ) -> ManagedLTX23Runtime:
        # Conditions alter pipeline class and operation binding; never reuse the
        # text-to-video supervisor even when both point to the same model folder.
        key = ("ltx23_condition", self._worker_operation, plan.pipeline_fingerprint)
        return RUNTIME_MANAGER.activate(
            key,
            lambda: ManagedLTX23Runtime(
                context.settings, plan, operation=self._worker_operation
            ),
        )

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        if isinstance(context.execution.recipe if context.execution else None, LTX23ComfyRuntimeRequest):
            return self._run_comfy_condition(context, inputs)
        start = AssetInput.model_validate(inputs["start_image"])
        end = AssetInput.model_validate(inputs["end_image"])
        plan = self._resolve_plan(context)
        context.record_provenance(runtime_plan=plan.provenance())
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")
        runtime = self._runtime(context, plan)
        try:
            metadata = runtime.generate(
                plan=plan,
                prompt=str(inputs["prompt"]),
                output_path=output_path,
                width=int(inputs["width"]),
                height=int(inputs["height"]),
                duration_seconds=float(inputs["duration_seconds"]),
                seed=int(inputs["seed"]),
                start_image_path=context.resolve_asset(start.asset_id),
                end_image_path=context.resolve_asset(end.asset_id),
                progress=context.progress,
                check_cancelled=context.check_cancelled,
            )
            context.record_provenance(
                runtime_result={
                    "pipeline_fingerprint": metadata["pipeline_fingerprint"],
                    "pipeline_warm": metadata["cache"]["pipeline_warm"],
                    "pipeline_kit": metadata["pipeline_kit"],
                    "cache": metadata["cache"],
                    "worker": runtime.status()["last_worker"],
                    "sampling": {
                        "steps": metadata["steps"],
                        "guidance_scale": metadata["guidance_scale"],
                        "sigmas": metadata["sigmas"],
                    },
                    "conditioning": metadata["conditioning"],
                    "audio_video": {
                        "fps": metadata["fps"],
                        "frame_count": metadata["frame_count"],
                        "duration_seconds": metadata["duration_seconds"],
                        "has_audio": metadata["has_audio"],
                    },
                }
            )
        finally:
            pass
        return [
            StoredArtifact(
                id=uuid4(),
                filename=output_path.name,
                content_type="video/mp4",
                path=output_path,
                role="primary",
                media_type="video",
                metadata=metadata,
            )
        ]

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime": "diffusers_disposable_worker",
            "pipeline": "LTX2ConditionPipeline",
            "model_family": "ltx_2_3",
            "artifact_contract": "complete_diffusers_bf16_native",
        }


class LTX23FirstFrameToVideoTool(LTX23ImageToVideoTool):
    """LTX 2.3 public first-frame-only operation.

    This is intentionally not an optional-endpoint schema.  A first+last
    request is a separate tool and recipe so authored callers cannot silently
    select a different conditioning topology by omitting a field.
    """

    _worker_operation = "first_frame"

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability()
        return ToolDescriptor(
            id=FIRST_FRAME_TO_VIDEO_ID,
            key="ltx23.image_to_video",
            schema_revision=2,
            name="First Frame to Video",
            description="Generate synchronized video and audio with LTX 2.3 from a first frame.",
            workflow_kind=WorkflowKind.IMAGE_TO_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=[_inputs()[0], ToolInput(
                key="start_image",
                label="First Frame",
                type=InputType.IMAGE,
                role=InputRole.START_IMAGE,
                required=True,
                ui=InputUi(group="Keyframes"),
            ), *_inputs()[1:]],
            requirements=[ToolRequirement(bundle_id="ltx23-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        if isinstance(context.execution.recipe if context.execution else None, LTX23ComfyRuntimeRequest):
            return self._run_comfy_condition(context, inputs)
        start = AssetInput.model_validate(inputs["start_image"])
        plan = self._resolve_plan(context)
        context.record_provenance(runtime_plan=plan.provenance())
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")
        runtime = self._runtime(context, plan)
        try:
            metadata = runtime.generate(
                plan=plan,
                prompt=str(inputs["prompt"]),
                output_path=output_path,
                width=int(inputs["width"]),
                height=int(inputs["height"]),
                duration_seconds=float(inputs["duration_seconds"]),
                seed=int(inputs["seed"]),
                start_image_path=context.resolve_asset(start.asset_id),
                end_image_path=None,
                progress=context.progress,
                check_cancelled=context.check_cancelled,
            )
            context.record_provenance(
                runtime_result={
                    "pipeline_fingerprint": metadata["pipeline_fingerprint"],
                    "pipeline_warm": metadata["cache"]["pipeline_warm"],
                    "pipeline_kit": metadata["pipeline_kit"],
                    "cache": metadata["cache"],
                    "worker": runtime.status()["last_worker"],
                    "sampling": {
                        "steps": metadata["steps"],
                        "guidance_scale": metadata["guidance_scale"],
                        "sigmas": metadata["sigmas"],
                    },
                    "conditioning": metadata["conditioning"],
                    "audio_video": {
                        "fps": metadata["fps"],
                        "frame_count": metadata["frame_count"],
                        "duration_seconds": metadata["duration_seconds"],
                        "has_audio": metadata["has_audio"],
                    },
                }
            )
        finally:
            pass
        return [
            StoredArtifact(
                id=uuid4(),
                filename=output_path.name,
                content_type="video/mp4",
                path=output_path,
                role="primary",
                media_type="video",
                metadata=metadata,
            )
        ]
