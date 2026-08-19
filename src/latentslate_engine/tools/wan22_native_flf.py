"""Curated native Wan 2.2 14B first/last-frame recipe tool."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID, uuid4

from ..protocol import (
    AssetInput,
    InputRole,
    InputType,
    InputUi,
    MediaType,
    ToolDescriptor,
    ToolInput,
    ToolOutput,
    WorkflowKind,
)
from ..runtime.wan22_i2v_conditioning import WAN_I2V_CANVAS
from ..runtime.manager import RUNTIME_MANAGER
from ..storage import StoredArtifact
from ..wan22_recipe import Wan22RuntimeRequest
from .base import ExecutionCapabilities, ExecutionRequest, Tool, ToolCancelled, ToolContext
from .wan22_native import NATIVE_WAN14B_FPS, _native_runtime_availability
from .wan22_native import _inputs as _i2v_inputs

NATIVE_WAN14B_FLF_ID = UUID("b9f5155b-83b0-52a5-bc63-a5feab0ed31f")
NATIVE_WAN14B_FLF_KEY = "wan22.native_first_last_frame_video"
NATIVE_WAN14B_FLF_RECIPE_TYPE = "wan22_flf_14b"
_FLF_OPERATIONS = frozenset({"wan22_flf_base", "wan22_flf_lightx2v_4step"})


def _inputs() -> list[ToolInput]:
    inputs = _i2v_inputs()
    inputs[0] = ToolInput(
        key="start_image",
        label="Start Image",
        type=InputType.IMAGE,
        role=InputRole.START_IMAGE,
        required=True,
        ui=InputUi(group="Input"),
    )
    inputs.insert(
        1,
        ToolInput(
            key="end_image",
            label="End Image",
            type=InputType.IMAGE,
            role=InputRole.END_IMAGE,
            required=True,
            ui=InputUi(group="Input"),
        ),
    )
    return inputs


class NativeWan14BFLFTool(Tool):
    def model_family(self) -> str:
        return "wan22"

    def variant_base_availability(self) -> tuple[bool, str | None]:
        return _native_runtime_availability()

    def execution_capabilities(self) -> ExecutionCapabilities:
        available, _ = _native_runtime_availability()
        return (
            ExecutionCapabilities(
                recipe_types=frozenset({NATIVE_WAN14B_FLF_RECIPE_TYPE}),
                lora_formats=frozenset({"safetensors"}),
                residency_policy=True,
            )
            if available
            else ExecutionCapabilities()
        )

    def validate_execution_request(self, request: ExecutionRequest) -> list[str]:
        errors = super().validate_execution_request(request)
        if request.recipe_type != NATIVE_WAN14B_FLF_RECIPE_TYPE:
            errors.append(
                f"native Wan 14B FLF requires recipe type {NATIVE_WAN14B_FLF_RECIPE_TYPE!r}"
            )
        if request.model_override:
            errors.append("native Wan recipes select explicit components, not a model override")
        return errors

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            id=NATIVE_WAN14B_FLF_ID,
            key=NATIVE_WAN14B_FLF_KEY,
            schema_revision=2,
            name="Native Wan 14B First/Last Frame Video",
            description="Engine-owned high/low Wan 2.2 14B first/last-frame video using exact stored I2V artifacts.",
            workflow_kind=WorkflowKind.FIRST_FRAME_LAST_FRAME_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=_inputs(),
            canvas=WAN_I2V_CANVAS,
            requirements=[],
            available=False,
            unavailable_reason="native Wan 14B FLF requires an explicit validated component recipe",
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        context.check_cancelled()
        recipe = context.execution.recipe if context.execution is not None else None
        if not isinstance(recipe, Wan22RuntimeRequest) or recipe.operation not in _FLF_OPERATIONS:
            raise TypeError("native Wan FLF execution requires a validated FLF recipe request")
        from ..runtime.wan22_flf_runtime import WanFLFRequest
        from ..runtime.wan22_native_managed import ManagedNativeWanI2VRuntime

        start_path = context.resolve_asset(
            AssetInput.model_validate(inputs["start_image"]).asset_id
        )
        end_path = context.resolve_asset(AssetInput.model_validate(inputs["end_image"]).asset_id)
        generation = WanFLFRequest(
            start_image=None,
            end_image=None,
            prompt=str(inputs["prompt"]),
            negative_prompt=str(inputs.get("negative_prompt") or ""),
            num_frames=int(inputs["num_frames"]),
            height=int(inputs["height"]),
            width=int(inputs["width"]),
            steps=int(inputs["steps"]),
            seed=int(inputs["seed"]),
            stage_policy=str(inputs["stage_policy"]),
            high_guidance=float(inputs["high_guidance"]),
            low_guidance=float(inputs["low_guidance"]),
            operation=recipe.operation,
        )
        runtime = RUNTIME_MANAGER.activate(
            ("wan22_native_flf_14b", recipe.fingerprint, context.settings.wan22_device),
            lambda: ManagedNativeWanI2VRuntime(recipe),
        )
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")

        def progress(completed: int, total: int, stage: str) -> None:
            context.check_cancelled()
            fraction = completed / max(1, total)
            context.progress(
                0.12 + 0.76 * fraction,
                f"Generating native Wan video ({completed}/{total}, {stage})",
            )

        input_assets = {
            "start_image_asset_id": str(AssetInput.model_validate(inputs["start_image"]).asset_id),
            "end_image_asset_id": str(AssetInput.model_validate(inputs["end_image"]).asset_id),
        }
        context.record_provenance(
            native_wan_recipe={
                "fingerprint": recipe.fingerprint,
                "type": NATIVE_WAN14B_FLF_RECIPE_TYPE,
                "components": recipe.public_component_manifest(),
                "operation": recipe.operation,
                "input_assets": input_assets,
                "configured_loras": [dict(item) for item in recipe.configured_loras],
                "active_loras": [item.public_dict() for item in recipe.active_loras],
            }
        )
        succeeded = False
        completed_output_owned = False
        try:
            context.progress(0.02, "Loading native Wan 14B component recipe")
            result = runtime.generate(
                generation,
                source_image_path=start_path,
                end_image_path=end_path,
                output_path=output_path,
                device=context.settings.wan22_device,
                fps=NATIVE_WAN14B_FPS,
                progress=progress,
                cancelled=context.cancel_event.is_set,
            )
            completed_output_owned = True
            context.check_cancelled()
            pipeline_warm = bool(getattr(result, "pipeline_warm", False))
            context.record_provenance(
                runtime_result={
                    "runtime": "NativeWanFLFRuntimePersistentWorker",
                    "recipe_fingerprint": recipe.fingerprint,
                    "pipeline_warm": pipeline_warm,
                    "execution_cache": {
                        "supported": True,
                        "hit": pipeline_warm,
                        "mode": "exact_recipe_persistent_worker",
                    },
                    "worker": {
                        "pid": result.worker_pid,
                        "exit_code": result.worker_exit_code,
                        "terminated": False,
                        "memory_boundary": "persistent_exact_recipe_worker",
                    },
                    "provenance": result.provenance,
                }
            )
            context.progress(1.0, "Complete")
            succeeded = True
            return [
                StoredArtifact(
                    id=uuid4(),
                    filename=output_path.name,
                    content_type="video/mp4",
                    path=output_path,
                    role="primary",
                    media_type="video",
                    metadata={
                        **result.stream_metadata,
                        "steps": generation.steps,
                        "seed": generation.seed,
                        "stage_policy": generation.stage_policy,
                        "high_guidance": generation.high_guidance,
                        "low_guidance": generation.low_guidance,
                        "recipe_fingerprint": recipe.fingerprint,
                        "components": recipe.public_component_manifest(),
                        "input_assets": input_assets,
                        "runtime_provenance": result.provenance,
                        "configured_loras": [dict(item) for item in recipe.configured_loras],
                        "active_loras": [item.public_dict() for item in recipe.active_loras],
                    },
                )
            ]
        except asyncio.CancelledError as exc:
            RUNTIME_MANAGER.evict_runtime(runtime, clear_cache=True)
            if completed_output_owned:
                try:
                    output_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    exc.add_note(f"native Wan canceled output cleanup failed: {cleanup_error}")
            raise ToolCancelled("Generation canceled") from exc
        except BaseException as exc:
            RUNTIME_MANAGER.evict_runtime(runtime, clear_cache=True)
            if completed_output_owned:
                try:
                    output_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    exc.add_note(f"native Wan failed output cleanup failed: {cleanup_error}")
            raise
        finally:
            if not succeeded:
                RUNTIME_MANAGER.evict_runtime(runtime, clear_cache=True)

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime": "native",
            "pipeline": "NativeWanFLFRuntime",
            "model_family": "wan_2_2_i2v_14b",
            "mode": "first_last_frame_to_video",
            "conversion": False,
        }
