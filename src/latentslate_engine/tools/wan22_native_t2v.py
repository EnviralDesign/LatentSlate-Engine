"""Hidden curated base for explicit native Wan 2.2 14B T2V recipes."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID, uuid4

from ..protocol import (
    ChoiceOption,
    InputRole,
    InputType,
    InputUi,
    MediaType,
    ToolDescriptor,
    ToolInput,
    ToolOutput,
    WorkflowKind,
)
from ..runtime.manager import RUNTIME_MANAGER
from ..storage import StoredArtifact
from ..wan22_recipe import Wan22RuntimeRequest
from .base import ExecutionCapabilities, ExecutionRequest, Tool, ToolCancelled, ToolContext
from .wan22_native import NATIVE_WAN14B_FPS, _native_runtime_availability

NATIVE_WAN14B_T2V_ID = UUID("ab75a3c3-0c01-5a41-bb10-b11aa3d0d12e")
NATIVE_WAN14B_T2V_KEY = "wan22.native_text_to_video"
NATIVE_WAN14B_T2V_RECIPE_TYPE = "wan22_t2v_14b"


def _inputs() -> list[ToolInput]:
    return [
        ToolInput(key="prompt", label="Prompt", type=InputType.TEXT, role=InputRole.PROMPT, required=True, ui=InputUi(group="Prompt", multiline=True)),
        ToolInput(key="negative_prompt", label="Negative Prompt", type=InputType.TEXT, role=InputRole.NEGATIVE_PROMPT, required=False, default="", ui=InputUi(group="Prompt", multiline=True, advanced=True)),
        ToolInput(key="num_frames", label="Frames", type=InputType.INTEGER, role=InputRole.FRAME_COUNT, required=True, default=81, ui=InputUi(group="Output", min=5, max=121, step=4, unit="frames")),
        ToolInput(key="width", label="Width", type=InputType.INTEGER, role=InputRole.WIDTH, required=True, default=640, ui=InputUi(group="Output", min=64, max=1280, step=16, unit="px")),
        ToolInput(key="height", label="Height", type=InputType.INTEGER, role=InputRole.HEIGHT, required=True, default=640, ui=InputUi(group="Output", min=64, max=1280, step=16, unit="px")),
        ToolInput(key="steps", label="Steps", type=InputType.INTEGER, required=True, default=20, ui=InputUi(group="Generation", min=2, max=100, step=1)),
        ToolInput(key="seed", label="Seed", type=InputType.INTEGER, role=InputRole.SEED, required=True, default=0, ui=InputUi(group="Advanced", advanced=True, min=0, step=1)),
        ToolInput(key="stage_policy", label="Stage Policy", type=InputType.CHOICE, required=True, default="expert_split", options=[ChoiceOption(value="expert_split", label="Expert split", description="Split the requested steps evenly across high and low noise.")], ui=InputUi(group="Generation", advanced=True)),
        ToolInput(key="high_guidance", label="High-Noise Guidance", type=InputType.NUMBER, required=True, default=3.5, ui=InputUi(group="Guidance", advanced=True, min=0, max=20, step=0.1)),
        ToolInput(key="low_guidance", label="Low-Noise Guidance", type=InputType.NUMBER, required=True, default=3.5, ui=InputUi(group="Guidance", advanced=True, min=0, max=20, step=0.1)),
    ]


class NativeWan14BT2VTool(Tool):
    def model_family(self) -> str:
        return "wan22"

    def variant_base_availability(self) -> tuple[bool, str | None]:
        return _native_runtime_availability()

    def execution_capabilities(self) -> ExecutionCapabilities:
        available, _ = _native_runtime_availability()
        return ExecutionCapabilities(recipe_types=frozenset({NATIVE_WAN14B_T2V_RECIPE_TYPE}), lora_formats=frozenset({"safetensors"}), residency_policy=True) if available else ExecutionCapabilities()

    def validate_execution_request(self, request: ExecutionRequest) -> list[str]:
        errors = super().validate_execution_request(request)
        if request.recipe_type != NATIVE_WAN14B_T2V_RECIPE_TYPE:
            errors.append(f"native Wan 14B T2V requires recipe type {NATIVE_WAN14B_T2V_RECIPE_TYPE!r}")
        if request.model_override:
            errors.append("native Wan recipes select explicit components, not a model override")
        return errors

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(id=NATIVE_WAN14B_T2V_ID, key=NATIVE_WAN14B_T2V_KEY, schema_revision=2, name="Native Wan 14B Text to Video", description="Engine-owned high/low Wan 2.2 14B T2V using exact stored artifacts.", workflow_kind=WorkflowKind.TEXT_TO_VIDEO, output=ToolOutput(type=MediaType.VIDEO), inputs=_inputs(), requirements=[], available=False, unavailable_reason="native Wan 14B T2V requires an explicit validated component recipe").with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        context.check_cancelled()
        recipe = context.execution.recipe if context.execution is not None else None
        if not isinstance(recipe, Wan22RuntimeRequest) or not recipe.operation.startswith("wan22_t2v_"):
            raise TypeError("native Wan T2V execution requires a validated T2V recipe request")
        from ..runtime.wan22_native_managed import ManagedNativeWanI2VRuntime
        from ..runtime.wan22_t2v_runtime import WanT2VRequest

        generation = WanT2VRequest(prompt=str(inputs["prompt"]), negative_prompt=str(inputs.get("negative_prompt") or ""), num_frames=int(inputs["num_frames"]), height=int(inputs["height"]), width=int(inputs["width"]), steps=int(inputs["steps"]), seed=int(inputs["seed"]), stage_policy=str(inputs["stage_policy"]), high_guidance=float(inputs["high_guidance"]), low_guidance=float(inputs["low_guidance"]))
        runtime = RUNTIME_MANAGER.activate(("wan22_native_t2v_14b", recipe.fingerprint, context.settings.wan22_device), lambda: ManagedNativeWanI2VRuntime(recipe))
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")

        def progress(completed: int, total: int, stage: str) -> None:
            context.check_cancelled()
            context.progress(0.12 + 0.76 * completed / max(1, total), f"Generating native Wan video ({completed}/{total}, {stage})")

        context.record_provenance(
            native_wan_recipe={
                "fingerprint": recipe.fingerprint,
                "type": NATIVE_WAN14B_T2V_RECIPE_TYPE,
                "components": recipe.public_component_manifest(),
                "operation": recipe.operation,
                "configured_loras": [dict(item) for item in recipe.configured_loras],
                "active_loras": [item.public_dict() for item in recipe.active_loras],
            }
        )
        succeeded = False
        completed_output_owned = False
        try:
            result = runtime.generate(generation, source_image_path=None, output_path=output_path, device=context.settings.wan22_device, fps=NATIVE_WAN14B_FPS, progress=progress, cancelled=context.cancel_event.is_set)
            completed_output_owned = True
            context.check_cancelled()
            pipeline_warm = bool(getattr(result, "pipeline_warm", False))
            context.record_provenance(runtime_result={"runtime": "NativeWanT2VRuntimePersistentWorker", "recipe_fingerprint": recipe.fingerprint, "pipeline_warm": pipeline_warm, "execution_cache": {"supported": True, "hit": pipeline_warm, "mode": "exact_recipe_persistent_worker"}, "worker": {"pid": result.worker_pid, "exit_code": result.worker_exit_code, "terminated": False, "memory_boundary": "persistent_exact_recipe_worker"}, "provenance": result.provenance})
            context.progress(1.0, "Complete")
            succeeded = True
            return [StoredArtifact(id=uuid4(), filename=output_path.name, content_type="video/mp4", path=output_path, role="primary", media_type="video", metadata={**result.stream_metadata, "steps": generation.steps, "seed": generation.seed, "stage_policy": generation.stage_policy, "high_guidance": generation.high_guidance, "low_guidance": generation.low_guidance, "recipe_fingerprint": recipe.fingerprint, "components": recipe.public_component_manifest(), "runtime_provenance": result.provenance, "configured_loras": [dict(item) for item in recipe.configured_loras], "active_loras": [item.public_dict() for item in recipe.active_loras]})]
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
        return {"runtime": "native", "pipeline": "NativeWanT2VRuntime", "model_family": "wan_2_2_t2v_14b", "mode": "text_to_video", "conversion": False}
