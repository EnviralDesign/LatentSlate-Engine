"""Hidden curated base for explicit native Wan 2.2 14B I2V recipe variants."""

from __future__ import annotations

import asyncio
import importlib.util
from functools import lru_cache
from typing import Any
from uuid import UUID, uuid4

from ..protocol import (
    AssetInput,
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
from ..runtime.wan22_i2v_conditioning import WAN_I2V_CANVAS
from ..storage import StoredArtifact
from ..wan22_recipe import Wan22RuntimeRequest
from .base import (
    ExecutionCapabilities,
    ExecutionRequest,
    Tool,
    ToolCancelled,
    ToolContext,
)
from .canvas import dimension_tool_inputs

NATIVE_WAN14B_I2V_ID = UUID("f2092e4f-52c8-5d65-90f1-3a8de4825df0")
NATIVE_WAN14B_I2V_KEY = "wan22.native_image_to_video"
NATIVE_WAN14B_RECIPE_TYPE = "wan22_i2v_14b"
# The exact pinned Comfy I2V-A14B workflow saves its 81 generated frames at
# 16 fps. This is inherent operation timing, not a generic Engine video default.
NATIVE_WAN14B_FPS = 16


@lru_cache(maxsize=1)
def _native_runtime_availability() -> tuple[bool, str | None]:
    """Report import prerequisites without loading the CUDA execution graph.

    Runtime construction performs the authoritative symbol and backend checks.
    Keeping catalog discovery import-free makes ordinary CLI inspection cheap.
    """

    required = (
        "torch",
        "PIL",
        "av",
        "diffusers",
        "transformers",
        "safetensors",
        "sentencepiece",
        "comfy_kitchen",
    )
    missing = [module for module in required if importlib.util.find_spec(module) is None]
    if missing:
        return False, "Run `uv sync`; missing native Wan packages: " + ", ".join(missing)
    return True, None


def _inputs() -> list[ToolInput]:
    return [
        ToolInput(
            key="source_image",
            label="Input Image",
            type=InputType.IMAGE,
            role=InputRole.SOURCE_IMAGE,
            required=True,
            ui=InputUi(group="Input"),
        ),
        ToolInput(
            key="prompt",
            label="Prompt",
            type=InputType.TEXT,
            role=InputRole.PROMPT,
            required=True,
            ui=InputUi(
                group="Prompt",
                multiline=True,
                placeholder="Describe the motion, camera, subject behavior, and shot.",
            ),
        ),
        ToolInput(
            key="negative_prompt",
            label="Negative Prompt",
            type=InputType.TEXT,
            role=InputRole.NEGATIVE_PROMPT,
            required=False,
            default="",
            ui=InputUi(group="Prompt", multiline=True, advanced=True),
        ),
        ToolInput(
            key="num_frames",
            label="Frames",
            type=InputType.INTEGER,
            role=InputRole.FRAME_COUNT,
            required=True,
            default=81,
            ui=InputUi(group="Output", min=5, max=121, step=4, unit="frames"),
        ),
        *dimension_tool_inputs(
            WAN_I2V_CANVAS,
            default_width=640,
            default_height=640,
            unit="px",
        ),
        ToolInput(
            key="steps",
            label="Steps",
            type=InputType.INTEGER,
            required=True,
            default=20,
            ui=InputUi(group="Generation", min=2, max=100, step=1),
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
        ToolInput(
            key="stage_policy",
            label="Stage Policy",
            type=InputType.CHOICE,
            required=True,
            default="expert_split",
            options=[
                ChoiceOption(
                    value="expert_split",
                    label="Expert split",
                    description="Split the requested steps evenly across high and low noise.",
                ),
                ChoiceOption(
                    value="diffusers_boundary",
                    label="Diffusers boundary",
                    description="Switch stages at the validated support-bundle boundary.",
                ),
            ],
            ui=InputUi(group="Generation", advanced=True),
        ),
        ToolInput(
            key="high_guidance",
            label="High-Noise Guidance",
            type=InputType.NUMBER,
            required=True,
            default=3.5,
            ui=InputUi(group="Guidance", advanced=True, min=0, max=20, step=0.1),
        ),
        ToolInput(
            key="low_guidance",
            label="Low-Noise Guidance",
            type=InputType.NUMBER,
            required=True,
            default=3.5,
            ui=InputUi(group="Guidance", advanced=True, min=0, max=20, step=0.1),
        ),
    ]


class NativeWan14BI2VTool(Tool):
    """Curated implementation base; only explicit recipe variants are cataloged."""

    def model_family(self) -> str:
        return "wan22"

    def variant_base_availability(self) -> tuple[bool, str | None]:
        return _native_runtime_availability()

    def execution_capabilities(self) -> ExecutionCapabilities:
        available, _ = _native_runtime_availability()
        if not available:
            return ExecutionCapabilities()
        return ExecutionCapabilities(
            recipe_types=frozenset({NATIVE_WAN14B_RECIPE_TYPE}),
            lora_formats=frozenset({"safetensors"}),
            residency_policy=True,
        )

    def validate_execution_request(self, request: ExecutionRequest) -> list[str]:
        errors = super().validate_execution_request(request)
        if request.recipe_type != NATIVE_WAN14B_RECIPE_TYPE:
            errors.append(f"native Wan 14B I2V requires recipe type {NATIVE_WAN14B_RECIPE_TYPE!r}")
        if request.model_override:
            errors.append("native Wan recipes select explicit components, not a model override")
        return errors

    @property
    def descriptor(self) -> ToolDescriptor:
        # The hidden base descriptor must stay protocol-only and cheap. Runtime
        # imports are checked only when an actual recipe variant is compiled.
        base_reason = "native Wan 14B I2V requires an explicit validated component recipe"
        return ToolDescriptor(
            id=NATIVE_WAN14B_I2V_ID,
            key=NATIVE_WAN14B_I2V_KEY,
            schema_revision=2,
            name="Native Wan 14B Image to Video",
            description=(
                "Engine-owned high/low Wan 2.2 14B I2V using exact stored artifacts."
            ),
            workflow_kind=WorkflowKind.IMAGE_TO_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=_inputs(),
            canvas=WAN_I2V_CANVAS,
            requirements=[],
            available=False,
            unavailable_reason=base_reason,
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        context.check_cancelled()
        execution = context.execution
        recipe = execution.recipe if execution is not None else None
        if not isinstance(recipe, Wan22RuntimeRequest):
            raise TypeError("native Wan I2V execution requires a validated recipe request")

        from ..runtime.wan22_i2v_runtime import WanI2VRequest
        from ..runtime.wan22_native_managed import ManagedNativeWanI2VRuntime

        asset = AssetInput.model_validate(inputs["source_image"])
        source_path = context.resolve_asset(asset.asset_id)
        context.check_cancelled()

        generation_request = WanI2VRequest(
            # The isolated exact-recipe worker decodes this identity-bound asset;
            # the parent never materializes a heavy native-Wan execution graph.
            image=None,
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
        )
        key = ("wan22_native_i2v_14b", recipe.fingerprint, context.settings.wan22_device)
        runtime = RUNTIME_MANAGER.activate(
            key,
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

        context.record_provenance(
            native_wan_recipe={
                "fingerprint": recipe.fingerprint,
                "type": NATIVE_WAN14B_RECIPE_TYPE,
                "components": recipe.public_component_manifest(),
                "operation": recipe.operation,
                "configured_loras": [dict(item) for item in recipe.configured_loras],
                "active_loras": [item.public_dict() for item in recipe.active_loras],
            }
        )
        succeeded = False
        completed_output_owned = False
        try:
            context.progress(0.02, "Loading native Wan 14B component recipe")
            result = runtime.generate(
                generation_request,
                source_image_path=source_path,
                output_path=output_path,
                device=context.settings.wan22_device,
                fps=NATIVE_WAN14B_FPS,
                progress=progress,
                cancelled=context.cancel_event.is_set,
            )
            # The managed runtime owns all in-flight cleanup.  Only after it
            # returns successfully can this tool own a final artifact for the
            # narrow late-cancellation window below.
            completed_output_owned = True
            context.check_cancelled()
            native_provenance = result.provenance
            pipeline_warm = bool(getattr(result, "pipeline_warm", False))
            context.record_provenance(
                runtime_result={
                    "runtime": "NativeWanI2VRuntimePersistentWorker",
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
                    "provenance": native_provenance,
                }
            )
            context.progress(1.0, "Complete")
            succeeded = True
            metadata: dict[str, object] = {
                **result.stream_metadata,
                "steps": generation_request.steps,
                "seed": generation_request.seed,
                "stage_policy": generation_request.stage_policy,
                "high_guidance": generation_request.high_guidance,
                "low_guidance": generation_request.low_guidance,
                "recipe_fingerprint": recipe.fingerprint,
                "components": recipe.public_component_manifest(),
                "runtime_provenance": native_provenance,
                "configured_loras": [dict(item) for item in recipe.configured_loras],
                "active_loras": [item.public_dict() for item in recipe.active_loras],
            }
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
            # A success intentionally retains the isolated, exact recipe/device
            # session for the next compatible render.
            if not succeeded:
                RUNTIME_MANAGER.evict_runtime(runtime, clear_cache=True)

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime": "native",
            "pipeline": "NativeWanI2VRuntime",
            "model_family": "wan_2_2_i2v_14b",
            "mode": "image_to_video",
            "conversion": False,
        }


def _public_runtime_provenance(provenance: Any) -> dict[str, object]:
    """Expose identities and contracts without leaking host filesystem paths."""

    return {
        "support_fingerprint": provenance.support_fingerprint,
        "tokenizer_sha256": provenance.tokenizer_sha256,
        "transformer_high_header_sha256": provenance.transformer_high_header_sha256,
        "transformer_low_header_sha256": provenance.transformer_low_header_sha256,
        "text_encoder_header_sha256": provenance.text_encoder_header_sha256,
        "vae_header_sha256": provenance.vae_header_sha256,
        "transformer_high_contract": provenance.transformer_high_contract,
        "transformer_low_contract": provenance.transformer_low_contract,
        "text_encoder_contract": provenance.text_encoder_contract,
        "stage_policy": provenance.stage_policy,
        "steps": provenance.steps,
        "seed": provenance.seed,
        "sampler": provenance.sampler,
        "scheduler": provenance.scheduler,
        "shift": provenance.shift,
        "transformer_high_size_bytes": provenance.transformer_high_size_bytes,
        "transformer_low_size_bytes": provenance.transformer_low_size_bytes,
        "text_encoder_size_bytes": provenance.text_encoder_size_bytes,
        "vae_size_bytes": provenance.vae_size_bytes,
        "transformer_high_mtime_ns": provenance.transformer_high_mtime_ns,
        "transformer_low_mtime_ns": provenance.transformer_low_mtime_ns,
        "text_encoder_mtime_ns": provenance.text_encoder_mtime_ns,
        "vae_mtime_ns": provenance.vae_mtime_ns,
    }
