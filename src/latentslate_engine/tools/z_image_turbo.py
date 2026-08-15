"""Public tool for the single exact Z-Image Turbo T2I recipe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ..protocol import (
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
from ..z_image_turbo_recipe import (
    ZImageTurboRuntimeRequest,
    revalidate_z_image_turbo_runtime_request,
)
from .base import ExecutionCapabilities, ExecutionRequest, Tool, ToolContext

Z_IMAGE_TURBO_ID = UUID("966f6431-5e34-5cab-9b79-8efed1652fca")
Z_IMAGE_TURBO_KEY = "zimage.turbo_text_to_image"
Z_IMAGE_TURBO_RECIPE_TYPE = "z_image_turbo_t2i"


class ZImageTurboTextToImageTool(Tool):
    """Managed, positive-only exact Turbo T2I; not yet recommendation-promoted."""

    def model_family(self) -> str:
        return "zimage"

    def variant_base_availability(self) -> tuple[bool, str | None]:
        # Deliberately lightweight: no parent CUDA probe or torch/model import.
        if sys.platform != "win32":
            return False, "Z-Image Turbo managed worker is currently Windows-only"
        if importlib.util.find_spec("comfy_kitchen") is None:
            return False, "Z-Image Turbo requires the installed comfy-kitchen package"
        return True, None

    def execution_capabilities(self) -> ExecutionCapabilities:
        return ExecutionCapabilities(
            recipe_types=frozenset({Z_IMAGE_TURBO_RECIPE_TYPE}), residency_policy=True
        )

    def validate_execution_request(self, request: ExecutionRequest) -> list[str]:
        errors = super().validate_execution_request(request)
        if request.recipe_type != Z_IMAGE_TURBO_RECIPE_TYPE:
            errors.append("Z-Image Turbo requires the exact official Turbo T2I component recipe")
        if request.model_override:
            errors.append(
                "Z-Image Turbo selects immutable explicit components, not a model override"
            )
        if request.loras:
            errors.append("Z-Image Turbo does not accept LoRAs")
        return errors

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = self.variant_base_availability()
        return ToolDescriptor(
            id=Z_IMAGE_TURBO_ID,
            key=Z_IMAGE_TURBO_KEY,
            schema_revision=1,
            name="Z-Image Turbo Text to Image",
            description="Exact managed-worker INT8 ConvRot Turbo T2I contract (hardware acceptance pending).",
            workflow_kind=WorkflowKind.TEXT_TO_IMAGE,
            output=ToolOutput(type=MediaType.IMAGE),
            inputs=[
                ToolInput(
                    key="prompt",
                    label="Prompt",
                    type=InputType.TEXT,
                    role=InputRole.PROMPT,
                    required=True,
                    ui=InputUi(group="Prompt", multiline=True),
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
            ],
            requirements=[],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        context.check_cancelled()
        recipe = context.execution.recipe if context.execution is not None else None
        if not isinstance(
            recipe, ZImageTurboRuntimeRequest
        ) or not revalidate_z_image_turbo_runtime_request(recipe):
            raise ValueError("Z-Image execution requires a revalidated immutable Turbo T2I request")
        available, reason = self.variant_base_availability()
        if not available:
            raise RuntimeError(reason or "Z-Image Turbo managed runtime is unavailable")
        from ..runtime.z_image_turbo_managed import ManagedZImageTurboRuntime

        output_path = Path(context.storage.artifact_path(context.job_id, "output.png"))
        key = (
            "z-image-turbo",
            recipe.fingerprint,
            recipe.components["transformer"]["header_sha256"],
            "worker-current-indexed-cuda",
            "bfloat16",
            "basic-guider/auraflow-shift3/simple/res-multistep/cpu-fp32-noise",
        )
        runtime = RUNTIME_MANAGER.activate(key, lambda: ManagedZImageTurboRuntime(recipe))
        keep_pipeline_loaded = bool(
            (context.execution.optimizations or {}).get("keep_pipeline_loaded", True)
            if context.execution is not None
            else True
        )
        context.record_provenance(
            runtime_plan={
                "runtime": "engine-native/z-image-turbo-persistent-worker",
                "request_fingerprint": recipe.fingerprint,
                "components": recipe.public_component_manifest(),
                "requested_device": str(context.settings.wan22_device),
                "device_resolution": "worker-current-indexed-cuda",
                "execution": "basic-guider/auraflow-shift3/simple/res-multistep/cpu-fp32-noise",
            }
        )
        try:
            result = runtime.generate(
                prompt=str(inputs["prompt"]),
                seed=int(inputs["seed"]),
                output_path=output_path,
                device=str(context.settings.wan22_device),
                progress=context.progress,
                check_cancelled=context.check_cancelled,
            )
            context.check_cancelled()
        except BaseException:
            RUNTIME_MANAGER.evict_runtime(runtime, clear_cache=True)
            output_path.unlink(missing_ok=True)
            raise
        if not keep_pipeline_loaded:
            context.record_provenance(
                runtime_unloaded_after_job=RUNTIME_MANAGER.unload_runtime(runtime)
            )
        # Capture after the requested eviction.  Reporting the pre-unload
        # worker PID as current status would make a released process look live.
        status = runtime.status()
        context.record_provenance(
            runtime_result={
                **result.metadata,
                "worker_pid": result.worker_pid,
                "pipeline_warm": result.pipeline_warm,
                "runtime_status": {
                    "loaded": status["loaded"],
                    "worker_pid": status["worker_pid"],
                    "cleanup_errors": status["cleanup_errors"],
                },
            }
        )
        return [
            StoredArtifact(
                id=uuid4(),
                filename=output_path.name,
                content_type="image/png",
                path=output_path,
                role="primary",
                media_type="image",
                metadata=result.metadata,
            )
        ]

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime": "native",
            "pipeline": "ZImageTurboNative",
            "model_family": "z_image_turbo",
            "mode": "text_to_image",
            "conversion": False,
            "fallback": "forbidden",
            "managed_worker": True,
        }
