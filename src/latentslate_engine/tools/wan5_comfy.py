"""Hidden curated bases for distinct Wan 2.2 TI2V 5B Comfy operations."""

from __future__ import annotations

import asyncio
import os
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
from ..wan22_ti2v5b_recipe import Wan5RuntimeRequest
from .base import ExecutionCapabilities, ExecutionRequest, Tool, ToolCancelled, ToolContext

WAN5_T2V_RECIPE_TYPE = "wan22_ti2v5b_comfy_t2v"


def _inputs() -> list[ToolInput]:
    return [
            ToolInput(key="prompt", label="Prompt", type=InputType.TEXT, role=InputRole.PROMPT, required=True, ui=InputUi(group="Prompt", multiline=True)),
            ToolInput(key="negative_prompt", label="Negative Prompt", type=InputType.TEXT, role=InputRole.NEGATIVE_PROMPT, required=False, default="", ui=InputUi(group="Prompt", multiline=True, advanced=True)),
            ToolInput(key="num_frames", label="Frames", type=InputType.INTEGER, role=InputRole.FRAME_COUNT, required=True, default=121, ui=InputUi(group="Output", min=5, max=121, step=4, unit="frames")),
            ToolInput(key="width", label="Width", type=InputType.INTEGER, role=InputRole.WIDTH, required=True, default=1280, ui=InputUi(group="Output", min=64, max=1280, step=32, unit="px")),
            ToolInput(key="height", label="Height", type=InputType.INTEGER, role=InputRole.HEIGHT, required=True, default=704, ui=InputUi(group="Output", min=64, max=1280, step=32, unit="px")),
            ToolInput(key="seed", label="Seed", type=InputType.INTEGER, role=InputRole.SEED, required=True, default=0, ui=InputUi(group="Advanced", advanced=True, min=0, step=1)),
    ]


class _Wan5ComfyTool(Tool):
    operation: str
    recipe_type: str

    def model_family(self) -> str:
        return "wan22"

    def variant_base_availability(self) -> tuple[bool, str | None]:
        root = Path(os.environ.get("LATENTSLATE_COMFYUI_ROOT", "C:/ComfyUI"))
        if not (root / "main.py").is_file() or not (root / ".venv" / "Scripts" / "python.exe").is_file():
            return False, f"Wan 5B Comfy runtime checkout is unavailable: {root}"
        return True, None

    def execution_capabilities(self) -> ExecutionCapabilities:
        return ExecutionCapabilities(
            recipe_types=frozenset({self.recipe_type}),
            residency_policy=True,
        )

    def validate_execution_request(self, request: ExecutionRequest) -> list[str]:
        errors = super().validate_execution_request(request)
        if request.recipe_type != self.recipe_type:
            errors.append(f"Wan 5B {self.operation} requires recipe type {self.recipe_type!r}")
        if request.model_override:
            errors.append("Wan 5B Comfy recipes select exact components")
        if request.loras:
            errors.append("Wan 5B Comfy LoRAs are not implemented")
        return errors

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        recipe = context.execution.recipe if context.execution is not None else None
        if not isinstance(recipe, Wan5RuntimeRequest) or recipe.operation != self.operation:
            raise TypeError(f"Wan 5B {self.operation} requires its exact recipe request")
        from ..runtime.wan5_comfy import (
            WAN5_CFG,
            WAN5_FPS,
            WAN5_NEGATIVE_PROMPT,
            WAN5_SAMPLER,
            WAN5_SCHEDULER,
            WAN5_SHIFT,
            WAN5_STEPS,
            ManagedWan5ComfyRuntime,
            Wan5ComfyRequest,
        )

        request = Wan5ComfyRequest(
            prompt=str(inputs["prompt"]),
            negative_prompt=str(inputs.get("negative_prompt") or WAN5_NEGATIVE_PROMPT),
            num_frames=int(inputs["num_frames"]),
            height=int(inputs["height"]),
            width=int(inputs["width"]),
            seed=int(inputs["seed"]),
        )
        key = ("wan22_ti2v5b_comfy", recipe.fingerprint)
        runtime = RUNTIME_MANAGER.activate(
            key,
            lambda: ManagedWan5ComfyRuntime(recipe, comfy_root=context.settings.comfyui_root),
        )
        output_path = context.storage.artifact_path(context.job_id, "output.webm")
        keep_loaded = bool((context.execution.optimizations or {}).get("keep_pipeline_loaded", True))
        context.record_provenance(
            wan5_comfy_recipe={
                "fingerprint": recipe.fingerprint,
                "type": self.recipe_type,
                "operation": self.operation,
                "components": recipe.public_component_manifest(),
            }
        )
        succeeded = False
        try:
            result = runtime.generate(
                request,
                output_path=output_path,
                progress=context.progress,
                check_cancelled=context.check_cancelled,
            )
            context.record_provenance(runtime_result=result.provenance)
            succeeded = True
            return [
                StoredArtifact(
                    id=uuid4(),
                    filename=output_path.name,
                    content_type="video/webm",
                    path=output_path,
                    role="primary",
                    media_type="video",
                    metadata={
                        "width": request.width,
                        "height": request.height,
                        "frame_count": request.num_frames,
                        "fps": WAN5_FPS,
                        "duration_seconds": request.num_frames / WAN5_FPS,
                        "has_audio": False,
                        "steps": WAN5_STEPS,
                        "guidance_scale": WAN5_CFG,
                        "sampler": WAN5_SAMPLER,
                        "scheduler": WAN5_SCHEDULER,
                        "shift": WAN5_SHIFT,
                        "seed": request.seed,
                        "operation": self.operation,
                        "source_image_asset_id": None,
                        "recipe_fingerprint": recipe.fingerprint,
                        "components": recipe.public_component_manifest(),
                        "runtime_provenance": result.provenance,
                    },
                )
            ]
        except asyncio.CancelledError as exc:
            RUNTIME_MANAGER.evict_runtime(runtime, clear_cache=True)
            raise ToolCancelled("Generation canceled") from exc
        except BaseException:
            RUNTIME_MANAGER.evict_runtime(runtime, clear_cache=True)
            raise
        finally:
            if succeeded and not keep_loaded:
                RUNTIME_MANAGER.unload_runtime(runtime)

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime": "comfyui_loopback",
            "pipeline": "official_wan22_ti2v5b_graph",
            "model_family": "wan_2_2_ti2v_5b",
            "mode": self.operation,
            "conversion": False,
        }


class Wan5ComfyTextToVideoTool(_Wan5ComfyTool):
    operation = "text_to_video"
    recipe_type = WAN5_T2V_RECIPE_TYPE

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            id=UUID("915bfad6-8756-56c5-86e8-c217ca13ea35"),
            key="wan22.comfy_text_to_video",
            schema_revision=1,
            name="Wan 2.2 TI2V 5B Comfy Text to Video",
            description="Exact official split-component Comfy text-to-video graph.",
            workflow_kind=WorkflowKind.TEXT_TO_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=_inputs(),
            available=False,
            unavailable_reason="requires an explicit validated Wan 5B component recipe",
        ).with_schema_hash()
