from __future__ import annotations

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
    ToolRequirement,
    WorkflowKind,
)
from ..runtime.kit import ResolvedRuntimePlan
from ..runtime.manager import RUNTIME_MANAGER
from ..runtime.wan22 import (
    WAN22_MAX_DURATION_SECONDS,
    WAN22_MIN_DURATION_SECONDS,
    Wan22Runtime,
    resolve_wan22_runtime_plan,
)
from ..runtime.wan22_support import wan22_runtime_support
from ..storage import StoredArtifact
from .base import (
    ExecutionCapabilities,
    ExecutionRequest,
    Tool,
    ToolContext,
)

TEXT_TO_VIDEO_ID = UUID("e2a558e6-533d-5e34-9231-f6388ef2ea20")

_WAN22_OFFLOAD_MODES = frozenset({"sequential", "group_leaf", "model"})
_WAN22_CACHE_MODES = frozenset({"none", "prompt"})


def _runtime_availability() -> tuple[bool, str | None]:
    support = wan22_runtime_support()
    return support.core_available, support.core_reason


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
                placeholder="Describe the shot, motion, camera, and visual style.",
            ),
        ),
        ToolInput(
            key="width",
            label="Width",
            type=InputType.INTEGER,
            role=InputRole.WIDTH,
            required=True,
            default=1280,
            ui=InputUi(group="Output", min=64, step=1, unit="pixels"),
        ),
        ToolInput(
            key="height",
            label="Height",
            type=InputType.INTEGER,
            role=InputRole.HEIGHT,
            required=True,
            default=704,
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
                min=WAN22_MIN_DURATION_SECONDS,
                max=WAN22_MAX_DURATION_SECONDS,
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


class Wan22TextToVideoTool(Tool):
    def model_family(self) -> str:
        return "wan22"

    def variant_base_availability(self) -> tuple[bool, str | None]:
        return _runtime_availability()

    def execution_capabilities(self) -> ExecutionCapabilities:
        support = wan22_runtime_support()
        if not support.core_available:
            return ExecutionCapabilities()

        return ExecutionCapabilities(
            model_formats=frozenset({"diffusers"}),
            lora_formats=frozenset(),
            attention_modes=frozenset({"native"}),
            offload_modes=_WAN22_OFFLOAD_MODES,
            quantization_modes=frozenset({"bf16"}),
            compile_modes=frozenset(),
            compile_fullgraph=False,
            compile_dynamic=False,
            vae_tiling_modes=frozenset({"on"}),
            vae_slicing_modes=frozenset(),
            cache_modes=_WAN22_CACHE_MODES,
            load_policy=False,
            residency_policy=True,
            runtime_parameters=False,
        )

    def validate_execution_request(self, request: ExecutionRequest) -> list[str]:
        errors = super().validate_execution_request(request)
        optimizations = request.optimizations
        quantization = str(optimizations.get("quantization", "inherit"))
        offload = str(optimizations.get("offload", "inherit"))
        attention = str(optimizations.get("attention", "inherit"))
        use_stream = bool(optimizations.get("group_offload_use_stream", False))
        record_stream = bool(optimizations.get("group_offload_record_stream", False))
        if (quantization == "inherit") != (offload == "inherit"):
            errors.append(
                "Wan 2.2 quantization and offload must either both inherit the "
                "configured profile or both be explicit"
            )
        if attention not in {"inherit", "native"}:
            errors.append(
                "Wan 2.2 recovery variants currently support only native attention"
            )
        if use_stream or record_stream:
            errors.append(
                "Wan 2.2 recovery variants disable group-offload streams to preserve "
                "the lowest predictable VRAM peak"
            )
        if offload == "model" and quantization in {"inherit", "bf16"}:
            errors.append(
                "Wan 2.2 BF16 model offload is not advertised as a 16 GB recovery "
                "path; use sequential or group_leaf"
            )
        return errors

    @property
    def descriptor(self) -> ToolDescriptor:
        available, reason = _runtime_availability()
        return ToolDescriptor(
            id=TEXT_TO_VIDEO_ID,
            key="wan22.text_to_video",
            schema_revision=2,
            name="Text to Video",
            description=(
                "Generate video with the dense Wan 2.2 TI2V-5B model in text-only mode."
            ),
            workflow_kind=WorkflowKind.TEXT_TO_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=_inputs(),
            requirements=[ToolRequirement(bundle_id="wan22-basic")],
            available=available,
            unavailable_reason=reason,
        ).with_schema_hash()

    def _resolve_plan(self, context: ToolContext) -> ResolvedRuntimePlan:
        return resolve_wan22_runtime_plan(context.settings, context.execution)

    def _runtime(
        self,
        context: ToolContext,
        plan: ResolvedRuntimePlan,
    ) -> Wan22Runtime:
        key = ("wan22_ti2v_5b", plan.pipeline_fingerprint)
        return RUNTIME_MANAGER.activate(
            key,
            lambda: Wan22Runtime(context.settings, plan),
        )

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        plan = self._resolve_plan(context)
        context.record_provenance(runtime_plan=plan.provenance())
        runtime = self._runtime(context, plan)
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")
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
                    "staged_text_encoder": metadata["staged_text_encoder"],
                    "prompt_stage": metadata["prompt_stage"],
                    "cache": metadata["cache"],
                }
            )
        finally:
            if not plan.keep_pipeline_loaded:
                unloaded = RUNTIME_MANAGER.unload_runtime(runtime)
                context.record_provenance(runtime_unloaded_after_job=unloaded)

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
            "runtime": "diffusers",
            "pipeline": "WanPipeline",
            "model_family": "wan_2_2_ti2v_5b",
            "mode": "text_to_video",
            "prompt_stage": "isolated_cpu_subprocess",
        }
