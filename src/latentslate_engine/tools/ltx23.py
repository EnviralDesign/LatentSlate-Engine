from __future__ import annotations

import importlib.util
from pathlib import Path
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
    LTX23_SIZE_PRESETS,
    LTX23Runtime,
    resolve_ltx23_runtime_plan,
)
from ..runtime.manager import RUNTIME_MANAGER
from ..storage import StoredArtifact
from .base import ExecutionCapabilities, Tool, ToolContext

TEXT_TO_VIDEO_ID = UUID("46bdb57c-3b19-5397-8949-4e20ffe757c9")


def _runtime_availability() -> tuple[bool, str | None]:
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
            key="size",
            label="Size",
            type=InputType.CHOICE,
            required=True,
            default="768x512",
            options=[ChoiceOption(value=value, label=value) for value in LTX23_SIZE_PRESETS],
            ui=InputUi(group="Output"),
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
    def model_family(self) -> str:
        return "ltx23"

    def variant_base_availability(self) -> tuple[bool, str | None]:
        return _runtime_availability()

    def execution_capabilities(self) -> ExecutionCapabilities:
        available, _reason = _runtime_availability()
        if not available:
            return ExecutionCapabilities()
        return ExecutionCapabilities(
            # LTX2Pipeline.from_pretrained consumes a complete repository.  A
            # standalone SafeTensors or GGUF resource is not a substitutable
            # pipeline, even when its name contains LTX.
            model_formats=frozenset({"diffusers"}),
            lora_formats=frozenset(),
            attention_modes=frozenset({"native"}),
            offload_modes=frozenset({"sequential", "model", "none"}),
            quantization_modes=frozenset({"bf16"}),
            compile_modes=frozenset(),
            vae_tiling_modes=frozenset({"on"}),
            vae_slicing_modes=frozenset(),
            cache_modes=frozenset({"none", "prompt"}),
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
            schema_revision=1,
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
    ) -> LTX23Runtime:
        key = ("ltx23", plan.pipeline_fingerprint)
        return RUNTIME_MANAGER.activate(
            key,
            lambda: LTX23Runtime(context.settings, plan),
        )

    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        plan = self._resolve_plan(context)
        context.record_provenance(runtime_plan=plan.provenance())
        output_path = context.storage.artifact_path(context.job_id, "output.mp4")
        runtime = self._runtime(context, plan)
        try:
            metadata = runtime.generate(
                plan=plan,
                prompt=str(inputs["prompt"]),
                output_path=output_path,
                size_name=str(inputs["size"]),
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
            "pipeline": "LTX2Pipeline",
            "model_family": "ltx_2_3",
            "artifact_contract": "complete_diffusers_bf16_native",
        }
