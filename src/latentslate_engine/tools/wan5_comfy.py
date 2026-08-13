"""Hidden curated bases for distinct Wan 2.2 TI2V 5B Comfy operations."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ..artifacts import probe_artifact
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
from ..resources import ResourceDescriptor, ResourceFormat, ResourceKind
from ..runtime.manager import RUNTIME_MANAGER
from ..storage import StoredArtifact
from ..wan22_ti2v5b_recipe import Wan5RuntimeRequest
from .base import ExecutionCapabilities, ExecutionRequest, Tool, ToolCancelled, ToolContext

WAN5_T2V_RECIPE_TYPE = "wan22_ti2v5b_comfy_t2v"
WAN5_I2V_RECIPE_TYPE = "wan22_ti2v5b_comfy_i2v"


def _inputs(*, image: bool) -> list[ToolInput]:
    values: list[ToolInput] = []
    if image:
        values.append(
            ToolInput(
                key="source_image",
                label="Input Image",
                type=InputType.IMAGE,
                role=InputRole.SOURCE_IMAGE,
                required=True,
                ui=InputUi(group="Input"),
            )
        )
    values.extend([
            ToolInput(key="prompt", label="Prompt", type=InputType.TEXT, role=InputRole.PROMPT, required=True, ui=InputUi(group="Prompt", multiline=True)),
            ToolInput(key="negative_prompt", label="Negative Prompt", type=InputType.TEXT, role=InputRole.NEGATIVE_PROMPT, required=False, default="", ui=InputUi(group="Prompt", multiline=True, advanced=True)),
            ToolInput(key="num_frames", label="Frames", type=InputType.INTEGER, role=InputRole.FRAME_COUNT, required=True, default=121, ui=InputUi(group="Output", min=5, max=121, step=4, unit="frames")),
            ToolInput(key="width", label="Width", type=InputType.INTEGER, role=InputRole.WIDTH, required=True, default=1280, ui=InputUi(group="Output", min=64, max=1280, step=32, unit="px")),
            ToolInput(key="height", label="Height", type=InputType.INTEGER, role=InputRole.HEIGHT, required=True, default=704, ui=InputUi(group="Output", min=64, max=1280, step=32, unit="px")),
            ToolInput(key="seed", label="Seed", type=InputType.INTEGER, role=InputRole.SEED, required=True, default=0, ui=InputUi(group="Advanced", advanced=True, min=0, step=1)),
    ])
    return values


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
            lora_formats=frozenset({"safetensors"}),
            residency_policy=True,
        )

    def validate_execution_request(self, request: ExecutionRequest) -> list[str]:
        errors = super().validate_execution_request(request)
        if request.recipe_type != self.recipe_type:
            errors.append(f"Wan 5B {self.operation} requires recipe type {self.recipe_type!r}")
        if request.model_override:
            errors.append("Wan 5B Comfy recipes select exact components")
        return errors

    def validate_lora_resource(
        self,
        resource: ResourceDescriptor,
        path: Path,
    ) -> list[str]:
        if (
            resource.kind != ResourceKind.LORA
            or resource.family != "wan22"
            or resource.format != ResourceFormat.SAFETENSORS
            or resource.base_model != "Wan-AI/Wan2.2-TI2V-5B"
            or resource.metadata.get("architecture")
            != "wan22_ti2v_5b_lora_30block"
            or not resource.sources
            or not resource.sources[0].sha256
            or not isinstance(resource.metadata.get("rank"), int)
            or isinstance(resource.metadata.get("rank"), bool)
        ):
            return ["LoRA descriptor is not an exact Wan 2.2 TI2V 5B SafeTensors adapter"]
        try:
            probe = probe_artifact(path)
        except (OSError, TypeError, ValueError) as exc:
            return [f"LoRA artifact probe failed: {exc}"]
        if (
            "wan22_ti2v_5b_lora_30block" not in probe.architecture_signals
            or "lora" not in probe.component_signals
            or probe.schema_sha256 != resource.metadata.get("schema_sha256")
        ):
            return ["LoRA header does not match the exact Wan 2.2 TI2V 5B topology"]
        observed_rank = probe.key_shape_signals.get("lora_rank")
        if observed_rank != resource.metadata["rank"]:
            return [
                (
                    "LoRA declared rank does not match its header topology: declared "
                    f"{resource.metadata['rank']}, observed {observed_rank}"
                )
            ]
        return []

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
            Wan5ComfyI2VRequest,
            Wan5ComfyLora,
            Wan5ComfyRequest,
        )

        source_asset_id = None
        source_manifest = None
        request_values = {
            "prompt": str(inputs["prompt"]),
            "negative_prompt": str(inputs.get("negative_prompt") or WAN5_NEGATIVE_PROMPT),
            "num_frames": int(inputs["num_frames"]),
            "height": int(inputs["height"]),
            "width": int(inputs["width"]),
            "seed": int(inputs["seed"]),
        }
        if self.operation == "image_to_video":
            asset = AssetInput.model_validate(inputs["source_image"])
            source_asset_id = asset.asset_id
            source = context.resolve_asset(asset.asset_id)
            request = Wan5ComfyI2VRequest(source_image=source, **request_values)
            source_manifest = _source_image_manifest(
                source,
                target_width=request.width,
                target_height=request.height,
            )
        else:
            request = Wan5ComfyRequest(**request_values)
        selected_loras = context.execution.loras
        if len(selected_loras) > 1:
            raise ValueError("Wan 5B Comfy accepts at most one model-only LoRA")
        lora = None
        if selected_loras:
            selected = selected_loras[0]
            probe = probe_artifact(selected.path)
            actual_sha256 = _sha256_file(selected.path)
            expected = {
                "sha256": selected.expected_sha256,
                "schema_sha256": selected.expected_schema_sha256,
                "architecture": selected.expected_architecture,
                "rank": selected.expected_rank,
            }
            if None in expected.values():
                raise ValueError("Wan 5B LoRA requires exact hash/schema/architecture/rank metadata")
            observed_rank = probe.key_shape_signals.get("lora_rank")
            if (
                actual_sha256 != selected.expected_sha256
                or probe.schema_sha256 != selected.expected_schema_sha256
                or "wan22_ti2v_5b_lora_30block" not in probe.architecture_signals
                or selected.expected_architecture != "wan22_ti2v_5b_lora_30block"
                or "lora" not in probe.component_signals
                or observed_rank != selected.expected_rank
            ):
                raise ValueError("Wan 5B LoRA artifact does not match its exact TI2V 5B contract")
            lora = Wan5ComfyLora(
                resource_id=selected.resource_id,
                path=selected.path,
                strength=selected.strength,
                sha256=actual_sha256,
                schema_sha256=probe.schema_sha256,
                rank=int(observed_rank),
            )
        key = ("wan22_ti2v5b_comfy", recipe.component_fingerprint)
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
                recipe=recipe,
                lora=lora,
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
                        "source_image_asset_id": (
                            str(source_asset_id) if source_asset_id is not None else None
                        ),
                        "source_image": source_manifest,
                        "recipe_fingerprint": recipe.fingerprint,
                "components": recipe.public_component_manifest(),
                "lora": (
                    {
                        "resource_id": lora.resource_id,
                        "sha256": lora.sha256,
                        "schema_sha256": lora.schema_sha256,
                        "rank": lora.rank,
                        "verified_rank": lora.rank,
                        "strength": lora.strength,
                    }
                    if lora is not None
                    else None
                ),
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
            inputs=_inputs(image=False),
            available=False,
            unavailable_reason="requires an explicit validated Wan 5B component recipe",
        ).with_schema_hash()


class Wan5ComfyImageToVideoTool(_Wan5ComfyTool):
    operation = "image_to_video"
    recipe_type = WAN5_I2V_RECIPE_TYPE

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            id=UUID("316975a5-b3b1-52b5-9531-a96dc833bda0"),
            key="wan22.comfy_image_to_video",
            schema_revision=1,
            name="Wan 2.2 TI2V 5B Comfy Image to Video",
            description="Exact official split-component Comfy first-frame image-to-video graph.",
            workflow_kind=WorkflowKind.IMAGE_TO_VIDEO,
            output=ToolOutput(type=MediaType.VIDEO),
            inputs=_inputs(image=True),
            available=False,
            unavailable_reason="requires an explicit validated Wan 5B component recipe",
        ).with_schema_hash()


def _source_image_manifest(
    source: Path,
    *,
    target_width: int,
    target_height: int,
) -> dict[str, Any]:
    from PIL import Image

    with Image.open(source) as image:
        source_width, source_height = image.size
        source_mode = image.mode
        source_format = image.format
    old_aspect = source_width / source_height
    new_aspect = target_width / target_height
    crop_x = (
        round((source_width - source_width * (new_aspect / old_aspect)) / 2)
        if old_aspect > new_aspect
        else 0
    )
    crop_y = (
        round((source_height - source_height * (old_aspect / new_aspect)) / 2)
        if old_aspect < new_aspect
        else 0
    )
    return {
        "sha256": _sha256_file(source),
        "width": source_width,
        "height": source_height,
        "mode": source_mode,
        "format": source_format,
        "preprocessing": {
            "node": "Wan22ImageToVideoLatent",
            "resize": "bilinear",
            "crop": "center",
            "crop_box": [
                crop_x,
                crop_y,
                source_width - crop_x,
                source_height - crop_y,
            ],
            "target_width": target_width,
            "target_height": target_height,
            "vae_encode": True,
            "first_latent_anchor": True,
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
