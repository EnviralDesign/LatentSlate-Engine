"""Minimal LatentSlate HTTP service for the public LTX, Klein, and Wan tools."""

from __future__ import annotations

import argparse
import gc
import hmac
import json
import logging
import math
import multiprocessing
import os
import queue
import shutil
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Protocol

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .authoring import compile_document, localize_document, validate_document
from .authoring_store import RecipeStore, StoreError
from .catalog import (
    RECIPE_TO_BUILTIN,
    TOOL_OPERATIONS,
    TOOLS,
    TOOLS_BY_ID,
    user_request_schema,
)
from .progress import ProgressCallback, report_progress
from .recipe import Recipe
from .validation import validate_u64
from .wan2214b.timing import native_frame_count, validate_duration_seconds

LOGGER = logging.getLogger(__name__)
PROTOCOL_VERSION = "1.0"
ENGINE_VERSION = "0.1.0"
MAX_ASSET_BYTES = 64 * 1024 * 1024
MAX_ASSET_COUNT = 128
MAX_ASSET_TOTAL_BYTES = 512 * 1024 * 1024
MAX_JOB_COUNT = 128
MAX_QUEUED_JOBS = 8


class EngineHttpError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class RuntimeBusyError(Exception):
    pass


USER_TOOL_NAMESPACE = uuid.UUID("2a3e9fab-7a47-5408-a16c-a7ba3012e25e")


def user_tool_id(recipe_id: str) -> str:
    """Keep user tool identity stable across recipe revisions and hosts."""
    tool_id = str(uuid.uuid5(USER_TOOL_NAMESPACE, recipe_id))
    if tool_id in TOOLS_BY_ID:
        raise EngineHttpError(409, "User tool identity collides with a built-in tool")
    return tool_id


def user_tool_schema(publication: dict, *, resolve_artifact=None) -> dict:
    """Project an immutable user revision using its existing family contract."""
    record = publication["record"]
    document = record["document"]
    template = TOOLS_BY_ID[RECIPE_TO_BUILTIN[document["operation"]]]
    validation = validate_document(document, resolve_artifact=resolve_artifact)
    tool_id = user_tool_id(document["id"])
    result = {
        **deepcopy(template),
        **user_request_schema(document),
        "id": tool_id,
        "key": f"user_recipe.{document['id']}",
        "name": document["name"],
        "schema_revision": publication["schema"]["revision"],
        "schema_hash": publication["schema"]["hash"],
        "recipe": {
            "id": document["id"],
            "revision": record["revision"],
            "definition_hash": record["definition_hash"],
        },
        "available": validation["recipe_compiles"]
        and validation["artifact_resolution"]["status"] == "resolved",
    }
    if not result["available"]:
        result["unavailable_reason"] = "; ".join(
            item["message"] for item in validation["issues"]
        )
    return result


@dataclass(frozen=True)
class LtxModelPaths:
    dev_checkpoint: Path
    distilled_checkpoint: Path
    text_checkpoint: Path
    transformer_lora: Path
    upsampler: Path

    @classmethod
    def from_home(cls, home: Path) -> LtxModelPaths:
        models = home / "models" / "ltx23"
        return cls(
            dev_checkpoint=models / "checkpoints" / "ltx-2.3-22b-dev-fp8.safetensors",
            distilled_checkpoint=(
                models / "checkpoints" / "ltx-2.3-22b-distilled-fp8.safetensors"
            ),
            text_checkpoint=(
                models / "text_encoders" / "gemma_3_12B_it_fp4_mixed.safetensors"
            ),
            transformer_lora=(
                home
                / "loras"
                / "ltx23"
                / "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
            ),
            upsampler=(
                models
                / "latent_upscalers"
                / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
            ),
        )

    def available(self) -> bool:
        return all(path.is_file() for path in self.__dict__.values())


@dataclass(frozen=True)
class KleinModelPaths:
    diffusion: Path
    text_encoder: Path
    vae: Path
    tokenizer: Path

    @classmethod
    def from_home(
        cls, home: Path, *, vae_override: Path | None = None
    ) -> KleinModelPaths:
        models = home / "models" / "klein9b"
        support = models / "support" / "bfl-distilled-pipeline-support"
        return cls(
            diffusion=models / "transformers" / "flux-2-klein-9b-fp8.safetensors",
            text_encoder=models / "text_encoders" / "qwen_3_8b_fp8mixed.safetensors",
            vae=vae_override
            or models / "vae" / "full_encoder_small_decoder.safetensors",
            tokenizer=support / "tokenizer",
        )

    def available(self) -> bool:
        tokenizer_files = (
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        )
        return (
            self.diffusion.is_file()
            and self.text_encoder.is_file()
            and self.vae.is_file()
            and self.tokenizer.is_dir()
            and all((self.tokenizer / name).is_file() for name in tokenizer_files)
            and (self.tokenizer.parent / "text_encoder" / "config.json").is_file()
        )


@dataclass(frozen=True)
class WanModelPaths:
    t2v_high_checkpoint: Path
    t2v_high_lora: Path
    t2v_low_checkpoint: Path
    t2v_low_lora: Path
    i2v_high_checkpoint: Path
    i2v_high_lora: Path
    i2v_low_checkpoint: Path
    i2v_low_lora: Path
    text_encoder: Path
    vae: Path

    @classmethod
    def from_root(cls, root: Path) -> WanModelPaths:
        diffusion = root / "diffusion_models" / "wan22"
        loras = root / "loras" / "wan"
        return cls(
            t2v_high_checkpoint=(
                diffusion / "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
            ),
            t2v_high_lora=(
                loras / "wan2.2_t2v_lightx2v_4steps_lora_v1_1_high_noise.safetensors"
            ),
            t2v_low_checkpoint=(
                diffusion / "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
            ),
            t2v_low_lora=(
                loras / "wan2.2_t2v_lightx2v_4steps_lora_v1_1_low_noise.safetensors"
            ),
            i2v_high_checkpoint=(
                diffusion / "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
            ),
            i2v_high_lora=(
                loras / "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
            ),
            i2v_low_checkpoint=(
                diffusion / "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
            ),
            i2v_low_lora=(
                loras / "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
            ),
            text_encoder=(
                root
                / "text_encoders"
                / "wan"
                / "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
            ),
            vae=root / "vae" / "wan" / "wan_2.1_vae.safetensors",
        )

    def available(self, operation: str) -> bool:
        shared = (self.text_encoder, self.vae)
        if operation == "wan_t2v":
            required = (
                self.t2v_high_checkpoint,
                self.t2v_high_lora,
                self.t2v_low_checkpoint,
                self.t2v_low_lora,
                *shared,
            )
        elif operation in {"wan_i2v", "wan_flf"}:
            required = (
                self.i2v_high_checkpoint,
                self.i2v_high_lora,
                self.i2v_low_checkpoint,
                self.i2v_low_lora,
                *shared,
            )
        else:
            raise ValueError("Unsupported Wan operation")
        return all(path.is_file() for path in required)


def _ltx_builtin_recipe(paths: LtxModelPaths, operation: str) -> Recipe:
    from .ltx23.recipes import ltx23_flf_recipe, ltx23_i2v_recipe, ltx23_t2v_recipe
    from .recipe import Adapter, Artifact

    if operation == "flf":
        return ltx23_flf_recipe(
            checkpoint=paths.distilled_checkpoint,
            text_checkpoint=paths.text_checkpoint,
            device_index=0,
        )
    factory = ltx23_t2v_recipe if operation == "t2v" else ltx23_i2v_recipe
    return factory(
        checkpoint=paths.dev_checkpoint,
        text_checkpoint=paths.text_checkpoint,
        upsampler=paths.upsampler,
        transformer_adapters=(Adapter(Artifact(paths.transformer_lora), 0.5),),
        device_index=0,
    )


def _ltx_model_identity(operation: str, definition: Recipe, inputs: dict | None = None):
    """Use the native family's portable identity for the worker lifetime boundary."""
    from .ltx23.recipes import (
        resolve_ltx23_flf,
        resolve_ltx23_flf_identity,
        resolve_ltx23_i2v,
        resolve_ltx23_i2v_identity,
        resolve_ltx23_t2v,
        resolve_ltx23_t2v_identity,
    )

    resolve, fixed_identity = {
        "t2v": (resolve_ltx23_t2v, resolve_ltx23_t2v_identity),
        "i2v": (resolve_ltx23_i2v, resolve_ltx23_i2v_identity),
        "flf": (resolve_ltx23_flf, resolve_ltx23_flf_identity),
    }[operation]
    if inputs is None:
        return fixed_identity(definition)
    return resolve(
        definition, {item["key"]: inputs[item["key"]] for item in definition.surface()}
    )[0]


class _LtxOperationRuntime:
    """Run one fixed LTX operation identity inside its GPU process."""

    def __init__(
        self,
        paths: LtxModelPaths,
        operation: str,
        recipe: Recipe | None = None,
        inputs: dict | None = None,
    ) -> None:
        self.paths = paths
        self.operation = operation
        self.runtime = self._create_runtime(operation, recipe, inputs)

    def generate(
        self,
        inputs: dict[str, Any],
        output_path: Path,
        progress: ProgressCallback | None = None,
        *,
        recipe: Recipe | None = None,
    ) -> None:
        media = self._generate_media(inputs, progress, recipe)
        report_progress(progress, 0.95, "Artifact encoding")
        media.save_mp4(output_path)
        report_progress(progress, 1.0, "Artifact encoding", stage_progress=1.0)

    def _create_runtime(
        self, operation: str, recipe: Recipe | None = None, inputs: dict | None = None
    ) -> Any:
        builtin = _ltx_builtin_recipe(self.paths, operation)
        identity = _ltx_model_identity(
            operation, recipe or builtin, inputs if recipe is not None else None
        )
        if operation == "t2v":
            from .ltx23.t2v import Ltx23T2VRuntime

            self._t2v_recipe = builtin
            return Ltx23T2VRuntime(identity)
        if operation == "i2v":
            from .ltx23.i2v import Ltx23I2VRuntime

            self._i2v_recipe = builtin
            return Ltx23I2VRuntime(identity)
        if operation == "flf":
            from .ltx23.flf import Ltx23FlfRuntime

            self._flf_recipe = builtin
            return Ltx23FlfRuntime(identity)
        raise ValueError("Unsupported LTX operation")

    def _generate_media(
        self,
        inputs: dict[str, Any],
        progress: ProgressCallback | None,
        recipe: Recipe | None = None,
    ) -> Any:
        if self.operation == "t2v":
            from .ltx23.recipes import resolve_ltx23_t2v

            definition = recipe or self._t2v_recipe
            _, request = resolve_ltx23_t2v(
                definition,
                {field["key"]: inputs[field["key"]] for field in definition.surface()},
            )
            return self.runtime.generate(**request, progress=progress)
        if self.operation == "i2v":
            from .ltx23.recipes import resolve_ltx23_i2v

            definition = recipe or self._i2v_recipe
            _, request = resolve_ltx23_i2v(
                definition,
                {field["key"]: inputs[field["key"]] for field in definition.surface()},
            )
            return self.runtime.generate(**request, progress=progress)
        from .ltx23.recipes import resolve_ltx23_flf

        definition = recipe or self._flf_recipe
        _, request = resolve_ltx23_flf(
            definition,
            {field["key"]: inputs[field["key"]] for field in definition.surface()},
        )
        return self.runtime.generate(**request, progress=progress)

    def close(self) -> None:
        runtime = self.runtime
        self.runtime = None
        if runtime is not None:
            try:
                runtime.close()
            finally:
                del runtime
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _ltx_worker_main(
    operation: str, paths: LtxModelPaths, connection: Connection
) -> None:
    runtime: _LtxOperationRuntime | None = None
    try:
        while True:
            message = connection.recv()
            if message["type"] == "close":
                return
            try:
                definition = (
                    compile_document(message["recipe"])
                    if message.get("recipe")
                    else None
                )
                if runtime is None:
                    runtime = _LtxOperationRuntime(
                        paths, operation, definition, message["inputs"]
                    )
                runtime.generate(
                    message["inputs"],
                    message["output_path"],
                    lambda event: connection.send({"type": "progress", "event": event}),
                    recipe=definition,
                )
            except Exception as error:
                LOGGER.exception("LTX worker generation failed")
                connection.send(
                    {
                        "type": "result",
                        "ok": False,
                        "error_type": type(error).__name__,
                    }
                )
                return
            connection.send({"type": "result", "ok": True})
    finally:
        if runtime is not None:
            runtime.close()
        connection.close()


def _klein_worker_main(paths: KleinModelPaths, connection: Connection) -> None:
    runtime: Any = None
    try:
        from .klein9b.recipes import (
            klein9b_t2i_recipe,
            klein9b_two_image_explicit_recipe,
            resolve_klein9b_fixed_identity,
            resolve_klein9b_t2i_request,
            resolve_klein9b_two_image_request,
        )
        from .klein9b.two_image import Klein9BTwoImageRuntime

        bindings = {
            "diffusion": paths.diffusion,
            "text_encoder": paths.text_encoder,
            "vae": paths.vae,
            "tokenizer": paths.tokenizer,
        }
        t2i_recipe = klein9b_t2i_recipe(**bindings)
        two_image_recipe = klein9b_two_image_explicit_recipe(**bindings)
        runtime = Klein9BTwoImageRuntime()
        while True:
            message = connection.recv()
            if message["type"] == "close":
                return
            try:
                operation = message["operation"]
                inputs = message["inputs"]
                output_path = message["output_path"]
                definition = (
                    compile_document(message["recipe"])
                    if message.get("recipe")
                    else t2i_recipe
                    if operation == "klein_t2i"
                    else two_image_recipe
                )
                identity = resolve_klein9b_fixed_identity(definition)
                inputs = {
                    item["key"]: inputs[item["key"]] for item in definition.surface()
                }
                if operation == "klein_t2i":
                    request = resolve_klein9b_t2i_request(definition, inputs)
                    result = runtime.generate(
                        identity=identity,
                        **request,
                        output=output_path,
                        progress=lambda event: connection.send(
                            {"type": "progress", "event": event}
                        ),
                    )
                    details = {
                        "conditioning_reused": result.conditioning_reused,
                        "models_reused": result.models_reused,
                    }
                elif operation == "klein_two_image":
                    request = resolve_klein9b_two_image_request(definition, inputs)
                    result = runtime.generate_two_image(
                        identity=identity,
                        **request,
                        output=output_path,
                        progress=lambda event: connection.send(
                            {"type": "progress", "event": event}
                        ),
                    )
                    details = {
                        "conditioning_reused": result.conditioning_reused,
                        "models_reused": result.models_reused,
                        "reference_reused": result.reference_reused,
                    }
                else:
                    raise ValueError("Unsupported Klein operation")
            except Exception as error:
                LOGGER.exception("Klein worker generation failed")
                connection.send(
                    {
                        "type": "result",
                        "ok": False,
                        "error_type": type(error).__name__,
                    }
                )
                return
            connection.send({"type": "result", "ok": True, "details": details})
    finally:
        if runtime is not None:
            runtime.close()
        connection.close()


class _WanFamilyRuntime:
    """Own exactly one current Wan operation session inside one family process."""

    def __init__(self, paths: WanModelPaths) -> None:
        self.paths = paths
        self.operation: str | None = None
        self.session: Any = None

    def generate(
        self,
        operation: str,
        inputs: dict[str, Any],
        output_path: Path,
        progress: ProgressCallback | None = None,
        *,
        recipe: Recipe | None = None,
    ) -> dict[str, Any]:
        session_reused = self.operation == operation and self.session is not None
        if not session_reused:
            self._close_session()
            self.session = self._create_session(operation, inputs, recipe)
            self.operation = operation
        previous_session = self.session
        if operation == "wan_t2v":
            from .wan2214b.recipes import resolve_wan2214b_t2v

            definition = recipe or self._t2v_recipe
            native_recipe, request = resolve_wan2214b_t2v(
                definition,
                {field["key"]: inputs[field["key"]] for field in definition.surface()},
            )
        elif operation == "wan_i2v":
            from .wan2214b.recipes import resolve_wan2214b_i2v

            definition = recipe or self._i2v_recipe
            native_recipe, request = resolve_wan2214b_i2v(
                definition,
                {field["key"]: inputs[field["key"]] for field in definition.surface()},
            )
        elif operation == "wan_flf":
            from .wan2214b.recipes import resolve_wan2214b_flf

            definition = recipe or self._flf_recipe
            native_recipe, request = resolve_wan2214b_flf(
                definition,
                {field["key"]: inputs[field["key"]] for field in definition.surface()},
            )
        else:
            raise ValueError("Unsupported Wan operation")
        self.session = self.session.replaced(native_recipe)
        details = self._reuse_details(operation, inputs)
        details["session_reused"] = session_reused and self.session is previous_session
        result = self.session.generate(
            output_path=output_path, **request, progress=progress
        )
        details["timings"] = result.timings
        return details

    def _create_session(
        self, operation: str, inputs: dict[str, Any], definition: Recipe | None = None
    ) -> Any:
        if operation == "wan_t2v":
            from .recipe import Adapter, Artifact
            from .wan2214b.pipeline import NEGATIVE_PROMPT, WanSession
            from .wan2214b.recipes import resolve_wan2214b_t2v, wan2214b_t2v_recipe

            self._t2v_recipe = wan2214b_t2v_recipe(
                high_checkpoint=self.paths.t2v_high_checkpoint,
                high_adapters=(
                    Adapter(Artifact(self.paths.t2v_high_lora), 1.0000000000000002),
                ),
                low_checkpoint=self.paths.t2v_low_checkpoint,
                low_adapters=(
                    Adapter(Artifact(self.paths.t2v_low_lora), 1.0000000000000002),
                ),
                text_encoder=self.paths.text_encoder,
                vae=self.paths.vae,
                negative_prompt=NEGATIVE_PROMPT,
            )
            recipe, _ = resolve_wan2214b_t2v(
                definition or self._t2v_recipe,
                {
                    field["key"]: inputs[field["key"]]
                    for field in (definition or self._t2v_recipe).surface()
                },
            )
            return WanSession(recipe)
        if operation == "wan_i2v":
            from .recipe import Adapter, Artifact
            from .wan2214b.i2v import NEGATIVE_PROMPT, WanI2VSession
            from .wan2214b.recipes import resolve_wan2214b_i2v, wan2214b_i2v_recipe

            self._i2v_recipe = wan2214b_i2v_recipe(
                high_checkpoint=self.paths.i2v_high_checkpoint,
                high_adapters=(
                    Adapter(Artifact(self.paths.i2v_high_lora), 1.0000000000000002),
                ),
                low_checkpoint=self.paths.i2v_low_checkpoint,
                low_adapters=(
                    Adapter(Artifact(self.paths.i2v_low_lora), 1.0000000000000002),
                ),
                text_encoder=self.paths.text_encoder,
                vae=self.paths.vae,
                negative_prompt=NEGATIVE_PROMPT,
            )
            recipe, _ = resolve_wan2214b_i2v(
                definition or self._i2v_recipe,
                {
                    field["key"]: inputs[field["key"]]
                    for field in (definition or self._i2v_recipe).surface()
                },
            )
            return WanI2VSession(recipe)
        if operation == "wan_flf":
            from .recipe import Adapter, Artifact
            from .wan2214b.flf import NEGATIVE_PROMPT, WanFLFSession
            from .wan2214b.recipes import resolve_wan2214b_flf, wan2214b_flf_recipe

            self._flf_recipe = wan2214b_flf_recipe(
                high_checkpoint=self.paths.i2v_high_checkpoint,
                high_adapters=(Adapter(Artifact(self.paths.i2v_high_lora), 1.0),),
                low_checkpoint=self.paths.i2v_low_checkpoint,
                low_adapters=(Adapter(Artifact(self.paths.i2v_low_lora), 1.0),),
                text_encoder=self.paths.text_encoder,
                vae=self.paths.vae,
                negative_prompt=NEGATIVE_PROMPT,
            )
            recipe, _ = resolve_wan2214b_flf(
                definition or self._flf_recipe,
                {
                    field["key"]: inputs[field["key"]]
                    for field in (definition or self._flf_recipe).surface()
                },
            )
            return WanFLFSession(recipe)
        raise ValueError("Unsupported Wan operation")

    def _reuse_details(self, operation: str, inputs: dict[str, Any]) -> dict[str, Any]:
        prompt_key = (inputs["prompt"], self.session.recipe.negative)
        details = {
            "conditioning_reused": (
                self.session._conditioning is not None
                and self.session._conditioning_key == prompt_key
            )
        }
        if operation == "wan_i2v":
            from .identity import FileContentIdentity
            from .wan2214b.i2v import ImageConditioningIdentity

            identity = ImageConditioningIdentity(
                source=FileContentIdentity.from_path(inputs["start_image"]),
                width=inputs["width"],
                height=inputs["height"],
                frame_count=native_frame_count(inputs["duration_seconds"]),
            )
            details["image_conditioning_reused"] = (
                self.session._image_conditioning is not None
                and self.session._image_conditioning.identity == identity
            )
        elif operation == "wan_flf":
            from .identity import FileContentIdentity
            from .wan2214b.flf import OrderedSourceIdentity

            identity = OrderedSourceIdentity(
                first=FileContentIdentity.from_path(inputs["start_image"]),
                last=FileContentIdentity.from_path(inputs["end_image"]),
                width=inputs["width"],
                height=inputs["height"],
                frame_count=native_frame_count(inputs["duration_seconds"]),
            )
            details["image_conditioning_reused"] = (
                self.session._flf_conditioning is not None
                and self.session._flf_conditioning.identity == identity
            )
        return details

    def close(self) -> None:
        self._close_session()

    def _close_session(self) -> None:
        session = self.session
        self.session = None
        self.operation = None
        if session is not None:
            try:
                session.destroy()
            finally:
                del session
        gc.collect()


def _wan_worker_main(paths: WanModelPaths, connection: Connection) -> None:
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    runtime: _WanFamilyRuntime | None = None
    try:
        runtime = _WanFamilyRuntime(paths)
        while True:
            message = connection.recv()
            if message["type"] == "close":
                return
            try:
                details = runtime.generate(
                    message["operation"],
                    message["inputs"],
                    message["output_path"],
                    lambda event: connection.send({"type": "progress", "event": event}),
                    recipe=compile_document(message["recipe"])
                    if message.get("recipe")
                    else None,
                )
            except Exception as error:
                LOGGER.exception("Wan worker generation failed")
                connection.send(
                    {
                        "type": "result",
                        "ok": False,
                        "error_type": type(error).__name__,
                    }
                )
                return
            connection.send({"type": "result", "ok": True, "details": details})
    finally:
        if runtime is not None:
            runtime.close()
        connection.close()


def _operation_family(operation: str) -> str:
    if operation in {"t2v", "i2v", "flf"}:
        return "ltx"
    if operation in {"klein_t2i", "klein_two_image"}:
        return "klein"
    if operation in {"wan_t2v", "wan_i2v", "wan_flf"}:
        return "wan"
    raise ValueError("Unsupported public operation")


class ActiveRuntimeOwner:
    """Own the single active GPU-family process and its earned reuse boundary."""

    def __init__(
        self,
        ltx_paths: LtxModelPaths,
        klein_paths: KleinModelPaths,
        wan_paths: WanModelPaths,
    ) -> None:
        self.ltx_paths = ltx_paths
        self.klein_paths = klein_paths
        self.wan_paths = wan_paths
        self._availability = {
            "t2v": ltx_paths.available(),
            "i2v": ltx_paths.available(),
            "flf": ltx_paths.available(),
            "klein_t2i": klein_paths.available(),
            "klein_two_image": klein_paths.available(),
            "wan_t2v": wan_paths.available("wan_t2v"),
            "wan_i2v": wan_paths.available("wan_i2v"),
            "wan_flf": wan_paths.available("wan_flf"),
        }
        self._lock = threading.Lock()
        self._family: str | None = None
        self._worker_operation: str | None = None
        self._ltx_identity = None
        self._last_operation: str | None = None
        self._last_generation: dict[str, Any] | None = None
        self._process: multiprocessing.Process | None = None
        self._connection: Connection | None = None
        self._generation_count = 0
        self._reuse_count = 0
        self._switch_count = 0
        self._release_count = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "family": self._family,
            "operation": self._last_operation,
            "worker_pid": self._process.pid if self._process is not None else None,
            "generation_count": self._generation_count,
            "reuse_count": self._reuse_count,
            "switch_count": self._switch_count,
            "release_count": self._release_count,
            "last_generation": self._last_generation,
        }

    def available(self, operation: str) -> bool:
        return self._availability[operation]

    def unavailable_reason(self, operation: str) -> str:
        family = _operation_family(operation)
        label = {"ltx": "LTX", "klein": "Klein", "wan": "Wan"}[family]
        return f"Required {label} model files are not installed."

    def generate(
        self,
        operation: str,
        inputs: dict[str, Any],
        output_path: Path,
        progress: ProgressCallback | None = None,
        *,
        recipe: dict | None = None,
    ) -> None:
        with self._lock:
            family = _operation_family(operation)
            if recipe is None and not self.available(operation):
                raise RuntimeError(self.unavailable_reason(operation))
            ltx_identity = None
            if family == "ltx":
                definition = (
                    compile_document(recipe)
                    if recipe is not None
                    else _ltx_builtin_recipe(self.ltx_paths, operation)
                )
                ltx_identity = _ltx_model_identity(
                    operation, definition, inputs if recipe is not None else None
                )
            same_worker = (
                self._family == family
                and self._process is not None
                and self._process.is_alive()
                and (family in {"klein", "wan"} or self._worker_operation == operation)
                and (family != "ltx" or self._ltx_identity == ltx_identity)
            )
            if same_worker:
                self._reuse_count += 1
                report_progress(progress, 0.0, "Preparing runtime")
            else:
                report_progress(progress, 0.0, "Loading runtime")
                if self._process is not None:
                    previous_family = self._family
                    previous_operation = self._worker_operation
                    previous_identity = self._ltx_identity
                    self._stop_worker()
                    if (
                        previous_family != family
                        or previous_operation != operation
                        or previous_identity != ltx_identity
                    ):
                        self._switch_count += 1
                self._start_worker(family, operation)
                self._ltx_identity = ltx_identity
            connection = self._connection
            if connection is None:
                raise RuntimeError("GPU worker did not start")
            try:
                connection.send(
                    {
                        "type": "generate",
                        "operation": operation,
                        "inputs": inputs,
                        "output_path": output_path,
                        "recipe": recipe,
                    }
                )
                while True:
                    result = connection.recv()
                    if result.get("type") != "progress":
                        break
                    if progress is not None:
                        progress(result["event"])
            except (BrokenPipeError, EOFError, OSError) as error:
                self._stop_worker()
                raise RuntimeError("GPU worker stopped unexpectedly") from error
            if not result.get("ok"):
                error_type = result.get("error_type", "NativeError")
                self._stop_worker()
                raise RuntimeError(f"GPU worker failed ({error_type})")
            self._generation_count += 1
            self._last_operation = operation
            self._last_generation = result.get("details")

    def release(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeBusyError("The active runtime is still executing a job")
        try:
            self._stop_worker()
            self._release_count += 1
        finally:
            self._lock.release()

    def _start_worker(self, family: str, operation: str) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        if family == "ltx":
            process = context.Process(
                target=_ltx_worker_main,
                args=(operation, self.ltx_paths, child),
                name=f"latentslate-ltx-{operation}",
                daemon=True,
            )
            worker_operation = operation
        elif family == "klein":
            process = context.Process(
                target=_klein_worker_main,
                args=(self.klein_paths, child),
                name="latentslate-klein9b",
                daemon=True,
            )
            worker_operation = None
        else:
            process = context.Process(
                target=_wan_worker_main,
                args=(self.wan_paths, child),
                name="latentslate-wan2214b",
                daemon=True,
            )
            worker_operation = None
        process.start()
        child.close()
        self._connection = parent
        self._process = process
        self._family = family
        self._worker_operation = worker_operation
        self._last_operation = None
        self._last_generation = None

    def _stop_worker(self) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        self._family = None
        self._worker_operation = None
        self._ltx_identity = None
        self._last_operation = None
        self._last_generation = None
        if process is None:
            return
        if process.is_alive() and connection is not None:
            try:
                connection.send({"type": "close"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        if connection is not None:
            connection.close()


class RuntimeExecutor(Protocol):
    def available(self, operation: str) -> bool: ...

    def unavailable_reason(self, operation: str) -> str: ...

    def generate(
        self,
        operation: str,
        inputs: dict[str, Any],
        output_path: Path,
        progress: ProgressCallback | None = None,
        *,
        recipe: dict | None = None,
    ) -> None: ...

    def release(self) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AssetRecord:
    id: uuid.UUID
    path: Path
    size: int


@dataclass
class JobRecord:
    id: uuid.UUID
    operation: str
    inputs: dict[str, Any]
    output_path: Path
    asset_ids: frozenset[uuid.UUID] = field(default_factory=frozenset)
    status: str = "queued"
    progress: float = 0.0
    stage: dict[str, Any] | None = None
    message: str = "Waiting for GPU"
    cancel_requested: bool = False
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, str] | None = None
    recipe: dict | None = None
    provenance: dict[str, Any] | None = None

    def public(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": str(self.id),
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "artifacts": list(self.artifacts),
        }
        if self.error is not None:
            result["error"] = dict(self.error)
        if self.stage is not None:
            result["stage"] = dict(self.stage)
        if self.provenance is not None:
            result.update(deepcopy(self.provenance))
        return result


class EngineService:
    def __init__(self, root: Path, executor: RuntimeExecutor) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self._temp = tempfile.TemporaryDirectory(prefix="session-", dir=root)
        self.session_root = Path(self._temp.name)
        self.asset_root = self.session_root / "assets"
        self.job_root = self.session_root / "jobs"
        self.asset_root.mkdir()
        self.job_root.mkdir()
        self.executor = executor
        self.authoring: RecipeStore | None = None
        self.materializer = None
        self._lock = threading.Lock()
        self._assets: dict[uuid.UUID, AssetRecord] = {}
        self._asset_bytes = 0
        self._jobs: dict[uuid.UUID, JobRecord] = {}
        self._closing = False
        self._pending: queue.Queue[uuid.UUID | None] = queue.Queue(MAX_QUEUED_JOBS)
        self._worker = threading.Thread(
            target=self._work, name="latentslate-gpu", daemon=True
        )
        self._worker.start()

    def store_asset(self, upload: UploadFile) -> AssetRecord:
        with self._lock:
            if self._closing:
                raise EngineHttpError(503, "The Engine service is shutting down")
            if len(self._assets) >= MAX_ASSET_COUNT:
                raise EngineHttpError(507, "The temporary asset limit has been reached")
        asset_id = uuid.uuid4()
        suffix = _safe_suffix(upload.filename)
        path = self.asset_root / f"{asset_id.hex}{suffix}"
        size = 0
        try:
            with path.open("xb") as destination:
                while chunk := upload.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_ASSET_BYTES:
                        raise EngineHttpError(413, "The uploaded asset is too large")
                    destination.write(chunk)
            if size == 0:
                raise EngineHttpError(422, "The uploaded asset is empty")
            with self._lock:
                if self._closing:
                    raise EngineHttpError(503, "The Engine service is shutting down")
                if len(self._assets) >= MAX_ASSET_COUNT:
                    raise EngineHttpError(
                        507, "The temporary asset limit has been reached"
                    )
                if self._asset_bytes + size > MAX_ASSET_TOTAL_BYTES:
                    raise EngineHttpError(
                        507, "The temporary asset byte limit has been reached"
                    )
                record = AssetRecord(asset_id, path, size)
                self._assets[asset_id] = record
                self._asset_bytes += size
                return record
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def submit(self, body: dict[str, Any]) -> JobRecord:
        with self._lock:
            if self._closing:
                raise EngineHttpError(503, "The Engine service is shutting down")
            recipe = provenance = None
            if not isinstance(body.get("tool_id"), str):
                raise EngineHttpError(422, "tool_id must be a UUID string")
            if body.get("tool_id") not in TOOLS_BY_ID and self.authoring is not None:
                operation, inputs, asset_ids, recipe, provenance = (
                    self._validate_user_job(body)
                )
            else:
                operation, inputs, asset_ids = self._validate_job(body)
            if len(self._jobs) >= MAX_JOB_COUNT:
                self._reclaim_oldest_terminal_job_locked()
            if len(self._jobs) >= MAX_JOB_COUNT:
                raise EngineHttpError(503, "The in-memory job limit has been reached")
            job_id = uuid.uuid4()
            directory = self.job_root / job_id.hex
            directory.mkdir()
            output_filename = (
                "output.png" if operation.startswith("klein_") else "output.mp4"
            )
            job = JobRecord(
                job_id,
                operation,
                inputs,
                directory / output_filename,
                asset_ids=asset_ids,
                recipe=recipe,
                provenance=provenance,
            )
            self._jobs[job_id] = job
            try:
                self._pending.put_nowait(job_id)
            except queue.Full as error:
                del self._jobs[job_id]
                shutil.rmtree(directory)
                raise EngineHttpError(503, "The GPU job queue is full") from error
            return job

    def get_job(self, job_id: uuid.UUID) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise EngineHttpError(404, "Job not found")
            return job

    def cancel(self, job_id: uuid.UUID) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise EngineHttpError(404, "Job not found")
            if job.status == "queued":
                job.cancel_requested = True
                job.status = "canceled"
                job.message = "Canceled"
                job.stage = {"label": "Canceled"}
                self._reclaim_job_assets_locked(job)
            elif job.status == "running":
                job.cancel_requested = True
                job.message = "Cancellation requested"
            return job

    def artifact(self, job_id: uuid.UUID, filename: str) -> Path:
        with self._lock:
            job = self._jobs.get(job_id)
            if (
                job is None
                or job.status != "succeeded"
                or filename != job.output_path.name
                or not job.output_path.is_file()
            ):
                raise EngineHttpError(404, "Artifact not found")
            return job.output_path

    def release_runtime(self) -> dict[str, Any]:
        try:
            self.executor.release()
        except RuntimeBusyError as error:
            raise EngineHttpError(
                409, "The active runtime is executing a job"
            ) from error
        return {"released": True, "runtime": self.executor.snapshot()}

    def close(self) -> None:
        with self._lock:
            self._closing = True
            for job in self._jobs.values():
                if job.status == "queued":
                    job.cancel_requested = True
                    job.status = "canceled"
                    job.message = "Canceled"
                    job.stage = {"label": "Canceled"}
                elif job.status == "running":
                    job.cancel_requested = True
                    job.message = "Cancellation requested"
            for job in self._jobs.values():
                if job.status == "canceled":
                    self._reclaim_job_assets_locked(job)
        self._pending.put(None)
        self._worker.join()
        try:
            self.executor.release()
        except RuntimeBusyError:
            pass
        self._temp.cleanup()

    def _work(self) -> None:
        while True:
            job_id = self._pending.get()
            if job_id is None:
                return
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                if job.status == "canceled":
                    continue
                job.status = "running"
                job.message = "Generating media"
                job.progress = 0.0
            try:

                def update_progress(
                    event: dict[str, Any], target_job: JobRecord = job
                ) -> None:
                    with self._lock:
                        if target_job.status != "running":
                            return
                        event_progress = event.get("progress")
                        if isinstance(event_progress, (int, float)):
                            target_job.progress = float(event_progress)
                        stage = event.get("stage")
                        if isinstance(stage, dict):
                            target_job.stage = dict(stage)

                self.executor.generate(
                    job.operation,
                    job.inputs,
                    job.output_path,
                    update_progress,
                    **(
                        {
                            "recipe": localize_document(
                                job.recipe,
                                lambda reference: self.materializer.resolve(
                                    reference, verify=True
                                ),
                            )
                            if self.materializer
                            else job.recipe
                        }
                        if job.recipe is not None
                        else {}
                    ),
                )
            except Exception as error:  # noqa: BLE001 - native failures end the job
                LOGGER.error("Engine job %s failed (%s)", job.id, type(error).__name__)
                job.output_path.unlink(missing_ok=True)
                with self._lock:
                    if job.cancel_requested:
                        job.status = "canceled"
                        job.message = "Canceled"
                        job.stage = {"label": "Canceled"}
                    else:
                        job.status = "failed"
                        job.message = "Generation failed"
                        job.stage = {"label": "Failed"}
                        job.error = {"message": "Engine generation failed."}
                    job.progress = 1.0
                    self._reclaim_job_assets_locked(job)
                continue
            with self._lock:
                if job.cancel_requested:
                    job.output_path.unlink(missing_ok=True)
                    job.status = "canceled"
                    job.message = "Canceled"
                    job.stage = {"label": "Canceled"}
                    job.progress = 1.0
                    self._reclaim_job_assets_locked(job)
                    continue
                job.status = "succeeded"
                job.message = "Complete"
                job.progress = 1.0
                job.stage = {"label": "Complete", "progress": 1.0}
                job.artifacts = [
                    {
                        "role": "primary",
                        "filename": job.output_path.name,
                        "download_url": (
                            f"/v1/artifacts/{job.id}/{job.output_path.name}"
                        ),
                    }
                ]
                self._reclaim_job_assets_locked(job)

    def _reclaim_oldest_terminal_job_locked(self) -> None:
        terminal = next(
            (
                (job_id, job)
                for job_id, job in self._jobs.items()
                if job.status not in {"queued", "running"}
            ),
            None,
        )
        if terminal is None:
            return
        job_id, job = terminal
        shutil.rmtree(job.output_path.parent)
        del self._jobs[job_id]

    def _reclaim_job_assets_locked(self, job: JobRecord) -> None:
        for asset_id in job.asset_ids:
            if any(
                other.status in {"queued", "running"} and asset_id in other.asset_ids
                for other in self._jobs.values()
            ):
                continue
            asset = self._assets.pop(asset_id, None)
            if asset is not None:
                self._asset_bytes -= asset.size
                asset.path.unlink(missing_ok=True)

    def _validate_user_job(self, body: dict[str, Any]):
        assert self.authoring is not None
        publication = None
        for record in self.authoring.list():
            recipe_id = record["document"]["id"]
            if user_tool_id(recipe_id) == body.get("tool_id"):
                try:
                    publication = self.authoring.publication(recipe_id)
                except StoreError as error:
                    if error.status != 404:
                        raise
                break
        if publication is None or not publication["enabled"]:
            raise EngineHttpError(422, "Unknown or disabled tool_id")
        record = publication["record"]
        resolve_artifact = self.materializer.resolve if self.materializer else None
        tool = user_tool_schema(publication, resolve_artifact=resolve_artifact)
        if (
            type(body.get("schema_revision")) is not int
            or not isinstance(body.get("recipe"), dict)
            or type(body["recipe"].get("revision")) is not int
            or body["schema_revision"] != tool["schema_revision"]
            or body.get("schema_hash") != tool["schema_hash"]
            or body.get("recipe") != tool["recipe"]
        ):
            raise EngineHttpError(
                409, "Recipe or request schema changed; refresh the catalog"
            )
        if not tool["available"]:
            raise EngineHttpError(503, tool["unavailable_reason"])
        raw = body.get("inputs")
        if not isinstance(raw, dict):
            raise EngineHttpError(422, "inputs must be an object")
        definition = compile_document(
            localize_document(record["document"], resolve_artifact)
            if resolve_artifact
            else record["document"]
        )
        surface = definition.surface()
        inputs = dict(raw)
        assets = []
        for item in surface:
            if item["type"] == "image" and item["key"] in inputs:
                asset = self._resolve_asset(inputs[item["key"]], None)
                inputs[item["key"]] = asset.path
                assets.append(asset)
        try:
            resolved = definition.resolve(inputs)
        except (TypeError, ValueError) as error:
            raise EngineHttpError(422, str(error)) from error
        operation = TOOL_OPERATIONS[RECIPE_TO_BUILTIN[record["document"]["operation"]]]
        if operation in {"i2v", "flf"}:
            from PIL import Image

            for asset in assets:
                with Image.open(asset.path) as image:
                    if image.size != (resolved["width"], resolved["height"]):
                        raise EngineHttpError(
                            422,
                            "Uploaded images must match the requested canvas dimensions",
                        )
        # Defaults are captured with the accepted revision, including hidden
        # geometry used for family media preparation. Native resolvers receive
        # only the recipe's caller surface.
        provenance = {
            "tool_id": tool["id"],
            "schema_revision": tool["schema_revision"],
            "schema_hash": tool["schema_hash"],
            "recipe": deepcopy(tool["recipe"]),
        }
        return (
            operation,
            resolved,
            frozenset(asset.id for asset in assets),
            deepcopy(record["document"]),
            provenance,
        )

    def _validate_job(
        self, body: dict[str, Any]
    ) -> tuple[str, dict[str, Any], frozenset[uuid.UUID]]:
        tool_id = body.get("tool_id")
        tool = TOOLS_BY_ID.get(tool_id)
        if tool is None:
            raise EngineHttpError(422, "Unknown tool_id")
        operation = TOOL_OPERATIONS[tool_id]
        if not self.executor.available(operation):
            raise EngineHttpError(503, self.executor.unavailable_reason(operation))
        if body.get("schema_revision") != tool["schema_revision"]:
            raise EngineHttpError(409, "The tool schema revision is stale")
        if body.get("schema_hash") != tool["schema_hash"]:
            raise EngineHttpError(409, "The tool schema hash is stale")
        raw_inputs = body.get("inputs")
        if not isinstance(raw_inputs, dict):
            raise EngineHttpError(422, "inputs must be an object")
        expected = {item["key"] for item in tool["inputs"]}
        if set(raw_inputs) - expected:
            raise EngineHttpError(422, "The request contains unknown inputs")
        missing = {
            item["key"]
            for item in tool["inputs"]
            if item.get("required") and item["key"] not in raw_inputs
        }
        if missing:
            raise EngineHttpError(422, "The request is missing required inputs")
        inputs = dict(raw_inputs)
        prompt = inputs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise EngineHttpError(422, "prompt must be non-empty text")
        for key in ("width", "height", "seed"):
            if isinstance(inputs.get(key), bool) or not isinstance(
                inputs.get(key), int
            ):
                raise EngineHttpError(422, f"{key} must be an integer")
        if operation in {"t2v", "i2v", "flf"}:
            duration = inputs.get("duration_seconds")
            if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                raise EngineHttpError(422, "duration_seconds must be numeric")
            alignment = 32 if operation == "flf" else 64
            try:
                _validate_ltx_product_request(
                    inputs["width"],
                    inputs["height"],
                    duration,
                    inputs["seed"],
                    alignment=alignment,
                )
            except (TypeError, ValueError) as error:
                raise EngineHttpError(422, str(error)) from error
            inputs["duration_seconds"] = float(duration)
        elif operation in {"klein_t2i", "klein_two_image"}:
            try:
                _validate_klein_product_request(
                    inputs["width"], inputs["height"], inputs["seed"]
                )
            except (TypeError, ValueError) as error:
                raise EngineHttpError(422, str(error)) from error
        else:
            duration = inputs.get("duration_seconds")
            try:
                _validate_wan_product_request(
                    inputs["width"],
                    inputs["height"],
                    duration,
                    inputs["seed"],
                )
            except (TypeError, ValueError) as error:
                raise EngineHttpError(422, str(error)) from error
            inputs["duration_seconds"] = float(duration)
            inputs["frame_count"] = native_frame_count(duration)
        asset_ids = set()
        for key in ("start_image", "end_image", "image_1", "image_2"):
            if key in expected:
                expected_size = (
                    (inputs["width"], inputs["height"])
                    if operation in {"i2v", "flf"}
                    else None
                )
                asset = self._resolve_asset(inputs.get(key), expected_size)
                inputs[key] = asset.path
                asset_ids.add(asset.id)
        return operation, inputs, frozenset(asset_ids)

    def _resolve_asset(
        self, value: Any, expected_size: tuple[int, int] | None
    ) -> AssetRecord:
        if not isinstance(value, dict) or value.get("type") != "asset":
            raise EngineHttpError(422, "Media inputs must reference an uploaded asset")
        try:
            asset_id = uuid.UUID(str(value.get("asset_id")))
        except (TypeError, ValueError, AttributeError) as error:
            raise EngineHttpError(422, "Media asset_id must be a UUID") from error
        asset = self._assets.get(asset_id)
        if asset is None:
            raise EngineHttpError(422, "Media asset was not found")
        try:
            from PIL import Image

            with Image.open(asset.path) as image:
                if expected_size is not None and image.size != expected_size:
                    raise EngineHttpError(
                        422,
                        "Uploaded images must match the requested canvas dimensions",
                    )
                image.verify()
        except EngineHttpError:
            raise
        except Exception as error:
            raise EngineHttpError(422, "Uploaded media is not a valid image") from error
        return asset


def _validate_ltx_product_request(
    width: int,
    height: int,
    duration_seconds: float,
    seed: int,
    *,
    alignment: int,
) -> None:
    if width < 64 or height < 64:
        raise ValueError("LTX width and height must each be at least 64")
    if width % alignment or height % alignment:
        raise ValueError(f"LTX width and height must each be divisible by {alignment}")
    if width * height > 942_080:
        raise ValueError("LTX width * height must not exceed 942080")
    duration = float(duration_seconds)
    if not math.isfinite(duration):
        raise ValueError("LTX duration_seconds must be finite")
    if not 1.0 <= duration <= 10.0:
        raise ValueError("LTX duration_seconds must be between 1.0 and 10.0")
    if not math.isclose(duration * 2.0, round(duration * 2.0), abs_tol=1e-9):
        raise ValueError("LTX duration_seconds must use 0.5-second increments")
    if seed < 0 or seed > (1 << 64) - 1:
        raise ValueError("LTX seed must be between 0 and 18446744073709551615")


def _validate_klein_product_request(width: int, height: int, seed: int) -> None:
    if width % 16 or height % 16:
        raise ValueError("Klein width and height must each be divisible by 16")
    if width < 256 or height < 256:
        raise ValueError("Klein width and height must each be at least 256")
    if width * height > 1_048_576:
        raise ValueError("Klein width * height must not exceed 1048576")
    if max(width, height) > min(width, height) * 4:
        raise ValueError("Klein aspect ratio must not exceed 4:1")
    validate_u64(seed, label="Klein seed")


def _validate_wan_product_request(
    width: int, height: int, duration_seconds: float, seed: int
) -> None:
    if width % 16 or height % 16:
        raise ValueError("Wan width and height must each be divisible by 16")
    if width < 480 or height < 480:
        raise ValueError("Wan width and height must each be at least 480")
    if width * height > 921_600:
        raise ValueError("Wan width * height must not exceed 921600")
    if max(width, height) * 9 > min(width, height) * 16:
        raise ValueError("Wan aspect ratio must not exceed 16:9")
    validate_duration_seconds(duration_seconds)
    validate_u64(seed, label="Wan seed")


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if 1 < len(suffix) <= 10 and suffix[1:].isalnum():
        return suffix
    return ".asset"


def create_app(
    *,
    home: Path | None = None,
    token: str | None = None,
    executor: RuntimeExecutor | None = None,
) -> FastAPI:
    repository_root = Path(__file__).resolve().parents[2]
    load_dotenv(repository_root / ".env")
    configured_home = os.environ.get("LATENTSLATE_ENGINE_HOME", "").strip()
    engine_home = home or (
        Path(configured_home)
        if configured_home
        else repository_root / "LatentSlateEngineData"
    )
    if not engine_home.is_absolute():
        engine_home = repository_root / engine_home
    configured_klein_vae = os.environ.get("LATENTSLATE_KLEIN9B_VAE", "").strip()
    klein_vae = Path(configured_klein_vae) if configured_klein_vae else None
    if klein_vae is not None and not klein_vae.is_absolute():
        klein_vae = engine_home / klein_vae
    configured_wan_root = os.environ.get("LATENTSLATE_WAN_MODEL_ROOT", "").strip()
    wan_root = (
        Path(configured_wan_root)
        if configured_wan_root
        else engine_home / "models" / "wan2214b"
    )
    if not wan_root.is_absolute():
        wan_root = engine_home / wan_root
    ltx_paths = LtxModelPaths.from_home(engine_home)
    klein_paths = KleinModelPaths.from_home(engine_home, vae_override=klein_vae)
    wan_paths = WanModelPaths.from_root(wan_root)
    runtime = executor or ActiveRuntimeOwner(ltx_paths, klein_paths, wan_paths)
    service = EngineService(engine_home / "runtime" / "http", runtime)
    auth_token = (
        token if token is not None else os.environ.get("LATENTSLATE_ENGINE_TOKEN", "")
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        materializer.close()
        service.close()

    app = FastAPI(title="LatentSlate Engine", lifespan=lifespan)
    app.state.engine_service = service

    from .artifact_library import ArtifactLibrary
    from .artifact_materialization import ArtifactMaterializer
    from .authoring_api import authoring_error, authoring_router
    from .authoring_builtins import builtin_documents
    from .authoring_store import RecipeStore, StoreError

    builtins = builtin_documents(ltx_paths, klein_paths, wan_paths)
    materializer = ArtifactMaterializer(engine_home / "artifacts")
    service.materializer = materializer
    app.state.artifact_materializer = materializer
    authoring = RecipeStore(
        engine_home / "authoring" / "recipes",
        builtin_ids=(document["id"] for document in builtins.values()),
        resolve_artifact=materializer.resolve,
    )
    authoring.reconcile_schema_lineage()
    service.authoring = authoring
    library = ArtifactLibrary(engine_home / "authoring" / "roots.json")
    app.include_router(
        authoring_router(
            authoring,
            builtins,
            library,
            lambda publication: user_tool_schema(
                publication, resolve_artifact=materializer.resolve
            ),
            materializer,
        )
    )
    app.add_exception_handler(StoreError, authoring_error)

    from fastapi.staticfiles import StaticFiles

    web_directory = Path(__file__).parent / "web"
    app.mount(
        "/authoring/assets",
        StaticFiles(directory=web_directory),
        name="authoring-assets",
    )

    @app.get("/authoring", include_in_schema=False)
    @app.get("/authoring/", include_in_schema=False)
    def authoring_ui():
        return FileResponse(web_directory / "index.html", media_type="text/html")

    @app.middleware("http")
    async def bearer_auth(request: Request, call_next):
        if auth_token and request.url.path.startswith("/v1/"):
            authorization = request.headers.get("authorization", "")
            expected = f"Bearer {auth_token}"
            if not hmac.compare_digest(authorization, expected):
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Bearer authentication required"}},
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)

    @app.exception_handler(EngineHttpError)
    async def engine_error(_: Request, error: EngineHttpError):
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"message": error.message}},
        )

    @app.get("/v1/health")
    def health():
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "engine_version": ENGINE_VERSION,
            "runtime": runtime.snapshot(),
        }

    @app.get("/v1/catalog")
    def catalog():
        tools = []
        for tool in TOOLS:
            operation = TOOL_OPERATIONS[tool["id"]]
            available = runtime.available(operation)
            public = {**tool, "available": available}
            if not available:
                public["unavailable_reason"] = runtime.unavailable_reason(operation)
            tools.append(public)
        for record in authoring.list():
            try:
                publication = authoring.publication(record["document"]["id"])
            except StoreError as error:
                if error.status != 404:
                    raise
                continue
            if publication["enabled"]:
                tools.append(
                    user_tool_schema(publication, resolve_artifact=materializer.resolve)
                )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "engine_version": ENGINE_VERSION,
            "bundles": [],
            "tools": tools,
        }

    @app.post("/v1/assets")
    def upload_asset(file: UploadFile):
        asset = service.store_asset(file)
        return {"id": str(asset.id)}

    @app.post("/v1/jobs")
    async def submit_job(request: Request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise EngineHttpError(400, "Request body must be valid JSON") from error
        if not isinstance(body, dict):
            raise EngineHttpError(422, "Request body must be an object")
        return service.submit(body).public()

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: uuid.UUID):
        return service.get_job(job_id).public()

    @app.delete("/v1/jobs/{job_id}")
    def cancel_job(job_id: uuid.UUID):
        return service.cancel(job_id).public()

    @app.get("/v1/artifacts/{job_id}/{filename}")
    def download_artifact(job_id: uuid.UUID, filename: str):
        path = service.artifact(job_id, filename)
        media_type = "image/png" if path.suffix.lower() == ".png" else "video/mp4"
        return FileResponse(
            path,
            media_type=media_type,
            filename=path.name,
        )

    @app.delete("/v1/runtime")
    def release_runtime():
        return service.release_runtime()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve LatentSlate Engine")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(), host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
