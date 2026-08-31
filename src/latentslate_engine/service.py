"""Minimal LatentSlate HTTP service for the three public LTX 2.3 operations."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Protocol

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

LOGGER = logging.getLogger(__name__)
PROTOCOL_VERSION = "1.0"
ENGINE_VERSION = "0.1.0"
MAX_ASSET_BYTES = 64 * 1024 * 1024
MAX_ASSET_COUNT = 128
MAX_ASSET_TOTAL_BYTES = 512 * 1024 * 1024
MAX_JOB_COUNT = 128
MAX_QUEUED_JOBS = 8

T2V_ID = "46bdb57c-3b19-5397-8949-4e20ffe757c9"
I2V_ID = "5d6e2d6f-216c-5f35-a4ec-1565d6e56ee7"
FLF_ID = "1a8f9c0b-410e-56e4-90de-23bcb9d644ca"


class EngineHttpError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class RuntimeBusyError(Exception):
    pass


def _input(
    key: str,
    label: str,
    input_type: str,
    *,
    required: bool = True,
    default: Any = None,
    role: str | None = None,
    ui: dict[str, Any] | None = None,
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "key": key,
        "label": label,
        "type": input_type,
        "required": required,
    }
    if default is not None:
        descriptor["default"] = default
    if role is not None:
        descriptor["role"] = role
    if ui is not None:
        descriptor["ui"] = ui
    return descriptor


def _tool_schema(
    tool_id: str,
    key: str,
    name: str,
    workflow_kind: str,
    alignment: int,
    media_inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": tool_id,
        "key": key,
        "schema_revision": 2,
        "name": name,
        "description": "Generate LTX 2.3 video with synchronized audio.",
        "workflow_kind": workflow_kind,
        "output": {"type": "video"},
        "inputs": [
            _input(
                "prompt",
                "Prompt",
                "text",
                ui={"multiline": True, "placeholder": "Describe the shot"},
            ),
            *media_inputs,
            _input(
                "width",
                "Width",
                "integer",
                default=512,
                role="width",
                ui={"min": 64, "step": alignment},
            ),
            _input(
                "height",
                "Height",
                "integer",
                default=512,
                role="height",
                ui={"min": 64, "step": alignment},
            ),
            _input(
                "duration_seconds",
                "Duration",
                "number",
                default=5.0,
                role="duration_seconds",
                ui={"min": 1.0, "max": 10.0, "step": 0.5, "unit": "seconds"},
            ),
            _input("seed", "Seed", "integer", default=0, role="seed"),
        ],
        "canvas": {
            "alignment": alignment,
            "min_side": 64,
            "max_pixels": 942_080,
        },
    }


def _schema_hash(schema: dict[str, Any]) -> str:
    encoded = json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _tool_definitions() -> list[dict[str, Any]]:
    image = _input("start_image", "Start Image", "image", role="start_image")
    first = _input("start_image", "First Frame", "image", role="start_image")
    last = _input("end_image", "Last Frame", "image", role="end_image")
    schemas = [
        _tool_schema(
            T2V_ID,
            "ltx23.text_to_video",
            "LTX 2.3 Text to Video",
            "text_to_video",
            64,
            [],
        ),
        _tool_schema(
            I2V_ID,
            "ltx23.image_to_video",
            "LTX 2.3 Image to Video",
            "image_to_video",
            64,
            [image],
        ),
        _tool_schema(
            FLF_ID,
            "ltx23.first_last_frame_to_video",
            "LTX 2.3 First/Last Frame to Video",
            "first_frame_last_frame_video",
            32,
            [first, last],
        ),
    ]
    return [{**schema, "schema_hash": _schema_hash(schema)} for schema in schemas]


TOOLS = _tool_definitions()
TOOLS_BY_ID = {tool["id"]: tool for tool in TOOLS}


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


class _LtxOperationRuntime:
    """Run one fixed LTX operation identity inside its GPU process."""

    def __init__(self, paths: LtxModelPaths, operation: str) -> None:
        self.paths = paths
        self.operation = operation
        self.runtime = self._create_runtime(operation)

    def generate(self, inputs: dict[str, Any], output_path: Path) -> None:
        media = self._generate_media(inputs)
        media.save_mp4(output_path)

    def _create_runtime(self, operation: str) -> Any:
        if operation == "t2v":
            from .ltx23.t2v import Ltx23T2VIdentity, Ltx23T2VRuntime

            return Ltx23T2VRuntime(
                Ltx23T2VIdentity(
                    checkpoint_path=str(self.paths.dev_checkpoint),
                    text_checkpoint_path=str(self.paths.text_checkpoint),
                    transformer_lora_path=str(self.paths.transformer_lora),
                    upsampler_path=str(self.paths.upsampler),
                )
            )
        if operation == "i2v":
            from .ltx23.i2v import Ltx23I2VIdentity, Ltx23I2VRuntime

            return Ltx23I2VRuntime(
                Ltx23I2VIdentity(
                    checkpoint_path=str(self.paths.dev_checkpoint),
                    text_checkpoint_path=str(self.paths.text_checkpoint),
                    transformer_lora_path=str(self.paths.transformer_lora),
                    upsampler_path=str(self.paths.upsampler),
                )
            )
        if operation == "flf":
            from .ltx23.flf import Ltx23FlfIdentity, Ltx23FlfRuntime

            return Ltx23FlfRuntime(
                Ltx23FlfIdentity(
                    checkpoint_path=str(self.paths.distilled_checkpoint),
                    text_checkpoint_path=str(self.paths.text_checkpoint),
                )
            )
        raise ValueError("Unsupported LTX operation")

    def _generate_media(self, inputs: dict[str, Any]) -> Any:
        common = {
            "prompt": inputs["prompt"],
            "width": inputs["width"],
            "height": inputs["height"],
            "duration_seconds": inputs["duration_seconds"],
            "seed": inputs["seed"],
        }
        if self.operation == "t2v":
            return self.runtime.generate(**common)
        if self.operation == "i2v":
            return self.runtime.generate(image_path=inputs["start_image"], **common)
        return self.runtime.generate(
            first_image_path=inputs["start_image"],
            last_image_path=inputs["end_image"],
            **common,
        )

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
        runtime = _LtxOperationRuntime(paths, operation)
        while True:
            message = connection.recv()
            if message["type"] == "close":
                return
            try:
                runtime.generate(message["inputs"], message["output_path"])
            except Exception as error:  # noqa: BLE001 - isolate native failure
                connection.send({"ok": False, "error_type": type(error).__name__})
                return
            connection.send({"ok": True})
    finally:
        if runtime is not None:
            runtime.close()
        connection.close()


class LtxRuntimeOwner:
    """Keep one operation-specific GPU process warm and replace it on switches."""

    def __init__(self, paths: LtxModelPaths) -> None:
        self.paths = paths
        self.available = paths.available()
        self._lock = threading.Lock()
        self._operation: str | None = None
        self._process: multiprocessing.Process | None = None
        self._connection: Connection | None = None
        self._generation_count = 0
        self._reuse_count = 0
        self._switch_count = 0
        self._release_count = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "operation": self._operation,
            "generation_count": self._generation_count,
            "reuse_count": self._reuse_count,
            "switch_count": self._switch_count,
            "release_count": self._release_count,
        }

    def generate(
        self, operation: str, inputs: dict[str, Any], output_path: Path
    ) -> None:
        with self._lock:
            if not self.available:
                raise RuntimeError("Required LTX model files are not installed")
            same_operation = (
                self._operation == operation
                and self._process is not None
                and self._process.is_alive()
            )
            if same_operation:
                self._reuse_count += 1
            else:
                if self._process is not None:
                    previous_operation = self._operation
                    self._stop_worker()
                    if previous_operation != operation:
                        self._switch_count += 1
                self._start_worker(operation)
            connection = self._connection
            if connection is None:
                raise RuntimeError("LTX worker did not start")
            try:
                connection.send(
                    {
                        "type": "generate",
                        "inputs": inputs,
                        "output_path": output_path,
                    }
                )
                result = connection.recv()
            except (BrokenPipeError, EOFError, OSError) as error:
                self._stop_worker()
                raise RuntimeError("LTX worker stopped unexpectedly") from error
            if not result.get("ok"):
                error_type = result.get("error_type", "NativeError")
                self._stop_worker()
                raise RuntimeError(f"LTX worker failed ({error_type})")
            self._generation_count += 1

    def release(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeBusyError("The active LTX runtime is still executing a job")
        try:
            self._stop_worker()
            self._release_count += 1
        finally:
            self._lock.release()

    def _start_worker(self, operation: str) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=_ltx_worker_main,
            args=(operation, self.paths, child),
            name=f"latentslate-ltx-{operation}",
            daemon=True,
        )
        process.start()
        child.close()
        self._connection = parent
        self._process = process
        self._operation = operation

    def _stop_worker(self) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        self._operation = None
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
    available: bool

    def generate(
        self, operation: str, inputs: dict[str, Any], output_path: Path
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
    message: str = "Waiting for GPU"
    cancel_requested: bool = False
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, str] | None = None

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
        self._lock = threading.Lock()
        self._assets: dict[uuid.UUID, AssetRecord] = {}
        self._asset_bytes = 0
        self._jobs: dict[uuid.UUID, JobRecord] = {}
        self._closing = False
        self._pending: queue.Queue[uuid.UUID | None] = queue.Queue(MAX_QUEUED_JOBS)
        self._worker = threading.Thread(
            target=self._work, name="latentslate-ltx-gpu", daemon=True
        )
        self._worker.start()

    def store_asset(self, upload: UploadFile) -> AssetRecord:
        with self._lock:
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
        operation, inputs, asset_ids = self._validate_job(body)
        with self._lock:
            if self._closing:
                raise EngineHttpError(503, "The Engine service is shutting down")
            if len(self._jobs) >= MAX_JOB_COUNT:
                self._reclaim_oldest_terminal_job_locked()
            if len(self._jobs) >= MAX_JOB_COUNT:
                raise EngineHttpError(503, "The in-memory job limit has been reached")
            job_id = uuid.uuid4()
            directory = self.job_root / job_id.hex
            directory.mkdir()
            job = JobRecord(
                job_id,
                operation,
                inputs,
                directory / "output.mp4",
                asset_ids=asset_ids,
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
                or filename != "output.mp4"
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
                self.executor.generate(job.operation, job.inputs, job.output_path)
            except Exception as error:  # noqa: BLE001 - native failures end the job
                LOGGER.error("Engine job %s failed (%s)", job.id, type(error).__name__)
                job.output_path.unlink(missing_ok=True)
                with self._lock:
                    if job.cancel_requested:
                        job.status = "canceled"
                        job.message = "Canceled"
                    else:
                        job.status = "failed"
                        job.message = "Generation failed"
                        job.error = {"message": "Engine generation failed."}
                    job.progress = 1.0
                    self._reclaim_job_assets_locked(job)
                continue
            with self._lock:
                if job.cancel_requested:
                    job.output_path.unlink(missing_ok=True)
                    job.status = "canceled"
                    job.message = "Canceled"
                    job.progress = 1.0
                    self._reclaim_job_assets_locked(job)
                    continue
                job.status = "succeeded"
                job.message = "Complete"
                job.progress = 1.0
                job.artifacts = [
                    {
                        "role": "primary",
                        "filename": "output.mp4",
                        "download_url": f"/v1/artifacts/{job.id}/output.mp4",
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

    def _validate_job(
        self, body: dict[str, Any]
    ) -> tuple[str, dict[str, Any], frozenset[uuid.UUID]]:
        if not self.executor.available:
            raise EngineHttpError(503, "Required LTX model files are not installed")
        tool_id = body.get("tool_id")
        tool = TOOLS_BY_ID.get(tool_id)
        if tool is None:
            raise EngineHttpError(422, "Unknown tool_id")
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
        duration = inputs.get("duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise EngineHttpError(422, "duration_seconds must be numeric")
        operation = {T2V_ID: "t2v", I2V_ID: "i2v", FLF_ID: "flf"}[tool_id]
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
        asset_ids = set()
        for key in ("start_image", "end_image"):
            if key in expected:
                asset = self._resolve_asset(inputs.get(key), inputs)
                inputs[key] = asset.path
                asset_ids.add(asset.id)
        return operation, inputs, frozenset(asset_ids)

    def _resolve_asset(self, value: Any, inputs: dict[str, Any]) -> AssetRecord:
        if not isinstance(value, dict) or value.get("type") != "asset":
            raise EngineHttpError(422, "Media inputs must reference an uploaded asset")
        try:
            asset_id = uuid.UUID(str(value.get("asset_id")))
        except (TypeError, ValueError, AttributeError) as error:
            raise EngineHttpError(422, "Media asset_id must be a UUID") from error
        with self._lock:
            asset = self._assets.get(asset_id)
        if asset is None:
            raise EngineHttpError(422, "Media asset was not found")
        try:
            from PIL import Image

            with Image.open(asset.path) as image:
                if image.size != (inputs["width"], inputs["height"]):
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
    runtime = executor or LtxRuntimeOwner(LtxModelPaths.from_home(engine_home))
    service = EngineService(engine_home / "runtime" / "http", runtime)
    auth_token = (
        token if token is not None else os.environ.get("LATENTSLATE_ENGINE_TOKEN", "")
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        service.close()

    app = FastAPI(title="LatentSlate Engine", lifespan=lifespan)
    app.state.engine_service = service

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
        available = runtime.available
        tools = []
        for tool in TOOLS:
            public = {**tool, "available": available}
            if not available:
                public["unavailable_reason"] = (
                    "Required LTX model files are not installed."
                )
            tools.append(public)
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
        return FileResponse(
            service.artifact(job_id, filename),
            media_type="video/mp4",
            filename="output.mp4",
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
