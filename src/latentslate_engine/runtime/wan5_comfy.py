"""Managed execution of the pinned official Comfy Wan 2.2 TI2V 5B graph."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from ..wan22_ti2v5b_recipe import (
    WAN5_COMFY_EXAMPLES_REVISION,
    WAN5_COMFY_RUNTIME_REVISION,
    WAN5_COMFY_SOURCE_REVISION,
    Wan5RuntimeRequest,
    revalidate_wan5_runtime_request,
    workflow_sha256,
)
from .kit import cleanup_accelerator_memory

WAN5_FPS = 24
WAN5_STEPS = 30
WAN5_CFG = 5.0
WAN5_SHIFT = 8.0
WAN5_SAMPLER = "uni_pc"
WAN5_SCHEDULER = "simple"
WAN5_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, paintings, still image, "
    "overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "messy background, three legs, many people in the background, walking backwards"
)
_POLL_SECONDS = 0.25
_START_TIMEOUT_SECONDS = 120.0
_JOB_TIMEOUT_SECONDS = 2 * 60 * 60.0


@dataclass(frozen=True, slots=True)
class Wan5ComfyRequest:
    prompt: str
    negative_prompt: str = WAN5_NEGATIVE_PROMPT
    num_frames: int = 121
    height: int = 704
    width: int = 1280
    seed: int = 0


@dataclass(frozen=True, slots=True)
class Wan5ComfyResult:
    video_path: Path
    provenance: dict[str, Any]


class ManagedWan5ComfyRuntime:
    """One isolated loopback Comfy worker tied to an exact component request."""

    def __init__(self, recipe: Wan5RuntimeRequest, *, comfy_root: Path) -> None:
        self.recipe = recipe
        self.comfy_root = Path(comfy_root).resolve()
        self._process: subprocess.Popen[str] | None = None
        self._workspace: TemporaryDirectory[str] | None = None
        self._base_url: str | None = None
        self._lock = RLock()
        self._jobs = 0

    def generate(
        self,
        request: Wan5ComfyRequest,
        *,
        output_path: Path,
        progress: Callable[[float, str | None], None],
        check_cancelled: Callable[[], None],
    ) -> Wan5ComfyResult:
        with self._lock:
            validate_wan5_comfy_request(request, self.recipe.operation)
            check_cancelled()
            if not revalidate_wan5_runtime_request(self.recipe):
                self.unload()
                raise RuntimeError("Wan 5B component recipe changed after catalog validation")
            cold_start = self._process is None
            started = time.monotonic()
            self._ensure_server(check_cancelled=check_cancelled)
            server_ready = time.monotonic()
            prompt_id = self._queue_prompt(self._workflow(request))
            queued = time.monotonic()
            try:
                output = self._wait_for_output(
                    prompt_id,
                    progress=progress,
                    check_cancelled=check_cancelled,
                )
                completed = time.monotonic()
                self._download_output(output, output_path)
                downloaded = time.monotonic()
                self._jobs += 1
                return Wan5ComfyResult(
                    output_path,
                    {
                        "backend": "comfyui/loopback-official-graph",
                        "operation": self.recipe.operation,
                        "recipe_fingerprint": self.recipe.fingerprint,
                        "comfy_source_revision": WAN5_COMFY_SOURCE_REVISION,
                        "comfy_runtime_revision": WAN5_COMFY_RUNTIME_REVISION,
                        "workflow_revision": WAN5_COMFY_EXAMPLES_REVISION,
                        "workflow_sha256": workflow_sha256(self.recipe.operation),
                        "schedule": {
                            "steps": WAN5_STEPS,
                            "cfg": WAN5_CFG,
                            "sampler": WAN5_SAMPLER,
                            "scheduler": WAN5_SCHEDULER,
                            "shift": WAN5_SHIFT,
                            "denoise": 1.0,
                        },
                        "cold_start": cold_start,
                        "timings_seconds": {
                            "server_start": server_ready - started,
                            "queue": queued - server_ready,
                            "generation": completed - queued,
                            "download": downloaded - completed,
                            "total": downloaded - started,
                        },
                    },
                )
            except BaseException:
                self._interrupt()
                self.unload()
                raise

    def unload(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            self._base_url = None
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            workspace = self._workspace
            self._workspace = None
            if workspace is not None:
                workspace.cleanup()
            cleanup_accelerator_memory()

    def clear_cache(self) -> None:
        self.unload()

    def status(self) -> dict[str, Any]:
        return {
            "family": "wan22",
            "operation": self.recipe.operation,
            "loaded": self._process is not None and self._process.poll() is None,
            "jobs": self._jobs,
            "backend": "comfyui/loopback-official-graph",
            "recipe_fingerprint": self.recipe.fingerprint,
        }

    def _ensure_server(self, *, check_cancelled: Callable[[], None]) -> None:
        if self._process is not None and self._process.poll() is None and self._base_url:
            return
        python = self.comfy_root / ".venv" / "Scripts" / "python.exe"
        main = self.comfy_root / "main.py"
        if not python.is_file() or not main.is_file():
            raise RuntimeError(f"ComfyUI checkout is incomplete: {self.comfy_root}")
        try:
            runtime_revision = subprocess.run(
                ["git", "-C", str(self.comfy_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("Could not establish the ComfyUI checkout revision") from exc
        if runtime_revision != WAN5_COMFY_RUNTIME_REVISION:
            raise RuntimeError(
                "ComfyUI checkout revision does not match the pinned Wan 5B runtime: "
                f"expected {WAN5_COMFY_RUNTIME_REVISION}, found {runtime_revision}"
            )
        self._workspace = TemporaryDirectory(prefix="latentslate-wan5-comfy-")
        root = Path(self._workspace.name)
        model_root = root / "models"
        for role, folder in (
            ("transformer", "diffusion_models"),
            ("text_encoder", "text_encoders"),
            ("vae", "vae"),
        ):
            target = model_root / folder / Path(str(self.recipe.components[role]["path"])).name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(str(self.recipe.components[role]["path"]), target)
        port = _free_port()
        self._base_url = f"http://127.0.0.1:{port}"
        log = root / "comfy.log"
        self._process = subprocess.Popen(
            [
                str(python),
                str(main),
                "--listen",
                "127.0.0.1",
                "--port",
                str(port),
                "--base-directory",
                str(root),
                "--output-directory",
                str(root / "output"),
                "--input-directory",
                str(root / "input"),
                "--temp-directory",
                str(root / "temp"),
                "--disable-all-custom-nodes",
                "--disable-auto-launch",
                "--preview-method",
                "none",
                "--cache-classic",
                "--deterministic",
                "--disable-metadata",
                "--log-stdout",
            ],
            cwd=self.comfy_root,
            stdout=log.open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + _START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            check_cancelled()
            if self._process.poll() is not None:
                tail = log.read_text(encoding="utf-8", errors="replace")[-8000:]
                raise RuntimeError(f"ComfyUI Wan worker exited during startup: {tail}")
            try:
                self._json_request("GET", "/system_stats")
                return
            except (HTTPError, URLError, TimeoutError, ValueError):
                time.sleep(_POLL_SECONDS)
        raise TimeoutError("ComfyUI Wan worker did not become ready")

    def _workflow(self, request: Wan5ComfyRequest) -> dict[str, Any]:
        transformer = Path(str(self.recipe.components["transformer"]["path"])).name
        text_encoder = Path(str(self.recipe.components["text_encoder"]["path"])).name
        vae = Path(str(self.recipe.components["vae"]["path"])).name
        workflow: dict[str, Any] = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": transformer, "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": text_encoder, "type": "wan", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
            "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": WAN5_SHIFT}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": request.prompt}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": request.negative_prompt}},
            "7": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["3", 0], "width": request.width, "height": request.height, "length": request.num_frames, "batch_size": 1}},
            "8": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0], "seed": request.seed, "steps": WAN5_STEPS, "cfg": WAN5_CFG, "sampler_name": WAN5_SAMPLER, "scheduler": WAN5_SCHEDULER, "denoise": 1.0}},
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
            "10": {"class_type": "SaveWEBM", "inputs": {"images": ["9", 0], "filename_prefix": "latentslate-wan5", "codec": "vp9", "fps": WAN5_FPS, "crf": 18.0}},
        }
        return workflow

    def _queue_prompt(self, workflow: dict[str, Any]) -> str:
        response = self._json_request(
            "POST",
            "/prompt",
            payload={"prompt": workflow, "client_id": f"latentslate-{uuid4().hex}"},
        )
        prompt_id = response.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise RuntimeError(f"ComfyUI rejected Wan prompt: {response}")
        return prompt_id

    def _wait_for_output(
        self,
        prompt_id: str,
        *,
        progress: Callable[[float, str | None], None],
        check_cancelled: Callable[[], None],
    ) -> dict[str, str]:
        deadline = time.monotonic() + _JOB_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            check_cancelled()
            history = self._json_request("GET", f"/history/{prompt_id}")
            item = history.get(prompt_id)
            if item is not None:
                status = item.get("status", {})
                if status.get("status_str") == "error" or status.get("completed") is False:
                    raise RuntimeError(f"ComfyUI Wan graph failed: {status.get('messages')}")
                outputs = item.get("outputs", {})
                node = outputs.get("10", {})
                videos = node.get("images") or node.get("gifs") or node.get("videos")
                if videos:
                    return videos[0]
            progress(0.10, "Generating Wan 2.2 video in official Comfy graph")
            time.sleep(_POLL_SECONDS)
        raise TimeoutError("ComfyUI Wan generation exceeded its bounded timeout")

    def _download_output(self, output: dict[str, str], target: Path) -> None:
        filename = output.get("filename")
        if not filename:
            raise RuntimeError("ComfyUI Wan output lacks a filename")
        import urllib.parse

        query = urllib.parse.urlencode(
            {
                "filename": filename,
                "subfolder": output.get("subfolder", ""),
                "type": output.get("type", "output"),
            }
        )
        payload = self._raw_request("GET", f"/view?{query}")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.{uuid4().hex}.partial")
        try:
            partial.write_bytes(payload)
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)

    def _interrupt(self) -> None:
        try:
            self._json_request("POST", "/interrupt", payload={})
        except (HTTPError, URLError, TimeoutError, ValueError):
            pass

    def _json_request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        raw = self._raw_request(method, path, payload=payload)
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise TypeError("ComfyUI returned a non-object JSON payload")
        return result

    def _raw_request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> bytes:
        if self._base_url is None:
            raise RuntimeError("ComfyUI Wan worker is not started")
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(
            self._base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        with urlopen(request, timeout=10) as response:
            return response.read()


def validate_wan5_comfy_request(request: Wan5ComfyRequest, operation: str) -> None:
    if operation != "text_to_video":
        raise ValueError("Wan 5B operation is invalid")
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise ValueError("Wan 5B prompt must be nonempty")
    if not isinstance(request.negative_prompt, str):
        raise TypeError("Wan 5B negative prompt must be text")
    for name, value in (("width", request.width), ("height", request.height), ("num_frames", request.num_frames)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Wan 5B {name} must be a positive integer")
    if request.width % 32 or request.height % 32:
        raise ValueError("Wan 5B width and height must be divisible by 32")
    if (request.num_frames - 1) % 4 or request.num_frames > 121:
        raise ValueError("Wan 5B frame count must be 4k+1 and at most 121")
    if request.width * request.height > 1280 * 704:
        raise ValueError("Wan 5B request exceeds the official 1280x704 pixel budget")
    if isinstance(request.seed, bool) or not isinstance(request.seed, int) or not 0 <= request.seed < 2**63:
        raise ValueError("Wan 5B seed must be in [0, 2^63)")


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
