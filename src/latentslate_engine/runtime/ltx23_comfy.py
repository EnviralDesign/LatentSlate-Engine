"""Disposable official-Comfy execution for the optimized LTX 2.3 operations.

Every request starts one loopback-only Comfy process in a Windows Job Object and
destroys that process tree before returning.  This is intentionally separate
from the native BF16 Diffusers worker: the official Comfy Dev and FLF graphs
have different artifact closures, node topology, sampling schedules, and
conditioning semantics.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from ..ltx23_comfy_recipe import (
    LTX23_COMFY_FPS,
    LTX23_COMFY_GUIDE_STRENGTH,
    LTX23_COMFY_MAIN_SIGMAS,
    LTX23_COMFY_MODEL_LORA_STRENGTH,
    LTX23_COMFY_RUNTIME_REVISION,
    LTX23_COMFY_TEMPLATE_REVISION,
    LTX23_COMFY_UPSCALE_SIGMAS,
    LTX23ComfyRuntimeRequest,
    revalidate_ltx23_comfy_runtime_request,
    template_sha256,
)
from .ltx23 import LTX23_MAX_DURATION_SECONDS, LTX23_MAX_PIXELS, LTX23_MIN_DURATION_SECONDS
from .windows_process import DisposableProcessTree

_POLL_SECONDS = 0.25
_START_TIMEOUT_SECONDS = 120.0
_JOB_TIMEOUT_SECONDS = 2 * 60 * 60.0
_QUEUE_RUNNING_MESSAGE = "Comfy queue running official LTX 2.3 graph"
_IO_CHUNK_BYTES = 1024 * 1024
_MAX_ENDPOINT_BYTES = 64 * 1024 * 1024
_MAX_OUTPUT_BYTES = 512 * 1024 * 1024

# Git blob object IDs for precisely the core files whose registered nodes the
# pinned official graphs submit.  This deliberately permits unrelated checkout
# edits, but rejects a dirty node implementation masquerading as the recorded
# Comfy revision.
_COMFY_REQUIRED_SOURCE_BLOBS = {
    "comfy_extras/nodes_custom_sampler.py": "d5aa730d22cd0bed99f8a995529d2fff33db3a45",
    "comfy_extras/nodes_dataset.py": "71e5ee368632f56f9d752943a4988c475a0b621c",
    "comfy_extras/nodes_hunyuan.py": "ce2997245840e6f038b3e0c26ceb900cb6d9910a",
    "comfy_extras/nodes_logic.py": "13c1685f7d8ba95d9d5ab640f8249e0f895889d5",
    "comfy_extras/nodes_lt.py": "a6e5c5d27b56ffe1b59e4bd4e26e62ee9278580d",
    "comfy_extras/nodes_lt_audio.py": "0924f3e9e9c56ac1031e4cbd1cc2fe2f21ab0cb7",
    "comfy_extras/nodes_lt_upsampler.py": "7e7975495cb6a722d5aa06700c2354edd2861022",
    "comfy_extras/nodes_post_processing.py": "763b8a52fa84fbee9432ee116000f131c0002400",
    "comfy_extras/nodes_primitive.py": "35761863ff49fba89af08f02314354f92770bc91",
    "comfy_extras/nodes_textgen.py": "40004652cb43b1b39bec50089b61178695a27c7e",
    "comfy_extras/nodes_video.py": "45394ce4df077c1bfb9593a21637675d9cc48b99",
    "nodes.py": "a7f91720f45761ec1f41e32c31646f9faefde88f",
}

_COMFY_REQUIRED_SOURCE_MARKERS = {
    "comfy_extras/nodes_lt.py": (
        "EmptyLTXVLatentVideo", "LTXVImgToVideoInplace", "LTXVPreprocess",
        "LTXVCropGuides", "LTXVAddGuide", "LTXVConditioning",
        "LTXVConcatAVLatent", "LTXVSeparateAVLatent",
    ),
    "comfy_extras/nodes_lt_audio.py": (
        "LTXAVTextEncoderLoader", "LTXVAudioVAELoader", "LTXVAudioVAEDecode",
        "LTXVEmptyLatentAudio",
    ),
    "comfy_extras/nodes_hunyuan.py": ("LatentUpscaleModelLoader",),
    "comfy_extras/nodes_lt_upsampler.py": ("LTXVLatentUpsampler",),
    "comfy_extras/nodes_textgen.py": ("TextGenerateLTX2Prompt",),
    "comfy_extras/nodes_video.py": ("class SaveVideo", "class CreateVideo"),
    "comfy_extras/nodes_dataset.py": ("ResizeImagesByLongerEdge",),
    "comfy_extras/nodes_post_processing.py": ("ResizeImageMaskNode",),
    "comfy_extras/nodes_logic.py": ("ComfySwitchNode",),
    "comfy_extras/nodes_primitive.py": ("PrimitiveStringMultiline",),
    "comfy_extras/nodes_custom_sampler.py": (
        "SamplerCustomAdvanced", "SamplerEulerAncestral", "RandomNoise", "KSamplerSelect",
        "ManualSigmas", "CFGGuider",
    ),
    "nodes.py": (
        "CheckpointLoaderSimple", "LoraLoader", "LoraLoaderModelOnly", "CLIPTextEncode",
        "VAEDecodeTiled", "LoadImage", "EmptyImage",
    ),
}


@dataclass(frozen=True, slots=True)
class LTX23ComfyRequest:
    prompt: str
    width: int
    height: int
    duration_seconds: float
    seed: int
    start_image: Path | None = None
    end_image: Path | None = None


@dataclass(frozen=True, slots=True)
class LTX23ComfyResult:
    output_path: Path
    provenance: dict[str, object]


class ManagedLTX23ComfyRuntime:
    """Fresh Comfy process tree for one exact LTX optimized operation."""

    def __init__(self, recipe: LTX23ComfyRuntimeRequest, *, comfy_root: Path) -> None:
        self.recipe = recipe
        self.comfy_root = Path(comfy_root).resolve()
        self._last_worker: dict[str, object] | None = None
        self._active_tree: DisposableProcessTree | None = None
        self._cleanup_errors: list[str] = []
        self._ownership = Lock()

    def generate(
        self,
        request: LTX23ComfyRequest,
        *,
        output_path: Path,
        progress: Callable[[float, str | None], None],
        check_cancelled: Callable[[], None],
    ) -> LTX23ComfyResult:
        _validate_request(request, self.recipe.operation)
        if not revalidate_ltx23_comfy_runtime_request(self.recipe):
            raise RuntimeError("LTX 2.3 Comfy component closure changed after catalog validation")
        check_cancelled()
        if not self._ownership.acquire(blocking=False):
            raise RuntimeError("LTX 2.3 Comfy disposable worker is already active")
        # A new request must never inherit a previous terminal worker claim if
        # checkout/staging fails before a new process is spawned.
        self._last_worker = None
        self._cleanup_errors = []
        workspace: Path | None = None
        process: subprocess.Popen[str] | None = None
        tree: DisposableProcessTree | None = None
        allocator_policy: str | None = None
        try:
            _validate_comfy_checkout(self.comfy_root)
            workspace = _workspace(self.recipe)
            _stage_components(workspace, self.recipe)
            start_name = (
                _stage_endpoint(request.start_image, workspace, "first", check_cancelled)
                if request.start_image
                else None
            )
            end_name = (
                _stage_endpoint(request.end_image, workspace, "last", check_cancelled)
                if request.end_image
                else None
            )
            port = _free_port()
            gate = workspace / "start.gate"
            process = _start_comfy(self.comfy_root, workspace, port, start_gate=gate)
            allocator_policy = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            tree = DisposableProcessTree(process)
            self._active_tree = tree
            # The gated bootstrap has not imported Comfy/Torch or started Comfy
            # until it is in the Job Object.  Tree ownership therefore precedes
            # any heavyweight worker action.
            gate.touch(exist_ok=False)
            base_url = f"http://127.0.0.1:{port}"
            _wait_ready(process, base_url, check_cancelled)
            _validate_live_object_info(_json_request(base_url, "GET", "/object_info"), workflow=None)
            progress(0.05, "Preparing exact official Comfy LTX 2.3 graph")
            workflow = build_workflow(self.recipe, request, start_name=start_name, end_name=end_name)
            _validate_live_object_info(_json_request(base_url, "GET", "/object_info"), workflow=workflow)
            workflow_sha256 = _sha256_json(workflow)
            prompt_id = _queue(base_url, workflow)
            video = _wait_video(
                base_url, prompt_id, progress, check_cancelled,
                save_node_id="25" if self.recipe.operation == "comfy_distilled_flf" else "33",
            )
            observed = _download_validate_and_publish(
                base_url, video, output_path, workspace, request, check_cancelled,
            )
            tree.terminate()
            tree.wait_for_empty()
            self._last_worker = {
                "pid": process.pid,
                "exit_code": process.poll(),
                "outcome": "succeeded",
                "terminated": True,
                "tree_empty": True,
                "memory_boundary": "disposable_process_exit",
                "allocator_policy": allocator_policy,
            }
            return LTX23ComfyResult(
                output_path,
                _provenance(self.recipe, request, workflow_sha256, self._last_worker, observed),
            )
        except BaseException as exc:
            tree_empty = _terminate_worker(tree, process, exc)
            if process is not None:
                terminated = tree_empty or process.poll() is not None
                self._last_worker = {
                    "pid": process.pid,
                    "exit_code": process.poll(),
                    "outcome": "canceled" if _is_cancelled(exc) else "failed",
                    "terminated": terminated,
                    # Direct Popen fallback proves only the root process exit;
                    # it must not claim Job Object tree-empty evidence.
                    "tree_empty": tree_empty,
                    "root_exited": process.poll() is not None,
                    "memory_boundary": (
                        "disposable_process_exit" if terminated else "termination_unproven"
                    ),
                    "allocator_policy": allocator_policy,
                }
            else:
                self._last_worker = {
                    "outcome": "canceled" if _is_cancelled(exc) else "failed",
                    "spawned": False,
                    "terminated": False,
                    "tree_empty": False,
                    "root_exited": False,
                    "memory_boundary": "no_worker_spawned",
                    "allocator_policy": allocator_policy,
                }
            _remove_failed_output(output_path, exc)
            raise
        finally:
            self._active_tree = None
            cleanup_errors: list[str] = []
            try:
                if tree is not None:
                    tree.close()
            except OSError:
                cleanup_errors.append("job_object")
            if workspace is not None:
                cleanup_errors.extend(_cleanup_workspace(workspace))
            self._cleanup_errors = sorted(set(cleanup_errors))
            self._ownership.release()

    def clear_cache(self) -> None:
        """A disposable worker retains no reusable Comfy model/cache state."""

    def unload(self) -> None:
        self.clear_cache()

    def status(self) -> dict[str, object]:
        return {
            "family": "ltx23",
            "runtime": "comfyui_disposable_worker",
            "loaded": False,
            "active_worker": self._active_tree is not None,
            "pipeline_fingerprint": self.recipe.component_fingerprint,
            "last_worker": self._last_worker,
            "cleanup_errors": list(self._cleanup_errors),
            "cache_support": {"prompt": False, "media": False},
        }


def build_workflow(
    recipe: LTX23ComfyRuntimeRequest,
    request: LTX23ComfyRequest,
    *,
    start_name: str | None,
    end_name: str | None,
) -> dict[str, dict[str, object]]:
    """Return the literal API graph for one pinned Comfy-template topology."""

    names = {role: Path(str(value["path"])).name for role, value in recipe.components.items()}
    frames = _frames(request.duration_seconds)
    if recipe.operation == "comfy_distilled_flf":
        if not start_name or not end_name:
            raise ValueError("LTX 2.3 Comfy FLF requires both uploaded endpoints")
        return _flf_graph(names, request, frames, start_name, end_name)
    if recipe.operation == "comfy_dev_i2v" and not start_name:
        raise ValueError("LTX 2.3 Comfy first-frame I2V requires one uploaded endpoint")
    return _dev_graph(names, request, frames, start_name)


def _dev_graph(names: Mapping[str, str], request: LTX23ComfyRequest, frames: int, start_name: str | None) -> dict[str, dict[str, object]]:
    # This is the two-stage graph from the current official t2v/i2v template:
    # 8-step Dev+Distilled-LoRA base denoise, x2 latent upscale, then 3-step refine.
    initial_width, initial_height = request.width // 2, request.height // 2
    prompt_enhance_inputs: dict[str, object] = {
        "clip": ["35", 1],
        "prompt": ["34", 0],
        "max_length": 2048,
        "sampling_mode": "on",
        "sampling_mode.temperature": 0.7,
        "sampling_mode.top_k": 64,
        "sampling_mode.top_p": 0.95,
        "sampling_mode.min_p": 0.05,
        "sampling_mode.repetition_penalty": 1.05,
        "sampling_mode.seed": 0,
        "thinking": False,
        "use_default_template": True,
    }
    if start_name is not None:
        # Current official I2V passes the longer-edge preprocessed first frame
        # to the enhancer and keeps enhancement opt-in; T2V defaults it on.
        prompt_enhance_inputs["image"] = ["42", 0]
    graph: dict[str, dict[str, object]] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": names["checkpoint"]}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": names["model_lora"], "strength_model": LTX23_COMFY_MODEL_LORA_STRENGTH}},
        "3": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": names["text_encoder"], "ckpt_name": names["checkpoint"], "device": "default"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ["37", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "pc game, console game, video game, cartoon, childish, ugly"}},
        "6": {"class_type": "LTXVConditioning", "inputs": {"positive": ["4", 0], "negative": ["5", 0], "frame_rate": LTX23_COMFY_FPS}},
        "7": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": names["checkpoint"]}},
        "8": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": initial_width, "height": initial_height, "length": frames, "batch_size": 1}},
        "9": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"audio_vae": ["7", 0], "frames_number": frames, "frame_rate": LTX23_COMFY_FPS, "batch_size": 1}},
        "10": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["1", 2], "image": ["40", 0], "latent": ["8", 0], "strength": 0.7, "bypass": start_name is None}},
        "12": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["10", 0], "audio_latent": ["9", 0]}},
        "40": {"class_type": "LTXVPreprocess", "inputs": {"image": ["42", 0], "img_compression": 18}},
        "41": {"class_type": "ResizeImageMaskNode", "inputs": {"input": ["11", 0], "resize_type": "scale dimensions", "resize_type.width": request.width, "resize_type.height": request.height, "resize_type.crop": "center", "scale_method": "lanczos"}},
        "42": {"class_type": "ResizeImagesByLongerEdge", "inputs": {"images": ["41", 0], "longer_edge": 1536}},
        "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": request.seed}},
        "14": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "15": {"class_type": "ManualSigmas", "inputs": {"sigmas": ", ".join(str(value) for value in LTX23_COMFY_MAIN_SIGMAS)}},
        # The official template crops guides between base sampling/upscale and
        # refinement; the base pass uses the un-cropped conditioning.
        "16": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["6", 0], "negative": ["6", 1], "latent": ["19", 0]}},
        "17": {"class_type": "CFGGuider", "inputs": {"model": ["2", 0], "positive": ["6", 0], "negative": ["6", 1], "cfg": 1}},
        "18": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["13", 0], "guider": ["17", 0], "sampler": ["14", 0], "sigmas": ["15", 0], "latent_image": ["12", 0]}},
        "19": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["18", 0]}},
        "20": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": names["latent_upscaler"]}},
        "21": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["19", 0], "upscale_model": ["20", 0], "vae": ["1", 2]}},
        "22": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["1", 2], "image": ["40", 0], "latent": ["21", 0], "strength": 1.0, "bypass": start_name is None}},
        "23": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["22", 0], "audio_latent": ["19", 1]}},
        "24": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "25": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "26": {"class_type": "ManualSigmas", "inputs": {"sigmas": ", ".join(str(value) for value in LTX23_COMFY_UPSCALE_SIGMAS)}},
        "27": {"class_type": "CFGGuider", "inputs": {"model": ["2", 0], "positive": ["16", 0], "negative": ["16", 1], "cfg": 1}},
        "28": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["24", 0], "guider": ["27", 0], "sampler": ["25", 0], "sigmas": ["26", 0], "latent_image": ["23", 0]}},
        "29": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["28", 0]}},
        "30": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["29", 0], "vae": ["1", 2], "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 4}},
        "31": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["29", 1], "audio_vae": ["7", 0]}},
        "32": {"class_type": "CreateVideo", "inputs": {"images": ["30", 0], "audio": ["31", 0], "fps": LTX23_COMFY_FPS}},
        "33": {"class_type": "SaveVideo", "inputs": {"video": ["32", 0], "filename_prefix": "latentslate-ltx23", "format": "auto", "codec": "auto"}},
        # The exact Dev templates enable this text-generation branch by default.
        # It uses the fixed text LoRA at 1.0 and switches generated text into the
        # actual CLIP encoder rather than merely staging that artifact.
        "34": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": request.prompt}},
        "35": {"class_type": "LoraLoader", "inputs": {"model": ["2", 0], "clip": ["3", 0], "lora_name": names["text_lora"], "strength_model": 1.0, "strength_clip": 1.0}},
        "36": {"class_type": "TextGenerateLTX2Prompt", "inputs": prompt_enhance_inputs},
        "37": {"class_type": "ComfySwitchNode", "inputs": {"on_false": ["34", 0], "on_true": ["36", 0], "switch": start_name is None}},
    }
    if start_name is None:
        graph["11"] = {"class_type": "EmptyImage", "inputs": {"width": request.width, "height": request.height, "batch_size": 1, "color": 0}}
    else:
        graph["11"] = {"class_type": "LoadImage", "inputs": {"image": start_name}}
    return graph


def _flf_graph(names: Mapping[str, str], request: LTX23ComfyRequest, frames: int, start_name: str, end_name: str) -> dict[str, dict[str, object]]:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": names["checkpoint"]}},
        "2": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": names["text_encoder"], "ckpt_name": names["checkpoint"], "device": "default"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": request.prompt}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "blurry, out of focus, overexposed, underexposed, low contrast"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": start_name}},
        "6": {"class_type": "LoadImage", "inputs": {"image": end_name}},
        "7": {"class_type": "LTXVPreprocess", "inputs": {"image": ["27", 0], "img_compression": 25}},
        "8": {"class_type": "LTXVPreprocess", "inputs": {"image": ["28", 0], "img_compression": 25}},
        "9": {"class_type": "LTXVConditioning", "inputs": {"positive": ["3", 0], "negative": ["4", 0], "frame_rate": LTX23_COMFY_FPS}},
        "10": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": request.width, "height": request.height, "length": frames, "batch_size": 1}},
        "11": {"class_type": "LTXVAddGuide", "inputs": {"positive": ["9", 0], "negative": ["9", 1], "vae": ["1", 2], "latent": ["10", 0], "image": ["7", 0], "frame_idx": 0, "strength": LTX23_COMFY_GUIDE_STRENGTH}},
        "12": {"class_type": "LTXVAddGuide", "inputs": {"positive": ["11", 0], "negative": ["11", 1], "vae": ["1", 2], "latent": ["11", 2], "image": ["8", 0], "frame_idx": -1, "strength": LTX23_COMFY_GUIDE_STRENGTH}},
        "13": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": names["checkpoint"]}},
        "14": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"audio_vae": ["13", 0], "frames_number": frames, "frame_rate": LTX23_COMFY_FPS, "batch_size": 1}},
        "15": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["12", 2], "audio_latent": ["14", 0]}},
        "16": {"class_type": "RandomNoise", "inputs": {"noise_seed": request.seed}},
        "17": {"class_type": "SamplerEulerAncestral", "inputs": {"eta": 0, "s_noise": 1}},
        "18": {"class_type": "ManualSigmas", "inputs": {"sigmas": ", ".join(str(value) for value in LTX23_COMFY_MAIN_SIGMAS)}},
        "19": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["12", 0], "negative": ["12", 1], "cfg": 1}},
        "20": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["16", 0], "guider": ["19", 0], "sampler": ["17", 0], "sigmas": ["18", 0], "latent_image": ["15", 0]}},
        # The official FLF graph separates the sampler's denoised output,
        # rather than its intermediate output slot.
        "21": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["20", 1]}},
        "22": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["26", 2], "vae": ["1", 2], "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 64}},
        "23": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["21", 1], "audio_vae": ["13", 0]}},
        "24": {"class_type": "CreateVideo", "inputs": {"images": ["22", 0], "audio": ["23", 0], "fps": LTX23_COMFY_FPS}},
        "25": {"class_type": "SaveVideo", "inputs": {"video": ["24", 0], "filename_prefix": "latentslate-ltx23-flf", "format": "auto", "codec": "auto"}},
        "26": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["12", 0], "negative": ["12", 1], "latent": ["21", 0]}},
        "27": {"class_type": "ResizeImageMaskNode", "inputs": {"input": ["5", 0], "resize_type": "scale dimensions", "resize_type.width": request.width, "resize_type.height": request.height, "resize_type.crop": "center", "scale_method": "nearest-exact"}},
        "28": {"class_type": "ResizeImageMaskNode", "inputs": {"input": ["6", 0], "resize_type": "scale dimensions", "resize_type.width": request.width, "resize_type.height": request.height, "resize_type.crop": "center", "scale_method": "nearest-exact"}},
    }


def _validate_request(request: LTX23ComfyRequest, operation: str) -> None:
    if not request.prompt.strip() or request.width <= 0 or request.height <= 0 or request.width % 32 or request.height % 32:
        raise ValueError("LTX 2.3 Comfy requires a prompt and dimensions divisible by 32")
    if request.width * request.height > LTX23_MAX_PIXELS:
        raise ValueError("LTX 2.3 Comfy dimensions exceed the bounded pixel area")
    if operation in {"comfy_dev_t2v", "comfy_dev_i2v"} and (
        request.width % 64 or request.height % 64
    ):
        raise ValueError("LTX 2.3 Comfy Dev dimensions must be divisible by 64")
    if not LTX23_MIN_DURATION_SECONDS <= request.duration_seconds <= LTX23_MAX_DURATION_SECONDS:
        raise ValueError("LTX 2.3 Comfy duration must be between 1 and 10 seconds")
    if _frames(request.duration_seconds) < 25:
        raise ValueError("LTX 2.3 Comfy duration must produce at least 25 frames")
    if operation == "comfy_dev_t2v" and (request.start_image or request.end_image):
        raise ValueError("LTX 2.3 Comfy T2V does not accept endpoint images")
    if operation == "comfy_dev_i2v" and (request.start_image is None or request.end_image is not None):
        raise ValueError("LTX 2.3 Comfy first-frame I2V requires only a first image")
    if operation == "comfy_distilled_flf" and (request.start_image is None or request.end_image is None):
        raise ValueError("LTX 2.3 Comfy FLF requires ordered first and last images")


def _frames(duration_seconds: float) -> int:
    frames = round(float(duration_seconds) * LTX23_COMFY_FPS) + 1
    if (frames - 1) % 8:
        raise ValueError("LTX 2.3 Comfy frames must satisfy 8n + 1 at 24 fps")
    return frames


def _workspace(recipe: LTX23ComfyRuntimeRequest) -> Path:
    source = Path(str(recipe.components["checkpoint"]["path"])).resolve(strict=True)
    return Path(mkdtemp(prefix=".latentslate-ltx23-comfy-", dir=source.parent))


def _stage_components(root: Path, recipe: LTX23ComfyRuntimeRequest) -> None:
    folders = {"checkpoint": "checkpoints", "text_encoder": "text_encoders", "model_lora": "loras", "text_lora": "loras", "latent_upscaler": "latent_upscale_models"}
    for role, values in recipe.components.items():
        source = Path(str(values["path"])).resolve(strict=True)
        target = root / "models" / folders[role] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.stat().st_dev != target.parent.stat().st_dev:
            raise RuntimeError("LTX 2.3 Comfy components must share a volume for zero-copy staging")
        os.link(source, target)


def _stage_endpoint(
    source: Path,
    workspace: Path,
    role: str,
    check_cancelled: Callable[[], None],
) -> str:
    """Copy one bounded endpoint into the job-private Comfy input directory."""

    check_cancelled()
    try:
        source_lstat = source.lstat()
        resolved = source.resolve(strict=True)
        source_stat = resolved.stat()
    except OSError as exc:
        raise ValueError("LTX 2.3 Comfy endpoint image is unavailable") from exc
    reparse_flag = getattr(source_lstat, "st_file_attributes", 0) & 0x400
    if (
        not stat.S_ISREG(source_lstat.st_mode)
        or source.is_symlink()
        or not resolved.is_file()
        or reparse_flag
    ):
        raise ValueError("LTX 2.3 Comfy endpoint image must be a regular non-reparse file")
    if source_stat.st_size > _MAX_ENDPOINT_BYTES:
        raise ValueError(
            f"LTX 2.3 Comfy endpoint image exceeds {_MAX_ENDPOINT_BYTES} bytes"
        )
    suffix = "".join(ch for ch in resolved.suffix.lower() if ch.isalnum() or ch == ".")[:16]
    if not suffix or suffix == ".":
        suffix = ".bin"
    name = f"latentslate-{role}-{uuid4().hex}{suffix}"
    target = workspace / "input" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    try:
        with resolved.open("rb") as reader, target.open("xb") as writer:
            while copied < source_stat.st_size:
                check_cancelled()
                chunk = reader.read(min(_IO_CHUNK_BYTES, source_stat.st_size - copied))
                if not chunk:
                    raise RuntimeError("LTX 2.3 Comfy endpoint image changed during staging")
                writer.write(chunk)
                copied += len(chunk)
                check_cancelled()
            if reader.read(1):
                raise RuntimeError("LTX 2.3 Comfy endpoint image changed during staging")
        check_cancelled()
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return name


def _validate_comfy_checkout(comfy_root: Path) -> None:
    """Fail closed before staging when the local Comfy API lacks this graph's nodes."""

    python = comfy_root / ".venv" / "Scripts" / "python.exe"
    main = comfy_root / "main.py"
    if not python.is_file() or not main.is_file():
        raise RuntimeError("LTX 2.3 Comfy checkout is incomplete")
    try:
        revision = subprocess.run(
            ["git", "-C", str(comfy_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("could not establish LTX 2.3 Comfy checkout revision") from exc
    if revision != LTX23_COMFY_RUNTIME_REVISION:
        raise RuntimeError("LTX 2.3 Comfy checkout revision is incompatible")
    for relative, markers in _COMFY_REQUIRED_SOURCE_MARKERS.items():
        path = comfy_root / relative
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError("LTX 2.3 Comfy checkout lacks required graph support") from exc
        if any(marker not in source for marker in markers):
            raise RuntimeError("LTX 2.3 Comfy checkout node schema is incompatible")
    for relative, expected_blob in _COMFY_REQUIRED_SOURCE_BLOBS.items():
        try:
            content = (comfy_root / relative).read_bytes()
        except OSError as exc:
            raise RuntimeError("LTX 2.3 Comfy checkout lacks pinned node source") from exc
        # Git's canonical blob uses LF while a clean Windows checkout may use
        # CRLF under core.autocrlf. Normalize only that documented checkout
        # transform; all source text and every other byte remain pinned.
        canonical_content = content.replace(b"\r\n", b"\n")
        actual_blob = hashlib.sha1(
            f"blob {len(canonical_content)}\0".encode() + canonical_content,
            usedforsecurity=False,
        ).hexdigest()
        if actual_blob != expected_blob:
            raise RuntimeError("LTX 2.3 Comfy required node source is modified")


def _validate_live_object_info(value: object, *, workflow: Mapping[str, object] | None) -> None:
    """Verify the running pinned Comfy server exposes the submitted API schema."""

    if not isinstance(value, dict):
        raise TypeError("LTX 2.3 Comfy object_info response is invalid")
    if workflow is None:
        return
    for node in workflow.values():
        if not isinstance(node, dict):
            raise TypeError("LTX 2.3 Comfy workflow node is invalid")
        class_type = node.get("class_type")
        inputs = node.get("inputs")
        schema = value.get(class_type) if isinstance(class_type, str) else None
        required = schema.get("input", {}).get("required", {}) if isinstance(schema, dict) else None
        optional = schema.get("input", {}).get("optional", {}) if isinstance(schema, dict) else None
        if not isinstance(required, dict) or not isinstance(optional, dict) or not isinstance(inputs, dict):
            raise TypeError("LTX 2.3 Comfy server lacks a submitted graph node schema")
        if any(name not in required and name not in optional for name in inputs):
            raise RuntimeError("LTX 2.3 Comfy server input schema is incompatible")


def _start_comfy(
    comfy_root: Path,
    workspace: Path,
    port: int,
    *,
    start_gate: Path,
) -> subprocess.Popen[str]:
    python, main = comfy_root / ".venv" / "Scripts" / "python.exe", comfy_root / "main.py"
    if not python.is_file() or not main.is_file():
        raise RuntimeError(f"ComfyUI checkout is incomplete: {comfy_root}")
    log = (workspace / "comfy.log").open("w", encoding="utf-8")
    comfy_command = [
        str(python), str(main), "--listen", "127.0.0.1", "--port", str(port),
        "--base-directory", str(workspace), "--output-directory", str(workspace / "output"),
        "--input-directory", str(workspace / "input"), "--temp-directory", str(workspace / "temp"),
        "--disable-all-custom-nodes", "--disable-auto-launch", "--preview-method", "none",
        "--cache-classic", "--deterministic", "--disable-metadata", "--log-stdout",
    ]
    # Exec only after the parent places this bootstrap PID in its kill-on-close
    # Job Object and releases the private start gate.
    bootstrap = (
        "import os, pathlib, sys, time; gate=pathlib.Path(sys.argv[1]); "
        "deadline=time.monotonic()+120; "
        "\nwhile not gate.exists():\n"
        " if time.monotonic() >= deadline: raise SystemExit('LTX Comfy start gate timed out')\n"
        " time.sleep(0.02)\n"
        "os.execv(sys.argv[2], sys.argv[2:])"
    )
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return subprocess.Popen(
        [sys.executable, "-c", bootstrap, str(start_gate), *comfy_command],
        cwd=comfy_root,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )


def _wait_ready(process: subprocess.Popen[str], base_url: str, check_cancelled: Callable[[], None]) -> None:
    deadline = time.monotonic() + _START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        check_cancelled()
        if process.poll() is not None:
            raise RuntimeError("LTX 2.3 Comfy worker exited during startup")
        try:
            _json_request(base_url, "GET", "/system_stats")
            return
        except (HTTPError, URLError, TimeoutError, ValueError):
            time.sleep(_POLL_SECONDS)
    raise TimeoutError("LTX 2.3 Comfy worker did not become ready")


def _queue(base_url: str, workflow: Mapping[str, object]) -> str:
    value = _json_request(base_url, "POST", "/prompt", {"prompt": workflow, "client_id": f"latentslate-{uuid4().hex}"})
    prompt_id = value.get("prompt_id") if isinstance(value, dict) else None
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError("ComfyUI rejected LTX 2.3 graph")
    return prompt_id


def _wait_video(
    base_url: str,
    prompt_id: str,
    progress: Callable[[float, str | None], None],
    check_cancelled: Callable[[], None],
    *,
    save_node_id: str,
) -> Mapping[str, str]:
    deadline = time.monotonic() + _JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        check_cancelled()
        history = _json_request(base_url, "GET", f"/history/{prompt_id}")
        item = history.get(prompt_id) if isinstance(history, dict) else None
        if isinstance(item, dict):
            status = item.get("status", {})
            if status.get("status_str") == "error" or status.get("completed") is False:
                raise RuntimeError("ComfyUI LTX 2.3 graph failed")
            outputs = item.get("outputs") or {}
            node = outputs.get(save_node_id) if isinstance(outputs, dict) else None
            values = node.get("gifs") or node.get("videos") if isinstance(node, dict) else None
            if isinstance(values, list) and len(values) == 1 and isinstance(values[0], dict):
                return values[0]
        queue = _json_request(base_url, "GET", "/queue")
        running = queue.get("queue_running") if isinstance(queue, dict) else None
        if isinstance(running, list) and any(
            isinstance(entry, (list, tuple)) and len(entry) > 1 and entry[1] == prompt_id
            for entry in running
        ):
            # Comfy's HTTP history endpoint does not expose node-step progress
            # while executing. Queue ownership is the earliest observable,
            # prompt-bound execution-adjacent fact; do not label it denoising.
            progress(0.10, _QUEUE_RUNNING_MESSAGE)
        time.sleep(_POLL_SECONDS)
    raise TimeoutError("LTX 2.3 Comfy generation exceeded its bounded timeout")


def _download_validate_and_publish(
    base_url: str,
    item: Mapping[str, str],
    output_path: Path,
    workspace: Path,
    request: LTX23ComfyRequest,
    check_cancelled: Callable[[], None],
) -> dict[str, object]:
    from urllib.parse import urlencode

    if not isinstance(item.get("filename"), str):
        raise TypeError("ComfyUI LTX 2.3 output lacks filename")
    query = urlencode({key: value for key, value in item.items() if key in {"filename", "subfolder", "type"}})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    private = workspace / "download.mp4"
    staged = output_path.with_name(f".{output_path.name}.{uuid4().hex}.partial")
    try:
        check_cancelled()
        with urlopen(base_url + "/view?" + query, timeout=120) as response:
            raw_length = getattr(response, "headers", {}).get("Content-Length")
            if raw_length is not None:
                try:
                    content_length = int(raw_length)
                except ValueError as exc:
                    raise RuntimeError("ComfyUI LTX 2.3 output length is invalid") from exc
                if content_length < 0 or content_length > _MAX_OUTPUT_BYTES:
                    raise RuntimeError("ComfyUI LTX 2.3 output exceeds its bounded size")
            total = 0
            with private.open("xb") as writer:
                while True:
                    check_cancelled()
                    chunk = response.read(_IO_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > _MAX_OUTPUT_BYTES:
                        raise RuntimeError("ComfyUI LTX 2.3 output exceeds its bounded size")
                    writer.write(chunk)
                    check_cancelled()
        check_cancelled()
        if total == 0:
            raise RuntimeError("ComfyUI LTX 2.3 output was empty")
        observed = _validate_mp4(private, request)
        check_cancelled()
        with private.open("rb") as reader, staged.open("xb") as writer:
            while True:
                check_cancelled()
                chunk = reader.read(_IO_CHUNK_BYTES)
                if not chunk:
                    break
                writer.write(chunk)
                check_cancelled()
        check_cancelled()
        os.replace(staged, output_path)
    finally:
        staged.unlink(missing_ok=True)
        private.unlink(missing_ok=True)
    return observed


def _validate_mp4(path: Path, request: LTX23ComfyRequest) -> dict[str, object]:
    """Validate observed mux facts before publishing an LTX result atomically."""

    command = [
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ]
    try:
        raw = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30).stdout
        probe = json.loads(raw)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError("could not validate LTX 2.3 Comfy MP4 output") from exc
    streams = probe.get("streams") if isinstance(probe, dict) else None
    container = probe.get("format") if isinstance(probe, dict) else None
    if not isinstance(streams, list):
        raise TypeError("LTX 2.3 Comfy output has no stream manifest")
    if not isinstance(container, dict) or "mp4" not in str(container.get("format_name", "")).split(","):
        raise RuntimeError("LTX 2.3 Comfy output is not an MP4 container")
    video = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"]
    audio = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"]
    expected_frames = _frames(request.duration_seconds)
    if len(video) != 1 or len(audio) != 1:
        raise RuntimeError("LTX 2.3 Comfy output must have exactly one video and one audio stream")
    video_stream, audio_stream = video[0], audio[0]
    frame_rate = str(video_stream.get("avg_frame_rate", ""))
    try:
        frame_count = int(video_stream["nb_read_frames"])
        fps = _fraction(frame_rate)
        video_duration = float(video_stream["duration"])
        audio_duration = float(audio_stream["duration"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeError("LTX 2.3 Comfy output lacks complete timing facts") from exc
    expected_duration = expected_frames / LTX23_COMFY_FPS
    if (
        video_stream.get("codec_name") != "h264"
        or video_stream.get("width") != request.width
        or video_stream.get("height") != request.height
        or abs(fps - LTX23_COMFY_FPS) > 1e-6
        or frame_count != expected_frames
        or audio_stream.get("sample_rate") != "48000"
        or audio_stream.get("channels") != 2
        or abs(video_duration - expected_duration) > 1 / LTX23_COMFY_FPS
        or abs(audio_duration - expected_duration) > 1 / LTX23_COMFY_FPS
        or abs(video_duration - audio_duration) > 1 / LTX23_COMFY_FPS
    ):
        raise RuntimeError("LTX 2.3 Comfy output does not meet the pinned MP4 A/V contract")
    return {
        "codec": "h264",
        "width": request.width,
        "height": request.height,
        "fps": LTX23_COMFY_FPS,
        "frame_count": expected_frames,
        "video_duration_seconds": video_duration,
        "audio_duration_seconds": audio_duration,
        "sample_rate": 48000,
        "channels": 2,
    }


def _fraction(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def _json_request(base_url: str, method: str, path: str, payload: object | None = None) -> object:
    data = None if payload is None else json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    request = Request(base_url + path, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read())


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _terminate_worker(
    tree: DisposableProcessTree | None,
    process: subprocess.Popen[str] | None,
    primary: BaseException,
) -> bool:
    """Attempt disposal without allowing cleanup to replace the root error."""

    try:
        if tree is not None:
            tree.terminate()
            tree.wait_for_empty()
            return True
        if process is not None:
            _terminate_process(process)
    except (OSError, RuntimeError, subprocess.SubprocessError) as cleanup_error:
        primary.add_note(
            "LTX 2.3 Comfy worker disposal failed: " + type(cleanup_error).__name__
        )
    return False


def _remove_failed_output(path: Path, primary: BaseException) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as cleanup_error:
        primary.add_note("LTX 2.3 Comfy output cleanup failed: " + type(cleanup_error).__name__)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _provenance(
    recipe: LTX23ComfyRuntimeRequest,
    request: LTX23ComfyRequest,
    workflow_sha256: str,
    worker: Mapping[str, object] | None,
    observed: Mapping[str, object],
) -> dict[str, object]:
    conditioning = {"mode": "text"}
    if recipe.operation == "comfy_dev_i2v":
        conditioning = {"mode": "first_frame", "ordered_indices": [0], "strength": 0.7}
    elif recipe.operation == "comfy_distilled_flf":
        conditioning = {"mode": "first_last_frame", "ordered_indices": [0, -1], "strength": LTX23_COMFY_GUIDE_STRENGTH}
    return {"backend": "comfyui/disposable-official-graph", "operation": recipe.operation, "template_revision": LTX23_COMFY_TEMPLATE_REVISION, "comfy_runtime_revision": LTX23_COMFY_RUNTIME_REVISION, "comfy_source_blobs": dict(_COMFY_REQUIRED_SOURCE_BLOBS), "raw_template_sha256": template_sha256(recipe.operation), "submitted_workflow_sha256": workflow_sha256, "recipe_fingerprint": recipe.fingerprint, "component_fingerprint": recipe.component_fingerprint, "components": recipe.public_component_manifest(), "sampling": {"main_sigmas": list(LTX23_COMFY_MAIN_SIGMAS), "upscale_sigmas": list(LTX23_COMFY_UPSCALE_SIGMAS) if recipe.operation != "comfy_distilled_flf" else None, "cfg": 1, "fps": LTX23_COMFY_FPS}, "conditioning": conditioning, "audio_video": {"has_audio": True, **dict(observed)}, "pipeline_warm": False, "cache": {"prompt_hit": False, "media_hit": False}, "worker": dict(worker or {})}


def _cleanup_workspace(workspace: Path) -> list[str]:
    try:
        shutil.rmtree(workspace)
    except OSError:
        # The status has no prompt, file name, or filesystem path, while still
        # making retained private hardlinks/inputs/logs observable to callers.
        return ["workspace"]
    return []


def _is_cancelled(exc: BaseException) -> bool:
    """Recognize the engine's cancellation type without a module import cycle.

    `tools.base` deliberately stays independent of runtimes. Importing its
    concrete exception only while handling a terminal error retains that
    direction and makes the accepted engine cancellation signal explicit.
    """

    if isinstance(exc, asyncio.CancelledError):
        return True
    from ..tools.base import ToolCancelled

    return isinstance(exc, ToolCancelled)
