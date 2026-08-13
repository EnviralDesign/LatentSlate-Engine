"""Identity-bound official support components for native Wan 2.2 I2V."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import IO, Any

from .umt5_stored_adapter import UMT5_XXL_CONFIG
from .wan21_vae_adapter import WAN21_VAE_CONFIG
from .wan22_prompt import ComfyWanTokenizer
from .wan22_stored_adapter import WAN22_14B_I2V_CONFIG

_MAX_JSON_BYTES = 1024 * 1024
_MAX_TOKENIZER_BYTES = 16 * 1024 * 1024
_REQUIRED_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/spiece.model",
    "transformer/config.json",
    "transformer_2/config.json",
    "text_encoder/config.json",
    "vae/config.json",
)
_PINNED_SCHEDULER_CONFIG = {
    "_class_name": "UniPCMultistepScheduler",
    "_diffusers_version": "0.35.0.dev0",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 3.0,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "bh2",
    "steps_offset": 0,
    "thresholding": False,
    "time_shift_type": "exponential",
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False,
}


@dataclass(frozen=True, slots=True)
class SupportFileIdentity:
    relative_path: str
    size_bytes: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class WanI2VSupportPlan:
    root: Path
    files: tuple[SupportFileIdentity, ...]
    fingerprint: str
    scheduler_config: Mapping[str, Any]
    tokenizer_payload: bytes
    tokenizer_sha256: str
    boundary_ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "scheduler_config", _deep_freeze(self.scheduler_config))

    def load_scheduler(self):
        from diffusers import UniPCMultistepScheduler

        if not revalidate_wan_i2v_support(self):
            raise ValueError("Wan support files changed after planning")
        return UniPCMultistepScheduler.from_config(_deep_thaw(self.scheduler_config))

    def load_tokenizer(self) -> ComfyWanTokenizer:
        return ComfyWanTokenizer.from_bytes(self.tokenizer_payload)


def plan_wan_i2v_support(root: Path) -> WanI2VSupportPlan:
    return _plan_wan_support(
        root,
        pipeline_class="WanImageToVideoPipeline",
        boundary_ratio=0.9,
        transformer_config=WAN22_14B_I2V_CONFIG,
    )


def _plan_wan_support(
    root: Path,
    *,
    pipeline_class: str,
    boundary_ratio: float,
    transformer_config: Mapping[str, Any],
) -> WanI2VSupportPlan:
    directory = Path(root).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Wan support root must be a directory")
    identities: list[SupportFileIdentity] = []
    payloads: dict[str, bytes] = {}
    for relative_path in _REQUIRED_FILES:
        identity, payload = _read_identity_bound_file(directory, relative_path)
        identities.append(identity)
        payloads[relative_path] = payload

    documents = {
        name: _parse_json(payloads[name], name)
        for name in _REQUIRED_FILES
        if name.endswith(".json")
    }
    _validate_support_documents(
        documents,
        pipeline_class=pipeline_class,
        boundary_ratio=boundary_ratio,
        transformer_config=transformer_config,
    )
    tokenizer = ComfyWanTokenizer.from_bytes(payloads["tokenizer/spiece.model"])
    scheduler = documents["scheduler/scheduler_config.json"]
    model_index = documents["model_index.json"]
    fingerprint = hashlib.sha256(
        json.dumps(
            [(item.relative_path, item.size_bytes, item.mtime_ns, item.sha256) for item in identities],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return WanI2VSupportPlan(
        root=directory,
        files=tuple(identities),
        fingerprint=fingerprint,
        scheduler_config=scheduler,
        tokenizer_payload=payloads["tokenizer/spiece.model"],
        tokenizer_sha256=tokenizer.model_sha256,
        boundary_ratio=float(model_index["boundary_ratio"]),
    )


def revalidate_wan_i2v_support(plan: WanI2VSupportPlan) -> bool:
    try:
        current = tuple(
            _read_identity_bound_file(plan.root, identity.relative_path, retain_payload=False)[0]
            for identity in plan.files
        )
    except (OSError, TypeError, ValueError):
        return False
    return current == plan.files


def _read_identity_bound_file(
    root: Path,
    relative_path: str,
    *,
    retain_payload: bool = True,
) -> tuple[SupportFileIdentity, bytes]:
    path = (root / relative_path).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Wan support file escapes its root: {relative_path!r}")
    limit = _MAX_TOKENIZER_BYTES if relative_path.endswith(".model") else _MAX_JSON_BYTES
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read(limit + 1)
        after = os.fstat(handle.fileno())
        opened_path = _opened_file_path(handle)
        if not opened_path.is_relative_to(root) or not stat.S_ISREG(after.st_mode):
            raise ValueError(f"Wan support file escapes its root after open: {relative_path!r}")
    if len(payload) > limit:
        raise ValueError(f"Wan support file exceeds its safety limit: {relative_path!r}")
    if not payload or _file_state(before) != _file_state(after):
        raise ValueError(f"Wan support file changed while reading: {relative_path!r}")
    digest = hashlib.sha256(payload).hexdigest()
    identity = SupportFileIdentity(relative_path, len(payload), after.st_mtime_ns, digest)
    return identity, payload if retain_payload else b""


def _file_state(stat_result: os.stat_result) -> tuple[int, int]:
    return stat_result.st_size, stat_result.st_mtime_ns


def _opened_file_path(handle: IO[bytes]) -> Path:
    if os.name == "nt":
        import ctypes
        import msvcrt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_final_path = kernel32.GetFinalPathNameByHandleW
        get_final_path.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32]
        get_final_path.restype = ctypes.c_uint32
        buffer = ctypes.create_unicode_buffer(32768)
        length = get_final_path(msvcrt.get_osfhandle(handle.fileno()), buffer, len(buffer), 0)
        if length == 0 or length >= len(buffer):
            raise OSError(ctypes.get_last_error(), "could not resolve opened Wan support file")
        value = buffer.value
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
        # GetFinalPathNameByHandleW with flags=0 already returns the normalized
        # DOS path for this exact open handle. A subsequent resolve/stat would
        # reintroduce a pathname race and discard that binding.
        return Path(value)

    for prefix in ("/proc/self/fd", "/dev/fd"):
        descriptor_path = Path(prefix) / str(handle.fileno())
        if descriptor_path.exists():
            return descriptor_path.resolve(strict=True)
    raise OSError("platform cannot resolve an opened Wan support file securely")


def _parse_json(payload: bytes, relative_path: str) -> dict[str, Any]:
    try:
        document = json.loads(payload, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Wan support JSON is invalid: {relative_path!r}") from exc
    if not isinstance(document, dict):
        raise TypeError(f"Wan support JSON must be an object: {relative_path!r}")
    return document


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _validate_support_documents(
    documents: Mapping[str, Mapping[str, Any]],
    *,
    pipeline_class: str,
    boundary_ratio: float,
    transformer_config: Mapping[str, Any],
) -> None:
    model_index = documents["model_index.json"]
    if (
        model_index.get("_class_name") != pipeline_class
        or model_index.get("transformer") != ["diffusers", "WanTransformer3DModel"]
        or model_index.get("transformer_2") != ["diffusers", "WanTransformer3DModel"]
        or model_index.get("vae") != ["diffusers", "AutoencoderKLWan"]
        or model_index.get("text_encoder") != ["transformers", "UMT5EncoderModel"]
        or not isinstance(model_index.get("boundary_ratio"), (int, float))
        or isinstance(model_index.get("boundary_ratio"), bool)
        or not math.isclose(float(model_index["boundary_ratio"]), boundary_ratio)
    ):
        raise ValueError("Wan support model_index does not match the pinned operation pipeline")

    scheduler = documents["scheduler/scheduler_config.json"]
    if _json_comparable(scheduler) != _json_comparable(_PINNED_SCHEDULER_CONFIG):
        raise ValueError("Wan support scheduler config does not exactly match the pinned runtime")

    for role in ("transformer", "transformer_2"):
        config = documents[f"{role}/config.json"]
        _require_fields(config, {"_class_name": "WanTransformer3DModel", **dict(transformer_config)}, role)

    text = documents["text_encoder/config.json"]
    _require_fields(
        text,
        {
            "architectures": ["UMT5EncoderModel"],
            **{
                key: UMT5_XXL_CONFIG[key]
                for key in (
                    "vocab_size",
                    "d_model",
                    "d_kv",
                    "d_ff",
                    "num_layers",
                    "num_heads",
                    "feed_forward_proj",
                    "relative_attention_num_buckets",
                    "relative_attention_max_distance",
                    "layer_norm_epsilon",
                )
            },
        },
        "text encoder",
    )

    vae = documents["vae/config.json"]
    _require_fields(
        vae,
        {
            "_class_name": "AutoencoderKLWan",
            **{
                key: WAN21_VAE_CONFIG[key]
                for key in (
                    "base_dim",
                    "dim_mult",
                    "num_res_blocks",
                    "temperal_downsample",
                    "z_dim",
                    "latents_mean",
                    "latents_std",
                )
            },
        },
        "VAE",
    )


def _require_fields(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    mismatches = [
        key
        for key, value in expected.items()
        if _json_comparable(actual.get(key)) != _json_comparable(value)
    ]
    if mismatches:
        raise ValueError(f"Wan support {label} config mismatch: {', '.join(sorted(mismatches))}")


def _json_comparable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_comparable(item) for item in value]
    if isinstance(value, list):
        return [_json_comparable(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _json_comparable(item) for key, item in value.items()}
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value
