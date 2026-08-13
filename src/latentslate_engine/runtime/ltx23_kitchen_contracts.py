"""CPU-only contracts for the existing LTX 2.3 stored optimized artifacts.

This module is deliberately a *plan* boundary.  It reads SafeTensors headers,
proves the six exact artifacts are internally coherent, and records every
tensor's role.  It does not import a workflow runtime, load tensor payloads,
or select a CUDA kernel.  A later Engine-owned direct Kitchen adapter consumes
these immutable plans.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..artifacts import (
    _MAX_HEADER_BYTES,
    ArtifactIdentity,
    probe_safetensors,
    revalidate_artifact,
)

LTX23_STORED_FP8_CONTRACT = "comfy_quant/float8_e4m3fn_global"
LTX23_GEMMA_MIXED_CONTRACT = "comfy_quant/mixed_fp8_nvfp4"
LTX23_NATIVE_BF16_CONTRACT = "native/bf16"

# This is architecture metadata for the Engine-owned A/V shell, not a request
# to instantiate a third-party pipeline.  It is part of the plan fingerprint so
# a future shell cannot silently drift from the reviewed stored checkpoint.
LTX23_AV_TRANSFORMER_CONFIG: Mapping[str, int | bool] = MappingProxyType(
    {
        "num_layers": 48,
        "in_channels": 128,
        "out_channels": 128,
        "caption_channels": 3840,
        "cross_attention_dim": 4096,
        "audio_in_channels": 128,
        "audio_out_channels": 128,
        "audio_cross_attention_dim": 2048,
        "video_num_attention_heads": 32,
        "video_attention_head_dim": 128,
        "audio_num_attention_heads": 32,
        "audio_attention_head_dim": 64,
        "cross_attn_mod": True,
        "audio_cross_attn_mod": True,
        "gated_attn": True,
        "audio_gated_attn": True,
        "rope_type": "split",
        "rope_double_precision": True,
    }
)
LTX23_GEMMA_TEXT_CONFIG: Mapping[str, int | bool] = MappingProxyType(
    {
        "num_hidden_layers": 48,
        "hidden_size": 3840,
        "attention_head_dim": 256,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "mixed_nvfp4_layers": 302,
        "mixed_fp8_layers": 34,
        "vision_branch_present": True,
    }
)
LTX23_SPATIAL_UPSCALER_CONFIG: Mapping[str, int | bool] = MappingProxyType(
    {"spatial_scale": 2, "temporal_upsample": False, "residual_block_count": 8}
)


@dataclass(frozen=True, slots=True)
class LTX23ArtifactContract:
    """Pinned identity and structural facts for one supported stored file."""

    name: str
    filename: str
    size_bytes: int
    source_sha256: str
    header_sha256: str
    schema_sha256: str
    tensor_count: int
    dtypes: tuple[str, ...]
    quantization_contract: str
    component_counts: Mapping[str, int]
    config: Mapping[str, int | bool | float | str]
    expected_quantized_counts: Mapping[str, int]


LTX23_DEV_FP8 = LTX23ArtifactContract(
    name="ltx23_dev_fp8_checkpoint",
    filename="ltx-2.3-22b-dev-fp8.safetensors",
    size_bytes=29_145_431_166,
    source_sha256="28606c5b5a06ce56f896d4dfcb20f212739e07a68fbe48e53638188449d26450",
    header_sha256="e21e55fbbc308ef8f476041fc298af4f3b22d65811f079f12c63902972be6511",
    schema_sha256="0a69321952b31131924aef3b568f759cb7c25d3d2738467973976bbf2061e746",
    tensor_count=8939,
    dtypes=("BF16", "F32", "F8_E4M3"),
    quantization_contract=LTX23_STORED_FP8_CONTRACT,
    component_counts=MappingProxyType(
        {"transformer": 7436, "vae": 170, "audio_vae": 102, "vocoder": 1227, "text_projection": 4}
    ),
    config=LTX23_AV_TRANSFORMER_CONFIG,
    expected_quantized_counts=MappingProxyType({"fp8": 1496}),
)
LTX23_DISTILLED_FP8 = LTX23ArtifactContract(
    name="ltx23_distilled_fp8_checkpoint",
    filename="ltx-2.3-22b-distilled-fp8.safetensors",
    size_bytes=29_531_884_062,
    source_sha256="d9646b6f2d5c42d337b23671634c43bfeece6989644f51b4a3aa088465ccd3b2",
    header_sha256="439f0abe2a6b6b6f220ab1099562fcbf5cd419632207b5326d5e16502abb53d1",
    schema_sha256="124c441187373cba2d758847ec2254fa28d3e6fc6f9bec292d905139732a5d73",
    tensor_count=8871,
    dtypes=("BF16", "F32", "F8_E4M3"),
    quantization_contract=LTX23_STORED_FP8_CONTRACT,
    component_counts=MappingProxyType(
        {"transformer": 7368, "vae": 170, "audio_vae": 102, "vocoder": 1227, "text_projection": 4}
    ),
    config=LTX23_AV_TRANSFORMER_CONFIG,
    expected_quantized_counts=MappingProxyType({"fp8": 1462}),
)
LTX23_GEMMA_MIXED = LTX23ArtifactContract(
    name="ltx23_gemma_3_12b_mixed_text",
    filename="gemma_3_12B_it_fp4_mixed.safetensors",
    size_bytes=9_447_702_218,
    source_sha256="aaca463d11e6d8d2a4bdb0d6299214c15ef78a3f73e0ef8113d5a9d0219b3f6d",
    header_sha256="7fcb15cd769553ffe4c1fd41c050d7c99ae6845e1aa03e3cd0c3803ffc1919c7",
    schema_sha256="ddf523b18b1a724da6d4a3b0a97d4305ad3ad02a89ab7ada299663a9047040cd",
    tensor_count=2040,
    dtypes=("BF16", "F32", "F8_E4M3", "U8"),
    quantization_contract=LTX23_GEMMA_MIXED_CONTRACT,
    component_counts=MappingProxyType(
        {"language_model": 1600, "vision_model": 437, "multimodal_projector": 2, "sentencepiece": 1}
    ),
    config=LTX23_GEMMA_TEXT_CONFIG,
    expected_quantized_counts=MappingProxyType({"nvfp4": 302, "fp8": 34}),
)
LTX23_MODEL_LORA = LTX23ArtifactContract(
    name="ltx23_distilled_model_lora",
    filename="ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
    size_bytes=2_741_024_390,
    source_sha256="31e0c0195fb841bf31af78e8b60858f489e87ddcea4a5239abc80943da65e3ac",
    header_sha256="c2496ef7db5834bb9d8e0823d25ec107578c2db6d2eb85ed33fea858d41d213b",
    schema_sha256="f5d65b851a5e6fe5eb7ad4e0e4e2051ff9d36bcea75557e72923f606de51134f",
    tensor_count=4979,
    dtypes=("BF16",),
    quantization_contract=LTX23_NATIVE_BF16_CONTRACT,
    component_counts=MappingProxyType({"model_lora": 4979}),
    config=MappingProxyType({"alpha_optional": True}),
    expected_quantized_counts=MappingProxyType({}),
)
LTX23_TEXT_LORA = LTX23ArtifactContract(
    name="ltx23_gemma_abliterated_text_lora",
    filename="gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
    size_bytes=628_203_616,
    source_sha256="87bcabeac9bec9f374232b5122d6511c2b2112d479e50176149e944b3712eb4a",
    header_sha256="fe14997e4f5a457b9c070421788fdb7b6674c52583db2982edd1d4838d85c233",
    schema_sha256="601c8857a7d830f05f80792e044f97df6df8ff125079d5a305f3de5a2999d027",
    tensor_count=1000,
    dtypes=("BF16",),
    quantization_contract=LTX23_NATIVE_BF16_CONTRACT,
    component_counts=MappingProxyType({"text_lora": 1000}),
    config=MappingProxyType({"rank": 64}),
    expected_quantized_counts=MappingProxyType({}),
)
LTX23_SPATIAL_UPSCALER = LTX23ArtifactContract(
    name="ltx23_spatial_upscaler_x2",
    filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    size_bytes=995_743_560,
    source_sha256="5f416311fa8172b65af67530758964708d29a317b830d689a51143b7f91913ed",
    header_sha256="235db5957b7680c3b586033d77617099a36753681089fbe7dc1b9e87103ee97c",
    schema_sha256="ccd7989113ce24be08ef9cfcaf135ff85594f3f44e34be231dd4b426d4482c34",
    tensor_count=72,
    dtypes=("BF16",),
    quantization_contract=LTX23_NATIVE_BF16_CONTRACT,
    component_counts=MappingProxyType({"latent_upscaler": 72}),
    config=LTX23_SPATIAL_UPSCALER_CONFIG,
    expected_quantized_counts=MappingProxyType({}),
)
LTX23_OPTIMIZED_ARTIFACTS: Mapping[str, LTX23ArtifactContract] = MappingProxyType(
    {
        item.name: item
        for item in (
            LTX23_DEV_FP8,
            LTX23_DISTILLED_FP8,
            LTX23_GEMMA_MIXED,
            LTX23_MODEL_LORA,
            LTX23_TEXT_LORA,
            LTX23_SPATIAL_UPSCALER,
        )
    }
)


@dataclass(frozen=True, slots=True)
class LTX23StoredArtifactPlan:
    """Header-only proof and complete role assignment for one artifact."""

    contract: LTX23ArtifactContract
    identity: ArtifactIdentity
    roles: Mapping[str, str]
    component_counts: Mapping[str, int]
    quantized_layers: Mapping[str, tuple[str, ...]]
    auxiliary_sources: tuple[str, ...]
    errors: tuple[str, ...]
    fingerprint: str

    @property
    def available(self) -> bool:
        return not self.errors

    def require_available(self) -> None:
        if self.errors:
            raise ValueError(f"{self.contract.name} is unavailable: " + "; ".join(self.errors))

    def revalidate(self) -> bool:
        """Re-read the file identity before a future payload materialization."""

        return revalidate_artifact(self.identity)


def plan_ltx23_stored_artifact(
    artifact_path: Path, contract: LTX23ArtifactContract
) -> LTX23StoredArtifactPlan:
    """Fail closed unless ``artifact_path`` exactly fits ``contract``'s header."""

    probe = probe_safetensors(Path(artifact_path).resolve(strict=True))
    entries, metadata = _read_header(probe.identity.path)
    errors = _identity_errors(probe, contract)
    roles, quantized, auxiliaries, structural_errors = _classify(entries, metadata, contract)
    errors.extend(structural_errors)
    classified = set(roles)
    if classified != set(entries):
        errors.append("every SafeTensors entry must receive exactly one stored-artifact role")
    component_counts = Counter(_component_for_role(role) for role in roles.values())
    expected_components = dict(contract.component_counts)
    if dict(component_counts) != expected_components:
        errors.append(
            f"component counts {dict(component_counts)!r} do not match {expected_components!r}"
        )
    actual_quantized_counts = {kind: len(values) for kind, values in quantized.items()}
    if actual_quantized_counts != dict(contract.expected_quantized_counts):
        errors.append(
            "quantized layer counts "
            f"{actual_quantized_counts!r} do not match {dict(contract.expected_quantized_counts)!r}"
        )
    fingerprint = _fingerprint(
        {
            "contract": contract.name,
            "source_sha256": contract.source_sha256,
            "schema_sha256": probe.schema_sha256,
            "config": dict(contract.config),
            "roles": roles,
            "quantized_layers": {kind: sorted(layers) for kind, layers in quantized.items()},
        }
    )
    return LTX23StoredArtifactPlan(
        contract=contract,
        identity=probe.identity,
        roles=MappingProxyType(dict(sorted(roles.items()))),
        component_counts=MappingProxyType(dict(sorted(component_counts.items()))),
        quantized_layers=MappingProxyType(
            {kind: tuple(sorted(values)) for kind, values in sorted(quantized.items())}
        ),
        auxiliary_sources=tuple(sorted(auxiliaries)),
        errors=tuple(errors),
        fingerprint=fingerprint,
    )


def _identity_errors(probe: Any, contract: LTX23ArtifactContract) -> list[str]:
    errors: list[str] = []
    facts = {
        "filename": probe.identity.path.name,
        "size_bytes": probe.identity.size_bytes,
        "header_sha256": probe.identity.header_sha256,
        "schema_sha256": probe.schema_sha256,
        "tensor_count": probe.tensor_count,
        "dtypes": probe.tensor_dtypes,
    }
    expected = {
        "filename": contract.filename,
        "size_bytes": contract.size_bytes,
        "header_sha256": contract.header_sha256,
        "schema_sha256": contract.schema_sha256,
        "tensor_count": contract.tensor_count,
        "dtypes": contract.dtypes,
    }
    for name, value in expected.items():
        if facts[name] != value:
            errors.append(f"{name} mismatch: expected {value!r}, found {facts[name]!r}")
    return errors


def _classify(
    entries: Mapping[str, Mapping[str, Any]], metadata: Mapping[str, Any], contract: LTX23ArtifactContract
) -> tuple[dict[str, str], dict[str, set[str]], set[str], list[str]]:
    if contract is LTX23_DEV_FP8 or contract is LTX23_DISTILLED_FP8 or contract.name == "checkpoint":
        return _classify_checkpoint(entries, metadata)
    if contract is LTX23_GEMMA_MIXED:
        return _classify_gemma(entries)
    if contract is LTX23_MODEL_LORA:
        return _classify_model_lora(entries)
    if contract is LTX23_TEXT_LORA:
        return _classify_text_lora(entries)
    if contract is LTX23_SPATIAL_UPSCALER:
        return _classify_upscaler(entries)
    return {}, {}, set(), [f"unsupported LTX 2.3 artifact contract {contract.name!r}"]


def _classify_checkpoint(
    entries: Mapping[str, Mapping[str, Any]], metadata: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, set[str]], set[str], list[str]]:
    roles: dict[str, str] = {}
    errors: list[str] = []
    components = {
        "model.diffusion_model.": "transformer",
        "vae.": "vae",
        "audio_vae.": "audio_vae",
        "vocoder.": "vocoder",
        "text_embedding_projection.": "text_projection",
    }
    for key in entries:
        component = next((role for prefix, role in components.items() if key.startswith(prefix)), None)
        if component is None:
            errors.append(f"unrecognized checkpoint component tensor {key!r}")
        else:
            roles[key] = component + "/dense"
    raw = metadata.get("_quantization_metadata")
    try:
        quantization = json.loads(raw) if isinstance(raw, str) else None
    except json.JSONDecodeError:
        quantization = None
    if not isinstance(quantization, dict) or quantization.get("format_version") != "1.0":
        return roles, {}, set(), [*errors, "missing or invalid FP8 quantization metadata"]
    layer_data = quantization.get("layers")
    if not isinstance(layer_data, dict) or not layer_data:
        return roles, {}, set(), [*errors, "FP8 quantization metadata has no layers"]
    layers: set[str] = set()
    auxiliaries: set[str] = set()
    for stem, description in layer_data.items():
        if not isinstance(stem, str) or description != {"format": "float8_e4m3fn"}:
            errors.append(f"invalid FP8 layer metadata for {stem!r}")
            continue
        weight, weight_scale, input_scale = (stem + suffix for suffix in (".weight", ".weight_scale", ".input_scale"))
        expected = {weight: "F8_E4M3", weight_scale: "F32", input_scale: "F32"}
        if any(entries.get(key, {}).get("dtype") != dtype for key, dtype in expected.items()):
            errors.append(f"incomplete FP8 payload/scale trio for {stem!r}")
            continue
        if not stem.startswith("model.diffusion_model."):
            errors.append(f"FP8 layer outside transformer component: {stem!r}")
            continue
        roles[weight] = "transformer/fp8_weight"
        roles[weight_scale] = "transformer/fp8_weight_scale"
        roles[input_scale] = "transformer/fp8_input_scale"
        layers.add(stem)
        auxiliaries.update((weight_scale, input_scale))
    actual_fp8_weights = {key.removesuffix(".weight") for key, entry in entries.items() if key.endswith(".weight") and entry.get("dtype") == "F8_E4M3"}
    if actual_fp8_weights != layers:
        errors.append("FP8 metadata layer set does not exactly match FP8 weight tensor set")
    return roles, {"fp8": layers}, auxiliaries, errors


def _classify_gemma(
    entries: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, str], dict[str, set[str]], set[str], list[str]]:
    roles: dict[str, str] = {}
    errors: list[str] = []
    for key in entries:
        if key.startswith("model."):
            roles[key] = "language_model/dense"
        elif key.startswith("vision_model."):
            roles[key] = "vision_model/dense"
        elif key.startswith("multi_modal_projector."):
            roles[key] = "multimodal_projector/dense"
        elif key == "spiece_model":
            roles[key] = "sentencepiece/dense"
        else:
            errors.append(f"unrecognized Gemma component tensor {key!r}")
    packed = {
        key.removesuffix(".weight")
        for key, entry in entries.items()
        if key.endswith(".weight") and entry.get("dtype") == "U8"
    }
    fp8 = {
        key.removesuffix(".weight")
        for key, entry in entries.items()
        if key.endswith(".weight") and entry.get("dtype") == "F8_E4M3"
    }
    auxiliaries: set[str] = set()
    for kind, stems, scale_dtype, second_scale in (
        ("nvfp4", packed, "F8_E4M3", True),
        ("fp8", fp8, "F32", False),
    ):
        for stem in stems:
            component = _component_for_role(roles.get(stem + ".weight", ""))
            companion = {stem + ".weight_scale": scale_dtype, stem + ".comfy_quant": "U8"}
            if second_scale:
                companion[stem + ".weight_scale_2"] = "F32"
            if any(entries.get(key, {}).get("dtype") != dtype for key, dtype in companion.items()):
                errors.append(f"incomplete {kind} Gemma tensor group for {stem!r}")
                continue
            roles[stem + ".weight"] = component + f"/{kind}_weight"
            for key in companion:
                roles[key] = component + f"/{kind}_auxiliary"
            auxiliaries.update(companion)
    if packed & fp8:
        errors.append("a Gemma weight stem cannot be both NVFP4 and FP8")
    return roles, {"nvfp4": packed, "fp8": fp8}, auxiliaries, errors


def _classify_model_lora(
    entries: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, str], dict[str, set[str]], set[str], list[str]]:
    return _classify_lora(entries, "model_lora", ".lora_A.weight", ".lora_B.weight", alpha_suffix=".alpha")


def _classify_text_lora(
    entries: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, str], dict[str, set[str]], set[str], list[str]]:
    roles, quantized, auxiliaries, errors = _classify_lora(
        entries, "text_lora", ".lora_down.weight", ".lora_up.weight", alpha_suffix=None
    )
    for key in entries:
        if not key.startswith("text_encoders.transformer."):
            errors.append(f"text LoRA target outside Gemma encoder {key!r}")
    for key, role in tuple(roles.items()):
        if ".vision_model." in key:
            roles[key] = "text_lora_vision/" + role.rsplit("/", 1)[1]
    if any(entry.get("shape", [None])[0] != 64 for key, entry in entries.items() if key.endswith(".lora_down.weight")):
        errors.append("text LoRA must retain rank 64 on every down projection")
    return roles, quantized, auxiliaries, errors


def _classify_lora(
    entries: Mapping[str, Mapping[str, Any]], component: str, down_suffix: str, up_suffix: str, *, alpha_suffix: str | None
) -> tuple[dict[str, str], dict[str, set[str]], set[str], list[str]]:
    roles: dict[str, str] = {}
    errors: list[str] = []
    down = {key.removesuffix(down_suffix) for key in entries if key.endswith(down_suffix)}
    up = {key.removesuffix(up_suffix) for key in entries if key.endswith(up_suffix)}
    if down != up:
        errors.append("LoRA down/up target sets differ")
    for stem in down | up:
        down_key, up_key = stem + down_suffix, stem + up_suffix
        down_entry, up_entry = entries.get(down_key), entries.get(up_key)
        if not isinstance(down_entry, dict) or not isinstance(up_entry, dict):
            continue
        down_shape, up_shape = down_entry.get("shape"), up_entry.get("shape")
        if (
            down_entry.get("dtype") != "BF16"
            or up_entry.get("dtype") != "BF16"
            or not _lora_shapes_match(down_shape, up_shape)
        ):
            errors.append(f"invalid BF16 LoRA pair for {stem!r}")
        roles[down_key] = component + "/down"
        roles[up_key] = component + "/up"
        if alpha_suffix is not None and stem + alpha_suffix in entries:
            alpha = entries[stem + alpha_suffix]
            if alpha.get("dtype") != "BF16" or alpha.get("shape") != []:
                errors.append(f"invalid LoRA alpha for {stem!r}")
            roles[stem + alpha_suffix] = component + "/alpha"
    unrecognized = set(entries) - set(roles)
    if unrecognized:
        errors.append(f"unrecognized LoRA tensors: {len(unrecognized)}")
    return roles, {}, set(), errors


def _classify_upscaler(
    entries: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, str], dict[str, set[str]], set[str], list[str]]:
    roots = ("initial_conv.", "initial_norm.", "res_blocks.", "upsampler.", "post_upsample_res_blocks.", "final_conv.")
    roles: dict[str, str] = {}
    errors: list[str] = []
    for key, entry in entries.items():
        if not key.startswith(roots):
            errors.append(f"unrecognized latent-upscaler tensor {key!r}")
            continue
        if entry.get("dtype") != "BF16":
            errors.append(f"latent-upscaler tensor {key!r} is not BF16")
        roles[key] = "latent_upscaler/dense"
    block_indices = {int(match.group(1)) for key in entries if (match := re.match(r"(?:res_blocks|post_upsample_res_blocks)\.(\d+)\.", key))}
    if block_indices != set(range(4)):
        errors.append("latent upscaler must contain four pre/post residual blocks")
    return roles, {}, set(), errors


def _component_for_role(role: str) -> str:
    component = role.split("/", 1)[0]
    return "text_lora" if component == "text_lora_vision" else component


def _lora_shapes_match(down: Any, up: Any) -> bool:
    return (
        isinstance(down, list)
        and isinstance(up, list)
        and len(down) == len(up) == 2
        and all(isinstance(value, int) and value > 0 for value in (*down, *up))
        and down[0] == up[1]
    )


def _read_header(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if header_size > _MAX_HEADER_BYTES:
            raise ValueError("LTX 2.3 SafeTensors header exceeds inspection limit")
        raw = stream.read(header_size)
    header = json.loads(raw)
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise TypeError("LTX 2.3 SafeTensors metadata must be an object")
    return header, metadata


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
