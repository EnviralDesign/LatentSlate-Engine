"""Exact support and dense-component contracts for Comfy-aligned Klein 4B.

The official Comfy workflows bind three standalone SafeTensors artifacts to a
small Diffusers pipeline shell.  This module keeps that topology literal: it
validates the pinned shell, Qwen3-4B encoder, and Flux2 VAE independently and
loads the two dense components directly from their SafeTensors files.  It does
not copy, convert, or silently source weights from the support directory.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from ..artifacts import ArtifactIdentity, probe_artifact, revalidate_artifact
from .kit import stable_fingerprint

_DISTILLED_SUPPORT_FILES: Mapping[str, tuple[int, str]] = MappingProxyType(
    {
        "model_index.json": (
            446,
            "51a76cb1cf3ed37423a1128c79c22faee8e6fbe7f5aaeb737f0a258930dbaac0",
        ),
        "scheduler/scheduler_config.json": (
            486,
            "067afb012cef64553a763447d1efd93daeffcc0123ca7e25b09f8de20b90762e",
        ),
        "text_encoder/config.json": (
            1_536,
            "214b4c29a0d975e9fddf9994a5673f22cb2c4c5750352f9227c2c3251ebeab40",
        ),
        "text_encoder/generation_config.json": (
            214,
            "4347b1aeed2b2b78bc059920a0b7f5fec71482e1344952b76d7665d638d71f13",
        ),
        "transformer/config.json": (
            541,
            "09733c74a3da6d17dd0a0472a091a8950c7c6935889c32c16cc800ede05029de",
        ),
        "vae/config.json": (
            821,
            "0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948",
        ),
        "tokenizer/added_tokens.json": (
            707,
            "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680",
        ),
        "tokenizer/chat_template.jinja": (
            4_168,
            "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        ),
        "tokenizer/merges.txt": (
            1_671_853,
            "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        ),
        "tokenizer/special_tokens_map.json": (
            613,
            "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
        ),
        "tokenizer/tokenizer.json": (
            11_422_654,
            "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        ),
        "tokenizer/tokenizer_config.json": (
            5_404,
            "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0",
        ),
        "tokenizer/vocab.json": (
            2_776_833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
    }
)
_BASE_SUPPORT_FILES: Mapping[str, tuple[int, str]] = MappingProxyType(
    {
        **_DISTILLED_SUPPORT_FILES,
        "model_index.json": (
            422,
            "de1fef01845f0af3e1e2f2581cd5e86f2dfa4a4776f7f0b26b4c77feac4ee70f",
        ),
        "transformer/config.json": (
            531,
            "c8a88c1433f24e52c01f46766c9f519e76d91d1c00bfb43fe0db2e4ed3c84d70",
        ),
    }
)
_SUPPORT_BY_MODE = {
    "base": _BASE_SUPPORT_FILES,
    "distilled": _DISTILLED_SUPPORT_FILES,
}

KLEIN_QWEN_SCHEMA_SHA256 = "e0a22a9523c6c3a8e298311bc7389a035f1ac6081133b71067629bd72ac5899d"
KLEIN_VAE_SCHEMA_SHA256 = "b7f4c62be021cd42d7cd16949f9532e2aceefaee4d933d838cc8f1773ef3cc99"
KLEIN_SMALL_VAE_SCHEMA_SHA256 = (
    "95077451a03ea87d23b895545a0b109dd6287f156de6a175b950e75a06e5ba2c"
)
KLEIN_DISTILLED_TRANSFORMER_SCHEMA_SHA256 = (
    "2ff21124fb997716c2da1597fab0824dc7bedcaf1aa182ade036e46788b79d6b"
)
KLEIN_BASE_TRANSFORMER_SCHEMA_SHA256 = (
    "ab6623aee9179bfe9fd287e196795e042b471facde107f98cd84e25110e2d6b3"
)
_SMALL_VAE_ARCHITECTURE = "flux2_small_decoder_full_encoder"
_FULL_VAE_ARCHITECTURE = "flux2_vae"
_QWEN_ARCHITECTURE = "qwen3_4b"
_SMALL_VAE_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ("DownEncoderBlock2D",) * 4,
        "up_block_types": ("UpDecoderBlock2D",) * 4,
        "block_out_channels": (128, 256, 512, 512),
        "decoder_block_out_channels": (96, 192, 384, 384),
        "layers_per_block": 2,
        "act_fn": "silu",
        "latent_channels": 32,
        "norm_num_groups": 32,
        "sample_size": 1024,
        "force_upcast": True,
        "use_quant_conv": True,
        "use_post_quant_conv": True,
        "mid_block_add_attention": True,
        "batch_norm_eps": 0.0001,
        "batch_norm_momentum": 0.1,
        "patch_size": (2, 2),
    }
)


@dataclass(frozen=True, slots=True)
class KleinPipelineSupportPlan:
    mode: str
    root: Path
    files: Mapping[str, tuple[int, str]]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class KleinDenseComponentPlan:
    role: str
    architecture: str
    identity: ArtifactIdentity
    schema_sha256: str
    tensor_count: int
    tensor_dtypes: tuple[str, ...]


def plan_klein_pipeline_support(path: Path, mode: str) -> KleinPipelineSupportPlan:
    """Validate the exact 15.9 MB config/tokenizer/scheduler shell."""

    try:
        support_files = _SUPPORT_BY_MODE[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported Klein support mode: {mode!r}") from exc
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Klein pipeline support must be a directory")

    present = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
        and ".cache" not in item.relative_to(root).parts
        and item.name != ".latentslate-model.toml"
    }
    expected = set(support_files)
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing={missing[:3]}")
        if extra:
            details.append(f"unexpected={extra[:3]}")
        raise ValueError("Klein pipeline support is not the exact bounded shell: " + "; ".join(details))

    verified: dict[str, tuple[int, str]] = {}
    for relative, expected_identity in support_files.items():
        candidate = (root / relative).resolve(strict=True)
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"Klein support file escapes its root: {relative}")
        before = candidate.stat()
        digest = _sha256_file(candidate)
        after = candidate.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Klein support file changed during validation: {relative}")
        actual = (after.st_size, digest)
        if actual != expected_identity:
            raise ValueError(f"Klein support file identity mismatch: {relative}")
        verified[relative] = actual

    _validate_support_semantics(root, mode)
    fingerprint = stable_fingerprint(
        "klein4-support",
        {"mode": mode, "files": [(name, *verified[name]) for name in sorted(verified)]},
    )
    return KleinPipelineSupportPlan(mode, root, MappingProxyType(verified), fingerprint)


def revalidate_klein_pipeline_support(plan: KleinPipelineSupportPlan) -> bool:
    try:
        refreshed = plan_klein_pipeline_support(plan.root, plan.mode)
    except (OSError, TypeError, ValueError):
        return False
    return refreshed.files == plan.files and refreshed.fingerprint == plan.fingerprint


def plan_klein_text_encoder(path: Path) -> KleinDenseComponentPlan:
    return _plan_dense_component(
        path,
        role="text_encoder",
        architecture=_QWEN_ARCHITECTURE,
        size_bytes=8_044_982_048,
        schema_sha256=KLEIN_QWEN_SCHEMA_SHA256,
        tensor_count=398,
        tensor_dtypes=("BF16",),
        contract="native/bf16",
    )


def plan_klein_vae(path: Path) -> KleinDenseComponentPlan:
    return _plan_dense_component(
        path,
        role="vae",
        architecture=_FULL_VAE_ARCHITECTURE,
        size_bytes=336_213_556,
        schema_sha256=KLEIN_VAE_SCHEMA_SHA256,
        tensor_count=251,
        tensor_dtypes=("F32", "I64"),
        contract="native/fp32",
    )


def plan_klein_small_vae(path: Path) -> KleinDenseComponentPlan:
    """Plan the exact FLUX.2 small-decoder file used by Comfy base I2I."""

    return _plan_dense_component(
        path,
        role="vae",
        architecture=_SMALL_VAE_ARCHITECTURE,
        size_bytes=249_519_092,
        schema_sha256=KLEIN_SMALL_VAE_SCHEMA_SHA256,
        tensor_count=251,
        tensor_dtypes=("F32", "I64"),
        contract="native/fp32",
    )


def revalidate_klein_dense_component(plan: KleinDenseComponentPlan) -> bool:
    try:
        planners = {
            _QWEN_ARCHITECTURE: plan_klein_text_encoder,
            _FULL_VAE_ARCHITECTURE: plan_klein_vae,
            _SMALL_VAE_ARCHITECTURE: plan_klein_small_vae,
        }
        refreshed = planners[plan.architecture](plan.identity.path)
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return refreshed == plan and revalidate_artifact(plan.identity)


def load_klein_text_encoder(plan: KleinDenseComponentPlan, support_root: Path) -> Any:
    """Materialize the exact standalone Qwen file without a copied HF folder."""

    if plan.role != "text_encoder" or not revalidate_klein_dense_component(plan):
        raise ValueError("Klein text encoder changed after planning")

    from accelerate import init_empty_weights, load_checkpoint_in_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config.from_pretrained(
        Path(support_root) / "text_encoder",
        local_files_only=True,
    )
    with init_empty_weights():
        model = Qwen3ForCausalLM(config)
    # The official standalone file intentionally omits the tied lm_head.  Its
    # complete 398-key schema is validated above; after direct low-memory load,
    # bind the sole absent parameter to the authoritative embedding weight.
    load_checkpoint_in_model(
        model,
        plan.identity.path,
        device_map={"": "cpu"},
        dtype=torch.bfloat16,
        strict=False,
    )
    meta = [name for name, value in model.state_dict().items() if value.is_meta]
    if meta != ["lm_head.weight"]:
        raise RuntimeError(f"Klein Qwen load left unexpected meta parameters: {meta[:3]}")
    model.lm_head.weight = model.model.embed_tokens.weight
    _require_loaded_state(model, {torch.bfloat16}, "Klein Qwen")
    model.eval()
    return model


def load_klein_vae(plan: KleinDenseComponentPlan, support_root: Path) -> Any:
    """Materialize the exact standalone FP32 Flux2 VAE from SafeTensors."""

    if plan.role != "vae" or not revalidate_klein_dense_component(plan):
        raise ValueError("Klein VAE changed after planning")

    from accelerate import init_empty_weights, load_checkpoint_in_model
    from diffusers import AutoencoderKLFlux2

    if plan.architecture == _SMALL_VAE_ARCHITECTURE:
        config = dict(_SMALL_VAE_CONFIG)
    elif plan.architecture == _FULL_VAE_ARCHITECTURE:
        config = AutoencoderKLFlux2.load_config(
            Path(support_root) / "vae", local_files_only=True
        )
    else:
        raise ValueError(f"Unsupported Klein VAE architecture: {plan.architecture!r}")
    with init_empty_weights():
        model = AutoencoderKLFlux2.from_config(config)
    if plan.architecture == _SMALL_VAE_ARCHITECTURE:
        _load_small_vae_checkpoint(model, plan.identity.path)
    else:
        load_checkpoint_in_model(
            model,
            plan.identity.path,
            device_map={"": "cpu"},
            dtype=None,
            strict=True,
        )
    _require_loaded_state(model, {torch.float32, torch.int64}, "Klein VAE")
    model.eval()
    return model


def _plan_dense_component(
    path: Path,
    *,
    role: str,
    architecture: str,
    size_bytes: int,
    schema_sha256: str,
    tensor_count: int,
    tensor_dtypes: tuple[str, ...],
    contract: str,
) -> KleinDenseComponentPlan:
    probe = probe_artifact(Path(path))
    errors: list[str] = []
    if probe.format != "safetensors":
        errors.append("container is not SafeTensors")
    if probe.identity.size_bytes != size_bytes:
        errors.append("file size differs from the pinned artifact")
    if probe.schema_sha256 != schema_sha256:
        errors.append("key/shape/dtype schema differs from the pinned artifact")
    if probe.tensor_count != tensor_count or probe.tensor_dtypes != tensor_dtypes:
        errors.append("tensor count or stored dtypes differ from the pinned artifact")
    if probe.quantization_contract != contract:
        errors.append("stored precision contract differs from the pinned artifact")
    if errors:
        raise ValueError(f"Klein {role} contract failed: " + "; ".join(errors))
    return KleinDenseComponentPlan(
        role,
        architecture,
        probe.identity,
        probe.schema_sha256,
        probe.tensor_count,
        probe.tensor_dtypes,
    )


def _load_small_vae_checkpoint(model: Any, path: Path) -> None:
    """Map the exact Comfy/LDM names into Diffusers without a converted copy."""

    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    target = model.state_dict()
    with safe_open(path, framework="pt", device="cpu") as tensors:
        # SafeTensors' native safe_open handle exposes keys() but is not iterable.
        mapping = {
            source: _map_small_vae_key(source)
            for source in tensors.keys()  # noqa: SIM118
        }
        if None in mapping.values() or len(set(mapping.values())) != len(mapping):
            raise RuntimeError("Klein small VAE key mapping is incomplete or ambiguous")
        if set(mapping.values()) != set(target):
            raise RuntimeError("Klein small VAE does not exactly cover the Diffusers module")

        for source, destination in mapping.items():
            assert destination is not None
            source_shape = tuple(tensors.get_slice(source).get_shape())
            target_shape = tuple(target[destination].shape)
            squeezable = (
                len(source_shape) in {3, 4}
                and source_shape[:2] == target_shape
                and all(value == 1 for value in source_shape[2:])
            )
            if source_shape != target_shape and not squeezable:
                raise RuntimeError(
                    f"Klein small VAE shape mismatch: {source} -> {destination}"
                )

        for source, destination in mapping.items():
            assert destination is not None
            value = tensors.get_tensor(source)
            if tuple(value.shape) != tuple(target[destination].shape):
                value = value.reshape(value.shape[:2])
            set_module_tensor_to_device(model, destination, "cpu", value=value)


def _map_small_vae_key(source: str) -> str | None:
    direct = {
        "encoder.norm_out.weight": "encoder.conv_norm_out.weight",
        "encoder.norm_out.bias": "encoder.conv_norm_out.bias",
        "decoder.norm_out.weight": "decoder.conv_norm_out.weight",
        "decoder.norm_out.bias": "decoder.conv_norm_out.bias",
        "encoder.quant_conv.weight": "quant_conv.weight",
        "encoder.quant_conv.bias": "quant_conv.bias",
        "decoder.post_quant_conv.weight": "post_quant_conv.weight",
        "decoder.post_quant_conv.bias": "post_quant_conv.bias",
    }
    if source in direct:
        return direct[source]
    if source.startswith(("encoder.conv_", "decoder.conv_", "bn.")):
        return source

    value = source.replace("nin_shortcut", "conv_shortcut")
    patterns = (
        (
            r"encoder\.down\.(\d+)\.block\.(\d+)\.(.+)",
            lambda match: (
                f"encoder.down_blocks.{match[1]}.resnets.{match[2]}.{match[3]}"
            ),
        ),
        (
            r"encoder\.down\.(\d+)\.downsample\.conv\.(.+)",
            lambda match: f"encoder.down_blocks.{match[1]}.downsamplers.0.conv.{match[2]}",
        ),
        (
            r"encoder\.mid\.block_(\d+)\.(.+)",
            lambda match: f"encoder.mid_block.resnets.{int(match[1]) - 1}.{match[2]}",
        ),
        (
            r"decoder\.up\.(\d+)\.block\.(\d+)\.(.+)",
            lambda match: (
                f"decoder.up_blocks.{3 - int(match[1])}.resnets.{match[2]}.{match[3]}"
            ),
        ),
        (
            r"decoder\.up\.(\d+)\.upsample\.conv\.(.+)",
            lambda match: (
                f"decoder.up_blocks.{3 - int(match[1])}.upsamplers.0.conv.{match[2]}"
            ),
        ),
        (
            r"decoder\.mid\.block_(\d+)\.(.+)",
            lambda match: f"decoder.mid_block.resnets.{int(match[1]) - 1}.{match[2]}",
        ),
    )
    for pattern, replacement in patterns:
        if match := re.fullmatch(pattern, value):
            return replacement(match)

    for prefix in ("encoder", "decoder"):
        marker = f"{prefix}.mid.attn_1."
        if value.startswith(marker):
            suffix = value.removeprefix(marker)
            attention = {
                "norm.weight": "group_norm.weight",
                "norm.bias": "group_norm.bias",
                "q.weight": "to_q.weight",
                "q.bias": "to_q.bias",
                "k.weight": "to_k.weight",
                "k.bias": "to_k.bias",
                "v.weight": "to_v.weight",
                "v.bias": "to_v.bias",
                "proj_out.weight": "to_out.0.weight",
                "proj_out.bias": "to_out.0.bias",
            }.get(suffix)
            return f"{prefix}.mid_block.attentions.0.{attention}" if attention else None
    return None


def _validate_support_semantics(root: Path, mode: str) -> None:
    def read(relative: str) -> dict[str, Any]:
        value = json.loads((root / relative).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"Klein support JSON must be an object: {relative}")
        return value

    index = read("model_index.json")
    if index.get("_class_name") != "Flux2KleinPipeline":
        raise ValueError("Klein support model_index does not describe Flux2KleinPipeline")
    if (index.get("is_distilled") is True) != (mode == "distilled"):
        raise ValueError("Klein support model_index mode differs from the recipe mode")
    if read("text_encoder/config.json").get("architectures") != ["Qwen3ForCausalLM"]:
        raise ValueError("Klein support text encoder config is not Qwen3ForCausalLM")
    if read("vae/config.json").get("_class_name") != "AutoencoderKLFlux2":
        raise ValueError("Klein support VAE config is not AutoencoderKLFlux2")
    if read("transformer/config.json").get("_class_name") != "Flux2Transformer2DModel":
        raise ValueError("Klein support transformer config is not Flux2Transformer2DModel")


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require_loaded_state(model: Any, allowed_dtypes: set[torch.dtype], label: str) -> None:
    state = model.state_dict()
    if not state or any(value.is_meta for value in state.values()):
        raise RuntimeError(f"{label} has missing/meta state after direct SafeTensors load")
    unexpected = {value.dtype for value in state.values()} - allowed_dtypes
    if unexpected:
        raise RuntimeError(f"{label} loaded unexpected dtypes: {sorted(map(str, unexpected))}")
