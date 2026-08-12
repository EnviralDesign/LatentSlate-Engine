from __future__ import annotations

import json
import tomllib
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from latentslate_engine.artifacts import ArtifactIdentity
from latentslate_engine.runtime import klein_components

# ``probe_artifact`` produced this header-only schema value from the exact BFL
# Base artifact selected by the declaration's immutable revision and full-file
# SHA-256.  Keeping the observed value here makes a stale schema pin fail
# offline, without a model download or a workstation-specific M: dependency.
_OFFICIAL_KLEIN_BASE_FP8_SCHEMA_SHA256 = (
    "ab66231a752c13d876075bd31111ca0e6b28a465d9209b972b84820e587fb5a6"
)


def test_klein_base_transformer_schema_pin_is_observed_and_mode_distinct():
    """Keep the exact Base header schema separate from Distilled's contract."""

    declaration_path = (
        Path(klein_components.__file__).resolve().parents[1]
        / "builtin_resource_declarations"
        / "klein4b-comfy-base-fp8-transformer.toml"
    )
    declaration = tomllib.loads(declaration_path.read_text(encoding="utf-8"))

    assert klein_components.KLEIN_BASE_TRANSFORMER_SCHEMA_SHA256 == (
        _OFFICIAL_KLEIN_BASE_FP8_SCHEMA_SHA256
    )
    assert declaration["resource"]["metadata"]["schema_sha256"] == (
        _OFFICIAL_KLEIN_BASE_FP8_SCHEMA_SHA256
    )
    assert (
        klein_components.KLEIN_BASE_TRANSFORMER_SCHEMA_SHA256
        != klein_components.KLEIN_DISTILLED_TRANSFORMER_SCHEMA_SHA256
    )


def test_accelerate_checkpoint_boundary_normalizes_windows_path(monkeypatch, tmp_path):
    captured = {}

    def fake_load(model, checkpoint, **kwargs):
        captured.update(model=model, checkpoint=checkpoint, **kwargs)

    monkeypatch.setattr("accelerate.load_checkpoint_in_model", fake_load)
    model = object()
    checkpoint = tmp_path / "component.safetensors"

    klein_components._load_accelerate_checkpoint(
        model,
        checkpoint,
        dtype=torch.bfloat16,
        strict=False,
    )

    assert captured == {
        "model": model,
        "checkpoint": str(checkpoint),
        "device_map": {"": "cpu"},
        "dtype": torch.bfloat16,
        "strict": False,
    }


def test_tied_qwen_state_has_no_unresolved_meta_parameters():
    class LoadedQwen:
        def __init__(self):
            self.embedding = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))

        def state_dict(self):
            return {
                "model.embed_tokens.weight": self.embedding,
                "lm_head.weight": self.embedding,
            }

    model = LoadedQwen()

    assert klein_components._unresolved_meta_parameters(model) == []

    model.embedding = torch.nn.Parameter(torch.empty(2, device="meta"))
    assert klein_components._unresolved_meta_parameters(model) == [
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]


@pytest.mark.parametrize(
    "architecture",
    [klein_components._FULL_VAE_ARCHITECTURE, klein_components._SMALL_VAE_ARCHITECTURE],
)
def test_klein_vae_materializes_comfy_bf16_runtime_dtype(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: str,
):
    captured: dict[str, object] = {}

    class FakeVae:
        @staticmethod
        def load_config(path, *, local_files_only):
            captured["config_path"] = path
            captured["local_files_only"] = local_files_only
            return {"architecture": "full"}

        @staticmethod
        def from_config(config):
            captured["config"] = config
            return FakeVae()

        def eval(self):
            captured["evaluated"] = True

    def fake_small_load(model, path, *, dtype):
        captured.update(loader="small", model=model, path=path, dtype=dtype)

    def fake_accelerate_load(model, path, *, dtype, strict):
        captured.update(
            loader="accelerate",
            model=model,
            path=path,
            dtype=dtype,
            strict=strict,
        )

    monkeypatch.setattr("accelerate.init_empty_weights", nullcontext)
    monkeypatch.setattr("diffusers.AutoencoderKLFlux2", FakeVae)
    monkeypatch.setattr(klein_components, "revalidate_klein_dense_component", lambda plan: True)
    monkeypatch.setattr(klein_components, "_load_small_vae_checkpoint", fake_small_load)
    monkeypatch.setattr(klein_components, "_load_accelerate_checkpoint", fake_accelerate_load)
    monkeypatch.setattr(
        klein_components,
        "_require_loaded_state",
        lambda model, allowed, label: captured.update(allowed=allowed, label=label),
    )

    checkpoint = tmp_path / "vae.safetensors"
    plan = klein_components.KleinDenseComponentPlan(
        role="vae",
        architecture=architecture,
        identity=ArtifactIdentity(checkpoint, 1, 1, "header"),
        schema_sha256="schema",
        tensor_count=1,
        tensor_dtypes=("F32",),
    )

    model = klein_components.load_klein_vae(plan, tmp_path)

    assert captured["model"] is model
    assert captured["dtype"] is torch.bfloat16
    assert captured["allowed"] == {torch.bfloat16, torch.int64}
    assert captured["label"] == "Klein VAE"
    assert captured["evaluated"] is True
    assert captured["loader"] == (
        "small"
        if architecture == klein_components._SMALL_VAE_ARCHITECTURE
        else "accelerate"
    )


def _support_tree(tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    contracts = (
        klein_components._DISTILLED_SUPPORT_FILES
        if mode == "distilled"
        else klein_components._BASE_SUPPORT_FILES
    )
    root = tmp_path / mode
    for relative, (size, _digest) in contracts.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "model_index.json":
            payload = {"_class_name": "Flux2KleinPipeline"}
            if mode == "distilled":
                payload["is_distilled"] = True
            raw = json.dumps(payload).encode()
        elif relative == "text_encoder/config.json":
            raw = json.dumps({"architectures": ["Qwen3ForCausalLM"]}).encode()
        elif relative == "vae/config.json":
            raw = json.dumps({"_class_name": "AutoencoderKLFlux2"}).encode()
        elif relative == "transformer/config.json":
            raw = json.dumps({"_class_name": "Flux2Transformer2DModel"}).encode()
        elif relative == "scheduler/scheduler_config.json":
            raw = json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler"}).encode()
        else:
            raw = b"x"
        path.write_bytes(raw + b" " * (size - len(raw)))

    expected_by_path = {
        str((root / relative).resolve()): digest for relative, (_size, digest) in contracts.items()
    }
    monkeypatch.setattr(
        klein_components,
        "_sha256_file",
        lambda path: expected_by_path[str(path.resolve())],
    )
    return root


@pytest.mark.parametrize("mode", ["base", "distilled"])
def test_klein_support_shell_is_mode_specific_and_weight_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
):
    root = _support_tree(tmp_path, mode, monkeypatch)
    plan = klein_components.plan_klein_pipeline_support(root, mode)

    assert plan.mode == mode
    assert plan.root == root.resolve()
    assert len(plan.files) == 13

    other_mode = "base" if mode == "distilled" else "distilled"
    with pytest.raises(ValueError, match="identity mismatch|mode differs"):
        klein_components.plan_klein_pipeline_support(root, other_mode)


def test_klein_support_shell_rejects_any_unbounded_extra_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = _support_tree(tmp_path, "distilled", monkeypatch)
    (root / "text_encoder" / "model.safetensors").write_bytes(b"must not be here")

    with pytest.raises(ValueError, match="exact bounded shell"):
        klein_components.plan_klein_pipeline_support(root, "distilled")


def test_small_decoder_plan_is_exact_and_separate_from_full_vae(monkeypatch, tmp_path):
    captured = {}

    def fake_plan(path, **kwargs):
        captured.update(path=path, **kwargs)
        return "small-plan"

    monkeypatch.setattr(klein_components, "_plan_dense_component", fake_plan)
    path = tmp_path / "full_encoder_small_decoder.safetensors"

    assert klein_components.plan_klein_small_vae(path) == "small-plan"
    assert captured == {
        "path": path,
        "role": "vae",
        "architecture": "flux2_small_decoder_full_encoder",
        "size_bytes": 249_519_092,
        "schema_sha256": klein_components.KLEIN_SMALL_VAE_SCHEMA_SHA256,
        "tensor_count": 251,
        "tensor_dtypes": ("F32", "I64"),
        "contract": "native/fp32",
    }


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("encoder.quant_conv.weight", "quant_conv.weight"),
        (
            "encoder.down.1.block.0.nin_shortcut.weight",
            "encoder.down_blocks.1.resnets.0.conv_shortcut.weight",
        ),
        (
            "decoder.up.3.upsample.conv.bias",
            "decoder.up_blocks.0.upsamplers.0.conv.bias",
        ),
        (
            "decoder.mid.attn_1.proj_out.weight",
            "decoder.mid_block.attentions.0.to_out.0.weight",
        ),
        ("bn.num_batches_tracked", "bn.num_batches_tracked"),
    ],
)
def test_small_decoder_key_mapping_is_bounded(source: str, expected: str):
    assert klein_components._map_small_vae_key(source) == expected


def test_small_decoder_key_mapping_rejects_unknown_keys():
    assert klein_components._map_small_vae_key("unexpected.weight") is None
