from __future__ import annotations

import hashlib
import json
from pathlib import Path

from latentslate_engine.config import Settings
from latentslate_engine.recipes import build_deployment_plan
from latentslate_engine.tools import default_registry

LTX_RESOURCE = "model:ltx23:diffusers--ltx-2.3-distilled-diffusers"
LTX_RECIPES = [
    "ltx-2-3.text-to-video.native-distilled-bf16",
    "ltx-2-3.image-to-video.native-distilled-bf16",
    "ltx-2-3.first-last-frame-to-video.native-distilled-bf16",
]
COMPONENT_FILES = {
    "audio_vae": ("config.json", "diffusion_pytorch_model.safetensors"),
    "connectors": (
        "config.json",
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
        "diffusion_pytorch_model.safetensors.index.json",
    ),
    "scheduler": ("scheduler_config.json",),
    "vae": ("config.json", "diffusion_pytorch_model.safetensors"),
    "vocoder": ("config.json", "diffusion_pytorch_model.safetensors"),
}
PROCESSOR_FILES = (
    "added_tokens.json", "chat_template.jinja", "preprocessor_config.json",
    "processor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
    "tokenizer_config.json",
)
TOKENIZER_FILES = (
    "added_tokens.json", "chat_template.jinja", "special_tokens_map.json", "tokenizer.json",
    "tokenizer.model", "tokenizer_config.json",
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )


def test_ltx23_exact_native_closure_is_a_50_file_immutable_snapshot(tmp_path: Path) -> None:
    registry = default_registry(_settings(tmp_path), emit_warnings=False)
    resource = registry.resources.by_id()[LTX_RESOURCE]
    snapshot = resource.metadata["upstream_snapshot"]
    files = snapshot["files"]

    assert resource.size_bytes == 94_977_693_482
    assert snapshot["aggregate_size_bytes"] == resource.size_bytes
    assert snapshot["revision"] == "432e0d3c2d1769aaa4d295f9243f7062bf6b47ee"
    assert snapshot["manifest_sha256"] == (
        "77a305d7a378520780949d3c6d7da3a4facf6b327242de4bf1f6e6f9145cfc4d"
    )
    assert len(files) == 50
    assert sum(item["size_bytes"] for item in files) == resource.size_bytes
    tuples = [(item["path"], item["size_bytes"], item["git_oid"]) for item in files]
    manifest = {"revision": snapshot["revision"], "files": tuples}
    assert hashlib.sha256(
        json.dumps(manifest, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest() == snapshot["manifest_sha256"]
    assert {item["path"] for item in files} == {
        "model_index.json",
        *{f"{component}/{name}" for component, names in COMPONENT_FILES.items() for name in names},
        *{f"processor/{name}" for name in PROCESSOR_FILES},
        *{f"tokenizer/{name}" for name in TOKENIZER_FILES},
        *{f"text_encoder/{name}" for name in {"config.json", "generation_config.json", "model.safetensors.index.json", *{f"model-{index:05d}-of-00011.safetensors" for index in range(1, 12)}}},
        "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors.index.json",
        *{f"transformer/diffusion_pytorch_model-{index:05d}-of-00008.safetensors" for index in range(1, 9)},
    }
    source = resource.sources[0]
    assert source.is_exact()
    assert set(source.allow_patterns) == {item["path"] for item in files}
    assert "README.md" not in source.allow_patterns and ".gitattributes" not in source.allow_patterns


def test_ltx23_three_operation_catalog_keeps_first_and_first_last_distinct(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    registry = default_registry(settings, emit_warnings=False)
    recipes = {entry.key: entry for entry in registry.variants}

    assert all(key in recipes for key in LTX_RECIPES)
    assert recipes[LTX_RECIPES[0]].recipe_type is None
    for key in LTX_RECIPES:
        assert recipes[key].model_resource == LTX_RESOURCE
        assert "reference" in recipes[key].tags
    descriptors = {descriptor.key: descriptor for descriptor in registry.descriptors()}
    first = descriptors[LTX_RECIPES[1]]
    first_last = descriptors[LTX_RECIPES[2]]
    first_inputs = {item.key: item for item in first.inputs}
    first_last_inputs = {item.key: item for item in first_last.inputs}
    assert "end_image" not in first_inputs
    assert first_last_inputs["end_image"].required is True
    assert first.schema_hash != first_last.schema_hash

    plan = build_deployment_plan(settings, registry, "ltx23-video")
    assert [recipe.key for recipe in plan.recipes] == LTX_RECIPES
    assert [resource.id for resource in plan.resources] == [LTX_RESOURCE]
    assert plan.total_bytes == 94_977_693_482
    assert plan.remote_provisionable
