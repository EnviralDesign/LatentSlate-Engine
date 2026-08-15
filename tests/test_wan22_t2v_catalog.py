from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from latentslate_engine import resources as resources_module
from latentslate_engine import variants as variants_module
from latentslate_engine.config import Settings
from latentslate_engine.resources import discover_resources
from latentslate_engine.tools import default_registry


def _settings(home: Path) -> Settings:
    settings = Settings(
        home=home,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    return settings


def test_t2v_support_closure_is_exact_and_excludes_checkpoint_weights(tmp_path: Path) -> None:
    registry = default_registry(_settings(tmp_path), emit_warnings=False)
    resources = {resource.id: resource for resource in registry.resources.resources}
    support = resources["model:wan22:wan22-14b-t2v-official-support"]
    source = support.sources[0]
    files = support.metadata["upstream_snapshot"]["files"]
    assert support.size_bytes == 529_069_135
    assert source.revision == "5be7df9619b54f4e2667b2755bc6a756675b5cd7"
    assert source.is_exact()
    assert len(files) == len(source.allow_patterns) == 12
    assert sum(item["size_bytes"] for item in files) == support.size_bytes
    assert [item["path"] for item in files] == list(source.allow_patterns)
    assert not any(
        path.startswith("transformer/") and not path.endswith("config.json")
        or path.startswith("transformer_2/") and not path.endswith("config.json")
        or path.startswith("text_encoder/") and not path.endswith("config.json")
        for path in source.allow_patterns
    )

    high = resources[
        "model:wan22:comfy-org-wan22-14b-t2v-fp8/"
        "wan2.2_t2v_high_noise_14b_fp8_scaled"
    ]
    low = resources[
        "model:wan22:comfy-org-wan22-14b-t2v-fp8/"
        "wan2.2_t2v_low_noise_14b_fp8_scaled"
    ]
    assert {high.sources[0].sha256, low.sources[0].sha256} == {
        "cad711ae211c8b23455ec68cd6a190a33a3d874234a77eb57266d73f8f0e6c9f",
        "e71b96d7c82e638694c5e7fb98fac4bfb0e4ddc5fbbb4b1df40da8f0f1278a97",
    }
    assert high.size_bytes == low.size_bytes == 14_293_923_632
    entry = next(
        item
        for item in registry.variants
        if item.key == "wan-2-2-14b-t2v.text-to-video.comfy-org-fp8"
    )
    assert entry.base_tool == "wan22.native_text_to_video"
    assert entry.recipe_type == "wan22_t2v_14b"
    assert entry.tags == [
        "builtin",
        "fallback",
        "hardware-proven",
        "wan2.2",
        "t2v",
        "14b",
        "comfy-org",
        "fp8",
        "native-stored-weights",
    ]

    lightx_high = resources[
        "lora:wan22:comfy-org/wan2.2_t2v_lightx2v_4steps_lora_v1_1_high_noise"
    ]
    lightx_low = resources[
        "lora:wan22:comfy-org/wan2.2_t2v_lightx2v_4steps_lora_v1_1_low_noise"
    ]
    assert (
        lightx_high.size_bytes,
        lightx_low.size_bytes,
        lightx_high.sources[0].revision,
        lightx_low.sources[0].revision,
        lightx_high.sources[0].sha256,
        lightx_low.sources[0].sha256,
    ) == (
        1_226_977_424,
        1_226_977_424,
        "fb1388adc906ab39ffc26ee40e96b22886b56bc4",
        "fb1388adc906ab39ffc26ee40e96b22886b56bc4",
        "698321cb86bd30c4af06c9b84e656a1048c8cb54e06d50694536fb5de37fde41",
        "ec95216e614b3c132c11bfb387b11feedf62163150ccc9068bca8a189771e75a",
    )
    expected_adapter_metadata = {
        "architecture": "wan22_t2v_14b_lightx2v_lora",
        "rank": 64,
        "header_sha256": "d65be4ded1d618bd2c8086f909717f99b95d638a74e33429173ee905e56b0636",
        "tensor_count": 1200,
        "alpha_tensor_count": 400,
        "alpha_dtype": "I64",
    }
    assert lightx_high.metadata == {**expected_adapter_metadata, "noise_stage": "high"}
    assert lightx_low.metadata == {**expected_adapter_metadata, "noise_stage": "low"}
    lightx = next(
        item
        for item in registry.variants
        if item.key == "wan-2-2-14b-t2v.text-to-video.comfy-org-fp8-lightx2v-4step"
    )
    assert lightx.base_tool == "wan22.native_text_to_video"
    assert lightx.recipe_type == "wan22_t2v_14b"
    assert lightx.tags == [
        "builtin",
        "quality-alternate",
        "hardware-proven",
        "wan2.2",
        "t2v",
        "14b",
        "comfy-org",
        "fp8",
        "lightx2v",
        "4step",
    ]
    assert lightx.recipe_resources == entry.recipe_resources
    assert lightx.fixed_resources[-2:] == [lightx_high.id, lightx_low.id]


def test_t2v_declarations_enrich_their_installed_direct_artifact_paths(
    tmp_path: Path, monkeypatch
) -> None:
    """The locally stored artifact IDs must survive a post-install rediscovery."""

    settings = _settings(tmp_path)
    root = settings.model_root / "wan22" / "comfy-org-wan22-14b-t2v-fp8"
    for filename in (
        "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
    ):
        (root / filename).parent.mkdir(parents=True, exist_ok=True)
        (root / filename).touch()

    def available(resource, _path, **_kwargs):
        return resource.model_copy(update={"available": True, "unavailable_reason": None})

    monkeypatch.setattr(resources_module, "_with_artifact_availability", available)
    monkeypatch.setattr(
        variants_module,
        "validate_native_wan22_i2v_14b_recipe",
        lambda *_args, **_kwargs: SimpleNamespace(errors=()),
    )
    inventory = discover_resources(settings)
    assert not [error for error in inventory.errors if "14b-t2v" in error]
    resources = {resource.id: resource for resource in inventory.resources}
    high_id = (
        "model:wan22:comfy-org-wan22-14b-t2v-fp8/"
        "wan2.2_t2v_high_noise_14b_fp8_scaled"
    )
    low_id = (
        "model:wan22:comfy-org-wan22-14b-t2v-fp8/"
        "wan2.2_t2v_low_noise_14b_fp8_scaled"
    )
    assert resources[high_id].sources
    assert resources[low_id].sources
    assert inventory.paths[high_id] == root / "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    assert inventory.paths[low_id] == root / "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"

    registry = default_registry(settings, emit_warnings=False)
    entry = next(
        item
        for item in registry.variants
        if item.key == "wan-2-2-14b-t2v.text-to-video.comfy-org-fp8"
    )
    assert entry.available
    assert entry.recipe_type == "wan22_t2v_14b"
    assert entry.recipe_resources == {
        "pipeline_support": "model:wan22:wan22-14b-t2v-official-support",
        "transformer_high_noise": high_id,
        "transformer_low_noise": low_id,
        "text_encoder": (
            "model:wan22:wan22-14b-i2v-comfy-support/"
            "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled"
        ),
        "vae": (
            "model:wan22:wan22-14b-i2v-comfy-support/"
            "split_files/vae/wan_2.1_vae"
        ),
    }
    assert len(entry.fixed_resources) == 5
    assert resources[high_id].base_model == resources[low_id].base_model == "comfy-org-wan22-14b-t2v-fp8"
