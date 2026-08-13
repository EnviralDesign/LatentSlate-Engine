from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from latentslate_engine import __main__ as engine_cli
from latentslate_engine.app import create_app
from latentslate_engine.cli_product import recipe_detail_payload
from latentslate_engine.config import Settings
from latentslate_engine.recipes import build_deployment_plan
from latentslate_engine.tools import default_registry


def _settings(home: Path) -> Settings:
    value = Settings(
        home=home,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    value.ensure_directories()
    return value


def test_builtin_recipes_are_exact_lean_and_unavailable_when_artifacts_are_absent(
    tmp_path: Path,
):
    value = _settings(tmp_path)
    registry = default_registry(value, emit_warnings=False)
    recipes = {entry.key: entry for entry in registry.variants}

    assert set(recipes) == {
        "flux2-klein-4b.image-to-image.comfy-base-fp8",
        "flux2-klein-4b.image-to-image.comfy-distilled-fp8",
        "flux2-klein-4b.image-to-image.bfl-distilled-nvfp4",
        "flux2-klein-4b.text-to-image.comfy-distilled-fp8",
        "flux2-klein-4b.text-to-image.bfl-distilled-nvfp4",
        "flux2-klein-4b.image-to-image.native-distilled-bf16",
        "flux2-klein-4b.text-to-image.native-distilled-bf16",
        "flux2-klein-9b.image-to-image.bfl-distilled-fp8",
        "flux2-klein-9b.image-to-image.bfl-distilled-nvfp4",
        "flux2-klein-9b.image-to-image.native-distilled-bf16",
        "flux2-klein-9b.text-to-image.bfl-distilled-fp8",
        "flux2-klein-9b.text-to-image.bfl-distilled-nvfp4",
        "flux2-klein-9b.text-to-image.native-distilled-bf16",
        "ltx-2-3.image-to-video.native-distilled-bf16",
        "ltx-2-3.text-to-video.native-distilled-bf16",
        "wan-2-2-14b-i2v.image-to-video.comfy-org-fp8",
        "wan-2-2-5b-ti2v.image-to-video.comfy-fp16",
        "wan-2-2-5b-ti2v.text-to-video.comfy-fp16",
        "wan-2-2-5b-ti2v.text-to-video.native-bf16",
    }
    assert all(not recipe.available for recipe in recipes.values())
    recipe_root = Path(__file__).parents[1] / "src/latentslate_engine/builtin_recipes/klein4b"
    distilled_i2i_source = (
        recipe_root / "flux2-klein-4b-image-to-image-comfy-distilled-fp8.toml"
    ).read_text(encoding="utf-8")
    base_i2i_source = (recipe_root / "flux2-klein-4b-image-to-image-comfy-base-fp8.toml").read_text(
        encoding="utf-8"
    )
    nvfp4_i2i_source = (
        recipe_root / "flux2-klein-4b-image-to-image-bfl-distilled-nvfp4.toml"
    ).read_text(encoding="utf-8")
    assert '"fallback"' in distilled_i2i_source
    assert '"recommended"' not in distilled_i2i_source
    assert '"recommended"' in nvfp4_i2i_source
    assert '"experimental"' not in nvfp4_i2i_source
    assert '"recommended"' not in base_i2i_source
    assert '"quality-alternate"' in base_i2i_source
    wan14_source = (
        Path(__file__).parents[1] / "src/latentslate_engine/builtin_recipes/wan22/"
        "wan-2-2-14b-i2v-image-to-video-comfy-org-fp8.toml"
    ).read_text(encoding="utf-8")
    assert '"experimental"' in wan14_source
    for key, recipe in recipes.items():
        reason = recipe.unavailable_reason or ""
        if (
            key.startswith("wan-2-2-5b-ti2v.")
            and key.endswith("comfy-fp16")
            or key == "wan-2-2-14b-i2v.image-to-video.comfy-org-fp8"
            or (
                key.startswith(("flux2-klein-4b.", "flux2-klein-9b."))
                and key.endswith(("comfy-base-fp8", "comfy-distilled-fp8", "bfl-distilled-nvfp4"))
            )
            or (key.startswith("flux2-klein-9b.") and key.endswith("bfl-distilled-fp8"))
        ):
            assert "inventory path is unavailable" in reason
        else:
            assert "artifact is not installed or incomplete" in reason

    legacy_recipe_key_parts = (
        ("klein4b", "comfy-fp8", "image-to-image"),
        ("klein4b", "comfy-fp8", "text-to-image"),
        ("klein4b", "reference-bf16", "image-to-image"),
        ("klein4b", "reference-bf16", "text-to-image"),
        ("ltx23", "distilled", "image-to-video"),
        ("ltx23", "distilled", "text-to-video"),
        ("wan22", "comfy-org-14b-i2v-fp8"),
        ("wan22", "ti2v5b", "text-to-video"),
    )
    legacy_recipe_keys = {".".join(parts) for parts in legacy_recipe_key_parts}
    assert legacy_recipe_keys.isdisjoint(recipes)
    for legacy_recipe_key in legacy_recipe_keys:
        with pytest.raises(KeyError, match="Unknown recipe"):
            recipe_detail_payload(value, registry, legacy_recipe_key)

    resources = {resource.id: resource for resource in registry.resources.resources}
    klein = resources["model:klein4b:black-forest-labs--flux.2-klein-4b"]
    klein_base = resources["model:klein4b:transformers/flux-2-klein-base-4b-fp8"]
    klein_distilled = resources["model:klein4b:transformers/flux-2-klein-4b-fp8"]
    klein_nvfp4 = resources["model:klein4b:transformers/flux-2-klein-4b-nvfp4"]
    klein_qwen = resources["model:klein4b:text_encoders/qwen_3_4b"]
    klein_vae = resources["model:klein4b:vae/flux2-vae"]
    klein_small_vae = resources["model:klein4b:vae/full_encoder_small_decoder"]
    klein_base_support = resources["model:klein4b:support/comfy-base-pipeline-support"]
    klein_distilled_support = resources["model:klein4b:support/comfy-distilled-pipeline-support"]
    klein9 = resources["model:klein9b:black-forest-labs--flux.2-klein-9b"]
    klein9_fp8 = resources["model:klein9b:transformers/flux-2-klein-9b-fp8"]
    klein9_nvfp4 = resources["model:klein9b:transformers/flux-2-klein-9b-nvfp4"]
    klein9_qwen = resources["model:klein9b:text_encoders/qwen_3_8b_fp8mixed"]
    klein9_support = resources["model:klein9b:support/bfl-distilled-pipeline-support"]
    ltx = resources["model:ltx23:diffusers--ltx-2.3-distilled-diffusers"]
    wan = resources["model:wan22:wan-ai--wan2.2-ti2v-5b-diffusers"]
    wan5_transformer = resources[
        "model:wan22:comfy-org-wan22-ti2v-5b/split_files/diffusion_models/wan2.2_ti2v_5b_fp16"
    ]
    wan5_text = resources[
        "model:wan22:comfy-org-wan22-ti2v-5b/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled"
    ]
    wan5_vae = resources["model:wan22:comfy-org-wan22-ti2v-5b/split_files/vae/wan2.2_vae"]
    wan5_crush_lora = resources["lora:wan22:ostris/wan22_5b_i2v_crush_it_lora"]
    wan5_hstoric_lora = resources["lora:wan22:alekseycalvin/hstoric_color_wan22_5b_lora"]
    wan14_support = resources["model:wan22:wan22-14b-i2v-official-support"]
    wan14_resources = [
        resources[
            "model:wan22:comfy-org-wan22-14b-i2v-fp8/split_files/diffusion_models/wan2.2_i2v_high_noise_14b_fp8_scaled"
        ],
        resources[
            "model:wan22:comfy-org-wan22-14b-i2v-fp8/split_files/diffusion_models/wan2.2_i2v_low_noise_14b_fp8_scaled"
        ],
        resources[
            "model:wan22:wan22-14b-i2v-comfy-support/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled"
        ],
        resources["model:wan22:wan22-14b-i2v-comfy-support/split_files/vae/wan_2.1_vae"],
        wan14_support,
    ]
    assert (klein.size_bytes, ltx.size_bytes, wan.size_bytes) == (
        23740007447,
        94977700554,
        34203021834,
    )
    assert klein_nvfp4.size_bytes == 2460413488
    assert klein_nvfp4.precision.value == "fp4"
    assert klein_nvfp4.quantization.value == "nvfp4"
    assert klein_nvfp4.metadata["schema_sha256"] == (
        "c6683e31192ed861a3068673e41d89555caacdad2e4a3a7357e5e576dcaea9d6"
    )
    assert (klein9_nvfp4.size_bytes, klein9_fp8.size_bytes, klein9_qwen.size_bytes) == (
        5_760_960_048,
        9_433_061_528,
        8_664_848_742,
    )
    assert klein9_nvfp4.metadata["schema_sha256"] == (
        "a222d48e4d796bfdb027b0c8e0eb3c8dc655d0901dbe9a7fdab41b434fb036f8"
    )
    assert klein9_fp8.metadata["schema_sha256"] == (
        "c25cec508eb68835ccd5833bb3a9886a1dea9cfb652ecf98b1ecf4d6d332940d"
    )
    assert klein9_qwen.metadata["schema_sha256"] == (
        "42333ea5d161147268b724ca269782a6be0b4db0e41c19216a4f739b869e0ff6"
    )
    assert klein9_nvfp4.sources[0].revision == ("e882f64f6aa086fcf8915a7763550e05af10ef13")
    assert klein9_nvfp4.sources[0].sha256 == (
        "5c72214496dd278f721a112e1bd1585fffed487bc0831c894bcbf30d12e9ee48"
    )
    assert klein9_fp8.sources[0].revision == "902d9d510b51533e07729f19211414a3648b77d2"
    assert klein9_qwen.sources[0].revision == "23fbc8aa8b621f29f2249cd1bd9c47e5d0eebd83"
    for operation in ("text-to-image", "image-to-image"):
        nvfp4_recipe = recipes[f"flux2-klein-4b.{operation}.bfl-distilled-nvfp4"]
        fp8_recipe = recipes[f"flux2-klein-4b.{operation}.comfy-distilled-fp8"]
        assert nvfp4_recipe.recipe_resources["transformer"] == klein_nvfp4.id
        assert {
            role: resource
            for role, resource in nvfp4_recipe.recipe_resources.items()
            if role != "transformer"
        } == {
            role: resource
            for role, resource in fp8_recipe.recipe_resources.items()
            if role != "transformer"
        }
    assert klein.sources[0].revision == "e7b7dc27f91deacad38e78976d1f2b499d76a294"
    assert ltx.sources[0].revision == "432e0d3c2d1769aaa4d295f9243f7062bf6b47ee"
    assert wan.sources[0].revision == "b8fff7315c768468a5333511427288870b2e9635"
    assert all(resource.sources[0].is_exact() for resource in (klein, ltx, wan))
    assert (wan5_transformer.size_bytes, wan5_text.size_bytes, wan5_vae.size_bytes) == (
        9_999_658_848,
        6_735_906_897,
        1_409_400_960,
    )
    assert all(
        resource.sources[0].is_exact() for resource in (wan5_transformer, wan5_text, wan5_vae)
    )
    assert (
        wan5_crush_lora.size_bytes,
        wan5_crush_lora.sources[0].revision,
        wan5_crush_lora.sources[0].sha256,
    ) == (
        161_293_208,
        "e4b85be20d75c2ca2ee1b901ba2cf49d9416e233",
        "00a3ed72d8e257b416e1232cce07acf76cfb3ad7538f8ba995b6818f0b560f23",
    )
    assert (
        wan5_hstoric_lora.size_bytes,
        wan5_hstoric_lora.sources[0].revision,
        wan5_hstoric_lora.sources[0].sha256,
    ) == (
        322_511_512,
        "fb47fbdfb7fa391ed6d29f1d1b06f78bc815d7c0",
        "5c2fc21b1e74d5088318fea72c676181650a0f771cc521151edfc43f6ea9ec77",
    )
    for operation in ("text-to-video", "image-to-video"):
        recipe = recipes[f"wan-2-2-5b-ti2v.{operation}.comfy-fp16"]
        assert recipe.recipe_resources == {
            "transformer": wan5_transformer.id,
            "text_encoder": wan5_text.id,
            "vae": wan5_vae.id,
        }

    for operation in ("text-to-image", "image-to-image"):
        nvfp4_recipe = recipes[f"flux2-klein-9b.{operation}.bfl-distilled-nvfp4"]
        fp8_recipe = recipes[f"flux2-klein-9b.{operation}.bfl-distilled-fp8"]
        assert nvfp4_recipe.recipe_type == fp8_recipe.recipe_type == "klein9_comfy"
        assert nvfp4_recipe.recipe_resources["transformer"] == klein9_nvfp4.id
        assert fp8_recipe.recipe_resources["transformer"] == klein9_fp8.id
        assert nvfp4_recipe.recipe_resources["text_encoder"] == klein9_qwen.id
        assert nvfp4_recipe.recipe_resources["pipeline_support"] == klein9_support.id
        assert nvfp4_recipe.recipe_resources["vae"] == klein_small_vae.id

    klein9_plan = build_deployment_plan(value, registry, "klein9b-image")
    assert [recipe.key for recipe in klein9_plan.recipes] == [
        "flux2-klein-9b.text-to-image.bfl-distilled-nvfp4",
        "flux2-klein-9b.image-to-image.bfl-distilled-nvfp4",
        "flux2-klein-9b.text-to-image.bfl-distilled-fp8",
        "flux2-klein-9b.image-to-image.bfl-distilled-fp8",
    ]
    assert [resource.id for resource in klein9_plan.resources] == sorted(
        resource.id
        for resource in (
            klein9_fp8,
            klein9_nvfp4,
            klein9_qwen,
            klein9_support,
            klein_small_vae,
        )
    )
    assert klein9_plan.total_bytes == 24_124_275_689
    assert klein9_plan.incremental_bytes == klein9_plan.total_bytes
    assert klein9_plan.remote_provisionable
    assert all(resource.sources[0].is_exact() for resource in klein9_plan.resources)

    klein9_reference = build_deployment_plan(value, registry, "klein9b-reference-bf16-image")
    assert [resource.id for resource in klein9_reference.resources] == [klein9.id]
    assert klein9_reference.total_bytes == 34_722_772_650

    wan14_plan = build_deployment_plan(value, registry, "wan22-14b-i2v-fp8")
    assert [resource.id for resource in wan14_plan.resources] == sorted(
        resource.id for resource in wan14_resources
    )
    assert wan14_plan.total_bytes == 36108276923
    assert wan14_plan.incremental_bytes == 36108276923
    assert not wan14_plan.locally_runnable
    # The 14B I2V support tree is an exact immutable filtered upstream
    # snapshot, rather than a locally assembled manual directory.  Its source
    # list is deliberately a path-by-path whitelist: this proves neither a 14B
    # transformer shard nor an UMT5 checkpoint can be acquired with it.
    assert all(resource.sources[0].is_exact() for resource in wan14_resources)
    support_source = wan14_support.sources[0]
    assert support_source.repo_id == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    assert support_source.revision == "596658fd9ca6b7b71d5057529bbf319ecbc61d74"
    expected_support_paths = {
        "README.md",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "tokenizer/special_tokens_map.json",
        "tokenizer/spiece.model",
        "tokenizer/tokenizer_config.json",
        "tokenizer/tokenizer.json",
        "transformer/config.json",
        "transformer_2/config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    }
    assert set(support_source.allow_patterns) == expected_support_paths
    assert not support_source.ignore_patterns
    assert all(
        not path.startswith(("transformer/", "transformer_2/", "text_encoder/model"))
        or path.endswith("/config.json")
        for path in support_source.allow_patterns
    )
    snapshot = wan14_support.metadata["upstream_snapshot"]
    assert snapshot["aggregate_size_bytes"] == wan14_support.size_bytes
    assert {entry["path"] for entry in snapshot["files"]} == expected_support_paths
    assert sum(entry["size_bytes"] for entry in snapshot["files"]) == wan14_support.size_bytes
    assert (
        next(
            entry
            for entry in snapshot["files"]
            if entry["path"] == "vae/diffusion_pytorch_model.safetensors"
        )["sha256"]
        == "d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793"
    )
    assert wan14_plan.remote_provisionable
    assert "no immutable remote source" not in " ".join(wan14_plan.warnings)

    plan = build_deployment_plan(value, registry, "klein4b-image")
    assert [recipe.key for recipe in plan.recipes] == [
        "flux2-klein-4b.text-to-image.bfl-distilled-nvfp4",
        "flux2-klein-4b.image-to-image.bfl-distilled-nvfp4",
        "flux2-klein-4b.text-to-image.comfy-distilled-fp8",
        "flux2-klein-4b.image-to-image.comfy-distilled-fp8",
        "flux2-klein-4b.image-to-image.comfy-base-fp8",
    ]
    assert [resource.id for resource in plan.resources] == sorted(
        resource.id
        for resource in (
            klein_base,
            klein_distilled,
            klein_nvfp4,
            klein_qwen,
            klein_vae,
            klein_small_vae,
            klein_base_support,
            klein_distilled_support,
        )
    )
    assert plan.total_bytes == 19_283_023_710
    assert plan.incremental_bytes == plan.total_bytes
    assert len(plan.resources) == len({resource.id for resource in plan.resources}) == 8
    assert not plan.locally_runnable
    assert plan.remote_provisionable
    assert all(resource.sources[0].is_exact() for resource in plan.resources)

    reference = build_deployment_plan(value, registry, "klein4b-reference-bf16-image")
    assert [resource.id for resource in reference.resources] == [klein.id]
    assert reference.total_bytes == klein.size_bytes

    ltx_plan = build_deployment_plan(value, registry, "ltx23-video")
    assert [recipe.key for recipe in ltx_plan.recipes] == [
        "ltx-2-3.text-to-video.native-distilled-bf16",
        "ltx-2-3.image-to-video.native-distilled-bf16",
    ]
    assert [resource.id for resource in ltx_plan.resources] == [ltx.id]
    assert ltx_plan.total_bytes == ltx.size_bytes
    assert ltx_plan.incremental_bytes == ltx.size_bytes


def test_builtin_catalog_is_exposed_through_api_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    value = _settings(tmp_path)
    registry = default_registry(value, emit_warnings=False)
    app = create_app(value, registry)

    with TestClient(app) as client:
        recipes = client.get("/v1/recipes")
        profiles = client.get("/v1/deployment/profiles")
        plan = client.get("/v1/deployment/plan/wan22-ti2v5b-text-to-video")

    assert recipes.status_code == 200
    assert len(recipes.json()["recipes"]) == 19
    assert profiles.status_code == 200
    assert [profile["key"] for profile in profiles.json()["profiles"]] == [
        "klein4b-image",
        "klein4b-reference-bf16-image",
        "klein9b-reference-bf16-image",
        "klein9b-image",
        "ltx23-video",
        "wan22-14b-i2v-fp8",
        "wan22-ti2v5b-comfy-video",
        "wan22-ti2v5b-text-to-video",
    ]
    assert plan.status_code == 200
    assert plan.json()["total_bytes"] == 34203021834

    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: value))
    monkeypatch.setattr(
        "latentslate_engine.tools.default_registry",
        lambda *_args, **_kwargs: registry,
    )
    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "deployments", "profiles"])
    engine_cli.main()
    assert "Deployment profiles · 8 saved recipe selections" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["latentslate-engine", "deployments", "profiles", "--json"],
    )
    engine_cli.main()
    assert '"key": "klein4b-image"' in capsys.readouterr().out


def test_builtin_wan14_profile_is_runnable_from_the_opt_in_workstation_home():
    """Prove package and retained local recipe keys coexist against actual artifacts."""

    configured_home = os.environ.get("LATENTSLATE_TEST_WAN14_HOME")
    if not configured_home:
        pytest.skip("set LATENTSLATE_TEST_WAN14_HOME for the workstation artifact proof")
    workstation_home = Path(configured_home)
    if not workstation_home.is_dir():
        pytest.skip("configured workstation Wan 14B artifact home is unavailable")

    # Do not create or alter anything in the opt-in artifact home.
    value = Settings(
        home=workstation_home,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    registry = default_registry(value, emit_warnings=False)
    recipes = {entry.key: entry for entry in registry.variants}

    assert registry.resources.errors == []
    assert registry.variant_errors == []

    package_recipe = recipes.get("wan-2-2-14b-i2v.image-to-video.comfy-org-fp8")
    retained_local_recipe = recipes.get("wan22.comfy_org_14b_i2v_fp8")
    if (
        package_recipe is None
        or retained_local_recipe is None
        or not package_recipe.available
        or not retained_local_recipe.available
    ):
        pytest.skip("opt-in workstation Wan 14B artifact closure is not installed")

    assert package_recipe.available
    assert retained_local_recipe.available
