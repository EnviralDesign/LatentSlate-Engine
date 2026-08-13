"""Offline qualification checks for the cataloged Winnougan ConvRot artifacts.

The candidate files are intentionally not downloaded by this suite: the three
payloads are tens of GiB and no clean operation contract exists yet.  These
assertions pin the independently inspected SafeTensors header facts alongside
the immutable Hugging Face content identities. The actual multi-GiB headers are
not fetched by this bounded suite, so generic ConvRot support is deliberately
kept distinct from exact-artifact adapter planning or native dispatch.
"""

from __future__ import annotations

from pathlib import Path

from latentslate_engine.config import Settings
from latentslate_engine.runtime.umt5_stored_adapter import _SUPPORTED as UMT5_SUPPORTED
from latentslate_engine.runtime.wan22_stored_adapter import (
    _SUPPORTED_ARTIFACT_CONTRACTS as WAN_SUPPORTED,
)
from latentslate_engine.tools import default_registry

_CONTRACT = "comfy_quant/int8_tensorwise_convrot"
_REPO = "Winnougan/Wan2.2-INT8-Convrot"


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


def test_wan14_convrot_resources_pin_audited_headers_and_are_not_recipe_reachable(
    tmp_path: Path,
) -> None:
    registry = default_registry(_settings(tmp_path), emit_warnings=False)
    resources = {item.id: item for item in registry.resources.resources}
    expected = {
        "model:wan22:winnougan-wan22-int8-convrot/wan2.2_i2v_high_noise_14b_int8_convrot": {
            "component": "transformer_high_noise",
            "filename": "wan2.2_i2v_high_noise_14B_int8_convrot.safetensors",
            "revision": "1f3ac81b2913e055f45bcbde0ac2e9848e73bf9e",
            "sha256": "a8e4d385b882b50ba79315245f16a8c014a03341fa2aea3493a467deeb440caf",
            "size": 14_536_737_888,
            "architecture": "wan2.2_i2v_14b",
            "group_size": 256,
            "header": "c61559c5c42dba63370e46aba11596cc72325c00d0e98d28c4868dd78108f757",
            "schema": "565c943bbf764460b3ea61f6dd48bd21689485f82286f33139749a182a6a080e",
            "tensor_count": 1895,
            "quantized_roles": 400,
            "quant_auxiliary": 800,
            "dense_tensors": 695,
        },
        "model:wan22:winnougan-wan22-int8-convrot/wan2.2_i2v_low_noise_14b_int8_convrot": {
            "component": "transformer_low_noise",
            "filename": "wan2.2_i2v_low_noise_14B_int8_convrot.safetensors",
            "revision": "99e38f9d97ef5493b73ccbfbb4af6af0f4964b32",
            "sha256": "5ff0d5cba86ebef79b4d82544779c89a15a9b04492a2e8dad7563d7ca8506492",
            "size": 14_536_737_888,
            "architecture": "wan2.2_i2v_14b",
            "group_size": 256,
            "header": "c61559c5c42dba63370e46aba11596cc72325c00d0e98d28c4868dd78108f757",
            "schema": "565c943bbf764460b3ea61f6dd48bd21689485f82286f33139749a182a6a080e",
            "tensor_count": 1895,
            "quantized_roles": 400,
            "quant_auxiliary": 800,
            "dense_tensors": 695,
        },
        "model:wan22:winnougan-wan22-int8-convrot/umt5_xxl_int8_convrot": {
            "component": "text_encoder",
            "filename": "umt5_xxl_int8_convrot.safetensors",
            "revision": "d8b25158d0359c1ca45425eb80fe719ea76e59e3",
            "sha256": "d82e9e154fc1eb9f16dbc432a3deb26a88d22e3cadb19b6b95efebb6eb08f269",
            "size": 6_743_849_737,
            "architecture": "umt5_xxl",
            "group_size": 64,
            "header": "0b37b71bdcb6a0378740e3c8d83bf03ce7647c4304582826e2c047bc72b5ec00",
            "schema": "639ef4e1307b93b044aee2484e62d82931d689f723cb33f89aeaa627cc0ee5fd",
            "tensor_count": 747,
            "quantized_roles": 168,
            "quant_auxiliary": 504,
            "dense_tensors": 74,
        },
    }

    for resource_id, facts in expected.items():
        resource = resources[resource_id]
        source = resource.sources[0]
        assert (resource.format, resource.precision, resource.quantization) == (
            "safetensors",
            "unknown",
            "int8",
        )
        assert (resource.component, resource.size_bytes, resource.metadata["architecture"]) == (
            facts["component"],
            facts["size"],
            facts["architecture"],
        )
        assert resource.metadata["quantization_contract"] == _CONTRACT
        assert resource.metadata["convrot_groupsize"] == facts["group_size"]
        assert resource.metadata["header_sha256"] == facts["header"]
        assert resource.metadata["schema_sha256"] == facts["schema"]
        assert resource.metadata["header_tensor_count"] == facts["tensor_count"]
        assert resource.metadata["quantized_role_count"] == facts["quantized_roles"]
        assert resource.metadata["quant_auxiliary_count"] == facts["quant_auxiliary"]
        assert resource.metadata["dense_tensor_count"] == facts["dense_tensors"]
        assert (source.repo_id, source.revision, source.filename, source.sha256) == (
            _REPO,
            facts["revision"],
            facts["filename"],
            facts["sha256"],
        )
        assert source.is_exact()

    assert sum(resources[resource_id].size_bytes for resource_id in expected) == 35_817_325_513

    assert all(
        resource_id not in recipe.recipe_resources.values()
        and resource_id not in recipe.fixed_resources
        for resource_id in expected
        for recipe in registry.variants
    )


def test_wan14_convrot_generic_contract_is_supported_by_existing_stored_adapters() -> None:
    """Generic ConvRot support is not a claim about these exact remote headers."""

    assert _CONTRACT in WAN_SUPPORTED
    assert _CONTRACT in UMT5_SUPPORTED
