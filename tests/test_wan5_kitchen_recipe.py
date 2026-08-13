from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from latentslate_engine.config import Settings
from latentslate_engine.resources import discover_resources
from latentslate_engine.tools import default_registry
from latentslate_engine.wan5_kitchen_recipe import (
    WAN5_BASE_MODEL,
    WAN5_FPS,
    WAN5_GUIDANCE_SCALE,
    WAN5_STEPS,
    Wan5KitchenRuntimeRequest,
    operation_execution_contract,
    rehydrate_wan5_kitchen_runtime_request,
    revalidate_wan5_kitchen_runtime_request,
)


def test_workflow_derived_operation_contract_is_exact() -> None:
    t2v = operation_execution_contract("wan5_t2v")
    i2v = operation_execution_contract("wan5_i2v")

    assert t2v["workflow_revision"] == "f9431bb000ce792094ff345446e22cac1ea6cef3"
    assert t2v["workflow_sha256"] == (
        "6a4d79e1891ae1257654fa78d6716936aff9f8c7e578e4e716eb112f4e5a57c4"
    )
    assert i2v["workflow_sha256"] == (
        "2b1784c9d6ecf03462651d6e8ded7b5cc5e18047e9eb5e2a885fb6c89c5ac515"
    )
    assert t2v["node_semantics_revision"] == "eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f"
    assert t2v["kitchen_revision"] == "78e6dd22fe4ebe7bde5062e050a045dc3a244ee4"
    assert (t2v["steps"], t2v["guidance_scale"], t2v["fps"]) == (
        WAN5_STEPS,
        WAN5_GUIDANCE_SCALE,
        WAN5_FPS,
    )
    assert t2v["scheduler_source_steps"] == 31
    assert t2v["discard_penultimate_sigma"] is True
    assert (t2v["saved_workflow_default_frames"], t2v["engine_product_default_frames"]) == (
        41,
        121,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", True, "schema_version"),
        ("schema_version", 2, "schema_version"),
        ("family", "ltx23", "family"),
        ("base_model", "wrong", "base_model"),
        ("operation", "wan5_flf", "operation"),
    ],
)
def test_worker_manifest_rejects_top_level_type_and_lineage_tamper_before_paths(
    field: str, value: object, message: str
) -> None:
    payload = {
        "schema_version": 1,
        "family": "wan22",
        "operation": "wan5_t2v",
        "base_model": WAN5_BASE_MODEL,
        "execution_contract": operation_execution_contract("wan5_t2v"),
        "component_fingerprint": "component",
        "fingerprint": "request",
        "components": {
            role: {"path": "must-not-be-resolved"}
            for role in ("pipeline_support", "transformer", "text_encoder", "vae")
        },
    }
    payload[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        rehydrate_wan5_kitchen_runtime_request(payload)


@pytest.mark.skipif(
    os.environ.get("LATENTSLATE_WAN5_REAL_HEADERS") != "1",
    reason="set LATENTSLATE_WAN5_REAL_HEADERS=1 for installed Wan 5B proof",
)
def test_installed_recipes_build_revalidate_and_reject_manifest_tamper() -> None:
    home = Path(os.environ["LATENTSLATE_WAN5_HOME"])
    settings = Settings(
        home=home,
        token=None,
        max_upload_bytes=16 * 1024**3,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    inventory = discover_resources(settings)
    assert not inventory.errors
    registry = default_registry(settings, emit_warnings=False)

    for key, operation in (
        ("wan-2-2-5b-ti2v.text-to-video.engine-stored-mixed", "wan5_t2v"),
        ("wan-2-2-5b-ti2v.image-to-video.engine-stored-mixed", "wan5_i2v"),
    ):
        variant = next(tool for tool in registry.tools() if tool.descriptor.key == key)
        request = variant._resolve_recipe_request()
        assert isinstance(request, Wan5KitchenRuntimeRequest)
        assert request.operation == operation
        assert revalidate_wan5_kitchen_runtime_request(request)
        rehydrated = rehydrate_wan5_kitchen_runtime_request(request.to_json_dict())
        assert rehydrated.to_json_dict() == request.to_json_dict()

        tampered = copy.deepcopy(request.to_json_dict())
        tampered["components"]["transformer"]["source_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="canonical contract"):
            rehydrate_wan5_kitchen_runtime_request(tampered)
