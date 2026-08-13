from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from latentslate_engine import ltx23_kitchen_recipe as recipe
from latentslate_engine.artifacts import ArtifactIdentity


def test_ltx23_operation_roles_are_distinct_and_minimal() -> None:
    assert recipe.required_ltx23_roles("ltx23_dev_t2v") == recipe.required_ltx23_roles(
        "ltx23_dev_i2v"
    )
    assert recipe.required_ltx23_roles("ltx23_dev_t2v") == {
        "pipeline_support",
        "checkpoint",
        "model_lora",
        "text_encoder",
        "text_lora",
        "latent_upscaler",
    }
    assert recipe.required_ltx23_roles("ltx23_distilled_flf") == {
        "pipeline_support",
        "checkpoint",
        "text_encoder",
    }


def test_ltx23_workflow_derived_schedule_is_fixed() -> None:
    assert recipe.LTX23_FPS == 24
    assert recipe.LTX23_GUIDANCE_SCALE == 1.0
    assert recipe.LTX23_MODEL_LORA_STRENGTH == 0.5
    assert recipe.LTX23_TEXT_LORA_STRENGTH == 1.0
    assert recipe.LTX23_GUIDE_STRENGTH == 0.7
    assert recipe.LTX23_MAIN_SIGMAS == (
        1.0,
        0.99375,
        0.9875,
        0.98125,
        0.975,
        0.909375,
        0.725,
        0.421875,
        0.0,
    )
    assert recipe.LTX23_REFINE_SIGMAS == (0.85, 0.725, 0.4219, 0.0)


def test_ltx23_pipeline_support_is_exact_and_revalidates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    content = b"pinned support"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        recipe,
        "_SUPPORT_FILES",
        MappingProxyType({"text_encoder/config.json": (len(content), digest)}),
    )
    target = tmp_path / "text_encoder" / "config.json"
    target.parent.mkdir()
    target.write_bytes(content)

    plan = recipe.plan_ltx23_pipeline_support(tmp_path)

    assert plan.files == {"text_encoder/config.json": (len(content), digest)}
    assert recipe.revalidate_ltx23_pipeline_support(plan)
    target.write_bytes(b"changed")
    assert not recipe.revalidate_ltx23_pipeline_support(plan)


def test_ltx23_pipeline_support_rejects_unexpected_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(recipe, "_SUPPORT_FILES", MappingProxyType({}))
    (tmp_path / "unexpected.json").write_text("{}", encoding="utf-8")

    try:
        recipe.plan_ltx23_pipeline_support(tmp_path)
    except ValueError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("unexpected LTX support file was accepted")


def test_runtime_request_uses_fresh_recipe_plans_without_immediate_reread(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The later spawn and worker gates, not build(), own stale-plan detection."""

    support = recipe.LTX23PipelineSupportPlan(tmp_path, MappingProxyType({}), "support")
    contracts = recipe._role_contracts("ltx23_distilled_flf")
    resolved = {
        "pipeline_support": SimpleNamespace(
            resource=SimpleNamespace(id="model:ltx23:support"), path=tmp_path
        )
    }
    plans = {"pipeline_support": support}
    for role in ("checkpoint", "text_encoder"):
        path = (tmp_path / f"{role}.safetensors").resolve()
        identity = ArtifactIdentity(path, 1, 1, "a" * 64)
        contract = contracts[role][4]
        assert contract is not None
        resolved[role] = SimpleNamespace(resource=SimpleNamespace(id=f"model:ltx23:{role}"), path=path)
        plans[role] = SimpleNamespace(
            identity=identity,
            contract=contract,
            available=True,
            fingerprint=f"{role}-plan",
            require_available=lambda: None,
            revalidate=lambda: (_ for _ in ()).throw(AssertionError("unexpected reread")),
        )
    validation = recipe.LTX23StoredRecipeValidation(
        True,
        (),
        MappingProxyType(resolved),
        MappingProxyType(plans),
    )
    monkeypatch.setattr(recipe, "validate_ltx23_stored_recipe", lambda *_args, **_kwargs: validation)

    request = recipe.build_ltx23_kitchen_runtime_request(
        SimpleNamespace(operation="ltx23_distilled_flf", base_model=recipe.LTX23_BASE_MODEL),
        SimpleNamespace(),
    )

    assert request.identities["checkpoint"] == plans["checkpoint"].identity
    assert request.identities["text_encoder"] == plans["text_encoder"].identity


def test_kitchen_worker_manifest_round_trips_and_binds_plans(
    monkeypatch,
    tmp_path: Path,
) -> None:
    support_root = tmp_path / "support"
    support_root.mkdir()
    support = recipe.LTX23PipelineSupportPlan(
        support_root,
        MappingProxyType({}),
        "support-fingerprint",
    )
    components: dict[str, dict[str, str | int]] = {
        "pipeline_support": {
            "resource_id": "model:ltx23:support",
            "path": str(support_root),
            "component": "pipeline_support",
            "support_fingerprint": support.fingerprint,
            "file_count": 0,
        }
    }
    identities = {}
    plans = {"pipeline_support": support}
    contracts = recipe._role_contracts("ltx23_distilled_flf")
    plans_by_path = {}
    for role in ("checkpoint", "text_encoder"):
        path = tmp_path / f"{role}.safetensors"
        path.write_bytes(b"header")
        identity = ArtifactIdentity(path.resolve(), 6, path.stat().st_mtime_ns, "a" * 64)
        contract = contracts[role][4]
        assert contract is not None
        plan = SimpleNamespace(
            identity=identity,
            contract=contract,
            available=True,
            fingerprint=f"{role}-plan",
            require_available=lambda: None,
            revalidate=lambda: True,
        )
        plans_by_path[path.resolve()] = plan
        plans[role] = plan
        identities[role] = identity
        components[role] = {
            "resource_id": f"model:ltx23:{role}",
            "path": str(path.resolve()),
            "component": role,
            "size_bytes": identity.size_bytes,
            "mtime_ns": identity.mtime_ns,
            "header_sha256": identity.header_sha256,
            "schema_sha256": contract.schema_sha256,
            "source_sha256": contract.source_sha256,
            "plan_fingerprint": plan.fingerprint,
        }
    request = recipe.LTX23KitchenRuntimeRequest(
        1,
        "ltx23",
        "ltx23_distilled_flf",
        recipe.LTX23_BASE_MODEL,
        dict(reversed(tuple(components.items()))),
        identities,
        plans,
    )

    monkeypatch.setattr(recipe, "plan_ltx23_pipeline_support", lambda _path: support)
    monkeypatch.setattr(
        recipe,
        "probe_artifact",
        lambda path: SimpleNamespace(
            identity=plans_by_path[Path(path).resolve()].identity,
            schema_sha256=plans_by_path[Path(path).resolve()].contract.schema_sha256,
        ),
    )
    monkeypatch.setattr(
        recipe,
        "plan_ltx23_stored_artifact",
        lambda path, _contract: plans_by_path[Path(path).resolve()],
    )
    monkeypatch.setattr(recipe, "revalidate_artifact", lambda _identity: True)

    payload = request.to_json_dict()
    assert payload["execution_contract"] == {
        "workflow_revision": "2b7f823136606344f0bccce249898d771b809aa1",
        "workflow_sha256": "168bc2584ef117133e76341f04e001aab2641b72b75d81b66b5c0b66e56c24a5",
        "node_semantics_revision": "725e6ec60621c6f001af04769173e7dbb3c53541",
        "kitchen_revision": "78e6dd22fe4ebe7bde5062e050a045dc3a244ee4",
        "pinned_workflow_default_width": 1280,
        "pinned_workflow_default_height": 720,
        "engine_acceptance_default_width": 768,
        "engine_acceptance_default_height": 512,
        "dimension_alignment": "dev=/64;distilled_flf=/32",
    }
    rebuilt = recipe.rehydrate_ltx23_kitchen_runtime_request(payload)
    assert rebuilt.to_json_dict() == payload
    assert rebuilt.fingerprint == request.fingerprint
    payload["components"]["checkpoint"]["plan_fingerprint"] = "tampered"
    try:
        recipe.rehydrate_ltx23_kitchen_runtime_request(payload)
    except ValueError as exc:
        assert "plan changed" in str(exc)
    else:
        raise AssertionError("tampered LTX stored plan fingerprint was accepted")
