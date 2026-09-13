"""Portable authoring contracts; no weights or native backend are required."""

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from test_service import FakeRuntime

from latentslate_engine.artifact_materialization import ArtifactMaterializer
from latentslate_engine.artifact_sources import HfFile
from latentslate_engine.authoring import (
    OPERATIONS,
    canonical_bytes,
    compile_document,
    definition_hash,
    operation_descriptors,
    parse_document,
    validate_authoring_contract,
    validate_document,
)
from latentslate_engine.authoring_builtins import builtin_documents
from latentslate_engine.authoring_store import RecipeStore, StoreError
from latentslate_engine.klein9b import authoring as klein
from latentslate_engine.ltx23 import authoring as ltx
from latentslate_engine.recipe import Adapter, Artifact, Capability
from latentslate_engine.service import (
    KreaModelPaths,
    KleinModelPaths,
    LtxModelPaths,
    WanModelPaths,
    create_app,
)
from latentslate_engine.wan2214b import authoring as wan


def test_three_family_ownership_uses_real_cross_field_rules():
    # LTX parallel collections, Klein paired geometry, Wan two ordered phases.
    cases = (
        (ltx, ltx.POLICIES[0], {"transformer_adapter_strengths": (0.5,)}),
        (klein, klein.POLICIES[1], {}),
        (wan, wan.POLICIES[0], {"negative_prompt": ""}),
    )
    bound = []
    for family, policy, extra in cases:
        declared = {field.capability.key for field in policy.fields}
        bindings = {}
        for capability in policy.capabilities.capabilities:
            key = capability.key
            if key in declared:
                continue
            if key in family.ARTIFACT_SLOTS:
                value = Artifact("absent.safetensors")
                if capability.value_type == "adapter":
                    value = Adapter(value, 1.0)
                bindings[key] = (value,) if capability.ordered else value
            elif key in family.HOST_BINDINGS:
                bindings[key] = family.HOST_BINDINGS[key]
            else:
                bindings[key] = extra[key]
        recipe = policy.bind(bindings)
        caller = {
            key: "validation placeholder"
            for key in family.CALLER_INPUTS
            if key in {c.key for c in policy.capabilities.capabilities}
        }
        recipe.resolve(caller)
        bound.append((recipe, caller))
    recipe, caller = bound[0]
    with pytest.raises(ValueError, match="matching order and length"):
        replace(
            recipe,
            fields=tuple(
                replace(f, value=())
                if f.capability.key == "transformer_adapter_strengths"
                else f
                for f in recipe.fields
            ),
        ).resolve(caller)

    recipe, caller = bound[1]
    with pytest.raises(ValueError, match="both be provided"):
        replace(
            recipe,
            fields=tuple(
                replace(f, value=None, nullable=True)
                if f.capability.key == "width"
                else f
                for f in recipe.fields
            ),
        ).resolve(caller)
    recipe, caller = bound[2]
    with pytest.raises(ValueError, match="at most primary and secondary"):
        replace(
            recipe,
            fields=tuple(
                replace(f, value=f.value * 3)
                if f.capability.key == "high_adapters"
                else f
                for f in recipe.fields
            ),
        ).resolve(caller)


@pytest.fixture
def builtins(tmp_path):
    return builtin_documents(
        LtxModelPaths.from_home(tmp_path),
        KleinModelPaths.from_home(tmp_path),
        WanModelPaths.from_root(tmp_path / "wan"),
        KreaModelPaths.from_root(tmp_path),
    )


def _field(document, key):
    return next(field for field in document["fields"] if field["key"] == key)


def _user(document):
    document = deepcopy(document)
    document["id"] = str(uuid4())
    return document


def _materialize(document, root):
    family, policy = OPERATIONS[document["operation"]]
    for field in document["fields"]:
        key = field["key"]
        if key not in family.ARTIFACT_SLOTS:
            continue
        capability = policy.capabilities[key]
        values = field["value"] if capability.ordered else [field["value"]]
        for value in values:
            reference = (
                value["artifact"] if capability.value_type == "adapter" else value
            )
            path = Path(reference["path"])
            assert path.resolve().is_relative_to(root.resolve()), path
            requirements = family.ARTIFACT_SLOTS[key]
            if requirements["kind"] == "directory":
                path.mkdir(parents=True, exist_ok=True)
                for name in requirements["required_files"]:
                    companion = path / name
                    companion.parent.mkdir(parents=True, exist_ok=True)
                    companion.touch()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()


def test_all_nine_builtins_compile_duplicate_and_keep_certified_surfaces(builtins):
    assert len(builtins) == len(operation_descriptors()) == 9
    for document in builtins.values():
        _, policy = OPERATIONS[document["operation"]]
        original = canonical_bytes(document)
        duplicate = _user(document)
        duplicate["name"] += " renamed"
        compiled = compile_document(duplicate)
        assert compiled.surface() == policy.surface()
        assert definition_hash(duplicate) == definition_hash(document)
        assert canonical_bytes(document) == original
        host_fields = {f.capability.key for f in compiled.fields} - {
            f["key"] for f in document["fields"]
        }
        assert host_fields == (
            {"device_index"} if document["operation"].startswith("ltx23.") else set()
        )


@pytest.mark.parametrize("operation", ("t2v", "i2v", "flf"))
@pytest.mark.parametrize(
    "key,certified,value_type,constraints",
    [
        ("steps", 4, "integer", {"min": 3, "max": 8, "step": 1}),
        ("shift", 5.000000000000001, "number", {"min": 4.0, "max": 6.0, "step": 0.5}),
    ],
)
def test_wan_sampling_fixed_exposed_and_family_presentation(
    builtins, operation, key, certified, value_type, constraints
):
    document = _user(
        next(d for d in builtins.values() if d["operation"] == f"wan2214b.{operation}")
    )
    setting = _field(document, key)
    assert setting == {"key": key, "mode": "fixed", "value": certified}
    baseline = compile_document(document).surface()
    setting["value"] = 6
    assert compile_document(document).surface() == baseline
    setting["mode"] = "exposed"
    exposed = next(f for f in compile_document(document).surface() if f["key"] == key)
    assert exposed["type"] == value_type
    assert exposed["constraints"] == constraints
    assert exposed["default"] == 6
    descriptor = next(
        op for op in operation_descriptors() if op["key"] == document["operation"]
    )
    field = next(f for f in descriptor["fields"] if f["key"] == key)
    assert field["presentation"] == wan.FIELD_PRESENTATION[key]
    assert field["presentation"]["certified_value"] == certified
    assert "not certified" in field["presentation"]["advanced_warning"]
    assert "presentation" not in setting and "presentation" not in exposed


def test_validation_layers_all_present_then_missing_and_wrong_kind(builtins, tmp_path):
    for original in builtins.values():
        document = _user(original)
        _materialize(document, tmp_path)
        result = validate_document(document)
        assert result["document_valid"] and result["recipe_compiles"]
        assert result["artifact_resolution"]["status"] == "resolved"
        assert result["execution_readiness"]["status"] == "unverified"
        assert result["execution_readiness"]["backend_checked"] is False
        assert result["issues"] == []
        field = next(
            f
            for f in document["fields"]
            if isinstance(f.get("value"), dict) and "path" in f["value"]
        )
        path = Path(field["value"]["path"])
        path.unlink()
        missing = validate_document(document)
        assert missing["document_valid"] and missing["recipe_compiles"]
        assert missing["artifact_resolution"]["status"] == "unresolved"
        assert missing["execution_readiness"]["status"] == "blocked"
        assert missing["issues"][0]["stage"] == "artifact"
        assert all(
            key in missing["issues"][0]
            for key in ("code", "path", "message", "remediation", "severity")
        )
        path.mkdir()
        assert (
            validate_document(document)["artifact_resolution"]["status"] == "unresolved"
        )
        path.rmdir()
        path.touch()


def test_klein_companion_and_policy_issues_are_independent(builtins, tmp_path):
    document = _user(builtins["flux2_klein9b.two_image.explicit.v1"])
    _materialize(document, tmp_path)
    tokenizer = Path(_field(document, "tokenizer")["value"]["path"])
    (tokenizer / "tokenizer_config.json").unlink()
    _field(document, "width")["maximum"] = 999999
    result = validate_document(document)
    assert result["document_valid"] and not result["recipe_compiles"]
    assert {issue["stage"] for issue in result["issues"]} == {"compile", "artifact"}
    assert any(
        issue["code"] == "missing_companion"
        and "tokenizer_config.json" in issue["message"]
        for issue in result["issues"]
    )
    assert result["execution_readiness"]["status"] == "blocked"


@pytest.mark.parametrize(
    "key,edit,expected",
    [
        (
            "ltx23.t2v.v1",
            lambda d: _field(d, "transformer_adapter_strengths").update(value=[]),
            "matching order and length",
        ),
        (
            "ltx23.t2v.v1",
            lambda d: _field(d, "width").update(value=2048),
            "width * height",
        ),
        (
            "ltx23.t2v.v1",
            lambda d: _field(d, "duration_seconds").update(minimum=0.5),
            "minimum",
        ),
        ("wan2214b.t2v.v1", lambda d: _field(d, "steps").update(value=9), "at most"),
        (
            "wan2214b.t2v.v1",
            lambda d: _field(d, "high_adapters").update(
                value=_field(d, "high_adapters")["value"] * 3
            ),
            "at most primary and secondary",
        ),
        (
            "flux2_klein9b.two_image.explicit.v1",
            lambda d: _field(d, "width").update(value=None, nullable=True),
            "both be provided",
        ),
        (
            "flux2_klein9b.t2i.v1",
            lambda d: _field(d, "width").update(nullable=True),
            "widen nullability",
        ),
    ],
)
def test_invalid_policy_uses_family_rules(builtins, key, edit, expected):
    document = _user(builtins[key])
    edit(document)
    result = validate_document(document)
    assert result["document_valid"] and not result["recipe_compiles"]
    assert any(expected in issue["message"] for issue in result["issues"])
    assert result["artifact_resolution"]["status"] == "unresolved"


def test_semantic_changes_and_order_change_hash_but_metadata_does_not(builtins):
    original = builtins["wan2214b.t2v.v1"]
    baseline = definition_hash(original)
    renamed = _user(original)
    renamed["name"] = "Other name"
    assert definition_hash(renamed) == baseline
    for key, value in (("negative_prompt", "changed"), ("width", 640)):
        changed = deepcopy(original)
        _field(changed, key)["value"] = value
        assert definition_hash(changed) != baseline
    changed = deepcopy(original)
    _field(changed, "width")["mode"] = "fixed"
    assert definition_hash(changed) != baseline
    changed = deepcopy(original)
    _field(changed, "width")["maximum"] = 640
    assert definition_hash(changed) != baseline
    changed = deepcopy(original)
    adapters = _field(changed, "high_adapters")["value"]
    adapters.append(
        {
            "artifact": {"source": "local", "path": "/other/lora.safetensors"},
            "strength": 0.25,
        }
    )
    before = definition_hash(changed)
    adapters.reverse()
    assert definition_hash(changed) != before
    adapters[0]["strength"] = 0.4
    assert definition_hash(changed) != before


def test_store_immutable_revisions_restart_stale_and_builtin_protection(
    tmp_path, builtins
):
    source = builtins["ltx23.t2v.v1"]
    store = RecipeStore(tmp_path / "store", builtin_ids=[source["id"]])
    with pytest.raises(StoreError, match="immutable"):
        store.save(source, base_revision=None)
    document = _user(source)
    first = store.save(document, base_revision=None)
    assert first["revision"] == 1
    revision_path = store.root / document["id"] / "revisions" / "1.json"
    original_bytes = revision_path.read_bytes()
    document["name"] = "Rename"
    second = store.save(document, base_revision=1)
    assert (
        second["revision"] == 2
        and second["definition_hash"] == first["definition_hash"]
    )
    with pytest.raises(StoreError) as stale:
        store.save(document, base_revision=1)
    assert stale.value.status == 409
    assert revision_path.read_bytes() == original_bytes
    reloaded = RecipeStore(store.root, builtin_ids=[source["id"]])
    assert reloaded.read(document["id"]) == second
    assert reloaded.read(document["id"], 1) == first
    assert reloaded.revisions(document["id"]) == [first, second]
    assert reloaded.list() == [second]


def test_publication_and_schema_lineage_are_separate_from_definition(
    tmp_path, builtins
):
    document = _user(builtins["ltx23.t2v.v1"])
    store = RecipeStore(tmp_path / "publication")
    first = store.save(document, base_revision=None)
    schema = store.publication(document["id"])["schema"]
    assert schema["revision"] == 1
    assert store.publication(document["id"])["enabled"] is False
    original_bytes = (store.root / document["id"] / "revisions/1.json").read_bytes()
    store.set_enabled(document["id"], True)
    store = RecipeStore(store.root)
    assert store.publication(document["id"]) == {
        "enabled": True,
        "schema": schema,
        "record": first,
    }
    assert (
        store.root / document["id"] / "revisions/1.json"
    ).read_bytes() == original_bytes
    document["name"] = "A new display name"
    renamed = store.save(document, base_revision=1)
    assert store.publication(document["id"])["schema"] == schema
    assert renamed["definition_hash"] == first["definition_hash"]
    _field(document, "checkpoint")["value"]["path"] = str(
        tmp_path / "different.safetensors"
    )
    hidden = store.save(document, base_revision=2)
    assert store.publication(document["id"])["schema"] == schema
    assert hidden["definition_hash"] != first["definition_hash"]
    _field(document, "seed")["value"] = 123
    exposed = store.save(document, base_revision=3)
    publication = store.publication(document["id"])
    assert publication["schema"]["revision"] == 2
    assert publication["schema"]["hash"] != schema["hash"]
    assert publication["enabled"] and publication["record"] == exposed
    store.set_enabled(document["id"], False)
    assert store.read(document["id"]) == exposed
    copied = RecipeStore(tmp_path / "imported").import_document(document)["record"]
    assert copied["document"] == exposed["document"]
    assert (
        RecipeStore(tmp_path / "imported").publication(document["id"])["enabled"]
        is False
    )


def test_delete_removes_all_history_and_publication_and_allows_clean_import(
    tmp_path, builtins
):
    from latentslate_engine.authoring_store import _filesystem_writer_lock

    store = RecipeStore(
        tmp_path / "store", builtin_ids=[doc["id"] for doc in builtins.values()]
    )
    document = _user(builtins["ltx23.t2v.v1"])
    recipe_id = document["id"]
    store.save(document, base_revision=None)
    document["name"] = "Second revision"
    store.save(document, base_revision=1)
    store.set_enabled(recipe_id, True)
    other = store.save(_user(document), base_revision=None)
    directory = store.root / recipe_id
    (directory / "revisions" / "3.json").write_text("orphan revision")

    for builtin in builtins.values():
        with pytest.raises(StoreError) as protected:
            store.delete(builtin["id"])
        assert protected.value.status == 409
    with _filesystem_writer_lock(store.root):
        with pytest.raises(StoreError) as busy:
            RecipeStore(store.root).delete(recipe_id)
        assert busy.value.status == 409
    assert store.publication(recipe_id)["enabled"]

    store.delete(recipe_id)
    assert not directory.exists()
    restarted = RecipeStore(store.root)
    assert restarted.list() == [other]
    for read in (restarted.read, restarted.revisions, restarted.publication):
        with pytest.raises(StoreError) as missing:
            read(recipe_id)
        assert missing.value.status == 404
    with pytest.raises(StoreError) as missing:
        store.delete(recipe_id)
    assert missing.value.status == 404
    with pytest.raises(StoreError) as invalid:
        store.delete("../store")
    assert invalid.value.status == 422

    assert restarted.preview_import(document)["status"] == "new"
    imported = restarted.import_document(document)["record"]
    assert imported["document"] == document
    assert imported["revision"] == 1 and imported["parent_revision"] is None
    assert restarted.revisions(recipe_id) == [imported]
    assert restarted.publication(recipe_id)["enabled"] is False
    assert "schema" not in json.loads((directory / "head.json").read_bytes())


def test_list_ignores_a_recipe_deleted_after_directory_discovery(
    tmp_path, builtins, monkeypatch
):
    store = RecipeStore(tmp_path / "store")
    record = store.save(_user(builtins["ltx23.t2v.v1"]), base_revision=None)
    original_read = store.read

    def deleted_before_read(recipe_id):
        store.delete(recipe_id)
        return original_read(recipe_id)

    monkeypatch.setattr(store, "read", deleted_before_read)
    assert store.list() == []
    assert not (store.root / record["document"]["id"]).exists()


def test_opaque_windows_and_posix_paths_roundtrip_hash_and_store(tmp_path, builtins):
    document = _user(builtins["ltx23.t2v.v1"])
    paths = [
        r"D:\Models\Mixed/../Model.safetensors",
        r"\\host\share\模型.safetensors",
        "/mnt/models/a/../模型.safetensors",
    ]
    keys = ["checkpoint", "text_checkpoint", "upsampler"]
    for key, path in zip(keys, paths, strict=True):
        _field(document, key)["value"]["path"] = path
    before = canonical_bytes(document)
    compiled = compile_document(document)
    assert compiled is not None
    assert canonical_bytes(parse_document(json.loads(before))) == before
    store = RecipeStore(tmp_path / "portable")
    saved = store.save(document, base_revision=None)
    assert canonical_bytes(saved["document"]) == before
    assert definition_hash(store.read(document["id"])["document"]) == definition_hash(
        document
    )
    assert [_field(saved["document"], key)["value"]["path"] for key in keys] == paths


def test_definition_hash_is_identical_on_every_host(builtins):
    document = _user(builtins["ltx23.t2v.v1"])
    paths = {
        "checkpoint": r"D:\Models\Mixed/../Model.safetensors",
        "text_checkpoint": r"\\host\share\model.safetensors",
        "upsampler": "/mnt/models/a/../model.safetensors",
    }
    for key, path in paths.items():
        _field(document, key)["value"]["path"] = path
    _field(document, "transformer_adapter_artifacts")["value"][0]["path"] = (
        "/opt/adapters/keep/../lora.safetensors"
    )
    # Fixed semantic document, independent of this test host's temporary paths.
    assert (
        definition_hash(document)
        == "bf20fed91dfde60c8174efffe0c13cca1ab4d138f0e96082d5868084300c924b"
    )


def test_http_auth_duplicate_save_conflict_reload_and_catalog_isolation(tmp_path):
    runtime = FakeRuntime()
    app = create_app(home=tmp_path, token="secret", executor=runtime)
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        assert client.get("/v1/authoring/operations").status_code == 401
        assert client.post("/v1/authoring/validate", json={}).status_code == 401
        client.headers.update(headers)
        catalog_before = client.get("/v1/catalog").content
        definitions = client.get("/v1/authoring/builtins").json()["recipes"]
        assert (
            len(definitions)
            == len(client.get("/v1/authoring/operations").json()["operations"])
            == 9
        )
        for builtin in definitions:
            response = client.post(
                f"/v1/authoring/builtins/{builtin['key']}/duplicate", json={}
            )
            assert response.status_code == 201, response.text
            record = response.json()
            assert record["revision"] == 1
            assert record["document"]["id"] != builtin["document"]["id"]
            assert definition_hash(record["document"]) == definition_hash(
                builtin["document"]
            )
        recipe_id = record["document"]["id"]
        record["document"]["name"] = "Saved rename"
        payload = {"base_revision": 1, "document": record["document"]}
        assert (
            client.put(f"/v1/authoring/recipes/{recipe_id}", json=payload).status_code
            == 200
        )
        assert (
            client.put(f"/v1/authoring/recipes/{recipe_id}", json=payload).status_code
            == 409
        )
        assert (
            client.get(f"/v1/authoring/recipes/{recipe_id}/revisions/1").json()[
                "document"
            ]["name"]
            != "Saved rename"
        )
        assert (
            len(
                client.get(f"/v1/authoring/recipes/{recipe_id}/revisions").json()[
                    "revisions"
                ]
            )
            == 2
        )
        assert (
            client.post(
                "/v1/authoring/recipes", json=definitions[0]["document"]
            ).status_code
            == 409
        )
        assert client.get("/v1/catalog").content == catalog_before
        assert runtime.operations == []
    with TestClient(
        create_app(home=tmp_path, token="secret", executor=FakeRuntime())
    ) as client:
        client.headers.update(headers)
        assert (
            client.get(f"/v1/authoring/recipes/{recipe_id}").json()["document"]["name"]
            == "Saved rename"
        )
        assert len(client.get("/v1/authoring/recipes").json()["recipes"]) == 9


def test_portable_authoring_does_not_import_or_probe_native_backend(tmp_path):
    script = """
import sys
from pathlib import Path
from latentslate_engine.authoring import validate_document
from latentslate_engine.authoring_builtins import builtin_documents
from latentslate_engine.service import LtxModelPaths, KleinModelPaths, WanModelPaths
home = Path(sys.argv[1])
paths = LtxModelPaths.from_home(home)
for path in vars(paths).values():
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
doc = builtin_documents(paths, KleinModelPaths.from_home(home), WanModelPaths.from_root(home))['ltx23.t2v.v1']
result = validate_document(doc)
assert result['document_valid'] and result['recipe_compiles']
assert result['artifact_resolution']['status'] == 'resolved'
assert result['execution_readiness']['status'] == 'unverified'
assert result['execution_readiness']['backend_checked'] is False
assert not {'torch', 'diffusers', 'transformers'} & sys.modules.keys()
assert not any(name.endswith(('.runtime', '.pipeline', '.two_image', '.i2v', '.flf', '.t2v')) for name in sys.modules if name.startswith('latentslate_engine.'))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.native
def test_builtin_wan_negative_prompts_match_native_paths(builtins):
    from latentslate_engine.wan2214b.flf import NEGATIVE_PROMPT as flf_negative
    from latentslate_engine.wan2214b.i2v import NEGATIVE_PROMPT as i2v_negative
    from latentslate_engine.wan2214b.pipeline import NEGATIVE_PROMPT as t2v_negative

    for key, expected in (
        ("wan2214b.t2v.v1", t2v_negative),
        ("wan2214b.i2v.v1_1", i2v_negative),
        ("wan2214b.flf.v1_1", flf_negative),
    ):
        assert _field(builtins[key], "negative_prompt")["value"] == expected


@pytest.mark.native
def test_all_builtin_bindings_match_service_construction(
    tmp_path, monkeypatch, builtins
):
    from types import SimpleNamespace

    from latentslate_engine import service
    from latentslate_engine.klein9b import recipes as klein_recipes
    from latentslate_engine.klein9b import two_image
    from latentslate_engine.ltx23 import flf as ltx_flf
    from latentslate_engine.ltx23 import i2v as ltx_i2v
    from latentslate_engine.ltx23 import t2v as ltx_t2v
    from latentslate_engine.wan2214b import flf as wan_flf
    from latentslate_engine.wan2214b import i2v as wan_i2v
    from latentslate_engine.wan2214b import pipeline

    def compare(actual):
        compiled = compile_document(builtins[actual.key])
        assert compiled.capabilities is actual.capabilities
        assert compiled.surface() == actual.surface()
        assert {f.capability.key: f for f in compiled.fields} == {
            f.capability.key: f for f in actual.fields
        }

    for module, name in (
        (ltx_t2v, "Ltx23T2VRuntime"),
        (ltx_i2v, "Ltx23I2VRuntime"),
        (ltx_flf, "Ltx23FlfRuntime"),
    ):
        monkeypatch.setattr(module, name, lambda identity: None)
    for operation in ("t2v", "i2v", "flf"):
        owner = service._LtxOperationRuntime(
            LtxModelPaths.from_home(tmp_path), operation
        )
        compare(getattr(owner, f"_{operation}_recipe"))

    for module, name in (
        (pipeline, "WanSession"),
        (wan_i2v, "WanI2VSession"),
        (wan_flf, "WanFLFSession"),
    ):
        monkeypatch.setattr(module, name, lambda recipe: None)
    owner = service._WanFamilyRuntime(WanModelPaths.from_root(tmp_path / "wan"))
    inputs = {
        "prompt": "test",
        "start_image": "first.png",
        "end_image": "last.png",
        "width": 512,
        "height": 512,
        "duration_seconds": 5.0,
        "seed": 0,
    }
    for operation in ("t2v", "i2v", "flf"):
        owner._create_session(f"wan_{operation}", inputs)
        compare(getattr(owner, f"_{operation}_recipe"))

    captured = []

    def identity(recipe):
        captured.append(recipe)
        return "same identity"

    monkeypatch.setattr(klein_recipes, "resolve_klein9b_fixed_identity", identity)

    def generated(**kwargs):
        return SimpleNamespace(
            conditioning_reused=False, models_reused=False, reference_reused=False
        )

    monkeypatch.setattr(
        two_image,
        "Klein9BTwoImageRuntime",
        lambda: SimpleNamespace(
            close=lambda: None, generate=generated, generate_two_image=generated
        ),
    )
    messages = iter(
        [
            {
                "type": "generate",
                "operation": operation,
                "inputs": {
                    "prompt": "test",
                    "width": 512,
                    "height": 512,
                    "seed": 0,
                    "image_1": "first.png",
                    "image_2": "last.png",
                },
                "output_path": tmp_path / "out.png",
            }
            for operation in ("klein_t2i", "klein_two_image")
        ]
        + [{"type": "close"}]
    )
    connection = SimpleNamespace(
        recv=lambda: next(messages), send=lambda value: None, close=lambda: None
    )
    service._klein_worker_main(KleinModelPaths.from_home(tmp_path), connection)
    assert len(captured) == 2
    for recipe in captured:
        compare(recipe)


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(format_version=2),
        lambda d: d.update(id="../escape"),
        lambda d: d.update(name=None),
        lambda d: d.update(fields="bad"),
        lambda d: d["fields"].append(
            {"key": "width", "mode": "exposed", "minimum": True}
        ),
        lambda d: d["fields"].append(
            {"key": "width", "mode": "exposed", "step": float("nan")}
        ),
    ],
)
def test_malformed_documents_are_structural_failures(builtins, edit):
    document = _user(builtins["ltx23.t2v.v1"])
    edit(document)
    result = validate_document(document)
    assert not result["document_valid"] and not result["recipe_compiles"]
    assert result["issues"][0]["stage"] == "document"


@pytest.mark.parametrize(
    "key, edit",
    [
        (
            "wan2214b.t2v.v1",
            lambda d: _field(d, "high_adapters")["value"][0].pop("artifact"),
        ),
        (
            "ltx23.t2v.v1",
            lambda d: _field(d, "checkpoint").update(
                value={"source": "remote", "path": "x"}
            ),
        ),
        (
            "ltx23.flf.v1_1",
            lambda d: d["fields"].append(
                {
                    "key": "upsampler",
                    "mode": "fixed",
                    "value": {"source": "local", "path": "/x"},
                }
            ),
        ),
        (
            "ltx23.t2v.v1",
            lambda d: d["fields"].append(
                {"key": "device_index", "mode": "fixed", "value": 2}
            ),
        ),
        (
            "ltx23.t2v.v1",
            lambda d: _field(d, "prompt").update(value="stored caller prompt"),
        ),
    ],
)
def test_invalid_artifacts_and_ownership_return_diagnostics(builtins, key, edit):
    document = _user(builtins[key])
    edit(document)
    result = validate_document(document)
    assert result["document_valid"] and not result["recipe_compiles"]
    assert result["issues"] and result["execution_readiness"]["status"] == "blocked"


def test_two_store_instances_cannot_overwrite_same_base_revision(tmp_path, builtins):
    from concurrent.futures import ThreadPoolExecutor

    document = _user(builtins["ltx23.t2v.v1"])
    root = tmp_path / "concurrent"
    RecipeStore(root).save(document, base_revision=None)

    def write(name):
        update = deepcopy(document)
        update["name"] = name
        try:
            return RecipeStore(root).save(update, base_revision=1)["revision"]
        except StoreError as error:
            return error.status

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, ("One", "Two")))
    assert sorted(results) == [2, 409]
    assert len(RecipeStore(root).revisions(document["id"])) == 2


def test_authoring_ownership_is_exhaustive_explicit_and_fail_closed():
    from types import SimpleNamespace

    for family in (ltx, klein, wan):
        partitions = validate_authoring_contract(family)
        for policy in family.POLICIES:
            assert set(partitions[policy.capabilities.key]) == {
                cap.key for cap in policy.capabilities.capabilities
            }
        values = {
            key: getattr(family, key)
            for key in (
                "POLICIES",
                "CALLER_INPUTS",
                "RECIPE_FIELDS",
                "ARTIFACT_SLOTS",
                "HOST_BINDINGS",
            )
        }
        first = family.POLICIES[0]
        capabilities = replace(
            first.capabilities,
            capabilities=(
                *first.capabilities.capabilities,
                Capability("internal_cache_mode", "boolean"),
            ),
        )
        unclassified = SimpleNamespace(
            **{
                **values,
                "POLICIES": (
                    replace(first, capabilities=capabilities),
                    *family.POLICIES[1:],
                ),
            }
        )
        with pytest.raises(
            ValueError, match="internal_cache_mode must have exactly one"
        ):
            validate_authoring_contract(unclassified)
        overlap = SimpleNamespace(
            **{**values, "RECIPE_FIELDS": family.RECIPE_FIELDS | {"prompt"}}
        )
        with pytest.raises(ValueError, match="exactly one"):
            validate_authoring_contract(overlap)
        stale = SimpleNamespace(
            **{**values, "RECIPE_FIELDS": family.RECIPE_FIELDS | {"obsolete_field"}}
        )
        with pytest.raises(ValueError, match="Stale authoring metadata"):
            validate_authoring_contract(stale)
        incompatible = SimpleNamespace(
            **{
                **values,
                "RECIPE_FIELDS": family.RECIPE_FIELDS - {"width"},
                "ARTIFACT_SLOTS": {**family.ARTIFACT_SLOTS, "width": {"kind": "file"}},
            }
        )
        with pytest.raises(ValueError, match="artifact slot must use"):
            validate_authoring_contract(incompatible)


def test_interrupted_head_write_never_resurrects_orphan_history(
    tmp_path, builtins, monkeypatch
):
    from latentslate_engine import authoring_store

    document = _user(builtins["ltx23.t2v.v1"])
    root = tmp_path / "interrupted"
    RecipeStore(root).save(document, base_revision=None)
    write = authoring_store._atomic_json

    def interrupt_head(path, value):
        if path.name == "head.json":
            raise OSError("simulated interruption before publishing head")
        write(path, value)

    document["name"] = "Never published"
    with monkeypatch.context() as patch:
        patch.setattr(authoring_store, "_atomic_json", interrupt_head)
        with pytest.raises(OSError, match="simulated interruption"):
            RecipeStore(root).save(document, base_revision=1)
    orphan_path = root / document["id"] / "revisions" / "2.json"
    orphan_bytes = orphan_path.read_bytes()
    assert RecipeStore(root).read(document["id"])["revision"] == 1
    document["name"] = "Later successful edit"
    saved = RecipeStore(root).save(document, base_revision=1)
    assert saved["revision"] == 3
    restarted = RecipeStore(root)
    assert [record["revision"] for record in restarted.revisions(document["id"])] == [
        1,
        3,
    ]
    assert restarted.read(document["id"])["document"]["name"] == "Later successful edit"
    with pytest.raises(StoreError) as error:
        restarted.read(document["id"], 2)
    assert error.value.status == 404
    assert orphan_path.read_bytes() == orphan_bytes


def test_import_preview_copy_conflicts_and_invalid_policy(tmp_path, builtins):
    reserved = builtins["ltx23.t2v.v1"]
    store = RecipeStore(tmp_path / "imports", builtin_ids=[reserved["id"]])
    document = _user(reserved)
    document["name"] = "Portable recipe"
    assert store.preview_import(document)["status"] == "new"
    assert not store.root.exists()  # Preview never creates store state.
    saved = store.import_document(document)["record"]
    assert saved["revision"] == 1 and saved["parent_revision"] is None
    assert store.preview_import(document)["status"] == "identical"
    assert store.import_document(document)["status"] == "already_present"
    assert len(store.revisions(document["id"])) == 1

    changed = deepcopy(document)
    changed["name"] = "Different name, same semantic hash"
    assert definition_hash(changed) == definition_hash(document)
    assert store.preview_import(changed)["status"] == "conflict"
    with pytest.raises(StoreError) as conflict:
        store.import_document(changed)
    assert conflict.value.status == 409
    copy = store.import_document(changed, as_copy=True)["record"]
    assert copy["document"]["id"] != document["id"]
    assert {**copy["document"], "id": changed["id"]} == changed
    assert copy["definition_hash"] == definition_hash(changed)
    assert copy["revision"] == 1 and copy["parent_revision"] is None
    assert store.read(document["id"]) == saved

    assert store.preview_import(reserved)["status"] == "builtin"
    with pytest.raises(StoreError):
        store.import_document(reserved)
    builtin_copy = store.import_document(reserved, as_copy=True)["record"]
    assert builtin_copy["document"]["id"] != reserved["id"]
    assert builtin_copy["definition_hash"] == definition_hash(reserved)
    assert not (store.root / reserved["id"]).exists()

    invalid = _user(reserved)
    _field(invalid, "width")["value"] = 513
    preview = store.preview_import(invalid)
    assert preview["status"] == "invalid"
    assert any(
        "increments of 64" in issue["message"]
        for issue in preview["validation"]["issues"]
    )
    with pytest.raises(StoreError) as rejected:
        store.import_document(invalid, as_copy=True)
    assert rejected.value.status == 422
    assert not (store.root / invalid["id"]).exists()
    malformed = store.preview_import({"revision": 20, "document": document})
    assert malformed["status"] == "invalid"
    assert not malformed["validation"]["document_valid"]


def test_import_rechecks_collision_after_preview(tmp_path, builtins, monkeypatch):
    store = RecipeStore(tmp_path / "imports")
    other_writer = RecipeStore(store.root)
    document = _user(builtins["ltx23.t2v.v1"])
    competing = deepcopy(document)
    competing["name"] = "Published by another client"
    original_save = store.save

    def interleaved_save(value, *, base_revision):
        other_writer.save(competing, base_revision=None)
        return original_save(value, base_revision=base_revision)

    monkeypatch.setattr(store, "save", interleaved_save)
    with pytest.raises(StoreError) as conflict:
        store.import_document(document)
    assert conflict.value.status == 409
    assert store.read(document["id"])["document"] == competing
    assert len(store.revisions(document["id"])) == 1


def test_http_export_import_preserves_canonical_bytes_across_homes(tmp_path):
    source_runtime, target_runtime = FakeRuntime(), FakeRuntime()
    with (
        TestClient(
            create_app(home=tmp_path / "source", token="", executor=source_runtime)
        ) as source,
        TestClient(
            create_app(
                home=tmp_path / "target", token="import-test", executor=target_runtime
            )
        ) as target,
    ):
        assert (
            target.post(
                "/v1/authoring/imports/preview", json={"document": {}}
            ).status_code
            == 401
        )
        assert (
            target.post("/v1/authoring/imports", json={"document": {}}).status_code
            == 401
        )
        target.headers["Authorization"] = "Bearer import-test"
        catalogs = source.get("/v1/catalog").content, target.get("/v1/catalog").content
        record = source.post(
            "/v1/authoring/builtins/ltx23.t2v.v1/duplicate", json={}
        ).json()
        document = record["document"]
        _field(document, "seed")["value"] = 18446744073709551615
        _field(document, "transformer_adapter_strengths")["value"] = [1.0]
        path = r"Z:\Foreign Models\Mixed/../Exact-模型.safetensors"
        _field(document, "checkpoint")["value"]["path"] = path
        saved = source.put(
            f"/v1/authoring/recipes/{document['id']}",
            json={"base_revision": 1, "document": document},
        ).json()
        exported = source.get(f"/v1/authoring/recipes/{document['id']}/export")
        assert exported.status_code == 200
        assert "attachment;" in exported.headers["content-disposition"]
        assert exported.content == canonical_bytes(document)
        assert b'"value":[1.0]' in exported.content
        assert set(exported.json()) == {
            "format_version",
            "id",
            "name",
            "operation",
            "fields",
        }
        preview = target.post(
            "/v1/authoring/imports/preview", json={"document": exported.json()}
        ).json()
        assert preview["status"] == "new"
        assert preview["validation"]["artifact_resolution"]["status"] == "unresolved"
        assert target.get("/v1/authoring/recipes").json()["recipes"] == []
        imported = target.post(
            "/v1/authoring/imports", json={"document": exported.json()}
        ).json()["record"]
        assert imported["revision"] == 1 and imported["parent_revision"] is None
        assert imported["definition_hash"] == saved["definition_hash"]
        assert _field(imported["document"], "seed")["value"] == 18446744073709551615
        assert _field(imported["document"], "checkpoint")["value"]["path"] == path
        reexported = target.get(f"/v1/authoring/recipes/{document['id']}/export")
        assert reexported.content == exported.content
        assert source.get("/v1/catalog").content == catalogs[0]
        assert target.get("/v1/catalog").content == catalogs[1]
        assert source_runtime.operations == target_runtime.operations == []


HF_TEST_BYTES = b"small portable artifact fixture\n"
HF_TEST_SHA = hashlib.sha256(HF_TEST_BYTES).hexdigest()
HF_TEST_COMMIT = "a" * 40


class TinyHfSource:
    def __init__(self, *, known_digest=True, blocked=False, payload=HF_TEST_BYTES):
        self.downloads = 0
        self.locators = []
        self.known_digest = known_digest
        self.blocked = blocked
        self.payload = payload
        self.started = threading.Event()

    def authentication_configured(self):
        return False

    def describe(self, locator):
        self.locators.append(locator)
        return HfFile(
            locator["repo"],
            HF_TEST_COMMIT,
            locator["file"],
            HF_TEST_SHA if self.known_digest else None,
            len(self.payload),
            "https://cdn.example/file",
        )

    def download(self, file, write, cancel):
        self.downloads += 1
        write(self.payload[:3])
        self.started.set()
        while self.blocked:
            cancel()
            time.sleep(0.01)
        write(self.payload[3:])


def _hf_reference(**changes):
    return {
        "source": "huggingface",
        "repo": "owner/model",
        "revision": HF_TEST_COMMIT,
        "file": "folder/model.safetensors",
        "sha256": HF_TEST_SHA,
        **changes,
    }


def _hf_document(builtins):
    from latentslate_engine.authoring import artifact_dependencies

    document = _user(builtins["ltx23.t2v.v1"])
    for dependency in artifact_dependencies(document):
        dependency["reference"].clear()
        dependency["reference"].update(_hf_reference())
    return document


def _civitai_reference(**changes):
    return {
        "source": "civitai",
        "model_version_id": 102,
        "file_id": 17,
        "sha256": HF_TEST_SHA,
        **changes,
    }


def _civitai_http(monkeypatch):
    """Exercise the real source adapter and redirect policy over mocked HTTP."""
    state = SimpleNamespace(
        downloads=0, requests=[], status=200, payload=HF_TEST_BYTES, stream=None
    )
    state.metadata = {
        "id": 102,
        "name": "v1",
        "model": {"name": "Tiny fixture"},
        "description": "not exposed",
        "files": [
            {
                "id": 16,
                "name": "other.safetensors",
                "type": "Model",
                "hashes": {"AutoV2": "abc"},
                "metadata": {"format": "SafeTensor"},
                "sizeKB": 2,
                "primary": False,
            },
            {
                "id": 17,
                "name": "chosen.safetensors",
                "type": "Model",
                "hashes": {"SHA256": HF_TEST_SHA.upper()},
                "metadata": {"format": "SafeTensor"},
                "sizeKB": len(HF_TEST_BYTES) / 1024,
                "primary": True,
                "downloadUrl": "https://civitai.com/api/download/models/102?type=Model&format=SafeTensor",
            },
        ],
    }

    def handle(request):
        state.requests.append(request)
        if request.url.path == "/api/v1/model-versions/102":
            return httpx.Response(200, json=state.metadata)
        if request.url.host == "civitai.com":
            if state.status != 200:
                return httpx.Response(
                    state.status, text="sensitive diagnostic must not escape"
                )
            return httpx.Response(
                302, headers={"Location": "https://cdn.example/blob?signed=private"}
            )
        state.downloads += 1
        if state.stream is not None:
            return httpx.Response(200, stream=state.stream)
        return httpx.Response(200, content=state.payload)

    transport = httpx.MockTransport(handle)
    real_client = httpx.Client

    def get(url, **kwargs):
        with real_client(transport=transport) as client:
            return client.get(url, **kwargs)

    from contextlib import contextmanager

    @contextmanager
    def stream(method, url, **kwargs):
        with (
            real_client(transport=transport) as client,
            client.stream(method, url, **kwargs) as response,
        ):
            yield response

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "stream", stream)
    return state


@pytest.mark.parametrize(
    "change",
    [
        {"model_version_id": 0},
        {"model_version_id": True},
        {"model_version_id": "102"},
        {"file_id": -1},
        {"file_id": 1.5},
        {"file_id": False},
        {"sha256": HF_TEST_SHA.upper()},
        {"sha256": "b" * 63},
        {"downloadUrl": "https://example.com"},
    ],
)
def test_civitai_canonical_identity_is_strict(builtins, change):
    document = _hf_document(builtins)
    _field(document, "checkpoint")["value"] = _civitai_reference(**change)
    assert not validate_document(document)["recipe_compiles"]


def test_civitai_inspect_exact_pin_and_download_hash(monkeypatch, tmp_path):
    from latentslate_engine.civitai_source import CivitaiSource, civitai_locator

    monkeypatch.delenv("CIVITAI_TOKEN", raising=False)
    http = _civitai_http(monkeypatch)
    source = CivitaiSource()
    assert not source.authentication_configured()
    locator = civitai_locator(
        {"url": "https://civitai.com/models/80/tiny?modelVersionId=102"}
    )
    assert locator == {"model_version_id": 102}
    inspected = source.inspect(locator)
    assert (
        inspected["model_name"] == "Tiny fixture" and inspected["version_name"] == "v1"
    )
    assert [file["file_id"] for file in inspected["files"]] == [16, 17]
    assert inspected["files"][1]["sha256"] == HF_TEST_SHA
    assert inspected["files"][0]["sha256"] is None
    assert "downloadUrl" not in json.dumps(inspected) and "description" not in inspected
    materializer = ArtifactMaterializer(tmp_path / "artifacts")
    known = _artifact_task(
        materializer,
        materializer.pin({"model_version_id": 102, "file_id": 17}, "civitai"),
    )
    assert known["result"]["reference"] == _civitai_reference()
    assert http.downloads == 0 and not known["result"]["downloaded_for_hash"]
    # Pin re-fetches rather than trusting inspection; AutoV2 is not a digest.
    http.metadata["files"][1]["hashes"] = {"AutoV2": "abcdef", "BLAKE3": "f" * 64}
    unknown = _artifact_task(
        materializer,
        materializer.pin({"model_version_id": 102, "file_id": 17}, "civitai"),
    )
    assert unknown["result"]["reference"] == _civitai_reference()
    assert unknown["result"]["downloaded_for_hash"] and http.downloads == 1
    assert materializer.resolve(_civitai_reference()).read_bytes() == HF_TEST_BYTES
    assert all("authorization" not in request.headers for request in http.requests)
    missing = _artifact_task(
        materializer,
        materializer.pin({"model_version_id": 102, "file_id": 999}, "civitai"),
    )
    assert missing["status"] == "failed" and "not present" in missing["error"]
    materializer.close()


@pytest.mark.parametrize("first", ["huggingface", "civitai"])
def test_cross_source_digest_dedup_both_directions(
    builtins, monkeypatch, tmp_path, first
):
    from latentslate_engine.authoring import artifact_dependencies, localize_document

    http = _civitai_http(monkeypatch)
    hf = TinyHfSource()
    materializer = ArtifactMaterializer(tmp_path / "artifacts", hf)
    hf_document = _hf_document(builtins)
    civitai_document = _user(hf_document)
    for dependency in artifact_dependencies(civitai_document):
        dependency["reference"].clear()
        dependency["reference"].update(_civitai_reference())
    documents = (
        [hf_document, civitai_document]
        if first == "huggingface"
        else [civitai_document, hf_document]
    )
    before = [canonical_bytes(document) for document in documents]
    assert materializer.plan(documents)["summary"]["unique_artifacts"] == 1
    result = _artifact_task(materializer, materializer.materialize(documents))
    assert result["status"] == "succeeded" and result["result"]["resolved"]
    assert hf.downloads + http.downloads == 1
    assert (hf.downloads == 1) == (first == "huggingface")
    cache = materializer.resolve(_civitai_reference())
    assert cache == materializer.resolve(_hf_reference())
    assert len(list((tmp_path / "artifacts").rglob("blob"))) == 1
    assert materializer.plan(documents)["summary"]["missing"] == 0
    assert _artifact_task(materializer, materializer.materialize(documents))["result"][
        "resolved"
    ]
    assert hf.downloads + http.downloads == 1
    for document in documents:
        assert compile_document(localize_document(document, materializer.resolve))
    assert [canonical_bytes(document) for document in documents] == before
    materializer.close()


@pytest.mark.parametrize("status", [200, 401, 403])
def test_civitai_auth_is_host_only_and_redirect_errors_are_sanitized(
    builtins, monkeypatch, tmp_path, status
):
    http = _civitai_http(monkeypatch)
    http.status = status
    monkeypatch.setenv("CIVITAI_TOKEN", "test-host-secret")
    materializer = ArtifactMaterializer(tmp_path / "artifacts")
    document = _hf_document(builtins)
    _field(document, "checkpoint")["value"] = _civitai_reference()
    # Other slots share that digest and require no second source download.
    task = _artifact_task(materializer, materializer.materialize([document]))
    for request in http.requests:
        assert "test-host-secret" not in str(request.url)
        if request.url.host == "civitai.com":
            assert request.headers["authorization"] == "Bearer test-host-secret"
        else:
            assert "authorization" not in request.headers
    assert all(
        value not in json.dumps(task) + json.dumps(document)
        for value in ("test-host-secret", "signed=private", "sensitive diagnostic")
    )
    if status == 200:
        assert task["result"]["resolved"]
    else:
        assert task["status"] == "failed" and "Host authentication" in task["error"]
        assert not materializer.cache.path(HF_TEST_SHA).exists()
    materializer.close()


@pytest.mark.parametrize(
    "value",
    [
        {"model_version_id": False},
        {"model_version_id": -1},
        {"url": "https://civitai.com/models/80/tiny"},
        {"url": "https://other.example/models/80?modelVersionId=102"},
        {"url": "https://civitai.com/models/80?modelVersionId=102&modelVersionId=103"},
        {"url": "http://civitai.com/models/80?modelVersionId=102"},
    ],
)
def test_civitai_invalid_locator_returns_structured_error(tmp_path, value):
    app = create_app(home=tmp_path, token="", executor=FakeRuntime())
    with TestClient(app) as client:
        response = client.post("/v1/authoring/sources/civitai/inspect", json=value)
        assert response.status_code == 422 and response.json()["error"]


def test_civitai_metadata_and_bytes_must_match_canonical_sha(
    builtins, monkeypatch, tmp_path
):
    http = _civitai_http(monkeypatch)
    materializer = ArtifactMaterializer(tmp_path / "artifacts")
    document = _hf_document(builtins)
    _field(document, "checkpoint")["value"] = _civitai_reference()
    http.metadata["files"][1]["hashes"]["SHA256"] = "0" * 64
    task = _artifact_task(materializer, materializer.materialize([document]))
    assert task["status"] == "failed" and "differs from the canonical" in task["error"]
    assert http.downloads == 0
    http.metadata["files"][1][
        "hashes"
    ] = {}  # No hash still requires byte verification.
    http.payload = b"wrong bytes"
    task = _artifact_task(materializer, materializer.materialize([document]))
    assert task["status"] == "failed" and "SHA-256 verification" in task["error"]
    assert http.downloads == 1
    assert not materializer.cache.path(HF_TEST_SHA).exists()
    assert not list((tmp_path / "artifacts").rglob("*.partial"))
    materializer.close()


def test_civitai_cancel_never_publishes_partial_bytes(builtins, monkeypatch, tmp_path):
    http = _civitai_http(monkeypatch)
    materializer = ArtifactMaterializer(tmp_path / "artifacts")
    started, release = threading.Event(), threading.Event()

    class SlowStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"x" * (1024 * 1024)
            started.set()
            assert release.wait(5)
            yield b"tail"

    http.stream = SlowStream()
    document = _hf_document(builtins)
    _field(document, "checkpoint")["value"] = _civitai_reference()
    task = materializer.materialize([document])
    try:
        assert started.wait(5)
        assert not materializer.cache.path(HF_TEST_SHA).exists()
        materializer.cancel(task["id"])
    finally:
        release.set()
    finished = _artifact_task(materializer, task)
    assert finished["status"] == "canceled"
    assert not materializer.cache.path(HF_TEST_SHA).exists()
    assert not list((tmp_path / "artifacts").rglob("*.partial"))
    materializer.close()


def _artifact_task(materializer, task):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = materializer.status(task["id"])
        if result["status"] != "running":
            return result
        time.sleep(0.01)
    pytest.fail("Artifact task did not terminate")


@pytest.mark.parametrize(
    "changes",
    [
        {"revision": "main"},
        {"revision": "release-v1"},
        {"revision": "a" * 39},
        {"sha256": "0" * 63},
        {"sha256": "A" * 64},
        {"repo": "../model"},
        {"repo": "https://huggingface.co/owner/model"},
        {"file": "../model"},
        {"file": "/model"},
        {"file": "a\\model"},
        {"file": "a//model"},
        {"token": "must-not-be-canonical"},
        {"cache_path": "/host/path"},
    ],
)
def test_hf_reference_rejects_mutable_or_host_specific_identity(builtins, changes):
    document = _hf_document(builtins)
    _field(document, "checkpoint")["value"] = _hf_reference(**changes)
    validation = validate_document(document)
    assert not validation["recipe_compiles"]
    assert any(item["code"] == "invalid_field_policy" for item in validation["issues"])


def test_hf_policy_is_valid_unmaterialized_but_execution_requires_localization(
    builtins,
):
    document = _hf_document(builtins)
    validation = validate_document(document)
    assert validation["document_valid"] and validation["recipe_compiles"]
    assert validation["artifact_resolution"]["status"] == "unresolved"
    assert "not materialized" in validation["issues"][0]["message"]
    assert canonical_bytes(parse_document(document)) == canonical_bytes(document)
    with pytest.raises(ValueError, match="localized before execution"):
        compile_document(document)
    assert (
        compile_document(document, policy_only=True).surface()
        == compile_document(builtins["ltx23.t2v.v1"]).surface()
    )


def test_hf_fresh_host_import_acquire_dedup_restart_and_corruption(builtins, tmp_path):
    from latentslate_engine.authoring import localize_document

    source = TinyHfSource()
    materializer = ArtifactMaterializer(tmp_path / "host-a" / "artifacts", source)
    document = _hf_document(builtins)
    before = canonical_bytes(document)
    second = _user(document)
    _field(second, "checkpoint")["value"].update(
        repo="another/repository", file="alias.bin"
    )
    plan = materializer.plan([document, second])
    assert plan["summary"]["recipes"] == 2
    assert plan["summary"]["unique_artifacts"] == plan["summary"]["missing"] == 1
    assert len(plan["dependencies"][0]["consumers"]) == 8
    assert plan["summary"]["unknown_sizes"] == 1
    finished = _artifact_task(
        materializer, materializer.materialize([document, second])
    )
    assert finished["status"] == "succeeded" and finished["result"]["resolved"]
    assert source.downloads == 1
    cache = materializer.resolve(_hf_reference())
    assert cache.read_bytes() == HF_TEST_BYTES
    assert (
        cache
        == tmp_path
        / "host-a"
        / "artifacts"
        / "sha256"
        / HF_TEST_SHA[:2]
        / HF_TEST_SHA
        / "blob"
    )
    localized = localize_document(document, materializer.resolve)
    assert _field(localized, "checkpoint")["value"] == {
        "source": "local",
        "path": str(cache),
    }
    assert compile_document(localized)
    assert canonical_bytes(document) == before
    assert _artifact_task(materializer, materializer.materialize([document]))["result"][
        "resolved"
    ]
    assert source.downloads == 1
    materializer.close()

    restarted = ArtifactMaterializer(tmp_path / "host-a" / "artifacts", TinyHfSource())
    assert restarted.plan([document])["summary"]["cached"] == 1
    assert (
        _artifact_task(restarted, restarted.materialize([document]))["status"]
        == "succeeded"
    )
    assert restarted.sources["huggingface"].downloads == 0
    cache.write_bytes(b"corrupt")
    assert restarted.plan([document])["summary"]["missing"] == 1
    assert (
        _artifact_task(restarted, restarted.materialize([document]))["status"]
        == "succeeded"
    )
    assert cache.read_bytes() == HF_TEST_BYTES
    assert restarted.sources["huggingface"].downloads == 1
    restarted.close()

    fresh = ArtifactMaterializer(tmp_path / "host-b" / "artifacts", TinyHfSource())
    store = RecipeStore(
        tmp_path / "host-b" / "authoring" / "recipes", resolve_artifact=fresh.resolve
    )
    imported = store.import_document(json.loads(before))["record"]
    assert imported["definition_hash"] == definition_hash(document)
    assert (
        store.preview_import(document)["validation"]["artifact_resolution"]["status"]
        == "unresolved"
    )
    assert (
        fresh.sources["huggingface"].downloads == 0
    )  # Import never downloads implicitly.
    assert _artifact_task(fresh, fresh.materialize([imported["document"]]))["result"][
        "resolved"
    ]
    assert fresh.sources["huggingface"].downloads == 1
    assert canonical_bytes(store.read(document["id"])["document"]) == before
    assert (
        store.preview_import(document)["validation"]["artifact_resolution"]["status"]
        == "resolved"
    )
    fresh.close()


def test_materialization_cancel_and_wrong_digest_never_publish(builtins, tmp_path):
    source = TinyHfSource(blocked=True)
    materializer = ArtifactMaterializer(tmp_path / "artifacts", source)
    document = _hf_document(builtins)
    task = materializer.materialize([document])
    assert source.started.wait(2)
    with pytest.raises(StoreError, match="active"):
        materializer.materialize([document])
    materializer.cancel(task["id"])
    assert _artifact_task(materializer, task)["status"] == "canceled"
    assert not materializer.cache.path(HF_TEST_SHA).exists()
    assert not list((tmp_path / "artifacts").rglob("*.partial"))
    materializer.sources["huggingface"] = TinyHfSource(
        known_digest=False, payload=b"wrong bytes"
    )
    failed = _artifact_task(materializer, materializer.materialize([document]))
    assert failed["status"] == "failed" and "SHA-256" in failed["error"]
    assert not materializer.cache.path(HF_TEST_SHA).exists()
    assert not list((tmp_path / "artifacts").rglob("*.partial"))
    materializer.close()


@pytest.mark.parametrize(
    "known_digest", [True, False], ids=["hub-sha256", "download-to-hash"]
)
def test_hf_pin_mutable_input_uses_official_metadata_and_bounded_download(
    tmp_path, monkeypatch, known_digest
):
    import httpx

    from latentslate_engine.artifact_sources import hf_locator

    metadata_requests = []
    downloads = []

    def url(repo, filename, **kwargs):
        metadata_requests.append((repo, filename, kwargs))
        return "https://huggingface.co/owner/model/resolve/main/model.bin"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_url=url,
            get_hf_file_metadata=lambda *args, **kwargs: SimpleNamespace(
                commit_hash=HF_TEST_COMMIT,
                etag=HF_TEST_SHA if known_digest else "b" * 40,
                size=len(HF_TEST_BYTES),
                location="https://cdn.example/file",
            ),
        ),
    )

    def serve(request):
        assert "authorization" not in request.headers
        downloads.append(str(request.url))
        return httpx.Response(200, content=HF_TEST_BYTES)

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        monkeypatch.setattr(
            httpx,
            "stream",
            lambda method, url, **kwargs: client.stream(method, url, **kwargs),
        )
        materializer = ArtifactMaterializer(tmp_path / "artifacts")
        friendly = {
            "url": "https://huggingface.co/owner/model/blob/main/model.bin?download=true"
        }
        assert hf_locator(friendly) == {
            "repo": "owner/model",
            "file": "model.bin",
            "revision": "main",
        }
        task = _artifact_task(materializer, materializer.pin(friendly))
        assert task["status"] == "succeeded", task
        assert task["result"]["reference"] == _hf_reference(file="model.bin")
        assert task["result"]["downloaded_for_hash"] is not known_digest
        assert len(downloads) == (0 if known_digest else 1)
        assert metadata_requests[0][2]["revision"] == "main"
        assert metadata_requests[1][2]["revision"] == HF_TEST_COMMIT
        materializer.close()


@pytest.mark.parametrize(
    "locator",
    [
        {"url": "https://evil.example/owner/model/blob/main/file"},
        {"url": "file:///tmp/model"},
        {"repo": "owner/model", "file": "../file"},
        {"repo": "owner/model", "file": "file", "revision": ""},
        {"repo": "owner/model", "file": "file", "revision": "../main"},
        {"repo": "owner/model", "file": "file", "token": "private"},
    ],
)
def test_hf_pin_bad_locator_is_structured_422_without_network(tmp_path, locator):
    materializer = ArtifactMaterializer(tmp_path / "artifacts", TinyHfSource())
    with pytest.raises(StoreError) as caught:
        materializer.pin(locator)
    assert caught.value.status == 422
    assert materializer.sources["huggingface"].locators == []
    materializer.close()
