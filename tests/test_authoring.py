"""Portable authoring contracts; no weights or native backend are required."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from test_service import FakeRuntime

from latentslate_engine.authoring import (
    OPERATIONS,
    canonical_bytes,
    compile_document,
    definition_hash,
    operation_descriptors,
    parse_document,
    validate_document,
)
from latentslate_engine.authoring_builtins import builtin_documents
from latentslate_engine.authoring_store import RecipeStore, StoreError
from latentslate_engine.klein9b import authoring as klein
from latentslate_engine.ltx23 import authoring as ltx
from latentslate_engine.recipe import Adapter, Artifact
from latentslate_engine.service import (
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
    )


def _field(document, key):
    return next(field for field in document["fields"] if field["key"] == key)


def _user(document):
    document = deepcopy(document)
    document["id"] = str(uuid4())
    return document


def _materialize(document):
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


def test_all_eight_builtins_compile_duplicate_and_keep_certified_surfaces(builtins):
    assert len(builtins) == len(operation_descriptors()) == 8
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


def test_validation_layers_all_present_then_missing_and_wrong_kind(builtins):
    for original in builtins.values():
        document = _user(original)
        _materialize(document)
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


def test_klein_companion_and_policy_issues_are_independent(builtins):
    document = _user(builtins["flux2_klein9b.two_image.explicit.v1"])
    _materialize(document)
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
        ("wan2214b.t2v.v1", lambda d: _field(d, "steps").update(value=8), "one of"),
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
            == 8
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
        assert len(client.get("/v1/authoring/recipes").json()["recipes"]) == 8


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
    monkeypatch.setattr(
        two_image, "Klein9BTwoImageRuntime", lambda: SimpleNamespace(close=lambda: None)
    )
    connection = SimpleNamespace(recv=lambda: {"type": "close"}, close=lambda: None)
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
