from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from latentslate_engine import __main__ as engine_cli
from latentslate_engine import app as engine_app
from latentslate_engine.acquisition import deployment_install as installer
from latentslate_engine.acquisition.resource_install import install_resource
from latentslate_engine.app import create_app
from latentslate_engine.authoring import inspection as source_inspection
from latentslate_engine.authoring.models import (
    AuthoringSourceType,
    RecipeDraftRequest,
    RecipePublishRequest,
    ResourceAddRequest,
    ResourceInspectRequest,
)
from latentslate_engine.authoring.service import (
    CatalogAuthoringError,
    add_resource,
    catalog_disk_revision,
    publish_recipe_draft,
    save_recipe_draft,
    validate_recipe,
)
from latentslate_engine.authoring.toml import (
    load_recipe_file,
    render_recipe_toml,
    render_resource_toml,
)
from latentslate_engine.config import Settings
from latentslate_engine.resources import (
    ResourceDescriptor,
    ResourceFormat,
    ResourceInventory,
    ResourceKind,
    ResourceSource,
    ResourceSourceKind,
    discover_resources,
)
from latentslate_engine.tools import ToolRegistry, default_registry
from latentslate_engine.variants import (
    VariantDefinition,
    VariantInputConfig,
    VariantLoraConfig,
)

PAYLOAD = b"small exact resource bytes"
SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


class _Response:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.body = body
        self.offset = 0
        self.status = status
        self.headers = headers or {}

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.body) - self.offset
        chunk = self.body[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk

    def getcode(self) -> int:
        return self.status

    def close(self) -> None:
        pass


def _settings(tmp_path: Path, *, token: str | None = None) -> Settings:
    value = Settings(
        home=tmp_path,
        token=token,
        max_upload_bytes=1024 * 1024,
        h3_model_id="unused",
        h3_profile="unused",
        h3_device="cpu",
    )
    value.ensure_directories()
    return value


def _safetensors_bytes() -> bytes:
    header = {
        "__metadata__": {"format": "pt"},
        "transformer.block.weight": {
            "dtype": "BF16",
            "shape": [1],
            "data_offsets": [0, 2],
        },
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + b"\x00\x00"


def _recipe(key: str = "test.custom-authoring") -> VariantDefinition:
    return VariantDefinition(
        key=key,
        name="Custom authoring test",
        family="klein4b",
        base_tool="flux2_klein4b.text_to_image",
        inputs={"prompt": VariantInputConfig()},
    )


def test_local_safetensors_inspection_addition_and_discovery(tmp_path: Path):
    settings = _settings(tmp_path / "engine")
    source = tmp_path / "tiny-bf16.safetensors"
    source.write_bytes(_safetensors_bytes())

    result = add_resource(
        settings,
        ResourceAddRequest(
            inspection=ResourceInspectRequest(source=str(source)),
            resource_id="lora:custom:tiny-bf16",
            kind=ResourceKind.LORA,
            family="custom",
            name="Tiny BF16",
            base_model="custom-base",
        ),
    )

    assert result.resource.id == "lora:custom:tiny-bf16"
    assert result.resource.format == ResourceFormat.SAFETENSORS
    assert result.inspection.facts.safetensors is not None
    assert result.inspection.facts.safetensors.tensor_keys == ["transformer.block.weight"]
    assert result.inspection.facts.safetensors.dtypes == ["BF16"]
    assert Path(result.artifact_path).read_bytes() == source.read_bytes()
    declaration = Path(result.declaration_path)
    assert declaration.is_file()
    assert "[[resource.sources]]" in declaration.read_text(encoding="utf-8")

    inventory = discover_resources(settings)
    assert inventory.errors == []
    assert inventory.is_installed(result.resource.id)
    assert inventory.resolve(result.resource.id).metadata["authoring_source_type"] == "local"


def test_direct_https_add_then_fetch_preserves_exact_remote_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(tmp_path / "engine")
    payload = _safetensors_bytes()
    digest = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(
        source_inspection,
        "open_remote_request",
        lambda *_args, **_kwargs: _Response(
            payload,
            headers={"Content-Length": str(len(payload))},
        ),
    )
    result = add_resource(
        settings,
        ResourceAddRequest(
            inspection=ResourceInspectRequest(
                source="https://models.example.invalid/tiny.safetensors",
                source_type=AuthoringSourceType.HTTPS,
                expected_size_bytes=len(payload),
                expected_sha256=digest,
            ),
            resource_id="model:custom:https-tiny",
            kind=ResourceKind.MODEL,
            family="custom",
            name="HTTPS Tiny",
        ),
    )

    target = Path(result.artifact_path)
    assert not target.exists()
    assert result.resource.sources[0].type == ResourceSourceKind.HTTPS
    assert result.resource.sources[0].url == "https://models.example.invalid/tiny.safetensors"
    assert result.resource.sources[0].sha256 == digest
    assert result.resource.available is False

    monkeypatch.setattr(
        installer,
        "urlopen",
        lambda *_args, **_kwargs: _Response(payload),
    )
    registry = SimpleNamespace(resources=discover_resources(settings))
    fetched = install_resource(settings, registry, result.resource.id)

    assert fetched.status == "installed"
    assert target.read_bytes() == payload


def test_direct_https_publication_requires_exact_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(tmp_path / "engine")
    monkeypatch.setattr(
        source_inspection,
        "open_remote_request",
        lambda *_args, **_kwargs: _Response(
            PAYLOAD,
            headers={"Content-Length": str(len(PAYLOAD))},
        ),
    )

    with pytest.raises(CatalogAuthoringError, match="exact declaration"):
        add_resource(
            settings,
            ResourceAddRequest(
                inspection=ResourceInspectRequest(
                    source="https://models.example.invalid/tiny.safetensors",
                    source_type=AuthoringSourceType.HTTPS,
                    expected_size_bytes=len(PAYLOAD),
                ),
                resource_id="model:custom:https-unpinned",
                kind=ResourceKind.MODEL,
                family="custom",
                name="HTTPS unpinned",
            ),
        )


def test_huggingface_file_and_filtered_snapshot_are_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(tmp_path)

    class FakeApi:
        def model_info(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                sha="a" * 40,
                siblings=[
                    SimpleNamespace(
                        rfilename="nested/model.safetensors",
                        size=10,
                        lfs={"sha256": "b" * 64, "size": 10},
                    ),
                    SimpleNamespace(rfilename="config.json", size=4, lfs=None),
                    SimpleNamespace(rfilename="README.md", size=7, lfs=None),
                ],
            )

    monkeypatch.setattr(source_inspection, "hf_api_factory", lambda **_kwargs: FakeApi())
    monkeypatch.setattr(
        source_inspection,
        "hf_safetensors_header_probe",
        lambda *_args: (None, None),
    )
    exact = source_inspection.inspect_source(
        ResourceInspectRequest(
            source="hf://example/model/nested/model.safetensors",
        ),
        settings,
    )
    assert exact.exact_source is not None
    assert exact.exact_source.revision == "a" * 40
    assert exact.exact_source.filename == "nested/model.safetensors"
    assert exact.facts.sha256 == "b" * 64

    snapshot = source_inspection.inspect_source(
        ResourceInspectRequest(
            source="hf://example/model",
            allow_patterns=["*.json", "nested/*"],
            ignore_patterns=["README*"],
        ),
        settings,
    )
    assert snapshot.exact_source is not None
    assert snapshot.exact_source.allow_patterns == ("*.json", "nested/*")
    assert snapshot.facts.size_bytes == 14
    assert snapshot.detected["selected_file_count"] == 2


def test_civitai_requires_explicit_file_when_version_is_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(tmp_path)
    metadata = {
        "baseModel": "Flux.2",
        "files": [
            {
                "id": 1,
                "name": "one.safetensors",
                "sizeBytes": 3,
                "hashes": {"SHA256": "1" * 64},
            },
            {
                "id": 2,
                "name": "two.safetensors",
                "sizeBytes": 4,
                "hashes": {"SHA256": "2" * 64},
            },
        ],
    }
    monkeypatch.setattr(source_inspection, "read_remote_json", lambda *_args: metadata)

    ambiguous = source_inspection.inspect_source(
        ResourceInspectRequest(source="civitai://version/9"),
        settings,
    )
    assert ambiguous.exact_source is None
    assert [candidate.id for candidate in ambiguous.candidates] == ["1", "2"]

    selected = source_inspection.inspect_source(
        ResourceInspectRequest(
            source="civitai://version/9",
            file_id=2,
            requires_auth=True,
        ),
        settings,
    )
    assert selected.exact_source is not None
    assert selected.exact_source.model_version_id == 9
    assert selected.exact_source.file_id == 2
    assert selected.exact_source.requires_auth is True
    assert selected.facts.size_bytes == 4

    round_tripped = source_inspection.inspect_source(
        ResourceInspectRequest(source=selected.canonical_source),
        settings,
    )
    assert round_tripped.exact_source is not None
    assert round_tripped.exact_source.model_version_id == 9
    assert round_tripped.exact_source.file_id == 2


def test_fixed_lora_recipe_enters_exact_authoring_closure(tmp_path: Path):
    settings = _settings(tmp_path / "engine")
    source = tmp_path / "style.safetensors"
    source.write_bytes(_safetensors_bytes())
    added = add_resource(
        settings,
        ResourceAddRequest(
            inspection=ResourceInspectRequest(source=str(source)),
            resource_id="lora:klein4b:style",
            kind=ResourceKind.LORA,
            family="klein4b",
            name="Klein style",
            base_model="flux.2-klein",
        ),
    )
    definition = _recipe("test.custom-lora-authoring").model_copy(
        update={
            "loras": [
                VariantLoraConfig(
                    slot="style",
                    resource=added.resource.id,
                    strength=0.8,
                )
            ]
        }
    )

    validation = validate_recipe(
        settings,
        RecipeDraftRequest(definition=definition),
        registry=default_registry(settings, emit_warnings=False),
    )

    assert validation.valid, validation.errors
    assert validation.closure is not None
    assert validation.closure.recipes[0].fixed_resources == [added.resource.id]
    assert [resource.id for resource in validation.closure.resources] == [added.resource.id]


def test_duplicate_resource_id_is_refused_without_clobber(tmp_path: Path):
    settings = _settings(tmp_path / "engine")
    source = tmp_path / "one.safetensors"
    source.write_bytes(PAYLOAD)
    request = ResourceAddRequest(
        inspection=ResourceInspectRequest(source=str(source)),
        resource_id="lora:custom:duplicate",
        kind=ResourceKind.LORA,
        family="custom",
        name="Duplicate",
        base_model="custom-base",
    )
    first = add_resource(settings, request)
    declaration_bytes = Path(first.declaration_path).read_bytes()
    artifact_bytes = Path(first.artifact_path).read_bytes()

    with pytest.raises(CatalogAuthoringError, match="already exists"):
        add_resource(settings, request)

    assert Path(first.declaration_path).read_bytes() == declaration_bytes
    assert Path(first.artifact_path).read_bytes() == artifact_bytes


def test_resource_publication_rolls_back_artifact_and_declaration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from latentslate_engine.authoring import resource_authoring

    settings = _settings(tmp_path / "engine")
    source = tmp_path / "rollback.safetensors"
    source.write_bytes(PAYLOAD)
    real_discover = discover_resources
    calls = 0

    def synthetic_discover(value: Settings) -> ResourceInventory:
        nonlocal calls
        calls += 1
        result = real_discover(value)
        if calls > 1:
            result.errors.append("synthetic post-publication catalog failure")
        return result

    monkeypatch.setattr(resource_authoring, "discover_resources", synthetic_discover)
    with pytest.raises(CatalogAuthoringError, match="synthetic"):
        add_resource(
            settings,
            ResourceAddRequest(
                inspection=ResourceInspectRequest(source=str(source)),
                resource_id="lora:custom:rollback",
                kind=ResourceKind.LORA,
                family="custom",
                name="Rollback",
                base_model="custom-base",
            ),
        )

    assert not list(settings.resource_declarations_root.glob("*rollback*.toml"))
    assert not list(settings.lora_root.rglob("rollback.safetensors"))


def test_single_resource_fetch_reuses_safe_installer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(tmp_path)
    resource = ResourceDescriptor(
        id="lora:custom:fetch-one",
        kind=ResourceKind.LORA,
        family="custom",
        name="Fetch One",
        relative_path="loras/custom/fetch-one.safetensors",
        format=ResourceFormat.SAFETENSORS,
        size_bytes=len(PAYLOAD),
        sources=[
            ResourceSource(
                type=ResourceSourceKind.CIVITAI,
                url="https://download.invalid/fetch-one",
                sha256=SHA256,
            )
        ],
    )
    target = settings.home / resource.relative_path
    inventory = ResourceInventory(resources=[resource], paths={resource.id: target})
    registry = SimpleNamespace(resources=inventory)
    monkeypatch.setattr(
        installer,
        "urlopen",
        lambda *_args, **_kwargs: _Response(PAYLOAD),
    )

    result = install_resource(settings, registry, resource.id)

    assert result.status == "installed"
    assert target.read_bytes() == PAYLOAD
    assert install_resource(settings, registry, resource.id).status == "skipped_installed"


def test_generated_resource_toml_is_stable_and_round_trips():
    resource = ResourceDescriptor(
        id="model:custom:stable",
        kind=ResourceKind.MODEL,
        family="custom",
        name="Stable",
        relative_path="models/custom/stable.safetensors",
        format=ResourceFormat.SAFETENSORS,
        size_bytes=len(PAYLOAD),
        metadata={"schema": "abc", "tensor_dtypes": ["BF16"]},
        sources=[ResourceSource(type=ResourceSourceKind.MANUAL, sha256=SHA256)],
    )
    rendered = render_resource_toml(resource)
    assert rendered == render_resource_toml(resource)
    assert rendered.index('id = "model:custom:stable"') < rendered.index('kind = "model"')


def test_recipe_toml_round_trip_preserves_typed_definition(tmp_path: Path):
    definition = _recipe()
    rendered = render_recipe_toml(definition)
    path = tmp_path / "recipe.toml"
    path.write_text(rendered, encoding="utf-8")

    assert load_recipe_file(path) == definition
    assert rendered == render_recipe_toml(definition)


def test_recipe_draft_validation_publication_and_rediscovery(tmp_path: Path):
    settings = _settings(tmp_path)
    registry = default_registry(settings, emit_warnings=False)
    request = RecipeDraftRequest(definition=_recipe())

    validation = validate_recipe(settings, request, registry=registry)
    assert validation.valid, validation.errors
    assert validation.closure is not None

    draft = save_recipe_draft(settings, request, registry=registry)
    assert Path(draft.draft_path).is_file()
    publication = publish_recipe_draft(
        settings,
        draft.draft_key,
        RecipePublishRequest(),
        registry=registry,
    )
    assert publication.activation.required_action == "next_cli_invocation"
    refreshed = default_registry(settings, emit_warnings=False)
    assert any(entry.key == draft.draft_key for entry in refreshed.variants)


def test_recipe_duplicate_key_requires_explicit_local_replace(tmp_path: Path):
    settings = _settings(tmp_path)
    registry = default_registry(settings, emit_warnings=False)
    request = RecipeDraftRequest(definition=_recipe())
    draft = save_recipe_draft(settings, request, registry=registry)
    publish_recipe_draft(settings, draft.draft_key, RecipePublishRequest(), registry=registry)

    refreshed = default_registry(settings, emit_warnings=False)
    duplicate = validate_recipe(settings, request, registry=refreshed)
    assert not duplicate.valid
    assert any("already exists" in error for error in duplicate.errors)


def test_authoring_api_is_authenticated_and_reports_stale_catalog(tmp_path: Path):
    settings = _settings(tmp_path, token="secret")
    app = create_app(settings, ToolRegistry([]))
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        assert client.get("/v1/authoring/status").status_code == 401
        initial = client.get("/v1/authoring/status", headers=headers)
        assert initial.status_code == 200
        assert initial.json()["stale"] is False

        denied = client.post(
            "/v1/authoring/resources/inspect",
            headers=headers,
            json={"source": str(tmp_path / "server-secret.safetensors")},
        )
        assert denied.status_code == 422
        assert "local filesystem" in denied.json()["detail"]

        denied_https = client.post(
            "/v1/authoring/resources/inspect",
            headers=headers,
            json={
                "source": "https://models.example.invalid/model.safetensors",
                "source_type": "https",
                "expected_size_bytes": 2,
                "expected_sha256": "0" * 64,
            },
        )
        assert denied_https.status_code == 422
        assert "direct HTTPS inspection is disabled" in denied_https.json()["detail"]

        settings.recipes_root.joinpath("manual.toml").write_text("# changed", encoding="utf-8")
        stale = client.get("/v1/authoring/status", headers=headers)
        assert stale.json()["stale"] is True
        assert stale.json()["required_action"] == "restart_engine"


def test_catalog_disk_revision_ignores_drafts(tmp_path: Path):
    settings = _settings(tmp_path)
    before = catalog_disk_revision(settings)
    draft = settings.home / "drafts" / "recipes" / "custom" / "draft.toml"
    draft.parent.mkdir(parents=True)
    draft.write_text("draft", encoding="utf-8")
    assert catalog_disk_revision(settings) == before


def test_app_catalog_revision_is_captured_before_registry_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(tmp_path)

    def mutate_catalog_during_registry_load(*_args: Any, **_kwargs: Any) -> ToolRegistry:
        settings.recipes_root.joinpath("concurrent.toml").write_text(
            "# published during startup\n",
            encoding="utf-8",
        )
        return ToolRegistry([])

    monkeypatch.setattr(engine_app, "default_registry", mutate_catalog_during_registry_load)
    app = engine_app.create_app(settings)

    assert app.state.loaded_catalog_revision != catalog_disk_revision(settings)


def test_resource_inspect_cli_json_is_exact_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    settings = _settings(tmp_path / "engine")
    source = tmp_path / "cli.safetensors"
    source.write_bytes(PAYLOAD)
    monkeypatch.setenv("LATENTSLATE_ENGINE_HOME", str(settings.home))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "latentslate-engine",
            "resources",
            "inspect",
            str(source),
            "--json",
        ],
    )

    engine_cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["source_type"] == "local"
    assert payload["facts"]["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
