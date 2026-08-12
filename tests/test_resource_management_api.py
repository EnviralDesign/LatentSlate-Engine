from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from latentslate_engine.app import create_app
from latentslate_engine.authoring import inspection as source_inspection
from latentslate_engine.authoring.models import (
    ResourceAddRequest,
    ResourceIdSuggestionRequest,
    ResourceInspectRequest,
    ResourceUpdateRequest,
)
from latentslate_engine.authoring.service import (
    CatalogAuthoringError,
    add_resource,
    delete_resource,
    inspect_resource_source,
    preview_resource,
    suggest_resource_id,
    update_resource,
)
from latentslate_engine.authoring.toml import render_resource_toml
from latentslate_engine.config import Settings
from latentslate_engine.resources import ResourceKind
from latentslate_engine.tools import ToolRegistry


def _settings(tmp_path: Path, *, token: str | None = None) -> Settings:
    settings = Settings(
        home=tmp_path,
        token=token,
        max_upload_bytes=1024 * 1024,
        h3_model_id="unused",
        h3_profile="unused",
        h3_device="cpu",
    )
    settings.ensure_directories()
    return settings


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


def _request(source: Path, *, name: str = "Local Style") -> ResourceAddRequest:
    return ResourceAddRequest(
        inspection=ResourceInspectRequest(source=str(source)),
        resource_id="lora:custom:local-style",
        kind=ResourceKind.LORA,
        family="custom",
        name=name,
        base_model="custom-base",
    )


def test_resource_editor_api_is_authenticated_and_discovers_fresh_local_catalog(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, token="secret")
    app = create_app(settings, ToolRegistry([]))
    source = tmp_path / "local-style.safetensors"
    source.write_bytes(_safetensors_bytes())

    add_resource(settings, _request(source))
    nested = add_resource(
        settings,
        ResourceAddRequest(
            inspection=ResourceInspectRequest(source=str(source)),
            resource_id="model:custom:nested/item",
            kind=ResourceKind.MODEL,
            family="custom",
            name="Nested item",
        ),
    )

    with TestClient(app) as client:
        assert client.get("/v1/authoring/resources").status_code == 401
        capabilities = client.get(
            "/v1/authoring/capabilities",
            headers={"Authorization": "Bearer secret"},
        )
        assert capabilities.status_code == 200
        assert capabilities.json()["resource_authoring"] == {
            "families": ["h3", "ltx23", "wan22", "klein4b", "klein9b", "custom"],
            "kinds": ["model", "lora"],
            "source_unchanged": True,
        }
        response = client.get(
            "/v1/authoring/resources",
            headers={"Authorization": "Bearer secret"},
        )
        assert response.status_code == 200
        catalog = response.json()
        suggestion = client.post(
            "/v1/authoring/resources/suggest-id",
            headers={"Authorization": "Bearer secret"},
            json={
                "kind": "model",
                "family": "custom",
                "name": "Suggested model",
                "source": "hf://example/suggested/model.safetensors",
            },
        )
        assert suggestion.status_code == 200
        assert suggestion.json()["resource_id"].startswith("model:custom:suggested-model-")
        local = next(
            resource
            for resource in catalog["resources"]
            if resource["id"] == "lora:custom:local-style"
        )
        assert local["editable"] is True
        assert local["declaration_origin"] == "local"
        assert local["declaration_path"] == "resource_declarations/lora--custom--local-style.toml"
        assert local["sources"][0]["type"] == "manual"
        retained_preview = client.post(
            "/v1/authoring/resources/preview",
            params={"existing_resource_id": local["id"]},
            headers={"Authorization": "Bearer secret"},
            json={
                "resource_id": local["id"],
                "kind": "lora",
                "family": "custom",
                "name": "Preview local update",
                "base_model": "custom-base",
            },
        )
        assert retained_preview.status_code == 200
        assert retained_preview.json()["valid"] is True
        assert retained_preview.json()["resource"]["name"] == "Preview local update"
        assert any(
            group["kind"] == "lora"
            and group["family"] == "custom"
            and local["id"] in group["resource_ids"]
            for group in catalog["groups"]
        )

        builtin = next(
            resource
            for resource in catalog["resources"]
            if resource["declaration_origin"] == "builtin"
        )
        assert builtin["editable"] is False
        read_only_update = client.put(
            f"/v1/authoring/resources/{builtin['id']}",
            headers={"Authorization": "Bearer secret"},
            json={
                "inspection": {"source": str(source)},
                "resource_id": builtin["id"],
                "kind": builtin["kind"],
                "family": builtin["family"],
                "name": "Attempted edit",
            },
        )
        assert read_only_update.status_code == 409
        detail = client.get(
            f"/v1/authoring/resources/{local['id']}",
            headers={"Authorization": "Bearer secret"},
        )
        assert detail.status_code == 200
        assert detail.json()["name"] == "Local Style"

        nested_detail = client.get(
            "/v1/authoring/resources/model:custom:nested/item",
            headers={"Authorization": "Bearer secret"},
        )
        assert nested_detail.status_code == 200
        assert nested_detail.json()["id"] == nested.resource.id
        nested_update = client.put(
            "/v1/authoring/resources/model:custom:nested/item",
            headers={"Authorization": "Bearer secret"},
            json={
                "resource_id": nested.resource.id,
                "kind": "model",
                "family": "custom",
                "name": "Updated nested item",
            },
        )
        assert nested_update.status_code == 200
        assert nested_update.json()["resource"]["name"] == "Updated nested item"

        credential_exfiltration = client.post(
            "/v1/authoring/resources/inspect",
            headers={"Authorization": "Bearer secret"},
            json={
                "source": "hf://example/model/model.safetensors",
                "token_env": "LATENTSLATE_ENGINE_TOKEN",
            },
        )
        assert credential_exfiltration.status_code == 422
        assert "only HF_TOKEN" in credential_exfiltration.json()["detail"]


def test_resource_id_suggestion_is_stable_and_avoids_existing_ids(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    request = ResourceIdSuggestionRequest(
        kind=ResourceKind.MODEL,
        family="custom",
        name="A Test Model",
        source="hf://example/test/model.safetensors",
    )

    first = suggest_resource_id(settings, request)
    assert suggest_resource_id(settings, request) == first

    source = tmp_path / "model.safetensors"
    source.write_bytes(_safetensors_bytes())
    add_resource(
        settings,
        ResourceAddRequest(
            inspection=ResourceInspectRequest(source=str(source)),
            resource_id=first.resource_id,
            kind=ResourceKind.MODEL,
            family="custom",
            name="A Test Model",
        ),
    )

    collision = suggest_resource_id(settings, request)
    assert collision.base_resource_id == first.resource_id
    assert collision.resource_id == f"{first.resource_id}-2"
    assert collision.collision_index == 1


def test_trusted_service_inspection_retains_custom_provider_token_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setenv("CUSTOM_HF_TOKEN", "trusted-cli-token")
    observed: list[str | None] = []

    class FakeApi:
        def __init__(self, *, token: str | None) -> None:
            observed.append(token)

        def model_info(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                sha="a" * 40,
                siblings=[
                    SimpleNamespace(
                        rfilename="model.safetensors",
                        size=2,
                        lfs={"sha256": "b" * 64, "size": 2},
                    )
                ],
            )

    monkeypatch.setattr(source_inspection, "hf_api_factory", FakeApi)
    result = inspect_resource_source(
        settings,
        ResourceInspectRequest(
            source="hf://example/model/model.safetensors",
            token_env="CUSTOM_HF_TOKEN",
        ),
    )
    assert result.exact_source is not None
    assert result.exact_source.token_env == "CUSTOM_HF_TOKEN"
    assert observed == ["trusted-cli-token"]


def test_local_resource_update_keeps_id_and_path_immutable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = tmp_path / "style.safetensors"
    source.write_bytes(_safetensors_bytes())
    created = add_resource(settings, _request(source))

    updated = update_resource(settings, created.resource.id, _request(source, name="Updated Style"))
    assert updated.resource.id == created.resource.id
    assert updated.resource.relative_path == created.resource.relative_path
    assert updated.resource.name == "Updated Style"
    assert updated.declaration_path == created.declaration_path

    moved = _request(source, name="Moved Style").model_copy(
        update={"relative_path": "loras/custom/moved-style.safetensors"}
    )
    with pytest.raises(CatalogAuthoringError, match="cannot move"):
        update_resource(settings, created.resource.id, moved)

    changed_id = _request(source).model_copy(update={"resource_id": "lora:custom:other"})
    with pytest.raises(CatalogAuthoringError, match="must match"):
        update_resource(settings, created.resource.id, changed_id)


def test_local_resource_deletion_can_keep_or_remove_an_unreferenced_artifact(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "engine")
    source = tmp_path / "style.safetensors"
    source.write_bytes(_safetensors_bytes())
    created = add_resource(settings, _request(source))
    declaration = Path(created.declaration_path)
    declaration_bytes = declaration.read_bytes()
    artifact = Path(created.artifact_path)

    retained = delete_resource(settings, created.resource.id)
    assert retained.declaration_removed is True
    assert retained.artifact_removed is False
    assert retained.resulting_resource is not None
    assert retained.resulting_resource.declaration_origin == "discovered"
    assert retained.resulting_resource.editable is False
    assert artifact.is_file()
    assert not declaration.exists()

    declaration.write_bytes(declaration_bytes)
    removed = delete_resource(settings, created.resource.id, delete_artifact=True)
    assert removed.declaration_removed is True
    assert removed.artifact_removed is True
    assert removed.resulting_resource is None
    assert not declaration.exists()
    assert not artifact.exists()


def test_resource_deletion_is_blocked_by_published_recipes_and_drafts(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "engine")
    source = tmp_path / "style.safetensors"
    source.write_bytes(_safetensors_bytes())
    created = add_resource(settings, _request(source))
    recipe = f'''[runnable_recipe]
key = "custom.local-style"
name = "Custom local style"
family = "custom"
base_tool = "custom.tool"

[[runnable_recipe.loras]]
slot = "style"
resource = "{created.resource.id}"
'''
    published_path = settings.recipes_root / "custom" / "local-style.toml"
    published_path.parent.mkdir(parents=True, exist_ok=True)
    published_path.write_text(recipe, encoding="utf-8")
    draft_path = settings.home / "drafts" / "recipes" / "custom" / "local-style.toml"
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path.write_text(
        f'''[runnable_recipe]
key = "custom.local-style-draft"
name = "Custom local style draft"
family = "custom"
base_tool = "custom.tool"

[[runnable_recipe.loras]]
slot = "style"
exposed = true
allowed = ["{created.resource.id}"]
default = "{created.resource.id}"
''',
        encoding="utf-8",
    )

    with pytest.raises(CatalogAuthoringError) as captured:
        delete_resource(settings, created.resource.id, delete_artifact=True)
    message = str(captured.value)
    assert "recipes depend on it" in message
    assert "custom.local-style" in message
    assert "custom.local-style-draft" in message
    assert Path(created.declaration_path).is_file()
    assert Path(created.artifact_path).is_file()


def test_resource_deletion_api_is_authenticated_local_only_and_dependency_safe(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "engine", token="secret")
    source = tmp_path / "style.safetensors"
    source.write_bytes(_safetensors_bytes())
    created = add_resource(
        settings,
        _request(source).model_copy(
            update={"resource_id": "lora:custom:nested/style", "name": "Nested style"}
        ),
    )
    app = create_app(settings, ToolRegistry([]))

    with TestClient(app) as client:
        endpoint = f"/v1/authoring/resources/{created.resource.id}"
        assert client.request("DELETE", endpoint, json={"delete_artifact": False}).status_code == 401
        builtin_id = "model:klein4b:black-forest-labs--flux.2-klein-4b"
        protected = client.request(
            "DELETE",
            f"/v1/authoring/resources/{builtin_id}",
            headers={"Authorization": "Bearer secret"},
            json={"delete_artifact": False},
        )
        assert protected.status_code == 409
        assert "not owned" in protected.json()["detail"]

        response = client.request(
            "DELETE",
            endpoint,
            headers={"Authorization": "Bearer secret"},
            json={"delete_artifact": True},
        )
        assert response.status_code == 200
        assert response.json()["declaration_removed"] is True
        assert response.json()["artifact_removed"] is True


def test_retained_source_update_and_preview_preserve_unowned_declaration_fields(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    source = tmp_path / "style.safetensors"
    source.write_bytes(_safetensors_bytes())
    created = add_resource(settings, _request(source))
    enriched = created.resource.model_copy(
        update={
            "trigger_words": ["local-style"],
            "default_strength": 0.75,
            "config": "style-config",
            "tags": ["existing"],
            "metadata": {"existing": True},
        }
    )
    declaration = Path(created.declaration_path)
    declaration.write_text(render_resource_toml(enriched), encoding="utf-8")
    before = declaration.read_bytes()
    retained = ResourceUpdateRequest(
        resource_id=created.resource.id,
        kind=ResourceKind.LORA,
        family="custom",
        name="Updated local style",
        base_model="custom-base",
    )

    preview = preview_resource(
        settings,
        retained,
        existing_resource_id=created.resource.id,
    )
    assert preview.valid
    assert preview.resource is not None
    assert preview.resource.trigger_words == ["local-style"]
    assert preview.resource.default_strength == 0.75
    assert preview.resource.config == "style-config"
    assert preview.resource.tags == ["existing"]
    assert preview.resource.metadata == {"existing": True}
    assert declaration.read_bytes() == before

    updated = update_resource(settings, created.resource.id, retained)
    assert updated.resource.name == "Updated local style"
    assert updated.resource.trigger_words == ["local-style"]
    assert updated.resource.default_strength == 0.75
    assert updated.resource.config == "style-config"
    assert updated.resource.tags == ["existing"]
    assert updated.resource.metadata == {"existing": True}


def test_resource_preview_does_not_publish_create(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "engine")
    source = tmp_path / "preview.safetensors"
    source.write_bytes(_safetensors_bytes())
    request = ResourceAddRequest(
        inspection=ResourceInspectRequest(source=str(source)),
        resource_id="model:custom:preview",
        kind=ResourceKind.MODEL,
        family="custom",
        name="Preview",
    )

    preview = preview_resource(settings, request)
    assert preview.valid
    assert preview.resource is not None
    assert preview.toml is not None
    assert not list(settings.resource_declarations_root.glob("*.toml"))
    assert not list(settings.model_root.rglob("preview.safetensors"))


def test_current_inspection_provenance_overwrites_stale_client_metadata(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "engine")
    source = tmp_path / "metadata.safetensors"
    source.write_bytes(_safetensors_bytes())
    result = add_resource(
        settings,
        ResourceAddRequest(
            inspection=ResourceInspectRequest(source=str(source)),
            resource_id="model:custom:metadata",
            kind=ResourceKind.MODEL,
            family="custom",
            name="Metadata",
            metadata={
                "authoring_source_type": "stale",
                "authoring_canonical_source": "stale://source",
                "format": "wrong",
                "unrelated": "retained",
            },
        ),
    )
    assert result.resource.metadata["authoring_source_type"] == "local"
    assert result.resource.metadata["authoring_canonical_source"] == "local-import"
    assert result.resource.metadata["format"] == "safetensors"
    assert result.resource.metadata["unrelated"] == "retained"


def test_package_owned_resource_cannot_be_updated(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = tmp_path / "style.safetensors"
    source.write_bytes(_safetensors_bytes())
    with pytest.raises(CatalogAuthoringError, match="not owned"):
        update_resource(
            settings,
            "model:klein4b:black-forest-labs--flux.2-klein-4b",
            ResourceAddRequest(
                inspection=ResourceInspectRequest(source=str(source)),
                resource_id="model:klein4b:black-forest-labs--flux.2-klein-4b",
                kind=ResourceKind.MODEL,
                family="klein4b",
                name="Attempted edit",
            ),
        )


def test_authored_resource_name_and_lora_base_model_are_required(tmp_path: Path) -> None:
    source = tmp_path / "style.safetensors"
    source.write_bytes(_safetensors_bytes())
    with pytest.raises(ValidationError, match="name"):
        ResourceAddRequest(
            inspection=ResourceInspectRequest(source=str(source)),
            resource_id="model:custom:no-name",
            kind=ResourceKind.MODEL,
            family="custom",
        )
    with pytest.raises(ValidationError, match="blank"):
        ResourceAddRequest(
            inspection=ResourceInspectRequest(source=str(source)),
            resource_id="model:custom:blank-name",
            kind=ResourceKind.MODEL,
            family="custom",
            name="   ",
        )

    settings = _settings(tmp_path / "engine")
    missing_base_model = ResourceAddRequest(
        inspection=ResourceInspectRequest(source=str(source)),
        resource_id="lora:custom:no-base-model",
        kind=ResourceKind.LORA,
        family="custom",
        name="No base model",
    )
    with pytest.raises(CatalogAuthoringError, match="require base_model"):
        add_resource(settings, missing_base_model)
