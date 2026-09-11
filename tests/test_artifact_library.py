"""Host-neutral discovery uses paths and directory structure, never model contents."""

import pytest
from fastapi.testclient import TestClient
from test_service import FakeRuntime

from latentslate_engine.artifact_library import ArtifactLibrary
from latentslate_engine.authoring import canonical_bytes, definition_hash
from latentslate_engine.authoring_store import StoreError
from latentslate_engine.klein9b.contracts import TOKENIZER_FILES
from latentslate_engine.service import create_app


def test_roots_persist_normalize_host_paths_and_retain_missing_directories(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    nested = folder / "nested"
    nested.mkdir()
    library = ArtifactLibrary(tmp_path / "config" / "roots.json")
    root = library.add_root(str(nested / ".."), "Shared models")
    assert root["path"] == str(folder.resolve())
    assert root["name"] == "Shared models" and root["available"]
    restarted = ArtifactLibrary(library.config_path)
    assert restarted.roots() == [root]
    with pytest.raises(StoreError) as duplicate:
        restarted.add_root(str(folder))
    assert duplicate.value.status == 409
    nested.rmdir()
    folder.rmdir()
    missing = restarted.roots()
    assert missing == [{**root, "available": False}]
    assert restarted.refresh()["issues"][0]["root_id"] == root["id"]
    assert ArtifactLibrary(library.config_path).roots() == missing
    restarted.remove_root(root["id"])
    assert ArtifactLibrary(library.config_path).roots() == []


def test_root_registration_requires_existing_absolute_directory(tmp_path):
    library = ArtifactLibrary(tmp_path / "roots.json")
    file = tmp_path / "model.safetensors"
    file.touch()
    for path in ("relative", str(tmp_path / "absent"), str(file), None):
        with pytest.raises(StoreError) as invalid:
            library.add_root(path)
        assert invalid.value.status == 422


def test_overlapping_roots_dedupe_and_typing_reuses_index(tmp_path, monkeypatch):
    outer = tmp_path / "models"
    inner = outer / "loras"
    inner.mkdir(parents=True)
    file = inner / "Klein-9B-example-adapter.safetensors"
    file.touch()
    library = ArtifactLibrary(tmp_path / "roots.json")
    library.add_root(str(outer), "All models")
    specific = library.add_root(str(inner), "My LoRAs")
    result = library.search("flux2_klein9b.two_image", "loras", "kln9adpt")
    assert result["total"] == 1
    candidate = result["results"][0]
    assert candidate["root_id"] == specific["id"]
    assert candidate["relative_path"] == file.name
    assert candidate["absolute_path"] == str(file.resolve())
    assert candidate["structural_status"] == "resolved"
    assert result["model_architecture_checked"] is False

    def forbidden(*args, **kwargs):
        raise AssertionError("Typing must not rescan unchanged roots")

    monkeypatch.setattr(library, "_scan", forbidden)
    for query in ("kle", "klein", "9b", "loras example", "not-a-match"):
        library.search("flux2_klein9b.two_image", "loras", query)


def test_refresh_discovers_added_removed_files_and_root_changes_invalidate(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    first = models / "first.safetensors"
    first.touch()
    library = ArtifactLibrary(tmp_path / "roots.json")
    root = library.add_root(str(models))
    assert library.search("ltx23.t2v", "checkpoint")["total"] == 1
    first.unlink()
    (models / "second.safetensors").touch()
    cached = library.search("ltx23.t2v", "checkpoint")
    assert cached["results"][0]["structural_status"] == "unresolved"
    library.refresh()
    current = library.search("ltx23.t2v", "checkpoint")
    assert (
        current["total"] == 1
        and current["results"][0]["relative_path"] == "second.safetensors"
    )
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "third.safetensors").touch()
    library.add_root(str(extra))
    assert library.search("ltx23.t2v", "checkpoint")["total"] == 2
    library.remove_root(root["id"])
    assert library.search("ltx23.t2v", "checkpoint")["total"] == 1


def test_contextual_directory_companions_and_file_slots(tmp_path):
    models = tmp_path / "models"
    tokenizer = models / "tokenizer"
    tokenizer.mkdir(parents=True)
    for name in TOKENIZER_FILES:
        (tokenizer / name).touch()
    library = ArtifactLibrary(tmp_path / "roots.json")
    library.add_root(str(models))
    result = library.search("flux2_klein9b.t2i", "tokenizer", "tokenizer")
    candidate = result["results"][0]
    assert candidate["kind"] == "directory"
    assert candidate["structural_status"] == "unresolved"
    assert {check["path"] for check in candidate["checks"] if not check["present"]} == {
        "../text_encoder/config.json"
    }
    (models / "text_encoder").mkdir()
    (models / "text_encoder" / "config.json").touch()
    assert (
        library.search("flux2_klein9b.t2i", "tokenizer", "tokenizer")["results"][0][
            "structural_status"
        ]
        == "resolved"
    )
    files = library.search("wan2214b.t2v", "high_adapters", "tokenizer")
    assert files["results"] and all(item["kind"] == "file" for item in files["results"])
    with pytest.raises(StoreError):
        library.search("wan2214b.t2v", "steps")
    with pytest.raises(StoreError):
        library.search("unknown", "checkpoint")


def test_http_discovery_auth_selection_and_manual_outside_path_preserve_documents(
    tmp_path,
):
    models = tmp_path / "External models"
    models.mkdir()
    selected = models / "Exact-Model.safetensors"
    selected.touch()
    outside = tmp_path / "outside.safetensors"
    outside.touch()
    runtime = FakeRuntime()
    with TestClient(
        create_app(home=tmp_path / "engine", token="test-token", executor=runtime)
    ) as client:
        page = client.get("/authoring")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        assert client.get("/authoring/").content == page.content
        for asset, content_type in (
            ("app.js", "javascript"),
            ("style.css", "text/css"),
        ):
            response = client.get(f"/authoring/assets/{asset}")
            assert response.status_code == 200
            assert content_type in response.headers["content-type"]
            assert b"test-token" not in response.content
        assert client.get("/v1/authoring/roots").status_code == 401
        assert client.post("/v1/authoring/artifacts/refresh").status_code == 401
        assert (
            client.get(
                "/v1/authoring/artifacts/search",
                params={"operation": "ltx23.t2v", "field": "checkpoint"},
            ).status_code
            == 401
        )
        client.headers["Authorization"] = "Bearer test-token"
        catalog = client.get("/v1/catalog").content
        root = client.post("/v1/authoring/roots", json={"path": str(models)}).json()
        assert root["available"]
        candidate = client.get(
            "/v1/authoring/artifacts/search",
            params={"operation": "ltx23.t2v", "field": "checkpoint", "q": "exctmdl"},
        ).json()["results"][0]
        record = client.post(
            "/v1/authoring/builtins/ltx23.t2v.v1/duplicate", json={}
        ).json()
        document = record["document"]
        fields = {field["key"]: field for field in document["fields"]}
        fields["checkpoint"]["value"] = candidate["artifact"]
        fields["text_checkpoint"]["value"]["path"] = str(outside)
        before = canonical_bytes(document)
        response = client.put(
            f"/v1/authoring/recipes/{document['id']}",
            json={"base_revision": 1, "document": document},
        )
        assert response.status_code == 200, response.text
        saved = response.json()
        assert canonical_bytes(saved["document"]) == before
        assert saved["definition_hash"] == definition_hash(document)
        assert fields["checkpoint"]["value"]["path"] == candidate["absolute_path"]
        assert client.get("/v1/catalog").content == catalog
        assert runtime.operations == []
        assert client.delete(f"/v1/authoring/roots/{root['id']}").status_code == 200
        assert selected.is_file() and outside.is_file()
