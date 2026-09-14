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
    assert {folder["absolute_path"] for folder in result["folders"]} == {
        str(outer.resolve()), str(inner.resolve())
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("Typing must not rescan unchanged roots")

    monkeypatch.setattr(library, "_scan", forbidden)
    for query in ("kle", "klein", "9b", "loras example", "not-a-match"):
        library.search("flux2_klein9b.two_image", "loras", query)


def test_folder_paths_narrow_full_fuzzy_results_before_limit(tmp_path, monkeypatch):
    models = tmp_path / "models"
    paths = [f"checkpoints/other/dragon-{index}.safetensors" for index in range(55)]
    paths += [
        "checkpoints/ltx2/dragon-target.safetensors",
        "checkpoints/ltx20/dragon-neighbor.safetensors",
        "loras/ltx2/dragon-adapter.safetensors",
        "other/dragon-ltx2-filename.safetensors",
    ]
    for relative in paths:
        file = models / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    library = ArtifactLibrary(tmp_path / "roots.json")
    root = library.add_root(str(models))
    other_models = tmp_path / "other-models"
    other_file = other_models / "checkpoints/ltx2/dragon.safetensors"
    other_file.parent.mkdir(parents=True)
    other_file.touch()
    other_root = library.add_root(str(other_models))
    result = library.search("ltx23.t2v", "checkpoint", "drgn", limit=1)
    assert result["total"] == 60 and len(result["results"]) == 1
    facets = {
        folder["relative_path"].replace("\\", "/"): folder
        for folder in result["folders"]
        if folder["root_id"] == root["id"]
    }
    checkpoints = facets["checkpoints"]
    checkpoint_ltx = facets["checkpoints/ltx2"]
    lora_ltx = facets["loras/ltx2"]
    assert checkpoints["count"] == 57
    assert checkpoint_ltx["count"] == lora_ltx["count"] == 1
    assert checkpoint_ltx["id"] != lora_ltx["id"]
    assert checkpoints["id"] in checkpoint_ltx["ancestors"]
    assert checkpoints["id"] not in lora_ltx["ancestors"]
    other_ltx = next(
        folder for folder in result["folders"]
        if folder["root_id"] == other_root["id"]
        and folder["relative_path"].replace("\\", "/") == "checkpoints/ltx2"
    )
    assert other_ltx["id"] != checkpoint_ltx["id"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Changing folder filters must reuse the index")

    monkeypatch.setattr(library, "_scan", forbidden)
    filtered = library.search(
        "ltx23.t2v", "checkpoint", "drgn", folders=[checkpoint_ltx["id"]]
    )
    assert filtered["total"] == 1
    assert filtered["results"][0]["absolute_path"] == str(
        (models / "checkpoints/ltx2/dragon-target.safetensors").resolve()
    )
    assert filtered["folders"] == result["folders"]
    for second in (lora_ltx, other_ltx):
        assert library.search(
            "ltx23.t2v", "checkpoint", "drgn",
            folders=[checkpoint_ltx["id"], second["id"]],
        )["total"] == 2
    assert (
        library.search(
            "ltx23.t2v", "checkpoint", "drgn", folders=[checkpoints["id"]]
        )["total"] == 57
    )
    assert library.search("ltx23.t2v", "checkpoint", "drgn", folders=[])["total"] == 60
    assert (
        library.search(
            "ltx23.t2v", "checkpoint", folders=[checkpoint_ltx["id"]]
        )["total"] == 1
    )
    assert (
        library.search(
            "ltx23.t2v", "checkpoint", "no-such-model", folders=[checkpoint_ltx["id"]]
        )["total"] == 0
    )
    assert (
        library.search(
            "ltx23.t2v", "checkpoint", "drgn", folders=[str(models.parent)]
        )["total"] == 0
    )
    for invalid in ("ltx2", [""], [None]):
        with pytest.raises(StoreError) as error:
            library.search("ltx23.t2v", "checkpoint", folders=invalid)
        assert error.value.status == 422


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
    filtered = library.search(
        "flux2_klein9b.t2i", "tokenizer", "tokenizer", folders=[str(tokenizer)]
    )
    assert filtered["results"] == result["results"]
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
    selected = models / "checkpoints" / "ltx2" / "Exact-Model.safetensors"
    selected.parent.mkdir(parents=True)
    selected.touch()
    (models / "Exact-Model.safetensors").touch()
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
            ("collections.js", "javascript"),
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
        filtered = client.get(
            "/v1/authoring/artifacts/search",
            params={
                "operation": "ltx23.t2v",
                "field": "checkpoint",
                "q": "exctmdl",
                "folder": [str(selected.parent), str(selected.parent.parent)],
            },
        ).json()
        assert filtered["total"] == 1
        candidate = filtered["results"][0]
        assert candidate["absolute_path"] == str(selected.resolve())
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
