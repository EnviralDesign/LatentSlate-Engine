import hashlib
import threading
import time
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest

from latentslate_engine.artifact_materialization import ArtifactMaterializer
from latentslate_engine.artifact_sources import HfFile
from latentslate_engine.authoring import artifact_dependencies
from latentslate_engine.authoring_builtins import builtin_documents
from latentslate_engine.bootstrap import manifest
from latentslate_engine.library_downloads import plan_downloads, start_downloads
from latentslate_engine.service import (
    KleinModelPaths,
    KreaModelPaths,
    LtxModelPaths,
    QwenModelPaths,
    ZImageModelPaths,
    WanModelPaths,
)


def documents(home):
    return builtin_documents(
        LtxModelPaths.from_home(home),
        KleinModelPaths.from_home(home),
        WanModelPaths.from_root(home / "models" / "wan2214b"),
        KreaModelPaths.from_root(home / "models"),
        QwenModelPaths.from_root(home / "models"),
        ZImageModelPaths.from_root(home / "models"),
    )


def wait(materializer, task):
    deadline = time.monotonic() + 10
    while task["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
        task = materializer.status(task["id"])
    assert task["status"] != "running"
    return task


def test_exact_builtin_selection_and_shared_tokenizers(tmp_path):
    builtins = documents(tmp_path)
    materializer = ArtifactMaterializer(tmp_path / "artifacts")
    report = plan_downloads(materializer, [(doc, True) for doc in builtins.values()])
    assert not report["summary"]["unresolved"]
    paths = {
        asset["path"]
        for item in report["dependencies"]
        for asset in item["bootstrap_assets"]
    }
    assert paths == {item["path"] for item in manifest()}
    assert report["summary"]["unique_artifacts"] == 33
    single = plan_downloads(materializer, [(builtins["wan2214b.t2v.v1"], True)])
    assert not any(
        "i2v" in a["path"]
        for e in single["dependencies"]
        for a in e["bootstrap_assets"]
    )
    materializer.close()


class Source:
    def __init__(self):
        self.payload = b"synthetic model"
        self.downloads = 0
        self.entered = threading.Event()
        self.block = False

    def describe(self, locator):
        return HfFile(
            **locator,
            sha256=hashlib.sha256(self.payload).hexdigest(),
            size=len(self.payload),
            location="https://example.invalid/file",
        )

    def download(self, remote, write, cancel):
        self.downloads += 1
        self.entered.set()
        while self.block:
            cancel()
            time.sleep(0.01)
        write(self.payload)


def test_builtin_and_custom_share_download_and_retry(tmp_path, monkeypatch):
    source = Source()
    digest = hashlib.sha256(source.payload).hexdigest()
    builtin = documents(tmp_path)["krea2.turbo.t2i.v1"]
    dependency = artifact_dependencies(builtin)[0]
    asset = {
        "path": str(
            Path(dependency["reference"]["path"]).relative_to(tmp_path)
        ).replace("\\", "/"),
        "families": ["krea2"],
        "size_bytes": len(source.payload),
        "reference": {
            "source": "huggingface",
            "repo": "example/model",
            "revision": "a" * 40,
            "file": "model.bin",
            "sha256": digest,
        },
    }
    monkeypatch.setattr(
        "latentslate_engine.library_downloads.selected_assets", lambda families: [asset]
    )
    user = deepcopy(builtin)
    user["id"] = str(uuid4())
    user["fields"][0]["value"] = deepcopy(asset["reference"])
    materializer = ArtifactMaterializer(tmp_path / "artifacts", source)
    selection = [(builtin, True), (user, False)]
    before = deepcopy(selection)
    report = plan_downloads(materializer, selection)
    assert report["summary"]["missing"] == 1
    assert report["summary"]["download_bytes_known"] == len(source.payload)
    task = wait(materializer, start_downloads(materializer, selection))
    assert task["status"] == "succeeded", task
    assert source.downloads == 1
    assert task["overall_bytes"] == len(source.payload)
    assert (
        task["result"]["needs_attention"] == 2
    )  # Other local dependencies remain absent.
    assert selection == before
    assert (
        wait(materializer, start_downloads(materializer, selection))["status"]
        == "succeeded"
    )
    assert source.downloads == 1
    materializer.close()


def test_cancel_keeps_local_files_and_retry_finishes(tmp_path):
    source = Source()
    doc = documents(tmp_path)["krea2.turbo.t2i.v1"]
    doc["id"] = str(uuid4())
    doc["fields"][0]["value"] = {
        "source": "huggingface",
        "repo": "example/model",
        "revision": "a" * 40,
        "file": "model.bin",
        "sha256": hashlib.sha256(source.payload).hexdigest(),
    }
    materializer = ArtifactMaterializer(tmp_path / "artifacts", source)
    source.block = True
    task = start_downloads(materializer, [(doc, False)])
    assert source.entered.wait(5)
    materializer.cancel(task["id"])
    assert wait(materializer, task)["status"] == "canceled"
    assert not list((tmp_path / "artifacts").glob(".staging/*"))
    source.block = False
    assert (
        wait(materializer, start_downloads(materializer, [(doc, False)]))["status"]
        == "succeeded"
    )
    materializer.close()


def test_preview_does_not_hash_or_download(tmp_path, monkeypatch):
    source = Source()
    materializer = ArtifactMaterializer(tmp_path / "artifacts", source)
    monkeypatch.setattr(
        materializer.cache,
        "verified",
        lambda *a, **k: pytest.fail("Preview must not hash"),
    )
    monkeypatch.setattr(
        source, "describe", lambda *a: pytest.fail("Preview must not fetch")
    )
    report = plan_downloads(
        materializer, [(doc, True) for doc in documents(tmp_path).values()]
    )
    assert report["summary"]["missing"] > 0
    assert plan_downloads(materializer, [])["summary"]["recipes"] == 0
    materializer.close()
