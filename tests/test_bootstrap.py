import hashlib

import pytest

from latentslate_engine.artifact_sources import HfFile
from latentslate_engine.bootstrap import install, manifest, plan, selected_assets


class Source:
    def __init__(self, payload=b"small model fixture"):
        self.payload = payload
        self.downloads = 0

    def describe(self, locator):
        return HfFile(
            **locator,
            sha256=hashlib.sha256(self.payload).hexdigest(),
            size=len(self.payload),
            location="https://example.invalid/file",
        )

    def download(self, remote, write, cancel):
        self.downloads += 1
        write(self.payload)


def assets(payload=b"small model fixture"):
    reference = {
        "source": "huggingface",
        "repo": "example/model",
        "revision": "a" * 40,
        "file": "model.bin",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return [
        {
            "path": f"models/{name}/model.bin",
            "families": [name],
            "size_bytes": len(payload),
            "reference": reference,
        }
        for name in ("one", "two")
    ]


def test_bootstrap_downloads_once_and_repeats_offline(tmp_path):
    source = Source()
    entries = assets()
    initial = plan(tmp_path, assets=entries)
    assert initial["download_bytes"] == len(source.payload)
    assert not list(tmp_path.iterdir())
    assert install(tmp_path, assets=entries, source=source)["complete"]
    assert source.downloads == 1
    one, two = [tmp_path / entry["path"] for entry in entries]
    assert one.read_bytes() == two.read_bytes() == source.payload
    assert not one.samefile(two)
    assert one.stat().st_nlink == two.stat().st_nlink == 1
    assert not list((tmp_path / "artifacts" / "sha256").rglob("blob"))
    source.describe = lambda _: pytest.fail(
        "repeat installation must not request metadata"
    )
    assert install(tmp_path, assets=entries, source=source)["complete"]
    assert source.downloads == 1
    assert plan(tmp_path, verify=True, assets=entries)["download_bytes"] == 0


@pytest.mark.parametrize("existing_index", [0, 1])
def test_bootstrap_reuses_existing_verified_file_without_download(
    tmp_path, existing_index
):
    source = Source()
    entries = assets()
    existing = tmp_path / entries[existing_index]["path"]
    existing.parent.mkdir(parents=True)
    existing.write_bytes(source.payload)
    install(tmp_path, assets=entries, source=source)
    assert source.downloads == 0
    assert existing.read_bytes() == (tmp_path / entries[1]["path"]).read_bytes()
    assert existing.stat().st_nlink == 1


def test_bootstrap_does_not_overwrite_conflicting_user_file(tmp_path):
    entries = assets()
    target = tmp_path / entries[0]["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x" * entries[0]["size_bytes"])
    source = Source()
    with pytest.raises(ValueError, match="no files were replaced"):
        install(tmp_path, assets=entries, source=source)
    assert target.read_bytes() == b"x" * entries[0]["size_bytes"]
    assert source.downloads == 0
    assert not (tmp_path / entries[1]["path"]).exists()


def test_bootstrap_rejects_bad_download_without_publishing(tmp_path):
    source = Source()
    source.download = lambda remote, write, cancel: write(b"x" * len(source.payload))
    with pytest.raises(ValueError, match="SHA-256"):
        install(tmp_path, assets=assets(), source=source)
    assert not (tmp_path / assets()[0]["path"]).exists()
    assert not list((tmp_path / "artifacts" / ".staging").iterdir())
    assert install(tmp_path, assets=assets(), source=Source())["complete"]


def test_bootstrap_family_selection_and_traversal(tmp_path):
    assert len(plan(tmp_path, ["one"], assets=assets())["assets"]) == 1
    with pytest.raises(ValueError, match="Unknown"):
        plan(tmp_path, ["unknown"], assets=assets())
    entries = assets()
    entries[0]["path"] = "../outside.bin"
    with pytest.raises(ValueError):
        plan(tmp_path, assets=entries)


def test_bootstrap_materializer_uses_same_canonical_file(tmp_path):
    from latentslate_engine.bootstrap import _cache

    entries = assets()
    source = Source()
    install(tmp_path, assets=entries, source=source)
    cache = _cache(tmp_path, entries)
    digest = entries[0]["reference"]["sha256"]
    assert cache.verified(digest) == tmp_path / entries[0]["path"]
    assert cache.acquire(
        digest,
        lambda *_: pytest.fail("must reuse installed model"),
        lambda: None,
        lambda *_: None,
    )["cache_hit"]
    assert source.downloads == 1


def test_pinned_manifest_covers_every_default_path(tmp_path):
    from latentslate_engine.service import (
        KleinModelPaths,
        KreaModelPaths,
        LtxModelPaths,
        QwenModelPaths,
        ZImageModelPaths,
        Ideogram4ModelPaths,
        SDXLModelPaths,
        WanModelPaths,
    )

    entries = selected_assets()
    for entry in entries:
        path = tmp_path / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert LtxModelPaths.from_home(tmp_path).available()
    assert KleinModelPaths.from_home(tmp_path).available()
    assert KreaModelPaths.from_root(tmp_path / "models").available()
    assert QwenModelPaths.from_root(tmp_path / "models").available()
    assert ZImageModelPaths.from_root(tmp_path / "models").available()
    assert Ideogram4ModelPaths.from_root(tmp_path / "models").available()
    assert SDXLModelPaths.from_root(tmp_path / "models").available()
    wan = WanModelPaths.from_root(tmp_path / "models" / "wan2214b")
    assert all(wan.available(op) for op in ("wan_t2v", "wan_i2v", "wan_flf"))
    assert {family for entry in manifest() for family in entry["families"]} == {
        "ltx23",
        "flux2_klein9b",
        "wan2214b",
        "krea2",
        "qwen2511",
        "zimage",
        "ideogram4",
        "sdxl",
    }
