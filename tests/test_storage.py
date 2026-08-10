from io import BytesIO
from pathlib import Path

from latentslate_engine.config import Settings
from latentslate_engine.storage import Storage


def settings(tmp_path: Path) -> Settings:
    return Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024 * 1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )


def test_asset_uploads_are_content_addressed_and_deduplicated(tmp_path: Path):
    storage = Storage(settings(tmp_path))

    first = storage.store_asset(
        BytesIO(b"same-content"),
        "first.png",
        "image/png",
        1024,
    )
    second = storage.store_asset(
        BytesIO(b"same-content"),
        "renamed.png",
        "image/png",
        1024,
    )

    assert first.id == second.id
    assert first.sha256 == second.sha256
    assert first.path == second.path
    assert storage.resolve_asset(first.id) == first.path
    folders = [
        path
        for path in storage.settings.assets_dir.iterdir()
        if path.is_dir() and path.name != ".incoming"
    ]
    assert folders == [storage.settings.assets_dir / str(first.id)]


def test_different_asset_content_produces_different_identity(tmp_path: Path):
    storage = Storage(settings(tmp_path))

    first = storage.store_asset(BytesIO(b"one"), "asset.bin", None, 1024)
    second = storage.store_asset(BytesIO(b"two"), "asset.bin", None, 1024)

    assert first.id != second.id
    assert first.sha256 != second.sha256
