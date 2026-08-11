import json
import os
from pathlib import Path

import pytest

from latentslate_engine import model_store
from latentslate_engine.bundles import BUNDLES, BundleDefinition, configured_bundles
from latentslate_engine.config import Settings
from latentslate_engine.model_store import (
    configure_library_cache_environment,
    configured_engine_home,
    configured_model_root,
    engine_data_directories,
    initialize_engine_data,
    repository_directory,
    repository_root,
    require_model_file,
    require_repository,
)
from latentslate_engine.protocol import BundleStatus


def test_engine_home_is_the_single_configurable_storage_root(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LATENTSLATE_ENGINE_HOME", str(tmp_path / "dedicated-drive"))
    ignored_cache = str(tmp_path / "ignored-hf-cache")
    for name in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_ASSETS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HUGGINGFACE_ASSETS_CACHE",
        "HF_XET_CACHE",
        "DIFFUSERS_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
    ):
        monkeypatch.setenv(name, ignored_cache)

    assert configured_engine_home() == (tmp_path / "dedicated-drive").resolve()
    assert configured_model_root() == (tmp_path / "dedicated-drive" / "models").resolve()
    cache_root = configure_library_cache_environment()
    assert cache_root == (tmp_path / "dedicated-drive" / "cache").resolve()
    huggingface_root = cache_root / "huggingface"
    assert os.environ["HF_HOME"] == str(huggingface_root)
    assert os.environ["HF_HUB_CACHE"] == str(huggingface_root / "hub")
    assert os.environ["HF_ASSETS_CACHE"] == str(huggingface_root / "assets")
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(huggingface_root / "hub")
    assert os.environ["HUGGINGFACE_ASSETS_CACHE"] == str(huggingface_root / "assets")
    assert os.environ["HF_XET_CACHE"] == str(huggingface_root / "xet")
    assert os.environ["DIFFUSERS_CACHE"] == str(huggingface_root / "hub")
    assert os.environ["TRANSFORMERS_CACHE"] == str(huggingface_root / "hub")
    assert os.environ["TORCH_HOME"] == str(cache_root / "torch")


def test_installed_package_falls_back_to_visible_launch_directory(monkeypatch, tmp_path: Path):
    launch_directory = tmp_path / "engine-launch"
    launch_directory.mkdir()
    installed_module = tmp_path / "python" / "site-packages" / "latentslate_engine"
    monkeypatch.chdir(launch_directory)
    monkeypatch.setattr(model_store, "__file__", str(installed_module / "model_store.py"))

    assert repository_root() == launch_directory


def test_repository_paths_are_organized_by_model_family(tmp_path: Path):
    path = repository_directory(
        tmp_path,
        "klein4b-basic",
        "black-forest-labs/FLUX.2-klein-4B",
    )

    assert path == tmp_path / "klein4b" / "black-forest-labs--FLUX.2-klein-4B"


def test_model_file_must_stay_within_its_owned_repository(tmp_path: Path):
    repository = repository_directory(tmp_path, "test-basic", "example/model")
    repository.mkdir(parents=True)

    for filename in ("../external.safetensors", str(tmp_path / "external.safetensors")):
        try:
            require_model_file(tmp_path, "test-basic", "example/model", filename)
        except ValueError as exc:
            assert "must stay within its repository" in str(exc)
        else:
            raise AssertionError(f"Expected {filename!r} to be rejected")


def test_model_repository_link_cannot_escape_model_root(tmp_path: Path):
    model_root = tmp_path / "models"
    repository = repository_directory(model_root, "test-basic", "example/model")
    repository.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "weights.bin").write_bytes(b"external")
    try:
        repository.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory links are unavailable: {exc}")

    with pytest.raises(ValueError, match="must stay within model root"):
        require_model_file(
            model_root,
            "test-basic",
            "example/model",
            "weights.bin",
        )


def test_data_initialization_creates_the_complete_layout(tmp_path: Path):
    engine_home = tmp_path / "LatentSlateEngineData"

    assert initialize_engine_data(engine_home) == engine_home
    assert all(path.is_dir() for path in engine_data_directories(engine_home))
    assert (engine_home / "models" / "custom").is_dir()
    assert (engine_home / "loras" / "custom").is_dir()
    assert (engine_home / "cache" / "huggingface" / "hub").is_dir()
    assert (engine_home / "logs").is_dir()
    assert (engine_home / "temp").is_dir()


def test_bundle_install_uses_owned_local_directories(
    monkeypatch,
    tmp_path: Path,
):
    bundle = BundleDefinition(
        id="test-basic",
        name="Test",
        description="Test bundle",
        repo_id="example/model",
    )
    calls = []

    def fake_snapshot_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "model_index.json").write_text("{}", encoding="utf-8")
        calls.append(kwargs)
        return str(local_dir)

    monkeypatch.setattr(
        "latentslate_engine.bundles.snapshot_download",
        fake_snapshot_download,
    )

    installed_path = bundle.install(tmp_path)

    expected = tmp_path / "test" / "example--model"
    assert installed_path == str(expected)
    assert calls[0]["local_dir"] == expected
    assert "cache_dir" not in calls[0]
    assert bundle.status(tmp_path) == BundleStatus.INSTALLED
    manifest = json.loads(
        (tmp_path / "test" / ".latentslate-installed.json").read_text(encoding="utf-8")
    )
    assert manifest["bundle_id"] == "test-basic"
    assert manifest["inventory"] == [
        {
            "path": "test/example--model/model_index.json",
            "size": 2,
        }
    ]
    assert require_repository(tmp_path, "test-basic", "example/model") == expected

    (expected / "model_index.json").unlink()
    assert bundle.status(tmp_path) == BundleStatus.MISSING


def test_canonical_bundle_install_writes_authoritative_artifact_sidecar(
    monkeypatch,
    tmp_path: Path,
):
    bundle = BundleDefinition(
        id="test-bf16",
        name="Test BF16",
        description="Test bundle",
        repo_id="example/bf16-model",
        artifact_precision="bf16",
        artifact_quantization="native",
    )

    def fake_snapshot_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "model_index.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(
        "latentslate_engine.bundles.snapshot_download",
        fake_snapshot_download,
    )

    bundle.install(tmp_path)
    sidecar = tmp_path / "test-bf16" / "example--bf16-model" / ".latentslate-model.toml"
    assert sidecar.read_text(encoding="utf-8") == (
        "# Generated by LatentSlate Engine for this complete canonical bundle.\n"
        'format = "diffusers"\n'
        'precision = "bf16"\n'
        'quantization = "native"\n'
    )

    manifest_path = tmp_path / "test-bf16" / ".latentslate-installed.json"
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy_manifest.pop("artifact")
    manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
    assert bundle.status(tmp_path) == BundleStatus.INSTALLED

    sidecar.unlink()
    assert bundle.status(tmp_path) == BundleStatus.MISSING


def test_configured_bundle_uses_the_same_model_id_as_the_runtime(tmp_path: Path):
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="example/custom-h3",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )

    assert configured_bundles(settings)["h3-basic"].repo_id == "example/custom-h3"
    configured = configured_bundles(settings)["h3-basic"]
    assert configured.revision is None
    assert configured.artifact_precision is None
    assert configured.artifact_quantization is None


def test_canonical_h3_and_ltx_bundles_pin_validated_upstream_revisions():
    assert BUNDLES["h3-basic"].revision == "9ac0dd7aabc2c651fcf0ace4c00b2bffd9c8c8a6"
    assert BUNDLES["ltx23-basic"].revision == "432e0d3c2d1769aaa4d295f9243f7062bf6b47ee"
