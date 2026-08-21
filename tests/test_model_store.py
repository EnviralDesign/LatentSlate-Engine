import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from latentslate_engine import model_store
from latentslate_engine.bundles import (
    BUNDLES,
    H3_FL2VA_ALLOW_PATTERNS,
    H3_FL2VA_CLOSURE_BYTES,
    H3_FL2VA_CLOSURE_FILES,
    BundleDefinition,
    configured_bundles,
)
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
from latentslate_engine.resources import discover_resources


def test_bundle_registry_does_not_eagerly_import_download_stack():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import latentslate_engine.bundles; "
                "assert 'huggingface_hub.file_download' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


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
    assert BUNDLES["h3-basic"].revision == "42ed227ee7df40d41602854ae760620d6eb651fe"
    assert BUNDLES["ltx23-basic"].revision == "432e0d3c2d1769aaa4d295f9243f7062bf6b47ee"
    assert BUNDLES["klein4b-basic"].revision == "e7b7dc27f91deacad38e78976d1f2b499d76a294"
    assert BUNDLES["klein9b-basic"].revision == "92196c8e11f7b6cf2b7493e037d8c5345c559216"
    assert BUNDLES["wan22-basic"].revision == "b8fff7315c768468a5333511427288870b2e9635"


def test_klein9_canonical_bundle_matches_reference_declaration_and_is_discoverable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Keep bundle installation and native-BF16 resource validation on one closure."""

    bundle = BUNDLES["klein9b-basic"]
    expected_patterns = (
        "model_index.json",
        "scheduler/**",
        "text_encoder/**",
        "tokenizer/**",
        "transformer/**",
        "vae/**",
    )
    declaration = tomllib.loads(
        (
            Path(__file__).parents[1]
            / "src"
            / "latentslate_engine"
            / "builtin_resource_declarations"
            / "klein9b-distilled-bf16.toml"
        ).read_text(encoding="utf-8")
    )["resource"]
    source = declaration["sources"][0]
    assert declaration["size_bytes"] == 34_722_772_650
    assert bundle.revision == source["revision"] == "92196c8e11f7b6cf2b7493e037d8c5345c559216"
    assert bundle.allow_patterns == tuple(source["allow_patterns"]) == expected_patterns

    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()

    def fake_snapshot_download(**kwargs):
        assert kwargs["repo_id"] == bundle.repo_id
        assert kwargs["revision"] == bundle.revision
        assert kwargs["allow_patterns"] == list(expected_patterns)
        local_dir = Path(kwargs["local_dir"])
        for relative, content in {
            "model_index.json": b"{}",
            "scheduler/scheduler_config.json": b"s",
            "text_encoder/config.json": b"t",
            "tokenizer/tokenizer.json": b"k",
            "transformer/diffusion_pytorch_model.safetensors": b"m",
            "vae/diffusion_pytorch_model.safetensors": b"v",
        }.items():
            target = local_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return str(local_dir)

    monkeypatch.setattr(
        "latentslate_engine.bundles.snapshot_download",
        fake_snapshot_download,
    )
    bundle.install(settings.model_root)
    assert bundle.status(settings.model_root) == BundleStatus.INSTALLED

    # Use the exact declaration source closure with a small synthesized inventory,
    # proving availability is based on the canonical artifact tree, not repo_id alone.
    fixture_patterns = ",\n  ".join(json.dumps(pattern) for pattern in source["allow_patterns"])
    (settings.resource_declarations_root / "klein9b-reference-fixture.toml").write_text(
        f'''
[resource]
id = "model:klein9b:black-forest-labs--flux.2-klein-9b"
kind = "model"
family = "klein9b"
name = "FLUX.2 Klein 9B Distilled (BF16)"
relative_path = "models/klein9b/black-forest-labs--FLUX.2-klein-9B"
format = "diffusers"
precision = "bf16"
quantization = "native"
size_bytes = 7

[[resource.sources]]
type = "huggingface"
repo_id = "black-forest-labs/FLUX.2-klein-9B"
revision = "{source["revision"]}"
requires_auth = true
allow_patterns = [
  {fixture_patterns}
]
''',
        encoding="utf-8",
    )

    resource = discover_resources(settings).resolve(
        "model:klein9b:black-forest-labs--flux.2-klein-9b"
    )
    assert resource.available
    assert resource.size_bytes == 7
    assert resource.sources[0].revision == bundle.revision
    assert tuple(resource.sources[0].allow_patterns) == bundle.allow_patterns


def test_ltx23_canonical_bundle_matches_reference_closure_and_is_discoverable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Keep the compatibility bundle byte-identical to the native LTX closure."""

    bundle = BUNDLES["ltx23-basic"]
    declaration = tomllib.loads(
        (
            Path(__file__).parents[1]
            / "src"
            / "latentslate_engine"
            / "builtin_resource_declarations"
            / "ltx23-distilled-bf16.toml"
        ).read_text(encoding="utf-8")
    )["resource"]
    source = declaration["sources"][0]
    expected_paths = set(source["allow_patterns"])
    ignored_paths = {"README.md", ".gitattributes"}

    assert declaration["size_bytes"] == 94_977_693_482
    assert len(expected_paths) == 50
    assert bundle.revision == source["revision"] == "432e0d3c2d1769aaa4d295f9243f7062bf6b47ee"
    assert bundle.allow_patterns == ()
    assert bundle.ignore_patterns == ("README.md", ".gitattributes")

    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    remote_paths = expected_paths | ignored_paths

    def fake_snapshot_download(**kwargs):
        assert kwargs["repo_id"] == bundle.repo_id
        assert kwargs["revision"] == bundle.revision
        assert kwargs["allow_patterns"] is None
        assert kwargs["ignore_patterns"] == list(bundle.ignore_patterns)
        selected = remote_paths - set(kwargs["ignore_patterns"])
        assert selected == expected_paths
        local_dir = Path(kwargs["local_dir"])
        for relative in sorted(selected):
            target = local_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative == "model_index.json":
                content = b"{}"
            elif relative.endswith(".index.json"):
                weights = sorted(
                    Path(candidate).name
                    for candidate in selected
                    if Path(candidate).parent == Path(relative).parent
                    and candidate.endswith(".safetensors")
                )
                content = json.dumps(
                    {"weight_map": {f"weight_{index}": weight for index, weight in enumerate(weights)}}
                ).encode("utf-8")
            else:
                content = b"x"
            target.write_bytes(content)
        return str(local_dir)

    monkeypatch.setattr(
        "latentslate_engine.bundles.snapshot_download",
        fake_snapshot_download,
    )
    bundle.install(settings.model_root)
    assert bundle.status(settings.model_root) == BundleStatus.INSTALLED

    installed_root = settings.model_root / "ltx23" / "diffusers--LTX-2.3-Distilled-Diffusers"
    fixture_size = sum(
        path.stat().st_size
        for path in installed_root.rglob("*")
        if path.is_file() and path.name != ".latentslate-model.toml"
    )
    fixture_patterns = ",\n  ".join(json.dumps(pattern) for pattern in source["allow_patterns"])
    (settings.resource_declarations_root / "ltx23-reference-fixture.toml").write_text(
        f'''
[resource]
id = "model:ltx23:diffusers--ltx-2.3-distilled-diffusers"
kind = "model"
family = "ltx23"
name = "LTX 2.3 Distilled BF16 Exact Native Closure"
relative_path = "models/ltx23/diffusers--LTX-2.3-Distilled-Diffusers"
format = "diffusers"
precision = "bf16"
quantization = "native"
size_bytes = {fixture_size}

[[resource.sources]]
type = "huggingface"
repo_id = "diffusers/LTX-2.3-Distilled-Diffusers"
revision = "{source["revision"]}"
allow_patterns = [
  {fixture_patterns}
]
''',
        encoding="utf-8",
    )

    resource = discover_resources(settings).resolve(
        "model:ltx23:diffusers--ltx-2.3-distilled-diffusers"
    )
    assert resource.available
    assert resource.size_bytes == fixture_size
    assert tuple(resource.sources[0].allow_patterns) == tuple(source["allow_patterns"])


def test_h3_bundle_remote_plan_is_the_exact_direct_fl2va_closure(monkeypatch, tmp_path: Path):
    bundle = BUNDLES["h3-basic"]
    remote_paths = {
        *H3_FL2VA_ALLOW_PATTERNS,
        "FL2VA/transformer/model-00001-of-00013.safetensors",
        "Ref2VA/transformer/model-00001-of-00013.safetensors",
        "transformer_ref/diffusion_pytorch_model-00001-of-00014.safetensors",
        "README.md",
    }
    requested: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        requested.update(kwargs)
        local_dir = Path(kwargs["local_dir"])
        for path in kwargs["allow_patterns"]:
            target = local_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
        return str(local_dir)

    monkeypatch.setattr("latentslate_engine.bundles.snapshot_download", fake_snapshot_download)
    bundle.install(tmp_path)

    assert bundle.allow_patterns == H3_FL2VA_ALLOW_PATTERNS
    assert len(bundle.allow_patterns) == 61
    assert H3_FL2VA_CLOSURE_BYTES == 144_051_143_011
    assert sum(size for _path, size in H3_FL2VA_CLOSURE_FILES) == H3_FL2VA_CLOSURE_BYTES
    selected_remote_paths = remote_paths & set(bundle.allow_patterns)
    assert selected_remote_paths == set(H3_FL2VA_ALLOW_PATTERNS)
    assert not any(
        path.startswith(("FL2VA/", "Ref2VA/", "transformer_ref/"))
        for path in bundle.allow_patterns
    )
    assert requested["revision"] == "42ed227ee7df40d41602854ae760620d6eb651fe"
    assert requested["allow_patterns"] == list(H3_FL2VA_ALLOW_PATTERNS)
    assert requested["ignore_patterns"] is None
