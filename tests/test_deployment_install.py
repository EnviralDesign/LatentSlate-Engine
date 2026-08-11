from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from latentslate_engine import __main__ as engine_cli
from latentslate_engine.acquisition import deployment_install as installer
from latentslate_engine.config import Settings
from latentslate_engine.recipes import (
    DeploymentLock,
    DeploymentLockResource,
    DeploymentPlan,
)
from latentslate_engine.resources import (
    ResourceDescriptor,
    ResourceFormat,
    ResourceInventory,
    ResourceKind,
    ResourceSource,
)
from latentslate_engine.tools import default_registry

PAYLOAD = b"exact model bytes"
SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


class _Registry:
    def __init__(self, resource: ResourceDescriptor, path: Path) -> None:
        self.resources = ResourceInventory(resources=[resource], paths={resource.id: path})


class _Response:
    def __init__(self, body: bytes, *, status: int = 200, headers: dict[str, str] | None = None):
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


def _settings(tmp_path: Path) -> Settings:
    value = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="unused",
        h3_device="cpu",
    )
    value.ensure_directories()
    return value


def _resource(source: ResourceSource) -> ResourceDescriptor:
    return ResourceDescriptor(
        id="lora:custom:exact",
        kind=ResourceKind.LORA,
        family="custom",
        name="Exact",
        relative_path="loras/custom/exact.safetensors",
        format=ResourceFormat.SAFETENSORS,
        size_bytes=len(PAYLOAD),
        sources=[source],
    )


def _plan(resource: ResourceDescriptor, *, provisionable: bool = True) -> DeploymentPlan:
    return DeploymentPlan(
        engine_version="test",
        profile_key="test",
        profile_name="test",
        recipes=[],
        resources=[],
        total_bytes=resource.size_bytes,
        incremental_bytes=resource.size_bytes,
        locally_runnable=False,
        remote_provisionable=provisionable,
    )


def _lock(resource: ResourceDescriptor) -> DeploymentLock:
    item = DeploymentLockResource(
        id=resource.id,
        family=resource.family,
        kind=resource.kind.value,
        format=resource.format.value,
        relative_path=resource.relative_path,
        size_bytes=resource.size_bytes,
        installed=False,
        sources=resource.sources,
    )
    return DeploymentLock(
        engine_version="test",
        generated_at="2026-01-01T00:00:00Z",
        profile_key="test",
        recipes=[],
        resources=[item, item],
        total_bytes=resource.size_bytes,
        incremental_bytes=resource.size_bytes,
        remote_provisionable=True,
    )


def _wire(monkeypatch: pytest.MonkeyPatch, resource: ResourceDescriptor, settings: Settings) -> _Registry:
    path = settings.home / resource.relative_path
    registry = _Registry(resource, path)
    monkeypatch.setattr(installer, "build_deployment_plan", lambda *_args: _plan(resource))
    monkeypatch.setattr(installer, "build_deployment_lock", lambda *_args: _lock(resource))
    monkeypatch.setattr(
        installer,
        "discover_resources",
        lambda _settings: ResourceInventory(resources=[resource], paths={resource.id: path}),
    )
    monkeypatch.setattr(
        "latentslate_engine.tools.default_registry",
        lambda *_args, **_kwargs: _Registry(resource, path),
    )
    return registry


def test_civitai_installs_only_the_declared_file_id_and_dedupes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    value = _settings(tmp_path)
    source = ResourceSource(type="civitai", model_version_id=9, file_id=2)
    resource = _resource(source)
    registry = _wire(monkeypatch, resource, value)
    requests: list[str] = []

    def fake_urlopen(request, timeout):
        requests.append(request.full_url)
        if request.full_url.endswith("/9"):
            return _Response(
                json.dumps(
                    {
                        "files": [
                            {"id": 1, "downloadUrl": "https://download.invalid/wrong"},
                            {
                                "id": 2,
                                "downloadUrl": "https://download.invalid/exact",
                                "hashes": {"SHA256": SHA256},
                            },
                        ]
                    }
                ).encode()
            )
        assert request.full_url == "https://download.invalid/exact"
        return _Response(PAYLOAD)

    monkeypatch.setattr(installer, "urlopen", fake_urlopen)
    result = installer.install_deployment_profile(value, registry, "test")

    assert result.installed_resource_ids == [resource.id]
    assert (value.home / resource.relative_path).read_bytes() == PAYLOAD
    assert requests == ["https://civitai.com/api/v1/model-versions/9", "https://download.invalid/exact"]


def test_missing_secret_and_unprovisionable_plan_never_call_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    value = _settings(tmp_path)
    resource = _resource(
        ResourceSource(
            type="civitai",
            url="https://civitai.com/api/download/models/1",
            sha256=SHA256,
            requires_auth=True,
        )
    )
    registry = _wire(monkeypatch, resource, value)
    monkeypatch.delenv("CIVITAI_TOKEN", raising=False)
    monkeypatch.setattr(installer, "urlopen", lambda *_args, **_kwargs: pytest.fail("network called"))
    with pytest.raises(installer.DeploymentInstallError, match="CIVITAI_TOKEN"):
        installer.install_deployment_profile(value, registry, "test")

    monkeypatch.setattr(installer, "build_deployment_plan", lambda *_args: _plan(resource, provisionable=False))
    with pytest.raises(installer.DeploymentInstallError, match="not remotely provisionable"):
        installer.install_deployment_profile(value, registry, "test")


def test_hash_mismatch_cleans_staging_and_refuses_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    value = _settings(tmp_path)
    resource = _resource(
        ResourceSource(type="civitai", url="https://download.invalid/exact", sha256=SHA256)
    )
    registry = _wire(monkeypatch, resource, value)
    monkeypatch.setattr(installer, "urlopen", lambda *_args, **_kwargs: _Response(b"wrong bytes"))

    with pytest.raises(installer.DeploymentInstallError, match="size|SHA256|completeness"):
        installer.install_deployment_profile(value, registry, "test")
    assert not (value.home / resource.relative_path).exists()
    assert not list((value.temp_dir / "deployment-installs").glob("*/payload"))


def test_installed_resource_skips_network_and_target_containment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    value = _settings(tmp_path)
    resource = _resource(
        ResourceSource(type="civitai", url="https://download.invalid/exact", sha256=SHA256)
    )
    target = value.home / resource.relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(PAYLOAD)
    registry = _wire(monkeypatch, resource, value)
    monkeypatch.setattr(installer, "urlopen", lambda *_args, **_kwargs: pytest.fail("network called"))
    assert installer.install_deployment_profile(value, registry, "test").skipped_resource_ids == [resource.id]

    escaped = resource.model_copy(update={"relative_path": "models/other/exact.safetensors"})
    with pytest.raises(installer.DeploymentInstallError, match="escapes"):
        installer._target_path(value, escaped)


def test_deployments_install_is_exposed_in_cli_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "deployments", "install", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        engine_cli.main()
    assert exc_info.value.code == 0
    assert "profile_key" in capsys.readouterr().out


def test_civitai_token_is_not_forwarded_to_delivery_redirect(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: list[tuple[str, str | None]] = []

    def fake_urlopen(request, timeout):
        seen.append((request.full_url, request.get_header("Authorization")))
        if request.full_url == "https://civitai.com/api/v1/model-versions/1":
            return _Response(b"", status=302, headers={"Location": "https://cdn.invalid/model"})
        return _Response(b"ok")

    monkeypatch.setattr(installer, "urlopen", fake_urlopen)
    response = installer._open_request(
        "https://civitai.com/api/v1/model-versions/1", "not-serialized-token"
    )
    response.close()
    assert seen == [
        ("https://civitai.com/api/v1/model-versions/1", "Bearer not-serialized-token"),
        ("https://cdn.invalid/model", None),
    ]


def test_huggingface_file_install_is_mocked_and_uses_pinned_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    value = _settings(tmp_path)
    resource = _resource(
        ResourceSource(
            type="huggingface",
            repo_id="example/model",
            filename="nested/exact.safetensors",
            sha256=SHA256,
        )
    )
    registry = _wire(monkeypatch, resource, value)
    calls: list[dict] = []

    def fake_hf_download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(PAYLOAD)
        return str(destination)

    monkeypatch.setattr(installer, "hf_hub_download", fake_hf_download)
    result = installer.install_deployment_profile(value, registry, "test")
    assert result.installed_resource_ids == [resource.id]
    assert calls[0]["repo_id"] == "example/model"


def test_huggingface_snapshot_install_is_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    value = _settings(tmp_path)
    resource = ResourceDescriptor(
        id="model:custom:snapshot",
        kind=ResourceKind.MODEL,
        family="custom",
        name="Snapshot",
        relative_path="models/custom/snapshot",
        format=ResourceFormat.DIFFUSERS,
        size_bytes=3,
        sources=[
            ResourceSource(
                type="huggingface",
                repo_id="example/snapshot",
                revision="a" * 40,
            )
        ],
    )
    registry = _wire(monkeypatch, resource, value)

    def fake_snapshot_download(**kwargs):
        destination = Path(kwargs["local_dir"])
        (destination / "model_index.json").write_text("{}", encoding="utf-8")
        (destination / "weights.safetensors").write_bytes(b"x")
        return str(destination)

    monkeypatch.setattr(installer, "snapshot_download", fake_snapshot_download)
    assert installer.install_deployment_profile(value, registry, "test").installed_resource_ids == [
        resource.id
    ]


def test_filtered_huggingface_snapshot_forwards_exact_patterns_and_never_serializes_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    value = _settings(tmp_path)
    token = "never-write-this-token"
    monkeypatch.setenv("TEST_HF_TOKEN", token)
    source = ResourceSource(
        type="huggingface",
        repo_id="example/snapshot",
        revision="a" * 40,
        allow_patterns=("model_index.json", "transformer/**", "vae/**"),
        ignore_patterns=("transformer_ref/**",),
        token_env="TEST_HF_TOKEN",
    )
    resource = ResourceDescriptor(
        id="model:custom:filtered-snapshot",
        kind=ResourceKind.MODEL,
        family="custom",
        name="Filtered snapshot",
        relative_path="models/custom/filtered-snapshot",
        format=ResourceFormat.DIFFUSERS,
        size_bytes=3,
        sources=[source],
    )
    registry = _wire(monkeypatch, resource, value)
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        (destination / "model_index.json").write_text("{}", encoding="utf-8")
        (destination / "weights.safetensors").write_bytes(b"x")
        return str(destination)

    monkeypatch.setattr(installer, "snapshot_download", fake_snapshot_download)
    assert installer.install_deployment_profile(value, registry, "test").installed_resource_ids == [
        resource.id
    ]
    assert calls == [
        {
            "repo_id": "example/snapshot",
            "revision": "a" * 40,
            "local_dir": str(installer._stage_directory(value, resource) / "payload"),
            "allow_patterns": ["model_index.json", "transformer/**", "vae/**"],
            "ignore_patterns": ["transformer_ref/**"],
            "token": token,
        }
    ]
    stage = installer._stage_directory(value, resource)
    installer._prepare_stage(stage, resource, source)
    assert token not in (stage / "manifest.json").read_text(encoding="utf-8")


def test_snapshot_patterns_are_lock_serialized_and_change_resume_identity(tmp_path: Path):
    value = _settings(tmp_path)
    source = ResourceSource(
        type="huggingface",
        repo_id="example/snapshot",
        revision="a" * 40,
        allow_patterns=("model_index.json", "transformer/**"),
        ignore_patterns=("transformer_ref/**",),
    )
    resource = ResourceDescriptor(
        id="model:custom:filtered-identity",
        kind=ResourceKind.MODEL,
        family="custom",
        name="Filtered identity",
        relative_path="models/custom/filtered-identity",
        format=ResourceFormat.DIFFUSERS,
        size_bytes=3,
        sources=[source],
    )

    serialized = _lock(resource).model_dump(mode="json")
    assert serialized["resources"][0]["sources"] == [
        {
            "type": "huggingface",
            "repo_id": "example/snapshot",
            "revision": "a" * 40,
            "filename": None,
            "url": None,
            "model_version_id": None,
            "file_id": None,
            "sha256": None,
            "allow_patterns": ["model_index.json", "transformer/**"],
            "ignore_patterns": ["transformer_ref/**"],
            "token_env": None,
            "requires_auth": False,
            "label": None,
        }
    ]

    stage = installer._stage_directory(value, resource)
    installer._prepare_stage(stage, resource, source)
    assert installer._stage_identity(resource, source) == installer._stage_identity(
        resource, source.model_copy(deep=True)
    )
    changed = source.model_copy(update={"allow_patterns": ("model_index.json", "vae/**")})
    with pytest.raises(installer.DeploymentInstallError, match="different source"):
        installer._preflight_stage(stage, resource, changed)


@pytest.mark.parametrize(
    "pattern",
    ["/weights/**", "../weights/**", "weights\\**", "weights//**", "weights/./**", " weights/**"],
)
def test_snapshot_patterns_reject_unsafe_relative_globs(pattern: str):
    with pytest.raises(ValueError, match="snapshot patterns"):
        ResourceSource(
            type="huggingface",
            repo_id="example/snapshot",
            revision="a" * 40,
            allow_patterns=(pattern,),
        )


def test_snapshot_patterns_are_restricted_to_immutable_huggingface_directories():
    with pytest.raises(ValueError, match="immutable revision"):
        ResourceSource(
            type="huggingface",
            repo_id="example/snapshot",
            revision="main",
            allow_patterns=("weights/**",),
        )
    with pytest.raises(ValueError, match="Hugging Face directory snapshots"):
        ResourceSource(type="civitai", model_version_id=1, file_id=1, allow_patterns=("weights/**",))
    with pytest.raises(ValueError, match="Hugging Face directory snapshots"):
        ResourceSource(type="manual", ignore_patterns=("weights/**",))
    with pytest.raises(ValueError, match="Hugging Face directory snapshots"):
        ResourceDescriptor(
            id="model:custom:filtered-file",
            kind=ResourceKind.MODEL,
            family="custom",
            name="Filtered file",
            relative_path="models/custom/filtered.safetensors",
            format=ResourceFormat.SAFETENSORS,
            size_bytes=3,
            sources=[
                ResourceSource(
                    type="huggingface",
                    repo_id="example/file",
                    revision="a" * 40,
                    filename="weights.safetensors",
                    allow_patterns=("weights/**",),
                )
            ],
        )


def test_real_catalog_profile_install_returns_post_install_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    value = _settings(tmp_path)
    resource_id = "model:wan22:test-snapshot"
    (value.resource_declarations_root / "test.toml").write_text(
        f'''[resource]
id = "{resource_id}"
kind = "model"
family = "wan22"
name = "Test snapshot"
relative_path = "models/wan22/test-snapshot"
format = "diffusers"
precision = "bf16"
quantization = "native"
size_bytes = 3

[[resource.sources]]
type = "huggingface"
repo_id = "example/test-snapshot"
revision = "{"a" * 40}"
''',
        encoding="utf-8",
    )
    (value.recipes_root / "test.toml").write_text(
        f'''[runnable_recipe]
key = "wan22.test-install"
name = "Test install"
family = "wan22"
base_tool = "wan22.text_to_video"

[runnable_recipe.model]
resource = "{resource_id}"

[runnable_recipe.optimizations]
attention = "native"
offload = "sequential"
quantization = "bf16"
cache = "prompt"
''',
        encoding="utf-8",
    )
    (value.deployment_profiles_root / "test.toml").write_text(
        '''[profile]
key = "test-install"
name = "Test install"
recipes = ["wan22.test-install"]
''',
        encoding="utf-8",
    )

    def fake_snapshot_download(**kwargs):
        destination = Path(kwargs["local_dir"])
        (destination / "model_index.json").write_text("{}", encoding="utf-8")
        (destination / "weights.safetensors").write_bytes(b"x")
        return str(destination)

    monkeypatch.setattr(installer, "snapshot_download", fake_snapshot_download)
    result = installer.install_deployment_profile(
        value, default_registry(value, emit_warnings=False), "test-install"
    )
    assert result.deployment_plan.locally_runnable
    assert result.deployment_plan.incremental_bytes == 0
    assert result.deployment_lock.incremental_bytes == 0
    assert result.deployment_lock.resources[0].installed


def test_multi_resource_local_obstruction_stops_all_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    value = _settings(tmp_path)
    first = _resource(ResourceSource(type="civitai", url="https://download.invalid/one", sha256=SHA256))
    second = first.model_copy(
        update={"id": "lora:custom:second", "relative_path": "loras/custom/second.safetensors"}
    )
    first_path = value.home / first.relative_path
    second_path = value.home / second.relative_path
    second_path.parent.mkdir(parents=True, exist_ok=True)
    second_path.write_bytes(b"incomplete")
    registry = type("Registry", (), {})()
    registry.resources = ResourceInventory(
        resources=[first, second], paths={first.id: first_path, second.id: second_path}
    )
    item_one = DeploymentLockResource(
        id=first.id, family=first.family, kind="lora", format="safetensors",
        relative_path=first.relative_path, size_bytes=first.size_bytes, installed=False, sources=first.sources,
    )
    item_two = item_one.model_copy(
        update={"id": second.id, "relative_path": second.relative_path, "sources": second.sources}
    )
    lock = DeploymentLock(
        engine_version="test", generated_at="2026-01-01T00:00:00Z", profile_key="test",
        recipes=[], resources=[item_one, item_two], total_bytes=first.size_bytes * 2,
        incremental_bytes=first.size_bytes * 2, remote_provisionable=True,
    )
    monkeypatch.setattr(installer, "build_deployment_plan", lambda *_args: _plan(first))
    monkeypatch.setattr(installer, "build_deployment_lock", lambda *_args: lock)
    monkeypatch.setattr(installer, "urlopen", lambda *_args, **_kwargs: pytest.fail("network called"))
    with pytest.raises(installer.DeploymentInstallError, match="incomplete"):
        installer.install_deployment_profile(value, registry, "test")


def test_civitai_part_recovery_and_bounded_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    destination = tmp_path / "payload"
    part = tmp_path / "payload.part"
    part.write_bytes(b"x")
    responses = [_Response(b"", status=416), _Response(PAYLOAD)]
    monkeypatch.setattr(installer, "urlopen", lambda *_args, **_kwargs: responses.pop(0))
    installer._download_http_file("https://cdn.invalid/model", destination, None, len(PAYLOAD), SHA256)
    assert destination.read_bytes() == PAYLOAD

    oversized = tmp_path / "oversized"
    monkeypatch.setattr(installer, "urlopen", lambda *_args, **_kwargs: _Response(PAYLOAD + b"!"))
    with pytest.raises(installer.DeploymentInstallError, match="exceeded|size"):
        installer._download_http_file("https://cdn.invalid/model", oversized, None, len(PAYLOAD), SHA256)


def test_no_clobber_file_publication_and_stage_symlink_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(PAYLOAD)
    target.write_bytes(b"existing")
    with pytest.raises(installer.DeploymentInstallError, match="appeared"):
        installer._publish_file_no_clobber(source, target, "test")
    assert target.read_bytes() == b"existing"

    value = _settings(tmp_path / "home")
    resource = _resource(ResourceSource(type="civitai", url="https://download.invalid/exact", sha256=SHA256))
    stage = installer._stage_directory(value, resource)
    stage.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    try:
        (stage / "payload").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    with pytest.raises(installer.DeploymentInstallError, match="link/reparse"):
        installer._preflight_stage(stage, resource, resource.sources[0])


def test_cli_install_dispatches_human_by_default_and_prints_json_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    value = _settings(tmp_path)
    resource = _resource(ResourceSource(type="civitai", url="https://download.invalid/exact", sha256=SHA256))
    registry = _Registry(resource, value.home / resource.relative_path)
    result = installer.DeploymentInstallResult(
        profile_key="test", deployment_plan=_plan(resource), deployment_lock=_lock(resource)
    )
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: value))
    monkeypatch.setattr("latentslate_engine.tools.default_registry", lambda *_args, **_kwargs: registry)
    monkeypatch.setattr(installer, "install_deployment_profile", lambda *_args: result)
    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "deployments", "install", "test"])
    engine_cli.main()
    assert "Deployment profile installation: test" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["latentslate-engine", "deployments", "install", "test", "--json"],
    )
    engine_cli.main()
    assert '"profile_key": "test"' in capsys.readouterr().out


def test_temp_capacity_creates_missing_root_and_fails_closed_on_probe_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    value = _settings(tmp_path)
    value.temp_dir.rmdir()
    installer._prepare_temp_capacity(value, 1)
    assert value.temp_dir.is_dir()
    monkeypatch.setattr(installer.shutil, "disk_usage", lambda _path: (_ for _ in ()).throw(OSError()))
    with pytest.raises(installer.DeploymentInstallError, match="unable to inspect"):
        installer._prepare_temp_capacity(value, 1)


def test_authenticated_civitai_exact_url_must_start_at_trusted_origin():
    with pytest.raises(ValueError, match="trusted civitai.com"):
        ResourceSource(
            type="civitai",
            url="https://cdn.invalid/model",
            sha256=SHA256,
            requires_auth=True,
        )
    with pytest.raises(ValueError, match="trusted civitai.com"):
        ResourceSource(
            type="civitai",
            url="https://mirror.invalid/model",
            sha256=SHA256,
            token_env="CIVITAI_TOKEN",
        )
