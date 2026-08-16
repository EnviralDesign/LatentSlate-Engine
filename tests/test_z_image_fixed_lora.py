from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from torch import nn

from latentslate_engine import z_image_turbo_recipe as recipe_contract
from latentslate_engine.artifacts import ArtifactIdentity
from latentslate_engine.runtime import z_image_stored_lora as fixed_lora


class _Slice:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = shape

    def get_shape(self) -> tuple[int, ...]:
        return self._shape


class _HeaderHandle:
    def __init__(self, keys: set[str]) -> None:
        self._keys = keys

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def keys(self):
        return self._keys

    def get_slice(self, key: str) -> _Slice:
        target = next(
            target
            for target in fixed_lora._EXPECTED_TARGETS
            if key in {target.down_key, target.up_key}
        )
        if key == target.down_key:
            return _Slice((16, target.in_features))
        return _Slice((target.out_features, 16))


def test_exact_planner_closes_all_240_rank16_targets(monkeypatch, tmp_path: Path):
    path = tmp_path / "fixed.safetensors"
    path.touch()
    identity = ArtifactIdentity(
        path,
        fixed_lora.Z_IMAGE_70S_HORROR_LORA_SIZE,
        1,
        fixed_lora.Z_IMAGE_70S_HORROR_LORA_HEADER_SHA256,
    )
    probe = SimpleNamespace(
        format="safetensors",
        identity=identity,
        schema_sha256=fixed_lora.Z_IMAGE_70S_HORROR_LORA_SCHEMA_SHA256,
        tensor_count=480,
        tensor_dtypes=("BF16",),
    )
    keys = set(fixed_lora._EXPECTED_KEYS)
    monkeypatch.setattr(fixed_lora, "probe_artifact", lambda _path: probe)
    monkeypatch.setattr(
        fixed_lora, "_sha256_file", lambda _path: fixed_lora.Z_IMAGE_70S_HORROR_LORA_SHA256
    )
    monkeypatch.setattr(fixed_lora, "safe_open", lambda *_args, **_kwargs: _HeaderHandle(keys))

    plan = fixed_lora.plan_z_image_70s_horror_lora(path)

    assert len(plan.targets) == 240
    assert len(plan.consumed_keys) == 480
    assert len({target.module_name for target in plan.targets}) == 180
    assert sum(target.module_name.endswith(".attention.qkv") for target in plan.targets) == 90
    assert {
        (target.row_start, target.row_count)
        for target in plan.targets
        if target.module_name == "layers.0.attention.qkv"
    } == {(0, 3840), (3840, 3840), (7680, 3840)}

    keys.add("unknown.weight")
    with pytest.raises(ValueError, match="key closure"):
        fixed_lora.plan_z_image_70s_horror_lora(path)


def test_sliced_bf16_additive_algebra_preserves_untargeted_rows():
    target = fixed_lora.ZImageFixedLoraTarget(
        "synthetic.q",
        "layers.0.attention.qkv",
        "down",
        "up",
        2,
        2,
        3,
        2,
    )
    branch = fixed_lora._ZImageFixedLoraBranch(
        target,
        torch.ones((16, 3), dtype=torch.bfloat16),
        torch.ones((2, 16), dtype=torch.bfloat16),
    )
    module = nn.Module()
    module._fixed_lora_branches = nn.ModuleDict({"fixed": branch})
    value = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.bfloat16)
    base = torch.arange(6, dtype=torch.bfloat16).reshape(1, 1, 6)

    actual = fixed_lora.apply_z_image_fixed_lora(module, value, base)

    expected = base.clone()
    expected[..., 2:4] += 16 * value.sum(dim=-1, keepdim=True)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[..., :2], base[..., :2])
    assert torch.equal(actual[..., 4:], base[..., 4:])
    assert branch.dispatch_count == 1


def test_fixed_lora_install_cancellation_rolls_back_every_branch(monkeypatch, tmp_path: Path):
    class FakeHandle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def keys(self):
            return {"a0", "b0", "a1", "b1", "a2", "b2"}

        def get_tensor(self, _key):
            return torch.ones((1, 1), dtype=torch.bfloat16)

    module = nn.Module()
    module._fixed_lora_branches = nn.ModuleDict()
    transformer = nn.Module()
    transformer.target = module
    targets = tuple(
        fixed_lora.ZImageFixedLoraTarget(
            f"target-{index}", "target", f"a{index}", f"b{index}", 0, 1, 1, 1
        )
        for index in range(3)
    )
    plan = fixed_lora.ZImageFixedLoraPlan(
        ArtifactIdentity(tmp_path / "x", 1, 1, "0" * 64),
        "1" * 64,
        "2" * 64,
        "resource",
        1.0,
        targets,
        frozenset({"a0", "b0", "a1", "b1", "a2", "b2"}),
    )

    def fake_add(target_module, name, target, _down, _up):
        branch = nn.Identity()
        branch.target_id = target.target_id
        branch.dispatch_count = 0
        target_module._fixed_lora_branches[name] = branch

    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    monkeypatch.setattr(fixed_lora, "revalidate_z_image_70s_horror_lora", lambda _plan: True)
    monkeypatch.setattr(fixed_lora, "safe_open", lambda *_args, **_kwargs: FakeHandle())
    monkeypatch.setattr(fixed_lora, "add_z_image_fixed_lora_branch", fake_add)
    lifecycle = fixed_lora.ZImageFixedLoraLifecycle()

    with pytest.raises(RuntimeError, match="canceled"):
        lifecycle.install(transformer, plan, cancelled=cancelled)

    assert len(module._fixed_lora_branches) == 0
    assert lifecycle.status()["loaded"] is False


def test_dispatch_proof_requires_every_exact_target():
    class FakeBranch(nn.Module):
        def __init__(self, target_id: str) -> None:
            super().__init__()
            self.target_id = target_id
            self.row_start = 0
            self.row_count = 1
            self.strength = 1.0
            self.down = nn.Parameter(torch.ones((16, 1), dtype=torch.bfloat16))
            self.up = nn.Parameter(torch.ones((1, 16), dtype=torch.bfloat16))
            self.dispatch_count = 8

    module = nn.Module()
    branches = {}
    target_map = {}
    targets = []
    for index in range(240):
        name = f"fixed_{index:03d}"
        branch = FakeBranch(f"target-{index}")
        branches[name] = branch
        target_map[branch.target_id] = ("target", name)
        targets.append(
            fixed_lora.ZImageFixedLoraTarget(
                branch.target_id, "target", "down", "up", 0, 1, 1, 1
            )
        )
    module._fixed_lora_branches = nn.ModuleDict(branches)
    transformer = nn.Module()
    transformer.target = module
    lifecycle = fixed_lora.ZImageFixedLoraLifecycle()
    lifecycle._targets = MappingProxyType(target_map)
    lifecycle._plan = SimpleNamespace(
        resource_id=fixed_lora.Z_IMAGE_70S_HORROR_LORA_RESOURCE_ID,
        strength=1.0,
        targets=tuple(targets),
    )
    before = {target_id: 0 for target_id in target_map}

    proof = lifecycle.verify_dispatch(transformer, before)

    assert proof["target_count"] == 240
    assert proof["total_dispatch_delta"] == 1920
    assert proof["complete"] is True
    assert proof["base_merged_or_dequantized"] is False
    module._fixed_lora_branches["fixed_010"].dispatch_count = 0
    with pytest.raises(RuntimeError, match="did not dispatch"):
        lifecycle.verify_dispatch(transformer, before)


def test_fixed_lora_component_round_trips_through_worker_manifest(monkeypatch, tmp_path: Path):
    roles = ("pipeline_support", "transformer", "text_encoder", "vae", "style_lora")
    (tmp_path / "pipeline_support").mkdir()
    for role in roles[1:]:
        (tmp_path / role).write_bytes(b"x")
    components = {
        role: {"path": str(tmp_path / role), "resource_id": f"resource:{role}"}
        for role in roles
    }
    plans = {"pipeline_support": SimpleNamespace(root=tmp_path / "pipeline_support")}
    identities = {}
    for index, role in enumerate(roles[1:], start=1):
        identity = ArtifactIdentity(tmp_path / role, index, index, f"{index:064x}")
        identities[role] = identity
        plans[role] = SimpleNamespace(identity=identity)
    request = recipe_contract.ZImageTurboRuntimeRequest(
        1,
        "base",
        recipe_contract.Z_IMAGE_OPERATION,
        dict(recipe_contract._SCHEDULE),
        components,
        identities,
        plans,
    )
    manifest = request.to_json_dict()
    planned_by_path = {
        str(Path(component["path"]).resolve(strict=False)): plans[role]
        for role, component in components.items()
    }
    monkeypatch.setattr(
        recipe_contract,
        "_plan_component",
        lambda _role, path: planned_by_path[str(path.resolve(strict=False))],
    )
    monkeypatch.setattr(recipe_contract, "revalidate_z_image_turbo_runtime_request", lambda _r: True)

    restored = recipe_contract.rehydrate_z_image_turbo_runtime_request(manifest)

    assert restored.fingerprint == request.fingerprint
    assert set(restored.components) == set(roles)
    changed = dict(manifest)
    changed["components"] = {
        role: dict(component) for role, component in manifest["components"].items()
    }
    changed["components"]["style_lora"]["resource_id"] = "tampered"
    with pytest.raises(ValueError, match="identity"):
        recipe_contract.rehydrate_z_image_turbo_runtime_request(changed)
