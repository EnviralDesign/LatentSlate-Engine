from __future__ import annotations

import asyncio
import importlib.util
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="native request construction requires the locked runtime group",
)
def test_native_wan_cancellation_is_checked_before_runtime_load(
    monkeypatch: pytest.MonkeyPatch,
):
    from latentslate_engine.runtime.wan22_i2v_runtime import WanI2VRequest
    from latentslate_engine.runtime.wan22_native_managed import (
        ManagedNativeWanI2VRuntime,
    )

    recipe = SimpleNamespace(fingerprint="recipe:test")
    managed = ManagedNativeWanI2VRuntime(recipe)  # type: ignore[arg-type]
    monkeypatch.setattr(
        managed,
        "_load",
        lambda: (_ for _ in ()).throw(AssertionError("runtime should not load")),
    )
    request = WanI2VRequest(
        image=object(),
        prompt="move",
        num_frames=5,
        height=64,
        width=64,
        steps=4,
    )

    with pytest.raises(asyncio.CancelledError):
        managed.generate(request, device="cpu", cancelled=lambda: True)


def test_invalid_recipe_unloads_cached_native_runtime(
    monkeypatch: pytest.MonkeyPatch,
):
    from latentslate_engine.runtime.wan22_native_managed import (
        ManagedNativeWanI2VRuntime,
    )

    recipe = SimpleNamespace(fingerprint="recipe:stale")
    managed = ManagedNativeWanI2VRuntime(recipe)  # type: ignore[arg-type]
    stale = SimpleNamespace()
    managed._runtime = stale
    monkeypatch.setattr(
        "latentslate_engine.runtime.wan22_native_managed.revalidate_runtime_request",
        lambda _request: False,
    )
    monkeypatch.setattr(
        "latentslate_engine.runtime.wan22_native_managed.cleanup_accelerator_memory",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="changed after catalog validation"):
        managed.generate(object(), device="cpu")

    assert managed._runtime is None


def test_managed_load_passes_catalog_bound_plans(monkeypatch: pytest.MonkeyPatch, tmp_path):
    from latentslate_engine.runtime import wan22_i2v_runtime as runtime_module
    from latentslate_engine.runtime.wan22_native_managed import (
        ManagedNativeWanI2VRuntime,
    )

    roles = ("transformer_high_noise", "transformer_low_noise", "text_encoder", "vae")
    identities = {}
    components = {"pipeline_support": {"path": str(tmp_path / "support")}}
    adapter_plans = {}
    for role in roles:
        path = tmp_path / f"{role}.safetensors"
        identity = SimpleNamespace(path=path)
        identities[role] = identity
        components[role] = {"path": str(path)}
        adapter_plans[role] = SimpleNamespace(identity=identity)
    support_plan = SimpleNamespace(root=tmp_path / "support")
    recipe = SimpleNamespace(
        fingerprint="recipe:bound",
        components=components,
        identities=identities,
        support_plan=support_plan,
        adapter_plans=adapter_plans,
    )
    managed = ManagedNativeWanI2VRuntime(recipe)  # type: ignore[arg-type]
    monkeypatch.setattr(
        "latentslate_engine.runtime.wan22_native_managed.revalidate_runtime_request",
        lambda _request: True,
    )
    captured = {}

    def fake_load(paths, *, support_plan, adapter_plans):
        captured["paths"] = paths
        captured["support_plan"] = support_plan
        captured["adapter_plans"] = adapter_plans
        return SimpleNamespace()

    monkeypatch.setattr(
        runtime_module.NativeWanI2VRuntime,
        "load",
        staticmethod(fake_load),
    )

    loaded = managed._load()

    assert loaded is managed._runtime
    assert captured["support_plan"] is support_plan
    assert captured["adapter_plans"] is adapter_plans
