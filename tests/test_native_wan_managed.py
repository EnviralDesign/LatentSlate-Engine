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
