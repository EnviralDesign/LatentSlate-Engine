import os
import weakref

import pytest

from latentslate_engine.runtime.framework.residency.dynamic import DynamicResidencyPoisoned
from latentslate_engine.runtime.manager import RuntimeManager


class CachePayload:
    pass


class FakeRuntime:
    def __init__(self):
        self.unload_count = 0
        self.clear_count = 0
        self.loaded = True

    def status(self):
        return {"loaded": self.loaded, "active_worker": False, "family": "fake"}

    def unload(self):
        self.unload_count += 1
        self.loaded = False

    def clear_cache(self):
        self.clear_count += 1


def test_runtime_manager_reuses_bounded_wrappers_and_clears_inactive_state_on_switch():
    manager = RuntimeManager()
    first = FakeRuntime()
    second = FakeRuntime()

    assert manager.activate(("h3", "model"), lambda: first) is first
    assert manager.activate(("h3", "model"), FakeRuntime) is first
    assert first.unload_count == 0

    assert manager.activate(("klein", "model"), lambda: second) is second
    assert first.unload_count == 1
    assert first.clear_count == 1
    assert second.unload_count == 0

    recovered = manager.activate(("h3", "model"), FakeRuntime)
    assert recovered is not first
    assert second.unload_count == 1
    assert second.clear_count == 1

    manager.clear()
    assert recovered.unload_count == 1


def test_runtime_manager_identity_switch_unload_failure_is_transactional() -> None:
    factory_calls = 0

    class PoisonedRuntime(FakeRuntime):
        def status(self):
            return {**super().status(), "poisoned": True}

        def unload(self):
            self.unload_count += 1
            raise DynamicResidencyPoisoned("device_quiescence_failed")

    old = PoisonedRuntime()
    manager = RuntimeManager()
    manager.activate("old", lambda: old)

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return FakeRuntime()

    with pytest.raises(DynamicResidencyPoisoned, match="device_quiescence_failed"):
        manager.activate("new", factory)

    assert factory_calls == 0
    assert manager.activate("old", factory) is old
    status = manager.status()
    assert status["active_runtime"] == "old"
    assert [item["key"] for item in status["runtimes"]] == ["old"]
    assert status["runtimes"][0]["poisoned"] is True
    assert any("identity_switch_unload old" in item for item in status["cleanup_errors"])


def test_runtime_manager_identity_switch_cache_failure_blocks_factory_and_can_retry() -> None:
    factory_calls = 0

    class CacheFailureRuntime(FakeRuntime):
        fail_cache = True

        def clear_cache(self):
            self.clear_count += 1
            if self.fail_cache:
                raise RuntimeError("cache purge failed")

    old = CacheFailureRuntime()
    manager = RuntimeManager()
    manager.activate("old", lambda: old)

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return FakeRuntime()

    with pytest.raises(RuntimeError, match="cache purge failed"):
        manager.activate("new", factory)
    assert factory_calls == 0
    assert manager.status()["active_runtime"] == "old"
    assert old.unload_count == old.clear_count == 1

    old.fail_cache = False
    new = manager.activate("new", factory)
    assert isinstance(new, FakeRuntime)
    assert factory_calls == 1
    assert old.unload_count == old.clear_count == 2
    status = manager.status()
    assert status["active_runtime"] == "new"
    assert [item["key"] for item in status["runtimes"]] == ["new"]


def test_runtime_manager_status_uses_public_typed_status_not_private_pipeline():
    manager = RuntimeManager()
    runtime = FakeRuntime()
    runtime._pipeline = None
    manager.activate(("fake", "model"), lambda: runtime)

    status = manager.status()["runtimes"][0]

    assert status["loaded"] is True
    assert status["worker_active"] is False
    assert status["poisoned"] is False
    assert status["family"] == "fake"


def test_mixed_runtime_switch_releases_inactive_parent_cache_references():
    manager = RuntimeManager(max_wrappers=8)
    first = FakeRuntime()
    payload = CachePayload()
    payload_ref = weakref.ref(payload)
    first.cache_payload = payload

    def clear_cache():
        first.clear_count += 1
        first.cache_payload = None

    first.clear_cache = clear_cache
    manager.activate(("family-a", "recipe-a"), lambda: first)
    del payload

    manager.activate(("family-b", "recipe-b"), FakeRuntime)

    assert payload_ref() is None
    assert first.clear_count == 1
    assert len(manager.status()["runtimes"]) == 1


def test_long_mixed_model_sequence_keeps_only_active_cache_payload_alive():
    manager = RuntimeManager(max_wrappers=8)
    payload_refs: list[weakref.ReferenceType[CachePayload]] = []

    class CachedRuntime(FakeRuntime):
        def __init__(self):
            super().__init__()
            payload = CachePayload()
            self.payload = payload
            payload_refs.append(weakref.ref(payload))

        def clear_cache(self):
            self.clear_count += 1
            self.payload = None

    for index in range(40):
        manager.activate((f"family-{index % 5}", f"recipe-{index}"), CachedRuntime)

    status = manager.status()
    assert len(status["runtimes"]) == 1
    assert sum(reference() is not None for reference in payload_refs) == 1
    assert status["host_process"]["pid"] > 0
    if os.name == "nt":
        assert status["host_process"]["private_bytes"] > 0
    else:
        assert status["host_process"]["private_bytes"] is None
