from __future__ import annotations

from types import SimpleNamespace

import pytest

from latentslate_engine.runtime.framework.residency.dynamic import (
    DynamicResidencyLease,
    DynamicResidencyPoisoned,
)
from latentslate_engine.runtime.framework.residency.leaf import (
    LeafResidencyDescriptor,
    LeafResidencyScheduler,
)


class _Backend:
    allocation_started = True
    required_virtual_bytes = 4096

    def __init__(self) -> None:
        self.groups = {}
        self.events: list[str] = []
        self.active: set[str] = set()
        self.maximum_active = 0
        self.seen: set[str] = set()
        self.hits = 0
        self.misses = 0
        self.closed = False

    def allocate_group(self, key, values):
        self.groups[key] = values
        self.events.append(f"allocate:{key}")

    def prioritize(self):
        self.events.append("prioritize")

    def _lease(self, key, phase):
        self.events.append(f"{phase}:{key}")
        self.active.add(key)
        self.maximum_active = max(self.maximum_active, len(self.active))
        if key in self.seen:
            self.hits += 1
        else:
            self.seen.add(key)
            self.misses += 1
        return DynamicResidencyLease(self.groups[key], SimpleNamespace(key=key))

    def acquire(self, key):
        return self._lease(key, "acquire")

    def prefetch(self, key):
        return self._lease(key, "prefetch")

    def wait(self, lease):
        self.events.append(f"wait:{lease.token.key}")

    def synchronize(self, lease):
        self.events.append(f"sync:{lease.token.key}")

    def release(self, lease):
        key = lease.token.key
        self.events.append(f"release:{key}")
        self.active.remove(key)

    def invalidate(self, *, reason):
        self.events.append(f"invalidate:{reason}")
        self.seen.clear()

    def invalidate_groups(self, keys, *, reason):
        self.events.append(f"invalidate-groups:{','.join(keys)}:{reason}")
        for key in keys:
            self.seen.discard(key)

    def prepare_stage(self, required_free_bytes):
        assert self.active <= {"tiny"}
        self.events.append(f"prepare-stage:{required_free_bytes}")

    def diagnostics(self):
        return {"backend": "fake"}

    def terminal_poison_reason(self):
        return None

    def close(self):
        assert not self.active
        self.closed = True
        self.events.append("close")


def _scheduler(backend: _Backend):
    bindings: dict[str, object] = {}

    def activate(descriptor, values):
        binding = object()
        bindings[descriptor.path] = binding
        backend.events.append(f"activate:{descriptor.path}:{values[0]}")
        return binding

    def restore(descriptor, binding):
        assert bindings.pop(descriptor.path) is binding
        backend.events.append(f"restore:{descriptor.path}")

    scheduler = LeafResidencyScheduler(
        backend,
        (
            LeafResidencyDescriptor("tiny", ("root",), (1,), 8, True),
            LeafResidencyDescriptor("block0.weight", ("block0",), (2,), 32_000),
            LeafResidencyDescriptor("block1.weight", ("block1",), (3,), 32_000),
        ),
        schedule_order=("root", "block0", "block1"),
        activate=activate,
        restore=restore,
    )
    return scheduler, bindings


def test_leaf_scheduler_prefetches_one_group_ahead_and_retains_warm_identity() -> None:
    backend = _Backend()
    scheduler, bindings = _scheduler(backend)
    backend.prioritize()
    scheduler.onload()

    for _ in range(2):
        scheduler.enter("root")
        scheduler.prefetch("block0")
        scheduler.enter("block0")
        scheduler.prefetch("block1")
        scheduler.leave("block0")
        scheduler.enter("block1")
        scheduler.leave("block1")
        scheduler.leave("root")
        scheduler.clear_stage()

    assert backend.misses == 3
    assert backend.hits == 2
    assert backend.maximum_active == 3  # tiny + current + one prefetched leaf
    assert list(bindings) == ["tiny"]
    assert scheduler.diagnostics() == {
        "leaf_allocation_count": 3,
        "force_resident_leaf_count": 1,
        "schedule_group_count": 3,
        "prefetch_groups": 4,
        "prefetch_leaves": 4,
        "deferred_waits": 4,
        "force_resident_waits": 1,
        "consumed_groups": 6,
        "active_groups": 0,
        "pending_prefetch": False,
    }
    scheduler.close()
    assert bindings == {}
    assert backend.closed is True


def test_leaf_scheduler_uses_one_group_release_without_per_leaf_sync() -> None:
    class _GroupedBackend(_Backend):
        def release_group(self, leases):
            keys = tuple(lease.token.key for lease in leases)
            self.events.append(f"release-group:{','.join(keys)}")
            for key in keys:
                self.active.remove(key)

    backend = _GroupedBackend()
    bindings: dict[str, object] = {}

    def activate(descriptor, _values):
        binding = object()
        bindings[descriptor.path] = binding
        backend.events.append(f"activate:{descriptor.path}")
        return binding

    def restore(descriptor, binding):
        assert bindings.pop(descriptor.path) is binding
        backend.events.append(f"restore:{descriptor.path}")

    scheduler = LeafResidencyScheduler(
        backend,
        (
            LeafResidencyDescriptor("block.a", ("block",), (1,), 8),
            LeafResidencyDescriptor("block.b", ("block",), (2,), 8),
        ),
        schedule_order=("block",),
        activate=activate,
        restore=restore,
    )
    scheduler.enter("block")
    scheduler.leave("block")

    assert "release-group:block.b,block.a" in backend.events
    assert not any(event.startswith("sync:") for event in backend.events)
    assert bindings == {}
    scheduler.close()


def test_leaf_scheduler_patch_invalidation_clears_temporary_state_before_epoch() -> None:
    backend = _Backend()
    scheduler, _bindings = _scheduler(backend)
    scheduler.onload()
    scheduler.prefetch("block0")
    scheduler.invalidate(reason="lora_to_base")
    assert backend.events.index("release:block0.weight") < backend.events.index(
        "invalidate:lora_to_base"
    )
    assert backend.active == {"tiny"}
    scheduler.close()


def test_leaf_scheduler_stage_boundary_clears_scheduling_before_backend_drain() -> None:
    backend = _Backend()
    scheduler, bindings = _scheduler(backend)
    scheduler.onload()
    scheduler.enter("root")
    scheduler.prefetch("block0")

    scheduler.prepare_stage(123_456)

    prepare_index = backend.events.index("prepare-stage:123456")
    assert backend.events.index("release:block0.weight") < prepare_index
    assert backend.active == {"tiny"}
    assert list(bindings) == ["tiny"]
    scheduler.close()


def test_leaf_scheduler_invalidation_releases_force_residents_before_backend_epoch() -> None:
    backend = _Backend()
    scheduler, bindings = _scheduler(backend)
    scheduler.onload()
    assert backend.active == {"tiny"}

    original_invalidate = backend.invalidate

    def strict_invalidate(*, reason):
        assert backend.active == set()
        original_invalidate(reason=reason)

    backend.invalidate = strict_invalidate
    scheduler.invalidate(reason="lora_to_base")

    assert backend.active == {"tiny"}
    assert list(bindings) == ["tiny"]
    assert backend.events.count("acquire:tiny") == 2
    scheduler.close()


def test_leaf_scheduler_selective_invalidation_preserves_other_signatures() -> None:
    backend = _Backend()
    scheduler, _bindings = _scheduler(backend)
    scheduler.onload()
    scheduler.enter("block0")
    scheduler.leave("block0")
    scheduler.invalidate(reason="lora_to_base", paths=("block0.weight",))

    assert "invalidate-groups:block0.weight:lora_to_base" in backend.events
    assert "tiny" in backend.seen
    assert "block0.weight" not in backend.seen
    scheduler.close()


@pytest.mark.parametrize("failure_site", ["prefetch", "wait", "synchronize", "release"])
def test_leaf_scheduler_poison_freezes_exact_graph_without_cleanup(failure_site: str) -> None:
    backend = _Backend()
    scheduler, bindings = _scheduler(backend)
    scheduler.onload()
    original = getattr(backend, failure_site)

    def poisoned(*args, **kwargs):
        del args, kwargs
        backend.events.append(f"poison:{failure_site}")
        raise DynamicResidencyPoisoned("failed_fill_quiescence_failed")

    setattr(backend, failure_site, poisoned)
    if failure_site == "prefetch":
        invoke = lambda: scheduler.prefetch("block0")
    elif failure_site == "wait":
        scheduler.prefetch("block0")
        invoke = lambda: scheduler.enter("block0")
    else:
        scheduler.prefetch("block0")
        scheduler.enter("block0")
        invoke = lambda: scheduler.leave("block0")

    with pytest.raises(DynamicResidencyPoisoned, match="failed_fill_quiescence_failed"):
        invoke()
    assert scheduler.terminal_poison_reason() == "failed_fill_quiescence_failed"
    frozen_events = list(backend.events)
    frozen_active = set(backend.active)
    frozen_bindings = dict(bindings)

    for operation in (scheduler.clear_stage, scheduler.close):
        with pytest.raises(DynamicResidencyPoisoned, match="failed_fill_quiescence_failed"):
            operation()
        assert backend.events == frozen_events
        assert backend.active == frozen_active
        assert bindings == frozen_bindings
    assert scheduler.diagnostics()["pending_prefetch"] in {True, False}
    assert backend.events == frozen_events
    setattr(backend, failure_site, original)
