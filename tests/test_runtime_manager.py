from latentslate_engine.runtime.manager import RuntimeManager


class FakeRuntime:
    def __init__(self):
        self.unload_count = 0

    def unload(self):
        self.unload_count += 1


def test_runtime_manager_reuses_active_runtime_and_evicts_on_family_switch():
    manager = RuntimeManager()
    first = FakeRuntime()
    second = FakeRuntime()

    assert manager.activate(("h3", "model"), lambda: first) is first
    assert manager.activate(("h3", "model"), FakeRuntime) is first
    assert first.unload_count == 0

    assert manager.activate(("klein", "model"), lambda: second) is second
    assert first.unload_count == 1
    assert second.unload_count == 0

    assert manager.activate(("h3", "model"), FakeRuntime) is first
    assert second.unload_count == 1

    manager.clear()
    assert first.unload_count == 2
