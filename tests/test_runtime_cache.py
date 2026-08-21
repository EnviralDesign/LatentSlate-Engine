from latentslate_engine.runtime.cache import BoundedCache, RuntimeCache, stable_cache_key
from latentslate_engine.runtime.manager import RuntimeManager


def test_stable_cache_key_is_order_independent():
    first = stable_cache_key("model", "prompt", {"prompt": "cat", "size": [512, 512]})
    second = stable_cache_key("model", "prompt", {"size": [512, 512], "prompt": "cat"})

    assert first == second
    assert first.startswith("model:prompt:")


def test_bounded_cache_tracks_hits_and_evicts_lru_entries():
    cache = BoundedCache("test", enabled=True, max_bytes=5, max_entries=4)

    assert cache.put("first", "abc") is True
    assert cache.get("first") == "abc"
    assert cache.put("second", "def") is True

    assert cache.get("first") is None
    assert cache.get("second") == "def"
    status = cache.status()
    assert status["entries"] == 1
    assert status["bytes"] == 3
    assert status["hits"] == 2
    assert status["misses"] == 1
    assert status["evictions"] == 1


def test_disabled_cache_is_a_noop():
    cache = BoundedCache("test", enabled=False, max_bytes=100, max_entries=4)

    assert cache.put("value", "cached") is False
    assert cache.get("value") is None
    assert cache.status()["entries"] == 0


def test_runtime_cache_splits_prompt_and_media_budgets():
    cache = RuntimeCache(
        "model",
        enabled=True,
        max_bytes=100,
        max_entries=8,
        prompt_fraction=0.25,
    )

    status = cache.status()
    assert status["prompt"]["max_bytes"] == 25
    assert status["media"]["max_bytes"] == 75


def test_runtime_manager_reports_and_clears_cache_without_unloading_pipeline():
    class FakeRuntime:
        def __init__(self):
            self._pipeline = object()
            self.cache_cleared = False
            self.unloaded = False

        def status(self):
            return {
                "loaded": not self.unloaded,
                "active_worker": False,
                "family": "fake",
                "cache": {"entries": 1},
            }

        def clear_cache(self):
            self.cache_cleared = True

        def unload(self):
            self.unloaded = True
            self._pipeline = None

    manager = RuntimeManager()
    runtime = manager.activate(("fake", "model"), FakeRuntime)

    status = manager.status()
    assert status["active_runtime"] == "fake:model"
    assert status["runtimes"][0]["active"] is True
    assert status["runtimes"][0]["loaded"] is True

    cleared = manager.clear_caches()
    assert runtime.cache_cleared is True
    assert runtime.unloaded is False
    assert cleared["runtimes"][0]["loaded"] is True
