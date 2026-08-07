from __future__ import annotations

from collections.abc import Callable, Hashable
from threading import RLock
from typing import Any, Protocol, TypeVar


class ManagedRuntime(Protocol):
    def unload(self) -> None:
        """Release heavyweight model state while keeping the wrapper reusable."""


RuntimeT = TypeVar("RuntimeT", bound=ManagedRuntime)


def _key_label(key: Hashable | None) -> str | None:
    if key is None:
        return None
    if isinstance(key, tuple):
        return ":".join(str(part) for part in key)
    return str(key)


class RuntimeManager:
    """Keeps one heavyweight model runtime active at a time.

    LatentSlate Engine executes jobs serially on one worker/GPU. Retaining several
    large pipelines would consume host RAM and VRAM, so activating a different
    runtime unloads the previous pipeline. Runtime wrappers and their bounded CPU
    conditioning caches remain reusable when the model becomes active again.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._runtimes: dict[Hashable, ManagedRuntime] = {}
        self._active_key: Hashable | None = None

    def activate(self, key: Hashable, factory: Callable[[], RuntimeT]) -> RuntimeT:
        with self._lock:
            if self._active_key != key:
                previous = self._runtimes.get(self._active_key)
                if previous is not None:
                    previous.unload()
                self._active_key = None

            runtime = self._runtimes.get(key)
            if runtime is None:
                runtime = factory()
                self._runtimes[key] = runtime
            self._active_key = key
            return runtime  # type: ignore[return-value]

    def status(self) -> dict[str, Any]:
        with self._lock:
            runtimes: list[dict[str, Any]] = []
            for key, runtime in self._runtimes.items():
                status_method = getattr(runtime, "status", None)
                details = status_method() if callable(status_method) else {}
                runtimes.append(
                    {
                        "key": _key_label(key),
                        "active": key == self._active_key,
                        "loaded": bool(getattr(runtime, "_pipeline", None) is not None),
                        **details,
                    }
                )
            runtimes.sort(key=lambda item: (not item["active"], item["key"] or ""))
            return {
                "active_runtime": _key_label(self._active_key),
                "runtimes": runtimes,
            }

    def clear_caches(self) -> dict[str, Any]:
        with self._lock:
            for runtime in self._runtimes.values():
                clear_method = getattr(runtime, "clear_cache", None)
                if callable(clear_method):
                    clear_method()
            return self.status()

    def clear(self) -> None:
        with self._lock:
            for runtime in self._runtimes.values():
                runtime.unload()
                clear_method = getattr(runtime, "clear_cache", None)
                if callable(clear_method):
                    clear_method()
            self._runtimes.clear()
            self._active_key = None


RUNTIME_MANAGER = RuntimeManager()
