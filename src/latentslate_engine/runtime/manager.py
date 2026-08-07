from __future__ import annotations

from collections.abc import Callable, Hashable
from threading import RLock
from typing import Protocol, TypeVar


class ManagedRuntime(Protocol):
    def unload(self) -> None:
        """Release heavyweight model state while keeping the lightweight wrapper reusable."""


RuntimeT = TypeVar("RuntimeT", bound=ManagedRuntime)


class RuntimeManager:
    """Keeps one heavyweight model runtime active at a time.

    LatentSlate Engine currently executes jobs serially on one worker/GPU. Retaining
    multiple large pipelines would only consume host RAM and VRAM, so activating a
    different runtime unloads the previous one. Lightweight wrapper objects remain
    cached and can reload their pipelines on the next request.
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

    def clear(self) -> None:
        with self._lock:
            for runtime in self._runtimes.values():
                runtime.unload()
            self._runtimes.clear()
            self._active_key = None


RUNTIME_MANAGER = RuntimeManager()
