from __future__ import annotations

from collections.abc import Callable, Hashable
from threading import RLock
from typing import Any, Protocol, TypeVar

from .process_memory import current_process_memory


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
    """Own bounded model wrappers and keep one heavyweight runtime active.

    Wrappers retain bounded CPU conditioning caches, but the manager itself is also
    bounded so a long session that explores many variant load plans cannot accumulate
    wrappers forever. Teardown is best-effort: cleanup failures are observable through
    ``status()`` but never mask the original generation/OOM error.
    """

    def __init__(self, *, max_wrappers: int = 8) -> None:
        self._lock = RLock()
        self._max_wrappers = max(1, int(max_wrappers))
        self._runtimes: dict[Hashable, ManagedRuntime] = {}
        self._active_key: Hashable | None = None
        self._cleanup_errors: list[str] = []

    def activate(self, key: Hashable, factory: Callable[[], RuntimeT]) -> RuntimeT:
        with self._lock:
            if self._active_key != key:
                previous = self._runtimes.get(self._active_key)
                if previous is not None:
                    self._best_effort_unload(previous, key=self._active_key)
                self._active_key = None

            runtime = self._runtimes.pop(key, None)
            if runtime is None:
                runtime = factory()
            self._runtimes[key] = runtime
            self._active_key = key
            self._prune_inactive()
            return runtime  # type: ignore[return-value]

    def unload_runtime(self, runtime: ManagedRuntime) -> bool:
        """Unload a pipeline but retain its wrapper and conditioning caches."""

        with self._lock:
            key = self._key_for_runtime(runtime)
            if key is None:
                return False
            self._best_effort_unload(runtime, key=key)
            return True

    def evict_runtime(
        self,
        runtime: ManagedRuntime,
        *,
        clear_cache: bool = True,
    ) -> str | None:
        """Remove a runtime wrapper after a poisoned load or execution failure."""

        with self._lock:
            key = self._key_for_runtime(runtime)
            if key is None:
                return None
            self._runtimes.pop(key, None)
            if self._active_key == key:
                self._active_key = None
            self._best_effort_unload(runtime, key=key)
            if clear_cache:
                self._best_effort_clear_cache(runtime, key=key)
            return _key_label(key)

    def evict_active(self, *, clear_cache: bool = True) -> str | None:
        with self._lock:
            runtime = self._runtimes.get(self._active_key)
            if runtime is None:
                self._active_key = None
                return None
            return self.evict_runtime(runtime, clear_cache=clear_cache)

    def status(self) -> dict[str, Any]:
        with self._lock:
            runtimes: list[dict[str, Any]] = []
            for key, runtime in self._runtimes.items():
                details: dict[str, Any] = {}
                status_method = getattr(runtime, "status", None)
                if callable(status_method):
                    try:
                        details = status_method()
                    except Exception as exc:  # noqa: BLE001 - status must remain inspectable
                        details = {"status_error": f"{type(exc).__name__}: {exc}"}
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
                "max_wrappers": self._max_wrappers,
                "runtimes": runtimes,
                "cleanup_errors": list(self._cleanup_errors),
                "host_process": current_process_memory(),
            }

    def clear_caches(self) -> dict[str, Any]:
        with self._lock:
            for key, runtime in self._runtimes.items():
                self._best_effort_clear_cache(runtime, key=key)
            return self.status()

    def clear(self) -> None:
        with self._lock:
            runtimes = list(self._runtimes.items())
            self._runtimes.clear()
            self._active_key = None
            for key, runtime in runtimes:
                self._best_effort_unload(runtime, key=key)
                self._best_effort_clear_cache(runtime, key=key)

    def _prune_inactive(self) -> None:
        while len(self._runtimes) > self._max_wrappers:
            victim_key = next(
                (key for key in self._runtimes if key != self._active_key),
                None,
            )
            if victim_key is None:
                return
            victim = self._runtimes.pop(victim_key)
            self._best_effort_unload(victim, key=victim_key)
            self._best_effort_clear_cache(victim, key=victim_key)

    def _best_effort_unload(
        self,
        runtime: ManagedRuntime,
        *,
        key: Hashable | None,
    ) -> None:
        try:
            runtime.unload()
        except Exception as exc:  # noqa: BLE001 - teardown must not mask original failure
            self._record_cleanup_error("unload", key, exc)

    def _best_effort_clear_cache(
        self,
        runtime: ManagedRuntime,
        *,
        key: Hashable | None,
    ) -> None:
        clear_method = getattr(runtime, "clear_cache", None)
        if not callable(clear_method):
            return
        try:
            clear_method()
        except Exception as exc:  # noqa: BLE001 - teardown must not mask original failure
            self._record_cleanup_error("clear_cache", key, exc)

    def _record_cleanup_error(
        self,
        operation: str,
        key: Hashable | None,
        error: Exception,
    ) -> None:
        self._cleanup_errors.append(
            f"{operation} {_key_label(key) or '<unknown>'}: "
            f"{type(error).__name__}: {error}"
        )
        del self._cleanup_errors[:-16]

    def _key_for_runtime(self, runtime: ManagedRuntime) -> Hashable | None:
        return next(
            (key for key, candidate in self._runtimes.items() if candidate is runtime),
            None,
        )


RUNTIME_MANAGER = RuntimeManager()
