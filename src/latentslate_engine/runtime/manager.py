from __future__ import annotations

import gc
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol, TypeVar, runtime_checkable

from .process_memory import current_process_memory


class ManagedRuntime(Protocol):
    def status(self) -> Mapping[str, Any]:
        """Return bounded public lifecycle facts and model-specific extensions."""

    def unload(self) -> None:
        """Release heavyweight model state while keeping the wrapper reusable."""


@runtime_checkable
class CacheClearingRuntime(Protocol):
    def clear_cache(self) -> None:
        """Release bounded parent-side caches."""


@dataclass(frozen=True, slots=True)
class RuntimeLifecycleStatus:
    """Typed common lifecycle core plus bounded model-specific status fields."""

    loaded: bool
    worker_active: bool
    poisoned: bool
    extensions: Mapping[str, Any]


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
    wrappers forever. Explicit identity switches are transactional: the old identity
    must prove unload/cache purge before a new factory may run. Other cleanup surfaces
    remain best-effort so they do not mask an original generation/OOM error.
    """

    def __init__(self, *, max_wrappers: int = 8) -> None:
        self._lock = RLock()
        self._max_wrappers = max(1, int(max_wrappers))
        self._runtimes: dict[Hashable, ManagedRuntime] = {}
        self._quiesced_keys: set[Hashable] = set()
        self._active_key: Hashable | None = None
        self._cleanup_errors: list[str] = []

    def activate(self, key: Hashable, factory: Callable[[], RuntimeT]) -> RuntimeT:
        with self._lock:
            if self._active_key != key:
                previous_key = self._active_key
                if previous_key is not None:
                    previous = self._runtimes.get(previous_key)
                    if previous is not None:
                        self._strict_identity_switch_unload(
                            previous, key=previous_key
                        )
                        self._strict_identity_switch_clear_cache(
                            previous, key=previous_key
                        )
                        self._runtimes.pop(previous_key, None)
                        self._quiesced_keys.discard(previous_key)
                        gc.collect()
                self._active_key = None

            runtime = self._runtimes.pop(key, None)
            if runtime is None:
                runtime = factory()
            self._runtimes[key] = runtime
            self._quiesced_keys.discard(key)
            self._active_key = key
            self._prune_inactive()
            return runtime  # type: ignore[return-value]

    def _strict_identity_switch_unload(
        self,
        runtime: ManagedRuntime,
        *,
        key: Hashable,
    ) -> None:
        try:
            runtime.unload()
        except Exception as exc:
            self._record_cleanup_error("identity_switch_unload", key, exc)
            raise

    def _strict_identity_switch_clear_cache(
        self,
        runtime: ManagedRuntime,
        *,
        key: Hashable,
    ) -> None:
        if not isinstance(runtime, CacheClearingRuntime):
            return
        try:
            runtime.clear_cache()
        except Exception as exc:
            self._record_cleanup_error("identity_switch_clear_cache", key, exc)
            raise

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
            was_quiesced = key in self._quiesced_keys
            self._quiesced_keys.discard(key)
            if self._active_key == key:
                self._active_key = None
            if not was_quiesced:
                self._best_effort_unload(runtime, key=key)
            if clear_cache and not was_quiesced:
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
                try:
                    lifecycle = _lifecycle_status(runtime)
                    details = dict(lifecycle.extensions)
                except Exception as exc:  # noqa: BLE001 - status must remain inspectable
                    lifecycle = RuntimeLifecycleStatus(
                        loaded=False,
                        worker_active=False,
                        poisoned=True,
                        extensions={"status_error": f"{type(exc).__name__}: {exc}"},
                    )
                    details = dict(lifecycle.extensions)
                runtimes.append(
                    {
                        "key": _key_label(key),
                        "active": key == self._active_key,
                        "loaded": lifecycle.loaded,
                        "worker_active": lifecycle.worker_active,
                        "poisoned": lifecycle.poisoned,
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
            self._quiesced_keys.clear()

    def _prune_inactive(self) -> None:
        while len(self._runtimes) > self._max_wrappers:
            victim_key = next(
                (key for key in self._runtimes if key != self._active_key),
                None,
            )
            if victim_key is None:
                return
            victim = self._runtimes.pop(victim_key)
            if victim_key not in self._quiesced_keys:
                self._best_effort_unload(victim, key=victim_key)
                self._best_effort_clear_cache(victim, key=victim_key)
            self._quiesced_keys.discard(victim_key)

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
        if not isinstance(runtime, CacheClearingRuntime):
            return
        try:
            runtime.clear_cache()
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


def _lifecycle_status(runtime: ManagedRuntime) -> RuntimeLifecycleStatus:
    details = dict(runtime.status())
    loaded = details.get("loaded")
    if not isinstance(loaded, bool):
        raise TypeError("runtime status loaded fact must be boolean")
    worker_active = details.get("active_worker", False)
    poisoned = details.get("poisoned", False)
    if not isinstance(worker_active, bool) or not isinstance(poisoned, bool):
        raise TypeError("runtime active/poisoned lifecycle facts must be boolean")
    details.pop("loaded", None)
    details.pop("active_worker", None)
    details.pop("poisoned", None)
    return RuntimeLifecycleStatus(
        loaded=loaded,
        worker_active=worker_active,
        poisoned=poisoned,
        extensions=details,
    )


RUNTIME_MANAGER = RuntimeManager()
