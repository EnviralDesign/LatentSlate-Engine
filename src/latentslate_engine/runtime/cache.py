from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Any


def stable_cache_key(namespace: str, kind: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{namespace}:{kind}:{digest}"


def _is_tensor(value: Any) -> bool:
    return all(
        hasattr(value, attribute)
        for attribute in ("detach", "to", "numel", "element_size")
    )


def freeze_for_cache(value: Any) -> Any:
    """Detach tensors onto CPU and recursively freeze container values.

    Torch is intentionally not imported at module import time so protocol-only CI
    and lightweight installations can import the Engine without model packages.
    """

    if _is_tensor(value):
        return value.detach().to(device="cpu").contiguous()
    if isinstance(value, tuple):
        return tuple(freeze_for_cache(item) for item in value)
    if isinstance(value, list):
        return [freeze_for_cache(item) for item in value]
    if isinstance(value, dict):
        return {key: freeze_for_cache(item) for key, item in value.items()}
    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001 - cache should not break inference
        return value


def materialize_cached(value: Any, *, device: Any | None = None) -> Any:
    """Return a request-local copy of a cached value on the requested device."""

    if _is_tensor(value):
        if device is None:
            return value.clone()
        moved = value.to(device=device, non_blocking=str(device) != "cpu")
        return moved.clone() if moved is value else moved
    if isinstance(value, tuple):
        return tuple(materialize_cached(item, device=device) for item in value)
    if isinstance(value, list):
        return [materialize_cached(item, device=device) for item in value]
    if isinstance(value, dict):
        return {
            key: materialize_cached(item, device=device)
            for key, item in value.items()
        }
    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001 - cache should not break inference
        return value


def estimate_cache_bytes(value: Any) -> int:
    if _is_tensor(value):
        try:
            return int(value.numel()) * int(value.element_size())
        except Exception:  # noqa: BLE001
            return 0
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, tuple | list):
        return sum(estimate_cache_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(
            estimate_cache_bytes(key) + estimate_cache_bytes(item)
            for key, item in value.items()
        )
    return 0


@dataclass(slots=True)
class _CacheEntry:
    value: Any
    size_bytes: int
    created_at: float
    last_access_at: float


class BoundedCache:
    def __init__(
        self,
        name: str,
        *,
        enabled: bool,
        max_bytes: int,
        max_entries: int,
    ) -> None:
        self.name = name
        self.enabled = enabled
        self.max_bytes = max(0, int(max_bytes))
        self.max_entries = max(0, int(max_entries))
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if not self.enabled:
                self._misses += 1
                return None
            entry = self._entries.pop(key, None)
            if entry is None:
                self._misses += 1
                return None
            entry.last_access_at = monotonic()
            self._entries[key] = entry
            self._hits += 1
            return entry.value

    def put(self, key: str, value: Any) -> bool:
        if not self.enabled or self.max_bytes <= 0 or self.max_entries <= 0:
            return False
        frozen = freeze_for_cache(value)
        size_bytes = estimate_cache_bytes(frozen)
        if size_bytes > self.max_bytes:
            return False

        now = monotonic()
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous.size_bytes
            self._entries[key] = _CacheEntry(
                value=frozen,
                size_bytes=size_bytes,
                created_at=now,
                last_access_at=now,
            )
            self._bytes += size_bytes
            self._evict_to_budget()
            return key in self._entries

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def status(self) -> dict[str, Any]:
        with self._lock:
            requests = self._hits + self._misses
            return {
                "name": self.name,
                "enabled": self.enabled,
                "entries": len(self._entries),
                "bytes": self._bytes,
                "max_bytes": self.max_bytes,
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": self._hits / requests if requests else None,
            }

    def _evict_to_budget(self) -> None:
        while self._entries and (
            len(self._entries) > self.max_entries or self._bytes > self.max_bytes
        ):
            _, entry = self._entries.popitem(last=False)
            self._bytes -= entry.size_bytes
            self._evictions += 1


class RuntimeCache:
    """Bounded CPU caches shared by all tools using one model runtime."""

    def __init__(
        self,
        namespace: str,
        *,
        enabled: bool,
        max_bytes: int,
        max_entries: int,
        prompt_fraction: float = 0.25,
    ) -> None:
        self.namespace = namespace
        prompt_bytes = int(max(0, max_bytes) * min(1.0, max(0.0, prompt_fraction)))
        media_bytes = max(0, int(max_bytes) - prompt_bytes)
        self.prompt = BoundedCache(
            "prompt",
            enabled=enabled,
            max_bytes=prompt_bytes,
            max_entries=max_entries,
        )
        self.media = BoundedCache(
            "media",
            enabled=enabled,
            max_bytes=media_bytes,
            max_entries=max_entries,
        )

    def key(self, kind: str, payload: dict[str, Any]) -> str:
        return stable_cache_key(self.namespace, kind, payload)

    def clear(self) -> None:
        self.prompt.clear()
        self.media.clear()

    def status(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "prompt": self.prompt.status(),
            "media": self.media.status(),
        }
