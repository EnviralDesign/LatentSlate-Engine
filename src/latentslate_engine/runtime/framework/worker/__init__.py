"""Authenticated file-IPC primitives for isolated Engine workers."""

from .auth import canonical_json, hmac_sha256, result_hmac_sha256, sha256_fingerprint
from .child import (
    DisposableChildContext,
    DisposableChildPaths,
    DisposableWorkerHandler,
    parse_disposable_child_paths,
    run_disposable_child,
)
from .disposable import (
    DisposableWorkerExited,
    DisposableWorkerLimits,
    DisposableWorkerPaths,
    DisposableWorkerProgressTruncated,
    DisposableWorkerRunState,
    DisposableWorkerSupervisor,
    is_worker_cancellation,
)
from .files import WorkerJsonFileError, atomic_write_json, read_bounded_json
from .persistent import (
    PersistentWatchdogPolicy,
    PersistentWorkerExited,
    PersistentWorkerFailedStart,
    PersistentWorkerPaths,
    PersistentWorkerSession,
    PersistentWorkerStreamError,
    PersistentWorkerSupervisor,
    PersistentWorkerTimeout,
)
from .persistent_child import (
    PersistentChildContext,
    PersistentChildPaths,
    PersistentWorkerHandler,
    parse_persistent_child_paths,
    run_persistent_child,
)
from .progress import (
    JsonlCursor,
    WorkerJsonlFileError,
    append_bounded_jsonl,
    drain_bounded_jsonl,
)

__all__ = (
    "DisposableChildContext",
    "DisposableChildPaths",
    "DisposableWorkerExited",
    "DisposableWorkerHandler",
    "DisposableWorkerLimits",
    "DisposableWorkerPaths",
    "DisposableWorkerProgressTruncated",
    "DisposableWorkerRunState",
    "DisposableWorkerSupervisor",
    "JsonlCursor",
    "PersistentChildContext",
    "PersistentChildPaths",
    "PersistentWatchdogPolicy",
    "PersistentWorkerExited",
    "PersistentWorkerFailedStart",
    "PersistentWorkerHandler",
    "PersistentWorkerPaths",
    "PersistentWorkerSession",
    "PersistentWorkerStreamError",
    "PersistentWorkerSupervisor",
    "PersistentWorkerTimeout",
    "WorkerJsonFileError",
    "WorkerJsonlFileError",
    "append_bounded_jsonl",
    "atomic_write_json",
    "canonical_json",
    "drain_bounded_jsonl",
    "hmac_sha256",
    "is_worker_cancellation",
    "parse_disposable_child_paths",
    "parse_persistent_child_paths",
    "read_bounded_json",
    "result_hmac_sha256",
    "run_disposable_child",
    "run_persistent_child",
    "sha256_fingerprint",
)
