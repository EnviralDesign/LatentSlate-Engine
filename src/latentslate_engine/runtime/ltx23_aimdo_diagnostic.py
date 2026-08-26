"""Operator-only real-hardware checks for LTX 2.3 Gemma AIMDO residency.

This module deliberately bypasses recipes and generation. It materializes only
the installed text encoder, exercises selected residency groups, and compares
stored physical tensors without dequantizing or retaining large CPU copies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NoReturn

import torch

from .framework.residency import DynamicResidencyPoisoned
from .framework.residency.aimdo import AimdoFileBackedValue, AimdoFileSpan
from .ltx23_kitchen_text import (
    LTX23GemmaMixedTextStage,
    load_ltx23_gemma_mixed_text_encoder,
    plan_ltx23_gemma_mixed_text_encoder,
)

DiagnosticProgress = Callable[[str], None]
DiagnosticHardExit = Callable[[int], NoReturn]
DIAGNOSTIC_POISON_EXIT_CODE = 86
_poisoned_diagnostic_retained: tuple[object, ...] | None = None


def representative_storage_groups(stage: LTX23GemmaMixedTextStage) -> tuple[tuple[str, Any], ...]:
    """Return root plus unique first/middle/last layer storage groups."""

    layers = stage._layer_storage
    indices = tuple(dict.fromkeys((0, len(layers) // 2, len(layers) - 1)))
    return (("root", stage._root_storage),) + tuple(
        (f"layer_{index}", layers[index]) for index in indices
    )


def physical_tensors(value: Any) -> tuple[tuple[str, torch.Tensor], ...]:
    """Flatten one ordinary or Kitchen tensor into its exact physical fields."""

    flatten = getattr(value, "__tensor_flatten__", None)
    if callable(flatten):
        names, _context = flatten()
        if not isinstance(names, (list, tuple)) or not all(isinstance(name, str) for name in names):
            raise TypeError("diagnostic tensor flatten contract is invalid")
        fields = tuple((name, getattr(value, name)) for name in names)
    elif isinstance(value, torch.Tensor):
        fields = (("tensor", value),)
    else:
        raise TypeError(f"diagnostic cannot flatten {type(value).__name__}")
    if not all(isinstance(tensor, torch.Tensor) for _name, tensor in fields):
        raise TypeError("diagnostic tensor flatten contract returned a non-tensor")
    return fields


def compare_resident_group(
    authoritative: tuple[Any, ...],
    resident: tuple[Any, ...],
    *,
    chunk_bytes: int = 16 * 1024**2,
    file_sources: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Compare exact physical bytes and logical aliases with bounded CPU memory."""

    if chunk_bytes <= 0:
        raise ValueError("diagnostic comparison chunk must be positive")
    if len(authoritative) != len(resident):
        raise RuntimeError("diagnostic resident logical-value count changed")
    alias_pairs = 0
    for left in range(len(authoritative)):
        for right in range(left):
            expected_alias = authoritative[left] is authoritative[right]
            actual_alias = resident[left] is resident[right]
            if expected_alias != actual_alias:
                raise RuntimeError("diagnostic logical alias identity changed")
            alias_pairs += int(expected_alias)

    physical_count = 0
    compared_bytes = 0
    file_physical_count = 0
    file_compared_bytes = 0
    for logical_index, (source_value, resident_value) in enumerate(
        zip(authoritative, resident, strict=True)
    ):
        source_template = (
            source_value.template
            if isinstance(source_value, AimdoFileBackedValue)
            else source_value
        )
        if type(source_template) is not type(resident_value):
            raise RuntimeError(
                f"diagnostic logical type changed at value {logical_index}: "
                f"{type(source_template).__name__} != {type(resident_value).__name__}"
            )
        source_fields = physical_tensors(source_template)
        resident_fields = physical_tensors(resident_value)
        if tuple(name for name, _tensor in source_fields) != tuple(
            name for name, _tensor in resident_fields
        ):
            raise RuntimeError("diagnostic quantized sidecar names changed")
        spans: tuple[AimdoFileSpan | None, ...] = (
            tuple(source_value.spans)
            if isinstance(source_value, AimdoFileBackedValue)
            else (None,) * len(source_fields)
        )
        if len(spans) != len(source_fields):
            raise RuntimeError("diagnostic file descriptor field count changed")
        for field_index, ((name, source), (_resident_name, candidate)) in enumerate(
            zip(source_fields, resident_fields, strict=True)
        ):
            span = spans[field_index]
            expected_dtype = source.dtype if span is None else span.dtype
            expected_shape = source.shape if span is None else torch.Size(span.shape)
            if expected_dtype != candidate.dtype or expected_shape != candidate.shape:
                raise RuntimeError(
                    f"diagnostic physical contract changed at value {logical_index}.{name}"
                )
            candidate_bytes = candidate.detach().reshape(-1).view(torch.uint8)
            source_size = source.numel() * source.element_size() if span is None else span.size
            if source_size != candidate_bytes.numel():
                raise RuntimeError("diagnostic physical byte length changed")
            source_bytes = source.detach().reshape(-1).view(torch.uint8) if span is None else None
            for offset in range(0, source_size, chunk_bytes):
                end = min(offset + chunk_bytes, source_size)
                if span is None:
                    expected = source_bytes[offset:end]
                else:
                    if file_sources is None or span.source_id not in file_sources:
                        raise RuntimeError("diagnostic file source is not bound")
                    payload = bytearray(end - offset)
                    handle = file_sources[span.source_id]
                    handle.seek(span.offset + offset)
                    if handle.readinto(payload) != len(payload):
                        raise RuntimeError("diagnostic authoritative file span is truncated")
                    expected = torch.frombuffer(payload, dtype=torch.uint8)
                observed = candidate_bytes[offset:end].to("cpu")
                if not torch.equal(expected, observed):
                    raise RuntimeError(
                        "diagnostic resident bytes differ at "
                        f"value {logical_index}.{name} offset {offset}"
                    )
            physical_count += 1
            compared_bytes += source_size
            if span is not None:
                file_physical_count += 1
                file_compared_bytes += source_size
    return {
        "logical_values": len(authoritative),
        "physical_tensors": physical_count,
        "compared_bytes": compared_bytes,
        "alias_pairs": alias_pairs,
        "file_physical_tensors": file_physical_count,
        "file_compared_bytes": file_compared_bytes,
        "cpu_physical_tensors": physical_count - file_physical_count,
        "cpu_compared_bytes": compared_bytes - file_compared_bytes,
    }


def _counter_delta(before: dict[str, Any], after: dict[str, Any], key: str) -> int:
    return int(after[key]) - int(before[key])


def _physical_copy_contract(
    authoritative: tuple[Any, ...],
) -> tuple[int, int, int, int]:
    physical: dict[tuple[object, ...], tuple[int, bool]] = {}
    for value in authoritative:
        template = value.template if isinstance(value, AimdoFileBackedValue) else value
        fields = physical_tensors(template)
        spans: tuple[AimdoFileSpan | None, ...] = (
            tuple(value.spans) if isinstance(value, AimdoFileBackedValue) else (None,) * len(fields)
        )
        if len(spans) != len(fields):
            raise RuntimeError("diagnostic file descriptor field count changed")
        for (_name, tensor), span in zip(fields, spans, strict=True):
            if span is None:
                size = int(tensor.numel() * tensor.element_size())
                key = ("cpu", int(tensor.data_ptr()), size)
                is_file = False
            else:
                size = span.size
                key = ("file", span.source_id, span.offset, span.size)
                is_file = True
            physical.setdefault(key, (size, is_file))
    file_values = tuple(size for size, is_file in physical.values() if is_file)
    return (
        len(physical),
        sum(size for size, _ in physical.values()),
        len(file_values),
        sum(file_values),
    )


def _require_acquire_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    label: str,
    hit: bool,
    physical_tensors: int,
    physical_bytes: int,
    staged_bytes: int,
    file_physical_tensors: int = 0,
    file_physical_bytes: int = 0,
    source_hit: bool = False,
) -> None:
    gathered = before.get("copy_strategy") == "gathered_host_buffer"
    expected = {
        "faults": 1,
        "signature_hits": int(hit),
        "signature_misses": int(not hit),
        "fault_none_temporaries": 0,
        "transfer_events": 0 if hit else 1 if gathered else physical_tensors,
        "transfer_waits": 0 if hit else 1 if gathered else physical_tensors,
        "unpin_calls": 1,
        "gathered_misses": int(not hit and gathered),
        "per_physical_misses": int(not hit and not gathered),
        "packed_source_bytes": 0 if hit or not gathered else physical_bytes,
        "gathered_h2d_bytes": 0 if hit or not gathered else staged_bytes,
        # This diagnostic releases each lease before the next acquire, so its
        # compute barrier also proves the shared host buffer reusable.
        "host_buffer_reuse_barriers": 0,
        "base_file_read_calls": 0 if hit else file_physical_tensors,
        "base_file_read_bytes": 0 if hit else file_physical_bytes,
        "host_source_pool_hits": int(not hit and gathered and source_hit),
        "host_source_pool_misses": int(not hit and gathered and not source_hit),
    }
    if any(_counter_delta(before, after, key) != value for key, value in expected.items()):
        raise RuntimeError(f"diagnostic {label} acquire counters are not exact")
    copied = _counter_delta(before, after, "pinned_copy_bytes") + _counter_delta(
        before, after, "pageable_copy_bytes"
    )
    expected_copied = 0 if hit else staged_bytes if gathered else physical_bytes
    if copied != expected_copied:
        raise RuntimeError(f"diagnostic {label} copy-byte counters are not exact")


def exercise_group(
    backend: Any,
    *,
    label: str,
    storage: Any,
    device: torch.device,
    authoritative: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """Prove miss, signature hit reuse, forced miss, and exact group bytes."""

    authoritative = (
        tuple(slot.cpu_value for slot in storage.slots) if authoritative is None else authoritative
    )
    physical_count, physical_bytes, file_count, file_bytes = _physical_copy_contract(authoritative)
    file_sources = getattr(backend, "_file_sources", {})
    before = backend.diagnostics()
    first = backend.acquire(id(storage))
    backend.synchronize(first)
    first_proof = compare_resident_group(authoritative, first.values, file_sources=file_sources)
    first_value_ids = tuple(id(value) for value in first.values)
    backend.release(first)
    del first
    after_first = backend.diagnostics()
    _require_acquire_delta(
        before,
        after_first,
        label=f"{label} initial miss",
        hit=False,
        physical_tensors=physical_count,
        physical_bytes=physical_bytes,
        staged_bytes=backend._groups[id(storage)].staged_bytes,
        file_physical_tensors=file_count,
        file_physical_bytes=file_bytes,
        source_hit=False,
    )

    second = backend.acquire(id(storage))
    backend.synchronize(second)
    hit_proof = compare_resident_group(authoritative, second.values, file_sources=file_sources)
    if first_value_ids != tuple(id(value) for value in second.values):
        raise RuntimeError(f"diagnostic {label} signature hit rebuilt logical views")
    backend.release(second)
    del second
    after_hit = backend.diagnostics()
    _require_acquire_delta(
        after_first,
        after_hit,
        label=f"{label} signature hit",
        hit=True,
        physical_tensors=physical_count,
        physical_bytes=physical_bytes,
        staged_bytes=backend._groups[id(storage)].staged_bytes,
        file_physical_tensors=file_count,
        file_physical_bytes=file_bytes,
        source_hit=False,
    )

    backend.invalidate(reason="diagnostic_force_miss")
    forced = backend.acquire(id(storage))
    backend.synchronize(forced)
    forced_proof = compare_resident_group(authoritative, forced.values, file_sources=file_sources)
    backend.release(forced)
    del forced
    after_forced = backend.diagnostics()
    _require_acquire_delta(
        after_hit,
        after_forced,
        label=f"{label} forced miss",
        hit=False,
        physical_tensors=physical_count,
        physical_bytes=physical_bytes,
        staged_bytes=backend._groups[id(storage)].staged_bytes,
        file_physical_tensors=0 if file_count == physical_count else file_count,
        file_physical_bytes=0 if file_count == physical_count else file_bytes,
        source_hit=file_count == physical_count,
    )
    torch.cuda.synchronize(device)
    return {
        "label": label,
        "initial_miss": first_proof,
        "signature_hit": hit_proof,
        "forced_miss": forced_proof,
        "source_contract": {
            "physical_tensors": physical_count,
            "physical_bytes": physical_bytes,
            "file_physical_tensors": file_count,
            "file_physical_bytes": file_bytes,
            "cpu_physical_tensors": physical_count - file_count,
            "cpu_physical_bytes": physical_bytes - file_bytes,
        },
    }


def _diagnostic_poison_reason(stage: Any, error: BaseException) -> str | None:
    if isinstance(error, DynamicResidencyPoisoned):
        return error.reason
    reason = stage.terminal_poison_reason()
    return reason if isinstance(reason, str) and reason else None


def _close_diagnostic_stage(
    stage: Any,
    *,
    report: DiagnosticProgress,
    primary: BaseException | None,
    hard_exit: DiagnosticHardExit,
) -> None:
    """Always offload, and never finalize Python after poisoned quiescence."""

    report_error: BaseException | None = None
    try:
        report("closing AIMDO text stage")
    except BaseException as exc:  # noqa: BLE001 - cleanup remains mandatory
        report_error = exc
    close_error: BaseException | None = None
    try:
        stage.offload()
    except BaseException as exc:  # noqa: BLE001 - classify poison below
        close_error = exc

    poison_error = close_error or primary
    poison_reason = None if poison_error is None else _diagnostic_poison_reason(stage, poison_error)
    if poison_reason is not None:
        retained_primary = primary or close_error or report_error
        if primary is not None and close_error is not None and close_error is not primary:
            primary.add_note(
                f"AIMDO diagnostic close also failed: {type(close_error).__name__}: {close_error}"
            )
        global _poisoned_diagnostic_retained
        _poisoned_diagnostic_retained = (
            stage,
            retained_primary,
            close_error,
            report_error,
            poison_reason,
        )
        hard_exit(DIAGNOSTIC_POISON_EXIT_CODE)
        raise RuntimeError("diagnostic hard-exit callback returned") from retained_primary

    effective = primary
    for secondary, label in (
        (report_error, "progress reporting"),
        (close_error, "AIMDO close"),
    ):
        if secondary is None:
            continue
        if effective is None:
            effective = secondary
        elif secondary is not effective:
            effective.add_note(
                f"AIMDO diagnostic {label} also failed: {type(secondary).__name__}: {secondary}"
            )
    if effective is not None:
        raise effective


def _require_file_backed_close_proof(before: Mapping[str, Any], closed: Mapping[str, Any]) -> None:
    """Authenticate source-backed ownership before and after synchronized close."""

    before_valid = bool(
        before.get("copy_strategy") == "gathered_host_buffer"
        and before.get("copy_fallback_reason") is None
        and before.get("base_file_backed") is True
        and before.get("base_file_source_live") is True
        and before.get("base_file_handle_live") is True
        and before.get("base_file_handle_opened") == 1
        and before.get("base_file_handle_closed") == 0
        and before.get("base_file_fallback_reason") is None
        and isinstance(before.get("base_file_read_calls"), int)
        and not isinstance(before.get("base_file_read_calls"), bool)
        and before["base_file_read_calls"] > 0
        and isinstance(before.get("base_file_read_bytes"), int)
        and not isinstance(before.get("base_file_read_bytes"), bool)
        and before["base_file_read_bytes"] > 0
    )
    closed_valid = bool(
        closed.get("copy_strategy") == "gathered_host_buffer"
        and closed.get("copy_fallback_reason") is None
        and closed.get("base_file_backed") is True
        and closed.get("base_file_source_live") is False
        and closed.get("base_file_handle_live") is False
        and closed.get("base_file_handle_opened") == 1
        and closed.get("base_file_handle_closed") == 1
        and closed.get("base_file_fallback_reason") is None
        and closed.get("base_file_read_calls") == before.get("base_file_read_calls")
        and closed.get("base_file_read_bytes") == before.get("base_file_read_bytes")
    )
    if not before_valid or not closed_valid:
        raise RuntimeError("diagnostic source-backed close proof is not exact")


def run_ltx23_aimdo_residency_diagnostic(
    text_artifact: Path,
    text_support_root: Path,
    *,
    device: str = "cuda",
    progress: DiagnosticProgress | None = None,
    hard_exit: DiagnosticHardExit,
) -> dict[str, Any]:
    """Materialize real Gemma in a hard-exit-contained dedicated process."""

    report = progress or (lambda _message: None)
    report("planning pinned LTX 2.3 Gemma artifact")
    plan = plan_ltx23_gemma_mixed_text_encoder(Path(text_artifact))
    report("materializing text-only Gemma CPU model")
    model = load_ltx23_gemma_mixed_text_encoder(plan, Path(text_support_root))
    stage = LTX23GemmaMixedTextStage(model, device, dynamic_policy="required")
    results: list[dict[str, Any]] = []
    primary: BaseException | None = None
    before_close: dict[str, Any] | None = None
    try:
        groups = representative_storage_groups(stage)
        report("initializing AIMDO VBAR groups")
        stage._initialize_backend()
        backend = stage._dynamic_backend
        if backend is None:
            raise RuntimeError("diagnostic required AIMDO but no backend was retained")
        for label, storage in groups:
            report(f"exercising {label}")
            results.append(
                exercise_group(
                    backend,
                    label=label,
                    storage=storage,
                    device=stage.execution_device,
                    authoritative=stage._storage_values(storage),
                )
            )
        before_close = stage.diagnostics()["dynamic_vram"]
    except BaseException as exc:  # noqa: BLE001 - close helper preserves primary
        primary = exc
    _close_diagnostic_stage(
        stage,
        report=report,
        primary=primary,
        hard_exit=hard_exit,
    )
    if before_close is None:
        raise RuntimeError("diagnostic completed without pre-close proof")
    closed = stage.diagnostics()["dynamic_vram"]
    _require_file_backed_close_proof(before_close, closed)
    if closed["live_allocations"] != 0 or closed["live_bytes"] != 0 or closed["loaded_bytes"] != 0:
        raise RuntimeError("diagnostic AIMDO close retained live residency")
    if (
        closed["free_calls"] != before_close["free_calls"] + 1
        or closed["unpin_calls"] != before_close["unpin_calls"]
        or before_close["live_allocations"] != before_close["allocation_count"]
        or before_close["live_bytes"] != before_close["virtual_bytes"]
        or closed["host_buffer_live"] is not False
        or closed["host_tensor_view_live"] is not False
        or closed["host_buffer_transfer_pending"] is not False
        or closed["host_buffer_unregistrations"] != closed["host_buffer_allocations"]
        or closed["host_buffer_frees"] != closed["host_buffer_allocations"]
    ):
        raise RuntimeError("diagnostic AIMDO close counters are not exact")
    return {
        "artifact_header_sha256": plan.identity.header_sha256,
        "device": str(stage.execution_device),
        "groups": results,
        "before_close": before_close,
        "after_close": closed,
    }
