from __future__ import annotations

import copy
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

import latentslate_engine.runtime.ltx23_aimdo_diagnostic as diagnostic_module
from latentslate_engine.runtime.framework.residency import DynamicResidencyPoisoned
from latentslate_engine.runtime.ltx23_aimdo_diagnostic import (
    DIAGNOSTIC_POISON_EXIT_CODE,
    _close_diagnostic_stage,
    compare_resident_group,
    exercise_group,
    physical_tensors,
    representative_storage_groups,
)


class _QuantizedFixture:
    def __init__(self, qdata: torch.Tensor, scale: torch.Tensor) -> None:
        self.qdata = qdata
        self.scale = scale

    def __tensor_flatten__(self):
        return ["qdata", "scale"], {"layout": "fixture"}


def test_diagnostic_selects_unique_root_first_middle_last_groups() -> None:
    layers = tuple(object() for _index in range(6))
    stage = SimpleNamespace(_root_storage="root", _layer_storage=layers)

    assert representative_storage_groups(stage) == (
        ("root", "root"),
        ("layer_0", layers[0]),
        ("layer_3", layers[3]),
        ("layer_5", layers[5]),
    )


def test_diagnostic_compares_aliases_and_quantized_sidecars_in_chunks() -> None:
    dense = torch.arange(32, dtype=torch.bfloat16)
    resident_dense = dense.clone()
    quantized = _QuantizedFixture(
        torch.arange(16, dtype=torch.uint8), torch.tensor([0.5], dtype=torch.float32)
    )
    resident_quantized = _QuantizedFixture(quantized.qdata.clone(), quantized.scale.clone())

    proof = compare_resident_group(
        (dense, dense, quantized),
        (resident_dense, resident_dense, resident_quantized),
        chunk_bytes=7,
    )

    assert proof == {
        "logical_values": 3,
        "physical_tensors": 4,
        "compared_bytes": 148,
        "alias_pairs": 1,
        "file_physical_tensors": 0,
        "file_compared_bytes": 0,
        "cpu_physical_tensors": 4,
        "cpu_compared_bytes": 148,
    }
    assert [name for name, _tensor in physical_tensors(quantized)] == ["qdata", "scale"]

    resident_quantized.scale.fill_(0.25)
    with pytest.raises(RuntimeError, match="resident bytes differ"):
        compare_resident_group(
            (dense, quantized),
            (resident_dense, resident_quantized),
            chunk_bytes=7,
        )


def test_diagnostic_compares_mixed_file_backed_kitchen_templates_without_cpu_master(
    tmp_path,
) -> None:
    from comfy_kitchen.tensor import (
        QuantizedTensor,
        TensorCoreFP8Layout,
        TensorCoreNVFP4Layout,
    )

    from latentslate_engine.runtime.framework.residency.aimdo import (
        AimdoFileBackedValue,
        AimdoFileSpan,
    )

    fp8_qdata = torch.arange(16, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(4, 4)
    fp8_scale = torch.tensor(0.5, dtype=torch.float32)
    nvfp4_qdata = torch.arange(128, dtype=torch.uint8).reshape(16, 8)
    nvfp4_scale = torch.tensor(0.25, dtype=torch.float32)
    nvfp4_block = torch.ones((16, 1), dtype=torch.float8_e4m3fn)

    fp8_template = QuantizedTensor(
        torch.empty((4, 4), dtype=torch.float8_e4m3fn, device="meta"),
        "TensorCoreFP8Layout",
        TensorCoreFP8Layout.Params(
            scale=torch.empty((), dtype=torch.float32, device="meta"),
            orig_dtype=torch.bfloat16,
            orig_shape=(4, 4),
        ),
    )
    nvfp4_template = QuantizedTensor(
        torch.empty((16, 8), dtype=torch.uint8, device="meta"),
        "TensorCoreNVFP4Layout",
        TensorCoreNVFP4Layout.Params(
            scale=torch.empty((), dtype=torch.float32, device="meta"),
            orig_dtype=torch.bfloat16,
            orig_shape=(16, 16),
            block_scale=torch.empty((16, 1), dtype=torch.float8_e4m3fn, device="meta"),
        ),
    )
    fp8_resident = QuantizedTensor(
        fp8_qdata,
        "TensorCoreFP8Layout",
        TensorCoreFP8Layout.Params(
            scale=fp8_scale,
            orig_dtype=torch.bfloat16,
            orig_shape=(4, 4),
        ),
    )
    nvfp4_resident = QuantizedTensor(
        nvfp4_qdata,
        "TensorCoreNVFP4Layout",
        TensorCoreNVFP4Layout.Params(
            scale=nvfp4_scale,
            orig_dtype=torch.bfloat16,
            orig_shape=(16, 16),
            block_scale=nvfp4_block,
        ),
    )
    physical = (
        ("fp8_q", fp8_qdata),
        ("fp8_scale", fp8_scale),
        ("nvfp4_q", nvfp4_qdata),
        ("nvfp4_scale", nvfp4_scale),
        ("nvfp4_block", nvfp4_block),
    )
    payload = bytearray(4096)
    spans = {}
    cursor = 128
    for key, tensor in physical:
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        payload[cursor : cursor + len(raw)] = raw
        spans[key] = AimdoFileSpan(
            "base",
            key,
            cursor,
            len(raw),
            tensor.dtype,
            tuple(tensor.shape),
        )
        cursor += len(raw) + 17
    path = tmp_path / "mixed.safetensors"
    path.write_bytes(payload)
    fp8 = AimdoFileBackedValue(fp8_template, (spans["fp8_q"], spans["fp8_scale"]))
    nvfp4 = AimdoFileBackedValue(
        nvfp4_template,
        (spans["nvfp4_q"], spans["nvfp4_scale"], spans["nvfp4_block"]),
    )
    cpu = torch.arange(9, dtype=torch.bfloat16)
    with path.open("rb") as handle:
        proof = compare_resident_group(
            (fp8, fp8, nvfp4, cpu),
            (fp8_resident, fp8_resident, nvfp4_resident, cpu.clone()),
            chunk_bytes=7,
            file_sources={"base": handle},
        )

    assert proof["alias_pairs"] == 1
    assert proof["file_physical_tensors"] == 7
    assert proof["cpu_physical_tensors"] == 1
    contract = diagnostic_module._physical_copy_contract((fp8, fp8, nvfp4, cpu))
    assert contract[0] == 6
    assert contract[2] == 5
    assert contract[1] == contract[3] + cpu.numel() * cpu.element_size()


def test_diagnostic_exercises_miss_hit_and_forced_rebuild(monkeypatch) -> None:
    source = torch.arange(16, dtype=torch.uint8)
    storage = SimpleNamespace(slots=(SimpleNamespace(cpu_value=source),))

    class Backend:
        def __init__(self) -> None:
            self.hits = 0
            self.misses = 0
            self.faults = 0
            self.events = 0
            self.waits = 0
            self.pinned_bytes = 0
            self.unpins = 0
            self.gathered_misses = 0
            self.per_physical_misses = 0
            self.cached = None
            self.released_refs: tuple[weakref.ReferenceType[torch.Tensor], ...] = ()
            self._groups = {id(storage): SimpleNamespace(staged_bytes=16)}

        def diagnostics(self):
            return {
                "faults": self.faults,
                "signature_hits": self.hits,
                "signature_misses": self.misses,
                "fault_none_temporaries": 0,
                "pinned_copy_bytes": self.pinned_bytes,
                "pageable_copy_bytes": 0,
                "transfer_events": self.events,
                "transfer_waits": self.waits,
                "unpin_calls": self.unpins,
                "copy_strategy": "per_physical",
                "gathered_misses": self.gathered_misses,
                "per_physical_misses": self.per_physical_misses,
                "packed_source_bytes": 0,
                "gathered_h2d_bytes": 0,
                "host_buffer_reuse_barriers": 0,
                "host_source_pool_hits": 0,
                "host_source_pool_misses": 0,
                "host_buffer_transfer_pending": False,
                "base_file_read_calls": 0,
                "base_file_read_bytes": 0,
            }

        def acquire(self, _key):
            self.faults += 1
            if self.cached is None:
                self.misses += 1
                self.cached = (source.clone(),)
                self.events += 1
                self.waits += 1
                self.pinned_bytes += source.numel()
                self.per_physical_misses += 1
            else:
                self.hits += 1
            return SimpleNamespace(values=self.cached)

        def synchronize(self, _lease) -> None:
            pass

        def release(self, _lease) -> None:
            self.unpins += 1

        def invalidate(self, *, reason: str) -> None:
            assert reason == "diagnostic_force_miss"
            self.released_refs = tuple(weakref.ref(value) for value in self.cached)
            self.cached = None
            gc.collect()
            assert all(reference() is None for reference in self.released_refs)

    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    proof = exercise_group(
        Backend(),
        label="layer_0",
        storage=storage,
        device=torch.device("cuda:0"),
    )

    assert proof["label"] == "layer_0"
    assert proof["initial_miss"]["compared_bytes"] == 16
    assert proof["signature_hit"] == proof["initial_miss"]
    assert proof["forced_miss"] == proof["initial_miss"]


def test_diagnostic_progress_failure_cannot_skip_stage_offload() -> None:
    events: list[str] = []

    class Stage:
        def offload(self) -> None:
            events.append("offload")

        def terminal_poison_reason(self):
            return None

    def report(_message: str) -> None:
        events.append("report")
        raise RuntimeError("progress failed")

    with pytest.raises(RuntimeError, match="progress failed"):
        _close_diagnostic_stage(
            Stage(),
            report=report,
            primary=None,
            hard_exit=lambda _code: (_ for _ in ()).throw(AssertionError()),
        )

    assert events == ["report", "offload"]


def test_diagnostic_poison_retains_stage_and_hard_exits_without_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class HardExit(BaseException):
        pass

    class Stage:
        def offload(self) -> None:
            events.append("offload")
            raise DynamicResidencyPoisoned("device_quiescence_failed")

        def terminal_poison_reason(self):
            return "device_quiescence_failed"

        def __del__(self):
            events.append("finalizer")

    stage = Stage()
    monkeypatch.setattr(diagnostic_module, "_poisoned_diagnostic_retained", None)

    def hard_exit(code: int):
        events.append(("hard_exit", code))
        raise HardExit

    with pytest.raises(HardExit):
        _close_diagnostic_stage(
            stage,
            report=lambda _message: events.append("report"),
            primary=RuntimeError("primary diagnostic failure"),
            hard_exit=hard_exit,
        )

    assert events == ["report", "offload", ("hard_exit", DIAGNOSTIC_POISON_EXIT_CODE)]
    retained = diagnostic_module._poisoned_diagnostic_retained
    assert retained is not None and retained[0] is stage
    assert "finalizer" not in events


def test_diagnostic_file_backed_close_proof_rejects_lifecycle_tampering() -> None:
    before = {
        "copy_strategy": "gathered_host_buffer",
        "copy_fallback_reason": None,
        "base_file_backed": True,
        "base_file_source_live": True,
        "base_file_handle_live": True,
        "base_file_handle_opened": 1,
        "base_file_handle_closed": 0,
        "base_file_fallback_reason": None,
        "base_file_read_calls": 12,
        "base_file_read_bytes": 4096,
    }
    closed = {
        **before,
        "base_file_source_live": False,
        "base_file_handle_live": False,
        "base_file_handle_closed": 1,
    }

    diagnostic_module._require_file_backed_close_proof(before, closed)

    before_mutations = (
        ("copy_strategy", "per_physical"),
        ("copy_fallback_reason", "host_buffer_setup_failed: fixture"),
        ("base_file_backed", False),
        ("base_file_source_live", False),
        ("base_file_handle_live", False),
        ("base_file_handle_opened", 0),
        ("base_file_handle_closed", 1),
        ("base_file_fallback_reason", "host_buffer_setup_failed: fixture"),
        ("base_file_read_calls", 0),
        ("base_file_read_bytes", 0),
    )
    closed_mutations = (
        ("copy_strategy", "per_physical"),
        ("copy_fallback_reason", "host_buffer_setup_failed: fixture"),
        ("base_file_backed", False),
        ("base_file_source_live", True),
        ("base_file_handle_live", True),
        ("base_file_handle_opened", 0),
        ("base_file_handle_closed", 0),
        ("base_file_fallback_reason", "host_buffer_setup_failed: fixture"),
        ("base_file_read_calls", 13),
        ("base_file_read_bytes", 4097),
    )
    for key, value in before_mutations:
        tampered = copy.deepcopy(before)
        tampered[key] = value
        with pytest.raises(RuntimeError, match="close proof is not exact"):
            diagnostic_module._require_file_backed_close_proof(tampered, closed)
    for key, value in closed_mutations:
        tampered = copy.deepcopy(closed)
        tampered[key] = value
        with pytest.raises(RuntimeError, match="close proof is not exact"):
            diagnostic_module._require_file_backed_close_proof(before, tampered)


def test_diagnostic_acquire_delta_rejects_hit_miss_or_transfer_contradictions() -> None:
    before = {
        "faults": 0,
        "signature_hits": 0,
        "signature_misses": 0,
        "fault_none_temporaries": 0,
        "transfer_events": 0,
        "transfer_waits": 0,
        "unpin_calls": 0,
        "pinned_copy_bytes": 0,
        "pageable_copy_bytes": 0,
        "copy_strategy": "per_physical",
        "gathered_misses": 0,
        "per_physical_misses": 0,
        "packed_source_bytes": 0,
        "gathered_h2d_bytes": 0,
        "host_buffer_reuse_barriers": 0,
        "host_source_pool_hits": 0,
        "host_source_pool_misses": 0,
        "host_buffer_transfer_pending": False,
        "base_file_read_calls": 0,
        "base_file_read_bytes": 0,
    }
    valid_hit = {
        **before,
        "faults": 1,
        "signature_hits": 1,
        "unpin_calls": 1,
    }
    diagnostic_module._require_acquire_delta(
        before,
        valid_hit,
        label="hit",
        hit=True,
        physical_tensors=1,
        physical_bytes=16,
        staged_bytes=1024,
    )

    for tampered in (
        {**valid_hit, "signature_misses": 1},
        {**valid_hit, "transfer_events": 1},
        {**valid_hit, "pageable_copy_bytes": 16},
        {**valid_hit, "unpin_calls": 0},
    ):
        with pytest.raises(RuntimeError, match="counters are not exact"):
            diagnostic_module._require_acquire_delta(
                before,
                tampered,
                label="hit",
                hit=True,
                physical_tensors=1,
                physical_bytes=16,
                staged_bytes=1024,
            )

    gathered_before = {
        **before,
        "copy_strategy": "gathered_host_buffer",
    }
    gathered_miss = {
        **gathered_before,
        "faults": 1,
        "signature_misses": 1,
        "pinned_copy_bytes": 1024,
        "transfer_events": 1,
        "transfer_waits": 1,
        "unpin_calls": 1,
        "gathered_misses": 1,
        "packed_source_bytes": 300,
        "gathered_h2d_bytes": 1024,
        "host_source_pool_misses": 1,
        "base_file_read_calls": 5,
        "base_file_read_bytes": 280,
    }
    diagnostic_module._require_acquire_delta(
        gathered_before,
        gathered_miss,
        label="mixed miss",
        hit=False,
        physical_tensors=6,
        physical_bytes=300,
        staged_bytes=1024,
        file_physical_tensors=5,
        file_physical_bytes=280,
    )
    with pytest.raises(RuntimeError, match="counters are not exact"):
        diagnostic_module._require_acquire_delta(
            gathered_before,
            {**gathered_miss, "base_file_read_calls": 4},
            label="mixed miss",
            hit=False,
            physical_tensors=6,
            physical_bytes=300,
            staged_bytes=1024,
            file_physical_tensors=5,
            file_physical_bytes=280,
        )
