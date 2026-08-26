from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from latentslate_engine.runtime import ltx23_av_aimdo as direct
from latentslate_engine.runtime import ltx23_av_stored_adapter as stored
from latentslate_engine.runtime.ltx23_av_aimdo import (
    LTX23AVAimdoState,
    _Layout,
    _Leaf,
    _Physical,
    _pin_regions,
    _PinRegion,
)
from latentslate_engine.runtime.ltx23_av_stored_adapter import LTX23AVFileSpan


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(65, 65)
        self.unused = nn.Parameter(torch.ones(4097))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(value)


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Linear(65, 65)
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(48)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.input_proj(value)
        for block in self.transformer_blocks:
            value = block(value)
        return value


class _TargetBlock(nn.Module):
    def __init__(self, target: nn.Module) -> None:
        super().__init__()
        self.target = target

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.target(value)


class _TargetTransformer(nn.Module):
    def __init__(self, target: nn.Module) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_TargetBlock(target), *(nn.Identity() for _ in range(47))]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            value = block(value)
        return value


def _stored_fp8_linear(features: int = 256) -> stored.LTX23StoredFP8Linear:
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

    qdata = torch.zeros((features, features), dtype=torch.float8_e4m3fn)
    params = TensorCoreFP8Layout.Params(
        scale=torch.tensor(0.25, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
        orig_shape=tuple(qdata.shape),
    )
    return stored.LTX23StoredFP8Linear(
        QuantizedTensor(qdata, "TensorCoreFP8Layout", params),
        torch.zeros(features, dtype=torch.bfloat16),
        input_scale=torch.tensor(0.5, dtype=torch.float32),
    )


def _current_slot_value(slot) -> torch.Tensor:
    values = slot.module._parameters if slot.parameter else slot.module._buffers
    return values[slot.name]


def test_cpu_characterization_uses_operation_hooks_and_block_queue_prefetch() -> None:
    transformer = _Transformer()
    original = tuple(transformer.state_dict().values())
    state = LTX23AVAimdoState(transformer, torch.device("cpu"))
    starts = 0

    def before_first() -> None:
        nonlocal starts
        starts += 1

    with state.forward_scope(before_first):
        first = transformer(torch.ones(1, 65))
        assert all(leaf.prefetched_values is None for leaf in state._leaves)
    with state.forward_scope(before_first):
        second = transformer(torch.ones(1, 65))

    assert first.shape == second.shape == (1, 65)
    assert starts == 2
    assert state.active is None
    assert state.policy["dynamic_vram"]["prefetch_calls"] == 194
    assert all(
        torch.equal(current, prior)
        for current, prior in zip(transformer.state_dict().values(), original, strict=True)
    )
    state.close()
    assert not state.handles


def test_fp8_lora_companion_uses_one_leaf_for_base_and_adapter_outer_forward(
    monkeypatch,
) -> None:
    target = _stored_fp8_linear()
    target.add_lora_adapter(
        "distilled",
        torch.full((32, 256), 0.01, dtype=torch.bfloat16),
        torch.full((256, 32), 0.01, dtype=torch.bfloat16),
        alpha_over_rank=1.0,
    )
    target.set_lora_strength("distilled", 0.5)
    transformer = _TargetTransformer(target)
    input = torch.ones((1, 256), dtype=torch.bfloat16)
    active_leaf: _Leaf | None = None
    base_observations: list[int] = []

    def direct_linear(flat, weight, bias, *, input_scale):
        del input_scale
        if active_leaf is not None:
            base_observations.append(active_leaf.users)
        return torch.zeros(
            (flat.shape[0], weight.shape[0]), dtype=flat.dtype, device=flat.device
        ) + bias

    monkeypatch.setattr(stored, "_direct_kitchen_fp8_linear", direct_linear)
    expected = transformer(input)
    state = LTX23AVAimdoState(transformer, torch.device("cpu"))
    target_leaves = [
        leaf
        for leaf in state._leaves
        if leaf.storage.operation_modules == (target,)
    ]
    assert len(target_leaves) == 1
    active_leaf = target_leaves[0]
    assert active_leaf.storage.path == "transformer_blocks.0.target"
    assert {slot.name for slot in active_leaf.storage.storage.slots} == {
        "weight",
        "bias",
        "input_scale",
    }
    assert {
        slot.name for slot in active_leaf.storage.companion_storage.slots
    } == {
        "down",
        "up",
    }
    assert not active_leaf.storage.force_resident
    assert active_leaf.companion_size > 0
    assert state.base_stored_bytes == active_leaf.storage.storage.physical_bytes
    assert (
        state.companion_stored_bytes
        == active_leaf.storage.companion_storage.physical_bytes
    )
    originals = {
        (id(slot.module), slot.name): slot.cpu_value
        for storage in (
            active_leaf.storage.storage,
            active_leaf.storage.companion_storage,
        )
        for slot in storage.slots
    }
    adapter_observations: list[int] = []
    handle = target._lora_adapters["distilled"].register_forward_pre_hook(
        lambda _module, _inputs: adapter_observations.append(active_leaf.users)
    )

    with state.forward_scope(lambda: None):
        actual = transformer(input)

    handle.remove()
    torch.testing.assert_close(actual, expected)
    assert base_observations[-1:] == [1]
    assert adapter_observations == [1]
    assert state.policy["leaf_allocation_count"] == 1
    assert state.policy["dynamic_vram"]["allocation_count"] == 1
    assert state.policy["dynamic_vram"]["prefetch_calls"] == 1
    assert active_leaf.users == 0 and active_leaf.binding is None
    assert all(
        _current_slot_value(slot) is originals[(id(slot.module), slot.name)]
        for storage in (
            active_leaf.storage.storage,
            active_leaf.storage.companion_storage,
        )
        for slot in storage.slots
    )
    state.close()
    assert not state.handles
    assert not active_leaf.pins


def test_dense_lora_companion_holds_base_and_adapter_through_wrapper_forward() -> None:
    base = nn.Linear(96, 96, bias=True, dtype=torch.bfloat16)
    target = stored.LTX23DenseLoraLinear(base)
    target.add_lora_adapter(
        "distilled",
        torch.full((65, 96), 0.005, dtype=torch.bfloat16),
        torch.full((96, 65), 0.005, dtype=torch.bfloat16),
        alpha_over_rank=1.0,
    )
    target.set_lora_strength("distilled", 0.5)
    transformer = _TargetTransformer(target)
    input = torch.linspace(-1.0, 1.0, 96, dtype=torch.bfloat16).unsqueeze(0)
    expected = transformer(input)
    state = LTX23AVAimdoState(transformer, torch.device("cpu"))
    target_leaves = [
        leaf
        for leaf in state._leaves
        if leaf.storage.operation_modules == (target,)
    ]
    assert len(target_leaves) == 1
    leaf = target_leaves[0]
    assert leaf.storage.path == "transformer_blocks.0.target.base"
    assert {slot.name for slot in leaf.storage.storage.slots} == {
        "weight",
        "bias",
    }
    assert {slot.name for slot in leaf.storage.companion_storage.slots} == {
        "down",
        "up",
    }
    assert not leaf.storage.force_resident
    originals = {
        (id(slot.module), slot.name): slot.cpu_value
        for storage in (leaf.storage.storage, leaf.storage.companion_storage)
        for slot in storage.slots
    }
    base_observations: list[int] = []
    adapter_observations: list[int] = []
    handles = (
        base.register_forward_pre_hook(
            lambda _module, _inputs: base_observations.append(leaf.users)
        ),
        target._lora_adapters["distilled"].register_forward_pre_hook(
            lambda _module, _inputs: adapter_observations.append(leaf.users)
        ),
    )

    with state.forward_scope(lambda: None):
        actual = transformer(input)

    for handle in handles:
        handle.remove()
    torch.testing.assert_close(actual, expected)
    assert base_observations == [1]
    assert adapter_observations == [1]
    assert state.policy["leaf_allocation_count"] == 1
    assert state.policy["dynamic_vram"]["allocation_count"] == 1
    assert state.policy["dynamic_vram"]["prefetch_calls"] == 1
    assert leaf.users == 0 and leaf.binding is None
    assert all(
        _current_slot_value(slot) is originals[(id(slot.module), slot.name)]
        for storage in (leaf.storage.storage, leaf.storage.companion_storage)
        for slot in storage.slots
    )
    state.close()
    assert not state.handles
    assert not leaf.pins


def test_force_resident_base_still_hooks_and_stages_dynamic_companion() -> None:
    base = nn.Linear(8, 8, bias=True, dtype=torch.bfloat16)
    target = stored.LTX23DenseLoraLinear(base)
    target.add_lora_adapter(
        "distilled",
        torch.ones((2, 8), dtype=torch.bfloat16),
        torch.ones((8, 2), dtype=torch.bfloat16),
        alpha_over_rank=1.0,
    )
    target.set_lora_strength("distilled", 0.5)
    transformer = _TargetTransformer(target)
    state = LTX23AVAimdoState(transformer, torch.device("cpu"))
    leaf = next(
        item for item in state._leaves if item.storage.operation_modules == (target,)
    )
    assert leaf.storage.force_resident
    assert leaf.storage.companion_storage is not None
    observations: list[int] = []
    handle = target._lora_adapters["distilled"].register_forward_pre_hook(
        lambda _module, _inputs: observations.append(leaf.users)
    )

    with state.forward_scope(lambda: None):
        output = transformer(torch.ones((1, 8), dtype=torch.bfloat16))

    handle.remove()
    assert output.shape == (1, 8)
    assert observations == [1]
    assert state.policy["dynamic_vram"]["allocation_count"] == 0
    assert state.policy["dynamic_vram"]["prefetch_calls"] == 1
    assert leaf.companion_prefetched_values is None
    assert leaf.companion_binding is None
    state.close()


def test_first_transformer_root_reprioritizes_after_conditioning_only_once() -> None:
    events: list[str] = []
    state = LTX23AVAimdoState.__new__(LTX23AVAimdoState)
    state.device = torch.device("cuda", 0)
    state.transformer = object()
    state._vbar = SimpleNamespace(prioritize=lambda: events.append("av_priority"))
    state._owner_thread = None
    state._before_first = None
    state._scope_started = False
    state._executing = False
    state._active_block = None
    state._closed = False
    state._poison_reason = None
    state._leaves = ()
    state._by_group = {"root": (), "transformer_blocks.47": ()}
    state._release_all_operation_state = MethodType(lambda _self: None, state)

    with state.forward_scope(lambda: events.append("before_first")):
        assert events == []
        events.append("video_vae_priority")
        state._root_pre(state.transformer, ())
        state._root_post(state.transformer, (), None)
        state._root_pre(state.transformer, ())
        state._root_post(state.transformer, (), None)

    assert events == ["video_vae_priority", "av_priority", "before_first"]


def test_active_av_path_does_not_import_retired_generic_residency() -> None:
    root = Path(__file__).parents[1]
    kitchen = (root / "src/latentslate_engine/runtime/ltx23_kitchen.py").read_text()
    direct = (root / "src/latentslate_engine/runtime/ltx23_av_aimdo.py").read_text()
    combined = kitchen + direct

    assert "LeafResidencyScheduler" not in combined
    assert "AimdoDynamicResidency" not in combined
    assert "AimdoHostSourcePool" not in combined
    assert ".prepare_stage(" not in kitchen


def test_first_pin_fill_populates_gpu_without_a_second_whole_pin_copy() -> None:
    state = LTX23AVAimdoState.__new__(LTX23AVAimdoState)
    state._h2d_bytes = 0
    state._can_pin = MethodType(lambda _self, _size: True, state)
    pin = torch.zeros(8, dtype=torch.uint8)
    state._make_pin = MethodType(lambda _self, _region: pin, state)

    def fill_pin(_self, region, created_pin, raw, stream) -> int:
        assert region.subset == "patches"
        assert created_pin is pin
        assert stream is None
        raw.fill_(7)
        return 8

    state._fill_pin = MethodType(fill_pin, state)
    source = torch.zeros(8, dtype=torch.uint8)
    physical = _Physical(source, None, 0, 8, torch.uint8, (8,))
    layout = _Layout(source, None, None, (physical,))
    leaf = _Leaf(None, (layout,), (0,), 8)
    raw = torch.zeros(8, dtype=torch.uint8)

    state._fill(leaf, raw, None)

    assert torch.equal(raw, torch.full_like(raw, 7))
    assert leaf.pins == {("patches", 0, 8): pin}
    assert state._h2d_bytes == 8


def test_first_file_pin_fill_passes_cpu_and_gpu_destinations_to_native_reader() -> None:
    calls: list[dict[str, object]] = []

    class HostBuffer:
        def __init__(self, pin: torch.Tensor) -> None:
            self.pin = pin

        def get_raw_address(self) -> int:
            return self.pin.data_ptr()

        def read_file_slice(self, file, file_offset, size, **kwargs) -> None:
            calls.append(
                {
                    "file": file,
                    "file_offset": file_offset,
                    "size": size,
                    **kwargs,
                }
            )

    state = LTX23AVAimdoState.__new__(LTX23AVAimdoState)
    state.device = torch.device("cuda", 0)
    state._file = object()
    state._source_read_calls = 0
    state._source_read_bytes = 0
    pin = torch.zeros(8, dtype=torch.uint8)
    raw = torch.zeros(8, dtype=torch.uint8)
    state._hostbufs = {"weights": HostBuffer(pin)}
    span = LTX23AVFileSpan("checkpoint", "weight", 4096, 8, torch.uint8, (8,))
    physical = _Physical(None, span, 0, 8, torch.uint8, (8,))
    region = _PinRegion("weights", 0, 8, (physical,))
    stream = SimpleNamespace(cuda_stream=1234)

    assert state._fill_pin(region, pin, raw, stream) == 8

    assert calls == [
        {
            "file": state._file,
            "file_offset": 4096,
            "size": 8,
            "offset": 0,
            "stream": 1234,
            "device_ptr": raw.data_ptr(),
            "device": 0,
        }
    ]
    assert state._source_read_calls == 1
    assert state._source_read_bytes == 8


def test_mixed_companion_preserves_distinct_base_and_patch_source_regions() -> None:
    base_span = LTX23AVFileSpan(
        "checkpoint", "weight", 4096, 8, torch.uint8, (8,)
    )
    base = _Physical(None, base_span, 0, 8, torch.uint8, (8,))
    patch_tensor = torch.ones(8, dtype=torch.uint8)
    patch = _Physical(patch_tensor, None, 1024, 8, torch.uint8, (8,))
    leaf = _Leaf(
        None,
        (
            _Layout(torch.empty(8, dtype=torch.uint8), None, None, (base,)),
            _Layout(patch_tensor, None, None, (patch,)),
        ),
        (0, 1),
        2048,
    )

    regions = _pin_regions(leaf)

    assert [(region.subset, region.offset, region.size) for region in regions] == [
        ("weights", 0, 8),
        ("patches", 1024, 8),
    ]
    assert regions[0].physical == (base,)
    assert regions[1].physical == (patch,)


def test_interleaved_source_regions_keep_distinct_cached_second_fill_ranges() -> None:
    first_span = LTX23AVFileSpan(
        "checkpoint", "first", 4096, 8, torch.uint8, (8,)
    )
    second_span = LTX23AVFileSpan(
        "checkpoint", "second", 8192, 8, torch.uint8, (8,)
    )
    patch_tensor = torch.ones(8, dtype=torch.uint8)
    physical = (
        _Physical(None, first_span, 0, 8, torch.uint8, (8,)),
        _Physical(patch_tensor, None, 1024, 8, torch.uint8, (8,)),
        _Physical(None, second_span, 2048, 8, torch.uint8, (8,)),
    )
    leaf = _Leaf(
        None,
        tuple(
            _Layout(
                torch.empty(8, dtype=torch.uint8),
                None,
                None,
                (item,),
            )
            for item in physical
        ),
        (0, 1, 2),
        3072,
    )
    regions = _pin_regions(leaf)
    assert [(region.subset, region.offset, region.size) for region in regions] == [
        ("weights", 0, 8),
        ("patches", 1024, 8),
        ("weights", 2048, 8),
    ]

    state = LTX23AVAimdoState.__new__(LTX23AVAimdoState)
    state._h2d_bytes = 0
    state._can_pin = MethodType(lambda _self, _size: True, state)
    state._make_pin = MethodType(
        lambda _self, region: torch.zeros(region.size, dtype=torch.uint8), state
    )

    def fill_pin(_self, region, pin, raw, _stream) -> int:
        marker = {0: 1, 1024: 2, 2048: 3}[region.offset]
        pin.fill_(marker)
        raw[region.offset : region.offset + region.size].copy_(pin)
        return region.size

    state._fill_pin = MethodType(fill_pin, state)
    state._fill(leaf, torch.zeros(3072, dtype=torch.uint8), None)
    second = torch.full((3072,), 99, dtype=torch.uint8)

    state._fill(leaf, second, None)

    assert set(leaf.pins) == {
        ("weights", 0, 8),
        ("patches", 1024, 8),
        ("weights", 2048, 8),
    }
    assert torch.equal(second[0:8], torch.full((8,), 1, dtype=torch.uint8))
    assert torch.equal(second[1024:1032], torch.full((8,), 2, dtype=torch.uint8))
    assert torch.equal(second[2048:2056], torch.full((8,), 3, dtype=torch.uint8))
    assert second[8:1024].eq(99).all()
    assert second[1032:2048].eq(99).all()


class _FailureStream:
    cuda_stream = 1234

    def __init__(self, failure: BaseException | None = None) -> None:
        self.synchronize_calls = 0
        self.failure = failure

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        if self.failure is not None:
            raise self.failure


class _FailureVBAR:
    def __init__(self, signature) -> None:
        self.signature = signature
        self.unpinned: list[object] = []

    def vbar_fault(self, _allocation):
        return self.signature

    @staticmethod
    def vbar_signature_compare(_actual, _expected) -> bool:
        return False

    def vbar_unpin(self, allocation) -> None:
        self.unpinned.append(allocation)


def _failure_state(signature, raw: torch.Tensor):
    state = LTX23AVAimdoState.__new__(LTX23AVAimdoState)
    state.device = torch.device("cuda", 0)
    state.transformer = nn.Module()
    state._model_vbar = _FailureVBAR(signature)
    state._aimdo_torch = SimpleNamespace(aimdo_to_tensor=lambda _allocation, _device: raw)
    state._streams = (_FailureStream(),)
    state._vrambufs = (SimpleNamespace(get=lambda _size, _offset: object()),)
    state._patch_vrambufs = (SimpleNamespace(get=lambda _size, _offset: object()),)
    state._stream_index = 0
    state._prefetch_calls = 0
    state._faults = 0
    state._signature_hits = 0
    state._signature_misses = 0
    state._fault_none = 0
    state._unpin_calls = 0
    state._h2d_bytes = 0
    state._source_read_calls = 0
    state._source_read_bytes = 0
    state._poison_reason = None
    state._owner_thread = None
    state._closed = False
    state._before_first = None
    return state


def test_signature_hit_still_stages_companion_in_patch_temporary_buffer(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    base_source = torch.ones(8, dtype=torch.uint8)
    patch_source = torch.full((8,), 2, dtype=torch.uint8)
    base_physical = _Physical(base_source, None, 0, 8, torch.uint8, (8,))
    patch_physical = _Physical(patch_source, None, 0, 8, torch.uint8, (8,))
    cached_base = (torch.full((8,), 3, dtype=torch.uint8),)
    leaf = _Leaf(
        SimpleNamespace(force_resident=False, companion_storage=object()),
        (_Layout(base_source, None, None, (base_physical,)),),
        (0,),
        1024,
        companion_layouts=(
            _Layout(patch_source, None, None, (patch_physical,)),
        ),
        companion_value_indices=(0,),
        companion_size=8,
        allocation=object(),
        signature=17,
        cached_values=cached_base,
    )
    patch_gets: list[tuple[int, int]] = []
    base_gets: list[tuple[int, int]] = []
    state = _failure_state(17, torch.zeros(8, dtype=torch.uint8))
    state._model_vbar.vbar_signature_compare = lambda _actual, _expected: True
    state._vrambufs = (
        SimpleNamespace(
            get=lambda size, offset: base_gets.append((size, offset)) or object()
        ),
    )
    state._patch_vrambufs = (
        SimpleNamespace(
            get=lambda size, offset: patch_gets.append((size, offset)) or object()
        ),
    )
    fills: list[bool] = []
    state._fill = MethodType(
        lambda _self, _leaf, _raw, _stream, *, companion=False: fills.append(
            companion
        ),
        state,
    )

    offsets = state._prefetch(leaf, stream_index=0)

    assert offsets == (0, 1024)
    assert state._faults == state._signature_hits == 1
    assert state._signature_misses == 0
    assert leaf.prefetched_values is cached_base
    assert leaf.prefetched_signature == 17
    assert leaf.companion_prefetched_values is not None
    assert leaf.companion_temporary_raw is not None
    assert base_gets == []
    assert patch_gets == [(8, 0)]
    assert fills == [True]


def test_signature_backed_prefetch_cpu_lora_copy_failure_is_terminal_and_unpins(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    base_span = LTX23AVFileSpan(
        "checkpoint", "weight", 4096, 8, torch.uint8, (8,)
    )
    base = _Physical(None, base_span, 0, 8, torch.uint8, (8,))
    invalid_patch = torch.ones(3, dtype=torch.uint8)
    patch = _Physical(invalid_patch, None, 0, 8, torch.uint8, (8,))
    allocation = object()
    leaf = _Leaf(
        SimpleNamespace(force_resident=False, companion_storage=object()),
        (_Layout(torch.empty(8, dtype=torch.uint8), None, None, (base,)),),
        (0,),
        1024,
        companion_layouts=(
            _Layout(invalid_patch, None, None, (patch,)),
        ),
        companion_value_indices=(0,),
        companion_size=1024,
        allocation=allocation,
        signature=7,
        cached_values=(object(),),
    )
    state = _failure_state(11, torch.zeros(2048, dtype=torch.uint8))
    state._can_pin = MethodType(lambda _self, _size: False, state)
    state._file = object()
    state._host_buffer = SimpleNamespace(read_file_to_device=lambda *_args, **_kwargs: None)

    with pytest.raises(direct.LTX23AVAimdoPoisoned) as caught:
        state._prefetch(leaf, stream_index=0)

    assert caught.value.reason == "stage_prepare_failed"
    assert state._streams[0].synchronize_calls == 1
    assert state._model_vbar.unpinned == [allocation]
    assert state._unpin_calls == 1
    assert leaf.prefetched_values is None
    assert leaf.prefetched_signature is None
    assert leaf.transfer_stream is None
    assert leaf.temporary_raw is None
    assert leaf.companion_prefetched_values is None
    assert leaf.companion_temporary_raw is None
    assert leaf.signature is None and leaf.cached_values is None
    assert leaf.users == 0 and leaf.binding is None
    assert state.transformer._latentslate_ltx23_residency_poisoned == caught.value.reason
    faults = state._faults
    with pytest.raises(direct.LTX23AVAimdoPoisoned):
        state._prefetch(leaf, stream_index=0)
    assert state._faults == faults
    with pytest.raises(direct.LTX23AVAimdoPoisoned), state.forward_scope(lambda: None):
        pass


def test_signature_none_prefetch_reconstruction_failure_clears_temporary_without_unpin(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    source = torch.ones(8, dtype=torch.uint8)
    physical = _Physical(source, None, 0, 8, torch.uint8, (8,))
    leaf = _Leaf(
        SimpleNamespace(force_resident=False, companion_storage=None),
        (_Layout(source, None, None, (physical,)),),
        (0,),
        1024,
        allocation=object(),
    )
    state = _failure_state(None, torch.zeros(1024, dtype=torch.uint8))
    state._fill = MethodType(lambda _self, _leaf, _raw, _stream: None, state)

    def fail_rebuild(_leaf, _raw, **_kwargs):
        raise RuntimeError("injected raw reconstruction failure")

    monkeypatch.setattr(direct, "_raw_values", fail_rebuild)

    with pytest.raises(direct.LTX23AVAimdoPoisoned) as caught:
        state._prefetch(leaf, stream_index=0)

    assert caught.value.reason == "stage_prepare_failed"
    assert state._streams[0].synchronize_calls == 1
    assert state._model_vbar.unpinned == []
    assert state._unpin_calls == 0
    assert leaf.temporary_raw is None
    assert leaf.prefetched_values is None
    assert leaf.transfer_stream is None
    assert state._fault_none == 1


def test_sync_failure_freezes_entire_owner_and_close_rejects_before_native_mutation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    native_events: list[str] = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _device: native_events.append("device_synchronize"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "cudart",
        lambda: native_events.append("cudart") or SimpleNamespace(),
    )
    source = torch.ones(8, dtype=torch.uint8)
    physical = _Physical(source, None, 0, 8, torch.uint8, (8,))
    layout = _Layout(source, None, None, (physical,))
    prior_allocation = object()
    prior_values = (object(),)
    prior_raw = torch.ones(8, dtype=torch.uint8)
    prior = _Leaf(
        SimpleNamespace(force_resident=False, companion_storage=None),
        (layout,),
        (0,),
        1024,
        allocation=prior_allocation,
        signature=5,
        cached_values=prior_values,
        prefetched_values=prior_values,
        prefetched_signature=9,
        temporary_raw=prior_raw,
    )
    failing_allocation = object()
    failing_values = (object(),)
    failing_raw = torch.full((8,), 2, dtype=torch.uint8)
    failing = _Leaf(
        SimpleNamespace(force_resident=False, companion_storage=None),
        (layout,),
        (0,),
        1024,
        allocation=failing_allocation,
        signature=7,
        cached_values=failing_values,
        prefetched_values=failing_values,
        prefetched_signature=10,
        temporary_raw=failing_raw,
    )
    state = _failure_state(11, torch.zeros(1024, dtype=torch.uint8))
    state._streams = (_FailureStream(RuntimeError("injected stream sync failure")),)
    state._fill = MethodType(
        lambda _self, _leaf, _raw, _stream: (_ for _ in ()).throw(
            RuntimeError("injected fill failure")
        ),
        state,
    )
    state._leaves = (prior, failing)
    state._scope_started = False
    state._executing = False
    state._active_block = None
    state._by_group = {"root": (), "transformer_blocks.47": ()}
    state._handles = []
    state._pins = [(torch.zeros(1, dtype=torch.uint8), True)]
    vbar = SimpleNamespace(_ptr=object(), _devctx=object())
    state._vbar = vbar
    state._model_vbar.lib = SimpleNamespace(
        vbar_free=lambda *_args: native_events.append("vbar_free")
    )
    file_owner = SimpleNamespace(close=lambda: native_events.append("file_close"))
    state._file = file_owner

    with pytest.raises(
        direct.LTX23AVAimdoPoisoned
    ) as caught, state.forward_scope(lambda: None):
        state._prefetch(failing, stream_index=0)

    assert caught.value.reason == "retirement_cleanup_failed"
    assert state._streams[0].synchronize_calls == 1
    assert state._model_vbar.unpinned == []
    assert state._unpin_calls == 0
    assert prior.prefetched_values is prior_values
    assert prior.prefetched_signature == 9
    assert prior.temporary_raw is prior_raw
    assert prior.signature == 5 and prior.cached_values is prior_values
    assert prior.users == 0
    assert failing.prefetched_values is not None
    assert failing.prefetched_signature == 11
    assert failing.temporary_raw is failing_raw
    assert failing.signature == 7 and failing.cached_values is failing_values
    assert failing.users == 0

    faults = state._faults
    with pytest.raises(direct.LTX23AVAimdoPoisoned):
        state._prefetch(failing, stream_index=0)
    assert state._faults == faults
    with pytest.raises(direct.LTX23AVAimdoPoisoned):
        state.close()
    assert native_events == []
    assert state._pins[0][1] is True
    assert state._vbar is vbar and vbar._ptr is not None
    assert state._file is file_owner


def test_prefetch_unpin_cleanup_failure_uses_canonical_terminal_poison(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    source = torch.ones(8, dtype=torch.uint8)
    physical = _Physical(source, None, 0, 8, torch.uint8, (8,))
    allocation = object()
    values = (object(),)
    leaf = _Leaf(
        SimpleNamespace(force_resident=False, companion_storage=None),
        (_Layout(source, None, None, (physical,)),),
        (0,),
        1024,
        allocation=allocation,
        signature=7,
        cached_values=values,
    )
    state = _failure_state(11, torch.zeros(1024, dtype=torch.uint8))
    unpin_attempts: list[object] = []

    def fail_unpin(item: object) -> None:
        unpin_attempts.append(item)
        raise RuntimeError("injected failed-prefetch unpin failure")

    state._model_vbar.vbar_unpin = fail_unpin
    state._fill = MethodType(
        lambda _self, _leaf, _raw, _stream: (_ for _ in ()).throw(
            RuntimeError("injected fill failure")
        ),
        state,
    )

    with pytest.raises(
        direct.LTX23AVAimdoPoisoned, match="retirement_cleanup_failed"
    ):
        state._prefetch(leaf, stream_index=0)

    assert state._streams[0].synchronize_calls == 1
    assert unpin_attempts == [allocation]
    assert state._unpin_calls == 0
    assert leaf.prefetched_values is not None
    assert leaf.prefetched_signature == 11
    assert leaf.signature == 7 and leaf.cached_values is values
    faults = state._faults
    with pytest.raises(
        direct.LTX23AVAimdoPoisoned, match="retirement_cleanup_failed"
    ):
        state._prefetch(leaf, stream_index=0)
    assert state._faults == faults
    assert unpin_attempts == [allocation]


@pytest.mark.parametrize(
    "failure",
    ("wrapper_raises", "nonzero", "clear_error_raises"),
)
def test_close_host_unregister_failure_poison_freezes_all_native_owners(
    monkeypatch, failure: str
) -> None:
    events: list[str] = []

    class Cudart:
        def cudaHostUnregister(self, _pointer: int) -> int:
            events.append("unregister")
            if failure == "wrapper_raises":
                raise RuntimeError("injected unregister wrapper failure")
            return 1

        def cudaGetLastError(self) -> int:
            events.append("clear_error")
            return 0

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _device: events.append("device_synchronize"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "cudart",
        lambda: events.append("cudart") or Cudart(),
    )
    if failure == "clear_error_raises":
        monkeypatch.setattr(
            LTX23AVAimdoState,
            "_clear_cuda_error",
            staticmethod(
                lambda _cudart: (_ for _ in ()).throw(
                    RuntimeError("injected clear-error failure")
                )
            ),
        )

    state = LTX23AVAimdoState.__new__(LTX23AVAimdoState)
    state.device = torch.device("cuda", 0)
    state.transformer = nn.Module()
    state._poison_reason = None
    state._closed = False
    state._owner_thread = None
    state._executing = False
    state._pins = [(torch.zeros(1, dtype=torch.uint8), True)]
    state._leaves = ()
    state._by_group = {}
    state._handles = []
    state._streams = ()
    state._vrambufs = ()
    state._patch_vrambufs = ()
    state._hostbufs = {}
    state._cleanup_calls = 0
    vbar = SimpleNamespace(_ptr=object(), _devctx=object())
    state._vbar = vbar
    state._model_vbar = SimpleNamespace(
        lib=SimpleNamespace(vbar_free=lambda *_args: events.append("vbar_free"))
    )
    file_owner = SimpleNamespace(close=lambda: events.append("file_close"))
    state._file = file_owner

    with pytest.raises(
        direct.LTX23AVAimdoPoisoned, match="retirement_cleanup_failed"
    ):
        state.close()

    assert state._poison_reason == "retirement_cleanup_failed"
    assert state.transformer._latentslate_ltx23_residency_poisoned == state._poison_reason
    assert state._pins[0][1] is True
    assert state._vbar is vbar and vbar._ptr is not None
    assert state._file is file_owner
    frozen_events = tuple(events)
    state._release_all_operation_state()
    with pytest.raises(
        direct.LTX23AVAimdoPoisoned, match="retirement_cleanup_failed"
    ):
        state.close()
    assert tuple(events) == frozen_events
    assert "vbar_free" not in events
    assert "file_close" not in events


class _FenceStream:
    def __init__(
        self,
        name: str,
        events: list[tuple[str, str, str]],
        failure: BaseException | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.failure = failure

    def wait_stream(self, other) -> None:
        self.events.append((self.name, "wait", other.name))
        if self.failure is not None:
            raise self.failure


def test_block_scope_uses_one_forward_and_reverse_fence_then_batch_unpins(
    monkeypatch,
) -> None:
    events: list[tuple[str, str, str]] = []
    compute = _FenceStream("compute", events)
    offload = _FenceStream("offload", events)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: compute)
    storage = SimpleNamespace(force_resident=False, companion_storage=None)
    leaves = tuple(
        _Leaf(
            storage,
            (),
            (),
            1024,
            allocation=object(),
            signature=index + 1,
        )
        for index in range(2)
    )
    state = LTX23AVAimdoState.__new__(LTX23AVAimdoState)
    state.device = torch.device("cuda", 0)
    state.transformer = nn.Module()
    state._poison_reason = None
    state._streams = (offload,)
    state._by_group = {"transformer_blocks.0": leaves}
    state._forward_stream_waits = 0
    state._reverse_stream_waits = 0
    state._unpin_calls = 0
    unpinned: list[object] = []
    state._model_vbar = SimpleNamespace(vbar_unpin=unpinned.append)

    def prefetch(
        _self,
        leaf,
        *,
        stream_index,
        temporary_offset,
        companion_offset,
    ):
        assert stream_index == 0
        leaf.prefetched_values = (object(),)
        leaf.prefetched_signature = leaf.signature
        return temporary_offset, companion_offset

    state._prefetch = MethodType(prefetch, state)

    state._prefetch_block(0)

    assert events == [("compute", "wait", "offload")]
    assert state._forward_stream_waits == 1
    for leaf in leaves:
        restored: list[str] = []
        leaf.users = 1
        leaf.binding = SimpleNamespace(
            restore_cpu=lambda restored=restored: restored.append("base")
        )
        state._leave_leaf(leaf)
        assert restored == ["base"]
        assert leaf.prefetched_values is not None
        assert leaf.prefetched_signature is not None
        assert leaf.transfer_stream is offload
        assert leaf.block_scoped
    assert events == [("compute", "wait", "offload")]
    assert unpinned == []

    state._release_prefetched_group("transformer_blocks.0")

    assert events == [
        ("compute", "wait", "offload"),
        ("offload", "wait", "compute"),
    ]
    assert state._reverse_stream_waits == 1
    assert unpinned == [leaf.allocation for leaf in leaves]
    assert all(
        leaf.prefetched_values is None
        and leaf.prefetched_signature is None
        and leaf.transfer_stream is None
        and not leaf.block_scoped
        for leaf in leaves
    )

    state._prefetch_block(0)
    assert events[-1] == ("compute", "wait", "offload")
    assert events.index(("offload", "wait", "compute")) < len(events) - 1


def test_block_forward_wait_failure_uses_canonical_poison_and_freezes(
    monkeypatch,
) -> None:
    events: list[tuple[str, str, str]] = []
    compute = _FenceStream(
        "compute", events, RuntimeError("injected block forward wait failure")
    )
    offload = _FenceStream("offload", events)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: compute)
    values = (object(),)
    leaf = _Leaf(
        SimpleNamespace(force_resident=False, companion_storage=None),
        (),
        (),
        1024,
        allocation=object(),
        signature=1,
    )
    state = LTX23AVAimdoState.__new__(LTX23AVAimdoState)
    state.device = torch.device("cuda", 0)
    state.transformer = nn.Module()
    state._poison_reason = None
    state._streams = (offload,)
    state._by_group = {"transformer_blocks.0": (leaf,)}
    state._forward_stream_waits = 0

    def prefetch(
        _self,
        target,
        *,
        stream_index,
        temporary_offset,
        companion_offset,
    ):
        target.prefetched_values = values
        target.prefetched_signature = 9
        return temporary_offset, companion_offset

    state._prefetch = MethodType(prefetch, state)

    with pytest.raises(
        direct.LTX23AVAimdoPoisoned, match="device_quiescence_failed"
    ):
        state._prefetch_block(0)

    assert leaf.prefetched_values is values
    assert leaf.prefetched_signature == 9
    assert leaf.transfer_stream is offload
    assert leaf.block_scoped
    assert state._forward_stream_waits == 0
    assert events == [("compute", "wait", "offload")]
    with pytest.raises(
        direct.LTX23AVAimdoPoisoned, match="device_quiescence_failed"
    ):
        state.close()


@pytest.mark.parametrize(
    ("failure", "reason"),
    (
        ("reverse_wait", "device_quiescence_failed"),
        ("batch_unpin", "retirement_release_failed"),
    ),
)
def test_block_release_failure_uses_canonical_poison_and_does_not_retry(
    monkeypatch, failure: str, reason: str
) -> None:
    events: list[tuple[str, str, str]] = []
    compute = _FenceStream("compute", events)
    offload = _FenceStream(
        "offload",
        events,
        RuntimeError("injected block reverse wait failure")
        if failure == "reverse_wait"
        else None,
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: compute)
    allocation = object()
    values = (object(),)
    raw = object()
    leaf = _Leaf(
        SimpleNamespace(force_resident=False, companion_storage=None),
        (),
        (),
        1024,
        allocation=allocation,
        prefetched_values=values,
        prefetched_signature=9,
        transfer_stream=offload,
        temporary_raw=raw,
        block_scoped=True,
    )
    unpin_attempts: list[object] = []

    def unpin(item: object) -> None:
        unpin_attempts.append(item)
        if failure == "batch_unpin":
            raise RuntimeError("injected block batch unpin failure")

    state = LTX23AVAimdoState.__new__(LTX23AVAimdoState)
    state.device = torch.device("cuda", 0)
    state.transformer = nn.Module()
    state._poison_reason = None
    state._by_group = {"transformer_blocks.0": (leaf,)}
    state._reverse_stream_waits = 0
    state._unpin_calls = 0
    state._model_vbar = SimpleNamespace(vbar_unpin=unpin)

    with pytest.raises(direct.LTX23AVAimdoPoisoned, match=reason):
        state._release_prefetched_group("transformer_blocks.0")

    assert leaf.prefetched_values is values
    assert leaf.prefetched_signature == 9
    assert leaf.transfer_stream is offload
    assert leaf.temporary_raw is raw
    assert leaf.block_scoped
    first_events = tuple(events)
    first_unpins = tuple(unpin_attempts)
    state._release_prefetched_group("transformer_blocks.0")
    assert tuple(events) == first_events
    assert tuple(unpin_attempts) == first_unpins
    with pytest.raises(direct.LTX23AVAimdoPoisoned, match=reason):
        state.close()


def test_root_leaf_keeps_operation_local_forward_reverse_wait_and_unpin(
    monkeypatch,
) -> None:
    events: list[tuple[str, str, str]] = []
    compute = _FenceStream("compute", events)
    offload = _FenceStream("offload", events)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: compute)

    class Binding:
        def __init__(self, *_args) -> None:
            self.restored = False

        def activate(self) -> None:
            pass

        def restore_cpu(self) -> None:
            self.restored = True

    monkeypatch.setattr(direct, "LTX23ModuleBinding", Binding)
    allocation = object()
    leaf = _Leaf(
        SimpleNamespace(
            force_resident=False,
            companion_storage=None,
            storage=object(),
        ),
        (),
        (),
        1024,
        allocation=allocation,
        prefetched_values=(object(),),
        prefetched_signature=13,
        transfer_stream=offload,
    )
    unpinned: list[object] = []
    state = LTX23AVAimdoState.__new__(LTX23AVAimdoState)
    state.device = torch.device("cuda", 0)
    state._poison_reason = None
    state._streams = (offload,)
    state._stream_index = 0
    state._forward_stream_waits = 0
    state._reverse_stream_waits = 0
    state._unpin_calls = 0
    state._model_vbar = SimpleNamespace(vbar_unpin=unpinned.append)

    state._enter_leaf(leaf)
    state._leave_leaf(leaf)

    assert events == [
        ("compute", "wait", "offload"),
        ("offload", "wait", "compute"),
    ]
    assert state._forward_stream_waits == state._reverse_stream_waits == 1
    assert unpinned == [allocation]
    assert leaf.prefetched_values is None


def test_operation_local_resident_hit_without_transfer_stream_unpins_without_fences(
    monkeypatch,
) -> None:
    def unexpected_current_stream(_device):
        raise AssertionError("resident signature hit must not manufacture a stream fence")

    monkeypatch.setattr(torch.cuda, "current_stream", unexpected_current_stream)

    class Binding:
        def __init__(self, *_args) -> None:
            self.restored = False

        def activate(self) -> None:
            pass

        def restore_cpu(self) -> None:
            self.restored = True

    monkeypatch.setattr(direct, "LTX23ModuleBinding", Binding)
    allocation = object()
    values = (object(),)
    leaf = _Leaf(
        SimpleNamespace(
            force_resident=False,
            companion_storage=None,
            storage=object(),
        ),
        (),
        (),
        1024,
        allocation=allocation,
        prefetched_values=values,
        prefetched_signature=17,
        transfer_stream=None,
    )
    unpinned: list[object] = []
    state = LTX23AVAimdoState.__new__(LTX23AVAimdoState)
    state.device = torch.device("cuda", 0)
    state._poison_reason = None
    state._streams = ()
    state._stream_index = 0
    state._forward_stream_waits = 0
    state._reverse_stream_waits = 0
    state._unpin_calls = 0
    state._model_vbar = SimpleNamespace(vbar_unpin=unpinned.append)

    state._enter_leaf(leaf)
    binding = leaf.binding
    state._leave_leaf(leaf)

    assert binding is not None and binding.restored
    assert state._forward_stream_waits == 0
    assert state._reverse_stream_waits == 0
    assert state._unpin_calls == 1
    assert unpinned == [allocation]
    assert leaf.users == 0 and leaf.binding is None
    assert leaf.prefetched_values is None
    assert leaf.prefetched_signature is None
    assert leaf.transfer_stream is None


def test_root_enter_forward_wait_failure_freezes_native_state_and_finalizers(
    monkeypatch,
) -> None:
    events: list[tuple[str, str, str]] = []
    compute = _FenceStream(
        "compute", events, RuntimeError("injected forward wait failure")
    )
    offload = _FenceStream("offload", events)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: compute)
    allocation = object()
    values = (object(),)
    raw = object()
    leaf = _Leaf(
        SimpleNamespace(
            force_resident=False,
            companion_storage=None,
            storage=object(),
        ),
        (),
        (),
        1024,
        allocation=allocation,
        prefetched_values=values,
        prefetched_signature=13,
        transfer_stream=offload,
        temporary_raw=raw,
    )
    unpinned: list[object] = []
    state = LTX23AVAimdoState.__new__(LTX23AVAimdoState)
    state.device = torch.device("cuda", 0)
    state.transformer = nn.Module()
    state._poison_reason = None
    state._streams = (offload,)
    state._stream_index = 0
    state._forward_stream_waits = 0
    state._model_vbar = SimpleNamespace(vbar_unpin=unpinned.append)

    with pytest.raises(
        direct.LTX23AVAimdoPoisoned, match="device_quiescence_failed"
    ):
        state._enter_leaf(leaf)

    assert state._poison_reason == "device_quiescence_failed"
    assert state.transformer._latentslate_ltx23_residency_poisoned == state._poison_reason
    assert state._forward_stream_waits == 0
    assert leaf.users == 0 and leaf.binding is None
    assert leaf.prefetched_values is values
    assert leaf.prefetched_signature == 13
    assert leaf.transfer_stream is offload
    assert leaf.temporary_raw is raw
    state._leave_leaf(leaf)
    state._release_all_operation_state()
    marker = object()
    assert state._root_post(nn.Module(), (), marker) is marker
    with pytest.raises(
        direct.LTX23AVAimdoPoisoned, match="device_quiescence_failed"
    ):
        state.close()
    assert events == [("compute", "wait", "offload")]
    assert unpinned == []


def test_root_leave_reverse_wait_failure_freezes_before_unpin_and_clear(
    monkeypatch,
) -> None:
    events: list[tuple[str, str, str]] = []
    compute = _FenceStream("compute", events)
    offload = _FenceStream(
        "offload", events, RuntimeError("injected reverse wait failure")
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: compute)
    allocation = object()
    values = (object(),)
    raw = object()
    restored: list[str] = []
    leaf = _Leaf(
        SimpleNamespace(force_resident=False, companion_storage=None),
        (),
        (),
        1024,
        allocation=allocation,
        prefetched_values=values,
        prefetched_signature=13,
        transfer_stream=offload,
        temporary_raw=raw,
        users=1,
        binding=SimpleNamespace(restore_cpu=lambda: restored.append("base")),
    )
    unpinned: list[object] = []
    state = LTX23AVAimdoState.__new__(LTX23AVAimdoState)
    state.device = torch.device("cuda", 0)
    state.transformer = nn.Module()
    state._poison_reason = None
    state._streams = (offload,)
    state._stream_index = 0
    state._reverse_stream_waits = 0
    state._model_vbar = SimpleNamespace(vbar_unpin=unpinned.append)

    with pytest.raises(
        direct.LTX23AVAimdoPoisoned, match="device_quiescence_failed"
    ):
        state._leave_leaf(leaf)

    assert restored == ["base"]
    assert leaf.users == 0 and leaf.binding is None
    assert leaf.prefetched_values is values
    assert leaf.prefetched_signature == 13
    assert leaf.transfer_stream is offload
    assert leaf.temporary_raw is raw
    assert state._reverse_stream_waits == 0
    state._leave_leaf(leaf)
    state._release_all_operation_state()
    with pytest.raises(
        direct.LTX23AVAimdoPoisoned, match="device_quiescence_failed"
    ):
        state.close()
    assert events == [("offload", "wait", "compute")]
    assert unpinned == []


def test_root_leave_unpin_failure_freezes_before_state_clear(monkeypatch) -> None:
    events: list[tuple[str, str, str]] = []
    compute = _FenceStream("compute", events)
    offload = _FenceStream("offload", events)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: compute)
    allocation = object()
    values = (object(),)
    raw = object()
    leaf = _Leaf(
        SimpleNamespace(force_resident=False, companion_storage=None),
        (),
        (),
        1024,
        allocation=allocation,
        prefetched_values=values,
        prefetched_signature=13,
        transfer_stream=offload,
        temporary_raw=raw,
        users=1,
    )
    unpin_attempts: list[object] = []

    def fail_unpin(item: object) -> None:
        unpin_attempts.append(item)
        raise RuntimeError("injected unpin failure")

    state = LTX23AVAimdoState.__new__(LTX23AVAimdoState)
    state.device = torch.device("cuda", 0)
    state.transformer = nn.Module()
    state._poison_reason = None
    state._streams = (offload,)
    state._stream_index = 0
    state._reverse_stream_waits = 0
    state._unpin_calls = 0
    state._model_vbar = SimpleNamespace(vbar_unpin=fail_unpin)

    with pytest.raises(
        direct.LTX23AVAimdoPoisoned, match="retirement_release_failed"
    ):
        state._leave_leaf(leaf)

    assert state._poison_reason == "retirement_release_failed"
    assert leaf.prefetched_values is values
    assert leaf.prefetched_signature == 13
    assert leaf.transfer_stream is offload
    assert leaf.temporary_raw is raw
    assert state._reverse_stream_waits == 1
    assert state._unpin_calls == 0
    state._leave_leaf(leaf)
    state._release_all_operation_state()
    with pytest.raises(
        direct.LTX23AVAimdoPoisoned, match="retirement_release_failed"
    ):
        state.close()
    assert events == [("offload", "wait", "compute")]
    assert unpin_attempts == [allocation]
