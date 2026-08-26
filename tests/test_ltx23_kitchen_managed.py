from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import latentslate_engine.runtime.ltx23_kitchen as kitchen_module
import latentslate_engine.runtime.ltx23_kitchen_managed as managed_module
import latentslate_engine.runtime.ltx23_kitchen_worker as worker_module
from latentslate_engine.runtime.framework.residency.dynamic import (
    DynamicResidencyLease,
    DynamicResidencyPoisoned,
)
from latentslate_engine.runtime.framework.residency.leaf import (
    LeafResidencyDescriptor,
    LeafResidencyScheduler,
)
from latentslate_engine.runtime.framework.worker import persistent_child as persistent_child_module
from latentslate_engine.runtime.ltx23_kitchen_managed import ManagedLTX23KitchenRuntime

_SECRET = bytes(range(32))


def _host_source_registration(
    *,
    budget_bytes: int = 0,
    attempts: int = 0,
    attempt_bytes: int = 0,
    successes: int = 0,
    failures: int = 0,
    failure_bytes: int = 0,
    registered_bytes: int = 0,
    unregistered_bytes: int = 0,
    live_bytes: int = 0,
    peak_bytes: int = 0,
    state_proven: bool = True,
) -> dict[str, object]:
    return {
        "policy": "aimdo_hostbuffer_registered_append",
        "budget_bytes": budget_bytes,
        "attempts": attempts,
        "attempt_bytes": attempt_bytes,
        "successes": successes,
        "failures": failures,
        "failure_bytes": failure_bytes,
        "registered_bytes": registered_bytes,
        "unregistered_bytes": unregistered_bytes,
        "live_bytes": live_bytes,
        "peak_bytes": peak_bytes,
        "state_proven": state_proven,
    }


def _generated_text_leaf_scheduler_proof() -> dict[str, object]:
    """Exercise the real scheduler's force-resident invalidation accounting."""

    class Backend:
        allocation_started = True
        required_virtual_bytes = 1

        def __init__(self) -> None:
            self.groups: dict[str, tuple[object, ...]] = {}
            self.active: set[str] = set()
            self.seen: set[str] = set()
            self.hits = 0
            self.misses = 0
            self.invalidations: list[tuple[str, ...]] = []

        def allocate_group(self, key: object, values: tuple[object, ...]) -> None:
            assert isinstance(key, str)
            self.groups[key] = values

        def prioritize(self) -> None:
            pass

        def acquire(self, key: object) -> DynamicResidencyLease:
            assert isinstance(key, str)
            if key in self.seen:
                self.hits += 1
            else:
                self.seen.add(key)
                self.misses += 1
            self.active.add(key)
            return DynamicResidencyLease(self.groups[key], SimpleNamespace(key=key))

        def prefetch(self, key: object) -> DynamicResidencyLease:
            raise AssertionError(f"text scheduling unexpectedly prefetched {key!r}")

        def wait(self, lease: DynamicResidencyLease) -> None:
            assert lease.token.key in self.active

        def synchronize(self, lease: DynamicResidencyLease) -> None:
            assert lease.token.key in self.active

        def release(self, lease: DynamicResidencyLease) -> None:
            self.active.remove(lease.token.key)

        def invalidate(self, *, reason: str) -> None:
            raise AssertionError(f"text scheduling invalidated all sources: {reason}")

        def invalidate_groups(self, keys: tuple[object, ...], *, reason: str) -> None:
            paths = tuple(str(key) for key in keys)
            self.invalidations.append(paths)
            for path in paths:
                self.seen.discard(path)

        def diagnostics(self) -> dict[str, object]:
            return {}

        def terminal_poison_reason(self) -> None:
            return None

        def close(self) -> None:
            assert not self.active

    schedule = tuple(f"g{index}" for index in range(98))
    descriptors = [
        LeafResidencyDescriptor(f"force{index}", ("g0",), (index,), 8, True)
        for index in range(10)
    ]
    descriptors.extend(
        LeafResidencyDescriptor(
            f"leaf{index}",
            ((f"g{index + 1}",) if index < 40 else (f"g{41 + (index - 40) % 57}",)),
            (index,),
            32_000,
        )
        for index in range(190)
    )
    backend = Backend()
    bindings: dict[str, object] = {}

    def activate(descriptor: LeafResidencyDescriptor, _values: tuple[object, ...]) -> object:
        binding = object()
        bindings[descriptor.path] = binding
        return binding

    def restore(descriptor: LeafResidencyDescriptor, binding: object) -> None:
        assert bindings.pop(descriptor.path) is binding

    scheduler = LeafResidencyScheduler(
        backend,
        descriptors,
        schedule_order=schedule,
        activate=activate,
        restore=restore,
    )
    scheduler.onload()
    for index in range(1, 41):
        scheduler.enter(f"g{index}")
        scheduler.leave(f"g{index}")
    scheduler.invalidate(reason="lora_to_base", paths=("leaf40",))
    for _ in range(87):
        scheduler.enter("g1")
        scheduler.leave("g1")
    scheduler.clear_stage(release_force_resident=True)

    proof = scheduler.diagnostics()
    assert proof["force_resident_waits"] == 20
    assert proof["deferred_waits"] == 127
    assert backend.misses == 50
    assert backend.hits == 97
    assert backend.invalidations == [("leaf40",)]
    assert not bindings
    return proof


def _aimdo_failure_counters(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "backend": "comfy-aimdo",
        "version": "0.4.15",
        "mode": "dynamic_vbar",
        "policy": "required",
        "physical_bytes": 1024,
        "staged_bytes": 1024,
        "virtual_bytes": 1536,
        "allocation_count": 2,
        "live_allocations": 2,
        "live_bytes": 1536,
        "loaded_bytes": None,
        "faults": 3,
        "signature_hits": 1,
        "signature_misses": 2,
        "fault_none_temporaries": 0,
        "pinned_copy_bytes": 512,
        "pageable_copy_bytes": 512,
        "transfer_events": 2,
        "transfer_waits": 2,
        "prioritize_calls": 1,
        "unpin_calls": 0,
        "free_calls": 0,
        "dirty_epoch": 1,
        "lora_invalidations": 1,
        "base_restores": 1,
        "copy_stream_count": 2,
        "copy_strategy": "per_physical",
        "copy_fallback_reason": "host_buffer_capability_unavailable: fixture",
        "gathered_host_buffer_requested": True,
        "host_buffer_capacity_bytes": 0,
        "host_buffer_allocations": 0,
        "host_buffer_unregistrations": 0,
        "host_buffer_frees": 0,
        "host_buffer_live": False,
        "host_tensor_view_live": False,
        "host_buffer_transfer_pending": False,
        "gathered_misses": 0,
        "per_physical_misses": 1,
        "packed_source_bytes": 0,
        "gathered_h2d_bytes": 0,
        "pressure_direct_transfers": 0,
        "pressure_direct_bytes": 0,
        "host_buffer_reuse_barriers": 0,
        "host_source_pool_generation": 0,
        "host_source_pool_lane_count": 0,
        "host_source_pool_capacity_bytes": 0,
        "host_source_pool_retained_slices": 0,
        "host_source_pool_retained_bytes": 0,
        "host_source_pool_temporary_slices": 0,
        "host_source_pool_temporary_bytes": 0,
        "host_source_pool_hits": 0,
        "host_source_pool_misses": 0,
        "host_source_pool_stale_rejections": 0,
        "host_source_pool_warm_ram_pressure_bypasses": 0,
        "host_source_pool_warm_zero_delta_extend_refusals": 0,
        "host_source_pool_warm_registration_refusals": 0,
        "host_source_pool_temporary_ram_pressure_bypasses": 0,
        "host_source_pool_temporary_zero_delta_extend_refusals": 0,
        "host_source_pool_temporary_registration_refusals": 0,
        "host_source_pool_poisoned": False,
        "host_source_pool_poison_reason": None,
        "host_source_registration": _host_source_registration(),
        "base_file_backed": False,
        "base_file_source_live": False,
        "base_file_read_calls": 0,
        "base_file_read_bytes": 0,
        "base_file_handle_live": False,
        "base_file_handle_opened": 0,
        "base_file_handle_closed": 0,
        "base_file_fallback_reason": None,
        "refill_failure_reason": None,
        "refill_target_bytes": None,
        "refill_root_already_bound": None,
        "refill_resident_bytes": None,
        "poisoned": True,
        "close_failed": True,
        "poison_reason": "device_quiescence_failed",
    }
    value.update(overrides)
    return value


class _Request:
    operation = "ltx23_dev_t2v"
    fingerprint = "request-fingerprint"
    component_fingerprint = "component-fingerprint"

    def to_json_dict(self):
        return {
            "schema_version": 1,
            "family": "ltx23",
            "operation": self.operation,
            "base_model": "Lightricks/LTX-2.3",
            "execution_contract": {
                "workflow_revision": "2b7f823136606344f0bccce249898d771b809aa1",
                "workflow_sha256": (
                    "75b10f3ee48c1fe00c7fb21b24c0c247b133e5ee34676144de4b652ac7dcbe7f"
                ),
                "node_semantics_revision": "b78cec879b9460d5cb25228a83a942fb78d2cd24",
                "kitchen_revision": "7c6ca3a5b63857d42c2d49777d6afb69de23f13f",
                "engine_parity_revision": "ltx23-comfy-baseline-v2",
                "pinned_workflow_default_width": 1280,
                "pinned_workflow_default_height": 720,
                "engine_acceptance_default_width": 1280,
                "engine_acceptance_default_height": 704,
                "dimension_alignment": "dev=/64;distilled_flf=/32",
                "audio_duration_contract": "source_derived_exact_duration_v1",
            },
            "component_fingerprint": self.component_fingerprint,
            "fingerprint": self.fingerprint,
            "components": {"pipeline_support": {"path": "not-used-before-binding"}},
        }

    def public_component_manifest(self):
        return {"pipeline_support": {"component": "pipeline_support"}}


def _memory_telemetry(operation: str = "ltx23_dev_t2v") -> dict[str, object]:
    phases = (
        (
            "after_text_offload",
            "after_main_denoise",
            "after_decode",
            "after_transient_clearing",
            "after_prompt_cache_publication",
        )
        if operation == "ltx23_distilled_flf"
        else (
            "after_text_offload",
            "after_stage1",
            "after_latent_upscaling",
            "after_stage2",
            "after_decode",
            "after_transient_clearing",
            "after_prompt_cache_publication",
        )
    )
    return {
        "schema_version": 1,
        "timestamp_clock": "time.time_ns",
        "elapsed_clock": "time.perf_counter_ns",
        "samples": [
            {
                "sequence": sequence,
                "phase": phase,
                "timestamp_unix_ns": 1_700_000_000_000_000_000 + sequence,
                "elapsed_ns": sequence * 10,
                "process": {
                    "status": "ok",
                    "error": None,
                    "pid": 42,
                    "private_bytes": 10_000 + sequence,
                    "working_set_bytes": 8_000 + sequence,
                },
                "system": {
                    "status": "ok",
                    "error": None,
                    "total_physical_bytes": 100_000,
                    "free_physical_bytes": 40_000 - sequence,
                    "used_physical_bytes": 60_000 + sequence,
                },
                "cuda": {
                    "status": "ok",
                    "error": None,
                    "device": "cuda:0",
                    "allocated_bytes": 2_000 + sequence,
                    "reserved_bytes": 3_000 + sequence,
                    "free_bytes": 6_000 - sequence,
                    "total_bytes": 10_000,
                },
            }
            for sequence, phase in enumerate(phases)
        ],
    }


def _metadata(
    request: _Request,
    *,
    seed: int = 7,
    frames: int = 25,
    requested_frames: int | None = None,
) -> dict[str, object]:
    requested_frames = frames if requested_frames is None else requested_frames
    audio_latent_frames = frames
    decoded_mel_frames = audio_latent_frames * 4 - 3
    decoded_samples = decoded_mel_frames * 160 * 48_000 // 16_000
    target_samples = frames * 48_000 // 25
    return {
        "family": "ltx23",
        "runtime": "engine-native/ltx23-kitchen",
        "operation": request.operation,
        "request_fingerprint": request.fingerprint,
        "component_fingerprint": request.component_fingerprint,
        "seed": seed,
        "width": 768,
        "height": 512,
        "num_frames": frames,
        "requested_num_frames": requested_frames,
        "fps": 25,
        "audio_sample_rate": 48_000,
        "audio_channels": 2,
        "container_format": "mov,mp4,m4a,3gp,3g2,mj2",
        "video_codec": "h264",
        "audio_codec": "aac",
        "audio_samples": target_samples,
        "video_duration_seconds": frames / 25,
        "audio_duration_seconds": target_samples / 48_000,
        "requested_duration_seconds": frames / 25,
        "effective_duration_seconds": frames / 25,
        "audio_duration_normalization": {
            "policy": "source_derived_exact_duration_v1",
            "reason": "independent_audio_grid_causal_tail",
            "video_frames": frames,
            "audio_latent_frames": audio_latent_frames,
            "expected_audio_latent_frames": audio_latent_frames,
            "audio_latent_channels": 8,
            "audio_latent_mel_bins": 16,
            "decoded_mel_frames": decoded_mel_frames,
            "expected_mel_frames": decoded_mel_frames,
            "decoded_mel_channels": 2,
            "decoded_mel_bins": 64,
            "decoded_samples": decoded_samples,
            "expected_decoded_samples": decoded_samples,
            "target_samples": target_samples,
            "fps": 25,
            "audio_channels": 2,
            "source_sample_rate": 16_000,
            "output_sample_rate": 48_000,
            "mel_hop_length": 160,
            "temporal_compression_ratio": 4,
            "causality_axis": "height",
            "is_causal": True,
            "trimmed_samples": 0,
            "padded_samples": target_samples - decoded_samples,
        },
        "output_sha256": "not-read-in-this-mocked-result",
        "components": request.public_component_manifest(),
        "prompt_enhanced": True,
        "prompt_enhancement_system_sha256": (
            "f00b22f47dad68358f5c2c7396c701db95095cf26dc3dbd6b5556eab04692071"
        ),
        "prompt_enhancement_seed": 0,
        "prompt_enhancement_max_new_tokens": 2048,
        "prompt_enhancement_stop_token_id": 106,
        "prompt_enhancement_template": "comfy_ltx2_gemma3_manual_v1",
        "prompt_enhancement_generation_settings": {
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 64,
            "top_p": 0.95,
            "min_p": 0.05,
            "repetition_penalty": 1.05,
        },
        "prompt_enhancement_memory": {
            "policy": "release_after_prompt_enhancement",
            "cache_present": False,
            "cache_type": None,
            "cuda_allocated_before_bytes": None,
            "cuda_allocated_after_bytes": None,
            "cuda_allocated_released_bytes": 0,
            "template": "comfy_ltx2_gemma3_manual_v1",
            "stop_token_id": 106,
            "generation_settings": {
                "do_sample": True,
                "temperature": 0.7,
                "top_k": 64,
                "top_p": 0.95,
                "min_p": 0.05,
                "repetition_penalty": 1.05,
            },
            "decoded_suffix_nonempty": True,
            "think_block_removed": False,
            "fallback_to_source_prompt": False,
        },
        "native_fp8": {
            "complete": True,
            "modules": 2,
            "dispatched_modules": 2,
            "native_dispatch_count": 4,
            "dense_fallback_count": 0,
        },
        "native_text": {
            "backend": "engine-native/comfy-strict-full-precision-mm",
            "policy": "full_precision_mm",
            "module_count": 2,
            "total_dispatches": 4,
            "minimum_module_dispatches": 1,
            "maximum_module_dispatches": 3,
            "native_quantized_dispatches": 0,
            "rejected_dispatches": 0,
            "dense_fallback_dispatches": 0,
        },
        "text_lora": {
            "backend": "engine-native/additive-lora",
            "policy": "prompt_enhancement_only",
            "target_module_count": 2,
            "total_dispatches": 4,
            "minimum_target_dispatches": 1,
            "maximum_target_dispatches": 3,
        },
        "negative_prompt": "pc game, console game, video game, cartoon, childish, ugly",
        "negative_encoding": {
            "prompt_sha256": "4f5c6b421252cd5565bd8305231eecdf1e36343bc7e52ec68a5608656b8df273",
            "max_sequence_length": 1024,
            "dtype": "bfloat16",
            "mask_dtype": "int64",
            "finite": True,
            "encoded": True,
            "used_for_cfg": False,
            "embeds_shape": [1, 1024, 188160],
            "mask_shape": [1, 1024],
        },
        "text_patch_state": {
            "policy": "prompt_enhancement_only",
            "lora_strength_enhancement": 1.0,
            "lora_strength_positive": 0.0,
            "lora_strength_negative": 0.0,
            "lora_to_base_transitions": 1,
            "restored_base_on_exit": True,
        },
        "text_residency": {
            "mode": "layer_streamed_cpu_master",
            "root_activation": "stage_onload",
            "layer_count": 48,
            "root_weight_bytes": 100,
            "largest_layer_weight_bytes": 200,
            "required_weight_bytes": 300,
            "root_transitions": 1,
            "layer_transitions": 144,
            "execution_policy": "strict_comfy_full_precision_mm",
            "native_quantized_dispatches": 0,
            "full_precision_dispatches": 12,
            "transfer_mode": "two_stream_nonblocking",
            "transfer_stream_count": 2,
            "transfer_events": 145,
            "transfer_waits": 145,
            "async_transfer_fallbacks": 0,
            "strict_cuda_parity": True,
            "host_registration": {
                "policy": "comfy_best_effort_in_place_cuda_host_register",
                "lifecycle": "text_stage_onload_through_synchronized_offload",
                "budget_bytes": 16_000,
                "candidates": 100,
                "candidate_bytes": 10_000,
                "deduplicated_aliases": 2,
                "already_registered": 5,
                "already_registered_bytes": 500,
                "attempts": 80,
                "attempt_bytes": 8_000,
                "successes": 60,
                "registered_bytes": 6_000,
                "failures": 20,
                "failure_bytes": 2_000,
                "ineligible": 13,
                "ineligible_bytes": 1_300,
                "unregistered": 60,
                "unregistered_bytes": 6_000,
                "unregister_failures": 0,
                "unregister_failure_bytes": 0,
                "owned_active": 0,
                "owned_active_bytes": 0,
                "categories": {
                    "unsupported_type": 1,
                    "non_cpu": 0,
                    "noncontiguous": 2,
                    "zero_pointer": 0,
                    "budget_exceeded": 10,
                    "eligibility_error": 0,
                    "register_error": 20,
                    "unregister_error": 0,
                },
            },
            "layer_compute_barriers": 144,
            "live_layer_bindings": 0,
            "live_layer_bytes": 0,
            "maximum_live_layer_bindings": 1,
            "maximum_live_layer_bytes": 200,
            "dynamic_vbar_prefetch": False,
            "leaf_allocation_count": 200,
            "force_resident_leaf_count": 10,
            "base_leaf_count": 150,
            "patch_leaf_count": 50,
            "schedule_group_count": 98,
            "leaf_scheduler": None,
            "warm_request_index": 1,
            "dynamic_vram": {
                "backend": "engine_hooks",
                "policy": "auto",
                "fallback_reason": "fixture: AIMDO unavailable before allocation",
                "prefetch": False,
                "allocator_plugin": False,
                "base_file_requested": True,
                "base_file_backed": False,
                "base_file_read_calls": 0,
                "base_file_read_bytes": 0,
                "base_file_handle_live": False,
                "base_file_handle_opened": 0,
                "base_file_handle_closed": 0,
                "base_file_fallback_reason": ("aimdo_backend_unavailable: fixture"),
            },
        },
        "timings": {
            "clock": "time.perf_counter",
            "unit": "seconds",
            "phases": {
                "materialization": 1.0,
                "text_onload": 0.5,
                "enhancement": 2.0,
                "positive_encode": 3.0,
                "negative_encode": 4.0,
                "text_offload": 5.0,
                "downstream": 6.0,
                "prompt_cache_publish": 0.0,
            },
            "cumulative": {
                "materialization": 1.0,
                "text_onload": 1.5,
                "enhancement": 3.5,
                "positive_encode": 6.5,
                "negative_encode": 10.5,
                "text_offload": 15.5,
                "downstream": 21.5,
                "prompt_cache_publish": 21.5,
            },
            "total_seconds": 21.5,
        },
        "memory_telemetry": _memory_telemetry(request.operation),
        "cache": {
            "pipeline_warm": False,
            "policy": "none",
            "prompt_hit": False,
            "prompt_published": False,
            "media_hit": False,
            "prompt": {
                "name": "prompt",
                "enabled": False,
                "entries": 0,
                "bytes": 0,
                "max_bytes": 1024 * 1024**2,
                "max_entries": 8,
                "hits": 0,
                "misses": 1,
                "evictions": 0,
                "hit_rate": 0.0,
            },
        },
        "dense_base_dequantizations": 0,
        "residency_policy": {
            "mode": "leaf_dynamic",
            "free_bytes": 16_000,
            "total_bytes": 16_000,
            "stored_bytes": 10_000,
            "reserved_headroom_bytes": 4_000,
            "resident_weight_budget_bytes": 5_000,
            "reason": "test",
            "root_bytes": 1_000,
            "resident_block_count": 0,
            "resident_block_bytes": 0,
            "streamed_block_count": 48,
            "streamed_block_bytes": 9_000,
            "streaming": "leaf_prefetch_aimdo_file_backed",
            "streamed_transitions": 49,
            "resident_refills": 0,
            "dynamic_acquires": 200,
            "dynamic_releases": 190,
            "group_count": 49,
            "root_group_count": 1,
            "layer_group_count": 48,
            "leaf_allocation_count": 200,
            "force_resident_leaf_count": 10,
            "prefetch_groups": 48,
            "prefetch_leaves": 190,
            "deferred_waits": 190,
            "force_resident_waits": 10,
            "prefetch": True,
            "base_file_backed": True,
            "base_file_bytes": 8_000,
            "base_file_handle_live": True,
            "base_file_handle_opened": 1,
            "base_file_handle_closed": 0,
            "cpu_source_bytes_base": 0,
            "cpu_source_bytes_lora_mutable": 2_000,
            "dynamic_vram": {
                "backend": "comfy-aimdo",
                "version": "0.4.15",
                "mode": "dynamic_vbar",
                "physical_bytes": 10_000,
                "staged_bytes": 200_000,
                "virtual_bytes": 200_000,
                "allocation_count": 200,
                "live_allocations": 200,
                "live_bytes": 200_000,
                "loaded_bytes": 10_000,
                "faults": 200,
                "signature_hits": 0,
                "signature_misses": 200,
                "fault_none_temporaries": 0,
                "pinned_copy_bytes": 200_000,
                "pageable_copy_bytes": 0,
                "transfer_events": 200,
                "transfer_waits": 200,
                "prioritize_calls": 1,
                "unpin_calls": 190,
                "free_calls": 0,
                "dirty_epoch": 0,
                "lora_invalidations": 0,
                "base_restores": 0,
                "copy_stream_count": 2,
                "copy_strategy": "gathered_host_buffer",
                "copy_fallback_reason": None,
                "gathered_host_buffer_requested": True,
                "host_buffer_capacity_bytes": 1_000,
                "host_buffer_allocations": 4,
                "host_buffer_unregistrations": 0,
                "host_buffer_frees": 0,
                "host_buffer_live": True,
                "host_tensor_view_live": True,
                "host_buffer_transfer_pending": False,
                "gathered_misses": 200,
                "per_physical_misses": 0,
                "packed_source_bytes": 150_000,
                "gathered_h2d_bytes": 200_000,
                "pressure_direct_transfers": 0,
                "pressure_direct_bytes": 0,
                "host_buffer_reuse_barriers": 0,
                "host_source_pool_generation": 1,
                "host_source_pool_lane_count": 4,
                "host_source_pool_capacity_bytes": 400_000,
                "host_source_pool_retained_slices": 200,
                "host_source_pool_retained_bytes": 200_000,
                "host_source_pool_temporary_slices": 0,
                "host_source_pool_temporary_bytes": 0,
                "host_source_pool_hits": 0,
                "host_source_pool_misses": 200,
                "host_source_pool_stale_rejections": 0,
                "host_source_pool_warm_ram_pressure_bypasses": 0,
                "host_source_pool_warm_zero_delta_extend_refusals": 0,
                "host_source_pool_warm_registration_refusals": 0,
        "host_source_pool_temporary_ram_pressure_bypasses": 0,
        "host_source_pool_temporary_zero_delta_extend_refusals": 0,
        "host_source_pool_temporary_registration_refusals": 0,
                "host_source_pool_poisoned": False,
                "host_source_pool_poison_reason": None,
                "host_source_registration": _host_source_registration(
                    budget_bytes=400_000,
                    attempts=200,
                    attempt_bytes=200_000,
                    successes=200,
                    registered_bytes=200_000,
                    live_bytes=200_000,
                    peak_bytes=200_000,
                ),
                "prefetch": True,
                "prefetch_calls": 190,
                "allocator_plugin": False,
                "poisoned": False,
                "close_failed": False,
                "poison_reason": None,
                "host_registration": {
                    "policy": "comfy_best_effort_in_place_cuda_host_register",
                    "lifecycle": "residency_stage_through_synchronized_close",
                    "budget_bytes": 16_000,
                    "candidates": 0,
                    "candidate_bytes": 0,
                    "deduplicated_aliases": 0,
                    "already_registered": 0,
                    "already_registered_bytes": 0,
                    "attempts": 0,
                    "attempt_bytes": 0,
                    "successes": 0,
                    "registered_bytes": 0,
                    "failures": 0,
                    "failure_bytes": 0,
                    "ineligible": 0,
                    "ineligible_bytes": 0,
                    "unregistered": 0,
                    "unregistered_bytes": 0,
                    "unregister_failures": 0,
                    "unregister_failure_bytes": 0,
                    "owned_active": 0,
                    "owned_active_bytes": 0,
                    "categories": {
                        "unsupported_type": 0,
                        "non_cpu": 0,
                        "noncontiguous": 0,
                        "zero_pointer": 0,
                        "budget_exceeded": 0,
                        "eligibility_error": 0,
                        "register_error": 0,
                        "unregister_error": 0,
                    },
                },
                "base_file_backed": True,
                "base_file_source_live": True,
                "base_file_read_calls": 100,
                "base_file_read_bytes": 18_000,
            },
        },
    }


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        name: tmp_path / name
        for name in (
            "request",
            "result",
            "progress",
            "heartbeat",
            "gate",
            "command",
            "cancel",
        )
    }


def _bound_generation(*, frames: int = 25, seed: int = 7) -> dict[str, object]:
    return {
        "seed": seed,
        "width": 768,
        "height": 512,
        "requested_num_frames": frames,
        "num_frames": frames,
        "duration_seconds": frames / 25,
    }


def _cleared_cache_status(*, policy: str = "prompt") -> dict[str, object]:
    return {
        "pipeline_warm": True,
        "policy": policy,
        "prompt_hit": False,
        "prompt_published": False,
        "media_hit": False,
        "prompt": {
            "name": "prompt",
            "enabled": policy == "prompt",
            "entries": 0,
            "bytes": 0,
            "max_bytes": 1024 * 1024**2,
            "max_entries": 8,
            "hits": 1 if policy == "prompt" else 0,
            "misses": 1,
            "evictions": 0,
            "hit_rate": 0.5 if policy == "prompt" else 0.0,
        },
    }


class _FakeProcess:
    def __init__(self, pid: int = 42) -> None:
        self.pid = pid
        self.exit_code: int | None = None

    def poll(self):
        return self.exit_code


class _FakeSupervisor:
    def __init__(
        self,
        paths: dict[str, Path],
        *,
        pid: int = 42,
        events: list[str] | None = None,
        start_error: BaseException | None = None,
        wait_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.paths = managed_module._persistent_paths(paths)
        self.process = _FakeProcess(pid)
        self.session = None
        self.failed_start = None
        self.events = [] if events is None else events
        self.start_error = start_error
        self.wait_error = wait_error
        self.close_error = close_error
        self.commands: list[object] = []

    def start(self, payload):
        self.events.append("start")
        if self.start_error is not None:
            raise self.start_error
        self.session = SimpleNamespace(process=self.process)
        self.commands.append(payload)
        self.paths.start_gate.touch(exist_ok=True)
        return self.session

    def send(self, payload):
        self.events.append("send")
        self.commands.append(payload)

    def wait(self, **_kwargs):
        self.events.append("wait")
        if self.wait_error is not None:
            raise self.wait_error

    def terminate(self):
        self.events.append("terminate")
        self.process.exit_code = 1

    def close(self):
        self.events.append("close")
        self.session = None
        if self.close_error is not None:
            raise self.close_error

    def cleanup_job(self):
        self.events.append("cleanup_job")
        for path in (
            self.paths.command,
            self.paths.result,
            self.paths.progress,
            self.paths.heartbeat,
        ):
            path.unlink(missing_ok=True)
        return []

    def cleanup_session(self):
        self.events.append("cleanup_session")
        for path in _paths(self.paths.request.parent).values():
            path.unlink(missing_ok=True)
        return []


@pytest.mark.parametrize("frames", [25, 33, 41, 121, 129])
def test_managed_accepts_source_derived_audio_duration_closure(frames: int) -> None:
    request = _Request()
    managed_module._validate_metadata(
        _metadata(request, frames=frames),
        request,  # type: ignore[arg-type]
        _bound_generation(frames=frames),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy", "arbitrary"),
        ("reason", "arbitrary"),
        ("audio_latent_frames", 27),
        ("decoded_mel_frames", 102),
        ("expected_mel_frames", 102),
        ("decoded_samples", 48_481),
        ("expected_decoded_samples", 48_481),
        ("target_samples", 50_001),
        ("video_frames", 33),
        ("fps", 23),
        ("audio_channels", 1),
        ("source_sample_rate", 16_001),
        ("source_sample_rate", 0),
        ("output_sample_rate", 47_999),
        ("mel_hop_length", 161),
        ("temporal_compression_ratio", 5),
        ("trimmed_samples", 1),
        ("padded_samples", 1_519),
        ("decoded_samples", True),
    ],
)
def test_managed_rejects_audio_duration_metadata_tamper(field: str, value: object) -> None:
    request = _Request()
    metadata = _metadata(request)
    metadata["audio_duration_normalization"][field] = value  # type: ignore[index]
    with pytest.raises(RuntimeError, match="audio duration normalization"):
        managed_module._validate_metadata(
            metadata,
            request,  # type: ignore[arg-type]
            _bound_generation(),
        )


@pytest.mark.parametrize("audio_latent_frames", [25, 27])
def test_managed_rejects_coherent_wrong_shorter_or_longer_audio_lattice(
    audio_latent_frames: int,
) -> None:
    request = _Request()
    metadata = _metadata(request)
    normalization = metadata["audio_duration_normalization"]
    mel_frames = audio_latent_frames * 4 - 3
    decoded_samples = mel_frames * 480
    normalization.update(  # type: ignore[union-attr]
        {
            "audio_latent_frames": audio_latent_frames,
            "expected_audio_latent_frames": audio_latent_frames,
            "decoded_mel_frames": mel_frames,
            "expected_mel_frames": mel_frames,
            "decoded_samples": decoded_samples,
            "expected_decoded_samples": decoded_samples,
            "trimmed_samples": 0,
            "padded_samples": 50_000 - decoded_samples,
        }
    )
    with pytest.raises(RuntimeError, match="audio duration normalization"):
        managed_module._validate_metadata(
            metadata,
            request,  # type: ignore[arg-type]
            _bound_generation(),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("video_duration_seconds", float("nan")),
        ("video_duration_seconds", float("inf")),
        ("video_duration_seconds", True),
        ("audio_duration_seconds", float("nan")),
        ("audio_duration_seconds", float("-inf")),
        ("audio_duration_seconds", False),
        ("audio_samples", True),
        ("audio_samples", 49_999),
        ("audio_samples", 51_024),
    ],
)
def test_managed_rejects_nonfinite_boolean_or_out_of_packet_media_arithmetic(
    field: str, value: object
) -> None:
    request = _Request()
    metadata = _metadata(request)
    metadata[field] = value
    if field == "audio_samples" and isinstance(value, int) and not isinstance(value, bool):
        metadata["audio_duration_seconds"] = value / 48_000
    with pytest.raises(RuntimeError, match="media provenance"):
        managed_module._validate_metadata(
            metadata,
            request,  # type: ignore[arg-type]
            _bound_generation(),
        )


def test_worker_rejects_tamper_before_recipe_rehydration_or_heavy_import(
    tmp_path: Path, monkeypatch
):
    request = _Request()
    payload = managed_module._payload(
        request,
        {
            "prompt": "scene",
            "width": 768,
            "height": 512,
            "duration_seconds": 1.0,
            "num_frames": 25,
            "seed": 7,
            "start_image_path": None,
            "end_image_path": None,
            "output_path": str(tmp_path / "output.mp4"),
        },
        device="cuda",
        cache_policy="none",
        secret=_SECRET,
    )
    tampered = copy.deepcopy(payload)
    tampered["generation"]["output_path"] = "C:/private/other.mp4"
    monkeypatch.setattr(
        worker_module,
        "_validate_generation_json",
        lambda _value: (_ for _ in ()).throw(AssertionError("parsed before binding")),
    )
    with pytest.raises(ValueError, match="binding"):
        worker_module._validate_bound_payload(tampered, _SECRET)


def test_worker_rejects_wrong_session_secret_before_generation_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation = {
        "prompt": "scene",
        "width": 768,
        "height": 512,
        "duration_seconds": 1.0,
        "requested_num_frames": 26,
        "num_frames": 25,
        "seed": 7,
        "start_image_path": None,
        "end_image_path": None,
        "start_image_identity": None,
        "end_image_identity": None,
        "output_path": str(tmp_path / "output.mp4"),
    }
    payload = managed_module._payload(
        _Request(), generation, device="cuda", cache_policy="none", secret=_SECRET
    )
    monkeypatch.setattr(
        worker_module,
        "_validate_generation_json",
        lambda _value: (_ for _ in ()).throw(AssertionError("parsed before authentication")),
    )
    with pytest.raises(ValueError, match="binding"):
        worker_module._validate_bound_payload(payload, b"x" * 32)


def _success_result(output: Path) -> dict[str, object]:
    metadata = _metadata(_Request())
    metadata["output_sha256"] = "3" * 64
    return {
        "schema_version": 2,
        "ok": True,
        "request_binding": "binding",
        "output_path": str(output),
        "output_size_bytes": 3,
        "metadata": metadata,
        "allocator_policy": "expandable_segments:True",
    }


@pytest.mark.parametrize("tamper", ["path", "size", "hash", "timing"])
def test_result_envelope_tamper_is_rejected_before_worker_content_consumption(
    tmp_path: Path, tamper: str
) -> None:
    output = tmp_path / "output.mp4"
    output.write_bytes(b"mp4")
    value = worker_module._signed_result(_success_result(output), _SECRET)
    if tamper == "path":
        value["output_path"] = str(tmp_path / "private-other.mp4")
    elif tamper == "size":
        value["output_size_bytes"] = 4
    elif tamper == "hash":
        value["metadata"]["output_sha256"] = "0" * 64  # type: ignore[index]
    else:
        value["metadata"]["audio_duration_normalization"]["target_samples"] += 1  # type: ignore[index,operator]
    result = tmp_path / "result.json"
    result.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not bind to its command"):
        managed_module._read_result(result, output, "binding", _SECRET)


def test_result_envelope_rejects_wrong_session_secret(tmp_path: Path) -> None:
    output = tmp_path / "output.mp4"
    output.write_bytes(b"mp4")
    value = worker_module._signed_result(_success_result(output), _SECRET)
    result = tmp_path / "result.json"
    result.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not bind to its command"):
        managed_module._read_result(result, output, "binding", b"x" * 32)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_authenticated_success_result_requires_exact_schema(tmp_path: Path, mutation: str) -> None:
    output = tmp_path / "output.mp4"
    output.write_bytes(b"mp4")
    value = _success_result(output)
    if mutation == "missing":
        value.pop("allocator_policy")
    else:
        value["unexpected"] = "field"
    signed = worker_module._signed_result(value, _SECRET)
    result = tmp_path / "result.json"
    result.write_text(json.dumps(signed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid success result"):
        managed_module._read_result(result, output, "binding", _SECRET)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_authenticated_failure_result_requires_exact_schema(tmp_path: Path, mutation: str) -> None:
    value: dict[str, object] = {
        "schema_version": 2,
        "ok": False,
        "request_binding": "binding",
        "error_type": "TypeError",
        "error": "private",
        "failure_stage": "materialize_text_encoder",
        "error_fingerprint": "a" * 64,
        "failure_location": "ltx23_kitchen_text.load",
        "cleanup_stage": None,
        "aimdo_counters": None,
    }
    if mutation == "missing":
        value.pop("error_fingerprint")
    else:
        value["unexpected"] = "field"
    signed = worker_module._signed_result(value, _SECRET)
    result = tmp_path / "result.json"
    result.write_text(json.dumps(signed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="failure result is invalid"):
        managed_module._read_result(result, tmp_path / "unused.mp4", "binding", _SECRET)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("terminal_exit_code", 1),
        ("poison_reason", "bad reason"),
        ("poison_reason", ""),
        ("poison_origin", "cleanup"),
        ("poison_origin", "arbitrary"),
    ],
)
def test_authenticated_poison_failure_requires_exact_terminal_contract(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    handler = worker_module._LTX23KitchenHandler(_SECRET)
    handler.failure.binding = "binding"
    exc = kitchen_module.LTX23KitchenWorkerPoisoned("failed_fill_quiescence_failed")
    value = dict(handler.failure_result(exc, SimpleNamespace(binding="binding")))
    result = tmp_path / "result.json"
    result.write_text(json.dumps(value), encoding="utf-8")
    accepted = managed_module._read_result(result, tmp_path / "unused.mp4", "binding", _SECRET)
    assert accepted["terminal_exit_code"] == 86
    assert accepted["poison_reason"] == "failed_fill_quiescence_failed"
    assert accepted["poison_origin"] == "primary"

    unsigned = {key: item for key, item in value.items() if key != "result_binding"}
    unsigned[field] = replacement
    tampered = worker_module._signed_result(unsigned, _SECRET)
    result.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="failure result is invalid"):
        managed_module._read_result(result, tmp_path / "unused.mp4", "binding", _SECRET)

    stripped = {
        key: item
        for key, item in unsigned.items()
        if key not in {"terminal_exit_code", "poison_reason", "poison_origin"}
    }
    result.write_text(json.dumps(worker_module._signed_result(stripped, _SECRET)), encoding="utf-8")
    with pytest.raises(RuntimeError, match="failure result is invalid"):
        managed_module._read_result(result, tmp_path / "unused.mp4", "binding", _SECRET)


def test_authenticated_poison_origin_is_bidirectionally_bound(
    tmp_path: Path,
) -> None:
    result = tmp_path / "result.json"
    output = tmp_path / "unused.mp4"
    handler = worker_module._LTX23KitchenHandler(_SECRET)
    handler.failure.binding = "binding"
    primary = dict(
        handler.failure_result(
            kitchen_module.LTX23KitchenWorkerPoisoned("device_quiescence_failed"),
            SimpleNamespace(binding="binding"),
        )
    )

    def assert_rejected(value: dict[str, object]) -> None:
        unsigned = {key: item for key, item in value.items() if key != "result_binding"}
        result.write_text(
            json.dumps(worker_module._signed_result(unsigned, _SECRET)),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="failure result is invalid"):
            managed_module._read_result(result, output, "binding", _SECRET)

    assert primary["poison_origin"] == "primary"
    assert_rejected({**primary, "cleanup_stage": "unload_runtime"})
    assert_rejected(
        {
            **primary,
            "poison_origin": "cleanup",
            "cleanup_stage": "unload_runtime",
        }
    )

    cleanup_unsigned: dict[str, object] = {
        "schema_version": 2,
        "ok": False,
        "request_binding": "binding",
        "error_type": "RuntimeError",
        "error": "ordinary primary",
        "failure_stage": "verify_output",
        "error_fingerprint": "a" * 64,
        "failure_location": "ltx23_kitchen_worker.execute",
        "cleanup_stage": "unload_runtime",
        "aimdo_counters": None,
        "terminal_exit_code": 86,
        "poison_reason": "device_quiescence_failed",
        "poison_origin": "cleanup",
    }
    cleanup = worker_module._signed_result(cleanup_unsigned, _SECRET)
    result.write_text(json.dumps(cleanup), encoding="utf-8")
    accepted = managed_module._read_result(result, output, "binding", _SECRET)
    assert accepted["error_type"] == "RuntimeError"
    assert accepted["poison_origin"] == "cleanup"

    assert_rejected({**cleanup, "error_type": "LTX23KitchenWorkerPoisoned"})
    assert_rejected({**cleanup, "poison_origin": "primary"})
    assert_rejected(
        {
            **cleanup,
            "error_type": "LTX23KitchenWorkerPoisoned",
            "poison_origin": "primary",
        }
    )


@pytest.mark.parametrize(
    "reason",
    (
        "device_quiescence_failed",
        "failed_fill_quiescence_failed",
        "host_source_pool_structural_failure",
        "host_source_pool_setup_cleanup_failed",
        "ltx23_av_dynamic_initialization_cleanup_failed",
    ),
)
def test_worker_and_managed_accept_only_canonical_terminal_poison_reasons(
    reason: str,
) -> None:
    exc = kitchen_module.LTX23KitchenWorkerPoisoned(reason)
    assert worker_module._poison_reason(exc) == reason
    assert reason in managed_module._AIMDO_POISON_REASONS


def test_worker_terminal_status_rejects_unknown_descriptive_runtime_reason() -> None:
    runtime = SimpleNamespace(
        terminal_poison_reason=lambda: "AIMDO arbitrary cleanup failed: private detail"
    )
    session = SimpleNamespace(runtime=runtime)
    handler = worker_module._LTX23KitchenHandler(_SECRET)

    assert (
        handler.terminal_exit_status(RuntimeError("ordinary failure"), session, SimpleNamespace())
        is None
    )

    canonical = DynamicResidencyPoisoned("device_quiescence_failed")
    wrapped = RuntimeError("descriptive transformer wrapper")
    wrapped.__cause__ = canonical
    terminal = handler.terminal_exit_status(wrapped, session, SimpleNamespace())
    assert terminal == persistent_child_module.PersistentChildTerminalExit(
        86, "device_quiescence_failed"
    )

    canonical_runtime = SimpleNamespace(
        terminal_poison_reason=lambda: "failed_fill_quiescence_failed"
    )
    canonical_session = SimpleNamespace(runtime=canonical_runtime)
    ordinary = RuntimeError("ordinary wrapper after runtime poison")
    terminal = handler.terminal_exit_status(ordinary, canonical_session, SimpleNamespace())
    assert terminal is None


def test_worker_announces_start_before_runtime_import(monkeypatch) -> None:
    events: list[tuple[float, str | None]] = []
    context = SimpleNamespace(
        publish_progress=lambda value, message: events.append((value, message)),
        binding="",
    )
    handler = worker_module._LTX23KitchenHandler(_SECRET)
    monkeypatch.setattr(
        handler,
        "_bind",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("stop before import")),
    )
    with pytest.raises(RuntimeError, match="stop before import"):
        handler.bind_initial({}, context)
    assert events == [(0.001, "LTX worker started")]


def test_worker_progress_log_phase_is_safe_and_specific() -> None:
    assert (
        managed_module._worker_progress_phase("Validating LTX worker request")
        == "validate_bound_request"
    )
    assert managed_module._worker_progress_phase("Importing LTX runtime") == "import_runtime"
    assert (
        managed_module._worker_progress_phase("Building LTX transformer shell")
        == "build_transformer_shell"
    )
    assert managed_module._worker_progress_phase("LTX denoise step 3/9") == "denoise"
    assert (
        managed_module._worker_progress_phase("LTX AIMDO first_acquire_after_fault")
        == "aimdo_text_residency"
    )
    assert managed_module._worker_progress_phase("prompt: private scene") == "working"
    assert (
        worker_module._progress_stage("Inspecting LTX transformer artifact")
        == "inspect_transformer"
    )
    assert (
        worker_module._progress_stage("Building LTX transformer shell") == "build_transformer_shell"
    )
    assert (
        worker_module._progress_stage("Planning LTX transformer materialization")
        == "plan_transformer_materialization"
    )


def test_result_failure_is_detected_before_worker_process_exits(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    value = worker_module._signed_result(
        {
            "schema_version": 2,
            "ok": False,
            "request_binding": "binding",
            "error_type": "TypeError",
            "error": "private child detail",
            "failure_stage": "materialize_text_encoder",
            "error_fingerprint": "a" * 64,
            "failure_location": "ltx23_kitchen_text.load",
            "cleanup_stage": None,
            "aimdo_counters": None,
        },
        _SECRET,
    )
    result.write_text(json.dumps(value), encoding="utf-8")
    failure = managed_module._worker_failure(result, 1, "binding", _SECRET)
    assert failure["stage"] == "materialize_text_encoder"
    assert "private child detail" not in failure["message"]


def test_worker_rejects_guide_content_changed_after_parent_binding(tmp_path: Path) -> None:
    guide = tmp_path / "guide.png"
    guide.write_bytes(b"first")
    generation = managed_module._generation(
        "ltx23_dev_i2v",
        prompt="scene",
        width=768,
        height=512,
        duration_seconds=1.0,
        num_frames=25,
        seed=7,
        start_image_path=guide,
        end_image_path=None,
        output_path=tmp_path / "output.mp4",
    )
    payload = managed_module._payload(
        _Request(), generation, device="cuda", cache_policy="none", secret=_SECRET
    )
    guide.write_bytes(b"second")

    with pytest.raises(ValueError, match="endpoint changed"):
        worker_module._validate_bound_payload(payload, _SECRET)


def test_supervisor_recomputes_published_output_hash(tmp_path: Path) -> None:
    output = tmp_path / "output.mp4"
    output.write_bytes(b"mp4")
    result = tmp_path / "result.json"
    value = worker_module._signed_result(
        {
            "schema_version": 2,
            "ok": True,
            "request_binding": "binding",
            "output_path": str(output),
            "output_size_bytes": 3,
            "metadata": {"output_sha256": "0" * 64},
            "allocator_policy": "expandable_segments:True",
        },
        _SECRET,
    )
    result.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(RuntimeError, match="output hash"):
        managed_module._read_result(result, output, "binding", _SECRET)


def test_managed_success_proves_empty_tree_operation_and_native_metadata(
    tmp_path: Path, monkeypatch
):
    request = _Request()
    runtime = ManagedLTX23KitchenRuntime(request)  # type: ignore[arg-type]
    paths, events = _paths(tmp_path), []
    supervisor = _FakeSupervisor(paths, events=events)

    output = tmp_path / "output.mp4"
    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(
        managed_module,
        "_supervisor",
        lambda *_args: (supervisor, "expandable_segments:True"),
    )
    monkeypatch.setattr(managed_module, "_wait_for_result", lambda *_args, **_kwargs: None)

    def result(_path, _expected, binding, _secret):
        output.write_bytes(b"mp4")
        return {
            "ok": True,
            "request_binding": binding,
            "output_size_bytes": 3,
            "metadata": _metadata(request, requested_frames=26),
            "allocator_policy": "expandable_segments:True",
        }

    monkeypatch.setattr(managed_module, "_read_result", result)
    progress_events: list[tuple[float, str | None]] = []
    generated = runtime.generate(
        prompt="scene",
        output_path=output,
        width=768,
        height=512,
        duration_seconds=1,
        seed=7,
        progress=lambda value, message: progress_events.append((value, message)),
        check_cancelled=lambda: None,
    )
    assert generated.output_path == output.resolve()
    assert events == ["start", "cleanup_job"]
    status = runtime.status()
    assert status["last_worker"]["outcome"] == "succeeded"
    assert status["last_worker"]["tree_empty"] is False
    assert status["loaded"] is True
    assert status["cache"]["pipeline_warm"] is False
    assert status["cache"]["policy"] == "none"
    assert all(not paths[name].exists() for name in ("request", "result", "progress", "command"))
    assert paths["gate"].exists()
    assert progress_events[:2] == [
        (0.0, "Validating LTX runtime request"),
        (0.0, "Starting isolated LTX worker"),
    ]


def test_managed_session_reuses_one_worker_for_compatible_jobs(tmp_path: Path, monkeypatch) -> None:
    request = _Request()
    runtime = ManagedLTX23KitchenRuntime(request)  # type: ignore[arg-type]
    paths = _paths(tmp_path)
    spawns: list[object] = []
    result_calls = 0
    supervisor = _FakeSupervisor(paths, pid=4242)

    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )

    def supervisor_factory(*_args):
        spawns.append(object())
        return supervisor, "expandable_segments:True"

    monkeypatch.setattr(managed_module, "_supervisor", supervisor_factory)
    monkeypatch.setattr(managed_module, "_wait_for_result", lambda *_args, **_kwargs: None)

    def result(_path, expected, binding, _secret):
        nonlocal result_calls
        expected.write_bytes(b"mp4")
        metadata = _metadata(request, seed=result_calls + 1, requested_frames=26)
        metadata["cache"]["pipeline_warm"] = result_calls > 0
        result_calls += 1
        return {
            "ok": True,
            "request_binding": binding,
            "output_size_bytes": 3,
            "metadata": metadata,
            "allocator_policy": "expandable_segments:True",
        }

    monkeypatch.setattr(managed_module, "_read_result", result)
    first = runtime.generate(
        prompt="first",
        output_path=tmp_path / "first.mp4",
        width=768,
        height=512,
        duration_seconds=1,
        seed=1,
        progress=lambda *_args: None,
        check_cancelled=lambda: None,
    )
    second = runtime.generate(
        prompt="second",
        output_path=tmp_path / "second.mp4",
        width=768,
        height=512,
        duration_seconds=1,
        seed=2,
        progress=lambda *_args: None,
        check_cancelled=lambda: None,
    )

    assert len(spawns) == 1
    assert first.worker_pid == second.worker_pid == 4242
    assert first.worker_exit_code is None
    assert second.worker_exit_code is None
    assert first.metadata["cache"]["pipeline_warm"] is False
    assert second.metadata["cache"]["pipeline_warm"] is True
    assert runtime.status()["cache_support"] == {"prompt": True, "media": False}


def test_managed_clear_cache_commands_live_worker_without_unloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _Request()
    runtime = ManagedLTX23KitchenRuntime(  # type: ignore[arg-type]
        request, cache_policy="prompt"
    )
    paths = _paths(tmp_path)
    supervisor = _FakeSupervisor(paths, pid=4242)
    supervisor.start({})
    runtime._session = managed_module._WorkerSession(
        supervisor, "expandable_segments:True", _SECRET
    )
    runtime._last_cache = {"pipeline_warm": True, "policy": "prompt"}
    monkeypatch.setattr(managed_module, "_wait_for_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        managed_module,
        "_read_clear_cache_result",
        lambda *_args: {
            "ok": True,
            "cache": {
                "pipeline_warm": True,
                "policy": "prompt",
                "prompt_hit": False,
                "prompt_published": False,
                "media_hit": False,
                "prompt": {"entries": 0, "bytes": 0},
            },
        },
    )

    runtime.clear_cache()

    command = supervisor.commands[-1]
    assert command["schema_version"] == 2
    assert command["command"] == "clear_cache"
    assert command["request_fingerprint"] == request.fingerprint
    assert command["cache_policy"] == "prompt"
    assert runtime.status()["loaded"] is True
    assert runtime.status()["cache"]["prompt"]["entries"] == 0


def test_authenticated_clear_cache_result_requires_canonical_live_status(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "clear-result.json"
    value = worker_module._signed_result(
        {
            "schema_version": 2,
            "ok": True,
            "request_binding": "binding",
            "command": "clear_cache",
            "cache": _cleared_cache_status(),
        },
        _SECRET,
    )
    result_path.write_text(json.dumps(value), encoding="utf-8")

    accepted = managed_module._read_clear_cache_result(result_path, "binding", _SECRET, "prompt")

    assert accepted["cache"] == _cleared_cache_status()


@pytest.mark.parametrize(
    ("target", "value"),
    (
        ("policy", "none"),
        ("pipeline_warm", False),
        ("prompt_hit", True),
        ("prompt_published", True),
        ("media_hit", True),
        ("entries", 1),
        ("bytes", 1),
        ("max_bytes", 1),
        ("max_entries", 7),
        ("hits", True),
        ("misses", 0),
        ("hit_rate", None),
        ("hit_rate", 0.75),
    ),
)
def test_authenticated_clear_cache_result_rejects_status_tamper(
    tmp_path: Path, target: str, value: object
) -> None:
    cache = _cleared_cache_status()
    if target in {
        "policy",
        "pipeline_warm",
        "prompt_hit",
        "prompt_published",
        "media_hit",
    }:
        cache[target] = value
    else:
        cache["prompt"][target] = value  # type: ignore[index]
    signed = worker_module._signed_result(
        {
            "schema_version": 2,
            "ok": True,
            "request_binding": "binding",
            "command": "clear_cache",
            "cache": cache,
        },
        _SECRET,
    )
    result_path = tmp_path / f"clear-tamper-{target}.json"
    result_path.write_text(json.dumps(signed), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid cache-clear result"):
        managed_module._read_clear_cache_result(result_path, "binding", _SECRET, "prompt")


def test_authenticated_clear_cache_result_rejects_extra_cache_field(
    tmp_path: Path,
) -> None:
    cache = _cleared_cache_status()
    cache["unexpected"] = True
    signed = worker_module._signed_result(
        {
            "schema_version": 2,
            "ok": True,
            "request_binding": "binding",
            "command": "clear_cache",
            "cache": cache,
        },
        _SECRET,
    )
    result_path = tmp_path / "clear-extra.json"
    result_path.write_text(json.dumps(signed), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid cache-clear result"):
        managed_module._read_clear_cache_result(result_path, "binding", _SECRET, "prompt")


@pytest.mark.parametrize(
    ("target", "value"),
    (("hit_rate", None), ("hit_rate", 0.5), ("prompt_hit", True)),
)
def test_generation_metadata_rejects_cache_counter_contradictions(
    target: str, value: object
) -> None:
    request = _Request()
    metadata = _metadata(request)
    if target == "prompt_hit":
        metadata["cache"][target] = value  # type: ignore[index]
    else:
        metadata["cache"]["prompt"][target] = value  # type: ignore[index]

    with pytest.raises(RuntimeError, match="cache provenance"):
        managed_module._validate_metadata(
            metadata,
            request,  # type: ignore[arg-type]
            _bound_generation(),
        )


def test_generation_metadata_requires_and_accepts_cold_prompt_publication() -> None:
    request = _Request()
    metadata = _metadata(request)
    cache = metadata["cache"]
    cache["policy"] = "prompt"  # type: ignore[index]
    cache["prompt"]["enabled"] = True  # type: ignore[index]

    with pytest.raises(RuntimeError, match="cache provenance"):
        managed_module._validate_metadata(
            metadata,
            request,  # type: ignore[arg-type]
            _bound_generation(),
            cache_policy="prompt",
        )

    cache["prompt_published"] = True  # type: ignore[index]
    cache["prompt"]["entries"] = 1  # type: ignore[index]
    cache["prompt"]["bytes"] = 385_359_999  # type: ignore[index]
    managed_module._validate_metadata(
        metadata,
        request,  # type: ignore[arg-type]
        _bound_generation(),
        cache_policy="prompt",
    )


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("native_text", "policy", "native_quantized_mm", "full-precision text"),
        ("native_text", "native_quantized_dispatches", 1, "full-precision text"),
        ("negative_encoding", "used_for_cfg", True, "negative text provenance"),
        ("negative_encoding", "mask_dtype", "float32", "negative text provenance"),
        ("negative_encoding", "finite", False, "negative text provenance"),
        ("negative_encoding", "embeds_shape", [1, 1024, 3840], "negative text provenance"),
        ("negative_encoding", "mask_shape", [1, 512], "negative text provenance"),
        ("text_patch_state", "restored_base_on_exit", False, "patch-state"),
        ("text_residency", "transfer_stream_count", 3, "text residency"),
        ("text_residency", "transfer_mode", "blocking_cpu", "text residency"),
        ("text_residency", "strict_cuda_parity", False, "text residency"),
        ("text_residency", "layer_compute_barriers", 143, "text residency"),
        ("text_residency", "maximum_live_layer_bindings", 2, "text residency"),
    ),
)
def test_generation_metadata_rejects_strict_text_provenance_tamper(
    section: str, field: str, value: object, message: str
) -> None:
    request = _Request()
    metadata = _metadata(request)
    metadata[section][field] = value  # type: ignore[index]

    with pytest.raises(RuntimeError, match=message):
        managed_module._validate_metadata(
            metadata,
            request,  # type: ignore[arg-type]
            _bound_generation(),
        )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("streaming",), "synchronous_cpu_master"),
        (("group_count",), 48),
        (("prefetch",), False),
        (("leaf_allocation_count",), 49),
        (("prefetch_leaves",), 0),
        (("base_file_handle_live",), False),
        (("cpu_source_bytes_base",), 1),
        (("dynamic_vram", "allocation_count"), 199),
        (("dynamic_vram", "prefetch_calls"), 189),
        (("dynamic_vram", "faults"), 27),
        (("dynamic_vram", "signature_hits"), 1),
        (("dynamic_vram", "copy_strategy"), "per_physical"),
        (("dynamic_vram", "base_file_source_live"), False),
        (("dynamic_vram", "base_file_read_bytes"), 0),
        (("dynamic_vram", "host_buffer_frees"), 1),
        (("dynamic_vram", "host_source_pool_lane_count"), 3),
        (("dynamic_vram", "host_source_pool_hits"), 1),
        (("dynamic_vram", "host_source_pool_stale_rejections"), 1),
        (("dynamic_vram", "host_source_pool_temporary_slices"), 1),
        (("dynamic_vram", "host_source_pool_poisoned"), True),
        (("dynamic_vram", "host_source_registration", "live_bytes"), 199_999),
        (("dynamic_vram", "host_source_registration", "state_proven"), False),
    ),
)
def test_generation_metadata_rejects_file_backed_transformer_residency_tamper(
    path: tuple[str, ...], value: object
) -> None:
    request = _Request()
    metadata = _metadata(request)
    target = metadata["residency_policy"]
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(RuntimeError, match="residency provenance"):
        managed_module._validate_metadata(
            metadata,
            request,  # type: ignore[arg-type]
            _bound_generation(),
        )


def test_generation_metadata_authenticates_file_and_cpu_pressure_direct_transfers() -> None:
    request = _Request()
    metadata = _metadata(request)
    dynamic = metadata["residency_policy"]["dynamic_vram"]  # type: ignore[index]
    dynamic.update(  # type: ignore[union-attr]
        {
            "pinned_copy_bytes": 198_000,
            "pressure_direct_transfers": 1,
            "pressure_direct_bytes": 2_000,
            "host_source_pool_misses": 199,
            "host_source_pool_retained_slices": 199,
            "host_source_pool_retained_bytes": 199_000,
            "host_source_pool_warm_ram_pressure_bypasses": 1,
            "host_source_pool_warm_zero_delta_extend_refusals": 0,
            "host_source_pool_warm_registration_refusals": 0,
        "host_source_pool_temporary_ram_pressure_bypasses": 0,
        "host_source_pool_temporary_zero_delta_extend_refusals": 0,
        "host_source_pool_temporary_registration_refusals": 0,
        }
    )
    dynamic["host_source_registration"].update(  # type: ignore[index]
        {
            "attempts": 199,
            "attempt_bytes": 199_000,
            "successes": 199,
            "registered_bytes": 199_000,
            "live_bytes": 199_000,
            "peak_bytes": 199_000,
        }
    )
    managed_module._validate_metadata(
        metadata,
        request,  # type: ignore[arg-type]
        _bound_generation(),
    )

    cpu_patch = copy.deepcopy(metadata)
    cpu_patch["residency_policy"]["dynamic_vram"]["pageable_copy_bytes"] = 1_000  # type: ignore[index]
    managed_module._validate_metadata(
        cpu_patch,
        request,  # type: ignore[arg-type]
        _bound_generation(),
    )

    excessive_pageable = copy.deepcopy(cpu_patch)
    excessive_pageable["residency_policy"]["dynamic_vram"]["pageable_copy_bytes"] = 2_001  # type: ignore[index]
    with pytest.raises(RuntimeError, match="residency provenance"):
        managed_module._validate_metadata(
            excessive_pageable,
            request,  # type: ignore[arg-type]
            _bound_generation(),
        )

    fabricated_direct = copy.deepcopy(metadata)
    fabricated_direct["residency_policy"]["dynamic_vram"][  # type: ignore[index]
        "host_source_pool_warm_ram_pressure_bypasses"
    ] = 0
    with pytest.raises(RuntimeError, match="residency provenance"):
        managed_module._validate_metadata(
            fabricated_direct,
            request,  # type: ignore[arg-type]
            _bound_generation(),
        )


def test_cross_group_alias_policy_bytes_remain_acceptable_managed_proof() -> None:
    shared = nn.Parameter(torch.ones(8), requires_grad=False)

    class _Block(nn.Module):
        def __init__(self, *, alias: bool) -> None:
            super().__init__()
            self.weight = shared if alias else nn.Parameter(torch.ones(8), requires_grad=False)

    transformer = nn.Module()
    transformer.root = shared
    transformer.transformer_blocks = nn.ModuleList(
        [_Block(alias=index == 0) for index in range(48)]
    )
    residency = kitchen_module._LTX23TransformerResidency(transformer, torch.device("cpu"))
    real_policy = residency.policy
    residency.close()
    assert (
        real_policy["root_bytes"] + real_policy["streamed_block_bytes"]
        == (real_policy["stored_bytes"])
    )

    request = _Request()
    metadata = _metadata(request)
    proof = metadata["residency_policy"]
    proof["stored_bytes"] = real_policy["stored_bytes"]
    proof["root_bytes"] = real_policy["root_bytes"]
    proof["resident_block_bytes"] = 0
    proof["streamed_block_bytes"] = real_policy["streamed_block_bytes"]
    proof["base_file_bytes"] = real_policy["stored_bytes"]
    proof["cpu_source_bytes_lora_mutable"] = 0
    proof["dynamic_vram"]["physical_bytes"] = real_policy["stored_bytes"]

    managed_module._validate_metadata(
        metadata,
        request,  # type: ignore[arg-type]
        _bound_generation(),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("prompt_enhanced", False),
        ("prompt_enhancement_system_sha256", "0" * 64),
        ("prompt_enhancement_seed", 7),
        ("prompt_enhancement_max_new_tokens", 1024),
        ("prompt_enhancement_stop_token_id", 1),
        ("prompt_enhancement_template", "transformers_chat_template"),
        ("prompt_enhancement_generation_settings", {"do_sample": False}),
    ),
)
def test_generation_metadata_rejects_prompt_enhancement_contract_tamper(
    field: str, value: object
) -> None:
    request = _Request()
    metadata = _metadata(request)
    metadata[field] = value

    with pytest.raises(RuntimeError, match="prompt-enhancement provenance"):
        managed_module._validate_metadata(
            metadata,
            request,  # type: ignore[arg-type]
            _bound_generation(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("template", "wrong"),
        ("stop_token_id", 1),
        ("generation_settings", {"do_sample": False}),
        ("decoded_suffix_nonempty", "yes"),
        ("think_block_removed", "yes"),
        ("fallback_to_source_prompt", "no"),
    ),
)
def test_generation_metadata_rejects_prompt_enhancement_memory_tamper(
    field: str, value: object
) -> None:
    request = _Request()
    metadata = _metadata(request)
    metadata["prompt_enhancement_memory"][field] = value  # type: ignore[index]

    with pytest.raises(RuntimeError, match="prompt-enhancement provenance"):
        managed_module._validate_metadata(
            metadata,
            request,  # type: ignore[arg-type]
            _bound_generation(),
        )


def test_non_enhanced_operation_requires_exact_none_prompt_provenance() -> None:
    metadata = _metadata(_Request())
    metadata.update(
        prompt_enhanced=False,
        prompt_enhancement_system_sha256=None,
        prompt_enhancement_seed=None,
        prompt_enhancement_max_new_tokens=None,
        prompt_enhancement_stop_token_id=None,
        prompt_enhancement_template=None,
        prompt_enhancement_generation_settings=None,
        prompt_enhancement_memory=None,
    )
    assert managed_module._valid_prompt_enhancement_provenance(metadata, "ltx23_dev_i2v")
    metadata["prompt_enhancement_stop_token_id"] = 106
    assert not managed_module._valid_prompt_enhancement_provenance(metadata, "ltx23_dev_i2v")


def test_generation_metadata_rejects_nonmonotonic_cumulative_timing() -> None:
    request = _Request()
    metadata = _metadata(request)
    metadata["timings"]["cumulative"]["negative_encode"] = 1.0  # type: ignore[index]

    with pytest.raises(RuntimeError, match="timing provenance"):
        managed_module._validate_metadata(
            metadata,
            request,  # type: ignore[arg-type]
            _bound_generation(),
        )


def test_generation_metadata_accepts_and_retains_authenticated_memory_telemetry() -> None:
    request = _Request()
    metadata = _metadata(request)
    expected = copy.deepcopy(metadata["memory_telemetry"])

    managed_module._validate_metadata(
        metadata,
        request,  # type: ignore[arg-type]
        _bound_generation(),
    )

    assert metadata["memory_telemetry"] == expected
    assert managed_module._valid_memory_telemetry(
        _memory_telemetry("ltx23_distilled_flf"), "ltx23_distilled_flf"
    )


def test_memory_telemetry_accepts_explicit_isolated_source_errors() -> None:
    value = _memory_telemetry()
    sample = value["samples"][2]  # type: ignore[index]
    sample["process"] = {  # type: ignore[index]
        "status": "error",
        "error": "OSError",
        "pid": None,
        "private_bytes": None,
        "working_set_bytes": None,
    }
    sample["cuda"] = {  # type: ignore[index]
        "status": "error",
        "error": "RuntimeError",
        "device": "cuda:0",
        "allocated_bytes": None,
        "reserved_bytes": None,
        "free_bytes": None,
        "total_bytes": None,
    }

    assert managed_module._valid_memory_telemetry(value, "ltx23_dev_t2v")


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("phase", "after_stage2"),
        ("sequence", 4),
        ("elapsed_ns", -1),
        ("timestamp_unix_ns", 0),
        ("process_private_bytes", None),
        ("system_used_bytes", 59_999),
        ("cuda_reserved_bytes", 1_000),
        ("cuda_free_bytes", 10_001),
        ("cuda_error_with_values", "RuntimeError"),
    ],
)
def test_generation_metadata_rejects_memory_telemetry_tamper(target: str, value: object) -> None:
    request = _Request()
    metadata = _metadata(request)
    sample = metadata["memory_telemetry"]["samples"][1]  # type: ignore[index]
    if target == "process_private_bytes":
        sample["process"]["private_bytes"] = value
    elif target == "system_used_bytes":
        sample["system"]["used_physical_bytes"] = value
    elif target == "cuda_reserved_bytes":
        sample["cuda"]["reserved_bytes"] = value
    elif target == "cuda_free_bytes":
        sample["cuda"]["free_bytes"] = value
    elif target == "cuda_error_with_values":
        sample["cuda"]["status"] = "error"
        sample["cuda"]["error"] = value
    else:
        sample[target] = value

    with pytest.raises(RuntimeError, match="memory telemetry"):
        managed_module._validate_metadata(
            metadata,
            request,  # type: ignore[arg-type]
            _bound_generation(),
        )


def test_memory_telemetry_rejects_nonmonotonic_elapsed_and_extra_fields() -> None:
    nonmonotonic = _memory_telemetry()
    nonmonotonic["samples"][3]["elapsed_ns"] = 5  # type: ignore[index]
    assert not managed_module._valid_memory_telemetry(nonmonotonic, "ltx23_dev_t2v")

    extra = _memory_telemetry()
    extra["samples"][0]["private_detail"] = "forbidden"  # type: ignore[index]
    assert not managed_module._valid_memory_telemetry(extra, "ltx23_dev_t2v")


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("top", "schema_version"),
        ("sample", "sequence"),
        ("sample", "timestamp_unix_ns"),
        ("sample", "elapsed_ns"),
        ("process", "pid"),
        ("process", "private_bytes"),
        ("process", "working_set_bytes"),
        ("system", "total_physical_bytes"),
        ("system", "free_physical_bytes"),
        ("system", "used_physical_bytes"),
        ("cuda", "allocated_bytes"),
        ("cuda", "reserved_bytes"),
        ("cuda", "free_bytes"),
        ("cuda", "total_bytes"),
    ],
)
def test_memory_telemetry_rejects_boolean_numeric_fields(section: str, field: str) -> None:
    value = _memory_telemetry()
    if section == "top":
        value[field] = True
    elif section == "sample":
        value["samples"][1][field] = True  # type: ignore[index]
    else:
        value["samples"][1][section][field] = True  # type: ignore[index]

    assert not managed_module._valid_memory_telemetry(value, "ltx23_dev_t2v")


class _CoercibleTelemetryInteger:
    def __int__(self) -> int:
        return 1


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("top", "schema_version"),
        ("sample", "sequence"),
        ("sample", "timestamp_unix_ns"),
        ("sample", "elapsed_ns"),
        ("process", "pid"),
        ("process", "private_bytes"),
        ("process", "working_set_bytes"),
        ("system", "total_physical_bytes"),
        ("system", "free_physical_bytes"),
        ("system", "used_physical_bytes"),
        ("cuda", "allocated_bytes"),
        ("cuda", "reserved_bytes"),
        ("cuda", "free_bytes"),
        ("cuda", "total_bytes"),
    ],
)
@pytest.mark.parametrize("invalid", [1.0, "1", _CoercibleTelemetryInteger()])
def test_memory_telemetry_rejects_coercible_non_integer_fields(
    section: str,
    field: str,
    invalid: object,
) -> None:
    value = _memory_telemetry()
    if section == "top":
        value[field] = invalid
    elif section == "sample":
        value["samples"][1][field] = invalid  # type: ignore[index]
    else:
        value["samples"][1][section][field] = invalid  # type: ignore[index]

    assert not managed_module._valid_memory_telemetry(value, "ltx23_dev_t2v")


def test_text_host_registration_accepts_truthful_zero_success_best_effort() -> None:
    value = {
        "policy": "comfy_best_effort_in_place_cuda_host_register",
        "lifecycle": "text_stage_onload_through_synchronized_offload",
        "budget_bytes": 100,
        "candidates": 2,
        "candidate_bytes": 200,
        "deduplicated_aliases": 0,
        "already_registered": 0,
        "already_registered_bytes": 0,
        "attempts": 1,
        "attempt_bytes": 100,
        "successes": 0,
        "registered_bytes": 0,
        "failures": 1,
        "failure_bytes": 100,
        "ineligible": 1,
        "ineligible_bytes": 100,
        "unregistered": 0,
        "unregistered_bytes": 0,
        "unregister_failures": 0,
        "unregister_failure_bytes": 0,
        "owned_active": 0,
        "owned_active_bytes": 0,
        "categories": {
            "unsupported_type": 0,
            "non_cpu": 0,
            "noncontiguous": 0,
            "zero_pointer": 0,
            "budget_exceeded": 1,
            "eligibility_error": 0,
            "register_error": 1,
            "unregister_error": 0,
        },
    }

    assert managed_module._valid_text_host_registration(value)


def test_managed_accepts_exact_aimdo_text_proof_and_rejects_tampering() -> None:
    residency = copy.deepcopy(_metadata(_Request())["text_residency"])
    registration = residency["host_registration"]
    registration["lifecycle"] = "residency_stage_through_synchronized_close"
    for field in (
        "candidates",
        "candidate_bytes",
        "deduplicated_aliases",
        "already_registered",
        "already_registered_bytes",
        "attempts",
        "attempt_bytes",
        "successes",
        "registered_bytes",
        "failures",
        "failure_bytes",
        "ineligible",
        "ineligible_bytes",
        "unregistered",
        "unregistered_bytes",
        "unregister_failures",
        "unregister_failure_bytes",
        "owned_active",
        "owned_active_bytes",
    ):
        registration[field] = 0
    registration["categories"] = {key: 0 for key in registration["categories"]}
    residency.update(
        {
            "mode": "dynamic_vbar_per_leaf",
            "root_activation": "per_model_forward_fault",
            "root_transitions": 3,
            "transfer_mode": "aimdo_two_stream_nonblocking",
            "transfer_events": 50,
            "transfer_waits": 50,
            "leaf_allocation_count": 200,
            "force_resident_leaf_count": 10,
            "base_leaf_count": 150,
            "patch_leaf_count": 50,
            "schedule_group_count": 98,
            "warm_request_index": 1,
            "leaf_scheduler": _generated_text_leaf_scheduler_proof(),
            "dynamic_vram": {
                "backend": "comfy-aimdo",
                "version": "0.4.15",
                "mode": "dynamic_vbar",
                "policy": "auto",
                "physical_bytes": 10_000,
                "staged_bytes": 12_288,
                "virtual_bytes": 16_384,
                "allocation_count": 200,
                "live_allocations": 200,
                "live_bytes": 16_384,
                # Released/unpinned VBAR pages may remain physically resident
                # for safe same-model warm reuse.
                "loaded_bytes": 8_192,
                "faults": 147,
                "signature_hits": 97,
                "signature_misses": 50,
                "fault_none_temporaries": 2,
                "pinned_copy_bytes": 24_576,
                "pageable_copy_bytes": 0,
                "transfer_events": 50,
                "transfer_waits": 50,
                "prioritize_calls": 1,
                "unpin_calls": 145,
                "free_calls": 0,
                "dirty_epoch": 1,
                "lora_invalidations": 1,
                "base_restores": 1,
                "copy_stream_count": 2,
                "copy_strategy": "gathered_host_buffer",
                "copy_fallback_reason": None,
                "gathered_host_buffer_requested": True,
                "host_buffer_capacity_bytes": 4_096,
                "host_buffer_allocations": 4,
                "host_buffer_unregistrations": 0,
                "host_buffer_frees": 0,
                "host_buffer_live": True,
                "host_tensor_view_live": True,
                "host_buffer_transfer_pending": False,
                "gathered_misses": 50,
                "per_physical_misses": 0,
                "packed_source_bytes": 20_000,
                "gathered_h2d_bytes": 24_576,
                "pressure_direct_transfers": 0,
                "pressure_direct_bytes": 0,
                "host_buffer_reuse_barriers": 0,
                "host_source_pool_generation": 2,
                "host_source_pool_lane_count": 4,
                "host_source_pool_capacity_bytes": 32_768,
                "host_source_pool_retained_slices": 50,
                "host_source_pool_retained_bytes": 20_000,
                "host_source_pool_temporary_slices": 0,
                "host_source_pool_temporary_bytes": 0,
                "host_source_pool_hits": 0,
                "host_source_pool_misses": 50,
                "host_source_pool_stale_rejections": 0,
                "host_source_pool_warm_ram_pressure_bypasses": 0,
                "host_source_pool_warm_zero_delta_extend_refusals": 0,
                "host_source_pool_warm_registration_refusals": 0,
        "host_source_pool_temporary_ram_pressure_bypasses": 0,
        "host_source_pool_temporary_zero_delta_extend_refusals": 0,
        "host_source_pool_temporary_registration_refusals": 0,
                "host_source_pool_poisoned": False,
                "host_source_pool_poison_reason": None,
                "host_source_registration": _host_source_registration(
                    budget_bytes=32_768,
                    attempts=50,
                    attempt_bytes=24_576,
                    successes=50,
                    registered_bytes=24_576,
                    live_bytes=24_576,
                    peak_bytes=24_576,
                ),
                "base_file_backed": True,
                "base_file_source_live": True,
                "base_file_read_calls": 80,
                "base_file_read_bytes": 18_000,
                "base_file_handle_live": True,
                "base_file_handle_opened": 1,
                "base_file_handle_closed": 0,
                "base_file_fallback_reason": None,
                "prefetch": False,
                "prefetch_calls": 0,
                "allocator_plugin": False,
                "poisoned": False,
                "close_failed": False,
                "poison_reason": None,
                "host_registration": copy.deepcopy(registration),
                "warm_request_index": 1,
                "request_delta": {
                    "faults": 147,
                    "signature_hits": 97,
                    "signature_misses": 50,
                    "fault_none_temporaries": 2,
                    "pinned_copy_bytes": 24_576,
                    "pageable_copy_bytes": 0,
                    "transfer_events": 50,
                    "transfer_waits": 50,
                    "unpin_calls": 145,
                    "dirty_epoch": 1,
                    "lora_invalidations": 1,
                    "base_restores": 1,
                    "gathered_misses": 50,
                    "per_physical_misses": 0,
                    "packed_source_bytes": 20_000,
                    "gathered_h2d_bytes": 24_576,
                    "pressure_direct_transfers": 0,
                    "pressure_direct_bytes": 0,
                    "host_buffer_reuse_barriers": 0,
                    "host_source_pool_hits": 0,
                    "host_source_pool_misses": 50,
                   "host_source_pool_stale_rejections": 0,
                    "host_source_pool_warm_ram_pressure_bypasses": 0,
                    "host_source_pool_warm_zero_delta_extend_refusals": 0,
                    "host_source_pool_warm_registration_refusals": 0,
        "host_source_pool_temporary_ram_pressure_bypasses": 0,
        "host_source_pool_temporary_zero_delta_extend_refusals": 0,
        "host_source_pool_temporary_registration_refusals": 0,
                    "base_file_read_calls": 80,
                    "base_file_read_bytes": 18_000,
                    "prefetch_calls": 0,
                },
            },
        }
    )

    assert managed_module._valid_text_residency(residency, lora_to_base_transitions=1)
    pressure_direct = copy.deepcopy(residency)
    pressure_dynamic = pressure_direct["dynamic_vram"]
    pressure_dynamic.update(
        pinned_copy_bytes=20_000,
        pressure_direct_transfers=1,
        pressure_direct_bytes=4_576,
        host_source_pool_misses=49,
        host_source_pool_warm_ram_pressure_bypasses=1,
        host_source_pool_warm_zero_delta_extend_refusals=0,
    )
    pressure_request = pressure_dynamic["request_delta"]
    pressure_request.update(
        pinned_copy_bytes=20_000,
        pressure_direct_transfers=1,
        pressure_direct_bytes=4_576,
        host_source_pool_misses=49,
        host_source_pool_warm_ram_pressure_bypasses=1,
        host_source_pool_warm_zero_delta_extend_refusals=0,
    )
    pressure_dynamic["host_source_registration"].update(
        attempts=49,
        successes=49,
    )
    assert managed_module._valid_text_residency(
        pressure_direct, lora_to_base_transitions=1
    )
    pressure_cpu_patch = copy.deepcopy(pressure_direct)
    pressure_cpu_dynamic = pressure_cpu_patch["dynamic_vram"]
    pressure_cpu_dynamic["pageable_copy_bytes"] = 1_024
    pressure_cpu_dynamic["request_delta"]["pageable_copy_bytes"] = 1_024
    assert managed_module._valid_text_residency(
        pressure_cpu_patch, lora_to_base_transitions=1
    )
    pressure_cpu_excess = copy.deepcopy(pressure_cpu_patch)
    pressure_cpu_excess["dynamic_vram"]["pageable_copy_bytes"] = 4_577
    assert not managed_module._valid_text_residency(
        pressure_cpu_excess, lora_to_base_transitions=1
    )
    fabricated_pressure_direct = copy.deepcopy(pressure_direct)
    fabricated_pressure_direct["dynamic_vram"][
        "host_source_pool_warm_ram_pressure_bypasses"
    ] = 0
    fabricated_pressure_direct["dynamic_vram"][
        "request_delta"
    ]["host_source_pool_warm_ram_pressure_bypasses"] = 0
    assert not managed_module._valid_text_residency(
        fabricated_pressure_direct, lora_to_base_transitions=1
    )
    wrong_pressure_cause_sum = copy.deepcopy(pressure_direct)
    wrong_pressure_cause_sum["dynamic_vram"][
        "host_source_pool_warm_ram_pressure_bypasses"
    ] = 2
    wrong_pressure_cause_sum["dynamic_vram"][
        "request_delta"
    ]["host_source_pool_warm_ram_pressure_bypasses"] = 2
    assert not managed_module._valid_text_residency(
        wrong_pressure_cause_sum, lora_to_base_transitions=1
    )
    second = copy.deepcopy(residency)
    second["warm_request_index"] = 2
    second["transfer_events"] = second["transfer_waits"] = 10
    dynamic_second = second["dynamic_vram"]
    dynamic_second.update(
        warm_request_index=2,
        faults=294,
        signature_hits=234,
        signature_misses=60,
        fault_none_temporaries=2,
        pinned_copy_bytes=28_672,
        transfer_events=60,
        transfer_waits=60,
        unpin_calls=292,
        dirty_epoch=2,
        lora_invalidations=2,
        base_restores=2,
        gathered_misses=60,
        packed_source_bytes=24_000,
        gathered_h2d_bytes=28_672,
        host_source_pool_misses=60,
        host_source_registration=_host_source_registration(
            budget_bytes=32_768,
            attempts=60,
            attempt_bytes=28_672,
            successes=60,
            registered_bytes=28_672,
            unregistered_bytes=4_096,
            live_bytes=24_576,
            peak_bytes=24_576,
        ),
        request_delta={
            "faults": 147,
            "signature_hits": 137,
            "signature_misses": 10,
            "fault_none_temporaries": 0,
            "pinned_copy_bytes": 4_096,
            "pageable_copy_bytes": 0,
            "transfer_events": 10,
            "transfer_waits": 10,
            "unpin_calls": 147,
            "dirty_epoch": 1,
            "lora_invalidations": 1,
            "base_restores": 1,
            "gathered_misses": 10,
            "per_physical_misses": 0,
            "packed_source_bytes": 4_000,
            "gathered_h2d_bytes": 4_096,
            "pressure_direct_transfers": 0,
            "pressure_direct_bytes": 0,
            "host_buffer_reuse_barriers": 0,
            "host_source_pool_hits": 0,
            "host_source_pool_misses": 10,
            "host_source_pool_stale_rejections": 0,
            "host_source_pool_warm_ram_pressure_bypasses": 0,
            "host_source_pool_warm_zero_delta_extend_refusals": 0,
            "host_source_pool_warm_registration_refusals": 0,
        "host_source_pool_temporary_ram_pressure_bypasses": 0,
        "host_source_pool_temporary_zero_delta_extend_refusals": 0,
        "host_source_pool_temporary_registration_refusals": 0,
            "base_file_read_calls": 0,
            "base_file_read_bytes": 0,
            "prefetch_calls": 0,
        },
    )
    assert managed_module._valid_text_residency(
        second, lora_to_base_transitions=1
    )
    assert dynamic_second["signature_hits"] > dynamic_second["request_delta"]["signature_hits"]
    assert dynamic_second["base_file_read_bytes"] > 0
    assert dynamic_second["request_delta"]["base_file_read_bytes"] == 0

    fallback = copy.deepcopy(residency)
    fallback_registration = copy.deepcopy(
        _metadata(_Request())["text_residency"]["host_registration"]
    )
    fallback_registration["lifecycle"] = "residency_stage_through_synchronized_close"
    fallback["host_registration"] = fallback_registration
    fallback["transfer_events"] = fallback["transfer_waits"] = 145
    fallback_dynamic = fallback["dynamic_vram"]
    fallback_dynamic.update(
        copy_strategy="per_physical",
        copy_fallback_reason="host_buffer_capability_unavailable: fixture",
        host_buffer_capacity_bytes=0,
        host_buffer_allocations=0,
        host_buffer_unregistrations=0,
        host_buffer_frees=0,
        gathered_misses=0,
        per_physical_misses=50,
        packed_source_bytes=0,
        gathered_h2d_bytes=0,
        host_buffer_reuse_barriers=0,
        host_source_pool_generation=0,
        host_source_pool_lane_count=0,
        host_source_pool_capacity_bytes=0,
        host_source_pool_retained_slices=0,
        host_source_pool_retained_bytes=0,
        host_source_pool_temporary_slices=0,
        host_source_pool_temporary_bytes=0,
        host_source_pool_hits=0,
        host_source_pool_misses=0,
        host_source_pool_stale_rejections=0,
        host_source_pool_poisoned=False,
        host_source_pool_poison_reason=None,
        host_source_registration=_host_source_registration(),
        base_file_backed=False,
        base_file_source_live=False,
        base_file_read_calls=0,
        base_file_read_bytes=0,
        base_file_handle_live=False,
        base_file_handle_opened=0,
        base_file_handle_closed=0,
        base_file_fallback_reason=None,
        pinned_copy_bytes=8_000,
        pageable_copy_bytes=2_000,
        transfer_events=145,
        transfer_waits=145,
        host_registration=copy.deepcopy(fallback_registration),
    )
    assert not managed_module._valid_text_residency(fallback, lora_to_base_transitions=1)
    safe_setup_fallback = copy.deepcopy(fallback)
    safe_dynamic = safe_setup_fallback["dynamic_vram"]
    safe_dynamic.update(
        host_buffer_capacity_bytes=4_096,
        host_buffer_allocations=2,
        host_buffer_unregistrations=0,
        host_buffer_frees=2,
        host_source_pool_generation=2,
        host_source_pool_lane_count=2,
        host_source_pool_capacity_bytes=8_192,
        host_source_registration=_host_source_registration(budget_bytes=32_768),
    )
    assert not managed_module._valid_text_residency(
        safe_setup_fallback, lora_to_base_transitions=1
    )
    safe_dynamic["host_buffer_frees"] = 1
    assert not managed_module._valid_text_residency(
        safe_setup_fallback, lora_to_base_transitions=1
    )
    for path, replacement in (
        (("dynamic_vram", "live_allocations"), 0),
        (("dynamic_vram", "loaded_bytes"), 16_385),
        (("dynamic_vram", "copy_stream_count"), 1),
        (("dynamic_vram", "allocator_plugin"), True),
        (("dynamic_vram", "poisoned"), True),
        (("dynamic_vram", "close_failed"), True),
        (("dynamic_vram", "signature_hits"), 99),
        (("dynamic_vram", "signature_misses"), 0),
        (("dynamic_vram", "pinned_copy_bytes"), 1),
        (("dynamic_vram", "host_buffer_capacity_bytes"), 0),
        (("dynamic_vram", "gathered_h2d_bytes"), 204_801),
        (("dynamic_vram", "pressure_direct_transfers"), True),
        (("dynamic_vram", "pressure_direct_transfers"), 51),
        (("dynamic_vram", "pressure_direct_bytes"), True),
        (("dynamic_vram", "pressure_direct_bytes"), 24_577),
        (("dynamic_vram", "pageable_copy_bytes"), 1),
        (("dynamic_vram", "gathered_misses"), 49),
        (("dynamic_vram", "host_buffer_frees"), 1),
        (("dynamic_vram", "base_file_read_bytes"), 20_001),
        (("dynamic_vram", "base_file_handle_closed"), 1),
        (("dynamic_vram", "base_file_source_live"), False),
        (("dynamic_vram", "base_file_fallback_reason"), "runtime_read_failed: bad"),
        (("dynamic_vram", "prefetch_calls"), True),
        (("dynamic_vram", "prefetch_calls"), 0.0),
        (("dynamic_vram", "prefetch_calls"), -1),
        (("dynamic_vram", "prefetch_calls"), 1),
        (("dynamic_vram", "host_source_pool_lane_count"), 3),
        (("dynamic_vram", "host_source_pool_hits"), 1),
        (("dynamic_vram", "host_source_pool_stale_rejections"), 1),
        (("dynamic_vram", "host_source_pool_retained_slices"), 0),
        (("dynamic_vram", "host_source_pool_poisoned"), True),
        (("dynamic_vram", "host_source_registration", "attempts"), 49),
        (("dynamic_vram", "host_source_registration", "live_bytes"), 1),
        (("dynamic_vram", "host_source_registration", "state_proven"), False),
        (("dynamic_vram", "host_registration", "budget_bytes"), 16_001),
        (("dynamic_vram", "host_registration", "owned_active"), 1),
        (("dynamic_vram", "allocation_count"), 49),
        (("leaf_allocation_count",), 49),
        (("leaf_scheduler", "prefetch_groups"), 1),
        (("leaf_scheduler", "force_resident_waits"), 0),
        (("leaf_scheduler", "force_resident_waits"), True),
        (("leaf_scheduler", "deferred_waits"), 128),
        (("leaf_scheduler", "pending_prefetch"), True),
        (("warm_request_index",), True),
        (("dynamic_vram", "warm_request_index"), 2),
        (("dynamic_vram", "request_delta", "dirty_epoch"), 0),
        (("dynamic_vram", "request_delta", "pressure_direct_transfers"), 51),
        (("dynamic_vram", "request_delta", "pressure_direct_bytes"), 1),
        (("dynamic_vram", "request_delta", "pageable_copy_bytes"), 1),
        (("dynamic_vram", "request_delta", "prefetch_calls"), 1),
        (("patch_leaf_count",), 0),
        (("root_transitions",), 2),
    ):
        tampered = copy.deepcopy(residency)
        target = tampered
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        assert not managed_module._valid_text_residency(
            tampered, lora_to_base_transitions=1
        ), path

    missing_prefetch_calls = copy.deepcopy(residency)
    del missing_prefetch_calls["dynamic_vram"]["prefetch_calls"]
    assert not managed_module._valid_text_residency(
        missing_prefetch_calls, lora_to_base_transitions=1
    )

    no_faults = copy.deepcopy(residency)
    no_faults["dynamic_vram"].update(
        faults=0, signature_hits=0, signature_misses=0, transfer_events=0, transfer_waits=0
    )
    no_faults.update(transfer_events=0, transfer_waits=0)
    assert not managed_module._valid_text_residency(no_faults, lora_to_base_transitions=1)
    assert not managed_module._valid_text_residency(residency, lora_to_base_transitions=0)

    invalid_fallback = copy.deepcopy(fallback)
    invalid_fallback["dynamic_vram"]["copy_fallback_reason"] = "host_buffer_runtime_failed: fixture"
    assert not managed_module._valid_text_residency(invalid_fallback, lora_to_base_transitions=1)


def test_text_forward_accounting_accepts_variable_autoregressive_length() -> None:
    proof = {
        "root_transitions": 175,
        "layer_count": 48,
        "layer_transitions": 8_400,
        "full_precision_dispatches": 58_800,
    }

    assert managed_module._valid_text_forward_accounting(proof, 1)
    assert not managed_module._valid_text_forward_accounting(
        {**proof, "root_transitions": 2}, 1
    )
    assert not managed_module._valid_text_forward_accounting(
        {**proof, "layer_transitions": 8_399}, 1
    )
    assert not managed_module._valid_text_forward_accounting(
        {**proof, "full_precision_dispatches": 58_799}, 1
    )


def test_text_residency_rejection_summary_is_bounded_and_excludes_unknown_fields() -> None:
    summary = managed_module._text_residency_rejection_summary(
        {
            "mode": "dynamic_vbar_per_leaf",
            "root_transitions": 175,
            "prompt": "must not be logged",
            "dynamic_vram": {
                "loaded_bytes": 123,
                "live_bytes": 456,
                "untrusted_detail": "must not be logged",
            },
        }
    )

    assert summary["mode"] == "dynamic_vbar_per_leaf"
    assert summary["root_transitions"] == 175
    assert summary["dynamic_vram"] == {
        "loaded_bytes": 123,
        "live_bytes": 456,
        "faults": None,
        "signature_hits": None,
        "signature_misses": None,
        "fault_none_temporaries": None,
        "unpin_calls": None,
        "transfer_events": None,
        "transfer_waits": None,
        "host_source_pool_hits": None,
        "host_source_pool_misses": None,
        "host_source_pool_poisoned": None,
        "poisoned": None,
        "close_failed": None,
    }
    assert "prompt" not in summary
    assert "untrusted_detail" not in summary["dynamic_vram"]
    assert isinstance(summary["proof_sha256"], str)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("registered_bytes", 20_000),
        ("owned_active", 1),
        ("unregistered", 59),
        ("deduplicated_aliases", 101),
    ),
)
def test_text_host_registration_rejects_counter_or_ownership_tamper(field: str, value: int) -> None:
    registration = copy.deepcopy(_metadata(_Request())["text_residency"]["host_registration"])
    registration[field] = value

    assert not managed_module._valid_text_host_registration(registration)


def test_generation_metadata_accepts_cached_negative_and_text_source_proofs() -> None:
    request = _Request()
    metadata = _metadata(request)
    for name in ("native_text", "negative_encoding", "text_patch_state", "text_lora"):
        metadata[name] = {
            "provenance": "cached_prompt_conditioning",
            "dispatch_performed": False,
            "source_proof": metadata[name],
        }
    metadata["text_residency"] = {
        "mode": "not_required_prompt_cache_hit",
        "source_proof": metadata["text_residency"],
    }
    cache = metadata["cache"]
    cache.update(policy="prompt", prompt_hit=True)  # type: ignore[union-attr]
    cache["prompt"].update(  # type: ignore[index,union-attr]
        enabled=True,
        entries=1,
        bytes=770_720_000,
        hits=1,
        misses=0,
        hit_rate=1.0,
    )
    metadata["prompt_enhancement_memory"] = {
        "policy": "not_required_prompt_cache_hit",
        "source_proof": metadata["prompt_enhancement_memory"],
    }

    managed_module._validate_metadata(
        metadata,
        request,  # type: ignore[arg-type]
        _bound_generation(),
        cache_policy="prompt",
    )


def test_unload_clears_session_state_even_when_tree_close_fails(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = ManagedLTX23KitchenRuntime(_Request())  # type: ignore[arg-type]
    paths = _paths(tmp_path)
    supervisor = _FakeSupervisor(paths, pid=77)
    supervisor.start({})

    def terminate():
        raise OSError("terminate failed")

    def close():
        supervisor.session = None
        raise OSError("close failed")

    monkeypatch.setattr(supervisor, "terminate", terminate)
    monkeypatch.setattr(supervisor, "close", close)

    session = managed_module._WorkerSession(supervisor, "policy", _SECRET)
    runtime._session = session
    runtime._last_cache = _cleared_cache_status()
    with pytest.raises(OSError, match="terminate failed"):
        runtime.unload()
    assert runtime.status()["loaded"] is False
    assert runtime.status()["active_worker"] is False
    assert runtime.status()["cache"] == managed_module._empty_cache_status("none")


def test_normal_unload_resets_dead_worker_cache_status(tmp_path: Path) -> None:
    runtime = ManagedLTX23KitchenRuntime(  # type: ignore[arg-type]
        _Request(), cache_policy="prompt"
    )
    supervisor = _FakeSupervisor(_paths(tmp_path), pid=79)
    supervisor.start({})
    runtime._session = managed_module._WorkerSession(supervisor, "policy", _SECRET)
    runtime._last_cache = _cleared_cache_status()

    runtime.unload()

    assert runtime.status()["loaded"] is False
    assert runtime.status()["cache"] == managed_module._empty_cache_status("prompt")


@pytest.mark.parametrize("failure_point", ["spawn", "job_object"])
def test_failed_session_start_cleans_private_ipc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    runtime = ManagedLTX23KitchenRuntime(_Request())  # type: ignore[arg-type]
    paths = _paths(tmp_path)
    existing = tmp_path / "existing.mp4"
    existing.write_bytes(b"keep")
    output = tmp_path / "target.mp4"
    supervisor = _FakeSupervisor(paths, start_error=OSError(f"{failure_point} failed"))
    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(
        managed_module,
        "_supervisor",
        lambda *_args: (supervisor, "expandable_segments:True"),
    )
    with pytest.raises(OSError):
        runtime.generate(
            prompt="scene",
            output_path=output,
            width=768,
            height=512,
            duration_seconds=1,
            seed=1,
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )
    assert existing.read_bytes() == b"keep"
    assert all(not path.exists() for path in paths.values())


def test_unload_surfaces_close_only_error_after_clearing_session(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = ManagedLTX23KitchenRuntime(_Request())  # type: ignore[arg-type]
    paths = _paths(tmp_path)
    supervisor = _FakeSupervisor(paths, pid=78, close_error=OSError("close only"))
    supervisor.start({})
    session = managed_module._WorkerSession(supervisor, "policy", _SECRET)
    runtime._session = session
    with pytest.raises(OSError, match="close only"):
        runtime.unload()
    assert runtime.status()["loaded"] is False


def test_ltx_runtime_reuses_materialized_components_then_unloads_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = object.__new__(kitchen_module.LTX23KitchenRuntime)
    runtime.request = SimpleNamespace(operation="ltx23_dev_t2v")
    runtime.device = kitchen_module.torch.device("cpu")
    runtime._components = None
    materialized: list[dict[str, object]] = []
    executed: list[dict[str, object]] = []
    released: list[dict[str, object]] = []
    components = {"transformer": object()}
    residency_closed: list[object] = []

    class _Residency:
        def __init__(self, transformer, _device):
            self.transformer = transformer

        def close(self):
            residency_closed.append(self.transformer)

    monkeypatch.setattr(kitchen_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        kitchen_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(kitchen_module, "validate_ltx23_kitchen_generation", lambda *_args: None)
    monkeypatch.setattr(kitchen_module, "_LTX23TransformerResidency", _Residency)
    monkeypatch.setattr(
        runtime,
        "_materialize",
        lambda *_args: materialized.append(components) or components,
    )

    def execute(component_set, generation, **_kwargs):
        executed.append(component_set)
        return kitchen_module.LTX23KitchenResult(Path(generation.output_path), {})

    monkeypatch.setattr(runtime, "_execute", execute)
    monkeypatch.setattr(
        kitchen_module, "_release_components", lambda value, _device: released.append(value)
    )
    first = SimpleNamespace(output_path=tmp_path / "one.mp4")
    second = SimpleNamespace(output_path=tmp_path / "two.mp4")
    runtime.generate(first, progress=lambda *_args: None, check_cancelled=lambda: None)
    runtime.generate(second, progress=lambda *_args: None, check_cancelled=lambda: None)
    runtime.unload()

    assert materialized == [components]
    assert executed == [components, components]
    assert released == [components]
    assert residency_closed == [components["transformer"]]


@pytest.mark.parametrize(
    ("cache_policy", "second_hit", "expected_hits", "expected_misses"),
    (("prompt", True, 1, 1), ("none", False, 0, 2)),
)
def test_ltx_runtime_publishes_cold_miss_then_policy_bound_warm_cache_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_policy: str,
    second_hit: bool,
    expected_hits: int,
    expected_misses: int,
) -> None:
    request = SimpleNamespace(operation="ltx23_dev_t2v", fingerprint="request")
    runtime = kitchen_module.LTX23KitchenRuntime(request, device="cuda", cache_policy=cache_policy)
    components = {"transformer": object()}

    class _Residency:
        def __init__(self, *_args) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(kitchen_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        kitchen_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(kitchen_module, "validate_ltx23_kitchen_generation", lambda *_args: None)
    monkeypatch.setattr(kitchen_module, "_LTX23TransformerResidency", _Residency)
    monkeypatch.setattr(runtime, "_materialize", lambda *_args: components)

    def execute(_components, generation, **_kwargs):
        cached = runtime._cache.prompt.get("conditioning")
        hit = cached is not None
        published = False
        if not hit:
            published = runtime._cache.prompt.put("conditioning", {"value": generation.seed})
        return kitchen_module.LTX23KitchenResult(
            Path(generation.output_path),
            {
                "_prompt_cache_hit": hit,
                "_prompt_cache_published": published,
            },
        )

    monkeypatch.setattr(runtime, "_execute", execute)
    first = SimpleNamespace(output_path=tmp_path / "cold.mp4", seed=1)
    second = SimpleNamespace(output_path=tmp_path / "warm.mp4", seed=2)

    cold = runtime.generate(first, progress=lambda *_args: None, check_cancelled=lambda: None)
    warm = runtime.generate(second, progress=lambda *_args: None, check_cancelled=lambda: None)

    assert cold.metadata["cache"]["prompt_hit"] is False
    assert cold.metadata["cache"]["prompt_published"] is (cache_policy == "prompt")
    assert cold.metadata["cache"]["prompt"]["entries"] == (1 if cache_policy == "prompt" else 0)
    assert (cold.metadata["cache"]["prompt"]["bytes"] > 0) is (cache_policy == "prompt")
    assert warm.metadata["cache"]["prompt_hit"] is second_hit
    assert warm.metadata["cache"]["prompt_published"] is False
    assert warm.metadata["cache"]["policy"] == cache_policy
    assert warm.metadata["cache"]["prompt"]["hits"] == expected_hits
    assert warm.metadata["cache"]["prompt"]["misses"] == expected_misses


def test_cold_materialization_reports_transformer_phases_before_first_payload_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The long meta-shell build is visible before CPU SafeTensors streaming starts."""

    import diffusers
    import transformers

    phases: list[tuple[float, str | None]] = []
    operations: list[str] = []
    payload_handle = object()

    class _PayloadContext:
        def __enter__(self):
            operations.append("open_payload")
            return payload_handle

        def __exit__(self, *_args):
            operations.append("close_payload")

    class _Component:
        def eval(self):
            return self

    checkpoint = SimpleNamespace(identity=SimpleNamespace(path=tmp_path / "checkpoint"))
    text = SimpleNamespace(identity=SimpleNamespace(path=tmp_path / "text"))
    upscaler = SimpleNamespace(identity=SimpleNamespace(path=tmp_path / "upscaler"))
    runtime = object.__new__(kitchen_module.LTX23KitchenRuntime)
    runtime.request = SimpleNamespace(
        operation="ltx23_dev_t2v",
        plans={
            "pipeline_support": SimpleNamespace(root=tmp_path / "support"),
            "checkpoint": checkpoint,
            "text_encoder": text,
            "latent_upscaler": upscaler,
            "model_lora": SimpleNamespace(identity=SimpleNamespace(path=tmp_path / "model_lora")),
            "text_lora": SimpleNamespace(identity=SimpleNamespace(path=tmp_path / "text_lora")),
        },
    )

    monkeypatch.setattr(
        kitchen_module,
        "inspect_ltx23_av_artifact",
        lambda *_args, **_kwargs: operations.append("inspect") or object(),
    )
    monkeypatch.setattr(
        kitchen_module,
        "build_ltx23_av_meta_shell",
        lambda _contract: operations.append("build_shell") or _Component(),
    )
    monkeypatch.setattr(
        kitchen_module,
        "plan_ltx23_av_materialization",
        lambda *_args, **_kwargs: operations.append("plan") or object(),
    )

    def materialize_transformer(shell, _plan, *, payload_handle):
        operations.append(f"transformer_payload:{payload_handle is expected_payload_handle}")
        return shell

    expected_payload_handle = payload_handle
    monkeypatch.setattr(
        kitchen_module,
        "materialize_ltx23_av",
        materialize_transformer,
    )
    monkeypatch.setattr(
        kitchen_module,
        "open_ltx23_av_payload",
        lambda _path: operations.append("open_payload_call") or _PayloadContext(),
    )
    monkeypatch.setattr(
        kitchen_module,
        "build_ltx23_connector_meta_shell",
        lambda _contract: _Component(),
    )
    monkeypatch.setattr(
        kitchen_module, "plan_ltx23_connector_materialization", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        kitchen_module,
        "materialize_ltx23_connectors",
        lambda shell, _plan, *, payload_handle: (
            operations.append(f"connector_payload:{payload_handle is expected_payload_handle}")
            or shell
        ),
    )
    monkeypatch.setattr(kitchen_module, "build_ltx23_media_shell", lambda _component: _Component())

    def plan_media(_artifact, component, _shell, *, payload_handle=None):
        operations.append(f"media_plan:{component}:{payload_handle is expected_payload_handle}")
        return component

    def materialize_media(shell, component, *, payload_handle=None):
        operations.append(f"media_payload:{component}:{payload_handle is expected_payload_handle}")
        return shell

    monkeypatch.setattr(kitchen_module, "plan_ltx23_media_component", plan_media)
    monkeypatch.setattr(kitchen_module, "materialize_ltx23_media_component", materialize_media)
    monkeypatch.setattr(
        kitchen_module, "plan_ltx23_gemma_mixed_text_encoder", lambda *_args: object()
    )
    monkeypatch.setattr(
        kitchen_module, "load_ltx23_gemma_mixed_text_encoder", lambda *_args: _Component()
    )
    monkeypatch.setattr(kitchen_module, "inspect_ltx23_model_lora", lambda *_args: object())
    monkeypatch.setattr(
        kitchen_module, "install_ltx23_model_lora", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(kitchen_module, "plan_ltx23_gemma_text_lora", lambda *_args: object())
    monkeypatch.setattr(
        kitchen_module, "install_ltx23_gemma_text_lora", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(kitchen_module, "ltx23_module_physical_bytes", lambda _value: 0)

    class _Processor:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return object()

    class _Scheduler:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return object()

    monkeypatch.setattr(
        transformers.processing_utils.ProcessorMixin,
        "from_pretrained",
        classmethod(lambda cls, *_args, **_kwargs: _Processor()),
    )
    monkeypatch.setattr(diffusers, "FlowMatchEulerDiscreteScheduler", _Scheduler)

    components = runtime._materialize(
        lambda: None, lambda value, message: phases.append((value, message))
    )

    assert operations == [
        "inspect",
        "build_shell",
        "plan",
        "open_payload_call",
        "open_payload",
        "transformer_payload:True",
        "connector_payload:True",
        "media_plan:video_vae:True",
        "media_payload:video_vae:True",
        "media_plan:audio_vae:True",
        "media_payload:audio_vae:True",
        "media_plan:vocoder:True",
        "media_payload:vocoder:True",
        "close_payload",
        "media_plan:latent_upsampler:False",
        "media_payload:latent_upsampler:False",
    ]
    assert phases[:4] == [
        (0.01, "Inspecting LTX transformer artifact"),
        (0.015, "Building LTX transformer shell"),
        (0.02, "Planning LTX transformer materialization"),
        (0.025, "Materializing LTX transformer"),
    ]
    assert phases[4:7] == [
        (0.027, "Building LTX connector shell"),
        (0.028, "Planning LTX connector payload mapping"),
        (0.03, "Materializing LTX connector payload"),
    ]
    assert [message for _value, message in phases if message and "video VAE" in message] == [
        "Building LTX video VAE shell",
        "Planning LTX video VAE payload mapping",
        "Materializing LTX video VAE payload",
    ]
    assert [value for value, _message in phases] == sorted(value for value, _message in phases)
    assert components["transformer"].__class__ is _Component


def test_worker_session_reuses_runtime_and_rejects_mismatched_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = [tmp_path / "one.mp4", tmp_path / "two.mp4"]
    initial, second, mismatch = object(), object(), object()
    request_data = {initial: "recipe-a", second: "recipe-a", mismatch: "recipe-b"}
    generations = {
        initial: {
            "output_path": str(outputs[0]),
            "prompt": "one",
            "width": 1,
            "height": 1,
            "duration_seconds": 1.0,
            "requested_num_frames": 26,
            "num_frames": 25,
            "seed": 1,
            "start_image_path": None,
            "end_image_path": None,
            "start_image_identity": None,
            "end_image_identity": None,
        },
        second: {
            "output_path": str(outputs[1]),
            "prompt": "two",
            "width": 1,
            "height": 1,
            "duration_seconds": 1.0,
            "requested_num_frames": 26,
            "num_frames": 25,
            "seed": 2,
            "start_image_path": None,
            "end_image_path": None,
            "start_image_identity": None,
            "end_image_identity": None,
        },
        mismatch: {
            "output_path": str(tmp_path / "bad.mp4"),
            "prompt": "bad",
            "width": 1,
            "height": 1,
            "duration_seconds": 1.0,
            "requested_num_frames": 26,
            "num_frames": 25,
            "seed": 3,
            "start_image_path": None,
            "end_image_path": None,
            "start_image_identity": None,
            "end_image_identity": None,
        },
    }
    results: list[dict[str, object]] = []
    created: list[object] = []

    monkeypatch.setattr(
        worker_module,
        "_validate_bound_payload",
        lambda payload, _secret: (
            "generate",
            {"recipe": request_data[payload]},
            generations[payload],
            "cuda",
            "none",
            f"binding-{generations[payload]['seed']}",
        ),
    )
    monkeypatch.setattr(
        "latentslate_engine.ltx23_kitchen_recipe.rehydrate_ltx23_kitchen_runtime_request",
        lambda value: SimpleNamespace(operation="ltx23_dev_t2v", fingerprint=value["recipe"]),
    )

    class _Generation:
        def __init__(self, *args):
            self.output_path = args[1]

    class _Runtime:
        def __init__(self, request, **_kwargs):
            created.append(request)
            self.cache_policy = "none"

        def generate(self, generation, **_kwargs):
            Path(generation.output_path).write_bytes(b"mp4")
            return SimpleNamespace(metadata={"cache": {"pipeline_warm": bool(results)}})

        def unload(self):
            pass

    monkeypatch.setattr(kitchen_module, "LTX23KitchenGeneration", _Generation)
    monkeypatch.setattr(kitchen_module, "LTX23KitchenRuntime", _Runtime)
    monkeypatch.setattr(kitchen_module, "validate_ltx23_kitchen_generation", lambda *_args: None)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    context = SimpleNamespace(publish_progress=lambda *_args: None, binding="")
    handler = worker_module._LTX23KitchenHandler(_SECRET)
    initial_command = handler.bind_initial(initial, context)
    session = handler.load(initial_command, context)
    results.append(dict(handler.execute(session, initial_command, context, cold=True)))
    second_command = handler.bind_command(second, session, context)
    results.append(dict(handler.execute(session, second_command, context, cold=False)))
    with pytest.raises(ValueError, match="does not match"):
        handler.bind_command(mismatch, session, context)
    assert len(created) == 1
    assert [item["request_binding"] for item in results] == ["binding-1", "binding-2"]
    assert results[1]["metadata"]["cache"]["pipeline_warm"] is True


def test_poisoned_ltx_child_hard_exits_without_normal_unload_or_finalizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = managed_module._persistent_paths(_paths(tmp_path))
    paths.request.write_text("{}", encoding="utf-8")
    paths.start_gate.touch()
    events: list[object] = []

    class _Session:
        def __del__(self):
            events.append("finalizer")

    class _Handler:
        def protocol_error(self, reason):
            return RuntimeError(reason)

        def bind_initial(self, _payload, _context):
            events.append("bind")
            return object()

        def load(self, _command, _context):
            events.append("load")
            return _Session()

        def execute(self, _session, _command, _context, *, cold):
            assert cold is True
            events.append("execute")
            raise kitchen_module.LTX23KitchenWorkerPoisoned("failed_fill_quiescence_failed")

        def terminal_exit_status(self, exc, _session, _context):
            events.append(("terminal", exc.reason))
            return persistent_child_module.PersistentChildTerminalExit(86, exc.reason)

        def failure_result(self, exc, _context):
            events.append(("failure", exc.reason))
            return {"ok": False, "poison_reason": exc.reason}

        def unload(self, _session, _context):
            events.append("unload")

    exits: list[int] = []
    monkeypatch.setattr(persistent_child_module, "_hard_exit_process", exits.append)
    monkeypatch.setattr(persistent_child_module, "_terminal_exit_retained", None)

    code = persistent_child_module.run_persistent_child(
        paths, _Handler(), maximum_bytes=1024, heartbeat_seconds=0.01
    )

    assert code == 86
    assert exits == [86]
    assert "unload" not in events
    assert "finalizer" not in events
    assert events[:3] == ["bind", "load", "execute"]
    assert ("terminal", "failed_fill_quiescence_failed") in events
    assert ("failure", "failed_fill_quiescence_failed") in events
    assert json.loads(paths.result.read_text(encoding="utf-8")) == {
        "ok": False,
        "poison_reason": "failed_fill_quiescence_failed",
    }
    retained = persistent_child_module._terminal_exit_retained
    assert retained is not None and any(isinstance(item, _Session) for item in retained)

    ordinary_session = SimpleNamespace(runtime=SimpleNamespace(terminal_poison_reason=lambda: None))
    assert (
        worker_module._LTX23KitchenHandler(_SECRET).terminal_exit_status(
            RuntimeError("ordinary failure"), ordinary_session, SimpleNamespace()
        )
        is None
    )


def test_av_constructor_cleanup_failure_is_signed_and_hard_exits_without_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = managed_module._persistent_paths(_paths(tmp_path))
    paths.request.write_text("{}", encoding="utf-8")
    paths.start_gate.touch()
    output = tmp_path / "must-not-publish.mp4"
    events: list[object] = []

    class _UnsafeBackend:
        def close(self):
            events.append("backend_close_attempt")
            raise RuntimeError("synthetic initialization quiescence loss")

        def __del__(self):
            events.append("backend_finalizer")

    class _SourceOwner:
        def close(self):
            events.append("source_close")

        def __del__(self):
            events.append("source_finalizer")

    class _Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.root = nn.Parameter(torch.ones(1))
            self.transformer_blocks = nn.ModuleList(
                [nn.Linear(1, 1, bias=False) for _ in range(48)]
            )
            # Constructor setup is monkeypatched below, so only the required
            # source-backed boundary proof is needed before entering it.
            self._latentslate_ltx23_av_source_descriptors = {-1: object()}
            self._latentslate_ltx23_av_source_plan = object()

    def fail_after_native_allocation(self):
        self._dynamic = _UnsafeBackend()
        self._base_file_handle = _SourceOwner()
        self._base_file_handle_opened = 1
        self._base_file_handle_closed = 0
        raise RuntimeError("synthetic setup failure after native allocation")

    monkeypatch.setattr(
        kitchen_module._LTX23TransformerResidency,
        "_initialize_dynamic_backend",
        fail_after_native_allocation,
    )
    monkeypatch.setattr(kitchen_module.torch.cuda, "current_device", lambda: 0)

    class _Runtime:
        def terminal_poison_reason(self):
            return None

    session = SimpleNamespace(runtime=_Runtime())

    class _Handler(worker_module._LTX23KitchenHandler):
        def bind_initial(self, _payload, context):
            context.binding = "binding"
            self.failure.binding = "binding"
            return object()

        def load(self, _command, _context):
            events.append("load")
            return session

        def execute(self, _session, _command, _context, *, cold):
            assert cold is True
            self.failure.stage = "generate"
            kitchen_module._LTX23TransformerResidency(
                _Transformer(), torch.device("cuda"), resident_weight_budget_bytes=0
            )
            raise AssertionError("constructor poison must escape")

        def unload(self, _session, _context):
            events.append("ordinary_unload")

    exits: list[int] = []
    monkeypatch.setattr(persistent_child_module, "_hard_exit_process", exits.append)
    monkeypatch.setattr(persistent_child_module, "_terminal_exit_retained", None)

    code = persistent_child_module.run_persistent_child(
        paths,
        _Handler(_SECRET),
        maximum_bytes=1024,
        heartbeat_seconds=0.01,
    )

    assert code == 86
    assert exits == [86]
    assert events == ["load", "backend_close_attempt"]
    assert "ordinary_unload" not in events
    assert "source_close" not in events
    assert "backend_finalizer" not in events
    assert "source_finalizer" not in events
    assert not output.exists()
    retained = persistent_child_module._terminal_exit_retained
    assert retained is not None
    poison = next(
        item for item in retained if isinstance(item, kitchen_module.LTX23KitchenWorkerPoisoned)
    )
    assert poison.reason == "ltx23_av_dynamic_initialization_cleanup_failed"

    accepted = managed_module._read_result(paths.result, output, "binding", _SECRET)
    assert accepted["terminal_exit_code"] == 86
    assert accepted["poison_reason"] == ("ltx23_av_dynamic_initialization_cleanup_failed")
    assert accepted["aimdo_counters"] is None


def test_generation_failure_then_raw_av_barrier_failure_uses_canonical_hard_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = managed_module._persistent_paths(_paths(tmp_path))
    paths.request.write_text("{}", encoding="utf-8")
    paths.start_gate.touch()
    output = tmp_path / "must-not-publish.mp4"
    events: list[str] = []

    class _Backend:
        def diagnostics(self):
            if residency._barrier_failed:
                raise AssertionError("native diagnostics after failed quiescence")
            events.append("backend_diagnostics")
            return {}

        def close(self):
            events.append("backend_close")

        def __del__(self):
            events.append("backend_finalizer")

    class _SourceOwner:
        def close(self):
            events.append("source_close")

        def __del__(self):
            events.append("source_finalizer")

    class _Handle:
        def remove(self):
            events.append("hook_remove")

    transformer = nn.Linear(1, 1, bias=False)
    backend = _Backend()
    source_owner = _SourceOwner()
    residency = object.__new__(kitchen_module._LTX23TransformerResidency)
    residency.transformer = transformer
    residency.device = torch.device("cuda")
    residency._handles = [_Handle()]
    residency._closed = False
    residency._owner_thread = None
    residency._executing = False
    residency._barrier_failed = False
    residency._streamed_binding = None
    residency._resident = {}
    residency._root_binding = None
    residency._dynamic = backend
    residency._base_file_handle = source_owner
    residency._base_file_handle_opened = 1
    residency._base_file_handle_closed = 0

    request = SimpleNamespace(operation="ltx23_dev_t2v", fingerprint="f" * 64)
    runtime = kitchen_module.LTX23KitchenRuntime(request, device="cuda")
    components = {"transformer": transformer, "sentinel": object()}
    runtime._components = components
    runtime._transformer_residency = residency
    runtime._active_text_stage = None

    def fail_generation(*_args, **_kwargs):
        events.append("ordinary_generation_failure")
        raise RuntimeError("ordinary generation failure")

    runtime._execute = fail_generation
    monkeypatch.setattr(kitchen_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        kitchen_module.torch.cuda,
        "synchronize",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("raw CUDA barrier failure")),
    )
    monkeypatch.setattr(
        kitchen_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(kitchen_module, "validate_ltx23_kitchen_generation", lambda *_args: None)
    monkeypatch.setattr(
        kitchen_module,
        "PhaseMemoryTelemetry",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    generation = {
        "prompt": "scene",
        "output_path": str(output),
        "width": 1280,
        "height": 704,
        "duration_seconds": 1.0,
        "requested_num_frames": 26,
        "num_frames": 25,
        "seed": 7,
        "start_image_path": None,
        "end_image_path": None,
        "start_image_identity": None,
        "end_image_identity": None,
    }
    command = worker_module._BoundCommand(
        "generate", request, generation, "cuda", "none", "binding"
    )
    session = worker_module._LoadedSession(
        request,
        runtime,
        kitchen_module.LTX23KitchenGeneration,
        lambda *_args: None,
    )

    class _Handler(worker_module._LTX23KitchenHandler):
        def bind_initial(self, _payload, context):
            context.binding = "binding"
            self.failure.binding = "binding"
            return command

        def load(self, _command, _context):
            events.append("load")
            return session

        def unload(self, _session, _context):
            events.append("ordinary_unload")

    exits: list[int] = []
    monkeypatch.setattr(persistent_child_module, "_hard_exit_process", exits.append)
    monkeypatch.setattr(persistent_child_module, "_terminal_exit_retained", None)

    code = persistent_child_module.run_persistent_child(
        paths,
        _Handler(_SECRET),
        maximum_bytes=1024,
        heartbeat_seconds=0.01,
    )

    assert code == 86
    assert exits == [86]
    assert events == [
        "load",
        "ordinary_generation_failure",
        "backend_diagnostics",
        "hook_remove",
    ]
    assert not output.exists()
    assert runtime._cache.prompt.status()["entries"] == 0
    assert runtime._components is components
    assert runtime._transformer_residency is residency
    assert residency._dynamic is backend
    assert residency._base_file_handle is source_owner
    diagnostic_events = list(events)
    assert residency.failure_diagnostics() == {}
    assert events == diagnostic_events
    assert "ordinary_unload" not in events
    assert "backend_close" not in events
    assert "source_close" not in events
    assert "backend_finalizer" not in events
    assert "source_finalizer" not in events
    retained = persistent_child_module._terminal_exit_retained
    assert retained is not None and session in retained

    accepted = managed_module._read_result(paths.result, output, "binding", _SECRET)
    assert accepted["error_type"] == "LTX23KitchenWorkerPoisoned"
    assert accepted["terminal_exit_code"] == 86
    assert accepted["poison_reason"] == "device_quiescence_failed"


def test_post_generation_verify_failure_then_cleanup_poison_hard_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = managed_module._persistent_paths(_paths(tmp_path))
    paths.request.write_text("{}", encoding="utf-8")
    paths.start_gate.touch()
    output = tmp_path / "missing-after-generation.mp4"
    events: list[str] = []

    class _Backend:
        def close(self):
            events.append("backend_close")

        def __del__(self):
            events.append("backend_finalizer")

    class _SourceOwner:
        def close(self):
            events.append("source_close")

        def __del__(self):
            events.append("source_finalizer")

    transformer = nn.Linear(1, 1, bias=False)
    backend = _Backend()
    source_owner = _SourceOwner()
    residency = object.__new__(kitchen_module._LTX23TransformerResidency)
    residency.transformer = transformer
    residency.device = torch.device("cuda")
    residency._handles = []
    residency._closed = False
    residency._owner_thread = None
    residency._executing = False
    residency._barrier_failed = False
    residency._streamed_binding = None
    residency._resident = {}
    residency._root_binding = None
    residency._dynamic = backend
    residency._base_file_handle = source_owner
    residency._base_file_handle_opened = 1
    residency._base_file_handle_closed = 0

    request = SimpleNamespace(operation="ltx23_dev_t2v", fingerprint="f" * 64)
    runtime = kitchen_module.LTX23KitchenRuntime(request, device="cuda")
    components = {"transformer": transformer, "sentinel": object()}
    runtime._components = components
    runtime._transformer_residency = residency
    runtime._active_text_stage = None
    runtime.generate = lambda *_args, **_kwargs: kitchen_module.LTX23KitchenResult(
        output,
        {"cache": {"pipeline_warm": True}},
    )

    synchronize_calls = 0

    def fail_barrier(*_args):
        nonlocal synchronize_calls
        synchronize_calls += 1
        raise RuntimeError("raw cleanup CUDA barrier failure")

    monkeypatch.setattr(kitchen_module.torch.cuda, "synchronize", fail_barrier)
    generation = {
        "prompt": "scene",
        "output_path": str(output),
        "width": 1280,
        "height": 704,
        "duration_seconds": 1.0,
        "requested_num_frames": 26,
        "num_frames": 25,
        "seed": 7,
        "start_image_path": None,
        "end_image_path": None,
        "start_image_identity": None,
        "end_image_identity": None,
    }
    command = worker_module._BoundCommand(
        "generate", request, generation, "cuda", "none", "binding"
    )
    session = worker_module._LoadedSession(
        request,
        runtime,
        kitchen_module.LTX23KitchenGeneration,
        lambda *_args: None,
    )

    class _Handler(worker_module._LTX23KitchenHandler):
        def bind_initial(self, _payload, context):
            context.binding = "binding"
            self.failure.binding = "binding"
            return command

        def load(self, _command, _context):
            return session

    exits: list[int] = []
    monkeypatch.setattr(persistent_child_module, "_hard_exit_process", exits.append)
    monkeypatch.setattr(persistent_child_module, "_terminal_exit_retained", None)

    code = persistent_child_module.run_persistent_child(
        paths,
        _Handler(_SECRET),
        maximum_bytes=1024,
        heartbeat_seconds=0.01,
    )

    assert code == 86
    assert exits == [86]
    assert synchronize_calls == 1
    assert not output.exists()
    assert runtime._cache.prompt.status()["entries"] == 0
    assert runtime._components is components
    assert runtime._transformer_residency is residency
    assert residency._dynamic is backend
    assert residency._base_file_handle is source_owner
    assert events == []
    retained = persistent_child_module._terminal_exit_retained
    assert retained is not None and session in retained
    retained_exceptions = [item for item in retained if isinstance(item, BaseException)]
    assert len(retained_exceptions) == 2
    assert type(retained_exceptions[0]) is RuntimeError
    assert str(retained_exceptions[0]) == ("LTX 2.3 Kitchen worker did not publish an MP4")
    assert isinstance(retained_exceptions[1], kitchen_module.LTX23KitchenWorkerPoisoned)

    accepted = managed_module._read_result(paths.result, output, "binding", _SECRET)
    assert accepted["error_type"] == "RuntimeError"
    assert accepted["failure_stage"] == "verify_output"
    assert accepted["cleanup_stage"] == "unload_runtime"
    assert accepted["terminal_exit_code"] == 86
    assert accepted["poison_reason"] == "device_quiescence_failed"
    assert accepted["poison_origin"] == "cleanup"


def test_persistent_child_cleanup_preserves_primary_materialization_stage(
    tmp_path: Path,
) -> None:
    paths = managed_module._persistent_paths(_paths(tmp_path))
    paths.request.write_text("{}", encoding="utf-8")
    paths.start_gate.touch()
    output = tmp_path / "must-not-publish.mp4"
    events: list[str] = []

    class _Runtime:
        cache_policy = "none"

        def generate(self, _generation, *, progress, check_cancelled):
            check_cancelled()
            progress(0.06, "Materializing LTX text encoder")
            raise ValueError("synthetic source descriptor mismatch")

        def failure_aimdo_counters(self):
            return None

        def terminal_poison_reason(self):
            return None

        def unload(self):
            events.append("runtime_unload")
            raise RuntimeError("ordinary nonterminal cleanup failure")

    request = SimpleNamespace(operation="ltx23_dev_t2v")
    runtime = _Runtime()
    generation = {
        "prompt": "scene",
        "output_path": str(output),
        "width": 1280,
        "height": 704,
        "duration_seconds": 1.0,
        "requested_num_frames": 26,
        "num_frames": 25,
        "seed": 7,
        "start_image_path": None,
        "end_image_path": None,
        "start_image_identity": None,
        "end_image_identity": None,
    }
    command = worker_module._BoundCommand(
        "generate", request, generation, "cuda", "none", "binding"
    )
    session = worker_module._LoadedSession(
        request,
        runtime,
        kitchen_module.LTX23KitchenGeneration,
        lambda *_args: None,
    )

    class _Handler(worker_module._LTX23KitchenHandler):
        def bind_initial(self, _payload, context):
            context.binding = "binding"
            self.failure.binding = "binding"
            return command

        def load(self, _command, _context):
            return session

    code = persistent_child_module.run_persistent_child(
        paths,
        _Handler(_SECRET),
        maximum_bytes=1024,
        heartbeat_seconds=0.01,
    )

    assert code == 1
    assert events == ["runtime_unload"]
    assert not output.exists()
    accepted = managed_module._read_result(paths.result, output, "binding", _SECRET)
    assert accepted["error_type"] == "ValueError"
    assert accepted["failure_stage"] == "materialize_text_encoder"
    assert accepted["cleanup_stage"] == "unload_runtime"


def test_managed_replaces_hard_exited_poisoned_child_on_next_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ManagedLTX23KitchenRuntime(_Request())  # type: ignore[arg-type]
    paths = _paths(tmp_path)

    class _HardExitSupervisor(_FakeSupervisor):
        def terminate(self):
            self.events.append("terminate")
            if self.process.exit_code is None:
                self.process.exit_code = 1

    first = _HardExitSupervisor(paths, pid=801)
    second = _HardExitSupervisor(paths, pid=802)
    supervisors = iter((first, second))
    secrets_seen: list[bytes] = []
    output = tmp_path / "fresh.mp4"
    reads = 0

    def supervisor_factory(_paths_value, secret):
        secrets_seen.append(secret)
        return next(supervisors), "expandable_segments:True"

    def wait_for_result(supervisor, *_args):
        if supervisor is first:
            supervisor.process.exit_code = 86

    def read_result(*_args):
        nonlocal reads
        reads += 1
        if reads == 1:
            return {
                "ok": False,
                "error_type": "LTX23KitchenWorkerPoisoned",
                "error": "poisoned",
                "failure_stage": "offload_text",
                "failure_location": "ltx23_kitchen.generate",
                "error_fingerprint": "a" * 64,
                "cleanup_stage": "offload_text",
                "aimdo_counters": None,
                "terminal_exit_code": 86,
                "poison_reason": "device_quiescence_failed",
                "poison_origin": "primary",
            }
        output.write_bytes(b"mp4")
        return {
            "ok": True,
            "output_size_bytes": 3,
            "metadata": _metadata(_Request()),
            "allocator_policy": "expandable_segments:True",
        }

    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(managed_module, "_supervisor", supervisor_factory)
    monkeypatch.setattr(managed_module, "_wait_for_result", wait_for_result)
    monkeypatch.setattr(managed_module, "_read_result", read_result)
    monkeypatch.setattr(managed_module, "_validate_metadata", lambda *_args: None)
    monkeypatch.setattr(
        managed_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )

    with pytest.raises(RuntimeError, match="worker failed"):
        runtime.generate(
            prompt="first",
            output_path=output,
            width=1280,
            height=704,
            duration_seconds=1,
            seed=1,
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )
    assert runtime.status()["loaded"] is False
    assert runtime.status()["last_worker"]["exit_code"] == 86

    result = runtime.generate(
        prompt="second",
        output_path=output,
        width=1280,
        height=704,
        duration_seconds=1,
        seed=2,
        progress=lambda *_args: None,
        check_cancelled=lambda: None,
    )
    assert result.worker_pid == 802
    assert runtime.status()["loaded"] is True
    assert first.events[:4] == ["start", "terminate", "close", "cleanup_session"]
    assert second.events[:3] == ["start", "cleanup_job"]
    assert len(secrets_seen) == 2 and secrets_seen[0] != secrets_seen[1]


def test_cancellation_terminates_tree_and_removes_partial_output(tmp_path: Path, monkeypatch):
    runtime = ManagedLTX23KitchenRuntime(_Request())  # type: ignore[arg-type]
    paths, events, output = _paths(tmp_path), [], tmp_path / "output.mp4"
    supervisor = _FakeSupervisor(paths, pid=99, events=events)

    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(
        managed_module,
        "_supervisor",
        lambda *_args: (supervisor, "expandable_segments:True"),
    )

    def cancel(*_args):
        output.write_bytes(b"partial")
        raise asyncio.CancelledError

    monkeypatch.setattr(managed_module, "_wait_for_result", cancel)
    runtime._last_cache = _cleared_cache_status(policy="none")
    with pytest.raises(asyncio.CancelledError):
        runtime.generate(
            prompt="scene",
            output_path=output,
            width=768,
            height=512,
            duration_seconds=1,
            seed=7,
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )
    assert events == ["start", "terminate", "close", "cleanup_session"]
    assert not output.exists()
    assert runtime.status()["last_worker"]["outcome"] == "canceled"
    assert runtime.status()["cache"] == managed_module._empty_cache_status("none")


def test_tool_cancellation_is_classified_without_importing_the_tools_layer(
    tmp_path: Path, monkeypatch
):
    runtime = ManagedLTX23KitchenRuntime(_Request())  # type: ignore[arg-type]
    paths, events = _paths(tmp_path), []

    class ToolCancelled(Exception):
        pass

    supervisor = _FakeSupervisor(paths, pid=101, events=events)

    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(
        managed_module,
        "_supervisor",
        lambda *_args: (supervisor, "expandable_segments:True"),
    )
    monkeypatch.setattr(
        managed_module,
        "_wait_for_result",
        lambda *_args: (_ for _ in ()).throw(ToolCancelled()),
    )
    with pytest.raises(ToolCancelled):
        runtime.generate(
            prompt="scene",
            output_path=tmp_path / "output.mp4",
            width=768,
            height=512,
            duration_seconds=1,
            seed=7,
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )
    assert runtime.status()["last_worker"]["outcome"] == "canceled"


def test_generation_failure_resets_dead_worker_cache_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ManagedLTX23KitchenRuntime(  # type: ignore[arg-type]
        _Request(), cache_policy="prompt"
    )
    paths = _paths(tmp_path)
    supervisor = _FakeSupervisor(paths, pid=102)
    monkeypatch.setattr(managed_module, "_paths", lambda _output: paths)
    monkeypatch.setattr(
        managed_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(
        managed_module,
        "_supervisor",
        lambda *_args: (supervisor, "expandable_segments:True"),
    )
    monkeypatch.setattr(
        managed_module,
        "_wait_for_result",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("worker failed")),
    )
    runtime._last_cache = _cleared_cache_status()

    with pytest.raises(RuntimeError, match="worker failed"):
        runtime.generate(
            prompt="scene",
            output_path=tmp_path / "failure.mp4",
            width=768,
            height=512,
            duration_seconds=1,
            seed=7,
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )

    assert runtime.status()["loaded"] is False
    assert runtime.status()["last_worker"]["outcome"] == "failed"
    assert runtime.status()["cache"] == managed_module._empty_cache_status("prompt")


def test_generation_operation_identity_and_exact_temporal_alignment(tmp_path: Path):
    assert managed_module.frames_for_duration(1.0) == 25
    assert managed_module.frames_for_duration(1.05) == 25
    assert managed_module.frames_for_duration(1.28) == 33
    assert managed_module.frames_for_duration(5.0) == 121
    assert managed_module.frames_for_duration(10.0) == 249
    with pytest.raises(ValueError, match="must not receive"):
        managed_module._validate_generation(
            "ltx23_dev_t2v",
            {
                "prompt": "scene",
                "width": 768,
                "height": 512,
                "duration_seconds": 1.0,
                "requested_num_frames": 26,
                "num_frames": 25,
                "seed": 1,
                "start_image_path": "guide.png",
                "end_image_path": None,
                "start_image_identity": {
                    "size_bytes": 1,
                    "mtime_ns": 1,
                    "sha256": "0" * 64,
                },
                "end_image_identity": None,
                "output_path": str(tmp_path / "out.mp4"),
            },
        )
    with pytest.raises(ValueError, match="frame count"):
        managed_module._validate_generation(
            "ltx23_distilled_flf",
            {
                "prompt": "scene",
                "width": 768,
                "height": 512,
                "duration_seconds": 1.0,
                "requested_num_frames": 26,
                "num_frames": 33,
                "seed": 1,
                "start_image_path": None,
                "end_image_path": None,
                "start_image_identity": None,
                "end_image_identity": None,
                "output_path": str(tmp_path / "out.mp4"),
            },
        )


def test_failure_result_must_be_bound_and_output_cleanup_is_owned(tmp_path: Path):
    result = tmp_path / "result.json"
    result.write_text(
        '{"schema_version":1,"ok":false,"request_binding":"other","error_type":"RuntimeError","error":"private"}',
        encoding="utf-8",
    )
    assert (
        managed_module._worker_error(result, 1, "expected", _SECRET)
        == "LTX 2.3 Kitchen worker exited with code 1"
    )
    staging = tmp_path / ".out.mp4.part.tmp.mp4"
    staging.write_bytes(b"partial")
    # Generic IPC cleanup is exact and must not delete similarly named encoder
    # artifacts outside the worker transport namespace.
    assert staging.exists()


def test_worker_removes_generated_output_when_success_signing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "unsigned.mp4"
    generation = {
        "prompt": "scene",
        "output_path": str(output),
        "width": 768,
        "height": 512,
        "num_frames": 25,
        "requested_num_frames": 26,
        "duration_seconds": 1.0,
        "seed": 7,
        "start_image_path": None,
        "end_image_path": None,
        "start_image_identity": None,
        "end_image_identity": None,
    }

    class _Runtime:
        def generate(self, built, **_kwargs):
            assert built.output_path == output
            output.write_bytes(b"complete-but-unsigned")
            return SimpleNamespace(metadata={"cache": {"prompt_published": True}})

    def build(*args):
        return SimpleNamespace(output_path=Path(args[1]))

    monkeypatch.setattr(
        worker_module,
        "_signed_result",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("signing failed")),
    )
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    request = _Request()
    session = worker_module._LoadedSession(request, _Runtime(), build, lambda *_args: None)
    command = worker_module._BoundCommand(
        "generate", request, generation, "cuda", "prompt", "binding"
    )
    context = SimpleNamespace(publish_progress=lambda *_args: None)

    with pytest.raises(RuntimeError, match="signing failed"):
        worker_module._LTX23KitchenHandler(_SECRET).execute(session, command, context, cold=True)

    assert not output.exists()


def test_worker_preserves_primary_enhancement_stage_and_safe_aimdo_failure_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "failed-enhancement.mp4"
    generation = {
        "prompt": "scene",
        "output_path": str(output),
        "width": 768,
        "height": 512,
        "num_frames": 25,
        "requested_num_frames": 26,
        "duration_seconds": 1.0,
        "seed": 7,
        "start_image_path": None,
        "end_image_path": None,
        "start_image_identity": None,
        "end_image_identity": None,
    }
    counters = _aimdo_failure_counters()

    class _Runtime:
        def generate(self, _built, *, progress, **_kwargs):
            progress(0.1, "Enhancing prompt")
            progress(0.2, "Offloaded base Gemma (phase_s=1.0)")
            raise RuntimeError("enhancement failed")

        def failure_aimdo_counters(self):
            return counters

    def build(*args):
        return SimpleNamespace(output_path=Path(args[1]))

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    request = _Request()
    session = worker_module._LoadedSession(request, _Runtime(), build, lambda *_args: None)
    command = worker_module._BoundCommand(
        "generate", request, generation, "cuda", "prompt", "binding"
    )
    context = SimpleNamespace(publish_progress=lambda *_args: None, binding="binding")
    handler = worker_module._LTX23KitchenHandler(_SECRET)

    with pytest.raises(RuntimeError, match="enhancement failed") as raised:
        handler.execute(session, command, context, cold=True)

    assert handler.failure.stage == "enhance_prompt"
    assert handler.failure.cleanup_stage == "offload_text"
    assert handler.failure.aimdo_counters == counters
    failure = handler.failure_result(raised.value, context)
    assert failure["failure_stage"] == "enhance_prompt"
    assert failure["cleanup_stage"] == "offload_text"
    assert failure["aimdo_counters"] == counters
    assert managed_module._valid_failure_diagnostic(failure)
    assert managed_module._valid_failure_aimdo_counters(failure["aimdo_counters"])


def test_failure_aimdo_counter_schema_is_exact_and_retained_in_worker_status() -> None:
    counters = _aimdo_failure_counters()
    assert kitchen_module._valid_bounded_aimdo_failure_counters(counters)
    assert managed_module._valid_failure_aimdo_counters(counters)
    assert (
        kitchen_module._bounded_aimdo_failure_counters(
            {"dynamic_vram": {**counters, "prefetch": False}}
        )
        == counters
    )

    for mutation in (
        {**counters, "faults": -1},
        {**counters, "backend": "engine_hooks"},
        {**counters, "unexpected": 1},
        {**counters, "copy_stream_count": True},
        {**counters, "host_source_pool_generation": True},
        {**counters, "host_source_pool_lane_count": 1},
        {**counters, "copy_fallback_reason": "host_buffer_runtime_failed: fixture"},
        {**counters, "base_file_read_calls": 1},
        {**counters, "base_file_source_live": True},
        {**counters, "base_file_handle_live": True},
        {**counters, "base_file_fallback_reason": "runtime_read_failed: fixture"},
        {**counters, "refill_failure_reason": "unknown"},
        {**counters, "refill_failure_reason": "unbound_root_exceeds_target"},
        {
            **counters,
            "refill_failure_reason": "unbound_root_exceeds_target",
            "refill_target_bytes": True,
            "refill_root_already_bound": False,
            "refill_resident_bytes": 0,
        },
    ):
        assert not kitchen_module._valid_bounded_aimdo_failure_counters(mutation)
        assert not managed_module._valid_failure_aimdo_counters(mutation)

    refill_failure = {
        **counters,
        "refill_failure_reason": "unbound_root_exceeds_target",
        "refill_target_bytes": 892_322_815,
        "refill_root_already_bound": False,
        "refill_resident_bytes": 0,
    }
    assert kitchen_module._valid_bounded_aimdo_failure_counters(refill_failure)
    assert managed_module._valid_failure_aimdo_counters(refill_failure)

    gathered = {
        **counters,
        "copy_strategy": "gathered_host_buffer",
        "copy_fallback_reason": None,
        "host_buffer_capacity_bytes": 1024,
        "host_buffer_allocations": 1,
        "host_buffer_live": True,
        "host_tensor_view_live": True,
        "host_buffer_transfer_pending": True,
        "gathered_misses": 2,
        "per_physical_misses": 0,
        "packed_source_bytes": 1024,
        "gathered_h2d_bytes": 2048,
        "pressure_direct_transfers": 1,
        "pressure_direct_bytes": 1536,
        "host_source_pool_warm_ram_pressure_bypasses": 1,
        "host_source_pool_warm_zero_delta_extend_refusals": 0,
        "host_source_pool_warm_registration_refusals": 0,
        "host_source_pool_temporary_ram_pressure_bypasses": 0,
        "host_source_pool_temporary_zero_delta_extend_refusals": 0,
        "host_source_pool_temporary_registration_refusals": 0,
        "host_source_pool_generation": 1,
        "host_source_pool_lane_count": 1,
        "host_source_pool_capacity_bytes": 1024,
        "host_source_pool_retained_slices": 0,
        "host_source_pool_retained_bytes": 0,
        "host_source_pool_temporary_slices": 1,
        "host_source_pool_temporary_bytes": 1024,
        "host_source_pool_hits": 0,
        "host_source_pool_misses": 1,
        "host_source_pool_stale_rejections": 0,
        "host_source_pool_poisoned": False,
        "host_source_pool_poison_reason": None,
        "host_source_registration": _host_source_registration(
            budget_bytes=1024,
            attempts=1,
            attempt_bytes=1024,
            successes=1,
            registered_bytes=1024,
            live_bytes=1024,
            peak_bytes=1024,
        ),
    }
    assert kitchen_module._valid_bounded_aimdo_failure_counters(gathered)
    assert managed_module._valid_failure_aimdo_counters(gathered)
    gathered_cpu_patch = {**gathered, "pageable_copy_bytes": 512}
    assert kitchen_module._valid_bounded_aimdo_failure_counters(gathered_cpu_patch)
    assert managed_module._valid_failure_aimdo_counters(gathered_cpu_patch)
    gathered_cpu_excess = {**gathered_cpu_patch, "pageable_copy_bytes": 1537}
    assert not kitchen_module._valid_bounded_aimdo_failure_counters(gathered_cpu_excess)
    assert not managed_module._valid_failure_aimdo_counters(gathered_cpu_excess)
    gathered_fabricated_direct = {
        **gathered,
        "host_source_pool_warm_ram_pressure_bypasses": 0,
    }
    assert not kitchen_module._valid_bounded_aimdo_failure_counters(
        gathered_fabricated_direct
    )
    assert not managed_module._valid_failure_aimdo_counters(gathered_fabricated_direct)
    gathered_excess_cause = {
        **gathered,
        "host_source_pool_warm_ram_pressure_bypasses": 3,
    }
    assert not kitchen_module._valid_bounded_aimdo_failure_counters(gathered_excess_cause)
    assert not managed_module._valid_failure_aimdo_counters(gathered_excess_cause)
    structural_registration = _host_source_registration(
        budget_bytes=1024,
        attempts=2,
        attempt_bytes=2048,
        successes=2,
        registered_bytes=2048,
        unregistered_bytes=1024,
        live_bytes=1024,
        peak_bytes=1024,
        state_proven=False,
    )
    structural = {
        **gathered,
        "poison_reason": "host_source_pool_structural_failure",
        "host_source_pool_poisoned": True,
        "host_source_pool_poison_reason": "host_buffer_rollback_failed",
        "host_source_registration": structural_registration,
    }
    produced = kitchen_module._bounded_aimdo_failure_counters(
        {"dynamic_vram": structural}
    )
    assert produced == structural
    assert managed_module._valid_failure_aimdo_counters(produced)
    excess_success = copy.deepcopy(structural)
    excess_success["host_source_registration"].update(
        attempts=3,
        attempt_bytes=3072,
        successes=3,
        registered_bytes=3072,
    )
    assert not kitchen_module._valid_bounded_aimdo_failure_counters(excess_success)
    assert not managed_module._valid_failure_aimdo_counters(excess_success)
    wrong_reason = copy.deepcopy(structural)
    wrong_reason["host_source_pool_poison_reason"] = "host_buffer_extend_failed"
    assert not kitchen_module._valid_bounded_aimdo_failure_counters(wrong_reason)
    assert not managed_module._valid_failure_aimdo_counters(wrong_reason)
    oversized_gather = {**gathered, "gathered_h2d_bytes": 2049}
    assert not kitchen_module._valid_bounded_aimdo_failure_counters(oversized_gather)
    assert not managed_module._valid_failure_aimdo_counters(oversized_gather)
    excess_pressure_transfers = {**gathered, "pressure_direct_transfers": 3}
    assert not kitchen_module._valid_bounded_aimdo_failure_counters(excess_pressure_transfers)
    assert not managed_module._valid_failure_aimdo_counters(excess_pressure_transfers)
    excess_pressure_bytes = {**gathered, "pressure_direct_bytes": 2049}
    assert not kitchen_module._valid_bounded_aimdo_failure_counters(excess_pressure_bytes)
    assert not managed_module._valid_failure_aimdo_counters(excess_pressure_bytes)

    process = SimpleNamespace(pid=7, poll=lambda: 86)
    status = managed_module._last_worker(
        process,
        "failed",
        tree_empty=True,
        allocator_policy="expandable_segments:True",
        worker_failure={
            "error_type": "RuntimeError",
            "stage": "enhance_prompt",
            "location": "ltx23_kitchen._enhance_prompt",
            "fingerprint": "a" * 64,
            "cleanup_stage": "offload_text",
            "aimdo_counters": counters,
        },
    )
    assert status["failure"]["cleanup_stage"] == "offload_text"
    assert status["failure"]["aimdo_counters"] == counters
    assert status["failure"]["aimdo_counters"]["pressure_direct_transfers"] == 0
    assert status["failure"]["aimdo_counters"]["pressure_direct_bytes"] == 0


@pytest.mark.parametrize(
    ("origin", "error_type", "cleanup_stage"),
    (
        ("primary", "LTX23KitchenWorkerPoisoned", None),
        ("cleanup", "RuntimeError", "unload_runtime"),
    ),
)
def test_last_worker_retains_authenticated_poison_status_and_serializes(
    origin: str,
    error_type: str,
    cleanup_stage: str | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    process = SimpleNamespace(pid=17, poll=lambda: 86)
    failure = {
        "error_type": error_type,
        "stage": "verify_output",
        "location": "ltx23_kitchen_worker.execute",
        "fingerprint": "b" * 64,
        "cleanup_stage": cleanup_stage,
        "aimdo_counters": None,
        "poison_reason": "device_quiescence_failed",
        "poison_origin": origin,
    }

    status = managed_module._last_worker(
        process,
        "failed",
        tree_empty=True,
        allocator_policy="expandable_segments:True",
        worker_failure=failure,
    )

    assert status["failure"]["poison_reason"] == "device_quiescence_failed"
    assert status["failure"]["poison_origin"] == origin
    assert json.loads(json.dumps(status))["failure"] == status["failure"]
    with caplog.at_level("ERROR"):
        managed_module._log_worker_failure(failure)
    assert "poison_reason=device_quiescence_failed" in caplog.text
    assert f"poison_origin={origin}" in caplog.text


def test_last_worker_and_logs_never_publish_unvalidated_poison_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_reason = "C:/private/prompt.txt"
    failure = {
        "error_type": "RuntimeError",
        "stage": "verify_output",
        "location": "ltx23_kitchen_worker.execute",
        "fingerprint": "b" * 64,
        "cleanup_stage": "unload_runtime",
        "aimdo_counters": None,
        "poison_reason": private_reason,
        "poison_origin": "cleanup",
    }
    status = managed_module._last_worker(
        SimpleNamespace(pid=18, poll=lambda: 1),
        "failed",
        tree_empty=True,
        allocator_policy="expandable_segments:True",
        worker_failure=failure,
    )

    assert "poison_reason" not in status["failure"]
    assert "poison_origin" not in status["failure"]
    with caplog.at_level("ERROR"):
        managed_module._log_worker_failure(failure)
    assert private_reason not in caplog.text
    assert "poison_reason=none" in caplog.text
    assert "poison_origin=none" in caplog.text


def test_worker_failure_publishes_and_logs_only_safe_diagnostics(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    result = tmp_path / "result.json"
    fingerprint = "a" * 64
    value = worker_module._signed_result(
        {
            "schema_version": 2,
            "ok": False,
            "request_binding": "expected",
            "error_type": "TypeError",
            "error": "prompt and C:/private/path must never be returned",
            "failure_stage": "materialize_text_encoder",
            "error_fingerprint": fingerprint,
            "failure_location": "ltx23_kitchen_text.load_ltx23_gemma_mixed_text_encoder",
            "cleanup_stage": "offload_text",
            "aimdo_counters": None,
        },
        _SECRET,
    )
    result.write_text(json.dumps(value), encoding="utf-8")

    failure = managed_module._worker_failure(result, 1, "expected", _SECRET)
    assert failure == {
        "message": (
            "LTX 2.3 Kitchen worker failed (TypeError during materialize_text_encoder at "
            "ltx23_kitchen_text.load_ltx23_gemma_mixed_text_encoder; diagnostic "
            "aaaaaaaaaaaa)"
        ),
        "error_type": "TypeError",
        "stage": "materialize_text_encoder",
        "location": "ltx23_kitchen_text.load_ltx23_gemma_mixed_text_encoder",
        "fingerprint": fingerprint,
        "cleanup_stage": "offload_text",
        "aimdo_counters": None,
    }
    assert "private" not in failure["message"]
    assert "prompt" not in failure["message"]
    with caplog.at_level("ERROR"):
        managed_module._log_worker_failure(failure)
    assert "TypeError" in caplog.text
    assert "materialize_text_encoder" in caplog.text
    assert "withheld_authenticated_ipc" in caplog.text
    assert "prompt" not in caplog.text
    assert "C:/private/path" not in caplog.text


def test_module_frame_failure_diagnostic_is_accepted_by_parent_validator() -> None:
    try:
        exec(  # noqa: S102 - synthesize a real traceback ``<module>`` frame.
            compile(
                "raise RuntimeError('private import failure')",
                "ltx23_import_fixture.py",
                "exec",
            ),
            {},
        )
    except RuntimeError as exc:
        diagnostic = worker_module._failure_diagnostic(
            exc,
            worker_module._FailureContext(stage="import_runtime"),
        )
    else:  # pragma: no cover - compile fixture must raise
        raise AssertionError("module failure fixture did not raise")

    assert diagnostic["failure_location"] == "ltx23_import_fixture.module"
    assert managed_module._valid_failure_diagnostic(diagnostic)


def test_preexisting_output_is_never_deleted_on_pre_spawn_validation_failure(tmp_path: Path):
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"existing")
    runtime = ManagedLTX23KitchenRuntime(_Request())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fresh MP4"):
        runtime.generate(
            prompt="scene",
            output_path=output,
            width=768,
            height=512,
            duration_seconds=1,
            seed=7,
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )
    assert output.read_bytes() == b"existing"
