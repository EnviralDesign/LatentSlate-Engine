"""Engine-native LTX 2.3 Kitchen runtime.

Diffusers owns the LTX pipeline and model forwards.  This module owns the
operation topology, explicit component residency, cancellation, provenance,
and final A/V mux. It converts no base model at runtime; stored quantized
linears dispatch directly through Comfy Kitchen as materialized by
:mod:`ltx23_av_stored_adapter`.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch import nn

from ..ltx23_kitchen_recipe import (
    LTX23_FPS,
    LTX23_GUIDANCE_SCALE,
    LTX23_GUIDE_STRENGTH,
    LTX23_MAIN_SIGMAS,
    LTX23_MODEL_LORA_STRENGTH,
    LTX23_REFINE_SIGMAS,
    LTX23_TEXT_LORA_STRENGTH,
    LTX23KitchenRuntimeRequest,
    revalidate_ltx23_kitchen_runtime_request,
)
from .cache import RuntimeCache, materialize_cached
from .ltx23_av_aimdo import LTX23AVAimdoState as _LTX23TransformerResidency
from .ltx23_av_stored_adapter import (
    LTX23StoredFP8Linear,
    build_ltx23_av_meta_shell,
    build_ltx23_connector_meta_shell,
    inspect_ltx23_av_artifact,
    inspect_ltx23_model_lora,
    install_ltx23_model_lora,
    ltx23_model_lora_dispatch_evidence,
    ltx23_module_physical_bytes,
    materialize_ltx23_av,
    materialize_ltx23_connectors,
    open_ltx23_av_payload,
    plan_ltx23_av_materialization,
    plan_ltx23_connector_materialization,
)
from .ltx23_kitchen_media import (
    build_ltx23_media_shell,
    materialize_ltx23_media_component,
    plan_ltx23_media_component,
)
from .ltx23_kitchen_text import (
    LTX23GemmaMixedTextStage,
    install_ltx23_gemma_text_lora,
    load_ltx23_gemma_mixed_text_encoder,
    plan_ltx23_gemma_mixed_text_encoder,
    plan_ltx23_gemma_text_lora,
)
from .ltx23_prompt import LTX23_T2V_SYSTEM_PROMPT
from .ltx23_video_vae_aimdo import (
    LTX23VideoVAEAimdoState as _LTX23VideoVAEResidency,
)
from .memory_telemetry import PhaseMemoryTelemetry

LTX23_AUDIO_SAMPLE_RATE = 48_000
LTX23_AUDIO_CHANNELS = 2
LTX23_AUDIO_SOURCE_SAMPLE_RATE = 16_000
LTX23_AUDIO_MEL_HOP_LENGTH = 160
LTX23_AUDIO_TEMPORAL_COMPRESSION_RATIO = 4
LTX23_AUDIO_DURATION_POLICY = "source_derived_exact_duration_v1"
LTX23_AAC_PACKET_SAMPLES = 1_024
LTX23_DEV_NEGATIVE_PROMPT = "pc game, console game, video game, cartoon, childish, ugly"
LTX23_FLF_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted "
    "proportions, unnatural skin tones, deformed facial features, asymmetrical face, "
    "missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts "
    "around text, unreadable text on shirt or hat, incorrect lettering on cap (\u201cPNTR\u201d), "
    "incorrect t-shirt slogan (\u201cJUST DO IT\u201d), missing microphone, misplaced microphone, "
    "inconsistent perspective, camera shake, incorrect depth of field, background too sharp, "
    "background clutter, distracting reflections, harsh shadows, inconsistent lighting "
    "direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, "
    "uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, "
    "smiling, laughing, exaggerated sadness, wrong gaze direction, eyes looking at camera, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, "
    "background noise, off-sync audio, missing sniff sounds, incorrect dialogue, added "
    "dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, "
    "unnatural transitions, inconsistent framing, tilted camera, missing door or shelves, "
    "missing shallow depth of field, flat lighting, inconsistent tone, cinematic "
    "oversaturation, stylized filters, or AI artifacts."
)
LTX23_PROMPT_ENHANCEMENT_SEED = 0
LTX23_PROMPT_MAX_NEW_TOKENS = 2_048
LTX23_PROMPT_STOP_TOKEN_ID = 106
LTX23_REFINE_SEED = 42
LTX23_GEMMA_PROMPT_EMBED_WIDTH = 3_840 * 49
LTX23_I2V_GUIDE_LONGER_EDGE = 1_536
LTX23_I2V_GUIDE_CRF = 18
LTX23_I2V_GUIDE_PRESET = "veryfast"
LTX23_I2V_GUIDE_PIXEL_FORMAT = "yuv420p"
LTX23_PROMPT_GENERATION_SETTINGS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 64,
    "top_p": 0.95,
    "min_p": 0.05,
    "repetition_penalty": 1.05,
}


class _ComfyLTX23LogitsProcessor:
    """Select one token with the pinned Comfy Gemma sampling semantics."""

    def __init__(
        self,
        *,
        prompt_length: int,
        device: torch.device,
        execution_dtype: torch.dtype,
        seed: int,
    ) -> None:
        self.prompt_length = prompt_length
        self.execution_dtype = execution_dtype
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[0] != 1 or scores.shape[0] != 1:
            raise RuntimeError("LTX prompt enhancement requires a single sequence")
        logits = scores.to(dtype=self.execution_dtype, copy=True)
        history = input_ids[:, self.prompt_length :]
        penalty = float(LTX23_PROMPT_GENERATION_SETTINGS["repetition_penalty"])
        if history.numel() and penalty != 1.0:
            token_ids = torch.unique(history)
            token_logits = logits[:, token_ids]
            token_logits = torch.where(
                token_logits < 0,
                token_logits * penalty,
                token_logits / penalty,
            )
            logits[:, token_ids] = token_logits

        temperature = float(LTX23_PROMPT_GENERATION_SETTINGS["temperature"])
        if temperature != 1.0:
            logits = logits / temperature

        top_k = min(int(LTX23_PROMPT_GENERATION_SETTINGS["top_k"]), logits.shape[-1])
        top_logits, top_indices = torch.topk(logits, top_k)
        min_p = float(LTX23_PROMPT_GENERATION_SETTINGS["min_p"])
        if min_p > 0.0:
            probabilities = torch.nn.functional.softmax(top_logits, dim=-1)
            threshold = min_p * probabilities.max(dim=-1, keepdim=True).values
            top_logits[probabilities < threshold] = torch.finfo(top_logits.dtype).min

        top_p = float(LTX23_PROMPT_GENERATION_SETTINGS["top_p"])
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(top_logits, descending=True)
            cumulative = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cumulative > top_p
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(top_logits, dtype=torch.bool)
            remove.scatter_(1, sorted_indices, sorted_remove)
            top_logits[remove] = torch.finfo(top_logits.dtype).min

        probabilities = torch.nn.functional.softmax(top_logits, dim=-1)
        sampled = torch.multinomial(probabilities, num_samples=1, generator=self.generator)
        token = top_indices.gather(1, sampled)

        # Transformers owns cache preparation and stopping, but its sampler
        # differs from Comfy. Publish the already-selected token as a forced
        # distribution so its final multinomial cannot alter this choice.
        forced = torch.full_like(scores, torch.finfo(scores.dtype).min)
        forced.scatter_(1, token, 0.0)
        return forced


# Diffusers retains all 49 Gemma hidden states for the 48-layer, 3,840-wide
# text model. At the pinned 1,024-token bf16 contract, each node output
# (embeddings plus its int64 mask) occupies 385,359,872 bytes. Comfy reuses the
# positive and negative CLIPTextEncode node outputs warm, so the smallest round
# bound that safely owns both plus provenance is 1 GiB.
LTX23_PROMPT_CACHE_MAX_BYTES = 1024 * 1024**2
LTX23_PROMPT_CACHE_MAX_ENTRIES = 8
LTX23_VAE_TILE_SAMPLE_MIN_HEIGHT = 768
LTX23_VAE_TILE_SAMPLE_MIN_WIDTH = 768
LTX23_VAE_TILE_SAMPLE_MIN_NUM_FRAMES = 4_096
LTX23_VAE_TILE_SAMPLE_STRIDE_HEIGHT = 704
LTX23_VAE_TILE_SAMPLE_STRIDE_WIDTH = 704
LTX23_VAE_TILE_SAMPLE_STRIDE_NUM_FRAMES = 4_088

LTX23KitchenProgress = Callable[[float, str | None], None]
LTX23KitchenCancellation = Callable[[], None]
_PROCESS_OWNERSHIP = threading.Lock()
_LTX23_REQUIRE_AIMDO_ENV = "LATENTSLATE_LTX23_REQUIRE_AIMDO"
_LTX23_TWO_STAGE_MEMORY_PHASES = (
    "after_text_offload",
    "after_stage1",
    "after_latent_upscaling",
    "after_stage2",
    "after_decode",
    "after_transient_clearing",
    "after_prompt_cache_publication",
)
_LTX23_SINGLE_STAGE_MEMORY_PHASES = (
    "after_text_offload",
    "after_main_denoise",
    "after_decode",
    "after_transient_clearing",
    "after_prompt_cache_publication",
)


class LTX23KitchenWorkerPoisoned(RuntimeError):
    """An unquiesced GPU child must terminate without Python finalization."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"LTX 2.3 Kitchen GPU child poisoned: {reason}")


_LTX23_CANONICAL_POISON_REASONS = frozenset(
    {
        "device_quiescence_failed",
        "failed_fill_quiescence_failed",
        "host_source_pool_structural_failure",
        "host_source_pool_setup_cleanup_failed",
        "host_registration_cleanup_failed",
        "ltx23_av_dynamic_initialization_cleanup_failed",
        "retirement_release_failed",
        "retirement_cleanup_failed",
        "retirement_query_failed",
        "retirement_quiescence_failed",
        "stage_prepare_failed",
    }
)


def _canonical_dynamic_poison_reason(exc: BaseException) -> str | None:
    """Recover only an authenticated child-terminal token from a cause chain."""

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        reason = getattr(current, "reason", None)
        if (
            type(current).__name__
            in {
                "LTX23KitchenWorkerPoisoned",
                "LTX23AVAimdoPoisoned",
                "DynamicResidencyPoisoned",
            }
            and reason in _LTX23_CANONICAL_POISON_REASONS
        ):
            return reason
        current = current.__cause__ or current.__context__
    return None


def _ltx23_text_dynamic_policy() -> str:
    """Use a child-only smoke gate without adding a recipe/user parameter."""

    value = os.environ.get(_LTX23_REQUIRE_AIMDO_ENV, "")
    if value not in {"", "0", "1"}:
        raise RuntimeError(f"{_LTX23_REQUIRE_AIMDO_ENV} must be unset, 0, or 1")
    return "required" if value == "1" else "auto"


def _ltx23_memory_telemetry_phases(operation: str) -> tuple[str, ...]:
    if operation in {"ltx23_dev_t2v", "ltx23_dev_i2v"}:
        return _LTX23_TWO_STAGE_MEMORY_PHASES
    if operation == "ltx23_distilled_flf":
        return _LTX23_SINGLE_STAGE_MEMORY_PHASES
    raise ValueError(f"unsupported LTX 2.3 memory telemetry operation: {operation}")


_AIMDO_FAILURE_COUNTER_FIELDS = (
    "physical_bytes",
    "staged_bytes",
    "virtual_bytes",
    "allocation_count",
    "live_allocations",
    "live_bytes",
    "loaded_bytes",
    "faults",
    "signature_hits",
    "signature_misses",
    "fault_none_temporaries",
    "pinned_copy_bytes",
    "pageable_copy_bytes",
    "transfer_events",
    "transfer_waits",
    "prioritize_calls",
    "unpin_calls",
    "free_calls",
    "dirty_epoch",
    "lora_invalidations",
    "base_restores",
    "copy_stream_count",
    "host_buffer_capacity_bytes",
    "host_buffer_allocations",
    "host_buffer_unregistrations",
    "host_buffer_frees",
    "gathered_misses",
    "per_physical_misses",
    "packed_source_bytes",
    "gathered_h2d_bytes",
    "pressure_direct_transfers",
    "pressure_direct_bytes",
    "host_buffer_reuse_barriers",
    "host_source_pool_generation",
    "host_source_pool_lane_count",
    "host_source_pool_capacity_bytes",
    "host_source_pool_retained_slices",
    "host_source_pool_retained_bytes",
    "host_source_pool_temporary_slices",
    "host_source_pool_temporary_bytes",
    "host_source_pool_hits",
    "host_source_pool_misses",
    "host_source_pool_stale_rejections",
    "host_source_pool_warm_ram_pressure_bypasses",
    "host_source_pool_warm_zero_delta_extend_refusals",
    "host_source_pool_warm_registration_refusals",
    "host_source_pool_temporary_ram_pressure_bypasses",
    "host_source_pool_temporary_zero_delta_extend_refusals",
    "host_source_pool_temporary_registration_refusals",
    "base_file_read_calls",
    "base_file_read_bytes",
    "base_file_handle_opened",
    "base_file_handle_closed",
)
_LTX23_REFILL_FAILURE_REASONS = frozenset(
    {
        "unbound_root_exceeds_target",
        "resident_trim_failed",
        "binding_acquire_failed",
    }
)

_AIMDO_HOST_BUFFER_FALLBACK_PREFIXES = (
    "host_buffer_capability_unavailable:",
    "host_buffer_setup_failed:",
)

_AIMDO_BASE_FILE_FALLBACK_PREFIXES = (
    "aimdo_backend_unavailable:",
    "aimdo_policy_or_device_unavailable:",
    *_AIMDO_HOST_BUFFER_FALLBACK_PREFIXES,
)


def _valid_aimdo_host_buffer_fallback_reason(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and 0 < len(value) <= 512
        and value.startswith(_AIMDO_HOST_BUFFER_FALLBACK_PREFIXES)
    )


def _valid_aimdo_base_file_fallback_reason(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and 0 < len(value) <= 512
        and value.startswith(_AIMDO_BASE_FILE_FALLBACK_PREFIXES)
    )


def _bounded_aimdo_failure_counters(text_residency: Mapping[str, Any]) -> dict[str, object] | None:
    dynamic = text_residency.get("dynamic_vram")
    if not isinstance(dynamic, Mapping) or dynamic.get("backend") != "comfy-aimdo":
        return None
    counters: dict[str, object] = {
        "backend": "comfy-aimdo",
        "version": dynamic.get("version"),
        "mode": dynamic.get("mode"),
        "policy": dynamic.get("policy"),
        "copy_strategy": dynamic.get("copy_strategy"),
        "copy_fallback_reason": dynamic.get("copy_fallback_reason"),
        "gathered_host_buffer_requested": dynamic.get("gathered_host_buffer_requested"),
        "host_buffer_live": dynamic.get("host_buffer_live"),
        "host_tensor_view_live": dynamic.get("host_tensor_view_live"),
        "host_buffer_transfer_pending": dynamic.get("host_buffer_transfer_pending"),
        "host_source_pool_poisoned": dynamic.get("host_source_pool_poisoned"),
        "host_source_pool_poison_reason": dynamic.get("host_source_pool_poison_reason"),
        "host_source_registration": dynamic.get("host_source_registration"),
        "base_file_backed": dynamic.get("base_file_backed"),
        "base_file_source_live": dynamic.get("base_file_source_live"),
        "base_file_handle_live": dynamic.get("base_file_handle_live"),
        "base_file_fallback_reason": dynamic.get("base_file_fallback_reason"),
        "refill_failure_reason": dynamic.get("refill_failure_reason"),
        "refill_target_bytes": dynamic.get("refill_target_bytes"),
        "refill_root_already_bound": dynamic.get("refill_root_already_bound"),
        "refill_resident_bytes": dynamic.get("refill_resident_bytes"),
        **{field: dynamic.get(field) for field in _AIMDO_FAILURE_COUNTER_FIELDS},
        "poisoned": dynamic.get("poisoned"),
        "close_failed": dynamic.get("close_failed"),
        "poison_reason": dynamic.get("poison_reason"),
    }
    if not _valid_bounded_aimdo_failure_counters(counters):
        raise RuntimeError("LTX AIMDO failure counters are not bounded canonical proof")
    return counters


def _valid_bounded_aimdo_failure_counters(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected = {
        "backend",
        "version",
        "mode",
        "policy",
        "copy_strategy",
        "copy_fallback_reason",
        "gathered_host_buffer_requested",
        "host_buffer_live",
        "host_tensor_view_live",
        "host_buffer_transfer_pending",
        "host_source_pool_poisoned",
        "host_source_pool_poison_reason",
        "host_source_registration",
        "base_file_backed",
        "base_file_source_live",
        "base_file_handle_live",
        "base_file_fallback_reason",
        "refill_failure_reason",
        "refill_target_bytes",
        "refill_root_already_bound",
        "refill_resident_bytes",
        *_AIMDO_FAILURE_COUNTER_FIELDS,
        "poisoned",
        "close_failed",
        "poison_reason",
    }
    return bool(
        set(value) == expected
        and value.get("backend") == "comfy-aimdo"
        and value.get("version") == "0.4.15"
        and value.get("mode") == "dynamic_vbar"
        and value.get("policy") in {"auto", "required"}
        and value.get("copy_strategy") in {"gathered_host_buffer", "per_physical"}
        and (
            value.get("copy_fallback_reason") is None
            or _valid_aimdo_host_buffer_fallback_reason(value.get("copy_fallback_reason"))
        )
        and isinstance(value.get("gathered_host_buffer_requested"), bool)
        and isinstance(value.get("host_buffer_live"), bool)
        and isinstance(value.get("host_tensor_view_live"), bool)
        and isinstance(value.get("host_buffer_transfer_pending"), bool)
        and isinstance(value.get("host_source_pool_poisoned"), bool)
        and (
            value.get("host_source_pool_poison_reason") is None
            or isinstance(value.get("host_source_pool_poison_reason"), str)
            and 0 < len(value["host_source_pool_poison_reason"]) <= 80
            and value["host_source_pool_poison_reason"].replace("_", "").isalnum()
        )
        and _valid_aimdo_host_source_registration(
            value.get("host_source_registration"),
            source_misses=value["host_source_pool_misses"],
            capacity_bytes=value["host_source_pool_capacity_bytes"],
            allow_unproven=(
                value.get("poison_reason")
                in {
                    "host_source_pool_structural_failure",
                    "host_source_pool_setup_cleanup_failed",
                }
                and value.get("host_source_pool_poisoned") is True
            ),
            pool_poison_reason=value.get("host_source_pool_poison_reason"),
        )
        and isinstance(value.get("base_file_backed"), bool)
        and isinstance(value.get("base_file_source_live"), bool)
        and isinstance(value.get("base_file_handle_live"), bool)
        and (
            value.get("base_file_fallback_reason") is None
            or _valid_aimdo_base_file_fallback_reason(value.get("base_file_fallback_reason"))
        )
        and (
            value.get("refill_failure_reason") is None
            and value.get("refill_target_bytes") is None
            and value.get("refill_root_already_bound") is None
            and value.get("refill_resident_bytes") is None
            or value.get("refill_failure_reason") in _LTX23_REFILL_FAILURE_REASONS
            and isinstance(value.get("refill_target_bytes"), int)
            and not isinstance(value.get("refill_target_bytes"), bool)
            and value["refill_target_bytes"] >= 0
            and isinstance(value.get("refill_root_already_bound"), bool)
            and isinstance(value.get("refill_resident_bytes"), int)
            and not isinstance(value.get("refill_resident_bytes"), bool)
            and value["refill_resident_bytes"] >= 0
            and (
                value["refill_failure_reason"] != "unbound_root_exceeds_target"
                or value["refill_root_already_bound"] is False
            )
        )
        and (not value["host_tensor_view_live"] or value["host_buffer_live"])
        and (
            not value["host_buffer_transfer_pending"]
            or value["host_buffer_live"]
            and value["host_tensor_view_live"]
        )
        and value["host_buffer_allocations"] <= 4
        and value["host_buffer_unregistrations"] <= value["host_buffer_allocations"]
        and value["host_buffer_frees"] <= value["host_buffer_unregistrations"]
        and value["gathered_misses"] <= value["signature_misses"]
        and value["per_physical_misses"] <= value["signature_misses"]
        and value["pressure_direct_transfers"] <= value["gathered_misses"]
        and value["pressure_direct_bytes"] <= value["gathered_h2d_bytes"]
        and (value["pressure_direct_transfers"] == 0) == (value["pressure_direct_bytes"] == 0)
        and value["pressure_direct_transfers"]
        <= value["host_source_pool_warm_ram_pressure_bypasses"]
        + value["host_source_pool_warm_zero_delta_extend_refusals"]
        + value["host_source_pool_warm_registration_refusals"]
        + value["host_source_pool_temporary_ram_pressure_bypasses"]
        + value["host_source_pool_temporary_zero_delta_extend_refusals"]
        + value["host_source_pool_temporary_registration_refusals"]
        <= value["pressure_direct_transfers"] + 1
        and value["host_buffer_reuse_barriers"] <= value["gathered_misses"]
        and value["host_source_pool_lane_count"] == value["host_buffer_allocations"]
        and value["host_source_pool_capacity_bytes"] >= value["host_buffer_capacity_bytes"]
        and value["host_source_pool_capacity_bytes"] <= 2 * value["virtual_bytes"]
        and value["gathered_misses"]
        <= value["host_source_pool_hits"]
        + value["host_source_pool_misses"]
        + value["pressure_direct_transfers"]
        <= value["signature_misses"]
        and value["host_source_pool_retained_bytes"] + value["host_source_pool_temporary_bytes"]
        <= value["host_source_pool_capacity_bytes"]
        and value["base_file_handle_opened"] <= 1
        and value["base_file_handle_closed"] <= value["base_file_handle_opened"]
        and value["base_file_handle_live"]
        == (value["base_file_handle_opened"] > value["base_file_handle_closed"])
        and (not value["base_file_source_live"] or value["base_file_handle_live"])
        and (not value["base_file_source_live"] or value["base_file_backed"])
        and (value["base_file_read_calls"] == 0) == (value["base_file_read_bytes"] == 0)
        and (value["base_file_read_calls"] == 0 or value["base_file_backed"])
        and (
            value["base_file_backed"]
            and value["base_file_handle_opened"] == 1
            and value["base_file_fallback_reason"] is None
            or not value["base_file_backed"]
            and value["base_file_read_calls"] == 0
        )
        and (
            not value["host_buffer_transfer_pending"]
            or value["copy_strategy"] == "gathered_host_buffer"
        )
        and (
            value["copy_strategy"] == "gathered_host_buffer"
            and value["gathered_host_buffer_requested"] is True
            and value["copy_fallback_reason"] is None
            and 1 <= value["host_buffer_allocations"] <= 4
            and value["host_source_pool_generation"] >= 1
            and value["host_source_pool_stale_rejections"] == 0
            and value["host_buffer_reuse_barriers"] == 0
            and value["host_buffer_capacity_bytes"] > 0
            and value["per_physical_misses"] == 0
            and value["pinned_copy_bytes"] + value["pressure_direct_bytes"]
            == value["gathered_h2d_bytes"]
            and value["pageable_copy_bytes"] <= value["pressure_direct_bytes"]
            and value["packed_source_bytes"] <= value["gathered_h2d_bytes"]
            and value["gathered_h2d_bytes"]
            <= value["host_buffer_capacity_bytes"] * value["gathered_misses"]
            or value["copy_strategy"] == "per_physical"
            and (
                value["copy_fallback_reason"] is None
                or _valid_aimdo_host_buffer_fallback_reason(value["copy_fallback_reason"])
            )
            and value["gathered_misses"] == 0
            and value["packed_source_bytes"] == 0
            and value["gathered_h2d_bytes"] == 0
            and value["pressure_direct_transfers"] == 0
            and value["pressure_direct_bytes"] == 0
            and value["host_source_pool_warm_ram_pressure_bypasses"] == 0
            and value["host_source_pool_warm_zero_delta_extend_refusals"] == 0
            and value["host_source_pool_warm_registration_refusals"] == 0
            and value["host_source_pool_temporary_ram_pressure_bypasses"] == 0
            and value["host_source_pool_temporary_zero_delta_extend_refusals"] == 0
            and value["host_source_pool_temporary_registration_refusals"] == 0
            and value["host_buffer_reuse_barriers"] == 0
            and _valid_clean_aimdo_host_source_pool_fallback(value)
        )
        and all(
            (
                field == "loaded_bytes"
                and value.get(field) is None
                or isinstance(value.get(field), int)
                and not isinstance(value.get(field), bool)
                and value[field] >= 0
            )
            for field in _AIMDO_FAILURE_COUNTER_FIELDS
        )
        and isinstance(value.get("poisoned"), bool)
        and isinstance(value.get("close_failed"), bool)
        and (
            value.get("poison_reason") is None
            or isinstance(value.get("poison_reason"), str)
            and value["poison_reason"] in _LTX23_CANONICAL_POISON_REASONS
        )
    )


def _valid_clean_aimdo_host_source_pool_fallback(value: Mapping[str, Any]) -> bool:
    allocations = value["host_buffer_allocations"]
    if not (
        value.get("host_source_pool_poisoned") is False
        and value.get("host_source_pool_poison_reason") is None
        and value["host_source_pool_retained_slices"] == 0
        and value["host_source_pool_retained_bytes"] == 0
        and value["host_source_pool_temporary_slices"] == 0
        and value["host_source_pool_temporary_bytes"] == 0
        and value["host_source_pool_hits"] == 0
        and value["host_source_pool_misses"] == 0
        and value["host_source_pool_stale_rejections"] == 0
    ):
        return False
    if allocations == 0:
        return bool(
            value["host_buffer_unregistrations"] == 0
            and value["host_buffer_frees"] == 0
            and value["host_source_pool_generation"] == 0
            and value["host_source_pool_lane_count"] == 0
            and value["host_source_pool_capacity_bytes"] == 0
        )
    return bool(
        1 <= allocations <= 4
        and value["host_buffer_unregistrations"] == allocations
        and value["host_buffer_frees"] == allocations
        and value["host_source_pool_generation"] >= 2
        and value["host_source_pool_lane_count"] == allocations
        and value["host_source_pool_capacity_bytes"] > 0
    )


def _valid_aimdo_host_source_registration(
    value: object,
    *,
    source_misses: int,
    capacity_bytes: int,
    allow_unproven: bool,
    pool_poison_reason: object,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    integer_fields = {
        "budget_bytes",
        "attempts",
        "attempt_bytes",
        "successes",
        "failures",
        "failure_bytes",
        "registered_bytes",
        "unregistered_bytes",
        "live_bytes",
        "peak_bytes",
    }
    if (
        set(value) != integer_fields | {"policy", "state_proven"}
        or value.get("policy") != "aimdo_hostbuffer_registered_append"
        or not isinstance(value.get("state_proven"), bool)
        or any(
            not isinstance(value.get(field), int)
            or isinstance(value.get(field), bool)
            or value[field] < 0
            for field in integer_fields
        )
    ):
        return False
    proven = value["state_proven"] is True
    return bool(
        (proven or allow_unproven)
        and value["attempts"] == value["successes"] + value["failures"]
        and value["attempt_bytes"] == value["registered_bytes"] + value["failure_bytes"]
        and source_misses <= value["successes"] <= value["attempts"]
        and (allow_unproven or value["successes"] == source_misses)
        and value["successes"] - source_misses <= 1
        and (
            value["successes"] == source_misses
            or pool_poison_reason
            in {"host_buffer_view_validation_failed", "host_buffer_rollback_failed"}
        )
        and value["unregistered_bytes"] <= value["registered_bytes"]
        and value["live_bytes"] <= value["peak_bytes"] <= capacity_bytes
        and value["peak_bytes"] <= value["budget_bytes"]
        and (
            not proven
            or value["live_bytes"] == value["registered_bytes"] - value["unregistered_bytes"]
        )
    )


@dataclass(frozen=True, slots=True)
class LTX23KitchenGeneration:
    """One exact, already-resolved LTX generation invocation."""

    prompt: str
    output_path: Path
    width: int
    height: int
    num_frames: int
    seed: int
    start_image_path: Path | None = None
    end_image_path: Path | None = None
    start_image_identity: Mapping[str, object] | None = None
    end_image_identity: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class LTX23KitchenResult:
    """Published output plus identity-bound native execution evidence."""

    output_path: Path
    metadata: Mapping[str, Any]


class _LTX23PhaseTimings:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.phases: dict[str, float] = {}
        self.cumulative: dict[str, float] = {}

    def record(self, name: str, started_at: float) -> float:
        now = time.perf_counter()
        duration = max(0.0, now - started_at)
        self.phases[name] = duration
        self.cumulative[name] = max(0.0, now - self.started_at)
        return duration

    def skip(self, name: str) -> None:
        self.phases[name] = 0.0
        self.cumulative[name] = max(0.0, time.perf_counter() - self.started_at)

    def metadata(self) -> dict[str, Any]:
        return {
            "clock": "time.perf_counter",
            "unit": "seconds",
            "phases": dict(self.phases),
            "cumulative": dict(self.cumulative),
            "total_seconds": max(0.0, time.perf_counter() - self.started_at),
        }


@dataclass(frozen=True, slots=True)
class LTX23KitchenOperationSpec:
    operation: str
    stages: tuple[str, ...]
    main_sigmas: tuple[float, ...]
    refine_sigmas: tuple[float, ...] | None
    prompt_enhancement: bool
    model_lora_strength: float | None
    text_lora_strength: float | None
    guide_strengths: tuple[float, ...]
    fps: int = LTX23_FPS
    audio_sample_rate: int = LTX23_AUDIO_SAMPLE_RATE
    audio_channels: int = LTX23_AUDIO_CHANNELS


@dataclass(frozen=True, slots=True)
class LTX23DecodedAudio:
    """Waveform plus closed evidence for the exact source decoder grid."""

    waveform: np.ndarray
    video_frames: int
    audio_latent_frames: int
    expected_audio_latent_frames: int
    audio_latent_channels: int
    audio_latent_mel_bins: int
    decoded_mel_frames: int
    expected_mel_frames: int
    decoded_mel_channels: int
    decoded_mel_bins: int
    decoded_samples: int
    expected_decoded_samples: int
    source_sample_rate: int
    output_sample_rate: int
    mel_hop_length: int
    temporal_compression_ratio: int
    causality_axis: str
    is_causal: bool


def ltx23_kitchen_operation_spec(operation: str) -> LTX23KitchenOperationSpec:
    """Return the fixed operation topology without loading a tensor payload."""

    if operation == "ltx23_dev_t2v":
        return LTX23KitchenOperationSpec(
            operation,
            ("prompt_enhance", "text", "main", "x2", "refine", "decode", "mux"),
            LTX23_MAIN_SIGMAS,
            LTX23_REFINE_SIGMAS,
            True,
            LTX23_MODEL_LORA_STRENGTH,
            LTX23_TEXT_LORA_STRENGTH,
            (),
        )
    if operation == "ltx23_dev_i2v":
        return LTX23KitchenOperationSpec(
            operation,
            (
                "prompt_enhance",
                "text",
                "guide_preprocess",
                "guide_half",
                "main",
                "x2",
                "guide_full",
                "refine",
                "decode",
                "mux",
            ),
            LTX23_MAIN_SIGMAS,
            LTX23_REFINE_SIGMAS,
            True,
            LTX23_MODEL_LORA_STRENGTH,
            LTX23_TEXT_LORA_STRENGTH,
            (LTX23_GUIDE_STRENGTH, 1.0),
        )
    if operation == "ltx23_distilled_flf":
        return LTX23KitchenOperationSpec(
            operation,
            ("text", "guide_first", "guide_last", "main", "decode", "mux"),
            LTX23_MAIN_SIGMAS,
            None,
            False,
            None,
            None,
            (LTX23_GUIDE_STRENGTH, LTX23_GUIDE_STRENGTH),
        )
    raise ValueError(f"unsupported LTX 2.3 Kitchen operation {operation!r}")


def validate_ltx23_kitchen_generation(operation: str, generation: LTX23KitchenGeneration) -> None:
    """Fail before loading models if invocation geometry or guides are invalid."""

    ltx23_kitchen_operation_spec(operation)
    if not isinstance(generation.prompt, str) or not generation.prompt.strip():
        raise ValueError("LTX 2.3 prompt must be nonempty")
    if generation.width <= 0 or generation.height <= 0:
        raise ValueError("LTX 2.3 dimensions must be positive")
    divisor = 64 if operation.startswith("ltx23_dev_") else 32
    if generation.width % divisor or generation.height % divisor:
        raise ValueError(f"LTX 2.3 {operation} dimensions must be divisible by {divisor}")
    if generation.num_frames <= 0 or generation.num_frames % 8 != 1:
        raise ValueError("LTX 2.3 frame count must be positive and of the form 8k+1")
    if isinstance(generation.seed, bool) or not isinstance(generation.seed, int):
        raise TypeError("LTX 2.3 seed must be an integer")
    expected = {
        "ltx23_dev_t2v": (False, False),
        "ltx23_dev_i2v": (True, False),
        "ltx23_distilled_flf": (True, True),
    }[operation]
    actual = (generation.start_image_path is not None, generation.end_image_path is not None)
    if actual != expected:
        raise ValueError(f"LTX 2.3 {operation} endpoint-image contract differs")
    identities = (
        generation.start_image_identity is not None,
        generation.end_image_identity is not None,
    )
    if identities != expected:
        raise ValueError(f"LTX 2.3 {operation} endpoint-image identity contract differs")
    for label, path, identity in (
        ("start", generation.start_image_path, generation.start_image_identity),
        ("end", generation.end_image_path, generation.end_image_identity),
    ):
        if path is not None and not Path(path).resolve(strict=True).is_file():
            raise ValueError(f"LTX 2.3 {label} guide is not a file")
        if identity is not None and set(identity) != {"size_bytes", "mtime_ns", "sha256"}:
            raise ValueError(f"LTX 2.3 {label} guide identity is not canonical")
    output = Path(generation.output_path).resolve(strict=False)
    if output.suffix.lower() != ".mp4":
        raise ValueError("LTX 2.3 output must be an MP4 path")


class LTX23KitchenRuntime:
    """Persistent request-bound runtime with explicit 16 GB-class staging."""

    def __init__(
        self,
        request: LTX23KitchenRuntimeRequest,
        *,
        device: torch.device | str = "cuda",
        cache_policy: str = "none",
    ) -> None:
        self.request = request
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("LTX 2.3 Kitchen runtime requires direct CUDA execution")
        if cache_policy not in {"none", "prompt"}:
            raise ValueError("LTX 2.3 Kitchen cache policy must be 'none' or 'prompt'")
        self.cache_policy = cache_policy
        self._cache = RuntimeCache(
            f"ltx23-kitchen:{request.fingerprint}",
            enabled=cache_policy == "prompt",
            max_bytes=LTX23_PROMPT_CACHE_MAX_BYTES,
            max_entries=LTX23_PROMPT_CACHE_MAX_ENTRIES,
            prompt_fraction=1.0,
        )
        self._components: dict[str, Any] | None = None
        self._transformer_residency: _LTX23TransformerResidency | None = None
        self._video_vae_residency: _LTX23VideoVAEResidency | None = None
        self._active_text_stage: LTX23GemmaMixedTextStage | None = None
        self._last_failure_aimdo: dict[str, object] | None = None

    def generate(
        self,
        generation: LTX23KitchenGeneration,
        *,
        progress: LTX23KitchenProgress,
        check_cancelled: LTX23KitchenCancellation,
    ) -> LTX23KitchenResult:
        """Materialize once, then execute compatible jobs against warmed components."""

        if not _PROCESS_OWNERSHIP.acquire(blocking=False):
            raise RuntimeError("an LTX 2.3 Kitchen runtime is already active in this process")
        timings = _LTX23PhaseTimings()
        memory_telemetry = PhaseMemoryTelemetry(
            _ltx23_memory_telemetry_phases(self.request.operation), self.device
        )
        self._last_failure_aimdo = None
        try:
            check_cancelled()
            if not torch.cuda.is_available():
                raise RuntimeError("LTX 2.3 Kitchen runtime requires an available CUDA device")
            if not revalidate_ltx23_kitchen_runtime_request(self.request):
                raise RuntimeError("LTX 2.3 Kitchen request changed after resolution")
            validate_ltx23_kitchen_generation(self.request.operation, generation)
            output = Path(generation.output_path).resolve(strict=False)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise FileExistsError(f"LTX 2.3 output already exists: {output}")
            pipeline_warm = self._components is not None
            if self._components is None:
                progress(0.006, "Loading LTX 2.3 components")
                phase_started = time.perf_counter()
                self._components = self._materialize(check_cancelled, progress)
                self._transformer_residency = _LTX23TransformerResidency(
                    self._components["transformer"], self.device
                )
                self._video_vae_residency = _LTX23VideoVAEResidency(
                    self._components["video_vae"], self.device
                )
                duration = timings.record("materialization", phase_started)
                progress(
                    0.076,
                    f"Materialized LTX components (phase_s={duration:.6f}, "
                    f"cumulative_s={timings.cumulative['materialization']:.6f})",
                )
            else:
                timings.skip("materialization")
                progress(0.006, "Reusing warmed LTX components")
            result = self._execute(
                self._components,
                generation,
                progress=progress,
                check_cancelled=check_cancelled,
                timings=timings,
                memory_telemetry=memory_telemetry,
            )
            prompt_hit = bool(result.metadata.pop("_prompt_cache_hit", False))
            prompt_published = bool(result.metadata.pop("_prompt_cache_published", False))
            prompt_cache = getattr(self, "_cache", None)
            result.metadata["cache"] = {
                "pipeline_warm": pipeline_warm,
                "policy": getattr(self, "cache_policy", "none"),
                "prompt_hit": prompt_hit,
                "prompt_published": prompt_published,
                "media_hit": False,
                "prompt": (
                    prompt_cache.prompt.status()
                    if prompt_cache is not None
                    else {
                        "name": "prompt",
                        "enabled": False,
                        "entries": 0,
                        "bytes": 0,
                        "max_bytes": LTX23_PROMPT_CACHE_MAX_BYTES,
                        "max_entries": LTX23_PROMPT_CACHE_MAX_ENTRIES,
                        "hits": 0,
                        "misses": 0,
                        "evictions": 0,
                        "hit_rate": None,
                    }
                ),
            }
            progress(1.0, "LTX 2.3 output ready")
            return result
        except BaseException as primary:
            if isinstance(primary, LTX23KitchenWorkerPoisoned):
                raise
            stage = getattr(self, "_active_text_stage", None)
            if stage is not None:
                try:
                    self._last_failure_aimdo = _bounded_aimdo_failure_counters(stage.diagnostics())
                except (KeyError, RuntimeError, TypeError, ValueError):
                    self._last_failure_aimdo = None
            # Once AV residency exists, its safe snapshot owns failures in the
            # denoise/upscale/decode portion, including ordinary module OOMs.
            # A wrapper-level failed CUDA barrier returns no snapshot and keeps
            # the earlier text counters as the only safe evidence.
            residency = getattr(self, "_transformer_residency", None)
            failure_diagnostics = getattr(residency, "failure_diagnostics", None)
            if callable(failure_diagnostics):
                try:
                    av_counters = _bounded_aimdo_failure_counters(failure_diagnostics())
                except (KeyError, RuntimeError, TypeError, ValueError):
                    av_counters = None
                if av_counters is not None:
                    self._last_failure_aimdo = av_counters
            try:
                Path(generation.output_path).unlink(missing_ok=True)
            except BaseException as output_cleanup_error:  # noqa: BLE001 - retain primary error
                primary.add_note(f"LTX 2.3 output cleanup also failed: {output_cleanup_error}")
            poison_reason = self.terminal_poison_reason()
            if poison_reason is not None:
                # Retain the entire component/stage graph. The persistent-child
                # terminal path will use a hard OS exit; normal unload or stack
                # destruction could invoke ModelVBAR.__del__ and native free on
                # an address still referenced by unquiesced GPU work.
                raise LTX23KitchenWorkerPoisoned(poison_reason) from primary
            # A failed execution can leave staged modules in an unknown state.
            # Do not attempt reuse until this process has rebuilt its exact recipe.
            # Cleanup is best effort here.  In particular, an incomplete
            # third-party module shell must never replace the actual inference
            # failure with a secondary ``.to(cpu)`` error from its intentional
            # meta state.
            try:
                self.unload()
            except LTX23KitchenWorkerPoisoned as cleanup_poison:
                primary.add_note(f"LTX 2.3 cleanup established terminal poison: {cleanup_poison}")
                raise
            except BaseException as cleanup_error:  # noqa: BLE001 - retain primary execution error
                primary.add_note(f"LTX 2.3 cleanup also failed: {cleanup_error}")
            raise
        finally:
            _PROCESS_OWNERSHIP.release()

    def clear_cache(self) -> None:
        """Clear bounded CPU prompt conditioning without unloading components."""

        cache = getattr(self, "_cache", None)
        if cache is not None:
            cache.clear()

    def failure_aimdo_counters(self) -> dict[str, object] | None:
        """Return safe authenticated AIMDO counters retained across failure unload."""

        return None if self._last_failure_aimdo is None else dict(self._last_failure_aimdo)

    def cache_status(self) -> dict[str, Any]:
        """Return bounded cache state suitable for authenticated worker status."""

        return {
            "pipeline_warm": self._components is not None,
            "policy": self.cache_policy,
            "prompt_hit": False,
            "prompt_published": False,
            "media_hit": False,
            "prompt": self._cache.prompt.status(),
        }

    def unload(self) -> None:
        """Release warmed components and establish a CUDA cleanup barrier."""

        poison_reason = self.terminal_poison_reason()
        if poison_reason is not None:
            raise LTX23KitchenWorkerPoisoned(poison_reason)

        components = self._components
        text_stage = getattr(self, "_active_text_stage", None)
        video_vae_residency = getattr(self, "_video_vae_residency", None)
        residency = getattr(self, "_transformer_residency", None)
        residency_error: BaseException | None = None
        if text_stage is not None:
            try:
                text_stage.close()
            except BaseException as exc:  # noqa: BLE001 - release remaining components too
                residency_error = exc
            else:
                self._active_text_stage = None
        if residency_error is None and video_vae_residency is not None:
            try:
                video_vae_residency.close()
            except BaseException as exc:  # noqa: BLE001 - preserve exact native owner
                residency_error = exc
            else:
                self._video_vae_residency = None
        if residency_error is None and residency is not None:
            try:
                residency.close()
            except BaseException as exc:  # noqa: BLE001 - release remaining components too
                residency_error = exc
            else:
                self._transformer_residency = None
        if residency_error is None and components is not None:
            try:
                _release_components(components, self.device)
            except BaseException as exc:  # noqa: BLE001 - preserve residency failure first
                residency_error = exc
            else:
                self._components = None
        if residency_error is not None:
            canonical = _canonical_dynamic_poison_reason(residency_error)
            if canonical is not None:
                # Failed quiescence keeps the runtime's exact residency and
                # component graph reachable until the persistent child takes
                # its hard OS exit.  Clearing either owner here can run native
                # finalizers against active GPU addresses.
                raise LTX23KitchenWorkerPoisoned(canonical) from residency_error
            raise RuntimeError(
                f"LTX 2.3 runtime unload failed: {residency_error}"
            ) from residency_error
        cache = getattr(self, "_cache", None)
        if cache is not None:
            cache.clear()

    def terminal_poison_reason(self) -> str | None:
        """Return a terminal AIMDO reason without releasing retained objects."""

        stage = getattr(self, "_active_text_stage", None)
        text_reason = None if stage is None else stage.terminal_poison_reason()
        if text_reason is not None:
            return text_reason
        video_vae_residency = getattr(self, "_video_vae_residency", None)
        video_reason = (
            None
            if video_vae_residency is None
            else video_vae_residency.terminal_poison_reason()
        )
        if video_reason is not None:
            return video_reason
        residency = getattr(self, "_transformer_residency", None)
        if residency is None:
            return None
        residency_reason = getattr(residency, "terminal_poison_reason", None)
        if callable(residency_reason):
            reason = residency_reason()
            if reason is not None:
                return reason
        transformer = getattr(residency, "transformer", None)
        if transformer is None:
            return None
        poisoned = getattr(transformer, "_latentslate_ltx23_residency_poisoned", None)
        return None if poisoned is None else str(poisoned)[:512]

    def _materialize(
        self,
        check_cancelled: LTX23KitchenCancellation,
        progress: LTX23KitchenProgress,
    ) -> dict[str, Any]:
        plans = self.request.plans
        support = plans["pipeline_support"].root
        checkpoint_path = plans["checkpoint"].identity.path
        variant: Literal["dev", "distilled"] = (
            "distilled" if self.request.operation == "ltx23_distilled_flf" else "dev"
        )

        # The meta shell is a necessary target-topology proof before any
        # SafeTensors payload can be assigned.  Report each real CPU phase so
        # a cold start never appears idle while preserving that exact order.
        progress(0.01, "Inspecting LTX transformer artifact")
        av_contract = inspect_ltx23_av_artifact(checkpoint_path, expected_variant=variant)
        progress(0.015, "Building LTX transformer shell")
        transformer = build_ltx23_av_meta_shell(av_contract)
        progress(0.02, "Planning LTX transformer materialization")
        transformer_plan = plan_ltx23_av_materialization(
            transformer, checkpoint_path, expected_variant=variant
        )
        # Keep one mapping alive across every embedded checkpoint closure.
        # Reopening this 29 GB file after CUDA initialization intermittently
        # faults inside Torch storage on Windows, before a Kitchen kernel runs.
        media: dict[str, nn.Module] = {}
        with open_ltx23_av_payload(checkpoint_path) as payload_handle:
            progress(0.025, "Materializing LTX transformer")
            transformer = materialize_ltx23_av(
                transformer,
                transformer_plan,
                payload_handle=payload_handle,
            )
            transformer.eval()
            check_cancelled()
            progress(0.027, "Building LTX connector shell")
            connector = build_ltx23_connector_meta_shell(av_contract)
            progress(0.028, "Planning LTX connector payload mapping")
            connector_plan = plan_ltx23_connector_materialization(
                connector, checkpoint_path, expected_variant=variant
            )
            progress(0.03, "Materializing LTX connector payload")
            connector = materialize_ltx23_connectors(
                connector,
                connector_plan,
                payload_handle=payload_handle,
            )
            connector.eval()
            media_progress = {
                "video_vae": (0.032, 0.034, 0.036, "LTX video VAE"),
                "audio_vae": (0.038, 0.04, 0.042, "LTX audio VAE"),
                "vocoder": (0.044, 0.046, 0.048, "LTX vocoder"),
            }
            for component in ("video_vae", "audio_vae", "vocoder"):
                shell_phase, plan_phase, payload_phase, label = media_progress[component]
                progress(shell_phase, f"Building {label} shell")
                shell = build_ltx23_media_shell(component)  # type: ignore[arg-type]
                progress(plan_phase, f"Planning {label} payload mapping")
                plan = plan_ltx23_media_component(
                    plans["checkpoint"],
                    component,
                    shell,
                    payload_handle=payload_handle,
                )  # type: ignore[arg-type]
                progress(payload_phase, f"Materializing {label} payload")
                media[component] = materialize_ltx23_media_component(
                    shell, plan, payload_handle=payload_handle
                )
                media[component].eval()
                check_cancelled()
        if variant == "dev":
            progress(0.051, "Building LTX latent upsampler shell")
            shell = build_ltx23_media_shell("latent_upsampler")
            progress(0.053, "Planning LTX latent upsampler payload mapping")
            up_plan = plan_ltx23_media_component(
                plans["latent_upscaler"], "latent_upsampler", shell
            )
            progress(0.055, "Materializing LTX latent upsampler payload")
            media["latent_upsampler"] = materialize_ltx23_media_component(shell, up_plan)
            media["latent_upsampler"].eval()

        progress(0.06, "Materializing LTX text encoder")
        text_plan = plan_ltx23_gemma_mixed_text_encoder(plans["text_encoder"].identity.path)
        text = load_ltx23_gemma_mixed_text_encoder(text_plan, support / "text_encoder")
        text.eval()
        check_cancelled()

        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import Gemma3Processor

        processor = Gemma3Processor.from_pretrained(support / "processor", local_files_only=True)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            support / "scheduler", local_files_only=True
        )

        model_lora = None
        text_lora = None
        if variant == "dev":
            progress(0.07, "Installing LTX model LoRA")
            model_contract = inspect_ltx23_model_lora(
                av_contract, plans["model_lora"].identity.path
            )
            model_lora = install_ltx23_model_lora(
                transformer,
                model_contract,
                adapter_name="latentslate_ltx23_distilled",
                strength=LTX23_MODEL_LORA_STRENGTH,
            )
            progress(0.075, "Installing LTX text LoRA")
            text_lora_plan = plan_ltx23_gemma_text_lora(plans["text_lora"].identity.path)
            text_lora = install_ltx23_gemma_text_lora(
                text,
                text_lora_plan,
                adapter_name="latentslate_ltx23_abliterated",
                strength=LTX23_TEXT_LORA_STRENGTH,
            )
        components = {
            "support": support,
            "transformer": transformer,
            "connector": connector,
            "text": text,
            "processor": processor,
            "scheduler": scheduler,
            "model_lora": model_lora,
            "text_lora": text_lora,
            **media,
        }
        components["_stage_bytes"] = {
            name: ltx23_module_physical_bytes(components[name])
            for name in (
                "connector",
                "text",
                "video_vae",
                "audio_vae",
                "vocoder",
            )
        }
        if "latent_upsampler" in components:
            components["_stage_bytes"]["latent_upsampler"] = ltx23_module_physical_bytes(
                components["latent_upsampler"]
            )
        return components

    def _execute(
        self,
        c: dict[str, Any],
        g: LTX23KitchenGeneration,
        *,
        progress: LTX23KitchenProgress,
        check_cancelled: LTX23KitchenCancellation,
        timings: _LTX23PhaseTimings,
        memory_telemetry: PhaseMemoryTelemetry,
    ) -> LTX23KitchenResult:
        base, conditioned, upsample = _build_pipelines(c, self.device)
        c["_video_processor"] = base.video_processor
        residency = self._transformer_residency
        if residency is None:
            raise RuntimeError("LTX transformer residency was not initialized")
        video_vae_residency = self._video_vae_residency
        if video_vae_residency is None:
            raise RuntimeError("LTX video VAE residency was not initialized")
        video_vae_before = video_vae_residency.diagnostics()
        operation_spec = ltx23_kitchen_operation_spec(self.request.operation)
        source_prompt = g.prompt
        negative_prompt = (
            LTX23_FLF_NEGATIVE_PROMPT
            if self.request.operation == "ltx23_distilled_flf"
            else LTX23_DEV_NEGATIVE_PROMPT
        )
        prompt_cache_key = _prompt_conditioning_cache_key(self._cache, self.request, source_prompt)
        cached_prompt = self._cache.prompt.get(prompt_cache_key)
        prompt_cache_hit = cached_prompt is not None
        prompt_cache_published = False
        prompt_cache_candidate: dict[str, Any] | None = None
        if cached_prompt is not None:
            _validate_cached_negative_conditioning(cached_prompt, negative_prompt)
            progress(0.13, "Reusing cached LTX prompt conditioning")
            prompt = cached_prompt["enhanced_prompt"]
            prompt_embeds = materialize_cached(cached_prompt["prompt_embeds"], device=self.device)
            prompt_mask = materialize_cached(cached_prompt["prompt_mask"], device=self.device)
            negative_encoding = _cached_dispatch_proof(cached_prompt["negative_encoding"])
            text_patch_state = _cached_dispatch_proof(cached_prompt["text_patch_state"])
            prompt_enhancement_memory = (
                {
                    "policy": "not_required_prompt_cache_hit",
                    "source_proof": cached_prompt["prompt_enhancement_memory"],
                }
                if operation_spec.prompt_enhancement
                else None
            )
            text_native_proof = _cached_dispatch_proof(cached_prompt["native_text"])
            text_proof = (
                None
                if cached_prompt["text_lora"] is None
                else _cached_dispatch_proof(cached_prompt["text_lora"])
            )
            text_residency = {
                "mode": "not_required_prompt_cache_hit",
                "source_proof": cached_prompt["text_residency"],
            }
            for phase in (
                "text_onload",
                "enhancement",
                "positive_encode",
                "negative_encode",
                "text_offload",
            ):
                timings.skip(phase)
        else:
            text_stage = self._active_text_stage
            if text_stage is None:
                text_stage = LTX23GemmaMixedTextStage(
                    c["text"],
                    self.device,
                    dynamic_policy=_ltx23_text_dynamic_policy(),
                    progress=progress,
                )
                self._active_text_stage = text_stage
            text_lora = c["text_lora"]
            if self.request.operation.startswith("ltx23_dev_") and text_lora is None:
                raise RuntimeError("LTX Dev text LoRA patcher is missing")
            # The mixed Gemma artifact is roughly the whole GPU on a 16 GB card
            # when staged as one module. Reserve only its largest live root+layer
            # binding plus activation headroom while retaining CPU masters.
            phase_started = time.perf_counter()
            progress(0.078, "Preparing streamed LTX text encoder")
            text_stage.onload()
            duration = timings.record("text_onload", phase_started)
            progress(
                0.079,
                f"Prepared streamed LTX text encoder (phase_s={duration:.6f}, "
                f"cumulative_s={timings.cumulative['text_onload']:.6f})",
            )
            prompt_enhancement_memory: dict[str, Any] | None = None
            text_patch_state = {
                "policy": (
                    "prompt_enhancement_only"
                    if operation_spec.prompt_enhancement
                    else "installed_inactive_base_encode"
                    if text_lora is not None
                    else "base_only"
                ),
                "lora_strength_enhancement": (1.0 if operation_spec.prompt_enhancement else None),
                "lora_strength_positive": 0.0,
                "lora_strength_negative": 0.0,
                "lora_entry_transitions": 0,
                "lora_to_base_transitions": 0,
                "restored_base_on_exit": False,
            }
            text_patch_state_lora_active = False
            primary_text_error: BaseException | None = None
            try:
                text_execution_before = _text_execution_snapshot(c["text"])
                text_before = text_lora.dispatch_snapshot() if text_lora else None
                prompt = source_prompt
                # Transformers generation otherwise retains its hybrid StaticCache
                # on ``model._cache``. Diffusers' subsequent 49-hidden-state
                # encoding is the high-water activation phase, so that cache must
                # be released at the exact enhancement/encoding boundary.
                with torch.inference_mode():
                    if operation_spec.prompt_enhancement:
                        if text_lora is not None:
                            text_lora.set_strength(1.0)
                            text_patch_state_lora_active = True
                            text_stage.invalidate_patch_state(to_base=False)
                            text_patch_state["lora_entry_transitions"] = 1
                        progress(0.08, "Enhancing prompt")
                        phase_started = time.perf_counter()
                        prompt, prompt_enhancement_memory = _enhance_prompt(
                            c["processor"],
                            c["text"],
                            prompt,
                            LTX23_PROMPT_ENHANCEMENT_SEED,
                            self.device,
                            check_cancelled,
                        )
                        duration = timings.record("enhancement", phase_started)
                        progress(
                            0.12,
                            f"Enhanced prompt (phase_s={duration:.6f}, "
                            f"cumulative_s={timings.cumulative['enhancement']:.6f})",
                        )
                    else:
                        timings.skip("enhancement")
                    if text_before is not None and operation_spec.prompt_enhancement:
                        text_proof = text_lora.verify_dispatch(text_before)
                        text_proof["policy"] = "prompt_enhancement_only"
                    elif text_lora is not None:
                        text_proof = {
                            **text_lora.provenance(),
                            "policy": "installed_inactive_base_encode",
                            "total_dispatches": 0,
                            "minimum_target_dispatches": 0,
                            "maximum_target_dispatches": 0,
                        }
                    else:
                        text_proof = None
                    if text_lora is not None:
                        text_lora.set_strength(0.0)
                        text_stage.invalidate_patch_state(to_base=True)
                        text_patch_state_lora_active = False
                        text_patch_state["lora_to_base_transitions"] = 1
                    base_lora_snapshot = (
                        text_lora.dispatch_snapshot() if text_lora is not None else None
                    )
                    progress(0.13, "Encoding positive prompt with base Gemma")
                    phase_started = time.perf_counter()
                    prompt_embeds, prompt_mask, _, _ = base.encode_prompt(
                        prompt=prompt,
                        negative_prompt=None,
                        do_classifier_free_guidance=False,
                        max_sequence_length=1024,
                        device=self.device,
                        dtype=torch.bfloat16,
                    )
                    duration = timings.record("positive_encode", phase_started)
                    progress(
                        0.145,
                        f"Encoded positive prompt (phase_s={duration:.6f}, "
                        f"cumulative_s={timings.cumulative['positive_encode']:.6f})",
                    )
                    phase_started = time.perf_counter()
                    progress(0.15, "Encoding negative prompt with base Gemma")
                    negative_embeds, negative_mask, _, _ = base.encode_prompt(
                        prompt=negative_prompt,
                        negative_prompt=None,
                        do_classifier_free_guidance=False,
                        max_sequence_length=1024,
                        device=self.device,
                        dtype=torch.bfloat16,
                    )
                    duration = timings.record("negative_encode", phase_started)
                    progress(
                        0.16,
                        f"Encoded negative prompt (phase_s={duration:.6f}, "
                        f"cumulative_s={timings.cumulative['negative_encode']:.6f})",
                    )
                check_cancelled()
                if text_lora is not None and text_lora.dispatch_snapshot() != base_lora_snapshot:
                    raise RuntimeError("LTX text LoRA executed during base prompt encoding")
                text_native_proof = _verify_text_execution(c["text"], text_execution_before)
                negative_finite = bool(torch.isfinite(negative_embeds).all())
                if (
                    tuple(negative_embeds.shape) != (1, 1024, LTX23_GEMMA_PROMPT_EMBED_WIDTH)
                    or negative_embeds.dtype is not torch.bfloat16
                    or not negative_finite
                    or tuple(negative_mask.shape) != (1, 1024)
                    or negative_mask.dtype not in {torch.int64, torch.bool}
                    or not bool(torch.logical_or(negative_mask == 0, negative_mask == 1).all())
                ):
                    raise RuntimeError("LTX negative text node output violates its exact contract")
                negative_encoding = {
                    "prompt_sha256": hashlib.sha256(negative_prompt.encode()).hexdigest(),
                    "max_sequence_length": 1024,
                    "dtype": "bfloat16",
                    "mask_dtype": str(negative_mask.dtype).removeprefix("torch."),
                    "finite": negative_finite,
                    "encoded": True,
                    "used_for_cfg": False,
                    "embeds_shape": list(negative_embeds.shape),
                    "mask_shape": list(negative_mask.shape),
                }
            except BaseException as exc:
                primary_text_error = exc
                raise
            finally:
                try:
                    if text_lora is not None:
                        text_lora.set_strength(0.0)
                        if text_patch_state_lora_active:
                            text_stage.invalidate_patch_state(to_base=True)
                            text_patch_state_lora_active = False
                            text_patch_state["lora_to_base_transitions"] = 1
                    text_patch_state["restored_base_on_exit"] = True
                    phase_started = time.perf_counter()
                    text_stage.offload()
                    duration = timings.record("text_offload", phase_started)
                    progress(
                        0.17,
                        f"Offloaded base Gemma (phase_s={duration:.6f}, "
                        f"cumulative_s={timings.cumulative['text_offload']:.6f})",
                    )
                except BaseException as cleanup_error:
                    if primary_text_error is None:
                        raise
                    primary_text_error.add_note(
                        "LTX base-text restoration/offload also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            text_residency = text_stage.diagnostics()
            # Retain the dormant per-leaf scheduler, VBAR allocations, file
            # source, and immutable HostBuffer lanes for the same runtime/model
            # identity. ``unload`` owns the strict close/purge boundary.
            # Retain only the live conditioning references here. Copying the
            # roughly 735 MiB positive+negative CPU entry before denoise
            # compounds the media peak.
            # Publication is delayed until the MP4 and its proof are complete.
            prompt_cache_candidate = {
                "enhanced_prompt": prompt,
                "prompt_embeds": prompt_embeds,
                "prompt_mask": prompt_mask,
                "negative_prompt_embeds": negative_embeds,
                "negative_prompt_mask": negative_mask,
                "negative_encoding": negative_encoding,
                "prompt_enhancement_memory": prompt_enhancement_memory,
                "native_text": text_native_proof,
                "text_lora": text_proof,
                "text_patch_state": text_patch_state,
                "text_residency": text_residency,
            }

        memory_telemetry.capture("after_text_offload")

        generator = torch.Generator(device=self.device).manual_seed(g.seed)
        fp8_before = _fp8_dispatch_snapshot(c["transformer"])
        _reset_model_lora_dispatch(c)

        downstream_started = time.perf_counter()
        transients: dict[str, Any] = {}
        guide_preprocessing: dict[str, Any] | None = None
        if self.request.operation == "ltx23_distilled_flf":
            first = _load_rgb(g.start_image_path, g.start_image_identity)
            last = _load_rgb(g.end_image_path, g.end_image_identity)
            conditions = _conditions(
                ((first, 0, LTX23_GUIDE_STRENGTH), (last, -1, LTX23_GUIDE_STRENGTH))
            )
            transients["denoised"] = _run_denoise(
                conditioned,
                c,
                residency=residency,
                conditions=conditions,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                generator=generator,
                width=g.width,
                height=g.height,
                num_frames=g.num_frames,
                sigmas=LTX23_MAIN_SIGMAS,
                noise_scale=None,
                progress_base=0.18,
                progress_span=0.58,
                progress=progress,
                check_cancelled=check_cancelled,
            )
            transients["video_latents"] = transients["denoised"].frames
            transients["audio_latents"] = transients["denoised"].audio
            memory_telemetry.capture("after_main_denoise")
        else:
            half_width, half_height = g.width // 2, g.height // 2
            conditions = None
            pipe = base
            operation_guide: Image.Image | None = None
            if self.request.operation == "ltx23_dev_i2v":
                progress(0.17, "Preprocessing LTX image guide")
                operation_guide, guide_preprocessing = _preprocess_ltx23_i2v_guide(
                    _load_rgb(g.start_image_path, g.start_image_identity),
                    width=g.width,
                    height=g.height,
                )
                guide_preprocessing["source_file_identity"] = dict(g.start_image_identity or {})
                conditions = _conditions(
                    (
                        (
                            operation_guide,
                            0,
                            LTX23_GUIDE_STRENGTH,
                        ),
                    )
                )
                pipe = conditioned
            transients["stage1"] = _run_denoise(
                pipe,
                c,
                residency=residency,
                conditions=conditions,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                generator=generator,
                width=half_width,
                height=half_height,
                num_frames=g.num_frames,
                sigmas=LTX23_MAIN_SIGMAS,
                noise_scale=None,
                progress_base=0.18,
                progress_span=0.34,
                progress=progress,
                check_cancelled=check_cancelled,
            )
            memory_telemetry.capture("after_stage1")
            check_cancelled()
            progress(0.54, "Upscaling LTX video latents")
            _move_module(c["latent_upsampler"], self.device)
            try:
                transients["upscaled"] = upsample(
                    latents=transients["stage1"].frames,
                    latents_normalized=False,
                    height=half_height,
                    width=half_width,
                    num_frames=g.num_frames,
                    output_type="latent",
                ).frames
            finally:
                _move_module(c["latent_upsampler"], "cpu")
            memory_telemetry.capture("after_latent_upscaling")
            check_cancelled()
            refine_conditions = None
            refine_pipe = base
            if self.request.operation == "ltx23_dev_i2v":
                if operation_guide is None:
                    raise RuntimeError("LTX I2V operation guide was not retained across stages")
                refine_conditions = _conditions(((operation_guide, 0, 1.0),))
                refine_pipe = conditioned
            transients["stage2"] = _run_denoise(
                refine_pipe,
                c,
                residency=residency,
                conditions=refine_conditions,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                generator=torch.Generator(device=self.device).manual_seed(LTX23_REFINE_SEED),
                width=g.width,
                height=g.height,
                num_frames=g.num_frames,
                sigmas=LTX23_REFINE_SIGMAS,
                noise_scale=LTX23_REFINE_SIGMAS[0],
                latents=transients["upscaled"],
                audio_latents=transients["stage1"].audio,
                progress_base=0.58,
                progress_span=0.18,
                progress=progress,
                check_cancelled=check_cancelled,
            )
            memory_telemetry.capture("after_stage2")
            # The official two-stage topology keeps stage-one audio. Stage two
            # uses a noised copy for cross-modal refinement but its audio output
            # is deliberately discarded.
            transients["video_latents"] = transients["stage2"].frames
            transients["audio_latents"] = transients["stage1"].audio

        fp8_proof = _native_dispatch_proof(c["transformer"], fp8_before)
        model_lora_proof = (
            ltx23_model_lora_dispatch_evidence(c["transformer"], c["model_lora"])
            if c["model_lora"] is not None
            else None
        )
        if model_lora_proof is not None and not model_lora_proof["complete"]:
            raise RuntimeError("LTX 2.3 model LoRA did not dispatch on every selected target")

        progress(0.79, "Decoding LTX video and audio")
        transients["frames"], transients["audio"] = _decode_media(
            c,
            transients["video_latents"],
            transients["audio_latents"],
            self.device,
            check_cancelled,
        )
        memory_telemetry.capture("after_decode")
        progress(0.91, "Muxing 25 fps video and 48 kHz stereo audio")
        output = Path(g.output_path).resolve(strict=False)
        audio_duration_normalization = _mux_mp4(
            transients["frames"],
            transients["audio"],
            output,
            check_cancelled=check_cancelled,
        )
        observed = _probe_mp4(output, check_cancelled)
        if (
            observed["width"] != g.width
            or observed["height"] != g.height
            or observed["num_frames"] != g.num_frames
            or observed["fps"] != LTX23_FPS
            or observed["audio_sample_rate"] != LTX23_AUDIO_SAMPLE_RATE
            or observed["audio_channels"] != LTX23_AUDIO_CHANNELS
        ):
            raise RuntimeError("LTX 2.3 published MP4 does not match its requested A/V contract")
        output_size = output.stat().st_size
        output_sha256 = _sha256_file(output, check_cancelled)
        duration = timings.record("downstream", downstream_started)
        progress(
            0.99,
            f"Completed LTX downstream phases (phase_s={duration:.6f}, "
            f"cumulative_s={timings.cumulative['downstream']:.6f})",
        )
        metadata = {
            "family": "ltx23",
            "runtime": "engine-native/ltx23-kitchen",
            "operation": self.request.operation,
            "request_fingerprint": self.request.fingerprint,
            "component_fingerprint": self.request.component_fingerprint,
            "seed": g.seed,
            **observed,
            "audio_duration_normalization": audio_duration_normalization,
            "output_size_bytes": output_size,
            "output_sha256": output_sha256,
            "components": self.request.public_component_manifest(),
            "main_sigmas": list(LTX23_MAIN_SIGMAS),
            "refine_sigmas": list(LTX23_REFINE_SIGMAS)
            if self.request.operation != "ltx23_distilled_flf"
            else None,
            "prompt_enhanced": operation_spec.prompt_enhancement,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_enhancement_system_sha256": _prompt_system_sha256()
            if operation_spec.prompt_enhancement
            else None,
            "prompt_enhancement_seed": LTX23_PROMPT_ENHANCEMENT_SEED
            if operation_spec.prompt_enhancement
            else None,
            "prompt_enhancement_max_new_tokens": LTX23_PROMPT_MAX_NEW_TOKENS
            if operation_spec.prompt_enhancement
            else None,
            "prompt_enhancement_stop_token_id": LTX23_PROMPT_STOP_TOKEN_ID
            if operation_spec.prompt_enhancement
            else None,
            "prompt_enhancement_template": "comfy_ltx2_gemma3_manual_v1"
            if operation_spec.prompt_enhancement
            else None,
            "prompt_enhancement_generation_settings": dict(LTX23_PROMPT_GENERATION_SETTINGS)
            if operation_spec.prompt_enhancement
            else None,
            "prompt_enhancement_memory": prompt_enhancement_memory,
            "negative_encoding": negative_encoding,
            "text_patch_state": text_patch_state,
            "refine_seed": LTX23_REFINE_SEED if operation_spec.refine_sigmas else None,
            "negative_prompt": negative_prompt,
            "guide_strengths": list(operation_spec.guide_strengths),
            "guide_preprocessing": guide_preprocessing,
            "model_lora_strength": operation_spec.model_lora_strength,
            "text_lora_strength": operation_spec.text_lora_strength,
            "native_fp8": fp8_proof,
            "native_text": text_native_proof,
            "model_lora": model_lora_proof,
            "text_lora": text_proof,
            "text_residency": text_residency,
            "timings": timings.metadata(),
            "dense_base_dequantizations": 0,
            "residency_policy": residency.policy,
            "video_vae_residency": {
                **video_vae_residency.policy,
                "job_delta": video_vae_residency.diagnostics_delta(video_vae_before),
            },
            "pipeline": "diffusers/LTX2Pipeline+LTX2ConditionPipeline+LTX2LatentUpsamplePipeline",
            "_prompt_cache_hit": prompt_cache_hit,
            "_prompt_cache_published": prompt_cache_published,
        }
        result = LTX23KitchenResult(output, metadata)
        return _finalize_ltx23_kitchen_result(
            result,
            transients=transients,
            cache=self._cache,
            cache_policy=self.cache_policy,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_candidate=prompt_cache_candidate,
            check_cancelled=check_cancelled,
            timings=timings,
            progress=progress,
            memory_telemetry=memory_telemetry,
        )


def _release_ltx23_generation_transients(transients: dict[str, Any]) -> None:
    """Drop large denoise, latent, decoded-frame, and decoded-audio references."""

    transients.clear()


def _finalize_ltx23_kitchen_result(
    result: LTX23KitchenResult,
    *,
    transients: dict[str, Any],
    cache: RuntimeCache,
    cache_policy: str,
    prompt_cache_key: str,
    prompt_cache_candidate: dict[str, Any] | None,
    check_cancelled: LTX23KitchenCancellation,
    timings: _LTX23PhaseTimings | None = None,
    progress: LTX23KitchenProgress | None = None,
    memory_telemetry: PhaseMemoryTelemetry | None = None,
) -> LTX23KitchenResult:
    """Release media peak, then fail-closed publish cold prompt conditioning."""

    published = False
    try:
        _release_ltx23_generation_transients(transients)
        if memory_telemetry is not None:
            memory_telemetry.capture("after_transient_clearing")
        check_cancelled()
        prompt_hit = result.metadata.get("_prompt_cache_hit") is True
        if cache_policy == "prompt" and not prompt_hit:
            if prompt_cache_candidate is None:
                raise RuntimeError("LTX cold prompt conditioning candidate is missing")
            phase_started = time.perf_counter()
            published = cache.prompt.put(prompt_cache_key, prompt_cache_candidate)
            if not published:
                raise RuntimeError(
                    "LTX prompt conditioning did not fit its bounded CPU cache; "
                    "select cache policy 'none' or update the pinned cache budget"
                )
            if timings is not None:
                duration = timings.record("prompt_cache_publish", phase_started)
                if progress is not None:
                    progress(
                        0.995,
                        f"Published LTX prompt cache (phase_s={duration:.6f}, "
                        "cumulative_s="
                        f"{timings.cumulative['prompt_cache_publish']:.6f})",
                    )
        elif timings is not None:
            timings.skip("prompt_cache_publish")
            if progress is not None:
                progress(
                    0.995,
                    "Skipped LTX prompt cache publication "
                    f"(phase_s=0.000000, cumulative_s="
                    f"{timings.cumulative['prompt_cache_publish']:.6f})",
                )
        if memory_telemetry is not None:
            memory_telemetry.capture("after_prompt_cache_publication")
        check_cancelled()
    except BaseException:
        if published:
            cache.clear()
        raise
    result.metadata["_prompt_cache_published"] = published
    if timings is not None:
        result.metadata["timings"] = timings.metadata()
    if memory_telemetry is not None:
        result.metadata["memory_telemetry"] = memory_telemetry.metadata()
    return result


def _build_pipelines(c: Mapping[str, Any], device: torch.device) -> tuple[Any, Any, Any]:
    from diffusers import LTX2ConditionPipeline, LTX2LatentUpsamplePipeline, LTX2Pipeline

    class BoundPipeline(LTX2Pipeline):
        @property
        def _execution_device(self) -> torch.device:
            return device

    class BoundConditionPipeline(LTX2ConditionPipeline):
        @property
        def _execution_device(self) -> torch.device:
            return device

    class BoundUpsamplePipeline(LTX2LatentUpsamplePipeline):
        @property
        def _execution_device(self) -> torch.device:
            return device

    common = {
        "scheduler": c["scheduler"],
        "vae": c["video_vae"],
        "audio_vae": c["audio_vae"],
        "text_encoder": c["text"],
        "tokenizer": c["processor"].tokenizer,
        "connectors": c["connector"],
        "transformer": c["transformer"],
        "vocoder": c["vocoder"],
    }
    base_common = {**common, "processor": c["processor"]}
    base = BoundPipeline(**base_common)
    conditioned = BoundConditionPipeline(**common)
    upsample = (
        BoundUpsamplePipeline(vae=c["video_vae"], latent_upsampler=c.get("latent_upsampler"))
        if c.get("latent_upsampler") is not None
        else None
    )
    return base, conditioned, upsample


def _run_denoise(
    pipeline: Any,
    c: Mapping[str, Any],
    *,
    residency: _LTX23TransformerResidency,
    conditions: Any,
    negative_prompt: str,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    generator: torch.Generator,
    width: int,
    height: int,
    num_frames: int,
    sigmas: tuple[float, ...],
    noise_scale: float | None,
    progress_base: float,
    progress_span: float,
    progress: LTX23KitchenProgress,
    check_cancelled: LTX23KitchenCancellation,
    latents: torch.Tensor | None = None,
    audio_latents: torch.Tensor | None = None,
) -> Any:
    connector_handles = [
        c["connector"].register_forward_pre_hook(
            lambda module, _inputs: _move_module(module, pipeline._execution_device)
        ),
        c["connector"].register_forward_hook(
            lambda module, _inputs, output: (_move_module(module, "cpu"), output)[1],
            always_call=True,
        ),
    ]
    def before_transformer() -> None:
        check_cancelled()

    step_count = len(sigmas) - 1

    def callback(_pipe: Any, index: int, _timestep: Any, values: dict[str, Any]) -> dict[str, Any]:
        check_cancelled()
        progress(
            progress_base + progress_span * ((index + 1) / step_count),
            f"LTX denoise step {index + 1}/{step_count}",
        )
        return values

    kwargs = {
        "prompt": None,
        "negative_prompt": negative_prompt,
        "prompt_embeds": prompt_embeds,
        "prompt_attention_mask": prompt_mask,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frame_rate": float(LTX23_FPS),
        "num_inference_steps": step_count,
        "sigmas": _diffusers_sigmas(sigmas),
        "guidance_scale": LTX23_GUIDANCE_SCALE,
        "generator": generator,
        "latents": latents,
        "audio_latents": audio_latents,
        "noise_scale": noise_scale,
        "use_cross_timestep": True,
        "output_type": "latent",
        "callback_on_step_end": callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }
    if conditions is not None:
        kwargs["conditions"] = conditions
    try:
        with residency.forward_scope(before_transformer):
            return pipeline(**kwargs)
    finally:
        for handle in connector_handles:
            handle.remove()
        _move_module(c["connector"], "cpu")


def _enhance_prompt(
    processor: Any,
    model: Any,
    prompt: str,
    seed: int,
    device: torch.device,
    check_cancelled: LTX23KitchenCancellation,
) -> tuple[str, dict[str, Any]]:
    """Pinned public Gemma generation path without moving its meta-only vision shell."""

    check_cancelled()
    template = _ltx23_prompt_enhancement_template(prompt)
    inputs = _tokenize_ltx23_prompt_enhancement(processor, template, device)
    from transformers import LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

    class CancellationCriteria(StoppingCriteria):
        def __call__(self, *_args: Any, **_kwargs: Any) -> bool:
            check_cancelled()
            return False

    execution_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float32
    )
    direct_comfy = bool(getattr(model, "_latentslate_comfy_gemma_direct", False))
    sampling = None
    if not direct_comfy:
        sampling = _ComfyLTX23LogitsProcessor(
            prompt_length=inputs["input_ids"].shape[1],
            device=device,
            execution_dtype=execution_dtype,
            seed=seed,
        )

    try:
        if direct_comfy:
            generated = model.generate(
                **inputs,
                max_new_tokens=LTX23_PROMPT_MAX_NEW_TOKENS,
                eos_token_id=LTX23_PROMPT_STOP_TOKEN_ID,
                do_sample=bool(LTX23_PROMPT_GENERATION_SETTINGS["do_sample"]),
                temperature=float(LTX23_PROMPT_GENERATION_SETTINGS["temperature"]),
                top_k=int(LTX23_PROMPT_GENERATION_SETTINGS["top_k"]),
                top_p=float(LTX23_PROMPT_GENERATION_SETTINGS["top_p"]),
                min_p=float(LTX23_PROMPT_GENERATION_SETTINGS["min_p"]),
                repetition_penalty=float(LTX23_PROMPT_GENERATION_SETTINGS["repetition_penalty"]),
                seed=seed,
                execution_dtype=execution_dtype,
                check_cancelled=check_cancelled,
            )
        else:
            generated = model.generate(
                **inputs,
                max_new_tokens=LTX23_PROMPT_MAX_NEW_TOKENS,
                eos_token_id=LTX23_PROMPT_STOP_TOKEN_ID,
                # Structural test doubles and explicit fallback shells retain
                # the previous Transformers compatibility seam.
                do_sample=False,
                repetition_penalty=1.0,
                logits_processor=LogitsProcessorList([sampling]),
                stopping_criteria=StoppingCriteriaList([CancellationCriteria()]),
            )
    except BaseException as generation_error:
        try:
            _release_transformers_generation_cache(model, device)
        except BaseException as cleanup_error:  # noqa: BLE001 - preserve primary failure
            generation_error.add_note(
                "LTX prompt cache cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise
    else:
        # Transformers 5.9's hybrid/static cache is intentionally persistent:
        # ``generate`` stores it on the model and later calls only reset its
        # contents.  That is counterproductive here because prompt enhancement
        # and 49-layer prompt encoding are mutually exclusive Engine stages.
        # Reset first for the cache contract, then remove the owning reference
        # so CUDA storage is actually reclaimable before encoding begins.
        cache_release = _release_transformers_generation_cache(model, device)
    check_cancelled()
    suffixes = [row[len(inputs["input_ids"][index]) :] for index, row in enumerate(generated)]
    values = processor.tokenizer.batch_decode(suffixes, skip_special_tokens=True)
    if len(values) != 1:
        raise RuntimeError("LTX 2.3 prompt enhancement returned an invalid batch")
    decoded = values[0]
    enhanced = re.sub(r"<think>.*?(?:</think>|$)", "", decoded, flags=re.DOTALL).strip()
    cache_release = {
        **cache_release,
        "template": "comfy_ltx2_gemma3_manual_v1",
        "stop_token_id": LTX23_PROMPT_STOP_TOKEN_ID,
        "generation_settings": dict(LTX23_PROMPT_GENERATION_SETTINGS),
        "decoded_suffix_nonempty": bool(decoded.strip()),
        "think_block_removed": decoded.strip() != enhanced,
        "fallback_to_source_prompt": not bool(enhanced),
    }
    return enhanced or prompt, cache_release


def _tokenize_ltx23_prompt_enhancement(
    processor: Any,
    template: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Match the pinned Comfy Gemma tokenizer boundary exactly.

    Comfy's SentencePiece wrapper inserts BOS, then its segmented special-token
    path drops the first token after BOS. Its SDTokenizer subsequently left
    pads to at least 1,024 positions. Gemma generation discards the tokenizer
    mask, so those pad positions remain causal context rather than being masked.
    """

    encoded = processor(text=template, images=None, return_tensors="pt")
    input_ids = encoded.input_ids
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError("LTX prompt enhancement requires one tokenized sequence")
    if input_ids.shape[1] < 2 or input_ids[0, :2].tolist() != [2, 105]:
        raise RuntimeError("LTX prompt enhancement tokenizer contract changed")

    input_ids = torch.cat((input_ids[:, :1], input_ids[:, 2:]), dim=1)
    pad_length = max(0, 1_024 - input_ids.shape[1])
    if pad_length:
        padding = torch.zeros(
            (1, pad_length),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat((padding, input_ids), dim=1)
    input_ids = input_ids.to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, device=device),
    }


def _ltx23_prompt_enhancement_template(prompt: str) -> str:
    """Reproduce the pinned Comfy Gemma-3 TextGenerateLTX2Prompt template."""

    return (
        f"<start_of_turn>system\n{_prompt_system_text().strip()}<end_of_turn>\n"
        f"<start_of_turn>user\n\nUser Raw Input Prompt: {prompt}.<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def _release_transformers_generation_cache(
    model: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Release a Transformers-owned persistent generation cache between text stages."""

    cache = getattr(model, "_cache", None)
    before_allocated = _cuda_allocated_bytes(device)
    if cache is None:
        return {
            "policy": "release_after_prompt_enhancement",
            "cache_present": False,
            "cache_type": None,
            "cuda_allocated_before_bytes": before_allocated,
            "cuda_allocated_after_bytes": before_allocated,
            "cuda_allocated_released_bytes": 0,
        }

    from transformers.cache_utils import Cache

    if not isinstance(cache, Cache):
        raise TypeError("LTX Gemma generation retained an unsupported cache owner")
    reset = getattr(cache, "reset", None)
    if not callable(reset):
        raise TypeError("LTX Gemma generation cache cannot be reset safely")
    cache_type = f"{type(cache).__module__}.{type(cache).__qualname__}"
    reset()
    delattr(model, "_cache")
    del cache
    if hasattr(model, "_cache"):
        raise RuntimeError("LTX Gemma generation cache owner survived explicit release")
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    after_allocated = _cuda_allocated_bytes(device)
    released = (
        None
        if before_allocated is None or after_allocated is None
        else max(0, before_allocated - after_allocated)
    )
    return {
        "policy": "release_after_prompt_enhancement",
        "cache_present": True,
        "cache_type": cache_type,
        "cuda_allocated_before_bytes": before_allocated,
        "cuda_allocated_after_bytes": after_allocated,
        "cuda_allocated_released_bytes": released,
    }


def _cuda_allocated_bytes(device: torch.device) -> int | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return int(torch.cuda.memory_allocated(device))


def _decode_media(
    c: Mapping[str, Any],
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    device: torch.device,
    check_cancelled: LTX23KitchenCancellation,
) -> tuple[np.ndarray, LTX23DecodedAudio]:
    vae = c["video_vae"]
    _configure_ltx23_video_vae(vae)
    check_cancelled()
    video = vae.decode(
        video_latents.to(device=device, dtype=vae.dtype), None, return_dict=False
    )[0]
    frames = c["_video_processor"].postprocess_video(video, output_type="np")
    check_cancelled()
    audio_vae = c["audio_vae"]
    _move_module(audio_vae, device)
    try:
        mel = audio_vae.decode(
            audio_latents.to(device=device, dtype=audio_vae.dtype), return_dict=False
        )[0]
    finally:
        _move_module(audio_vae, "cpu")
    check_cancelled()
    vocoder = c["vocoder"]
    _move_module(vocoder, device)
    try:
        audio = vocoder(mel.to(device=device, dtype=vocoder.dtype))
        audio = audio[0].float().cpu().numpy()
    finally:
        _move_module(vocoder, "cpu")
    if isinstance(frames, list) or isinstance(frames, np.ndarray) and frames.ndim == 5:
        frames = frames[0]
    frames = np.asarray(frames)
    decoded_audio = _decoded_audio_proof(
        audio_latents, mel, audio, audio_vae, vocoder, video_frames=int(frames.shape[0])
    )
    return frames, decoded_audio


def _configure_ltx23_video_vae(vae: Any) -> None:
    """Apply the hidden Comfy VAE decode defaults to the Diffusers VAE."""

    enable_tiling = getattr(vae, "enable_tiling", None)
    if not callable(enable_tiling):
        raise TypeError("LTX 2.3 video VAE does not expose tiled decoding")
    enable_tiling(
        tile_sample_min_height=LTX23_VAE_TILE_SAMPLE_MIN_HEIGHT,
        tile_sample_min_width=LTX23_VAE_TILE_SAMPLE_MIN_WIDTH,
        tile_sample_min_num_frames=LTX23_VAE_TILE_SAMPLE_MIN_NUM_FRAMES,
        tile_sample_stride_height=LTX23_VAE_TILE_SAMPLE_STRIDE_HEIGHT,
        tile_sample_stride_width=LTX23_VAE_TILE_SAMPLE_STRIDE_WIDTH,
        tile_sample_stride_num_frames=LTX23_VAE_TILE_SAMPLE_STRIDE_NUM_FRAMES,
    )
    vae.use_framewise_decoding = True


def _decoded_audio_proof(
    audio_latents: torch.Tensor,
    mel: torch.Tensor,
    waveform: torch.Tensor | np.ndarray,
    audio_vae: Any,
    vocoder: Any,
    *,
    video_frames: int,
) -> LTX23DecodedAudio:
    """Close the LTX audio decoder arithmetic before duration normalization."""

    if isinstance(video_frames, bool) or not isinstance(video_frames, int) or video_frames <= 0:
        raise ValueError("LTX 2.3 decoded video frame count is invalid")
    if (
        audio_latents.ndim != 4
        or tuple(audio_latents.shape[:2]) != (1, 8)
        or audio_latents.shape[3] != 16
    ):
        raise ValueError("LTX 2.3 audio latents must have layout [1,8,L,16]")
    if mel.ndim != 4 or tuple(mel.shape[:2]) != (1, 2) or mel.shape[3] != 64:
        raise ValueError("LTX 2.3 decoded mel must have layout [1,2,M,64]")
    source_sample_rate = _exact_config_int(audio_vae.config, "sample_rate")
    mel_hop_length = _exact_config_int(audio_vae.config, "mel_hop_length")
    temporal_compression_ratio = _exact_int(
        getattr(audio_vae, "temporal_compression_ratio", None),
        "audio VAE temporal compression ratio",
    )
    vocoder_source_sample_rate = _exact_config_int(vocoder.config, "input_sampling_rate")
    output_sample_rate = _exact_config_int(vocoder.config, "output_sampling_rate")
    latent_channels = _exact_config_int(audio_vae.config, "latent_channels")
    mel_bins = _exact_config_int(audio_vae.config, "mel_bins")
    output_channels = _exact_config_int(audio_vae.config, "output_channels")
    vocoder_channels = _exact_config_int(vocoder.config, "out_channels")
    bwe_channels = _exact_config_int(vocoder.config, "bwe_out_channels")
    causality_axis = getattr(audio_vae.config, "causality_axis", None)
    is_causal = getattr(audio_vae.config, "is_causal", None)
    if (
        source_sample_rate != LTX23_AUDIO_SOURCE_SAMPLE_RATE
        or vocoder_source_sample_rate != source_sample_rate
        or mel_hop_length != LTX23_AUDIO_MEL_HOP_LENGTH
        or temporal_compression_ratio != LTX23_AUDIO_TEMPORAL_COMPRESSION_RATIO
        or output_sample_rate != LTX23_AUDIO_SAMPLE_RATE
        or latent_channels != 8
        or mel_bins != 64
        or output_channels != LTX23_AUDIO_CHANNELS
        or vocoder_channels != LTX23_AUDIO_CHANNELS
        or bwe_channels != LTX23_AUDIO_CHANNELS
        or causality_axis != "height"
        or is_causal is not True
    ):
        raise ValueError("LTX 2.3 audio decoder configuration does not match the pinned contract")

    audio_latent_frames = int(audio_latents.shape[2])
    expected_audio_latent_frames = round((video_frames / LTX23_FPS) * 25)
    decoded_mel_frames = int(mel.shape[2])
    if audio_latent_frames <= 0 or decoded_mel_frames <= 0:
        raise ValueError("LTX 2.3 audio decoder grids must be nonempty")
    if audio_latent_frames != expected_audio_latent_frames:
        raise ValueError("LTX 2.3 audio latent frame count does not match its decoded video grid")
    expected_mel_frames = audio_latent_frames * temporal_compression_ratio - (
        temporal_compression_ratio - 1
    )
    if decoded_mel_frames != expected_mel_frames:
        raise ValueError("LTX 2.3 decoded mel frame count does not match its source latent grid")
    expected_decoded_samples = (
        expected_mel_frames * mel_hop_length * output_sample_rate // source_sample_rate
    )
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] != LTX23_AUDIO_CHANNELS or not audio.shape[1]:
        raise ValueError("LTX 2.3 decoded waveform must have layout [2,S]")
    if not np.isfinite(audio).all():
        raise ValueError("LTX 2.3 decoded waveform samples must be finite")
    decoded_samples = int(audio.shape[1])
    if decoded_samples != expected_decoded_samples:
        raise ValueError("LTX 2.3 decoded waveform length does not match its source mel grid")
    return LTX23DecodedAudio(
        waveform=np.ascontiguousarray(audio),
        video_frames=video_frames,
        audio_latent_frames=audio_latent_frames,
        expected_audio_latent_frames=expected_audio_latent_frames,
        audio_latent_channels=int(audio_latents.shape[1]),
        audio_latent_mel_bins=int(audio_latents.shape[3]),
        decoded_mel_frames=decoded_mel_frames,
        expected_mel_frames=expected_mel_frames,
        decoded_mel_channels=int(mel.shape[1]),
        decoded_mel_bins=int(mel.shape[3]),
        decoded_samples=decoded_samples,
        expected_decoded_samples=expected_decoded_samples,
        source_sample_rate=source_sample_rate,
        output_sample_rate=output_sample_rate,
        mel_hop_length=mel_hop_length,
        temporal_compression_ratio=temporal_compression_ratio,
        causality_axis=causality_axis,
        is_causal=is_causal,
    )


def _exact_config_int(config: Any, name: str) -> int:
    return _exact_int(getattr(config, name, None), f"audio decoder config {name}")


def _exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"LTX 2.3 {label} must be an integer")
    return value


def _mux_mp4(
    frames: np.ndarray,
    audio: LTX23DecodedAudio,
    output: Path,
    *,
    check_cancelled: LTX23KitchenCancellation,
) -> dict[str, int | str]:
    import av

    frames = _uint8_frames(frames)
    if audio.video_frames != int(frames.shape[0]):
        raise ValueError("LTX 2.3 decoded audio proof does not bind to the mux video frame count")
    required_audio_samples = frames.shape[0] * LTX23_AUDIO_SAMPLE_RATE // LTX23_FPS
    audio, audio_duration_normalization = _normalize_audio_duration(audio, required_audio_samples)
    audio = _audio_for_encoding(audio)
    audio_duration_normalization.update(
        {
            "fps": LTX23_FPS,
            "audio_channels": LTX23_AUDIO_CHANNELS,
        }
    )
    staging = output.with_name(f".{output.name}.{os.getpid()}.tmp.mp4")
    staging.unlink(missing_ok=True)
    try:
        with av.open(str(staging), "w") as container:
            video_stream = container.add_stream("libx264", rate=LTX23_FPS)
            video_stream.width = int(frames.shape[2])
            video_stream.height = int(frames.shape[1])
            video_stream.pix_fmt = "yuv420p"
            audio_stream = container.add_stream("aac", rate=LTX23_AUDIO_SAMPLE_RATE)
            audio_stream.layout = "stereo"
            for index, pixels in enumerate(frames):
                check_cancelled()
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts = index
                frame.time_base = Fraction(1, LTX23_FPS)
                for packet in video_stream.encode(frame):
                    container.mux(packet)
            for packet in video_stream.encode():
                container.mux(packet)
            for offset in range(0, audio.shape[1], 1024):
                check_cancelled()
                chunk = np.ascontiguousarray(audio[:, offset : offset + 1024])
                frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo")
                frame.sample_rate = LTX23_AUDIO_SAMPLE_RATE
                frame.pts = offset
                frame.time_base = Fraction(1, LTX23_AUDIO_SAMPLE_RATE)
                for packet in audio_stream.encode(frame):
                    container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)
        check_cancelled()
        os.replace(staging, output)
    finally:
        staging.unlink(missing_ok=True)
    return audio_duration_normalization


def _audio_for_encoding(audio: np.ndarray) -> np.ndarray:
    """Clip a proven finite waveform exactly once at the codec boundary."""

    return np.ascontiguousarray(np.clip(audio, -1.0, 1.0))


def _normalize_audio_duration(
    decoded: LTX23DecodedAudio, required_samples: int
) -> tuple[np.ndarray, dict[str, int | str]]:
    """Bind proven source-valid audio exactly to the decoded picture duration."""

    if (
        isinstance(required_samples, bool)
        or not isinstance(required_samples, int)
        or required_samples <= 0
    ):
        raise ValueError("LTX 2.3 target audio length must be a positive integer")
    _validate_decoded_audio_proof(decoded)
    audio = decoded.waveform
    decoded_samples = decoded.decoded_samples
    if decoded_samples > required_samples:
        normalized = audio[:, :required_samples]
        trimmed_samples = decoded_samples - required_samples
        padded_samples = 0
    elif decoded_samples < required_samples:
        padded_samples = required_samples - decoded_samples
        normalized = np.pad(audio, ((0, 0), (0, padded_samples)), mode="constant")
        trimmed_samples = 0
    else:
        normalized = audio
        trimmed_samples = 0
        padded_samples = 0
    return np.ascontiguousarray(normalized), {
        "policy": LTX23_AUDIO_DURATION_POLICY,
        "reason": "independent_audio_grid_causal_tail",
        "video_frames": decoded.video_frames,
        "audio_latent_frames": decoded.audio_latent_frames,
        "expected_audio_latent_frames": decoded.expected_audio_latent_frames,
        "audio_latent_channels": decoded.audio_latent_channels,
        "audio_latent_mel_bins": decoded.audio_latent_mel_bins,
        "decoded_mel_frames": decoded.decoded_mel_frames,
        "expected_mel_frames": decoded.expected_mel_frames,
        "decoded_mel_channels": decoded.decoded_mel_channels,
        "decoded_mel_bins": decoded.decoded_mel_bins,
        "decoded_samples": decoded_samples,
        "expected_decoded_samples": decoded.expected_decoded_samples,
        "target_samples": required_samples,
        "source_sample_rate": decoded.source_sample_rate,
        "output_sample_rate": decoded.output_sample_rate,
        "mel_hop_length": decoded.mel_hop_length,
        "temporal_compression_ratio": decoded.temporal_compression_ratio,
        "causality_axis": decoded.causality_axis,
        "is_causal": decoded.is_causal,
        "trimmed_samples": trimmed_samples,
        "padded_samples": padded_samples,
    }


def _validate_decoded_audio_proof(decoded: LTX23DecodedAudio) -> None:
    if not isinstance(decoded, LTX23DecodedAudio):
        raise TypeError("LTX 2.3 mux requires source-derived decoded audio evidence")
    integer_fields = (
        decoded.audio_latent_frames,
        decoded.expected_audio_latent_frames,
        decoded.audio_latent_channels,
        decoded.audio_latent_mel_bins,
        decoded.decoded_mel_frames,
        decoded.expected_mel_frames,
        decoded.decoded_mel_channels,
        decoded.decoded_mel_bins,
        decoded.decoded_samples,
        decoded.expected_decoded_samples,
        decoded.source_sample_rate,
        decoded.output_sample_rate,
        decoded.mel_hop_length,
        decoded.temporal_compression_ratio,
        decoded.video_frames,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in integer_fields
    ):
        raise ValueError("LTX 2.3 decoded audio evidence contains invalid integer fields")
    if (
        decoded.source_sample_rate != LTX23_AUDIO_SOURCE_SAMPLE_RATE
        or decoded.output_sample_rate != LTX23_AUDIO_SAMPLE_RATE
        or decoded.mel_hop_length != LTX23_AUDIO_MEL_HOP_LENGTH
        or decoded.temporal_compression_ratio != LTX23_AUDIO_TEMPORAL_COMPRESSION_RATIO
        or decoded.causality_axis != "height"
        or decoded.is_causal is not True
        or decoded.audio_latent_channels != 8
        or decoded.audio_latent_mel_bins != 16
        or decoded.decoded_mel_channels != 2
        or decoded.decoded_mel_bins != 64
    ):
        raise ValueError("LTX 2.3 decoded audio evidence does not match the pinned contract")
    expected_audio_latent_frames = round((decoded.video_frames / LTX23_FPS) * 25)
    expected_mel_frames = decoded.audio_latent_frames * decoded.temporal_compression_ratio - (
        decoded.temporal_compression_ratio - 1
    )
    expected_decoded_samples = (
        expected_mel_frames
        * decoded.mel_hop_length
        * decoded.output_sample_rate
        // decoded.source_sample_rate
    )
    if (
        decoded.audio_latent_frames != expected_audio_latent_frames
        or decoded.expected_audio_latent_frames != expected_audio_latent_frames
        or decoded.decoded_mel_frames != expected_mel_frames
        or decoded.expected_mel_frames != expected_mel_frames
        or decoded.decoded_samples != expected_decoded_samples
        or decoded.expected_decoded_samples != expected_decoded_samples
    ):
        raise ValueError("LTX 2.3 decoded audio evidence does not close arithmetically")
    waveform = np.asarray(decoded.waveform)
    if (
        waveform.ndim != 2
        or waveform.shape != (LTX23_AUDIO_CHANNELS, decoded.decoded_samples)
        or not np.issubdtype(waveform.dtype, np.floating)
        or not np.isfinite(waveform).all()
    ):
        raise ValueError("LTX 2.3 decoded audio evidence does not match its finite stereo waveform")


def _probe_mp4(
    output: Path, check_cancelled: LTX23KitchenCancellation
) -> dict[str, int | float | str]:
    """Observe the published container instead of echoing request-side media facts."""

    import av

    check_cancelled()
    with av.open(str(output)) as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) != 1:
            raise ValueError("LTX 2.3 output must contain exactly one video and one audio stream")
        video = container.streams.video[0]
        audio = container.streams.audio[0]
        rate = video.average_rate
        if rate is None or rate.denominator == 0:
            raise ValueError("LTX 2.3 output video frame rate is unavailable")
        observed: dict[str, int | float | str] = {
            "container_format": container.format.name,
            "video_codec": video.codec_context.name,
            "audio_codec": audio.codec_context.name,
            "width": int(video.codec_context.width),
            "height": int(video.codec_context.height),
            "fps": int(rate) if rate.denominator == 1 else float(rate),
            "audio_sample_rate": int(audio.codec_context.sample_rate),
            "audio_channels": len(audio.codec_context.layout.channels),
        }
        frame_count = 0
        for _frame in container.decode(video=0):
            check_cancelled()
            frame_count += 1
    with av.open(str(output)) as container:
        audio_samples = 0
        for frame in container.decode(audio=0):
            check_cancelled()
            audio_samples += int(frame.samples)
    observed["num_frames"] = frame_count
    observed["audio_samples"] = audio_samples
    observed["video_duration_seconds"] = frame_count / float(observed["fps"])
    observed["audio_duration_seconds"] = audio_samples / int(observed["audio_sample_rate"])
    if "mp4" not in str(observed["container_format"]).split(","):
        raise ValueError("LTX 2.3 output container is not MP4")
    if observed["video_codec"] != "h264" or observed["audio_codec"] != "aac":
        raise ValueError("LTX 2.3 output codecs are not H.264/AAC")
    # The mux input is duration-normalized exactly. AAC may expose only the
    # final one-sided encoder packet containing the end padding.
    target_audio_samples = frame_count * int(observed["audio_sample_rate"]) // int(observed["fps"])
    if not target_audio_samples <= audio_samples < target_audio_samples + LTX23_AAC_PACKET_SAMPLES:
        raise ValueError("LTX 2.3 output audio/video durations drift beyond tolerance")
    return observed


def _uint8_frames(value: np.ndarray) -> np.ndarray:
    frames = np.asarray(value)
    if frames.ndim != 4 or frames.shape[-1] != 3 or not frames.shape[0]:
        raise ValueError("LTX 2.3 video must be nonempty FHWC RGB")
    if frames.dtype == np.uint8:
        return np.ascontiguousarray(frames)
    if not np.issubdtype(frames.dtype, np.floating) or not np.isfinite(frames).all():
        raise ValueError("LTX 2.3 video samples must be finite floats or uint8")
    return np.ascontiguousarray((np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8))


def _stereo_audio(value: np.ndarray) -> np.ndarray:
    audio = np.asarray(value, dtype=np.float32)
    if audio.ndim != 2:
        raise ValueError("LTX 2.3 audio must be a two-dimensional stereo waveform")
    if audio.shape[0] != LTX23_AUDIO_CHANNELS and audio.shape[1] == LTX23_AUDIO_CHANNELS:
        audio = audio.T
    if audio.shape[0] != LTX23_AUDIO_CHANNELS or not audio.shape[1]:
        raise ValueError("LTX 2.3 audio must contain exactly two nonempty channels")
    if not np.isfinite(audio).all():
        raise ValueError("LTX 2.3 audio samples must be finite")
    return np.ascontiguousarray(np.clip(audio, -1.0, 1.0))


def _conditions(items: tuple[tuple[Image.Image, int, float], ...]) -> list[Any]:
    from diffusers.pipelines.ltx2 import LTX2VideoCondition

    return [
        LTX2VideoCondition(frames=image, index=index, strength=strength)
        for image, index, strength in items
    ]


def ltx23_guide_identity(path: Path) -> dict[str, int | str]:
    """Hash one guide image with a stable stat envelope for request binding."""

    candidate = Path(path).resolve(strict=True)
    before = candidate.stat()
    digest = _sha256_file(candidate, lambda: None)
    after = candidate.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("LTX 2.3 guide changed while its identity was measured")
    return {
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
    }


def _load_rgb(path: Path | None, identity: Mapping[str, object] | None) -> Image.Image:
    if path is None or identity is None:
        raise ValueError("LTX 2.3 guide path or identity is missing")
    if ltx23_guide_identity(path) != dict(identity):
        raise ValueError("LTX 2.3 guide image changed after request binding")
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _preprocess_ltx23_i2v_guide(
    image: Image.Image,
    *,
    width: int,
    height: int,
) -> tuple[Image.Image, dict[str, Any]]:
    """Apply the pinned Comfy I2V guide nodes once for both denoise stages.

    This is a narrow source adaptation of pinned ComfyUI's
    ``ResizeImageMaskNode(scale dimensions, lanczos, center)`` followed by
    ``ResizeImagesByLongerEdge(1536)`` and ``LTXVPreprocess(crf=18)``.  The
    latter is the exact single-frame H.264 roundtrip from
    ``comfy_extras/nodes_lt.py``.  The returned RGB image is operation-local;
    callers deliberately reuse that same object for both guide conditions.
    """

    import av

    if width <= 0 or height <= 0:
        raise ValueError("LTX I2V guide dimensions must be positive")
    source = image.convert("RGB")
    source_width, source_height = source.size
    # Comfy LoadImage/pil_to_tensor represents images as float32 [0, 1], and
    # both Lanczos nodes reconstruct PIL with multiply+clip+uint8 truncation.
    source_uint8 = np.clip(
        (np.asarray(source, dtype=np.float32) / 255.0) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    source = Image.fromarray(source_uint8, mode="RGB")
    old_aspect = source_width / source_height
    new_aspect = width / height
    crop_x = 0
    crop_y = 0
    if old_aspect > new_aspect:
        crop_x = round((source_width - source_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        crop_y = round((source_height - source_height * (old_aspect / new_aspect)) / 2)
    crop_box = (
        crop_x,
        crop_y,
        source_width - crop_x,
        source_height - crop_y,
    )
    resized = source.crop(crop_box).resize(
        (width, height),
        resample=Image.Resampling.LANCZOS,
    )
    resized_uint8 = np.clip(
        (np.asarray(resized, dtype=np.float32) / 255.0) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    resized = Image.fromarray(resized_uint8, mode="RGB")

    resized_width, resized_height = resized.size
    if resized_width > resized_height:
        longer_width = LTX23_I2V_GUIDE_LONGER_EDGE
        longer_height = int(resized_height * (LTX23_I2V_GUIDE_LONGER_EDGE / resized_width))
    else:
        longer_height = LTX23_I2V_GUIDE_LONGER_EDGE
        longer_width = int(resized_width * (LTX23_I2V_GUIDE_LONGER_EDGE / resized_height))
    longer = resized.resize(
        (longer_width, longer_height),
        resample=Image.Resampling.LANCZOS,
    )

    # Match Comfy's ``(image * 255).byte()`` input and even-dimension crop
    # before libx264.  PIL RGB already contains exactly those uint8 samples.
    image_array = np.clip(
        (np.asarray(longer, dtype=np.float32) / 255.0) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    encoded_height = (image_array.shape[0] // 2) * 2
    encoded_width = (image_array.shape[1] // 2) * 2
    image_array = np.ascontiguousarray(image_array[:encoded_height, :encoded_width])
    with BytesIO() as output_file:
        container = av.open(output_file, "w", format="mp4")
        try:
            stream = container.add_stream(
                "libx264",
                rate=1,
                options={
                    "crf": str(LTX23_I2V_GUIDE_CRF),
                    "preset": LTX23_I2V_GUIDE_PRESET,
                },
            )
            stream.height = encoded_height
            stream.width = encoded_width
            frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(
                format=LTX23_I2V_GUIDE_PIXEL_FORMAT
            )
            container.mux(stream.encode(frame))
            container.mux(stream.encode())
        finally:
            container.close()
        encoded = output_file.getvalue()

    with BytesIO(encoded) as video_file:
        container = av.open(video_file)
        try:
            stream = next(stream for stream in container.streams if stream.type == "video")
            decoded = next(container.decode(stream)).to_ndarray(format="rgb24")
        finally:
            container.close()
    decoded = np.ascontiguousarray(decoded)
    operation_image = Image.fromarray(decoded, mode="RGB")
    operation_identity = hashlib.sha256(decoded.tobytes()).hexdigest()
    proof: dict[str, Any] = {
        "policy": "pinned_comfy_i2v_guide_v1",
        "ordering": [
            "resize_dimensions_center_lanczos",
            "resize_longer_edge_1536_pil_lanczos",
            "h264_single_frame_roundtrip",
        ],
        "source_size": [source_width, source_height],
        "center_crop_box": list(crop_box),
        "resize_dimensions_size": [width, height],
        "resize_dimensions_method": "pil_lanczos_common_upscale_uint8",
        "longer_edge": LTX23_I2V_GUIDE_LONGER_EDGE,
        "longer_edge_size": [longer_width, longer_height],
        "compression_codec": "libx264",
        "compression_crf": LTX23_I2V_GUIDE_CRF,
        "compression_preset": LTX23_I2V_GUIDE_PRESET,
        "compression_pixel_format": LTX23_I2V_GUIDE_PIXEL_FORMAT,
        "operation_image_size": [operation_image.width, operation_image.height],
        "operation_image_identity_sha256": operation_identity,
        "stage_image_identities": [operation_identity, operation_identity],
        "stage_dimensions": [[width // 2, height // 2], [width, height]],
        "stage_strengths": [LTX23_GUIDE_STRENGTH, 1.0],
        "shared_operation_image": True,
        "persistent_guide_cache": False,
    }
    return operation_image, proof


def _move_module(module: nn.Module, device: torch.device | str) -> None:
    module.to(device=device)
    for nested in module.modules():
        if isinstance(nested, LTX23StoredFP8Linear):
            nested.move_stored_storage(device)


def _canonical_device(device: torch.device | str) -> torch.device:
    target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return target


def _reset_model_lora_dispatch(c: Mapping[str, Any]) -> None:
    installation = c.get("model_lora")
    if installation is not None:
        ltx23_model_lora_dispatch_evidence(c["transformer"], installation, reset=True)


def _text_execution_snapshot(model: nn.Module) -> dict[str, tuple[int, int, int, int]]:
    expected = getattr(model, "_latentslate_ltx23_gemma_quant_modules", None)
    if not isinstance(expected, Mapping) or not expected:
        raise RuntimeError("LTX 2.3 mixed text execution contract is missing")
    snapshot = {
        name: (
            int(getattr(model.get_submodule(name), "full_precision_dispatch_count", -1)),
            int(getattr(model.get_submodule(name), "native_dispatch_count", -1)),
            int(getattr(model.get_submodule(name), "rejected_dispatch_count", -1)),
            int(getattr(model.get_submodule(name), "dense_fallback_count", -1)),
        )
        for name in expected
    }
    if any(value < 0 for counters in snapshot.values() for value in counters):
        raise RuntimeError("LTX 2.3 mixed text execution counters are missing")
    return snapshot


def _verify_text_execution(
    model: nn.Module, before: Mapping[str, tuple[int, int, int, int]]
) -> dict[str, int | str]:
    after = _text_execution_snapshot(model)
    if set(after) != set(before):
        raise RuntimeError("LTX 2.3 mixed text quantized module identity changed")
    full_precision = {name: after[name][0] - before[name][0] for name in after}
    native = sum(after[name][1] - before[name][1] for name in after)
    rejected = sum(after[name][2] - before[name][2] for name in after)
    fallback = sum(after[name][3] - before[name][3] for name in after)
    if not full_precision or any(value <= 0 for value in full_precision.values()):
        missed = sorted(name for name, value in full_precision.items() if value <= 0)
        raise RuntimeError(
            f"LTX 2.3 mixed text did not use every full-precision layer: {missed[:3]}"
        )
    if native or rejected or fallback:
        raise RuntimeError("LTX 2.3 strict text execution used a forbidden quantized path")
    return {
        "backend": "engine-native/comfy-strict-full-precision-mm",
        "policy": "full_precision_mm",
        "module_count": len(full_precision),
        "total_dispatches": sum(full_precision.values()),
        "minimum_module_dispatches": min(full_precision.values()),
        "maximum_module_dispatches": max(full_precision.values()),
        "native_quantized_dispatches": native,
        "rejected_dispatches": rejected,
        "dense_fallback_dispatches": fallback,
    }


def _prompt_conditioning_cache_key(
    cache: RuntimeCache,
    request: LTX23KitchenRuntimeRequest,
    source_prompt: str,
) -> str:
    """Bind reusable conditioning to every text-semantic input except media state."""

    enhancement = ltx23_kitchen_operation_spec(request.operation).prompt_enhancement
    return cache.key(
        "prompt-conditioning-v2",
        {
            "request_fingerprint": request.fingerprint,
            "component_fingerprint": request.component_fingerprint,
            "operation": request.operation,
            "source_prompt_sha256": hashlib.sha256(source_prompt.encode()).hexdigest(),
            "enhancement": {
                "enabled": enhancement,
                "system_sha256": _prompt_system_sha256() if enhancement else None,
                "seed": LTX23_PROMPT_ENHANCEMENT_SEED if enhancement else None,
                "max_new_tokens": LTX23_PROMPT_MAX_NEW_TOKENS if enhancement else None,
                "stop_token_id": LTX23_PROMPT_STOP_TOKEN_ID if enhancement else None,
                "template": "comfy_ltx2_gemma3_manual_v1" if enhancement else None,
                "generation_settings": (LTX23_PROMPT_GENERATION_SETTINGS if enhancement else None),
            },
            "encoding": {
                "max_sequence_length": 1_024,
                "classifier_free_guidance": False,
                "dtype": "bfloat16",
                "positive_patch_state": "base",
                "negative_prompt_sha256": hashlib.sha256(
                    (
                        LTX23_FLF_NEGATIVE_PROMPT
                        if request.operation == "ltx23_distilled_flf"
                        else LTX23_DEV_NEGATIVE_PROMPT
                    ).encode()
                ).hexdigest(),
                "negative_patch_state": "base",
                "negative_used_for_cfg": False,
                "text_execution_policy": "full_precision_mm",
            },
            "text_lora_strength": ltx23_kitchen_operation_spec(
                request.operation
            ).text_lora_strength,
        },
    )


def _cached_dispatch_proof(source_proof: Mapping[str, Any]) -> dict[str, Any]:
    """Describe reused validated conditioning without claiming a fresh dispatch."""

    return {
        "provenance": "cached_prompt_conditioning",
        "dispatch_performed": False,
        "source_proof": dict(source_proof),
    }


def _validate_cached_negative_conditioning(
    cached_prompt: Mapping[str, Any], expected_prompt: str
) -> None:
    """Fail closed if Comfy's reusable negative node output is incomplete."""

    embeds = cached_prompt.get("negative_prompt_embeds")
    mask = cached_prompt.get("negative_prompt_mask")
    proof = cached_prompt.get("negative_encoding")
    if (
        not isinstance(embeds, torch.Tensor)
        or embeds.device.type != "cpu"
        or embeds.ndim != 3
        or tuple(embeds.shape) != (1, 1024, LTX23_GEMMA_PROMPT_EMBED_WIDTH)
        or embeds.dtype is not torch.bfloat16
        or not bool(torch.isfinite(embeds).all())
        or not isinstance(mask, torch.Tensor)
        or mask.device.type != "cpu"
        or tuple(mask.shape) != (1, 1024)
        or mask.dtype not in {torch.int64, torch.bool}
        or not bool(torch.logical_or(mask == 0, mask == 1).all())
        or not isinstance(proof, Mapping)
        or proof.get("prompt_sha256") != hashlib.sha256(expected_prompt.encode()).hexdigest()
        or proof.get("max_sequence_length") != 1024
        or proof.get("dtype") != "bfloat16"
        or proof.get("mask_dtype") != str(mask.dtype).removeprefix("torch.")
        or proof.get("finite") is not True
        or proof.get("encoded") is not True
        or proof.get("used_for_cfg") is not False
        or proof.get("embeds_shape") != list(embeds.shape)
        or proof.get("mask_shape") != list(mask.shape)
    ):
        raise RuntimeError("LTX cached negative text node output is invalid")


def _fp8_dispatch_snapshot(transformer: nn.Module) -> dict[str, tuple[int, int, int]]:
    return {
        name: (
            module.native_dispatch_count,
            module.rejected_dispatch_count,
            module.dense_fallback_count,
        )
        for name, module in transformer.named_modules()
        if isinstance(module, LTX23StoredFP8Linear)
    }


def _native_dispatch_proof(
    transformer: nn.Module, before: Mapping[str, tuple[int, int, int]]
) -> dict[str, Any]:
    after = _fp8_dispatch_snapshot(transformer)
    if set(after) != set(before):
        raise RuntimeError("LTX 2.3 native FP8 module topology changed during execution")
    native = sum(after[name][0] - before[name][0] for name in after)
    rejected = sum(after[name][1] - before[name][1] for name in after)
    fallback = sum(after[name][2] - before[name][2] for name in after)
    dispatched_modules = sum(after[name][0] > before[name][0] for name in after)
    if dispatched_modules != len(after) or rejected or fallback:
        raise RuntimeError(
            "LTX 2.3 native FP8 proof failed: "
            f"modules={dispatched_modules}/{len(after)}, native={native}, "
            f"rejected={rejected}, dense_fallback={fallback}"
        )
    return {
        "backend": "comfy_kitchen.tensorcore_fp8",
        "modules": len(after),
        "dispatched_modules": dispatched_modules,
        "complete": dispatched_modules == len(after),
        "native_dispatch_count": native,
        "rejected_dispatch_count": rejected,
        "dense_fallback_count": fallback,
    }


def _prompt_system_text() -> str:
    """Return the pinned Comfy LTX 2.3 T2V enhancement instruction."""

    return LTX23_T2V_SYSTEM_PROMPT.strip()


def _prompt_system_sha256() -> str:
    return hashlib.sha256(_prompt_system_text().encode()).hexdigest()


def _diffusers_sigmas(saved_sigmas: tuple[float, ...]) -> list[float]:
    """Remove the saved terminal zero that Diffusers appends internally."""

    if (
        len(saved_sigmas) < 2
        or saved_sigmas[-1] != 0.0
        or any(left <= right for left, right in pairwise(saved_sigmas))
    ):
        raise ValueError("LTX 2.3 saved sigma schedule must strictly descend to zero")
    return list(saved_sigmas[:-1])


def _release_components(c: dict[str, Any], device: torch.device) -> None:
    errors: list[str] = []
    for name, value in c.items():
        if isinstance(value, nn.Module):
            if getattr(value, "_latentslate_ltx23_residency_poisoned", None):
                continue
            try:
                if name == "text":
                    # The exact active text stage is owned and closed by the
                    # runtime before component release. Never reconstruct a
                    # fresh stage here: it would not own the live VBAR/hooks.
                    continue
                elif _module_has_meta_state(value):
                    # A partially materialized component has no safe generic
                    # ``Module.to`` transition: PyTorch cannot copy a meta
                    # tensor to CPU.  The component dictionary is being
                    # discarded immediately and this worker is not reused on a
                    # failed generation, so dropping its owner is the bounded
                    # cleanup.  This also protects any future intentional
                    # meta-only auxiliary branch without guessing its topology.
                    continue
                else:
                    _move_module(value, "cpu")
            except Exception as exc:  # noqa: BLE001 - attempt every safety transition
                errors.append(
                    f"slot={name} type={type(value).__module__}.{type(value).__qualname__} "
                    f"state={_module_release_state(value)} error={type(exc).__name__}: {exc}"
                )
    c.clear()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    if errors:
        raise RuntimeError(f"LTX 2.3 component release failed: {errors[0]}")


def _module_has_meta_state(module: nn.Module) -> bool:
    """Return whether a component contains an intentionally/unexpected meta slot.

    Release must not call ``Module.to(cpu)`` on a mixed meta/materialized tree.
    The caller owns the final reference and discards it directly instead.
    """

    return any(value.is_meta for value in (*module.parameters(), *module.buffers()))


def _module_release_state(module: nn.Module) -> str:
    """Summarize component residency without exposing tensor names or payloads."""

    counts = {"meta": 0, "cpu": 0, "cuda": 0, "other": 0}
    for value in (*module.parameters(), *module.buffers()):
        if value.is_meta:
            counts["meta"] += 1
        elif value.device.type in counts:
            counts[value.device.type] += 1
        else:
            counts["other"] += 1
    return ",".join(f"{name}={counts[name]}" for name in ("meta", "cpu", "cuda", "other"))


def _sha256_file(path: Path, check_cancelled: LTX23KitchenCancellation) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            check_cancelled()
            digest.update(chunk)
    return digest.hexdigest()
