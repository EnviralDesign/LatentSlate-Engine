from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from latentslate_engine.runtime import ltx23_kitchen as kitchen_module
from latentslate_engine.runtime.ltx23_kitchen import (
    LTX23_AUDIO_CHANNELS,
    LTX23_AUDIO_DURATION_POLICY,
    LTX23_AUDIO_MEL_HOP_LENGTH,
    LTX23_AUDIO_SAMPLE_RATE,
    LTX23_AUDIO_SOURCE_SAMPLE_RATE,
    LTX23_AUDIO_TEMPORAL_COMPRESSION_RATIO,
    LTX23_FLF_NEGATIVE_PROMPT,
    LTX23_I2V_GUIDE_CRF,
    LTX23_I2V_GUIDE_LONGER_EDGE,
    LTX23_I2V_GUIDE_PIXEL_FORMAT,
    LTX23_I2V_GUIDE_PRESET,
    LTX23_PROMPT_ENHANCEMENT_SEED,
    LTX23_PROMPT_GENERATION_SETTINGS,
    LTX23_PROMPT_MAX_NEW_TOKENS,
    LTX23_PROMPT_STOP_TOKEN_ID,
    LTX23_REFINE_SEED,
    LTX23DecodedAudio,
    LTX23KitchenGeneration,
    _audio_for_encoding,
    _ComfyLTX23LogitsProcessor,
    _decoded_audio_proof,
    _diffusers_sigmas,
    _enhance_prompt,
    _ltx23_prompt_enhancement_template,
    _LTX23TransformerResidency,
    _mux_mp4,
    _normalize_audio_duration,
    _preprocess_ltx23_i2v_guide,
    _probe_mp4,
    _prompt_system_sha256,
    _release_transformers_generation_cache,
    _stereo_audio,
    _tokenize_ltx23_prompt_enhancement,
    _uint8_frames,
    ltx23_guide_identity,
    ltx23_kitchen_operation_spec,
    validate_ltx23_kitchen_generation,
)
from latentslate_engine.runtime.ltx23_kitchen_text import (
    LTX23GemmaMixedTextStage,
    _materialize_ltx23_gemma_runtime_buffers,
)


class _MetaVisionGemmaShell(nn.Module):
    """Gemma-shaped text-only shell with intentionally unused meta vision state."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(4, 3)
        self.model.language_model.layers = nn.ModuleList((nn.Linear(3, 3),))
        self.model.vision_tower = nn.Linear(3, 3, device="meta")
        self.model.multi_modal_projector = nn.Linear(3, 3, device="meta")
        self.lm_head = nn.Linear(3, 4, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight


def _decoded_audio(
    audio_latent_frames: int, *, video_frames: int, value: float = 0.25
) -> LTX23DecodedAudio:
    mel_frames = audio_latent_frames * LTX23_AUDIO_TEMPORAL_COMPRESSION_RATIO - 3
    samples = (
        mel_frames
        * LTX23_AUDIO_MEL_HOP_LENGTH
        * LTX23_AUDIO_SAMPLE_RATE
        // LTX23_AUDIO_SOURCE_SAMPLE_RATE
    )
    return LTX23DecodedAudio(
        waveform=np.full((2, samples), value, dtype=np.float32),
        video_frames=video_frames,
        audio_latent_frames=audio_latent_frames,
        expected_audio_latent_frames=video_frames,
        audio_latent_channels=8,
        audio_latent_mel_bins=16,
        decoded_mel_frames=mel_frames,
        expected_mel_frames=mel_frames,
        decoded_mel_channels=2,
        decoded_mel_bins=64,
        decoded_samples=samples,
        expected_decoded_samples=samples,
        source_sample_rate=LTX23_AUDIO_SOURCE_SAMPLE_RATE,
        output_sample_rate=LTX23_AUDIO_SAMPLE_RATE,
        mel_hop_length=LTX23_AUDIO_MEL_HOP_LENGTH,
        temporal_compression_ratio=LTX23_AUDIO_TEMPORAL_COMPRESSION_RATIO,
        causality_axis="height",
        is_causal=True,
    )


def _decoded_audio_for_video_frames(num_frames: int) -> LTX23DecodedAudio:
    return _decoded_audio(num_frames, video_frames=num_frames)


def _audio_decoder_modules() -> tuple[SimpleNamespace, SimpleNamespace]:
    audio_vae = SimpleNamespace(
        config=SimpleNamespace(
            sample_rate=16_000,
            mel_hop_length=160,
            latent_channels=8,
            mel_bins=64,
            output_channels=2,
            causality_axis="height",
            is_causal=True,
        ),
        temporal_compression_ratio=4,
    )
    vocoder = SimpleNamespace(
        config=SimpleNamespace(
            input_sampling_rate=16_000,
            output_sampling_rate=48_000,
            out_channels=2,
            bwe_out_channels=2,
        )
    )
    return audio_vae, vocoder


class _FailingReleaseModule(nn.Module):
    """CPU-only cleanup double that identifies the exact release slot."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def to(self, *_args, **_kwargs):
        raise RuntimeError("release move failed")


class _RebuildableRotary(nn.Module):
    """Small stand-in for Gemma's nonpersistent RoPE buffer holder."""

    def __init__(self, config: object, *, meta: bool = False) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "inv_freq", torch.ones(2, device="meta" if meta else "cpu"), persistent=False
        )


class _MetaRuntimeBufferGemmaShell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(4, 3)
        self.model.language_model.layers = nn.ModuleList((nn.Linear(3, 3),))
        self.model.language_model.rotary_emb = _RebuildableRotary(object(), meta=True)
        self.model.vision_tower = nn.Linear(3, 3, device="meta")
        self.lm_head = nn.Linear(3, 4, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight


def test_failed_encode_prompt_cleanup_releases_only_gemma_language_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed prompt must not route Gemma's intentional meta branches to ``.to``."""

    runtime = object.__new__(kitchen_module.LTX23KitchenRuntime)
    runtime.request = SimpleNamespace(operation="ltx23_dev_t2v")
    runtime.device = torch.device("cpu")
    runtime._components = None
    runtime._transformer_residency = None
    text = _MetaVisionGemmaShell()
    components = {"transformer": object(), "text": text}

    class _Residency:
        def __init__(self, *_args) -> None:
            pass

        def close(self) -> None:
            pass

    def failed_encode_prompt(*_args, **_kwargs):
        raise RuntimeError("encode_prompt failed")

    monkeypatch.setattr(kitchen_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        kitchen_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(kitchen_module, "validate_ltx23_kitchen_generation", lambda *_args: None)
    monkeypatch.setattr(kitchen_module, "_LTX23TransformerResidency", _Residency)
    monkeypatch.setattr(runtime, "_materialize", lambda *_args: components)
    monkeypatch.setattr(runtime, "_execute", failed_encode_prompt)

    with pytest.raises(RuntimeError, match="encode_prompt failed"):
        runtime.generate(
            SimpleNamespace(output_path=tmp_path / "failed.mp4"),
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )

    assert runtime._components is None
    assert components == {}
    assert text.model.language_model.embed_tokens.weight.device.type == "cpu"
    assert text.model.vision_tower.weight.is_meta
    assert text.model.multi_modal_projector.weight.is_meta


def test_release_components_drops_unknown_meta_component_without_generic_cpu_move() -> None:
    """Unknown mixed-meta topology is discarded rather than sent through ``.to``."""

    component = _MetaVisionGemmaShell()
    components = {"unknown": component}

    kitchen_module._release_components(components, torch.device("cpu"))

    assert components == {}
    assert component.model.vision_tower.weight.is_meta


def test_gemma_text_stage_rejects_unmaterialized_nonpersistent_runtime_buffers() -> None:
    """RoPE buffers are not in state_dict and must be initialized before staging."""

    text = _MetaRuntimeBufferGemmaShell()

    with pytest.raises(ValueError, match="text-only materialized model"):
        LTX23GemmaMixedTextStage(text, "cpu")

    _materialize_ltx23_gemma_runtime_buffers(text)
    stage = LTX23GemmaMixedTextStage(text, "cpu")
    stage.onload()
    stage.offload()
    assert text.model.language_model.rotary_emb.inv_freq.device.type == "cpu"


def test_failed_generation_preserves_primary_error_when_release_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public failure stays at inference rather than being replaced by unload."""

    runtime = object.__new__(kitchen_module.LTX23KitchenRuntime)
    runtime.request = SimpleNamespace(operation="ltx23_dev_t2v")
    runtime.device = torch.device("cpu")
    runtime._components = None
    runtime._transformer_residency = None
    components = {"transformer": object(), "bad_slot": _FailingReleaseModule()}

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
    monkeypatch.setattr(
        runtime,
        "_execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("encode failed")),
    )

    with pytest.raises(RuntimeError, match="encode failed") as raised:
        runtime.generate(
            SimpleNamespace(output_path=tmp_path / "failed-release.mp4"),
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )

    assert components == {}
    assert any(
        "slot=bad_slot" in note and "_FailingReleaseModule" in note
        for note in raised.value.__notes__
    )


def test_generation_prefers_terminal_av_failure_counters_over_retained_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An AV backend poison must not be masked by an earlier text snapshot."""

    runtime = object.__new__(kitchen_module.LTX23KitchenRuntime)
    runtime.request = SimpleNamespace(operation="ltx23_dev_t2v")
    runtime.device = torch.device("cpu")
    runtime._components = None
    runtime._transformer_residency = None

    class _Stage:
        def diagnostics(self):
            return {"source": "text"}

        def terminal_poison_reason(self):
            return None

    class _Residency:
        def __init__(self, *_args) -> None:
            self.failure_calls = 0

        def terminal_poison_reason(self):
            return "device_quiescence_failed"

        def failure_diagnostics(self):
            self.failure_calls += 1
            return {"source": "av"}

        def close(self) -> None:
            pytest.fail("terminal residency must not be normally closed")

    stage = _Stage()
    runtime._active_text_stage = stage
    created: list[_Residency] = []

    def construct_residency(*args):
        residency = _Residency(*args)
        created.append(residency)
        return residency

    monkeypatch.setattr(kitchen_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        kitchen_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(kitchen_module, "validate_ltx23_kitchen_generation", lambda *_args: None)
    monkeypatch.setattr(kitchen_module, "_LTX23TransformerResidency", construct_residency)
    monkeypatch.setattr(runtime, "_materialize", lambda *_args: {"transformer": object()})
    monkeypatch.setattr(
        runtime,
        "_execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("AV failed")),
    )
    monkeypatch.setattr(
        kitchen_module,
        "_bounded_aimdo_failure_counters",
        lambda diagnostics: diagnostics["source"],
    )

    with pytest.raises(kitchen_module.LTX23KitchenWorkerPoisoned, match="device_quiescence_failed"):
        runtime.generate(
            SimpleNamespace(output_path=tmp_path / "terminal-av.mp4"),
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )

    assert runtime._last_failure_aimdo == "av"
    assert created[0].failure_calls == 1


def test_generation_prefers_safe_av_failure_counters_without_terminal_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = object.__new__(kitchen_module.LTX23KitchenRuntime)
    runtime.request = SimpleNamespace(operation="ltx23_dev_t2v")
    runtime.device = torch.device("cpu")
    runtime._components = None
    runtime._transformer_residency = None

    class _Stage:
        def diagnostics(self):
            return {"source": "text"}

        def terminal_poison_reason(self):
            return None

        def close(self) -> None:
            pass

    class _Residency:
        def __init__(self, *_args) -> None:
            self.failure_calls = 0

        def terminal_poison_reason(self):
            return None

        def failure_diagnostics(self):
            self.failure_calls += 1
            return {"source": "av"}

        def close(self) -> None:
            pass

    runtime._active_text_stage = _Stage()
    monkeypatch.setattr(kitchen_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        kitchen_module, "revalidate_ltx23_kitchen_runtime_request", lambda _request: True
    )
    monkeypatch.setattr(kitchen_module, "validate_ltx23_kitchen_generation", lambda *_args: None)
    created: list[_Residency] = []

    def construct_residency(*args):
        residency = _Residency(*args)
        created.append(residency)
        return residency

    monkeypatch.setattr(kitchen_module, "_LTX23TransformerResidency", construct_residency)
    monkeypatch.setattr(runtime, "_materialize", lambda *_args: {"transformer": object()})
    monkeypatch.setattr(
        runtime,
        "_execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("execution failed")),
    )
    monkeypatch.setattr(
        kitchen_module,
        "_bounded_aimdo_failure_counters",
        lambda diagnostics: diagnostics["source"],
    )

    with pytest.raises(RuntimeError, match="execution failed"):
        runtime.generate(
            SimpleNamespace(output_path=tmp_path / "text-fallback.mp4"),
            progress=lambda *_args: None,
            check_cancelled=lambda: None,
        )

    assert runtime._last_failure_aimdo == "av"
    assert created[0].failure_calls == 1


def test_exact_operation_topologies() -> None:
    t2v = ltx23_kitchen_operation_spec("ltx23_dev_t2v")
    assert t2v.prompt_enhancement is True
    assert t2v.model_lora_strength == 0.5
    assert t2v.text_lora_strength == 1.0
    assert len(t2v.main_sigmas) - 1 == 8
    assert len(t2v.refine_sigmas or ()) - 1 == 3
    assert "x2" in t2v.stages

    i2v = ltx23_kitchen_operation_spec("ltx23_dev_i2v")
    assert i2v.prompt_enhancement is True
    assert i2v.guide_strengths == (0.7, 1.0)
    assert i2v.stages[:3] == ("prompt_enhance", "text", "guide_preprocess")

    flf = ltx23_kitchen_operation_spec("ltx23_distilled_flf")
    assert flf.refine_sigmas is None
    assert flf.guide_strengths == (0.7, 0.7)
    assert flf.stages.index("guide_first") < flf.stages.index("guide_last")
    assert flf.fps == 25
    assert flf.audio_sample_rate == LTX23_AUDIO_SAMPLE_RATE
    assert flf.audio_channels == LTX23_AUDIO_CHANNELS
    assert LTX23_REFINE_SEED == 42


def test_i2v_guide_preprocess_matches_pinned_comfy_order_and_contract() -> None:
    pixels = np.zeros((80, 160, 3), dtype=np.uint8)
    pixels[:, :40, 0] = 255
    pixels[:, 40:80, 1] = 255
    pixels[:, 80:, 2] = 255

    image, proof = _preprocess_ltx23_i2v_guide(
        Image.fromarray(pixels, mode="RGB"),
        width=768,
        height=512,
    )

    assert image.mode == "RGB"
    assert image.size == (1536, 1024)
    assert proof == {
        "policy": "pinned_comfy_i2v_guide_v1",
        "ordering": [
            "resize_dimensions_center_lanczos",
            "resize_longer_edge_1536_pil_lanczos",
            "h264_single_frame_roundtrip",
        ],
        "source_size": [160, 80],
        "center_crop_box": [20, 0, 140, 80],
        "resize_dimensions_size": [768, 512],
        "resize_dimensions_method": "pil_lanczos_common_upscale_uint8",
        "longer_edge": LTX23_I2V_GUIDE_LONGER_EDGE,
        "longer_edge_size": [1536, 1024],
        "compression_codec": "libx264",
        "compression_crf": LTX23_I2V_GUIDE_CRF,
        "compression_preset": LTX23_I2V_GUIDE_PRESET,
        "compression_pixel_format": LTX23_I2V_GUIDE_PIXEL_FORMAT,
        "operation_image_size": [1536, 1024],
        "operation_image_identity_sha256": proof["operation_image_identity_sha256"],
        "stage_image_identities": [
            proof["operation_image_identity_sha256"],
            proof["operation_image_identity_sha256"],
        ],
        "stage_dimensions": [[384, 256], [768, 512]],
        "stage_strengths": [0.7, 1.0],
        "shared_operation_image": True,
        "persistent_guide_cache": False,
    }
    assert len(proof["operation_image_identity_sha256"]) == 64


def test_i2v_reuses_one_preprocessed_guide_and_transformer_residency() -> None:
    source = Path("src/latentslate_engine/runtime/ltx23_kitchen.py").read_text(encoding="utf-8")
    execute = source[
        source.index("    def _execute(") : source.index(
            "\n\ndef _release_ltx23_generation_transients"
        )
    ]

    assert execute.count("_preprocess_ltx23_i2v_guide(") == 1
    # One load belongs to FLF; I2V's one load is nested directly in its one
    # preprocessing call and does not recur at refinement.
    assert execute.count("_load_rgb(g.start_image_path, g.start_image_identity)") == 2
    assert execute.count("operation_guide,") == 3  # assignment plus both guide conditions
    assert execute.count("residency=residency") == 3
    assert "LTX2Gemma" not in execute


def test_video_vae_uses_exact_hidden_comfy_decode_defaults() -> None:
    observed: dict[str, object] = {}

    class _VAE:
        use_framewise_decoding = False

        def enable_tiling(self, **kwargs):
            observed.update(kwargs)

    vae = _VAE()
    kitchen_module._configure_ltx23_video_vae(vae)

    assert observed == {
        "tile_sample_min_height": 768,
        "tile_sample_min_width": 768,
        "tile_sample_min_num_frames": 4096,
        "tile_sample_stride_height": 704,
        "tile_sample_stride_width": 704,
        "tile_sample_stride_num_frames": 4088,
    }
    assert vae.use_framewise_decoding is True


def test_cached_text_proof_preserves_source_without_claiming_new_dispatch() -> None:
    source = {
        "backend": "comfy_kitchen/cuda/mixed-fp8-nvfp4",
        "module_count": 336,
        "total_dispatches": 61_488,
        "minimum_module_dispatches": 1,
    }

    cached = kitchen_module._cached_dispatch_proof(source)

    assert cached == {
        "provenance": "cached_prompt_conditioning",
        "dispatch_performed": False,
        "source_proof": source,
    }
    assert "total_dispatches" not in cached


def test_prompt_cache_key_binds_text_semantics_but_not_video_state() -> None:
    request = SimpleNamespace(
        fingerprint="request-a",
        component_fingerprint="components-a",
        operation="ltx23_dev_t2v",
    )
    cache = kitchen_module.RuntimeCache(
        "ltx23-test",
        enabled=True,
        max_bytes=kitchen_module.LTX23_PROMPT_CACHE_MAX_BYTES,
        max_entries=8,
        prompt_fraction=1.0,
    )

    first = kitchen_module._prompt_conditioning_cache_key(cache, request, "scene")
    same = kitchen_module._prompt_conditioning_cache_key(cache, request, "scene")
    changed_prompt = kitchen_module._prompt_conditioning_cache_key(
        cache, request, "different scene"
    )
    changed_stack = kitchen_module._prompt_conditioning_cache_key(
        cache,
        SimpleNamespace(
            fingerprint="request-b",
            component_fingerprint="components-b",
            operation="ltx23_dev_t2v",
        ),
        "scene",
    )

    assert first == same
    assert first != changed_prompt
    assert first != changed_stack
    assert cache.prompt.status()["max_entries"] == 8
    assert cache.prompt.status()["max_bytes"] == 1024 * 1024**2


def test_i2v_prompt_cache_binds_the_same_manual_enhancement_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SimpleNamespace(
        fingerprint="request-i2v",
        component_fingerprint="components-dev",
        operation="ltx23_dev_i2v",
    )
    cache = kitchen_module.RuntimeCache(
        "ltx23-i2v-enhancement-key",
        enabled=True,
        max_bytes=kitchen_module.LTX23_PROMPT_CACHE_MAX_BYTES,
        max_entries=8,
        prompt_fraction=1.0,
    )
    first = kitchen_module._prompt_conditioning_cache_key(cache, request, "scene")
    monkeypatch.setattr(kitchen_module, "LTX23_PROMPT_ENHANCEMENT_SEED", 1)

    assert kitchen_module._prompt_conditioning_cache_key(cache, request, "scene") != first


def test_prompt_cache_key_binds_exact_negative_node_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SimpleNamespace(
        fingerprint="request-a",
        component_fingerprint="components-a",
        operation="ltx23_dev_t2v",
    )
    cache = kitchen_module.RuntimeCache(
        "ltx23-negative-key",
        enabled=True,
        max_bytes=kitchen_module.LTX23_PROMPT_CACHE_MAX_BYTES,
        max_entries=8,
        prompt_fraction=1.0,
    )
    first = kitchen_module._prompt_conditioning_cache_key(cache, request, "scene")
    monkeypatch.setattr(kitchen_module, "LTX23_DEV_NEGATIVE_PROMPT", "changed negative")

    assert kitchen_module._prompt_conditioning_cache_key(cache, request, "scene") != first


def test_cached_negative_node_output_is_required_and_shape_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kitchen_module, "LTX23_GEMMA_PROMPT_EMBED_WIDTH", 8)
    embeds = torch.zeros((1, 1024, 8), dtype=torch.bfloat16)
    mask = torch.ones((1, 1024), dtype=torch.int64)
    prompt = kitchen_module.LTX23_DEV_NEGATIVE_PROMPT
    cached = {
        "negative_prompt_embeds": embeds,
        "negative_prompt_mask": mask,
        "negative_encoding": {
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "max_sequence_length": 1024,
            "dtype": "bfloat16",
            "mask_dtype": "int64",
            "finite": True,
            "encoded": True,
            "used_for_cfg": False,
            "embeds_shape": list(embeds.shape),
            "mask_shape": list(mask.shape),
        },
    }

    kitchen_module._validate_cached_negative_conditioning(cached, prompt)
    cached.pop("negative_prompt_mask")
    with pytest.raises(RuntimeError, match="negative text node output"):
        kitchen_module._validate_cached_negative_conditioning(cached, prompt)


@pytest.mark.parametrize("tamper", ("embed_dtype", "nonfinite", "mask_dtype", "mask_value"))
def test_cached_negative_node_output_rejects_tensor_tamper(
    tamper: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kitchen_module, "LTX23_GEMMA_PROMPT_EMBED_WIDTH", 2)
    embeds = torch.zeros((1, 1024, 2), dtype=torch.bfloat16)
    mask = torch.ones((1, 1024), dtype=torch.int64)
    prompt = kitchen_module.LTX23_DEV_NEGATIVE_PROMPT
    if tamper == "embed_dtype":
        embeds = embeds.float()
    elif tamper == "nonfinite":
        embeds[0, 0, 0] = torch.inf
    elif tamper == "mask_dtype":
        mask = mask.float()
    else:
        mask[0, 0] = 2
    cached = {
        "negative_prompt_embeds": embeds,
        "negative_prompt_mask": mask,
        "negative_encoding": {
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "max_sequence_length": 1024,
            "dtype": "bfloat16",
            "mask_dtype": str(mask.dtype).removeprefix("torch."),
            "finite": True,
            "encoded": True,
            "used_for_cfg": False,
            "embeds_shape": list(embeds.shape),
            "mask_shape": list(mask.shape),
        },
    }

    with pytest.raises(RuntimeError, match="negative text node output"):
        kitchen_module._validate_cached_negative_conditioning(cached, prompt)


def test_prompt_cache_budget_holds_pinned_gemma_conditioning_without_large_allocation() -> None:
    class _SizedTensor:
        def __init__(self, shape: tuple[int, ...], element_bytes: int) -> None:
            self.shape = shape
            self._element_bytes = element_bytes

        def detach(self):
            return self

        def to(self, **_kwargs):
            return self

        def contiguous(self):
            return self

        def numel(self) -> int:
            result = 1
            for dimension in self.shape:
                result *= dimension
            return result

        def element_size(self) -> int:
            return self._element_bytes

    prompt_embeds = _SizedTensor((1, 1_024, 3_840 * 49), 2)
    prompt_mask = _SizedTensor((1, 1_024), 8)
    negative_prompt_embeds = _SizedTensor((1, 1_024, 3_840 * 49), 2)
    negative_prompt_mask = _SizedTensor((1, 1_024), 8)
    tensor_bytes = (
        prompt_embeds.numel() * prompt_embeds.element_size()
        + prompt_mask.numel() * prompt_mask.element_size()
        + negative_prompt_embeds.numel() * negative_prompt_embeds.element_size()
        + negative_prompt_mask.numel() * negative_prompt_mask.element_size()
    )
    assert tensor_bytes == 770_719_744
    assert tensor_bytes > 512 * 1024**2
    assert tensor_bytes < kitchen_module.LTX23_PROMPT_CACHE_MAX_BYTES

    cache = kitchen_module.RuntimeCache(
        "ltx23-realistic-conditioning",
        enabled=True,
        max_bytes=kitchen_module.LTX23_PROMPT_CACHE_MAX_BYTES,
        max_entries=8,
        prompt_fraction=1.0,
    )
    payload = {
        "enhanced_prompt": "prompt",
        "prompt_embeds": prompt_embeds,
        "prompt_mask": prompt_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_mask": negative_prompt_mask,
        "prompt_enhancement_memory": {"policy": "test"},
        "native_text": {"backend": "test"},
        "text_lora": {"backend": "test"},
        "text_residency": {"mode": "test"},
    }

    assert cache.prompt.put("fits", payload) is True
    published = cache.prompt.status()
    assert published["entries"] == 1
    assert tensor_bytes < published["bytes"] < 1024 * 1024**2

    oversized = {
        "prompt_embeds": _SizedTensor((kitchen_module.LTX23_PROMPT_CACHE_MAX_BYTES // 2 + 1,), 2)
    }
    assert cache.prompt.put("too-large", oversized) is False
    rejected = cache.prompt.status()
    assert rejected["entries"] == 1
    assert rejected["bytes"] == published["bytes"]
    assert rejected["evictions"] == published["evictions"]


def test_cold_prompt_publication_follows_result_proof_and_transient_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    transients = {"frames": object(), "audio": object(), "stage2": object()}
    result = kitchen_module.LTX23KitchenResult(
        tmp_path / "proven.mp4",
        {
            "_prompt_cache_hit": False,
            "_prompt_cache_published": False,
            "output_sha256": "a" * 64,
            "container_format": "mp4",
        },
    )
    original_release = kitchen_module._release_ltx23_generation_transients

    def release(values):
        assert result.metadata["output_sha256"] == "a" * 64
        assert result.metadata["container_format"] == "mp4"
        events.append("release")
        original_release(values)

    class _PromptCache:
        def put(self, key, candidate):
            assert key == "conditioning"
            assert candidate == {"prompt_embeds": "live-gpu-reference"}
            assert transients == {}
            events.append("publish")
            return True

    class _Telemetry:
        def capture(self, phase):
            if phase == "after_transient_clearing":
                assert transients == {}
            events.append(phase)

        def metadata(self):
            return {"proof": "memory"}

    monkeypatch.setattr(kitchen_module, "_release_ltx23_generation_transients", release)
    finalized = kitchen_module._finalize_ltx23_kitchen_result(
        result,
        transients=transients,
        cache=SimpleNamespace(prompt=_PromptCache(), clear=lambda: None),
        cache_policy="prompt",
        prompt_cache_key="conditioning",
        prompt_cache_candidate={"prompt_embeds": "live-gpu-reference"},
        check_cancelled=lambda: events.append("cancel_check"),
        memory_telemetry=_Telemetry(),
    )

    assert finalized is result
    assert events == [
        "release",
        "after_transient_clearing",
        "cancel_check",
        "publish",
        "after_prompt_cache_publication",
        "cancel_check",
    ]
    assert result.metadata["_prompt_cache_published"] is True
    assert result.metadata["memory_telemetry"] == {"proof": "memory"}


def test_warm_prompt_hit_skips_late_publication_without_changing_provenance(
    tmp_path: Path,
) -> None:
    class _PromptCache:
        def put(self, *_args):
            raise AssertionError("warm hit must not republish prompt conditioning")

    result = kitchen_module.LTX23KitchenResult(
        tmp_path / "warm.mp4",
        {"_prompt_cache_hit": True, "_prompt_cache_published": False},
    )
    phases: list[str] = []
    telemetry = SimpleNamespace(
        capture=phases.append,
        metadata=lambda: {"proof": "warm"},
    )
    finalized = kitchen_module._finalize_ltx23_kitchen_result(
        result,
        transients={"frames": object()},
        cache=SimpleNamespace(prompt=_PromptCache(), clear=lambda: None),
        cache_policy="prompt",
        prompt_cache_key="conditioning",
        prompt_cache_candidate=None,
        check_cancelled=lambda: None,
        memory_telemetry=telemetry,
    )

    assert finalized.metadata == {
        "_prompt_cache_hit": True,
        "_prompt_cache_published": False,
        "memory_telemetry": {"proof": "warm"},
    }
    assert phases == ["after_transient_clearing", "after_prompt_cache_publication"]


def test_cache_none_still_captures_both_final_memory_phases(tmp_path: Path) -> None:
    class _PromptCache:
        def put(self, *_args):
            raise AssertionError("cache none must not publish prompt conditioning")

    phases: list[str] = []
    result = kitchen_module.LTX23KitchenResult(
        tmp_path / "none.mp4",
        {"_prompt_cache_hit": False, "_prompt_cache_published": False},
    )
    kitchen_module._finalize_ltx23_kitchen_result(
        result,
        transients={"frames": object()},
        cache=SimpleNamespace(prompt=_PromptCache(), clear=lambda: None),
        cache_policy="none",
        prompt_cache_key="conditioning",
        prompt_cache_candidate=None,
        check_cancelled=lambda: None,
        memory_telemetry=SimpleNamespace(
            capture=phases.append,
            metadata=lambda: {"proof": "none"},
        ),
    )

    assert phases == ["after_transient_clearing", "after_prompt_cache_publication"]
    assert result.metadata["memory_telemetry"] == {"proof": "none"}


def test_ltx_memory_telemetry_phase_lists_are_operation_specific() -> None:
    assert kitchen_module._ltx23_memory_telemetry_phases("ltx23_dev_t2v") == (
        "after_text_offload",
        "after_stage1",
        "after_latent_upscaling",
        "after_stage2",
        "after_decode",
        "after_transient_clearing",
        "after_prompt_cache_publication",
    )
    assert kitchen_module._ltx23_memory_telemetry_phases("ltx23_dev_i2v") == (
        "after_text_offload",
        "after_stage1",
        "after_latent_upscaling",
        "after_stage2",
        "after_decode",
        "after_transient_clearing",
        "after_prompt_cache_publication",
    )
    assert kitchen_module._ltx23_memory_telemetry_phases("ltx23_distilled_flf") == (
        "after_text_offload",
        "after_main_denoise",
        "after_decode",
        "after_transient_clearing",
        "after_prompt_cache_publication",
    )


def test_poisoned_runtime_unload_retains_complete_component_graph() -> None:
    runtime = object.__new__(kitchen_module.LTX23KitchenRuntime)
    components = {"text": object(), "transformer": object()}

    class _Stage:
        def terminal_poison_reason(self) -> str:
            return "device_quiescence_failed"

        def offload(self) -> None:
            raise AssertionError("poisoned runtime must not normally offload")

    stage = _Stage()
    residency = object()
    runtime._components = components
    runtime._active_text_stage = stage
    runtime._transformer_residency = residency

    with pytest.raises(
        kitchen_module.LTX23KitchenWorkerPoisoned,
        match="device_quiescence_failed",
    ):
        runtime.unload()

    assert runtime._components is components
    assert runtime._active_text_stage is stage
    assert runtime._transformer_residency is residency


def test_successful_runtime_unload_closes_av_source_once_and_is_idempotent() -> None:
    events: list[str] = []

    class _SourceOwner:
        def close(self) -> None:
            events.append("source_close")

    residency = object.__new__(kitchen_module._LTX23TransformerResidency)
    residency.transformer = nn.Linear(1, 1, bias=False)
    residency.device = torch.device("cpu")
    residency._handles = []
    residency._closed = False
    residency._owner_thread = None
    residency._executing = False
    residency._barrier_failed = False
    residency._streamed_binding = None
    residency._resident = {}
    residency._root_binding = None
    residency._dynamic = None
    residency._base_file_handle = _SourceOwner()
    residency._base_file_handle_opened = 1
    residency._base_file_handle_closed = 0

    runtime = object.__new__(kitchen_module.LTX23KitchenRuntime)
    runtime.device = torch.device("cpu")
    runtime._components = {}
    runtime._active_text_stage = None
    runtime._transformer_residency = residency
    runtime._cache = SimpleNamespace(clear=lambda: events.append("cache_clear"))

    runtime.unload()
    runtime.unload()

    assert events.count("source_close") == 1
    assert runtime._components is None
    assert runtime._active_text_stage is None
    assert runtime._transformer_residency is None
    assert residency._base_file_handle is None
    assert residency._base_file_handle_opened == residency._base_file_handle_closed == 1


def test_ltx23_acceptance_env_requires_aimdo_without_recipe_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = "LATENTSLATE_LTX23_REQUIRE_AIMDO"
    monkeypatch.delenv(variable, raising=False)
    assert kitchen_module._ltx23_text_dynamic_policy() == "auto"

    monkeypatch.setenv(variable, "0")
    assert kitchen_module._ltx23_text_dynamic_policy() == "auto"
    monkeypatch.setenv(variable, "1")
    assert kitchen_module._ltx23_text_dynamic_policy() == "required"

    monkeypatch.setenv(variable, "yes")
    with pytest.raises(RuntimeError, match="must be unset, 0, or 1"):
        kitchen_module._ltx23_text_dynamic_policy()


@pytest.mark.parametrize(
    ("failure", "message"),
    (("publication", "did not fit"), ("post_put_cancel", "cancelled")),
)
def test_post_mux_prompt_finalization_failure_removes_output_and_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    request = SimpleNamespace(
        operation="ltx23_dev_t2v", fingerprint="request", component_fingerprint="components"
    )
    runtime = kitchen_module.LTX23KitchenRuntime(request, device="cuda", cache_policy="prompt")
    components = {"transformer": object()}
    finalizer_checks = 0
    in_finalizer = False

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
    monkeypatch.setattr(kitchen_module, "_release_components", lambda value, _device: value.clear())
    if failure == "publication":
        monkeypatch.setattr(runtime._cache.prompt, "put", lambda *_args: False)

    def check_cancelled() -> None:
        nonlocal finalizer_checks
        if in_finalizer:
            finalizer_checks += 1
            if failure == "post_put_cancel" and finalizer_checks == 2:
                raise RuntimeError("cancelled after prompt publication")

    def execute(_components, generation, **_kwargs):
        nonlocal in_finalizer
        output = Path(generation.output_path)
        output.write_bytes(b"proven-mp4")
        result = kitchen_module.LTX23KitchenResult(
            output,
            {
                "_prompt_cache_hit": False,
                "_prompt_cache_published": False,
                "output_sha256": "a" * 64,
            },
        )
        in_finalizer = True
        return kitchen_module._finalize_ltx23_kitchen_result(
            result,
            transients={"frames": object(), "video_latents": object()},
            cache=runtime._cache,
            cache_policy="prompt",
            prompt_cache_key="conditioning",
            prompt_cache_candidate={"prompt_embeds": "small-test-conditioning"},
            check_cancelled=check_cancelled,
        )

    monkeypatch.setattr(runtime, "_execute", execute)
    output = tmp_path / f"{failure}.mp4"
    generation = SimpleNamespace(output_path=output)

    with pytest.raises(RuntimeError, match=message):
        runtime.generate(
            generation,
            progress=lambda *_args: None,
            check_cancelled=check_cancelled,
        )

    assert not output.exists()
    assert runtime._cache.prompt.status()["entries"] == 0
    assert runtime._components is None


def test_saved_sigma_contract_has_exact_diffusers_step_count() -> None:
    from diffusers import FlowMatchEulerDiscreteScheduler

    schedules = (
        (ltx23_kitchen_operation_spec("ltx23_dev_t2v").main_sigmas, 8),
        (ltx23_kitchen_operation_spec("ltx23_dev_t2v").refine_sigmas, 3),
    )
    for saved, expected_steps in schedules:
        assert saved is not None
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(sigmas=_diffusers_sigmas(saved))
        assert len(scheduler.timesteps) == expected_steps
        assert len(scheduler.sigmas) == expected_steps + 1
        assert scheduler.sigmas[-1].item() == 0.0
        assert scheduler.sigmas[-2].item() > 0.0


def test_prompt_enhancement_uses_pinned_first_party_contract() -> None:
    assert _prompt_system_sha256() == (
        "f00b22f47dad68358f5c2c7396c701db95095cf26dc3dbd6b5556eab04692071"
    )
    assert len(kitchen_module._prompt_system_text()) == 4_174
    assert len(kitchen_module._prompt_system_text().splitlines()) == 39
    assert LTX23_PROMPT_ENHANCEMENT_SEED == 0
    assert LTX23_PROMPT_MAX_NEW_TOKENS == 2_048
    assert LTX23_PROMPT_STOP_TOKEN_ID == 106
    assert LTX23_PROMPT_GENERATION_SETTINGS == {
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 64,
        "top_p": 0.95,
        "min_p": 0.05,
        "repetition_penalty": 1.05,
    }
    assert hashlib.sha256(LTX23_FLF_NEGATIVE_PROMPT.encode()).hexdigest() == (
        "89b4453c73ab7a46c5f6ab4f3466cb68f3d2c245df9ae497c2e5b3c09056e435"
    )


def test_prompt_enhancement_releases_persistent_cache_before_return(monkeypatch) -> None:
    from transformers.cache_utils import Cache

    events: list[str] = []

    class ResetLayer:
        def reset(self) -> None:
            events.append("cache_reset")

    class FakeModel:
        def generate(self, **inputs):
            events.append("generate")
            assert inputs["eos_token_id"] == 106
            assert inputs["do_sample"] is False
            assert inputs["repetition_penalty"] == 1.0
            assert len(inputs["logits_processor"]) == 1
            self._cache = Cache(layers=[ResetLayer()])
            return torch.cat((inputs["input_ids"], torch.tensor([[13]])), dim=1)

    class FakeTokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return "template"

        def batch_decode(self, values, **_kwargs):
            assert [value.tolist() for value in values] == [[13]]
            events.append("decode")
            return ["enhanced prompt"]

    class FakeInputs(dict):
        input_ids = torch.tensor([[2, 105, 11, 12]])

        def to(self, _device):
            return self

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def __call__(self, **kwargs):
            assert kwargs["text"] == _ltx23_prompt_enhancement_template("prompt")
            return FakeInputs(input_ids=FakeInputs.input_ids)

    processor = FakeProcessor()
    model = FakeModel()

    enhanced, memory = _enhance_prompt(
        processor, model, "prompt", 0, torch.device("cpu"), lambda: None
    )

    assert enhanced == "enhanced prompt"
    assert events == ["generate", "cache_reset", "decode"]
    assert not hasattr(model, "_cache")
    assert memory == {
        "policy": "release_after_prompt_enhancement",
        "cache_present": True,
        "cache_type": "transformers.cache_utils.Cache",
        "cuda_allocated_before_bytes": None,
        "cuda_allocated_after_bytes": None,
        "cuda_allocated_released_bytes": None,
        "template": "comfy_ltx2_gemma3_manual_v1",
        "stop_token_id": 106,
        "generation_settings": LTX23_PROMPT_GENERATION_SETTINGS,
        "decoded_suffix_nonempty": True,
        "think_block_removed": False,
        "fallback_to_source_prompt": False,
    }


def test_prompt_enhancement_tokenizer_matches_comfy_segment_and_padding_contract() -> None:
    class Inputs:
        input_ids = torch.tensor([[2, 105, 11, 12]])

    class Processor:
        def __call__(self, **_kwargs):
            return Inputs()

    inputs = _tokenize_ltx23_prompt_enhancement(Processor(), "template", torch.device("cpu"))

    assert set(inputs) == {"input_ids", "attention_mask"}
    assert inputs["input_ids"].shape == (1, 1_024)
    assert inputs["input_ids"][0, :1_021].count_nonzero().item() == 0
    assert inputs["input_ids"][0, -3:].tolist() == [2, 11, 12]
    assert inputs["attention_mask"].sum().item() == 1_024


def test_comfy_prompt_sampler_penalizes_generated_history_not_prompt(monkeypatch) -> None:
    def choose_argmax(probabilities, num_samples, generator):
        assert num_samples == 1
        assert generator is not None
        return probabilities.argmax(dim=-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", choose_argmax)
    scores = torch.tensor([[0.0, 10.0, 9.6]], dtype=torch.float32)
    prompt_only = torch.tensor([[1, 7, 8]])
    with_generated_repeat = torch.tensor([[1, 7, 8, 1]])

    without_history = _ComfyLTX23LogitsProcessor(
        prompt_length=3,
        device=torch.device("cpu"),
        execution_dtype=torch.bfloat16,
        seed=0,
    )
    with_history = _ComfyLTX23LogitsProcessor(
        prompt_length=3,
        device=torch.device("cpu"),
        execution_dtype=torch.bfloat16,
        seed=0,
    )

    assert without_history(prompt_only, scores).argmax(dim=-1).item() == 1
    assert with_history(with_generated_repeat, scores).argmax(dim=-1).item() == 2


def test_prompt_enhancement_manual_template_think_strip_and_empty_fallback() -> None:
    template = _ltx23_prompt_enhancement_template("a bird")
    assert template.startswith(
        "<start_of_turn>system\n" + kitchen_module._prompt_system_text().strip()
    )
    assert (
        "<end_of_turn>\n<start_of_turn>user\n\n"
        "User Raw Input Prompt: a bird.<end_of_turn>\n<start_of_turn>model\n"
    ) in template

    class Inputs(dict):
        input_ids = torch.tensor([[2, 105, 1, 2]])

        def to(self, _device):
            return self

    class Tokenizer:
        output = ""

        def batch_decode(self, _values, **_kwargs):
            return [self.output]

    class Processor:
        tokenizer = Tokenizer()

        def __call__(self, **_kwargs):
            return Inputs(input_ids=Inputs.input_ids)

    class Model:
        def generate(self, **kwargs):
            return torch.cat((kwargs["input_ids"], torch.tensor([[3]])), dim=1)

    processor = Processor()
    processor.tokenizer.output = "<think>reasoning</think> enhanced scene"
    enhanced, proof = _enhance_prompt(
        processor, Model(), "original", 0, torch.device("cpu"), lambda: None
    )
    assert enhanced == "enhanced scene"
    assert proof["think_block_removed"] is True
    assert proof["fallback_to_source_prompt"] is False

    processor.tokenizer.output = "<think>unfinished"
    fallback, proof = _enhance_prompt(
        processor, Model(), "original", 0, torch.device("cpu"), lambda: None
    )
    assert fallback == "original"
    assert proof["decoded_suffix_nonempty"] is True
    assert proof["think_block_removed"] is True
    assert proof["fallback_to_source_prompt"] is True

    processor.tokenizer.output = ""
    fallback, proof = _enhance_prompt(
        processor, Model(), "original", 0, torch.device("cpu"), lambda: None
    )
    assert fallback == "original"
    assert proof["decoded_suffix_nonempty"] is False
    assert proof["think_block_removed"] is False
    assert proof["fallback_to_source_prompt"] is True


def test_prompt_cache_release_rejects_unknown_owner() -> None:
    model = SimpleNamespace(_cache=object())

    with pytest.raises(TypeError, match="unsupported cache owner"):
        _release_transformers_generation_cache(model, torch.device("cpu"))

    assert hasattr(model, "_cache")


def test_t2v_prompt_enhancement_does_not_inherit_the_video_seed() -> None:
    source = Path("src/latentslate_engine/runtime/ltx23_kitchen.py").read_text(encoding="utf-8")
    call = source[
        source.index("prompt, prompt_enhancement_memory = _enhance_prompt(") : source.index(
            ")\n", source.index("prompt, prompt_enhancement_memory = _enhance_prompt(")
        )
    ]
    assert "LTX23_PROMPT_ENHANCEMENT_SEED" in call
    assert "g.seed" not in call


def test_text_failure_cleanup_restores_patch_state_before_warm_offload() -> None:
    source = Path("src/latentslate_engine/runtime/ltx23_kitchen.py").read_text(encoding="utf-8")
    cleanup = source[
        source.index("            finally:\n", source.index("primary_text_error")) : source.index(
            "            text_residency = text_stage.diagnostics()"
        )
    ]

    strength_reset = cleanup.index("text_lora.set_strength(0.0)")
    active_guard = cleanup.index("if text_patch_state_lora_active:")
    base_invalidation = cleanup.index("text_stage.invalidate_patch_state(to_base=True)")
    offload = cleanup.index("text_stage.offload()")
    assert strength_reset < active_guard < base_invalidation < offload


def test_generation_contract_enforces_guides_and_two_stage_geometry(tmp_path: Path) -> None:
    guide = tmp_path / "guide.png"
    Image.new("RGB", (64, 64), "red").save(guide)
    valid = LTX23KitchenGeneration("prompt", tmp_path / "out.mp4", 768, 512, 121, 7)
    validate_ltx23_kitchen_generation("ltx23_dev_t2v", valid)

    with pytest.raises(ValueError, match="divisible by 64"):
        validate_ltx23_kitchen_generation(
            "ltx23_dev_t2v",
            LTX23KitchenGeneration("prompt", tmp_path / "bad.mp4", 736, 512, 121, 7),
        )
    with pytest.raises(ValueError, match="endpoint-image"):
        validate_ltx23_kitchen_generation(
            "ltx23_distilled_flf",
            LTX23KitchenGeneration("prompt", tmp_path / "bad.mp4", 768, 512, 121, 7, guide),
        )
    validate_ltx23_kitchen_generation(
        "ltx23_dev_i2v",
        LTX23KitchenGeneration(
            "prompt",
            tmp_path / "i2v.mp4",
            768,
            512,
            121,
            7,
            guide,
            None,
            ltx23_guide_identity(guide),
        ),
    )


def test_output_normalization_rejects_wrong_media_contract() -> None:
    frames = _uint8_frames(np.full((2, 4, 6, 3), 0.5, dtype=np.float32))
    assert frames.dtype == np.uint8
    assert frames.shape == (2, 4, 6, 3)
    audio = _stereo_audio(np.zeros((64, 2), dtype=np.float32))
    assert audio.shape == (2, 64)
    with pytest.raises(ValueError, match="FHWC RGB"):
        _uint8_frames(np.zeros((2, 4, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="exactly two"):
        _stereo_audio(np.zeros((1, 64), dtype=np.float32))


def test_decoder_audio_proof_closes_exact_source_configuration_and_counts() -> None:
    audio_vae, vocoder = _audio_decoder_modules()
    proof = _decoded_audio_proof(
        torch.zeros((1, 8, 25, 16)),
        torch.zeros((1, 2, 97, 64)),
        torch.zeros((2, 46_560)),
        audio_vae,
        vocoder,
        video_frames=25,
    )
    assert proof.audio_latent_frames == 25
    assert proof.decoded_mel_frames == proof.expected_mel_frames == 97
    assert proof.decoded_samples == proof.expected_decoded_samples == 46_560


@pytest.mark.parametrize("audio_latent_frames", [24, 26])
def test_decoder_audio_proof_rejects_coherent_wrong_audio_lattice(
    audio_latent_frames: int,
) -> None:
    audio_vae, vocoder = _audio_decoder_modules()
    mel_frames = audio_latent_frames * 4 - 3
    with pytest.raises(ValueError, match="decoded video grid"):
        _decoded_audio_proof(
            torch.zeros((1, 8, audio_latent_frames, 16)),
            torch.zeros((1, 2, mel_frames, 64)),
            torch.zeros((2, mel_frames * 480)),
            audio_vae,
            vocoder,
            video_frames=25,
        )


@pytest.mark.parametrize(
    "latent_shape,mel_shape",
    [
        ((2, 8, 25, 16), (1, 2, 97, 64)),
        ((1, 7, 25, 16), (1, 2, 97, 64)),
        ((1, 8, 25, 15), (1, 2, 97, 64)),
        ((1, 8, 25, 16), (2, 2, 97, 64)),
        ((1, 8, 25, 16), (1, 1, 97, 64)),
        ((1, 8, 25, 16), (1, 2, 97, 63)),
    ],
)
def test_decoder_audio_proof_rejects_noncanonical_tensor_shapes(
    latent_shape: tuple[int, ...], mel_shape: tuple[int, ...]
) -> None:
    audio_vae, vocoder = _audio_decoder_modules()
    with pytest.raises(ValueError, match="layout"):
        _decoded_audio_proof(
            torch.zeros(latent_shape),
            torch.zeros(mel_shape),
            torch.zeros((2, 46_560)),
            audio_vae,
            vocoder,
            video_frames=25,
        )


@pytest.mark.parametrize("mel_delta,sample_delta", [(1, 0), (-1, 0), (0, 1), (0, -1)])
def test_decoder_audio_proof_rejects_off_by_one_counts(mel_delta: int, sample_delta: int) -> None:
    audio_vae, vocoder = _audio_decoder_modules()
    with pytest.raises(ValueError, match="source (latent|mel) grid"):
        _decoded_audio_proof(
            torch.zeros((1, 8, 25, 16)),
            torch.zeros((1, 2, 97 + mel_delta, 64)),
            torch.zeros((2, 46_560 + sample_delta)),
            audio_vae,
            vocoder,
            video_frames=25,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", 16_001),
        ("vocoder_source", 16_001),
        ("hop", 161),
        ("compression", 5),
        ("output", 47_999),
    ],
)
def test_decoder_audio_proof_rejects_wrong_rate_or_grid_config(field: str, value: int) -> None:
    settings = {
        "source": 16_000,
        "vocoder_source": 16_000,
        "hop": 160,
        "compression": 4,
        "output": 48_000,
    }
    settings[field] = value
    audio_vae, vocoder = _audio_decoder_modules()
    audio_vae.config.sample_rate = settings["source"]
    audio_vae.config.mel_hop_length = settings["hop"]
    audio_vae.temporal_compression_ratio = settings["compression"]
    vocoder.config.input_sampling_rate = settings["vocoder_source"]
    vocoder.config.output_sampling_rate = settings["output"]
    with pytest.raises(ValueError, match="pinned contract"):
        _decoded_audio_proof(
            torch.zeros((1, 8, 25, 16)),
            torch.zeros((1, 2, 97, 64)),
            torch.zeros((2, 46_560)),
            audio_vae,
            vocoder,
            video_frames=25,
        )


@pytest.mark.parametrize(
    "owner,field,value",
    [
        ("vae", "latent_channels", 7),
        ("vae", "mel_bins", 63),
        ("vae", "output_channels", 1),
        ("vae", "causality_axis", "width"),
        ("vae", "is_causal", False),
        ("vocoder", "out_channels", 1),
        ("vocoder", "bwe_out_channels", 1),
    ],
)
def test_decoder_audio_proof_rejects_wrong_shape_or_causal_config(
    owner: str, field: str, value: object
) -> None:
    audio_vae, vocoder = _audio_decoder_modules()
    target = audio_vae.config if owner == "vae" else vocoder.config
    setattr(target, field, value)
    with pytest.raises(ValueError, match="pinned contract"):
        _decoded_audio_proof(
            torch.zeros((1, 8, 25, 16)),
            torch.zeros((1, 2, 97, 64)),
            torch.zeros((2, 46_560)),
            audio_vae,
            vocoder,
            video_frames=25,
        )


@pytest.mark.parametrize(
    "waveform",
    [
        np.zeros((1, 2, 46_560), dtype=np.float32),
        np.zeros((46_560, 2), dtype=np.float32),
        np.zeros((1, 46_560), dtype=np.float32),
        np.full((2, 46_560), np.nan, dtype=np.float32),
        np.full((2, 46_560), np.inf, dtype=np.float32),
    ],
)
def test_decoder_audio_proof_rejects_wrong_layout_channels_and_nonfinite(
    waveform: np.ndarray,
) -> None:
    audio_vae, vocoder = _audio_decoder_modules()
    with pytest.raises(ValueError, match="layout|finite"):
        _decoded_audio_proof(
            torch.zeros((1, 8, 25, 16)),
            torch.zeros((1, 2, 97, 64)),
            waveform,
            audio_vae,
            vocoder,
            video_frames=25,
        )


def test_audio_duration_normalization_exact_pad_trim_and_malformed() -> None:
    proof = _decoded_audio(4, video_frames=4)
    exact_target = proof.decoded_samples
    normalized, exact_metadata = _normalize_audio_duration(proof, exact_target)
    assert np.array_equal(normalized, proof.waveform)
    assert exact_metadata["policy"] == LTX23_AUDIO_DURATION_POLICY
    assert exact_metadata["reason"] == "independent_audio_grid_causal_tail"
    assert exact_metadata["trimmed_samples"] == exact_metadata["padded_samples"] == 0
    assert exact_metadata["decoded_samples"] == exact_metadata["expected_decoded_samples"]

    pad_target = exact_target + 2_160
    normalized, pad_metadata = _normalize_audio_duration(proof, pad_target)
    assert normalized.shape == (2, pad_target)
    assert np.array_equal(normalized[:, :exact_target], proof.waveform)
    assert np.count_nonzero(normalized[:, exact_target:]) == 0
    assert pad_metadata["padded_samples"] == 2_160
    assert pad_metadata["trimmed_samples"] == 0

    trim_target = exact_target - 17
    normalized, trim_metadata = _normalize_audio_duration(proof, trim_target)
    assert np.array_equal(normalized, proof.waveform[:, :trim_target])
    assert trim_metadata["trimmed_samples"] == 17
    assert trim_metadata["padded_samples"] == 0

    for malformed in (
        replace(proof, waveform=proof.waveform[:, :-1]),
        replace(proof, waveform=np.pad(proof.waveform, ((0, 0), (0, 1)))),
    ):
        with pytest.raises(ValueError, match="finite stereo waveform"):
            _normalize_audio_duration(malformed, exact_target)


def test_audio_encoding_clips_positive_and_negative_excursions_once() -> None:
    source = np.array([[1.5, -1.5, 0.25], [-2.0, 2.0, -0.25]], dtype=np.float32)
    encoded = _audio_for_encoding(source)
    assert np.array_equal(
        encoded,
        np.array([[1.0, -1.0, 0.25], [-1.0, 1.0, -0.25]], dtype=np.float32),
    )
    assert np.array_equal(source, np.array([[1.5, -1.5, 0.25], [-2.0, 2.0, -0.25]]))


def test_audio_duration_arithmetic_exhausts_every_legal_frame_count_and_residue() -> None:
    for num_frames in range(25, 250, 8):
        proof = _decoded_audio_for_video_frames(num_frames)
        target = num_frames * 48_000 // 25
        normalized, metadata = _normalize_audio_duration(proof, target)
        assert normalized.shape == (2, target)
        assert metadata["padded_samples"] == 1_440
        assert metadata["trimmed_samples"] == 0
        assert metadata["decoded_samples"] + metadata["padded_samples"] == target


@pytest.mark.parametrize("num_frames", [25, 33, 41, 121, 129, 249])
def test_audio_duration_mandatory_frame_cases(num_frames: int) -> None:
    proof = _decoded_audio_for_video_frames(num_frames)
    _normalized, metadata = _normalize_audio_duration(proof, num_frames * 48_000 // 25)
    assert metadata["padded_samples"] == 1_440


def test_mux_publishes_25fps_48khz_stereo(tmp_path: Path) -> None:
    import av

    output = tmp_path / "native.mp4"
    frames = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    audio = _decoded_audio_for_video_frames(3)
    checks = 0

    def check_cancelled() -> None:
        nonlocal checks
        checks += 1

    normalization = _mux_mp4(frames, audio, output, check_cancelled=check_cancelled)
    assert output.stat().st_size > 0
    assert checks >= 3 + 6
    observed = _probe_mp4(output, check_cancelled)
    assert observed["container_format"].split(",").count("mp4") == 1
    assert observed["video_codec"] == "h264"
    assert observed["audio_codec"] == "aac"
    assert observed["width"] == observed["height"] == 32
    assert observed["num_frames"] == 3
    assert observed["fps"] == 25
    assert observed["audio_sample_rate"] == 48_000
    assert observed["audio_channels"] == 2
    assert normalization["target_samples"] == 5_760
    assert normalization["video_frames"] == 3
    assert normalization["fps"] == 25
    assert normalization["audio_channels"] == 2
    assert normalization["trimmed_samples"] == 0
    assert normalization["padded_samples"] == 1_440
    with av.open(str(output)) as container:
        video = container.streams.video[0]
        sound = container.streams.audio[0]
        assert video.average_rate == 25
        assert sound.sample_rate == 48_000
        assert sound.layout.name == "stereo"


def test_mux_pads_one_ltx_audio_decoder_tail_with_silence(tmp_path: Path) -> None:
    output = tmp_path / "padded-tail.mp4"
    frames = np.zeros((25, 32, 32, 3), dtype=np.uint8)
    decoded_samples = 46_560
    normalization = _mux_mp4(
        frames,
        _decoded_audio_for_video_frames(25),
        output,
        check_cancelled=lambda: None,
    )
    assert normalization == {
        "policy": "source_derived_exact_duration_v1",
        "reason": "independent_audio_grid_causal_tail",
        "video_frames": 25,
        "audio_latent_frames": 25,
        "expected_audio_latent_frames": 25,
        "audio_latent_channels": 8,
        "audio_latent_mel_bins": 16,
        "decoded_mel_frames": 97,
        "expected_mel_frames": 97,
        "decoded_mel_channels": 2,
        "decoded_mel_bins": 64,
        "decoded_samples": decoded_samples,
        "expected_decoded_samples": decoded_samples,
        "target_samples": 48_000,
        "fps": 25,
        "audio_channels": 2,
        "source_sample_rate": 16_000,
        "output_sample_rate": 48_000,
        "mel_hop_length": 160,
        "temporal_compression_ratio": 4,
        "causality_axis": "height",
        "is_causal": True,
        "trimmed_samples": 0,
        "padded_samples": 1_440,
    }
    observed = _probe_mp4(output, check_cancelled=lambda: None)
    assert observed["num_frames"] == 25
    assert observed["audio_samples"] >= normalization["target_samples"]
    assert observed["audio_samples"] - normalization["target_samples"] < 1024


def test_mux_is_atomic_on_cancellation(tmp_path: Path) -> None:
    output = tmp_path / "cancel.mp4"
    calls = 0

    def cancel() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        _mux_mp4(
            np.zeros((3, 32, 32, 3), dtype=np.uint8),
            _decoded_audio_for_video_frames(3),
            output,
            check_cancelled=cancel,
        )
    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp.mp4"))


def test_transformer_residency_is_family_local_and_removes_every_hook() -> None:
    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.root = nn.Linear(2, 2)
            self.transformer_blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(48)])

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            value = self.root(value)
            for block in self.transformer_blocks:
                value = block(value)
            return value

    model = TinyTransformer()
    transitions = 0

    def before_first() -> None:
        nonlocal transitions
        transitions += 1

    manager = _LTX23TransformerResidency(model, torch.device("cpu"))
    with manager.forward_scope(before_first):
        assert model(torch.ones(1, 2)).shape == (1, 2)
        assert model(torch.ones(1, 2)).shape == (1, 2)
    assert transitions == 1
    assert manager.handles
    manager.close()
    assert not manager.handles
    assert manager.active is None
    assert all(not block._forward_hooks for block in model.transformer_blocks)
    assert all(not block._forward_pre_hooks for block in model.transformer_blocks)


def test_transformer_residency_uses_leaf_allocations_and_warm_block_prefetch() -> None:
    class WeightedBlock(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.arange(size, dtype=torch.float32))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + self.weight[0] * 0

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.root = nn.Parameter(torch.ones(3))
            self.transformer_blocks = nn.ModuleList(
                [WeightedBlock(5_000 + index) for index in range(48)]
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            for block in self.transformer_blocks:
                value = block(value)
            return value

    model = TinyTransformer()
    originals = {name: parameter for name, parameter in model.named_parameters()}
    manager = _LTX23TransformerResidency(
        model,
        torch.device("cpu"),
    )

    for _job in range(2):
        with manager.forward_scope(lambda: None):
            assert torch.equal(model(torch.ones(1)), torch.ones(1))

    policy = manager.policy
    assert policy["resident_block_count"] == 0
    assert policy["leaf_allocation_count"] == 49
    assert policy["force_resident_leaf_count"] == 1
    assert policy["prefetch"] is True
    assert policy["prefetch_groups"] == 48 * 2
    assert policy["prefetch_leaves"] == 48 * 2
    manager.close()
    assert all(dict(model.named_parameters())[name] is value for name, value in originals.items())


def test_transformer_policy_counts_cross_group_alias_physical_bytes_once() -> None:
    shared = nn.Parameter(torch.ones(8), requires_grad=False)

    class Block(nn.Module):
        def __init__(self, *, alias: bool = False) -> None:
            super().__init__()
            self.weight = shared if alias else nn.Parameter(torch.ones(8), requires_grad=False)

        def forward(self, value):
            return value + self.weight[0] * 0

    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.root = shared
            self.transformer_blocks = nn.ModuleList(
                [Block(alias=index == 0) for index in range(48)]
            )

        def forward(self, value):
            for block in self.transformer_blocks:
                value = block(value)
            return value

    manager = _LTX23TransformerResidency(Transformer(), torch.device("cpu"))
    policy = manager.policy

    assert policy["root_bytes"] + policy["streamed_block_bytes"] == policy["stored_bytes"]
    assert policy["resident_block_bytes"] == 0
    assert manager.leaf_storage[0].schedule_groups == (
        "root",
        "transformer_blocks.0",
    )
    manager.close()


def test_transformer_leaf_activation_failure_rolls_back_to_cpu_originals(monkeypatch) -> None:
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(5_000))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + self.weight[0] * 0

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.root = nn.Parameter(torch.ones(2))
            self.transformer_blocks = nn.ModuleList([Block() for _ in range(48)])

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            for block in self.transformer_blocks:
                value = block(value)
            return value

    model = TinyTransformer()
    originals = {name: parameter for name, parameter in model.named_parameters()}
    manager = _LTX23TransformerResidency(model, torch.device("cpu"))
    original_activate = manager._activate_leaf

    def fail_selected(descriptor, values):
        if descriptor.path == "transformer_blocks.1":
            raise RuntimeError("synthetic leaf activation failure")
        return original_activate(descriptor, values)

    manager._scheduler._activate = fail_selected

    with (
        pytest.raises(RuntimeError, match="synthetic leaf activation failure"),
        manager.forward_scope(lambda: None),
    ):
        model(torch.ones(1, 2))

    assert all(dict(model.named_parameters())[name] is value for name, value in originals.items())


def test_transformer_residency_barrier_failure_poisons_without_rebinding(monkeypatch) -> None:
    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.root = nn.Parameter(torch.ones(2))
            self.transformer_blocks = nn.ModuleList(
                [nn.Linear(2, 2, bias=False) for _ in range(48)]
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            for block in self.transformer_blocks:
                value = block(value)
            return value

    model = TinyTransformer()
    manager = _LTX23TransformerResidency(
        model, torch.device("cpu"), resident_weight_budget_bytes=10_000
    )
    with manager.forward_scope(lambda: None):
        model(torch.ones(1, 2))
    resident_values = {
        name: dict(model.named_parameters())[name]
        for name in model.state_dict()
        if name.startswith("transformer_blocks")
    }
    monkeypatch.setattr(
        manager,
        "_barrier",
        lambda _label: (_ for _ in ()).throw(RuntimeError("synthetic barrier loss")),
    )

    with pytest.raises(
        kitchen_module.DynamicResidencyPoisoned,
        match="device_quiescence_failed",
    ):
        manager.close()

    assert "barrier failed" in model._latentslate_ltx23_residency_poisoned
    assert not manager.handles
    assert any(
        dict(model.named_parameters()).get(name) is value for name, value in resident_values.items()
    )


def test_leaf_manager_poison_skips_finally_diagnostics_and_close_native_paths() -> None:
    events: list[str] = []

    class _Scheduler:
        def terminal_poison_reason(self):
            events.append("terminal_reason")
            return "failed_fill_quiescence_failed"

        def clear_stage(self):
            pytest.fail("poisoned scheduler must not clear")

        def close(self):
            pytest.fail("poisoned scheduler must not close")

    manager = object.__new__(_LTX23TransformerResidency)
    manager.transformer = nn.Module()
    manager.device = torch.device("cuda")
    manager._scheduler = _Scheduler()
    manager._dynamic = SimpleNamespace(
        diagnostics=lambda: pytest.fail("poisoned backend diagnostics are forbidden")
    )
    manager._terminal_dynamic_poison_reason = None
    manager._barrier_failed = False
    manager._closed = False
    manager._owner_thread = None
    manager._executing = False
    manager._before_first = None
    manager._scope_started = False
    manager._root_active = False
    manager._active_block = None
    manager._handles = []

    with pytest.raises(RuntimeError, match="primary"), manager.forward_scope(lambda: None):
        raise RuntimeError("primary")

    assert manager.terminal_poison_reason() == "failed_fill_quiescence_failed"
    assert manager.failure_diagnostics() == {}
    with pytest.raises(
        kitchen_module.DynamicResidencyPoisoned,
        match="failed_fill_quiescence_failed",
    ):
        manager.close()
    assert manager._scheduler is not None
    assert manager._dynamic is not None
    assert manager.transformer._latentslate_ltx23_residency_poisoned == (
        "failed_fill_quiescence_failed"
    )
    assert events == ["terminal_reason"]


def test_transformer_backend_poison_failure_diagnostics_keep_safe_counters() -> None:
    """Backend-terminal diagnostics are safe, including unknown loaded VBAR bytes."""

    calls: list[str] = []

    class _Backend:
        def terminal_poison_reason(self):
            calls.append("terminal_reason")
            return "failed_fill_quiescence_failed"

        def diagnostics(self):
            calls.append("diagnostics")
            return {"backend": "comfy-aimdo", "loaded_bytes": None, "faults": 17}

    manager = object.__new__(_LTX23TransformerResidency)
    manager.transformer = nn.Module()
    manager._barrier_failed = False
    manager._dynamic = _Backend()
    manager._base_file_handle = object()
    manager._base_file_handle_opened = 1
    manager._base_file_handle_closed = 0

    assert manager.failure_diagnostics() == {
        "dynamic_vram": {
            "backend": "comfy-aimdo",
            "loaded_bytes": None,
            "faults": 17,
            "policy": "required",
            "base_file_handle_live": True,
            "base_file_handle_opened": 1,
            "base_file_handle_closed": 0,
            "base_file_fallback_reason": None,
        }
    }
    assert calls == ["diagnostics"]


def test_transformer_wrapper_poison_failure_diagnostics_never_query_native_backend() -> None:
    class _Backend:
        def terminal_poison_reason(self):
            pytest.fail("barrier-only poison must not query backend terminal state")

        def diagnostics(self):
            pytest.fail("wrapper-only poison must not query native diagnostics")

    manager = object.__new__(_LTX23TransformerResidency)
    manager.transformer = nn.Module()
    manager.transformer._latentslate_ltx23_residency_poisoned = "wrapper barrier failure"
    manager._barrier_failed = True
    manager._dynamic = _Backend()

    assert manager.failure_diagnostics() == {}


def test_file_backed_transformer_residency_uses_per_leaf_allocations_and_tiny_residents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from latentslate_engine.runtime.framework.residency import aimdo as aimdo_module

    payload = tmp_path / "av-base.bin"
    payload.write_bytes(
        b"".join(
            torch.tensor([index], dtype=torch.float32).numpy().tobytes() for index in range(49)
        )
    )

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(1, device="meta"), requires_grad=False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + self.weight

    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.root = nn.Parameter(torch.empty(1, device="meta"), requires_grad=False)
            self.transformer_blocks = nn.ModuleList([Block() for _ in range(48)])

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            value = value + self.root
            for block in self.transformer_blocks:
                value = block(value)
            return value

    model = Transformer()
    descriptors = {}
    all_parameters = [model.root, *(block.weight for block in model.transformer_blocks)]
    for index, parameter in enumerate(all_parameters):
        span = aimdo_module.AimdoFileSpan(
            "ltx23_av_base",
            f"value.{index}",
            index * 4,
            4,
            torch.float32,
            (1,),
        )
        descriptors[id(parameter)] = aimdo_module.AimdoFileBackedValue(parameter, (span,))
    contract = SimpleNamespace(path=payload, variant="dev")
    model._latentslate_ltx23_av_source_descriptors = descriptors
    model._latentslate_ltx23_av_source_plan = SimpleNamespace(contract=contract)

    backend_instances = []

    class FakeAimdo:
        def __init__(self, _device, *, virtual_bytes, gathered_host_transfer):
            self.virtual_bytes = virtual_bytes
            self.groups = {}
            self.source = None
            self.seen = set()
            self.faults = 0
            self.hits = 0
            self.misses = 0
            self.releases = 0
            self.closed = False
            backend_instances.append(self)

        @staticmethod
        def group_bytes(values):
            return sum(
                sum(span.size for span in value.spans)
                if isinstance(value, aimdo_module.AimdoFileBackedValue)
                else value.numel() * value.element_size()
                for value in values
            )

        def allocate_group(self, key, values):
            self.groups[key] = values

        def prioritize(self):
            pass

        def bind_file_source(self, source_id, handle):
            assert source_id == "ltx23_av_base"
            self.source = handle

        def acquire(self, key):
            self.faults += 1
            if key in self.seen:
                self.hits += 1
            else:
                self.seen.add(key)
                self.misses += 1
            values = []
            for source in self.groups[key]:
                if isinstance(source, aimdo_module.AimdoFileBackedValue):
                    span = source.spans[0]
                    self.source.seek(span.offset)
                    raw = self.source.read(span.size)
                    value = torch.frombuffer(bytearray(raw), dtype=span.dtype).view(span.shape)
                    if isinstance(source.template, nn.Parameter):
                        value = nn.Parameter(value, requires_grad=False)
                    values.append(value)
                else:
                    values.append(source)
            return SimpleNamespace(values=tuple(values), token=key)

        def prefetch(self, key):
            return self.acquire(key)

        def wait(self, _lease):
            pass

        def synchronize(self, _lease):
            pass

        def release(self, _lease):
            self.releases += 1

        def diagnostics(self):
            return {
                "copy_strategy": "gathered_host_buffer",
                "copy_fallback_reason": None,
                "allocation_count": len(self.groups),
                "faults": self.faults,
                "signature_hits": self.hits,
                "signature_misses": self.misses,
            }

        def invalidate(self, *, reason):
            del reason

        def close(self):
            self.closed = True

    monkeypatch.setattr(aimdo_module, "AimdoDynamicResidency", FakeAimdo)
    monkeypatch.setattr(kitchen_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        kitchen_module, "inspect_ltx23_av_artifact", lambda *_args, **_kwargs: contract
    )

    manager = _LTX23TransformerResidency(
        model, torch.device("cuda"), resident_weight_budget_bytes=4
    )
    manager._barrier = lambda _label: None
    backend = backend_instances[-1]
    assert list(backend.groups) == [
        "<root>",
        *(f"transformer_blocks.{index}" for index in range(48)),
    ]
    assert manager._base_file_handle is not None

    for _ in range(2):
        with manager.forward_scope(lambda: None):
            torch.testing.assert_close(model(torch.zeros(1)), torch.tensor([1176.0]))
    assert (backend.faults, backend.misses, backend.hits) == (49, 49, 0)
    assert manager.policy["prefetch"] is True
    assert manager.policy["group_count"] == 49
    assert manager.policy["leaf_allocation_count"] == 49
    assert manager.policy["force_resident_leaf_count"] == 49
    assert manager.policy["base_file_handle_live"] is True
    assert manager.policy["cpu_source_bytes_base"] == 0

    manager.close()
    assert backend.closed is True
    assert manager._base_file_handle is None
    assert manager._base_file_handle_opened == manager._base_file_handle_closed == 1
    assert all(parameter.is_meta for parameter in model.parameters())


def test_cuda_transformer_residency_fails_closed_without_file_backed_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Module()
    model.root = nn.Parameter(torch.ones(1))
    model.transformer_blocks = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(48)])
    monkeypatch.setattr(kitchen_module.torch.cuda, "current_device", lambda: 0)

    with pytest.raises(RuntimeError, match="requires authenticated file-backed sources"):
        _LTX23TransformerResidency(model, torch.device("cuda"))


def test_failed_dynamic_initialization_retains_file_owner_when_close_cannot_quiesce() -> None:
    manager = object.__new__(_LTX23TransformerResidency)
    manager.transformer = nn.Module()

    class _Backend:
        def close(self):
            raise RuntimeError("synthetic quiescence loss")

    class _Handle:
        closed = False

        def close(self):
            self.closed = True

    backend = _Backend()
    handle = _Handle()
    manager._dynamic = backend
    manager._base_file_handle = handle
    manager._base_file_handle_opened = 1
    manager._base_file_handle_closed = 0
    primary = RuntimeError("setup failed")

    assert manager._cleanup_failed_initialization(primary) is False
    assert manager._dynamic is backend
    assert manager._base_file_handle is handle
    assert handle.closed is False
    assert manager._base_file_handle_closed == 0
    assert "quiescence loss" in manager.transformer._latentslate_ltx23_residency_poisoned
    assert any("cleanup also failed" in note for note in primary.__notes__)


def test_terminal_pool_setup_poison_skips_initialization_cleanup_and_retains_graph() -> None:
    manager = object.__new__(_LTX23TransformerResidency)
    manager.transformer = nn.Module()
    manager._terminal_dynamic_poison_reason = None
    events: list[str] = []

    class _Scheduler:
        def close(self):
            events.append("scheduler-close")

        def terminal_poison_reason(self):
            return "host_source_pool_setup_cleanup_failed"

    scheduler = _Scheduler()
    manager._scheduler = scheduler
    manager._dynamic = object()
    manager._base_file_handle = object()
    primary = kitchen_module.DynamicResidencyPoisoned("host_source_pool_setup_cleanup_failed")

    assert manager._cleanup_failed_initialization(primary) is False
    assert events == []
    assert manager._scheduler is scheduler
    assert manager._dynamic is not None
    assert manager._base_file_handle is not None
    assert manager.terminal_poison_reason() == "host_source_pool_setup_cleanup_failed"
