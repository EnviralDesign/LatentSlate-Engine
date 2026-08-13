from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from latentslate_engine.runtime.ltx23_kitchen import (
    LTX23_AUDIO_CHANNELS,
    LTX23_AUDIO_SAMPLE_RATE,
    LTX23_FLF_NEGATIVE_PROMPT,
    LTX23_PROMPT_ENHANCEMENT_SEED,
    LTX23_PROMPT_GENERATION_SETTINGS,
    LTX23_PROMPT_MAX_NEW_TOKENS,
    LTX23_REFINE_SEED,
    LTX23KitchenGeneration,
    _diffusers_sigmas,
    _LTX23TransformerResidency,
    _mux_mp4,
    _probe_mp4,
    _prompt_system_sha256,
    _stereo_audio,
    _uint8_frames,
    ltx23_guide_identity,
    ltx23_kitchen_operation_spec,
    validate_ltx23_kitchen_generation,
)


def test_exact_operation_topologies() -> None:
    t2v = ltx23_kitchen_operation_spec("ltx23_dev_t2v")
    assert t2v.prompt_enhancement is True
    assert t2v.model_lora_strength == 0.5
    assert t2v.text_lora_strength == 1.0
    assert len(t2v.main_sigmas) - 1 == 8
    assert len(t2v.refine_sigmas or ()) - 1 == 3
    assert "x2" in t2v.stages

    i2v = ltx23_kitchen_operation_spec("ltx23_dev_i2v")
    assert i2v.prompt_enhancement is False
    assert i2v.guide_strengths == (0.7, 1.0)

    flf = ltx23_kitchen_operation_spec("ltx23_distilled_flf")
    assert flf.refine_sigmas is None
    assert flf.guide_strengths == (0.7, 0.7)
    assert flf.stages.index("guide_first") < flf.stages.index("guide_last")
    assert flf.fps == 24
    assert flf.audio_sample_rate == LTX23_AUDIO_SAMPLE_RATE
    assert flf.audio_channels == LTX23_AUDIO_CHANNELS
    assert LTX23_REFINE_SEED == 42


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
        "11851da115bddb2dc0c13f574083175d0aa37df15b8b83b1ab5582879dda1bc5"
    )
    assert LTX23_PROMPT_ENHANCEMENT_SEED == 0
    assert LTX23_PROMPT_MAX_NEW_TOKENS == 2_048
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


def test_t2v_prompt_enhancement_does_not_inherit_the_video_seed() -> None:
    source = Path("src/latentslate_engine/runtime/ltx23_kitchen.py").read_text(encoding="utf-8")
    call = source[
        source.index("prompt = _enhance_prompt(") : source.index(
            ")\n", source.index("prompt = _enhance_prompt(")
        )
    ]
    assert "LTX23_PROMPT_ENHANCEMENT_SEED" in call
    assert "g.seed" not in call


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


def test_mux_publishes_24fps_48khz_stereo(tmp_path: Path) -> None:
    import av

    output = tmp_path / "native.mp4"
    frames = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    audio = np.zeros((2, 6000), dtype=np.float32)
    checks = 0

    def check_cancelled() -> None:
        nonlocal checks
        checks += 1

    _mux_mp4(frames, audio, output, check_cancelled=check_cancelled)
    assert output.stat().st_size > 0
    assert checks >= 3 + 6
    observed = _probe_mp4(output, check_cancelled)
    assert observed["container_format"].split(",").count("mp4") == 1
    assert observed["video_codec"] == "h264"
    assert observed["audio_codec"] == "aac"
    assert observed["width"] == observed["height"] == 32
    assert observed["num_frames"] == 3
    assert observed["fps"] == 24
    assert observed["audio_sample_rate"] == 48_000
    assert observed["audio_channels"] == 2
    with av.open(str(output)) as container:
        video = container.streams.video[0]
        sound = container.streams.audio[0]
        assert video.average_rate == 24
        assert sound.sample_rate == 48_000
        assert sound.layout.name == "stereo"


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
            np.zeros((2, 6000), dtype=np.float32),
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


def test_transformer_residency_retains_largest_budgeted_blocks_across_jobs() -> None:
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
                [WeightedBlock(index + 1) for index in range(48)]
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            for block in self.transformer_blocks:
                value = block(value)
            return value

    model = TinyTransformer()
    originals = {name: parameter for name, parameter in model.named_parameters()}
    root_bytes = model.root.numel() * model.root.element_size()
    largest_two = (48 + 47) * torch.tensor([], dtype=torch.float32).element_size()
    manager = _LTX23TransformerResidency(
        model,
        torch.device("cpu"),
        resident_weight_budget_bytes=root_bytes + largest_two,
    )

    for _job in range(2):
        with manager.forward_scope(lambda: None):
            assert torch.equal(model(torch.ones(1)), torch.ones(1))

    policy = manager.policy
    assert policy["resident_block_count"] == 2
    assert manager._desired_resident == (
        "transformer_blocks.46",
        "transformer_blocks.47",
    )
    assert policy["streamed_transitions"] == 46 * 2
    assert policy["resident_refills"] == 1
    manager.close()
    assert all(dict(model.named_parameters())[name] is value for name, value in originals.items())


def test_transformer_residency_refill_rolls_back_to_cpu_originals(monkeypatch) -> None:
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
    originals = {name: parameter for name, parameter in model.named_parameters()}
    manager = _LTX23TransformerResidency(
        model, torch.device("cpu"), resident_weight_budget_bytes=10_000
    )
    failing = manager.block_storage["transformer_blocks.1"]
    original_copy = type(failing).copy_to

    def fail_selected(storage, device):
        if storage is failing:
            raise RuntimeError("synthetic refill failure")
        return original_copy(storage, device)

    monkeypatch.setattr(
        type(failing),
        "copy_to",
        fail_selected,
    )

    with (
        pytest.raises(RuntimeError, match="synthetic refill failure"),
        manager.forward_scope(lambda: None),
    ):
        model(torch.ones(1, 2))

    assert "refill failed" in model._latentslate_ltx23_residency_poisoned
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

    with pytest.raises(RuntimeError, match="teardown barrier failed"):
        manager.close()

    assert "barrier failed" in model._latentslate_ltx23_residency_poisoned
    assert not manager.handles
    assert any(
        dict(model.named_parameters()).get(name) is value
        for name, value in resident_values.items()
    )
