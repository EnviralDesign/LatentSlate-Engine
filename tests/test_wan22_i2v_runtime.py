from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from torch import nn

import latentslate_engine.runtime.wan22_i2v_runtime as runtime_module
from latentslate_engine.runtime.wan22_i2v_runtime import (
    NativeWanI2VRuntime,
    WanI2VArtifactPaths,
    WanI2VRequest,
)
from latentslate_engine.runtime.wan22_native_managed import ManagedNativeWanI2VRuntime
from latentslate_engine.runtime.wan22_prompt import WanPromptTokens


class _Plan:
    def __init__(self, name: str, contract: str = "contract"):
        self.identity = SimpleNamespace(
            path=name,
            header_sha256="q" * 64,
            size_bytes=100,
            mtime_ns=200,
        )
        self.artifact_contract = contract
        self.config_fingerprint = f"{name}-config"
        self.mapping_fingerprint = f"{name}-mapping"

    def require_available(self):
        return None


class _Tokenizer:
    model_sha256 = "t" * 64

    def tokenize_pair(self, prompt, negative):
        assert prompt == "a test"
        assert negative == ""
        return WanPromptTokens(
            input_ids=torch.zeros((2, 512), dtype=torch.int64),
            attention_mask=torch.zeros((2, 512), dtype=torch.int64),
            token_counts=(1, 1),
            sequence_length=512,
            tokenizer_sha256=self.model_sha256,
        )


class _Scheduler:
    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=1000)

    def set_timesteps(self, steps, device):
        assert steps == 4
        self.timesteps = torch.tensor([999, 900, 750, 500], device=device)

    def step(self, prediction, timestep, latents, return_dict=False):
        assert return_dict is False
        return (latents,)


class _Support:
    fingerprint = "s" * 64
    tokenizer_sha256 = "t" * 64
    boundary_ratio = 0.9

    def load_tokenizer(self):
        return _Tokenizer()

    def load_scheduler(self):
        return _Scheduler()


class _TextSession:
    entered = 0
    exited = 0

    def __init__(self, encoder, **kwargs):
        self.tokenizer_sha256 = encoder._latentslate_tokenizer_sha256

    def __enter__(self):
        type(self).entered += 1
        return self

    def __exit__(self, *args):
        type(self).exited += 1

    def encode(self, input_ids, attention_mask, *, sequence_length):
        assert sequence_length == 512
        return torch.zeros((2, 512, 4096), dtype=torch.float16)


class _VaeSession:
    entered = 0
    exited = 0

    def __init__(self, vae, plan, **kwargs):
        self.vae = vae

    def __enter__(self):
        type(self).entered += 1
        return self

    def __exit__(self, *args):
        type(self).exited += 1

    def encode(self, video):
        return torch.zeros((1, 16, 2, 8, 8), dtype=torch.bfloat16)

    def decode(self, latents):
        return torch.zeros((1, 3, 5, 64, 64), dtype=torch.bfloat16)


class _TransformerSession:
    entered: ClassVar[list[str]] = []
    exited: ClassVar[list[str]] = []

    def __init__(self, model, plan, **kwargs):
        self.transformer = model
        self.active = False

    def __enter__(self):
        self.active = True
        type(self).entered.append(self.transformer.name)
        return self

    def __exit__(self, *args):
        self.active = False
        type(self).exited.append(self.transformer.name)


class _Forward:
    def __init__(self, condition):
        assert condition.shape == (1, 20, 2, 8, 8)

    def __call__(self, model, session, latents, timestep, conditioning, identity):
        assert session.active
        assert identity == "cond"
        return torch.zeros_like(latents, dtype=torch.float16)


def _runtime(monkeypatch):
    monkeypatch.setattr(runtime_module, "revalidate_wan_i2v_support", lambda support: True)
    monkeypatch.setattr(runtime_module, "UMT5EncoderResidencySession", _TextSession)
    monkeypatch.setattr(runtime_module, "WanVaeResidencySession", _VaeSession)
    monkeypatch.setattr(runtime_module, "WanTransformerResidencySession", _TransformerSession)
    monkeypatch.setattr(runtime_module, "WanI2VForward", _Forward)
    def stored_snapshot(model):
        return {f"fake.{model.name}": {"bound": 1}}

    def verify_stored_dispatch(model, before):
        assert before == stored_snapshot(model)
        return {
            "fp8_module_count": 1,
            "fp8_modules": {
                f"fake.{model.name}": {
                    "native_dispatch_delta": 1,
                    "rejected_delta": 0,
                    "dense_fallback_delta": 0,
                }
            },
            "int8_module_count": 0,
            "int8_modules": {},
            "dense_fallback_count": 0,
            "rejected_count": 0,
        }

    monkeypatch.setattr(runtime_module, "wan_stored_dispatch_snapshot", stored_snapshot)
    monkeypatch.setattr(runtime_module, "verify_wan_stored_dispatch", verify_stored_dispatch)
    monkeypatch.setattr(
        runtime_module,
        "preprocess_wan_i2v_image",
        lambda image, height, width: torch.zeros((1, 3, height, width)),
    )
    high = nn.Linear(1, 1)
    high.name = "high"
    high._latentslate_compute_dtype = torch.float16
    high._latentslate_wan_config_fingerprint = "h-config"
    high._latentslate_wan_mapping_fingerprint = "h-mapping"
    high._latentslate_wan_artifact_identity = None
    low = nn.Linear(1, 1)
    low.name = "low"
    low._latentslate_compute_dtype = torch.float16
    low._latentslate_wan_config_fingerprint = "l-config"
    low._latentslate_wan_mapping_fingerprint = "l-mapping"
    low._latentslate_wan_artifact_identity = None
    text = nn.Linear(1, 1)
    text._latentslate_tokenizer_sha256 = "t" * 64
    text._latentslate_umt5_config_fingerprint = "x-config"
    text._latentslate_umt5_mapping_fingerprint = "x-mapping"
    text._latentslate_umt5_artifact_identity = None
    vae = nn.Linear(1, 1)
    vae._latentslate_vae_config_fingerprint = "v-config"
    vae._latentslate_vae_mapping_fingerprint = "v-mapping"
    vae._latentslate_vae_artifact_identity = None
    runtime = NativeWanI2VRuntime(
        support=_Support(),
        high_plan=_Plan("h"),
        low_plan=_Plan("l"),
        text_plan=_Plan("x", "text-contract"),
        vae_plan=_Plan("v"),
        high_model=high,
        low_model=low,
        text_encoder=text,
        vae=vae,
        high_residency=object(),
        low_residency=object(),
    )
    high._latentslate_wan_artifact_identity = runtime.high_plan.identity
    low._latentslate_wan_artifact_identity = runtime.low_plan.identity
    text._latentslate_umt5_artifact_identity = runtime.text_plan.identity
    vae._latentslate_vae_artifact_identity = runtime.vae_plan.identity
    return runtime


def test_load_materializes_catalog_bound_plans_without_replanning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    support_root = tmp_path / "support"
    support_root.mkdir()
    paths_by_role = {}
    for role in ("transformer_high_noise", "transformer_low_noise", "text_encoder", "vae"):
        path = tmp_path / f"{role}.safetensors"
        path.write_bytes(role.encode())
        paths_by_role[role] = path.resolve()
    paths = WanI2VArtifactPaths(
        support=support_root,
        transformer_high=paths_by_role["transformer_high_noise"],
        transformer_low=paths_by_role["transformer_low_noise"],
        text_encoder=paths_by_role["text_encoder"],
        vae=paths_by_role["vae"],
    )
    support = SimpleNamespace(root=support_root.resolve())
    plans = {
        role: _Plan(path, "shared" if role.startswith("transformer") else role)
        for role, path in paths_by_role.items()
    }
    monkeypatch.setattr(runtime_module, "revalidate_wan_i2v_support", lambda _plan: True)
    for planner in (
        "plan_wan_i2v_support",
        "plan_stored_wan_transformer",
        "plan_stored_umt5_encoder",
        "plan_stored_wan21_vae",
    ):
        monkeypatch.setattr(
            runtime_module,
            planner,
            lambda *_args: (_ for _ in ()).throw(AssertionError("must not replan")),
        )
    monkeypatch.setattr(
        runtime_module, "materialize_wan_transformer", lambda *_a, **_k: nn.Linear(1, 1)
    )
    monkeypatch.setattr(
        runtime_module, "materialize_umt5_encoder", lambda *_a, **_k: nn.Linear(1, 1)
    )
    monkeypatch.setattr(runtime_module, "materialize_wan21_vae", lambda *_a, **_k: nn.Linear(1, 1))
    monkeypatch.setattr(runtime_module, "plan_wan_root_residency", lambda _model: object())
    monkeypatch.setattr(NativeWanI2VRuntime, "_validate_component_binding", lambda _self: None)

    runtime = NativeWanI2VRuntime.load(
        paths,
        support_plan=support,
        adapter_plans=plans,
    )

    assert runtime.support is support
    assert runtime.high_plan is plans["transformer_high_noise"]
    assert runtime.low_plan is plans["transformer_low_noise"]
    assert runtime.text_plan is plans["text_encoder"]
    assert runtime.vae_plan is plans["vae"]


def test_runtime_composes_stages_and_returns_cpu_video(monkeypatch):
    runtime = _runtime(monkeypatch)
    progress = []
    result = runtime.generate(
        WanI2VRequest(image=object(), prompt="a test"),
        device="cpu",
        progress=lambda completed, total, stage: progress.append((completed, total, stage)),
    )
    assert result.video.shape == (1, 3, 5, 64, 64)
    assert result.video.device.type == "cpu"
    assert [item[2] for item in progress] == ["high", "high", "low", "low"]
    assert _TransformerSession.entered[-2:] == ["high", "low"]
    assert _TransformerSession.exited[-2:] == ["high", "low"]
    assert _TextSession.entered == _TextSession.exited
    assert _VaeSession.entered == _VaeSession.exited
    assert result.provenance.stage_policy == "expert_split"
    assert (result.provenance.sampler, result.provenance.scheduler, result.provenance.shift) == (
        "euler",
        "simple",
        5.0,
    )
    assert result.provenance.transformer_dispatch == {
        stage: {
            "fp8_module_count": 1,
            "fp8_modules": {
                f"fake.{stage}": {
                    "native_dispatch_delta": 1,
                    "rejected_delta": 0,
                    "dense_fallback_delta": 0,
                }
            },
            "int8_module_count": 0,
            "int8_modules": {},
            "dense_fallback_count": 0,
            "rejected_count": 0,
        }
        for stage in ("high", "low")
    }


def test_runtime_rejects_support_text_identity_mismatch(monkeypatch):
    runtime = _runtime(monkeypatch)
    runtime.text_encoder._latentslate_tokenizer_sha256 = "z" * 64
    with pytest.raises(ValueError, match="tokenizer"):
        runtime.generate(WanI2VRequest(image=object(), prompt="a test"), device="cpu")


def test_runtime_rejects_same_header_high_low_model_swap(monkeypatch):
    runtime = _runtime(monkeypatch)
    assert runtime.high_plan.identity.header_sha256 == runtime.low_plan.identity.header_sha256
    runtime.high_model, runtime.low_model = runtime.low_model, runtime.high_model
    with pytest.raises(ValueError, match="high transformer"):
        runtime.generate(WanI2VRequest(image=object(), prompt="a test"), device="cpu")


def test_runtime_release_dematerializes_cpu_components_and_is_terminal(monkeypatch):
    runtime = _runtime(monkeypatch)

    runtime.release()

    for module in (runtime.high_model, runtime.low_model, runtime.text_encoder, runtime.vae):
        assert all(value.is_meta for value in module.parameters())
    runtime.release()  # idempotent cleanup is required for manager failure paths.
    with pytest.raises(RuntimeError, match="released"):
        runtime.generate(WanI2VRequest(image=object(), prompt="a test"), device="cpu")


def test_managed_unload_terminates_an_active_disposable_worker():
    terminated = []
    waited = []

    class _Process:
        def __init__(self):
            self.exited = False

        def poll(self):
            return 0 if self.exited else None

        def wait(self, timeout):
            waited.append(timeout)
            self.exited = True
            return 0

    class _Tree:
        process = _Process()

        def active_processes(self):
            return 1

        def terminate(self):
            terminated.append(True)

        def wait_for_empty(self, timeout=15.0):
            waited.append(f"tree:{timeout}")

        def close(self):
            terminated.append("closed")

    managed = object.__new__(ManagedNativeWanI2VRuntime)
    managed._active_tree = _Tree()
    managed.unload()

    assert terminated == [True, "closed"]
    assert waited == [15, "tree:15.0"]
    assert managed._active_tree is None


def test_runtime_cancellation_closes_active_transformer_session(monkeypatch):
    runtime = _runtime(monkeypatch)
    progress = []

    def cancelled():
        return bool(progress)

    with pytest.raises(asyncio.CancelledError):
        runtime.generate(
            WanI2VRequest(image=object(), prompt="a test"),
            device="cpu",
            progress=lambda completed, total, stage: progress.append(stage),
            cancelled=cancelled,
        )
    assert progress == ["high"]
    assert _TransformerSession.entered[-1:] == ["high"]
    assert _TransformerSession.exited[-1:] == ["high"]


@pytest.mark.parametrize("steps", [1, 1001, True])
def test_runtime_rejects_invalid_step_budget(monkeypatch, steps):
    runtime = _runtime(monkeypatch)
    with pytest.raises((TypeError, ValueError), match="steps"):
        runtime.generate(
            WanI2VRequest(image=object(), prompt="a test", steps=steps),
            device="cpu",
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"num_frames": 4},
        {"height": 65},
        {"width": 1296},
        {"seed": -1},
        {"seed": True},
    ),
)
def test_runtime_rejects_invalid_request_before_residency(monkeypatch, overrides):
    runtime = _runtime(monkeypatch)
    text_entries = _TextSession.entered
    vae_entries = _VaeSession.entered
    with pytest.raises(ValueError):
        runtime.generate(
            WanI2VRequest(image=object(), prompt="a test", **overrides),
            device="cpu",
        )
    assert _TextSession.entered == text_entries
    assert _VaeSession.entered == vae_entries
