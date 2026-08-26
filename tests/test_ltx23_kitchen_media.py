from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from latentslate_engine.artifacts import probe_safetensors
from latentslate_engine.runtime import ltx23_kitchen_media as media_module
from latentslate_engine.runtime.ltx23_kitchen_contracts import (
    LTX23_DEV_FP8,
    LTX23_SPATIAL_UPSCALER,
    LTX23StoredArtifactPlan,
    plan_ltx23_stored_artifact,
)
from latentslate_engine.runtime.ltx23_kitchen_media import (
    _active_meta_tensor_names,
    _LTX23MediaRuntimeBuffer,
    build_ltx23_media_shell,
    ltx23_media_component_residency,
    materialize_ltx23_media_component,
    plan_ltx23_media_component,
    unload_ltx23_media_component,
)


def _ensure(root: nn.Module, path: str) -> nn.Module:
    current = root
    for part in path.split(".") if path else ():
        if part not in current._modules:
            current.add_module(part, nn.Module())
        current = current._modules[part]
    return current


def _shell(targets: dict[str, torch.Tensor]) -> nn.Module:
    shell = nn.Module()
    for target, value in targets.items():
        parent_path, _, leaf = target.rpartition(".")
        _ensure(shell, parent_path).register_parameter(
            leaf, nn.Parameter(torch.empty(tuple(value.shape), dtype=value.dtype, device="meta"), requires_grad=False)
        )
    return shell


def _stored_plan(path: Path, roles: dict[str, str]) -> LTX23StoredArtifactPlan:
    probe = probe_safetensors(path)
    return LTX23StoredArtifactPlan(
        contract=LTX23_DEV_FP8,
        identity=probe.identity,
        roles=MappingProxyType(roles),
        component_counts=MappingProxyType({}),
        quantized_layers=MappingProxyType({}),
        auxiliary_sources=(),
        errors=(),
        fingerprint="synthetic-media-plan",
    )


def _fixture(component: str, tmp_path: Path) -> tuple[LTX23StoredArtifactPlan, nn.Module]:
    count = {"video_vae": 170, "audio_vae": 102, "vocoder": 1227, "latent_upsampler": 72}[component]
    if component == "video_vae":
        keys = ["vae.up_blocks.7.res_blocks.0.weight"]
        keys += [f"vae.down_blocks.0.unit_{index}.weight" for index in range(167)]
        keys += [
            "vae.per_channel_statistics.mean-of-means",
            "vae.per_channel_statistics.std-of-means",
        ]
        role = "vae/dense"
    elif component == "audio_vae":
        keys = ["audio_vae.per_channel_statistics.mean-of-means", "audio_vae.per_channel_statistics.std-of-means"]
        keys += [f"audio_vae.blocks.{index}.weight" for index in range(100)]
        role = "audio_vae/dense"
    elif component == "vocoder":
        keys = [f"vocoder.generator.ups.0.unit_{index}.weight" for index in range(count)]
        role = "vocoder/dense"
    else:
        keys = [f"resnets.{index}.weight" for index in range(count)]
        role = "latent_upscaler/dense"
    values = {key: torch.zeros((1,), dtype=torch.bfloat16) for key in keys}
    path = tmp_path / f"{component}.safetensors"
    save_file(values, path)
    roles = {key: role for key in keys}
    plan = _stored_plan(path, roles)

    # The test shell is deliberately tiny, but it has the exact converted
    # topology.  Production uses the pinned Diffusers meta shells instead.
    targets: dict[str, torch.Tensor] = {}
    for source, value in values.items():
        prefix = "" if component == "latent_upsampler" else component.replace("video_vae", "vae").replace("audio_vae", "audio_vae") + "."
        key = source.removeprefix(prefix)
        if component == "video_vae":
            key = (
                key.replace("up_blocks.7", "up_blocks.3.upsamplers.0")
                .replace("res_blocks", "resnets")
                .replace("per_channel_statistics.mean-of-means", "latents_mean")
                .replace("per_channel_statistics.std-of-means", "latents_std")
            )
        elif component == "audio_vae":
            key = key.replace("per_channel_statistics.mean-of-means", "latents_mean").replace("per_channel_statistics.std-of-means", "latents_std")
        elif component == "vocoder":
            key = key.replace(".ups.", ".upsamplers.")
        targets[key] = value
    assert len(keys) == count
    return plan, _shell(targets)


@pytest.mark.parametrize("component", ["video_vae", "audio_vae", "vocoder", "latent_upsampler"])
def test_media_component_plans_prove_exact_source_and_shell_closure(component: str, tmp_path: Path) -> None:
    stored, shell = _fixture(component, tmp_path)

    plan = plan_ltx23_media_component(stored, component, shell)  # type: ignore[arg-type]

    assert plan.source_count == {"video_vae": 170, "audio_vae": 102, "vocoder": 1227, "latent_upsampler": 72}[component]
    if component == "video_vae":
        assert plan.ignored_sources == ()
        assert any(item.target == "up_blocks.3.upsamplers.0.resnets.0.weight" for item in plan.tensors)
    if component == "vocoder":
        assert plan.tensors[0].target.startswith("generator.upsamplers.0.")


@pytest.mark.parametrize("component", ["video_vae", "audio_vae", "vocoder", "latent_upsampler"])
def test_media_components_materialize_and_unload_independently(component: str, tmp_path: Path) -> None:
    stored, shell = _fixture(component, tmp_path)
    plan = plan_ltx23_media_component(stored, component, shell)  # type: ignore[arg-type]

    materialize_ltx23_media_component(shell, plan)

    assert ltx23_media_component_residency(shell) == "cpu"
    assert all(not value.is_meta for value in shell.state_dict().values())
    unload_ltx23_media_component(shell)
    assert ltx23_media_component_residency(shell) == "meta"


def test_caller_owned_handle_prevents_media_reopen_and_preserves_payload_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored, shell = _fixture("audio_vae", tmp_path)
    with safe_open(str(stored.identity.path), framework="pt", device="cpu") as handle:
        monkeypatch.setattr(
            media_module,
            "safe_open",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("caller-owned media handle must prevent reopen")
            ),
        )
        plan = plan_ltx23_media_component(
            stored, "audio_vae", shell, payload_handle=handle
        )
        source_values = {
            item.target: handle.get_tensor(item.source) for item in plan.tensors
        }
        materialize_ltx23_media_component(shell, plan, payload_handle=handle)

    state = shell.state_dict()
    assert all(
        state[target].data_ptr() == source.data_ptr()
        for target, source in source_values.items()
    )


def test_independent_media_helpers_open_their_artifact_without_a_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored, shell = _fixture("latent_upsampler", tmp_path)
    opens = 0
    real_safe_open = media_module.safe_open

    def counted_safe_open(*args, **kwargs):
        nonlocal opens
        opens += 1
        return real_safe_open(*args, **kwargs)

    monkeypatch.setattr(media_module, "safe_open", counted_safe_open)
    plan = plan_ltx23_media_component(stored, "latent_upsampler", shell)
    materialize_ltx23_media_component(shell, plan)

    assert opens == 2


def test_shared_handle_key_mismatch_fails_plan_and_poisons_materialization(
    tmp_path: Path,
) -> None:
    stored, shell = _fixture("audio_vae", tmp_path)

    class _WrongHandle:
        def keys(self):
            return ()

    with pytest.raises(ValueError, match="key set changed while opening header"):
        plan_ltx23_media_component(
            stored, "audio_vae", shell, payload_handle=_WrongHandle()
        )

    plan = plan_ltx23_media_component(stored, "audio_vae", shell)
    with pytest.raises(ValueError, match="key set changed during materialization"):
        materialize_ltx23_media_component(
            shell, plan, payload_handle=_WrongHandle()
        )
    assert "ValueError" in shell._latentslate_ltx23_media_poisoned  # type: ignore[attr-defined]


def test_media_materialization_restores_nonpersistent_runtime_buffers(tmp_path: Path) -> None:
    """A meta-shell transition must not strand vocoder-style runtime filters."""

    stored, shell = _fixture("vocoder", tmp_path)
    runtime = _ensure(shell, "resampler")
    expected = torch.tensor([[[0.25, 0.5, 0.25]]], dtype=torch.float32)
    runtime.register_buffer("filter", expected, persistent=False)
    shell._latentslate_ltx23_media_runtime_buffers = (  # type: ignore[attr-defined]
        _LTX23MediaRuntimeBuffer("resampler", "filter", expected.clone()),
    )
    shell.to_empty(device="meta")
    plan = plan_ltx23_media_component(stored, "vocoder", shell)

    materialize_ltx23_media_component(shell, plan)

    assert _active_meta_tensor_names(shell) == []
    assert torch.equal(shell.resampler.filter, expected)
    shell.to("cpu")


def test_vocoder_meta_shell_keeps_exact_noncheckpoint_resampler_filter() -> None:
    """The pinned Diffusers shell exposes the sole non-checkpoint closure."""

    shell = build_ltx23_media_shell("vocoder")

    buffers = shell._latentslate_ltx23_media_runtime_buffers  # type: ignore[attr-defined]
    assert [(item.target, tuple(item.value.shape), item.value.dtype) for item in buffers] == [
        ("resampler.filter", (1, 1, 43), torch.float32),
    ]
    assert shell.resampler.filter.is_meta


@pytest.mark.skipif(
    os.environ.get("LATENTSLATE_LTX23_REAL_HEADERS") != "1",
    reason="set LATENTSLATE_LTX23_REAL_HEADERS=1 to inspect installed LTX headers",
)
def test_opt_in_installed_ltx23_media_component_shell_closures() -> None:
    root = Path(r"M:\LatentSlateEngineData")
    checkpoint = root / "models/ltx23/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
    upsampler = root / "models/ltx23/latent_upscalers/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    if not checkpoint.is_file() or not upsampler.is_file():
        pytest.skip("LTX 2.3 media artifacts are not installed")
    combined = plan_ltx23_stored_artifact(checkpoint, LTX23_DEV_FP8)
    spatial = plan_ltx23_stored_artifact(upsampler, LTX23_SPATIAL_UPSCALER)
    for component, artifact in (
        ("video_vae", combined),
        ("audio_vae", combined),
        ("vocoder", combined),
        ("latent_upsampler", spatial),
    ):
        plan = plan_ltx23_media_component(artifact, component, build_ltx23_media_shell(component))  # type: ignore[arg-type]
        assert plan.source_count == {"video_vae": 170, "audio_vae": 102, "vocoder": 1227, "latent_upsampler": 72}[component]
