from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import latentslate_engine.runtime.wan21_vae_adapter as vae_adapter
from latentslate_engine.runtime.wan21_vae_adapter import (
    WanVaeResidencySession,
    WanVaeSemantics,
    build_wan21_vae_skeleton,
    comfy_key_for_wan21_vae_target,
    configure_wan21_vae_memory,
    encode_wan21_latents,
    materialize_wan21_vae,
    plan_comfy_wan21_vae,
)

_CFG = {
    "base_dim": 8,
    "dim_mult": [1, 1],
    "num_res_blocks": 1,
    "temperal_downsample": [False],
    "z_dim": 16,
    "in_channels": 3,
    "out_channels": 3,
    "scale_factor_temporal": 1,
    "scale_factor_spatial": 2,
    "latents_mean": [0.1] * 16,
    "latents_std": [1.1] * 16,
}


def _write(path: Path):
    m = build_wan21_vae_skeleton(_CFG)
    tensors = {}
    for target, value in m.state_dict().items():
        source = comfy_key_for_wan21_vae_target(target)
        assert source, target
        tensors[source] = torch.zeros(tuple(value.shape), dtype=torch.bfloat16)
    save_file(tensors, path)


def _plan(path, monkeypatch):
    real = vae_adapter.probe_safetensors
    monkeypatch.setattr(
        vae_adapter,
        "probe_safetensors",
        lambda p: replace(
            real(p), architecture_signals=("wan_vae_2_1",), component_signals=("vae",)
        ),
    )
    return plan_comfy_wan21_vae(path, _CFG)


def test_small_vae_plan_materializes_and_forward(tmp_path, monkeypatch):
    p = tmp_path / "vae.safetensors"
    _write(p)
    plan = _plan(p, monkeypatch)
    assert plan.available
    vae = materialize_wan21_vae(plan, _CFG)
    assert not any(x.is_meta for x in vae.parameters())
    x = torch.zeros((1, 3, 1, 8, 8), dtype=torch.bfloat16)
    z = vae.encode(x).latent_dist.mode()
    out = vae.decode(z).sample
    assert z.shape[1] == 16 and out.shape[1] == 3


def test_vae_plan_rejects_bf16_mapping_gap(tmp_path, monkeypatch):
    p = tmp_path / "bad.safetensors"
    _write(p)
    from safetensors import safe_open

    with safe_open(str(p), framework="pt", device="cpu") as h:
        keys = h.keys()
        t = {k: h.get_tensor(k) for k in keys}
    t.pop(next(iter(t)))
    save_file(t, p)
    assert not _plan(p, monkeypatch).available


def test_vae_semantics_residency_and_memory_controls(tmp_path, monkeypatch):
    p = tmp_path / "resident.safetensors"
    _write(p)
    vae = materialize_wan21_vae(_plan(p, monkeypatch), _CFG)
    semantics = WanVaeSemantics()
    mean = torch.zeros(16)
    std = torch.ones(16)
    z = torch.ones((1, 16, 1, 1, 1), dtype=torch.bfloat16)
    assert torch.equal(semantics.denormalize(semantics.normalize(z, mean, std), mean, std), z)
    configure_wan21_vae_memory(vae, tiling=True, slicing=True)
    plan = _plan(p, monkeypatch)
    with WanVaeResidencySession(vae, plan, onload_device="cpu"):
        assert all(x.device.type == "cpu" for x in vae.parameters())
    assert all(x.device.type == "cpu" for x in vae.parameters())


def test_vae_multiframe_ratio_and_nontrivial_normalization(tmp_path, monkeypatch):
    p = tmp_path / "ratio.safetensors"
    _write(p)
    plan = _plan(p, monkeypatch)
    vae = materialize_wan21_vae(plan, _CFG)
    sem = plan.semantics
    mean = torch.arange(16, dtype=torch.bfloat16)
    std = torch.arange(1, 17, dtype=torch.bfloat16)
    raw = torch.ones((1, 16, 5, 8, 8), dtype=torch.bfloat16) * 3
    normalized = sem.normalize(raw, mean, std)
    assert torch.equal(normalized[:, 3], (raw[:, 3] - mean[3]) / std[3])
    assert torch.equal(sem.denormalize(normalized, mean, std), raw)
    video = torch.zeros((1, 3, 5, 16, 16), dtype=torch.bfloat16)
    with WanVaeResidencySession(vae, plan, onload_device="cpu") as session:
        latent = session.encode(video)
        out = session.decode(latent)
    assert latent.shape[1:] == (
        sem.latent_channels,
        ((video.shape[2] - 1) // sem.temporal_ratio) + 1,
        video.shape[3] // sem.spatial_ratio,
        video.shape[4] // sem.spatial_ratio,
    )
    assert out.shape == video.shape


def test_vae_rejects_temporal_lengths_that_cannot_round_trip():
    semantics = WanVaeSemantics(
        temporal_ratio=4,
        mean=(0.0,) * 16,
        std_values=(1.0,) * 16,
    )

    class MustNotEncode:
        def encode(self, _video):
            raise AssertionError("invalid temporal length reached the VAE")

    video = torch.zeros((1, 3, 4, 16, 16), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="4k\\+1"):
        encode_wan21_latents(MustNotEncode(), video, semantics)


def test_vae_paused_active_close_is_rejected_then_cleans(tmp_path, monkeypatch):
    p = tmp_path / "paused.safetensors"
    _write(p)
    plan = _plan(p, monkeypatch)
    vae = materialize_wan21_vae(plan, _CFG)
    session = WanVaeResidencySession(vae, plan, onload_device="cpu")
    started = threading.Event()
    release = threading.Event()
    errors = []
    original_encode = vae.encode

    def pause(video):
        started.set()
        release.wait(5)
        return original_encode(video)

    def worker():
        try:
            with session:
                session.encode(torch.zeros((1, 3, 1, 8, 8), dtype=torch.bfloat16))
        except BaseException as exc:  # noqa: BLE001 - captured for thread assertion
            errors.append(exc)

    monkeypatch.setattr(vae, "encode", pause)
    worker_thread = threading.Thread(target=worker)
    worker_thread.start()
    assert started.wait(2)
    with pytest.raises(RuntimeError, match="owning thread"):
        session.close()
    with pytest.raises(RuntimeError, match="owning residency"):
        session.encode(torch.zeros((1, 3, 1, 8, 8), dtype=torch.bfloat16))
    release.set()
    worker_thread.join()
    assert not errors and not session.active
    assert all(x.device.type == "cpu" for x in vae.parameters())


def test_vae_owner_active_close_and_base_exception_cleanup(tmp_path, monkeypatch):
    p = tmp_path / "active.safetensors"
    _write(p)
    plan = _plan(p, monkeypatch)
    vae = materialize_wan21_vae(plan, _CFG)
    session = WanVaeResidencySession(vae, plan, onload_device="cpu")

    def close_during_encode(_video):
        session.close()

    monkeypatch.setattr(vae, "encode", close_during_encode)
    with pytest.raises(RuntimeError, match="while encode/decode is active"), session:
        session.encode(torch.zeros((1, 3, 1, 8, 8), dtype=torch.bfloat16))
    assert not session.active
    assert all(x.device.type == "cpu" for x in vae.parameters())

    vae = materialize_wan21_vae(plan, _CFG)
    session = WanVaeResidencySession(vae, plan, onload_device="cpu")

    def interrupt(_video):
        raise KeyboardInterrupt

    monkeypatch.setattr(vae, "encode", interrupt)
    with pytest.raises(KeyboardInterrupt), session:
        session.encode(torch.zeros((1, 3, 1, 8, 8), dtype=torch.bfloat16))
    assert not session.active
    assert all(x.device.type == "cpu" for x in vae.parameters())


def test_vae_plan_rejects_duplicate_target(tmp_path, monkeypatch):
    p = tmp_path / "duplicate.safetensors"
    _write(p)
    from safetensors import safe_open

    with safe_open(str(p), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        tensors = {key: handle.get_tensor(key) for key in keys}
    canonical = "encoder.downsamples.0.residual.2.weight"
    tensors["encoder.downsamples.0.conv1.weight"] = tensors[canonical].clone()
    save_file(tensors, p)
    plan = _plan(p, monkeypatch)
    assert not plan.available
    assert plan.duplicate_targets
