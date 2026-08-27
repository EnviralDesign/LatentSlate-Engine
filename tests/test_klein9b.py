from __future__ import annotations

from pathlib import Path

import torch

from latentslate_engine.klein9b.model import KleinTransformer
from latentslate_engine.klein9b.runtime import (
    Klein9BIdentity,
    Klein9BRuntime,
    _sigmas,
    _unpack_latent,
)


def _identity(root: Path, suffix: str = "") -> Klein9BIdentity:
    paths = []
    for name in ("diffusion", "text", "vae"):
        path = root / f"{name}{suffix}.safetensors"
        path.write_bytes(name.encode())
        paths.append(path)
    tokenizer = root / f"tokenizer{suffix}"
    tokenizer.mkdir()
    return Klein9BIdentity.from_paths(*paths, tokenizer)


def test_exact_identity_reuses_state_and_change_releases_it(tmp_path: Path) -> None:
    first = _identity(tmp_path)
    second = _identity(tmp_path, "-changed")
    runtime = Klein9BRuntime(device="cpu")

    assert runtime.ensure_identity(first) is False
    transformer = object()
    vae = object()
    conditioning = ("prompt", torch.zeros(1))
    runtime.transformer = transformer  # type: ignore[assignment]
    runtime.vae = vae  # type: ignore[assignment]
    runtime.conditioning = conditioning

    assert runtime.ensure_identity(first) is True
    assert runtime.transformer is transformer
    assert runtime.vae is vae
    assert runtime.conditioning is conditioning

    assert runtime.ensure_identity(second) is False
    assert runtime.identity == second
    assert runtime.transformer is None
    assert runtime.vae is None
    assert runtime.conditioning is None


def test_packed_latent_becomes_flux2_vae_shape() -> None:
    packed = torch.arange(128 * 2 * 3).reshape(1, 128, 2, 3)
    unpacked = _unpack_latent(packed)
    assert unpacked.shape == (1, 32, 4, 6)
    assert torch.equal(unpacked[:, :, 0::2, 0::2], packed[:, 0::4])


def test_canonical_schedule_has_four_descending_steps() -> None:
    schedule = _sigmas(4, torch.device("cpu"))
    assert schedule.shape == (5,)
    assert schedule[0] == 1
    assert schedule[-1] == 0
    assert torch.all(schedule[:-1] > schedule[1:])


def test_transformer_schema_matches_canonical_checkpoint_shape() -> None:
    state = KleinTransformer().state_dict()
    assert len(state) == 201
    assert state["img_in.weight"].shape == (4096, 128)
    assert state["txt_in.weight"].shape == (4096, 12288)
    assert state["double_blocks.7.img_attn.qkv.weight"].shape == (12288, 4096)
    assert state["single_blocks.23.linear1.weight"].shape == (36864, 4096)
    assert state["final_layer.linear.weight"].shape == (128, 4096)
