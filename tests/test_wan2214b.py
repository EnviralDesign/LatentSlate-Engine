from __future__ import annotations

from pathlib import Path

import av
import pytest
import torch
from comfy_kitchen.tensor import QuantizedTensor

from latentslate_engine.wan2214b.pipeline import (
    WanRecipe,
    WanSession,
    canonical_sigmas,
    save_video,
)
from latentslate_engine.wan2214b.weights import (
    ArtifactIdentity,
    WanWeights,
)


class _Store:
    def __init__(self, values: dict[str, torch.Tensor], name: str):
        self.values = values
        self.keys = frozenset(values)
        self.identity = ArtifactIdentity(name, 1, 1)

    def tensor(self, key: str) -> torch.Tensor:
        return self.values[key]


def _small_weights(prefix: str, device: torch.device) -> WanWeights:
    base_float = torch.tensor([[0.25, -0.5], [0.75, 1.0]], dtype=torch.float16)
    scale = base_float.abs().max().float() / 448
    qdata = (base_float / scale.to(base_float.dtype)).to(torch.float8_e4m3fn)
    lora_prefix = f"diffusion_model.{prefix}"
    weights = object.__new__(WanWeights)
    weights.base = _Store(
        {f"{prefix}.weight": qdata, f"{prefix}.scale_weight": scale}, "base"
    )
    weights.lora = _Store(
        {
            f"{lora_prefix}.lora_up.weight": torch.tensor([[1.0], [2.0]]),
            f"{lora_prefix}.lora_down.weight": torch.tensor([[0.5, -0.25]]),
            f"{lora_prefix}.alpha": torch.tensor(1.0),
        },
        "lora",
    )
    weights.lora_strength = 1.0
    weights.native_fp8 = True
    weights._patched_weights = {}
    weights._active_weights = {}
    weights._active_qk_norms = {}
    weights._active_device = device
    weights._base_reopened = False
    weights._materialized_since_reopen = 0
    weights._reopen_before_next_access = False
    weights._prefetch_stream = None
    weights._prefetched_live = None
    return weights


def test_canonical_recipe_and_schedule_are_fixed() -> None:
    recipe = WanRecipe()
    assert recipe.frame_count == 81
    assert (recipe.steps, recipe.split_step, recipe.cfg, recipe.shift) == (
        4,
        2,
        1.0,
        5.000000000000001,
    )
    assert (recipe.width, recipe.height, recipe.duration, recipe.fps) == (
        512,
        512,
        5.0,
        16.0,
    )
    torch.testing.assert_close(
        canonical_sigmas(recipe.shift, recipe.steps),
        torch.tensor([1.0, 0.9375, 0.8333333134651184, 0.625, 0.0]),
        rtol=0,
        atol=0,
    )


def test_recipe_identity_consumes_both_models_loras_and_strengths(
    tmp_path: Path,
) -> None:
    paths = []
    for name in ("high", "high-lora", "low", "low-lora", "text", "vae"):
        path = tmp_path / f"{name}.safetensors"
        path.write_bytes(name.encode())
        paths.append(str(path))
    recipe = WanRecipe(*paths)
    identity = recipe.identity

    changed = WanRecipe(*paths, high_lora_strength=0.5)
    assert changed.identity != identity

    Path(paths[3]).write_bytes(b"changed low lora")
    assert recipe.identity != identity


def test_live_lora_rebuilds_from_immutable_base_without_accumulation() -> None:
    prefix = "blocks.0.self_attn.q"
    weights = _small_weights(prefix, torch.device("cpu"))
    original = weights.base.tensor(f"{prefix}.weight").clone()

    first = weights._patched_weight(prefix, torch.device("cpu"), torch.float16)
    second = weights._patched_weight(prefix, torch.device("cpu"), torch.float16)

    assert isinstance(first, torch.Tensor)
    assert isinstance(second, torch.Tensor)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(weights.base.tensor(f"{prefix}.weight"), original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FP8 is required")
def test_materialized_lora_cache_is_stable_across_phase_reactivation() -> None:
    prefix = "blocks.3.ffn.0"
    device = torch.device("cuda")
    weights = _small_weights(prefix, device)
    weights._active_device = device

    first = weights._patched_weight(prefix, device, torch.float16)
    assert isinstance(first, QuantizedTensor)
    first_qdata = first._qdata.clone()
    weights.deactivate()
    weights.activate(device)
    second = weights._patched_weight(prefix, device, torch.float16)

    assert isinstance(second, QuantizedTensor)
    assert torch.equal(first_qdata, second._qdata)


def test_destroyed_session_is_unusable() -> None:
    session = object.__new__(WanSession)
    session._alive = True
    session._identity = ("recipe",)
    session._conditioning = (torch.zeros(1), torch.zeros(1))
    session._token_data = {"ids": torch.zeros(1)}
    session._vae = object()
    session.high_weights = object()
    session.low_weights = object()
    session.text_weights = object()

    session.destroy()

    assert session._conditioning is None
    assert session.high_weights is None
    with pytest.raises(RuntimeError, match="destructively replaced"):
        _ = session.identity


def test_video_writer_emits_canonical_media_metadata(tmp_path: Path) -> None:
    path = tmp_path / "wan.mp4"
    images = torch.linspace(0, 1, 2 * 16 * 16 * 3).reshape(2, 16, 16, 3)
    save_video(images, path, 16.0)

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.codec_context.format.name == "yuv420p"
        assert stream.average_rate == 16
        assert stream.frames == 2
