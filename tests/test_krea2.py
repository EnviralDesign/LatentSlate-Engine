"""Small regressions for the measured Krea Turbo boundaries."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

from latentslate_engine.krea2.contracts import validate_request
from latentslate_engine.krea2.recipes import krea2_t2i_recipe, resolve_krea2_request
from latentslate_engine.krea2.sampling import noise, sigmas
from latentslate_engine.krea2.weights import Linear

pytestmark = pytest.mark.native


def test_noise_and_schedule_match_frozen_comfy_oracle():
    state = torch.get_rng_state().clone()
    value = noise(594361197674106, 1024, 1024)
    assert (
        hashlib.sha256(value.numpy().tobytes()).hexdigest()
        == "fa127c93b1f26bab6330802ceb8c104191e0d6bd6b15e37a5011159e988dfc03"
    )
    assert torch.equal(state, torch.get_rng_state())
    assert sigmas().tolist() == [
        1.0,
        0.956723690032959,
        0.9045307636260986,
        0.8403487801551819,
        0.7595109343528748,
        0.6545668244361877,
        0.5128440856933594,
        0.31090107560157776,
        0.0,
    ]


def test_certified_recipe_preserves_eight_aligned_landscape():
    recipe = krea2_t2i_recipe(
        **{key: Path(key) for key in ("diffusion", "text_encoder", "vae", "tokenizer")}
    )
    request = resolve_krea2_request(
        recipe, {"prompt": "A glass", "width": 1368, "height": 768, "seed": 2**64 - 1}
    )
    assert request == {
        "prompt": "A glass",
        "width": 1368,
        "height": 768,
        "seed": 2**64 - 1,
    }
    assert {field["key"] for field in recipe.surface()} == {
        "prompt",
        "width",
        "height",
        "seed",
    }
    with pytest.raises(ValueError):
        resolve_krea2_request(recipe, {"prompt": "A glass", "steps": 4})


@pytest.mark.parametrize(
    "width,height,seed",
    [
        (1023, 1024, 0),
        (2048, 2048, 0),
        (True, 768, 0),
        (768, 768, True),
        (768, 768, 2**64),
    ],
)
def test_invalid_requests_fail_before_loading(width, height, seed):
    with pytest.raises((TypeError, ValueError)):
        validate_request(width, height, seed)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP8 regression requires CUDA"
)
def test_missing_fp8_input_scale_means_one():
    generator = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(32, 16, device="cuda", generator=generator).mul(3).bfloat16()
    raw = torch.randn(16, 16, device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    scale = torch.tensor(0.3, device="cuda")
    unpinned = []
    layer = Linear(16, 16, bias=False)
    layer.binding = SimpleNamespace(
        full_precision=False,
        materialize=lambda: {"weight": raw, "weight_scale": scale},
        unpin=lambda: unpinned.append(True),
    )
    weight = QuantizedTensor(
        raw,
        "TensorCoreFP8Layout",
        TensorCoreFP8Layout.Params(
            scale=scale, orig_dtype=x.dtype, orig_shape=raw.shape
        ),
    )
    expected = torch.nn.functional.linear(
        QuantizedTensor.from_float(x, "TensorCoreFP8Layout", scale=1.0), weight
    )
    assert torch.equal(layer(x), expected)
    assert unpinned == [True]


@pytest.mark.parametrize("fail", [False, True])
def test_generation_restores_process_math_precision(tmp_path, monkeypatch, fail):
    from latentslate_engine.krea2 import runtime as module
    runtime = module.Krea2Runtime(device="cpu")
    identity = object()
    runtime.identity = identity
    runtime.model = object()
    runtime.conditioning = ("prompt", "expanded", torch.zeros(1))
    runtime.vae = SimpleNamespace(decode=lambda latent: torch.zeros(1, 3, 1, 8, 8))
    def sample(*args, **kwargs):
        assert torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        if fail:
            raise RuntimeError("sample failure")
        return torch.zeros(1)
    monkeypatch.setattr(module, "sample", sample)
    previous = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    try:
        if fail:
            with pytest.raises(RuntimeError, match="sample failure"):
                runtime.generate(identity, "prompt", 0, tmp_path / "output.png")
            assert runtime.identity is None
        else:
            runtime.generate(identity, "prompt", 0, tmp_path / "output.png")
        assert not torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    finally:
        runtime.close()
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous)
