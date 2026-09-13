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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA host registration")
def test_cached_weight_registration_released_with_model(tmp_path):
    from comfy_aimdo.torch import hostbuf_to_tensor
    from safetensors.torch import save_file

    from latentslate_engine.krea2.weights import KreaWeights

    weight = torch.eye(16, dtype=torch.bfloat16)
    checkpoint = tmp_path / "weight.safetensors"
    save_file({"0.weight": weight}, checkpoint)
    model = torch.nn.Sequential(Linear(16, 16, bias=False, dtype=torch.bfloat16))
    weights = KreaWeights(checkpoint, model, torch.device("cuda", 0))
    cache = weights.host_cache
    host = hostbuf_to_tensor(cache)
    try:
        value = torch.ones((1, 16), dtype=torch.bfloat16, device="cuda")
        assert torch.equal(model(value), value)
        assert host.is_pinned()
        assert torch.equal(model(value), value)
    finally:
        weights.close()
    assert not host.is_pinned()
    assert cache.size == 0
    assert cache.get_raw_address() == 0


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
        "prompt_suffix": "",
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
        format="float8_e4m3fn",
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
    runtime.conditioning = (("prompt", ""), "expanded", torch.zeros(1))
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Krea mapped CUDA weights")
@pytest.mark.parametrize("representation", ["bf16", "fp8", "int8"])
def test_plain_and_quantized_checkpoints_preserve_reference_norm_precision(tmp_path, representation):
    import json
    from safetensors.torch import save_file
    from latentslate_engine.krea2.model import RMSNorm
    from latentslate_engine.krea2.weights import KreaWeights

    model = torch.nn.Module()
    columns = 256 if representation == "int8" else 16
    model.linear = Linear(columns, 16, bias=False, dtype=torch.bfloat16)
    model.norm = RMSNorm(16, dtype=torch.bfloat16)
    scale = torch.linspace(-0.15, 0.23, 16, dtype=torch.float32)
    tensors = {"linear.weight": torch.eye(16, columns, dtype=torch.bfloat16), "norm.scale": scale}
    metadata = None
    if representation == "fp8":
        tensors["linear.weight"] = tensors["linear.weight"].to(torch.float8_e4m3fn)
        tensors["linear.weight_scale"] = torch.tensor(1.0)
        metadata = {"_quantization_metadata": json.dumps({"layers": {
            "linear": {"format": "float8_e4m3fn", "full_precision_matrix_mult": True}
        }})}
    elif representation == "int8":
        tensors["linear.weight"] = tensors["linear.weight"].to(torch.int8)
        tensors["linear.weight_scale"] = torch.ones(16, 1)
        tensors["linear.comfy_quant"] = torch.tensor(list(json.dumps({
            "format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256,
        }).encode()), dtype=torch.uint8)
    path = tmp_path / "norm.safetensors"
    save_file(tensors, path, metadata=metadata)
    weights = KreaWeights(path, model, torch.device("cuda"))
    try:
        reference_scale = scale if representation == "fp8" else scale.bfloat16()
        x = torch.arange(1, 17, device="cuda", dtype=torch.bfloat16).unsqueeze(0)
        expected = torch.nn.functional.rms_norm(
            x.float(), (16,), reference_scale.cuda().float() + 1.0, eps=1e-5
        ).bfloat16()
        assert torch.equal(model.norm(x), expected)
    finally:
        weights.close()


def test_prompt_suffix_is_appended_after_enhancement_and_invalidates_conditioning(tmp_path, monkeypatch):
    from latentslate_engine.krea2 import runtime as module
    encoded = []
    enhanced = []
    class Encoder:
        def __init__(self, *args):
            pass
        def enhance(self, prompt):
            enhanced.append(prompt)
            return "expanded scene"
        def encode(self, text):
            encoded.append(text)
            return torch.zeros(1)
        def close(self):
            pass
    identity = SimpleNamespace(text_encoder=SimpleNamespace(path=Path("text")), tokenizer=Path("tokenizer"))
    runtime = module.Krea2Runtime(device="cpu")
    runtime.identity = identity
    runtime.model = object()
    runtime.vae = SimpleNamespace(decode=lambda latent: torch.zeros(1, 3, 1, 8, 8))
    monkeypatch.setattr(module, "KreaTextEncoder", Encoder)
    monkeypatch.setattr(module, "sample", lambda *args, **kwargs: torch.zeros(1))
    try:
        first = runtime.generate(identity, "scene", 1, tmp_path / "a.png", prompt_suffix="ink style")
        second = runtime.generate(identity, "scene", 1, tmp_path / "b.png", prompt_suffix="ink style")
        third = runtime.generate(identity, "scene", 1, tmp_path / "c.png", prompt_suffix="anime style")
        assert first.expanded_prompt == "expanded scene, ink style"
        assert second.conditioning_reused
        assert not third.conditioning_reused
        assert third.models_reused
        assert encoded == ["expanded scene, ink style", "expanded scene, anime style"]
        assert enhanced == ["scene", "scene"]
    finally:
        runtime.close()



def test_adapter_pairs_preserve_strength_order_zero_and_immutable_base(tmp_path):
    from safetensors.torch import save_file
    from latentslate_engine.krea2.adapters import load_updates, patch_weight
    paths = []
    for name, up, down in (("a", 1.0, 2.0), ("b", 3.0, 4.0)):
        path = tmp_path / f"{name}.safetensors"
        save_file({
            "transformer.img_in.lora_A.weight": torch.tensor([[down]]),
            "transformer.img_in.lora_B.weight": torch.tensor([[up]]),
        }, path)
        paths.append(SimpleNamespace(path=path))
    modules = {"first": Linear(1, 1, bias=False)}
    updates = load_updates(tuple(zip(paths, (0.5, 1.0))), modules, "cpu")
    assert [strength for _, _, strength in updates["first"]] == [0.5, 1.0]
    base = torch.tensor([[1.0]], dtype=torch.bfloat16)
    assert patch_weight(base, updates["first"]).item() == 14.0
    assert patch_weight(base, updates["first"]).item() == 14.0
    assert base.item() == 1.0
    assert load_updates(((paths[0], 0.0),), modules, "cpu") == {}
    invalid = tmp_path / "unpaired.safetensors"
    save_file({"transformer.img_in.lora_A.weight": torch.ones(1, 1)}, invalid)
    with pytest.raises(ValueError, match="Unsupported or duplicate"):
        load_updates(((SimpleNamespace(path=invalid), 1.0),), modules, "cpu")


def test_fusion_projector_patch_preserves_reference_rounding(tmp_path):
    from safetensors.torch import save_file
    from latentslate_engine.krea2.adapters import load_updates

    path = tmp_path / "projector.safetensors"
    save_file({
        "transformer.text_fusion.projector.lora_A.weight": torch.tensor([[-0.0029]]),
        "transformer.text_fusion.projector.lora_B.weight": torch.ones(1, 1),
    }, path)
    layer = Linear(1, 1, bias=False)
    layer.updates = load_updates(
        ((SimpleNamespace(path=path), 1.0),), {"txtfusion.projector": layer}, "cpu"
    )["txtfusion.projector"]
    base = torch.tensor([[0.37109375]])
    layer.binding = SimpleNamespace(
        name="txtfusion.projector", format=None, materialize=lambda: {"weight": base},
        unpin=lambda: None,
    )
    x = torch.ones(1, 1, dtype=torch.bfloat16)
    assert layer(x).item() == 0.3671875
    assert layer(x).item() == 0.3671875
    assert base.item() == 0.37109375
