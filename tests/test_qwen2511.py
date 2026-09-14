"""Observable regressions for the curated native Qwen image-edit contract."""

import hashlib
import os
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from latentslate_engine.qwen2511.preprocessing import picture_prompt
from latentslate_engine.qwen2511.sampling import noise, sigmas


def test_curated_noise_and_schedule_match_frozen_oracle():
    def digest(tensor):
        return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

    assert digest(noise(677909188488042, 1024, 1024)) == "94240e1549a80dc6cf13d890f298c66d8f2264becd87b0add93565a73827398f"
    assert digest(sigmas()) == "cbc9f93fdf8a08db577f6629bfd25ea5bdd628b0adb3afb815801537b966345e"


def test_sparse_reference_prompt_preserves_logical_slots():
    assert picture_prompt("Edit", (1, 3)) == (
        "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
        "Picture 3: <|vision_start|><|image_pad|><|vision_end|>Edit"
    )


def test_curated_recipe_exposes_only_request_inputs(tmp_path):
    from latentslate_engine.qwen2511.recipes import qwen2511_edit_recipe, resolve_qwen2511_request

    recipe = qwen2511_edit_recipe(diffusion="diffusion", text_encoder="text", vae="vae", tokenizer="tokenizer")
    values = resolve_qwen2511_request(recipe, {"prompt": "Edit", "image_1": "one.png", "image_3": "three.png"})
    assert values == dict(prompt="Edit", image_1="one.png", image_2=None, image_3="three.png", seed=0, steps=40, cfg=4.0, shift=3.1)
    for forbidden, value in (("width", 512), ("steps", 4), ("cfg", 1), ("adapters", ())):
        with pytest.raises(ValueError):
            resolve_qwen2511_request(recipe, {"prompt": "Edit", "image_1": "one.png", forbidden: value})
    with pytest.raises(ValueError, match="image_1"):
        resolve_qwen2511_request(recipe, {"prompt": "Edit"})


def test_runtime_reuses_and_invalidates_consumed_inputs(monkeypatch, tmp_path):
    from latentslate_engine.qwen2511 import runtime as module

    calls = []

    class VAE:
        def encode(self, pixels):
            calls.append(("vae_encode", tuple(pixels.shape)))
            return torch.zeros(1, 16, 1, pixels.shape[1] // 8, pixels.shape[2] // 8)

        def decode(self, latent):
            return torch.zeros(1, 3, 1, latent.shape[-2] * 8, latent.shape[-1] * 8)

    class TextEncoder:
        def __init__(self, *args):
            calls.append(("text_load",))

        def encode(self, prompt, images, slots):
            calls.append(("text_encode", prompt, slots))
            return torch.zeros(1, 1, 3584)

        def offload(self):
            pass

        def close(self):
            calls.append(("text_close",))

    class Model:
        def __init__(self, **kwargs):
            calls.append(("model_load",))

        def eval(self):
            return self

        def requires_grad_(self, value):
            return self

    class Weights:
        def __init__(self, *args):
            pass

        def close(self):
            calls.append(("weights_close",))

    def sample(model, positive, negative, references, seed, width, height, *args, **kwargs):
        calls.append(("sample", seed, width, height))
        return torch.zeros(1, 16, 1, height // 8, width // 8)

    monkeypatch.setattr(module, "load_vae", lambda *args: VAE())
    monkeypatch.setattr(module, "QwenTextEncoder", TextEncoder)
    monkeypatch.setattr(module, "QwenImageTransformer2DModel", Model)
    monkeypatch.setattr(module, "QwenWeights", Weights)
    monkeypatch.setattr(module, "sample", sample)
    one, three = tmp_path / "one.png", tmp_path / "three.png"
    Image.new("RGB", (64, 64), "red").save(one)
    Image.new("RGB", (32, 64), "blue").save(three)
    identity = SimpleNamespace(
        diffusion=SimpleNamespace(path=tmp_path / "diffusion"),
        text_encoder=SimpleNamespace(path=tmp_path / "text"),
        vae=SimpleNamespace(path=tmp_path / "vae"), tokenizer=tmp_path,
        adapters=(),
    )
    runtime = module.Qwen2511Runtime("cpu")

    def generate(prompt="Edit", seed=1, image_3=None):
        return runtime.generate(identity, prompt, seed, tmp_path / "result.png", image_1=one, image_3=image_3)

    first = generate()
    original_bytes = first.output.read_bytes()
    assert not first.models_reused and not first.references_reused
    second = generate(seed=2)
    assert second.models_reused and second.references_reused and second.positive_reused and second.negative_reused
    changed_prompt = generate("Different")
    assert changed_prompt.references_reused and not changed_prompt.positive_reused and changed_prompt.negative_reused
    sparse = generate(image_3=three)
    assert sparse.models_reused and not sparse.references_reused
    assert sparse.reference_slots_reused == (1,)
    assert ("text_encode", "Edit", (1, 3)) in calls
    assert (sparse.width, sparse.height) == (1024, 1024)

    before = three.stat()
    Image.new("RGB", (32, 64), "green").save(three)
    os.utime(three, ns=(before.st_atime_ns, before.st_mtime_ns))
    changed_reference = generate(image_3=three)
    assert not changed_reference.references_reused and not changed_reference.negative_reused
    assert changed_reference.reference_slots_reused == (1,)
    returned = generate()
    assert returned.models_reused and returned.output.read_bytes() == original_bytes
    assert calls.count(("model_load",)) == 1
    assert calls.count(("text_load",)) == 1
    assert len([call for call in calls if call[0] == "sample"]) == 6

    runtime.ensure_identity("different identity")
    assert runtime.model is None and runtime.vae is None and runtime.text_encoder is None
    assert runtime.references is None and runtime.positive is None and runtime.negative is None
    assert calls.count(("weights_close",)) == 1 and calls.count(("text_close",)) == 1
    runtime.close()


def test_missing_image_rejected_before_native_loading(tmp_path):
    from latentslate_engine.qwen2511.runtime import Qwen2511Runtime

    runtime = Qwen2511Runtime("cpu")
    with pytest.raises(ValueError, match="requires image_1"):
        runtime.generate(None, "Edit", 1, tmp_path / "out.png", image_1=None)
    assert runtime.identity is None and runtime.model is None


def test_fp8_adapter_residency_and_eviction_preserve_source(monkeypatch):
    from latentslate_engine.qwen2511 import weights as module

    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn)
    original = source.float().clone()

    class Binding:
        name = "transformer_blocks.0.attn.to_q"
        resident = False
        signature = object()

        def __init__(self):
            self.values = {"weight": source.clone(), "weight_scale": torch.tensor(1.0)}

        def materialize(self):
            return self.values

        def unpin(self):
            pass

    binding = Binding()
    linear = module.Linear(2, 2, bias=False)
    linear.binding = binding
    linear.updates = ((torch.ones(2, 1), torch.ones(1, 2), 0.2),)
    # A deliberately coarse rounded payload makes reuse distinguishable from
    # reapplying the delta, independently of Kitchen's CUDA rounding kernel.
    monkeypatch.setattr(module, "requantize_fp8", lambda weight, name: (
        weight.round().to(torch.float8_e4m3fn), torch.tensor(1.0),
    ))
    x = torch.eye(2)
    first = linear(x)
    assert torch.equal(first, (original + 0.2).T)
    binding.resident = True
    assert torch.equal(linear(x), original.T)
    assert torch.equal(linear(x), original.T)
    binding.resident = False
    binding.values["weight"].copy_(source)
    assert torch.equal(linear(x), first)
    assert torch.equal(source.float(), original)
    binding.signature = None
    binding.values["weight"].copy_(source)
    assert torch.equal(linear(x), first)
    assert torch.equal(linear(x), first)


def test_qwen_adapter_consumes_complete_pairs_and_alpha(tmp_path):
    from safetensors.torch import save_file
    from latentslate_engine.qwen2511.adapters import load_updates, patch_weight

    path = tmp_path / "adapter.safetensors"
    tensors = {
        "layer.lora_down.weight": torch.ones(2, 3),
        "layer.lora_up.weight": torch.ones(4, 2),
        "layer.alpha": torch.tensor(0.5),
    }
    save_file(tensors, path)
    adapter = ((SimpleNamespace(path=path), 1.0),)
    modules = {"layer": SimpleNamespace(in_features=3, out_features=4)}
    updates = load_updates(adapter, modules, "cpu")
    source = torch.zeros(4, 3, dtype=torch.bfloat16)
    assert torch.equal(patch_weight(source, updates["layer"]), torch.full_like(source, 0.5))
    assert torch.count_nonzero(source) == 0
    tensors["unknown"] = torch.ones(1)
    save_file(tensors, path)
    with pytest.raises(ValueError, match="Unsupported Qwen adapter tensors"):
        load_updates(adapter, modules, "cpu")
