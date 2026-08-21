from __future__ import annotations

import hashlib
import inspect
import json
import os
import tomllib
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import MappingProxyType

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from latentslate_engine import resources
from latentslate_engine import z_image_turbo_recipe as contract
from latentslate_engine.artifacts import probe_artifact
from latentslate_engine.config import Settings
from latentslate_engine.runtime import z_image_conditioning as conditioning
from latentslate_engine.runtime import z_image_qwen_architecture as qwen_architecture
from latentslate_engine.runtime import z_image_qwen_checkpoint as qwen_checkpoint
from latentslate_engine.runtime import z_image_qwen_runtime as qwen_runtime
from latentslate_engine.runtime import z_image_sampler as sampler
from latentslate_engine.runtime import z_image_stored_adapter as stored_adapter
from latentslate_engine.runtime.z_image_nextdit import (
    ZImageNextDiTConfig,
    ZImageNextDiTShell,
    _z_image_nextdit_stored_bias_keys,
    build_z_image_nextdit_shell,
)
from latentslate_engine.runtime.z_image_turbo import (
    ZImagePhase,
    ZImageTurboCancelled,
    ZImageTurboLifecycle,
    z_image_initial_noise,
)
from latentslate_engine.runtime.z_image_vae import (
    ZImagePngPublicationCancelled,
    _map_flux_ae_key,
    decode_z_image_flux_ae,
    write_z_image_png_atomic,
)
from latentslate_engine.tools.z_image_turbo import ZImageTurboTextToImageTool
from latentslate_engine.z_image_turbo_recipe import (
    ZImageDensePlan,
    ZImageTurboRecipe,
    ZImageTurboRuntimeRequest,
    _plan_transformer,
    z_image_turbo_schedule,
)

_INSTALLED_Z_IMAGE_TURBO_ROOT = Path(
    r"M:\LatentSlateEngineData\models\zimage\comfy-org-z-image-turbo"
)


def _marker() -> torch.Tensor:
    return torch.tensor(
        list(
            json.dumps(
                {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": 256,
                    "per_row": True,
                }
            ).encode()
        ),
        dtype=torch.uint8,
    )


def _save_transformer(
    path: Path, *, include_adaln_bias: bool = False, include_extra_stored_bias: bool = False
) -> None:
    tensors: dict[str, torch.Tensor] = {}
    stems = [
        *(f"layers.{index}.attention.qkv" for index in range(34)),
        *(f"layers.{index}.attention.out" for index in range(34)),
        *(f"layers.{index}.feed_forward.w1" for index in range(34)),
        *(f"layers.{index}.feed_forward.w2" for index in range(34)),
        *(f"layers.{index}.feed_forward.w3" for index in range(34)),
        *(f"layers.{index}.adaLN_modulation.0" for index in range(32)),
    ]
    assert len(stems) == 202
    for stem in stems:
        tensors[stem + ".weight"] = torch.zeros((1, 256), dtype=torch.int8)
        tensors[stem + ".weight_scale"] = torch.tensor([[0.25]])
        tensors[stem + ".comfy_quant"] = _marker()
        if include_adaln_bias and stem.endswith("adaLN_modulation.0"):
            tensors[stem + ".bias"] = torch.zeros(1, dtype=torch.float32)
    if include_extra_stored_bias:
        tensors["layers.0.attention.qkv.bias"] = torch.zeros(1, dtype=torch.float32)
    # The upstream header uses empty global metadata; every ConvRot fact comes
    # from the exact per-layer U8 marker payload.
    save_file(tensors, path, metadata={})


def test_official_base_and_exact_fixed_lora_catalog_contracts():
    root = Path(__file__).parents[1] / "src/latentslate_engine"
    recipe = tomllib.loads(
        (
            root / "builtin_recipes/zimage/z-image-turbo-text-to-image-comfy-int8-convrot.toml"
        ).read_text()
    )["runnable_recipe"]
    assert recipe["family"] == "zimage"
    assert recipe["recipe"]["type"] == "z_image_turbo_t2i"
    assert recipe["recipe"]["operation"] == "zimage_turbo_t2i_int8_convrot"
    assert "recommended" in recipe["tags"]
    assert "engine-native" in recipe["tags"]
    assert "experimental" not in recipe["tags"]
    assert "Native GPU acceptance is pending" not in recipe["description"]
    assert "fixed" not in recipe
    assert [item.key for item in ZImageTurboTextToImageTool().descriptor.inputs] == [
        "prompt",
        "seed",
    ]
    declarations = list((root / "builtin_resource_declarations").glob("zimage-*.toml"))
    assert len(declarations) == 5
    sources = [tomllib.loads(path.read_text())["resource"]["sources"][0] for path in declarations]
    assert {source["repo_id"] for source in sources} == {
        "Comfy-Org/z_image_turbo",
        "Kutches/ImageZV2",
        "Tongyi-MAI/Z-Image-Turbo",
    }
    assert all(len(source["revision"]) == 40 for source in sources)
    assert sum("sha256" in source for source in sources) == 4
    resources_by_path = [tomllib.loads(path.read_text())["resource"] for path in declarations]
    assert sum("recommended" in resource["tags"] for resource in resources_by_path) == 4
    assert all("engine-native" in resource["tags"] for resource in resources_by_path)
    lora = next(resource for resource in resources_by_path if resource["kind"] == "lora")
    assert lora["default_strength"] == 1.0
    assert lora["id"] == "lora:zimage:kutches--imagezv2/70s-horror-movie-b"
    assert lora["metadata"]["license"] == "not-declared-upstream"
    assert lora["metadata"]["target_count"] == 240
    # Bounded real-header facts captured from the immutable revisions, not a
    # mutable filename heuristic. The synthetic files below preserve exactly
    # these counts/marker lengths while staying small enough for CI.
    assert (
        contract._Z_TRANSFORMER_HEADER_SHA256
        == "01e93cae3aa75eb2106025889f1a78df19628a95c433b45d9447562b04907814"
    )
    assert (
        qwen_checkpoint.QWEN_HEADER_SHA256
        == "7537b0cd31f4fc963d334b4f997cedee6f51c62aa8518b7b7a852b182144aed9"
    )


def test_fixed_lora_declaration_resolves_from_managed_lora_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    home = tmp_path / "LatentSlateEngineData"
    settings = Settings(
        home=home,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    artifact = settings.lora_root / "zimage/Kutches--ImageZV2/70s-Horror-Movie-b.safetensors"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fixture")

    original_artifact_complete = resources._artifact_complete

    def artifact_complete(path, resource, *, verification_cache_root=None):
        if path.resolve() == artifact.resolve():
            return True
        return original_artifact_complete(
            path,
            resource,
            verification_cache_root=verification_cache_root,
        )

    monkeypatch.setattr(resources, "_artifact_complete", artifact_complete)

    inventory = resources.discover_resources(settings)
    resource_id = "lora:zimage:kutches--imagezv2/70s-horror-movie-b"
    descriptor = inventory.resolve(resource_id)

    assert inventory.path_for(resource_id).resolve() == artifact.resolve()
    assert descriptor.relative_path == (
        "loras/zimage/Kutches--ImageZV2/70s-Horror-Movie-b.safetensors"
    )
    assert descriptor.available is True
    assert not [error for error in inventory.errors if "70s-Horror-Movie-b" in error]


def test_prompt_template_and_token_ids_match_pinned_first_party_z_image_bytes():
    class FirstPartyZImageTokenizerFixture:
        """Byte IDs make the token comparison independent of chat-template behavior."""

        def __call__(self, text, **kwargs):
            assert kwargs == {
                "add_special_tokens": False,
                "padding": False,
                "truncation": False,
                "return_tensors": "pt",
            }
            return {
                "input_ids": torch.tensor([[*text.encode("utf-8")]], dtype=torch.int64),
                "attention_mask": torch.ones((1, len(text.encode("utf-8"))), dtype=torch.int64),
            }

    prompt = "a glass kiwi"
    expected_text = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    assert (
        conditioning.Z_IMAGE_PROMPT_TEMPLATE
        == "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
    assert conditioning.z_image_prompt_envelope(prompt) == expected_text
    tokenizer = FirstPartyZImageTokenizerFixture()
    actual = conditioning.tokenize_z_image_prompt(tokenizer, prompt)
    expected_ids = torch.tensor([[*expected_text.encode("utf-8")]], dtype=torch.int64)
    assert torch.equal(actual.input_ids, expected_ids)
    assert torch.equal(actual.attention_mask, torch.ones_like(expected_ids))


def test_prompt_has_no_fixed_truncation_and_process_tokens_mask_matches_source():
    observed = {}

    class LongTokenizer:
        def __call__(self, _text, **kwargs):
            observed.update(kwargs)
            return {
                "input_ids": torch.tensor([[151643, 7, 8, 151643, 9]], dtype=torch.int64),
                "attention_mask": torch.ones((1, 5), dtype=torch.int64),
            }

    tokenized = conditioning.tokenize_z_image_prompt(LongTokenizer(), "long")
    assert tokenized.attention_mask.tolist() == [[0, 1, 1, 0, 0]]
    assert observed["truncation"] is False
    assert "max_length" not in observed


def test_z_image_pipeline_support_requires_exact_four_file_architecture_closure(tmp_path: Path):
    expected = {
        "text_encoder/config.json",
        "tokenizer/merges.txt",
        "tokenizer/vocab.json",
        "tokenizer/tokenizer_config.json",
    }
    for relative in expected:
        candidate = tmp_path / relative
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("x")
    metadata = {"architecture": "z_image_turbo_pipeline_support"}
    assert resources._has_pipeline_support_files(tmp_path, "zimage", metadata)
    assert not resources._has_pipeline_support_files(tmp_path, "zimage", {})
    (tmp_path / "unexpected.json").write_text("x")
    assert not resources._has_pipeline_support_files(tmp_path, "zimage", metadata)


def test_support_declaration_is_exactly_the_runtime_four_file_closure():
    declaration = (
        Path(__file__).parents[1] / "src/latentslate_engine/builtin_resource_declarations/"
        "zimage-tongyi-mai-turbo-pipeline-support.toml"
    )
    resource = tomllib.loads(declaration.read_text())["resource"]
    source = resource["sources"][0]
    expected = contract._Z_PIPELINE_SUPPORT_FILES
    assert resource["size_bytes"] == sum(size for size, _digest in expected.values())
    assert resource["metadata"]["architecture"] == "z_image_turbo_pipeline_support"
    assert source["revision"] == contract._IMMUTABLE_COMPONENTS["pipeline_support"][1]
    assert set(source["allow_patterns"]) == set(expected)
    assert len(source["allow_patterns"]) == len(expected) == 4


def test_saved_res_multistep_contract_is_executable_and_exact():
    fixture = sampler.z_image_sampler_contract()
    assert fixture["kind"] == "deterministic_res_multistep" and fixture["executable"] is True
    assert fixture["sampler"] == "res_multistep"
    assert fixture["ancestral_eta"] == 0.0
    assert fixture["guider"] == "basic"
    assert "guidance_scale" not in fixture
    assert "negative_conditioning" not in fixture
    assert fixture["sigmas"] == sampler.Z_IMAGE_AURAFLOW_SHIFT_3_SIGMAS


def test_saved_schedule_uses_exact_1000_point_fp32_index_lookup():
    base = torch.arange(1, 1001, dtype=torch.float32) / 1000
    shifted = (3.0 * base) / (1.0 + 2.0 * base)
    expected = torch.cat(
        (shifted[torch.tensor([999, 874, 749, 624, 499, 374, 249, 124])], torch.zeros(1))
    )
    actual = sampler.z_image_auraflow_shift_3_sigma_tensor()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert actual.tolist() == [
        1.0,
        0.9545454382896423,
        0.8999999761581421,
        0.8333333134651184,
        0.75,
        0.6428571343421936,
        0.5,
        0.30000001192092896,
        0.0,
    ]


def test_initial_noise_is_cpu_fp32_seeded_then_transferred(monkeypatch):
    original_generator = torch.Generator
    original_randn = torch.randn
    seen = []

    def generator(*args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        seen.append(("generator", str(device)))
        return original_generator(*args, **kwargs)

    def randn(*args, **kwargs):
        seen.append(("randn", str(kwargs.get("device")), kwargs.get("dtype")))
        return original_randn(*args, **kwargs)

    expected = original_randn(
        (1, 16, 2, 2),
        generator=original_generator(device="cpu").manual_seed(123),
        device="cpu",
        dtype=torch.float32,
    )
    monkeypatch.setattr(torch, "Generator", generator)
    monkeypatch.setattr(torch, "randn", randn)
    actual = z_image_initial_noise(123, height=16, width=16, device="cpu")
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert seen == [("generator", "cpu"), ("randn", "cpu", torch.float32)]


def test_fused_nextdit_forward_is_cpu_testable_without_splitting_qkv():
    config = ZImageNextDiTConfig(
        in_channels=2,
        dim=12,
        layers=1,
        refiner_layers=1,
        heads=2,
        kv_heads=2,
        cap_dim=4,
        axes_dims=(2, 2, 2),
        axes_lens=(128, 128, 128),
    )
    torch.manual_seed(4)
    shell = ZImageNextDiTShell(config).eval()
    output = shell(
        torch.randn((1, 2, 4, 4)),
        torch.tensor([0.75]),
        torch.randn((1, 3, 4)),
    )
    assert output.shape == (1, 2, 4, 4)
    assert torch.isfinite(output).all()
    assert shell.layers[0].attention.qkv.out_features == 36
    seam = shell.forward_contract()
    assert seam["executable"] is True and seam["source_sequence_order"] == "caption_then_image"
    assert seam["image_output_offset"] == "cap_size"


def test_nextdit_timestep_is_fp32_and_caption_prefix_is_excluded_from_output():
    config = ZImageNextDiTConfig(
        in_channels=1,
        dim=4,
        layers=0,
        refiner_layers=0,
        heads=1,
        kv_heads=1,
        cap_dim=4,
        axes_dims=(2, 1, 1),
        axes_lens=(128, 128, 128),
    )
    shell = ZImageNextDiTShell(config).eval()
    shell.x_embedder = nn.Identity()
    shell.cap_embedder = nn.Identity()

    class TimestepSpy(nn.Module):
        def __init__(self):
            super().__init__()
            self.observed = None

        def forward(self, value):
            self.observed = value.detach().clone()
            return torch.zeros((len(value), 4), dtype=torch.float32)

    class FinalIdentity(nn.Module):
        def forward(self, value, _condition):
            return value

    spy = TimestepSpy()
    shell.t_embedder = spy
    shell.final_layer = FinalIdentity()
    latents = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    captions = torch.full((1, 3, 4), -100.0)
    output = shell(latents, torch.tensor([0.75], dtype=torch.float64), captions)
    assert spy.observed.dtype is torch.float32
    assert spy.observed.item() == 250.0
    assert torch.equal(output, -latents)


def test_fused_nextdit_matches_independent_apache_split_reference():
    from diffusers import ZImageTransformer2DModel

    config = ZImageNextDiTConfig(
        in_channels=2,
        dim=12,
        layers=1,
        refiner_layers=1,
        heads=2,
        kv_heads=2,
        cap_dim=4,
        axes_dims=(2, 2, 2),
        axes_lens=(128, 128, 128),
    )
    torch.manual_seed(91)
    fused = ZImageNextDiTShell(config).eval()
    split = ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=2,
        dim=12,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        cap_feat_dim=4,
        axes_dims=[2, 2, 2],
        axes_lens=[128, 128, 128],
    ).eval()
    mapped = {}
    for key, value in fused.state_dict().items():
        destination = key
        destination = destination.replace("x_embedder.", "all_x_embedder.2-1.")
        destination = destination.replace("final_layer.", "all_final_layer.2-1.")
        destination = destination.replace(".attention.out.", ".attention.to_out.0.")
        destination = destination.replace(".attention.q_norm.", ".attention.norm_q.")
        destination = destination.replace(".attention.k_norm.", ".attention.norm_k.")
        if ".attention.qkv.weight" in key:
            query, key_weight, value_weight = value.chunk(3, dim=0)
            mapped[destination.replace("qkv", "to_q")] = query
            mapped[destination.replace("qkv", "to_k")] = key_weight
            mapped[destination.replace("qkv", "to_v")] = value_weight
        else:
            mapped[destination] = value
    split.load_state_dict(mapped, strict=True)
    latents = torch.randn((1, 2, 4, 4))
    captions = torch.randn((1, 3, 4))
    sigma = torch.tensor([0.75])
    actual = fused(latents, sigma, captions)
    reference = split(
        [latents[0].unsqueeze(1)],
        1.0 - sigma,
        [captions[0]],
        return_dict=False,
    )[0][0].squeeze(1)
    assert torch.allclose(actual[0], -reference, rtol=1e-5, atol=1e-5)


def test_res_multistep_normalized_golden_fixture_and_boundaries():
    calls = []
    progress = []

    def denoiser(value, sigma):
        calls.append(float(sigma[0]))
        return value * 0.25 + sigma.reshape(-1, 1)

    result = sampler.z_image_res_multistep(
        denoiser,
        torch.tensor([[0.25, -0.5]], dtype=torch.float64),
        progress=progress.append,
    )
    assert result[0].tolist() == pytest.approx([0.4273414701277809, 0.35192826951699596])
    assert calls == pytest.approx(sampler.Z_IMAGE_AURAFLOW_SHIFT_3_SIGMAS[:-1])
    assert [step.method for step in progress] == [
        "euler",
        "res_second_order",
        "res_second_order",
        "res_second_order",
        "res_second_order",
        "res_second_order",
        "res_second_order",
        "euler",
    ]
    assert all(step.sigma == pytest.approx(calls[step.index]) for step in progress)


def test_conditioning_uses_fp32_penultimate_hidden_and_no_negative_branch():
    class Tokenizer:
        def __call__(self, text, **_kwargs):
            length = len(text.encode())
            return {
                "input_ids": torch.arange(length).reshape(1, -1),
                "attention_mask": torch.ones((1, length), dtype=torch.int64),
            }

    class Encoder(nn.Module):
        def forward_conditioning(self, input_ids, attention_mask, **_kwargs):
            assert input_ids.dtype is torch.int64
            assert attention_mask.dtype is torch.int64
            assert attention_mask.ndim == 2
            return torch.full((*input_ids.shape, 2560), 3.0, dtype=torch.float32)

    encoded = conditioning.encode_z_image_prompt(Encoder(), Tokenizer(), "glass kiwi", device="cpu")
    assert torch.count_nonzero(encoded.positive != 3.0) == 0
    assert encoded.positive.dtype is torch.float32
    assert encoded.attention_mask.dtype is torch.int64
    assert not hasattr(encoded, "negative")
    assert encoded.token_count == encoded.positive.shape[1]


def test_atomic_png_reports_observed_dimensions_hash_and_size(tmp_path: Path):
    from PIL import Image

    artifact = write_z_image_png_atomic(Image.new("RGB", (7, 5), (1, 2, 3)), tmp_path / "x.png")
    assert (artifact.width, artifact.height) == (7, 5)
    assert artifact.size_bytes == artifact.path.stat().st_size
    assert artifact.sha256 == hashlib.sha256(artifact.path.read_bytes()).hexdigest()
    assert not list(tmp_path.glob("*.tmp"))

    prior = artifact.path.read_bytes()
    with pytest.raises(ValueError, match="dimensions differ"):
        write_z_image_png_atomic(Image.new("RGB", (8, 5)), artifact.path, expected_size=(7, 5))
    assert artifact.path.read_bytes() == prior
    assert not list(tmp_path.glob("*.tmp"))

    with pytest.raises(ZImagePngPublicationCancelled, match="before atomic PNG publication"):
        write_z_image_png_atomic(
            Image.new("RGB", (7, 5), (9, 8, 7)), artifact.path, cancelled=lambda: True
        )
    assert artifact.path.read_bytes() == prior
    assert not list(tmp_path.glob("*.tmp"))


def test_flux_ae_decode_uses_exact_scale_shift_and_cancellation():
    class TinyVae(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.observed = None

        def decode(self, value, return_dict=False):
            assert return_dict is False
            self.observed = value.detach().clone()
            return (torch.zeros((value.shape[0], 3, 4, 6), device=value.device),)

    model = TinyVae()
    latents = torch.full((1, 16, 1, 1), 0.3611)
    images = decode_z_image_flux_ae(model, latents)
    assert len(images) == 1 and images[0].size == (6, 4)
    assert torch.allclose(model.observed, torch.full_like(model.observed, 1.1159), atol=1e-6)
    with pytest.raises(RuntimeError, match="before VAE"):
        decode_z_image_flux_ae(model, latents, cancelled=lambda: True)


def test_z_image_vae_uses_its_own_flux_ae_mapping_contract():
    assert _map_flux_ae_key("encoder.norm_out.weight") == "encoder.conv_norm_out.weight"
    assert _map_flux_ae_key("decoder.up.0.block.1.conv1.weight") == (
        "decoder.up_blocks.3.resnets.1.conv1.weight"
    )
    assert _map_flux_ae_key("unexpected.weight") is None


def test_installed_kitchen_pins_tensorwise_int8_convrot_direct_native_primitive():
    import comfy_kitchen as kitchen

    assert (
        stored_adapter.Z_IMAGE_CONVROT_NATIVE_PRIMITIVE
        == "comfy_kitchen.registry/int8_linear@cuda"
    )
    assert callable(kitchen.registry.get_implementation)
    assert "scaled_mm_int8" not in kitchen.registry.list_backends().get("cuda", {}).get(
        "capabilities", ()
    )


def test_installed_kitchen_int8_callable_requires_weight_scale_schema():
    from comfy_kitchen import registry
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        kwargs = {
            "x": torch.zeros((2, 256), device="cuda", dtype=torch.bfloat16),
            "weight": torch.zeros((1, 256), device="cuda", dtype=torch.int8),
            "weight_scale": torch.ones((1, 1), device="cuda", dtype=torch.float32),
            "bias": torch.zeros(1, device="cuda", dtype=torch.float32),
            "out_dtype": torch.bfloat16,
            "convrot": True,
            "convrot_groupsize": 256,
            "input_act": None,
        }
        validation = registry.validate_backend_for_call("cuda", "int8_linear", kwargs)
        assert validation.success is True
        implementation = registry.get_implementation(
            "int8_linear", backend="cuda", kwargs=kwargs
        )
        signature = inspect.signature(implementation)
        assert "weight_scale" in signature.parameters
        assert "scale" not in signature.parameters
        signature.bind(**kwargs)

        wrong = {**kwargs, "scale": kwargs["weight_scale"]}
        wrong.pop("weight_scale")
        with pytest.raises(TypeError, match="required argument: 'weight_scale'"):
            signature.bind(**wrong)


def test_opt_in_real_z_image_headers_and_support_closure():
    """Run against locally supplied immutable files; never fetch a model in CI.

    Set ``LATENTSLATE_Z_IMAGE_TURBO_FIXTURE_ROOT`` to a directory containing
    ``pipeline-support/`` plus the three declared SafeTensors filenames.
    """

    root_value = os.environ.get("LATENTSLATE_Z_IMAGE_TURBO_FIXTURE_ROOT")
    if not root_value:
        pytest.skip("set LATENTSLATE_Z_IMAGE_TURBO_FIXTURE_ROOT for real Z-Image headers")
    root = Path(root_value)
    transformer = contract._plan_transformer(root / "z_image_turbo_int8_convrot.safetensors")
    qwen = qwen_checkpoint.plan_z_image_mixed_qwen(root / "qwen_3_4b_fp8_mixed.safetensors")
    from latentslate_engine.runtime.z_image_vae import plan_z_image_flux_ae

    vae = plan_z_image_flux_ae(root / "ae.safetensors")
    support = contract.plan_z_image_pipeline_support(root / "pipeline-support")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(root / "pipeline-support", local_files_only=True)
    prompt = "real tokenizer equivalence"
    expected_ids = tokenizer(
        "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
        padding=False,
        return_tensors="pt",
    )["input_ids"]
    actual_ids = conditioning.tokenize_z_image_prompt(tokenizer, prompt).input_ids
    assert transformer.stored_layer_count == 202
    assert (len(qwen.dense_sources), len(qwen.fp8_sources), len(qwen.nvfp4_sources)) == (
        209,
        177,
        12,
    )
    assert qwen.first_linear_format == "fp8"
    assert qwen_checkpoint.QWEN_FIRST_LINEAR_SOURCE in qwen.fp8_sources
    assert len(vae.source_to_target) == 244
    assert dict(support.files) == dict(contract._Z_PIPELINE_SUPPORT_FILES)
    assert torch.equal(actual_ids, expected_ids)


def test_installed_real_headers_prove_qwen_nextdit_and_flux_ae_metadata_only():
    """Exercise the local immutable headers only; no tensor payload is opened."""

    if not _INSTALLED_Z_IMAGE_TURBO_ROOT.is_dir():
        pytest.skip("installed Z-Image artifacts are not available")
    transformer = contract._plan_transformer(
        _INSTALLED_Z_IMAGE_TURBO_ROOT / "z_image_turbo_int8_convrot.safetensors"
    )
    qwen = qwen_checkpoint.plan_z_image_mixed_qwen(
        _INSTALLED_Z_IMAGE_TURBO_ROOT / "qwen_3_4b_fp8_mixed.safetensors"
    )
    shell = build_z_image_nextdit_shell(transformer)
    from latentslate_engine.runtime.z_image_vae import plan_z_image_flux_ae

    vae = plan_z_image_flux_ae(_INSTALLED_Z_IMAGE_TURBO_ROOT / "ae.safetensors")
    assert transformer.stored_layer_count == 202
    assert len(qwen.source_to_target) == 398
    assert set(qwen.source_to_target) == set(qwen_checkpoint.expected_qwen_weight_shapes())
    assert (len(qwen.dense_sources), len(qwen.fp8_sources), len(qwen.nvfp4_sources)) == (
        209,
        177,
        12,
    )
    assert qwen.first_linear_format == "fp8"
    assert qwen_checkpoint.QWEN_FIRST_LINEAR_SOURCE in qwen.fp8_sources
    assert len(shell._latentslate_z_image_convrot_bias_keys) == 32
    assert all(key in shell.state_dict() for key in shell._latentslate_z_image_convrot_bias_keys)
    assert shell.forward_contract()["executable"] is True
    assert (
        Counter(
            contract._z_image_transformer_stored_category(source)
            for source in transformer.stored_layers
        )
        == contract._Z_TRANSFORMER_STORED_CATEGORY_COUNTS
    )
    assert len(vae.source_to_target) == 244
    raw, header = contract._read_z_safetensors_header(
        _INSTALLED_Z_IMAGE_TURBO_ROOT / "ae.safetensors",
        (_INSTALLED_Z_IMAGE_TURBO_ROOT / "ae.safetensors").stat().st_size,
    )
    assert (
        hashlib.sha256(raw).hexdigest()
        == "6753860d781c5040a82e9aee0726719966ae774c1513d38789b264b30c496a39"
    )
    contract_keys = sorted(key for key in header if key in _map_flux_ae_attention_sources())
    for source in contract_keys:
        assert tuple(header[source]["shape"])[2:] == (1, 1)
        assert vae.source_to_target[source].endswith(
            ("to_q.weight", "to_k.weight", "to_v.weight", "to_out.0.weight")
        )
    assert len(contract_keys) == 8


def _map_flux_ae_attention_sources() -> frozenset[str]:
    return frozenset(
        f"{stage}.mid.attn_1.{projection}.weight"
        for stage in ("encoder", "decoder")
        for projection in ("q", "k", "v", "proj_out")
    )


def test_stored_convrot_plan_materializes_without_dense_fallback(tmp_path: Path, monkeypatch):
    path = tmp_path / "z-int8.safetensors"
    _save_transformer(path)
    raw, _ = contract._read_z_safetensors_header(path, path.stat().st_size)
    monkeypatch.setattr(contract, "_Z_TRANSFORMER_HEADER_SHA256", hashlib.sha256(raw).hexdigest())
    plan = _plan_transformer(path)
    assert plan.stored_layer_count == 202
    plan.require_stored_layout()
    assert (
        Counter(
            contract._z_image_transformer_stored_category(source) for source in plan.stored_layers
        )
        == contract._Z_TRANSFORMER_STORED_CATEGORY_COUNTS
    )
    restored = next(iter(plan.stored_layers.values())).materialize(torch.float32)
    assert restored.storage_dtype == torch.int8
    module = stored_adapter.ZImageStoredConvRotLinear(restored, torch.zeros(1, dtype=torch.float32))
    assert module.bias is not None and module.bias.dtype is torch.float32
    with pytest.raises(TypeError, match="bias must be exact"):
        stored_adapter.ZImageStoredConvRotLinear(restored, torch.zeros(2, dtype=torch.float32))


def test_convrot_dispatch_resolves_explicit_cuda_registry_without_global_policy(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "z-int8.safetensors"
    _save_transformer(path)
    raw, _ = contract._read_z_safetensors_header(path, path.stat().st_size)
    monkeypatch.setattr(contract, "_Z_TRANSFORMER_HEADER_SHA256", hashlib.sha256(raw).hexdigest())
    restored = next(iter(contract._plan_transformer(path).stored_layers.values())).materialize(
        torch.float32
    )
    bias = torch.zeros(1, dtype=torch.float32)
    observed = {}
    from comfy_kitchen import registry

    def implementation(**kwargs):
        observed["invocation"] = kwargs
        return torch.zeros((kwargs["x"].shape[0], kwargs["weight"].shape[0]))

    def resolve(name, *, backend, kwargs):
        observed["resolve"] = (name, backend, kwargs)
        return implementation

    monkeypatch.setattr(registry, "get_implementation", resolve)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    value = torch.zeros((2, 256), dtype=torch.bfloat16)
    module = stored_adapter.ZImageStoredConvRotLinear(restored, bias)
    result = module._dispatch_flat(value)
    assert result.shape == (2, 1)
    assert module.native_dispatch_count == 1
    assert module.rejected_dispatch_count == module.dense_fallback_count == 0
    name, backend, kwargs = observed["resolve"]
    assert (name, backend) == ("int8_linear", "cuda")
    assert kwargs.keys() == observed["invocation"].keys()
    assert kwargs["x"] is observed["invocation"]["x"]
    assert kwargs["weight"] is observed["invocation"]["weight"]
    assert kwargs["x"].dtype is torch.bfloat16
    assert kwargs["weight"].dtype is torch.int8
    assert kwargs["weight_scale"].dtype is torch.float32
    assert kwargs["bias"] is module.bias
    assert kwargs["out_dtype"] is torch.bfloat16
    assert kwargs["convrot"] is True and kwargs["convrot_groupsize"] == 256
    assert kwargs["input_act"] is None

    def fail_resolve(*_args, **_kwargs):
        raise RuntimeError("injected resolver failure")

    monkeypatch.setattr(registry, "get_implementation", fail_resolve)
    with pytest.raises(RuntimeError, match="dense fallback is forbidden"):
        module._dispatch_flat(value)
    assert module.native_dispatch_count == 1
    assert module.rejected_dispatch_count == 1

    def fail_invoke(**_kwargs):
        raise TypeError("injected implementation failure")

    monkeypatch.setattr(registry, "get_implementation", lambda *_args, **_kwargs: fail_invoke)
    with pytest.raises(RuntimeError, match="dense fallback is forbidden"):
        module._dispatch_flat(value)
    assert module.native_dispatch_count == 1
    assert module.rejected_dispatch_count == 2

    with pytest.raises(RuntimeError, match="dense fallback is forbidden"):
        module(torch.zeros((2, 256)))
    assert module.native_dispatch_count == 1
    assert module.rejected_dispatch_count == 3
    assert module.dense_fallback_count == 0


def test_nextdit_residency_rebuild_failure_rolls_back_and_poison_ejects(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "z-int8.safetensors"
    _save_transformer(path)
    raw, _ = contract._read_z_safetensors_header(path, path.stat().st_size)
    monkeypatch.setattr(contract, "_Z_TRANSFORMER_HEADER_SHA256", hashlib.sha256(raw).hexdigest())
    restored = next(iter(contract._plan_transformer(path).stored_layers.values())).materialize(
        torch.float32
    )
    model = nn.Module()
    model.linear = stored_adapter.ZImageStoredConvRotLinear(restored)
    original_qdata = model.linear.weight._qdata.clone()
    import comfy_kitchen.tensor as kitchen_tensor

    def fail_rebuild(*_args, **_kwargs):
        raise RuntimeError("injected rebuild failure")

    monkeypatch.setattr(kitchen_tensor, "QuantizedTensor", fail_rebuild)
    with pytest.raises(RuntimeError, match="injected"):
        stored_adapter.move_z_image_nextdit_storage(model, "cpu")
    assert model.linear.weight is not None
    assert torch.equal(model.linear.weight._qdata, original_qdata)
    assert model.linear.weight.device.type == "cpu"
    assert "RuntimeError" in model._latentslate_z_image_residency_poisoned
    stored_adapter.move_z_image_nextdit_storage(model, "cpu")


def test_nextdit_shell_requires_exact_adaln_bias_closure(tmp_path: Path, monkeypatch):
    path = tmp_path / "z-int8-bias.safetensors"
    _save_transformer(path, include_adaln_bias=True)
    raw, _ = contract._read_z_safetensors_header(path, path.stat().st_size)
    monkeypatch.setattr(contract, "_Z_TRANSFORMER_HEADER_SHA256", hashlib.sha256(raw).hexdigest())
    plan = contract._plan_transformer(path)
    _raw, header = contract._read_z_safetensors_header(path, path.stat().st_size)
    bias_keys = _z_image_nextdit_stored_bias_keys(plan, header)
    assert len(bias_keys) == 32
    assert all(key.endswith("adaLN_modulation.0.bias") for key in bias_keys)


@pytest.mark.parametrize(
    ("include_adaln_bias", "include_extra_stored_bias"),
    [(False, False), (True, True)],
)
def test_nextdit_shell_rejects_missing_or_extra_stored_biases(
    tmp_path: Path, monkeypatch, include_adaln_bias: bool, include_extra_stored_bias: bool
):
    path = tmp_path / "z-int8-invalid-bias.safetensors"
    _save_transformer(
        path,
        include_adaln_bias=include_adaln_bias,
        include_extra_stored_bias=include_extra_stored_bias,
    )
    raw, _ = contract._read_z_safetensors_header(path, path.stat().st_size)
    monkeypatch.setattr(contract, "_Z_TRANSFORMER_HEADER_SHA256", hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="bias stems"):
        plan = contract._plan_transformer(path)
        _raw, header = contract._read_z_safetensors_header(path, path.stat().st_size)
        _z_image_nextdit_stored_bias_keys(plan, header)


def test_lifecycle_rejects_cancel_and_never_claims_warm_cache(tmp_path: Path, monkeypatch):
    transformer_path = tmp_path / "z-int8.safetensors"
    _save_transformer(transformer_path)
    raw, _ = contract._read_z_safetensors_header(transformer_path, transformer_path.stat().st_size)
    monkeypatch.setattr(contract, "_Z_TRANSFORMER_HEADER_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(
        "latentslate_engine.runtime.z_image_qwen_checkpoint.revalidate_z_image_mixed_qwen",
        lambda _plan: True,
    )
    transformer = _plan_transformer(transformer_path)
    dense_identity = probe_artifact(transformer_path).identity
    dense = ZImageDensePlan(dense_identity, "synthetic", "text_encoder", 1)
    request = ZImageTurboRuntimeRequest(
        1,
        "comfy-org-z-image-turbo-int8-convrot",
        "zimage_turbo_t2i_int8_convrot",
        {
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "guider": "basic",
            "sampling": "auraflow_shift_3",
            "sampler": "res_multistep",
            "scheduler": "simple",
        },
        {
            role: {
                "path": str(transformer_path.resolve()),
                "header_sha256": dense_identity.header_sha256,
                "source_revision": contract._IMMUTABLE_COMPONENTS[role][1],
                "source_filename": contract._IMMUTABLE_COMPONENTS[role][2],
                "source_sha256": contract._IMMUTABLE_COMPONENTS[role][3],
            }
            for role in ("transformer", "text_encoder", "vae")
        },
        {role: dense_identity for role in ("transformer", "text_encoder", "vae")},
        {
            "transformer": transformer,
            "text_encoder": dense,
            "vae": ZImageDensePlan(dense_identity, "synthetic", "vae", 1),
        },
    )
    lifecycle = ZImageTurboLifecycle(request)
    with pytest.raises(ZImageTurboCancelled):
        lifecycle.checkpoint(ZImagePhase.TEXT_ENCODER, lambda: True)
    assert lifecycle.ejected
    provenance = lifecycle.public_provenance()
    assert provenance["execution_cache"]["supported"] is False
    assert provenance["request_fingerprint"] == request.fingerprint
    assert provenance["components"] == request.public_component_manifest()
    assert provenance["native_transformer_dispatch"] == {
        "proven": False,
        "count": 0,
        "reason": "lifecycle-only provenance does not include transformer dispatch proof",
    }


def test_lifecycle_requires_text_transformer_vae_order(tmp_path: Path, monkeypatch):
    transformer_path = tmp_path / "z-int8.safetensors"
    _save_transformer(transformer_path)
    raw, _ = contract._read_z_safetensors_header(transformer_path, transformer_path.stat().st_size)
    monkeypatch.setattr(contract, "_Z_TRANSFORMER_HEADER_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(
        "latentslate_engine.runtime.z_image_qwen_checkpoint.revalidate_z_image_mixed_qwen",
        lambda _plan: True,
    )
    identity = probe_artifact(transformer_path).identity
    transformer = _plan_transformer(transformer_path)
    dense = ZImageDensePlan(identity, "synthetic", "text_encoder", 1)
    request = ZImageTurboRuntimeRequest(
        1,
        "comfy-org-z-image-turbo-int8-convrot",
        "zimage_turbo_t2i_int8_convrot",
        {
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "guider": "basic",
            "sampling": "auraflow_shift_3",
            "sampler": "res_multistep",
            "scheduler": "simple",
        },
        {
            role: {
                "path": str(transformer_path.resolve()),
                "header_sha256": identity.header_sha256,
                "source_revision": contract._IMMUTABLE_COMPONENTS[role][1],
                "source_filename": contract._IMMUTABLE_COMPONENTS[role][2],
                "source_sha256": contract._IMMUTABLE_COMPONENTS[role][3],
            }
            for role in ("transformer", "text_encoder", "vae")
        },
        {role: identity for role in ("transformer", "text_encoder", "vae")},
        {
            "transformer": transformer,
            "text_encoder": dense,
            "vae": ZImageDensePlan(identity, "synthetic", "vae", 1),
        },
    )
    lifecycle = ZImageTurboLifecycle(request)
    with pytest.raises(ValueError, match="invalid after"):
        lifecycle.checkpoint(ZImagePhase.VAE, lambda: False)
    assert lifecycle.ejected


def test_schedule_rejects_any_base_or_edit_like_deviation():
    recipe = ZImageTurboRecipe.__new__(ZImageTurboRecipe)
    object.__setattr__(recipe, "width", 1024)
    object.__setattr__(recipe, "height", 1024)
    object.__setattr__(recipe, "steps", 9)
    object.__setattr__(recipe, "guider", "basic")
    object.__setattr__(recipe, "sampling", "auraflow_shift_3")
    object.__setattr__(recipe, "sampler", "res_multistep")
    object.__setattr__(recipe, "scheduler", "simple")
    with pytest.raises(ValueError, match="exact pinned schedule"):
        z_image_turbo_schedule(recipe)


def test_mixed_qwen_full_precision_proof_requires_all_189_low_bit_modules():
    from latentslate_engine.stored_quant import (
        restore_global_fp8_tensor,
        restore_nvfp4_tensor,
    )

    def wrapper(kind: str):
        if kind == "fp8":
            return qwen_runtime.ZImageFullPrecisionFP8Linear(
                restore_global_fp8_tensor(
                    torch.zeros((8, 8), dtype=torch.float8_e4m3fn),
                    torch.tensor(0.25, dtype=torch.float32),
                    torch.bfloat16,
                ),
            )
        return qwen_runtime.ZImageFullPrecisionNVFP4Linear(
            restore_nvfp4_tensor(
                torch.zeros((128, 128), dtype=torch.uint8),
                torch.ones((128, 16), dtype=torch.float8_e4m3fn),
                torch.tensor(0.25, dtype=torch.float32),
                (128, 256),
                torch.bfloat16,
            ),
        )

    model = nn.Module()
    model.quantized = nn.ModuleList(
        [wrapper("fp8") for _ in range(177)] + [wrapper("nvfp4") for _ in range(12)]
    )
    names = {f"quantized.{index}": "fp8" if index < 177 else "nvfp4" for index in range(189)}
    model._latentslate_z_image_quant_modules = MappingProxyType(names)
    before = qwen_runtime.z_image_mixed_dispatch_snapshot(model)
    for module in (model.quantized[0], model.quantized[177]):
        stored_qdata = module.weight._qdata
        converted = module.weight.to(dtype=torch.float32)
        assert module.weight.params.orig_dtype is torch.bfloat16
        assert converted.params.orig_dtype is torch.float32
        assert converted._qdata.dtype is stored_qdata.dtype
        assert torch.equal(converted._qdata, stored_qdata)
        assert converted.dequantize().dtype is torch.float32
    for module in model.quantized:
        stored_qdata = module.weight._qdata
        stored_scale = module.weight.params.scale
        stored_block_scale = getattr(module.weight.params, "block_scale", None)
        assert module.weight.params.orig_dtype is torch.bfloat16
        output = module(torch.zeros((1, 2, module.weight.shape[1]), dtype=torch.float32))
        assert output.shape == (1, 2, module.weight.shape[0]) and output.dtype is torch.float32
        assert module.weight._qdata is stored_qdata and module.weight.params.orig_dtype is torch.bfloat16
        assert module.weight.params.scale is stored_scale
        assert getattr(module.weight.params, "block_scale", None) is stored_block_scale
        assert module.per_op_move_count == 1
    proof = qwen_runtime.verify_z_image_mixed_dispatch(model, before)
    assert proof["module_count"] == proof["dequantized_modules"] == proof["f_linear_modules"] == 189
    assert proof["fp8_modules"] == 177 and proof["nvfp4_modules"] == 12
    assert proof["total_dequantizations"] == proof["total_f_linear_calls"] == 189
    assert proof["rejected_dispatch_count"] == proof["dense_checkpoint_fallback_count"] == 0
    assert proof["activation_quantized"] is False and proof["scaled_mm_calls"] == 0

    # Aggregate-equal totals are insufficient: every stored module must pair
    # its Kitchen dequantization with its own ordinary F.linear invocation.
    crossed = qwen_runtime.z_image_mixed_dispatch_snapshot(model)
    model.quantized[0].native_dequant_count += 1
    model.quantized[1].f_linear_count += 1
    with pytest.raises(RuntimeError, match="did not dequantize and F.linear every stored layer"):
        qwen_runtime.verify_z_image_mixed_dispatch(model, crossed)

    model.quantized[0] = nn.Identity()
    with pytest.raises(TypeError, match="wrapper type"):
        qwen_runtime.z_image_mixed_dispatch_snapshot(model)

    model.quantized[0] = wrapper("nvfp4")
    with pytest.raises(TypeError, match="wrapper type"):
        qwen_runtime.z_image_mixed_dispatch_snapshot(model)

    model.quantized[0] = wrapper("fp8")
    model.quantized[0].dense_checkpoint_fallback_count = 1
    with pytest.raises(ValueError, match="checkpoint-fallback history"):
        qwen_runtime.z_image_mixed_dispatch_snapshot(model)


@pytest.mark.parametrize("kind", ("fp8", "nvfp4"))
def test_mixed_qwen_first_linear_preflight_validates_both_kitchen_layouts(kind):
    from latentslate_engine.stored_quant import (
        restore_global_fp8_tensor,
        restore_nvfp4_tensor,
    )

    if kind == "fp8":
        shape = (8, 8)
        module = qwen_runtime.ZImageFullPrecisionFP8Linear(
            restore_global_fp8_tensor(
                torch.zeros(shape, dtype=torch.float8_e4m3fn),
                torch.tensor(0.25, dtype=torch.float32),
                torch.bfloat16,
            )
        )
    else:
        shape = (128, 256)
        module = qwen_runtime.ZImageFullPrecisionNVFP4Linear(
            restore_nvfp4_tensor(
                torch.zeros((128, 128), dtype=torch.uint8),
                torch.ones((128, 16), dtype=torch.float8_e4m3fn),
                torch.tensor(0.25, dtype=torch.float32),
                shape,
                torch.bfloat16,
            )
        )
    before = (
        module.native_dequant_count,
        module.f_linear_count,
        module.rejected_dispatch_count,
        module.per_op_move_count,
    )
    stages: list[str] = []
    proof = qwen_runtime._preflight_z_image_full_precision_linear(
        module,
        "cpu",
        expected_shape=shape,
        diagnostic=stages.append,
    )
    assert proof["first_linear_preflight"] is True
    assert proof["first_linear_format"] == kind
    assert proof["first_linear_layout_registered"] is True
    assert proof["first_linear_logical_shape"] == f"{shape[0]}x{shape[1]}"
    expected_stages = [
        f"conditioning.preflight_{kind}_cuda_sync",
        f"conditioning.preflight_{kind}_uint8_allocate",
        f"conditioning.preflight_{kind}_ordinary_uint8_copy",
        f"conditioning.preflight_{kind}_ordinary_uint8_sync",
        f"conditioning.preflight_{kind}_ordinary_uint8_readback",
        f"conditioning.preflight_{kind}_origin_flat_prepare",
        f"conditioning.preflight_{kind}_origin_uint8_copy",
    ]
    if kind == "fp8":
        expected_stages.append(f"conditioning.preflight_{kind}_flat_dtype_view")
    expected_stages.extend(
        [
            f"conditioning.preflight_{kind}_shape_restore",
            f"conditioning.preflight_{kind}_scale_move",
            f"conditioning.preflight_{kind}_bit_verify",
            f"conditioning.preflight_{kind}_direct_fp32_dequant",
            f"conditioning.preflight_{kind}_f_linear",
            f"conditioning.preflight_{kind}_validate",
        ]
    )
    assert stages == expected_stages
    assert before == (
        module.native_dequant_count,
        module.f_linear_count,
        module.rejected_dispatch_count,
        module.per_op_move_count,
    )


def test_mixed_qwen_raw_transport_moves_nvfp4_fields_without_source_mutation():
    from latentslate_engine.stored_quant import restore_nvfp4_tensor

    weight = restore_nvfp4_tensor(
        torch.zeros((128, 128), dtype=torch.uint8),
        torch.ones((128, 16), dtype=torch.float8_e4m3fn),
        torch.tensor(0.25, dtype=torch.float32),
        (128, 256),
        torch.bfloat16,
    )
    qdata, scale, block_scale = weight._qdata, weight.params.scale, weight.params.block_scale
    moved = qwen_runtime._transport_z_image_quantized_weight(weight, "cpu", verify_bits=True)
    assert moved.qdata is not qdata
    assert moved.scale is not scale
    assert moved.block_scale is not block_scale
    assert torch.equal(moved.qdata, qdata)
    assert torch.equal(moved.scale, scale)
    assert torch.equal(moved.block_scale, block_scale)
    assert moved.qdata.dtype is torch.uint8
    assert moved.scale.dtype is torch.float32
    assert moved.block_scale.dtype is torch.float8_e4m3fn
    assert moved.orig_shape == weight.params.orig_shape == (128, 256)
    assert weight._qdata is qdata
    assert weight.params.scale is scale
    assert weight.params.block_scale is block_scale


def test_kitchen_public_direct_fp32_api_signatures_are_exact():
    import comfy_kitchen

    fp8 = inspect.signature(comfy_kitchen.dequantize_per_tensor_fp8)
    nvfp4 = inspect.signature(comfy_kitchen.dequantize_nvfp4)
    assert tuple(fp8.parameters) == ("x", "scale", "output_type")
    assert fp8.parameters["output_type"].default is torch.bfloat16
    assert tuple(nvfp4.parameters) == (
        "qx",
        "per_tensor_scale",
        "block_scales",
        "output_type",
        "hi_first",
    )
    assert nvfp4.parameters["output_type"].default is torch.bfloat16
    assert nvfp4.parameters["hi_first"].default is True


@pytest.mark.parametrize("kind", ("fp8", "nvfp4"))
def test_direct_fp32_dequant_matches_working_logical_cast_reference(kind):
    from latentslate_engine.stored_quant import (
        restore_global_fp8_tensor,
        restore_nvfp4_tensor,
    )

    if kind == "fp8":
        qdata = torch.arange(64, dtype=torch.float32).reshape(8, 8).to(torch.float8_e4m3fn)
        weight = restore_global_fp8_tensor(
            qdata,
            torch.tensor(0.1234567, dtype=torch.float32),
            torch.bfloat16,
        )
    else:
        weight = restore_nvfp4_tensor(
            torch.arange(128 * 128, dtype=torch.int64).reshape(128, 128).to(torch.uint8),
            torch.ones((128, 16), dtype=torch.float8_e4m3fn),
            torch.tensor(0.1234567, dtype=torch.float32),
            (128, 256),
            torch.bfloat16,
        )
    reference = weight.to(dtype=torch.float32).dequantize().contiguous()
    transported = qwen_runtime._transport_z_image_quantized_weight(
        weight, "cpu", verify_bits=True
    )
    actual = qwen_runtime._direct_fp32_dequantize_z_image_weight(transported)
    assert actual.dtype is reference.dtype is torch.float32
    assert actual.shape == reference.shape == weight.shape
    assert torch.equal(actual, reference)
    assert weight.params.orig_dtype is torch.bfloat16


def test_direct_fp32_dequant_is_not_bf16_dequant_then_widened():
    from latentslate_engine.stored_quant import restore_global_fp8_tensor

    weight = restore_global_fp8_tensor(
        torch.arange(64, dtype=torch.float32).reshape(8, 8).to(torch.float8_e4m3fn),
        torch.tensor(0.1234567, dtype=torch.float32),
        torch.bfloat16,
    )
    transported = qwen_runtime._transport_z_image_quantized_weight(
        weight, "cpu", verify_bits=True
    )
    direct = qwen_runtime._direct_fp32_dequantize_z_image_weight(transported)
    widened = weight.dequantize().float()
    assert direct.dtype is widened.dtype is torch.float32
    assert torch.any(direct != widened)


def test_production_fp8_linear_never_uses_quantized_tensor_logical_cast(
    monkeypatch,
):
    import comfy_kitchen
    from comfy_kitchen.tensor import QuantizedTensor

    from latentslate_engine.stored_quant import restore_global_fp8_tensor

    module = qwen_runtime.ZImageFullPrecisionFP8Linear(
        restore_global_fp8_tensor(
            torch.ones((8, 8), dtype=torch.float8_e4m3fn),
            torch.tensor(0.1234567, dtype=torch.float32),
            torch.bfloat16,
        )
    )
    observed: list[torch.dtype] = []
    public_dequant = comfy_kitchen.dequantize_per_tensor_fp8

    def observed_dequant(x, scale, output_type=torch.bfloat16):
        observed.append(output_type)
        return public_dequant(x, scale, output_type=output_type)

    monkeypatch.setattr(comfy_kitchen, "dequantize_per_tensor_fp8", observed_dequant)
    monkeypatch.setattr(
        QuantizedTensor,
        "to",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("logical wrapper cast is forbidden")
        ),
        raising=False,
    )
    output = module(torch.zeros((1, 1, 8), dtype=torch.float32))
    assert output.shape == (1, 1, 8) and output.dtype is torch.float32
    assert observed == [torch.float32]


def test_direct_fp32_nvfp4_dequant_crops_padded_storage_to_orig_shape():
    from latentslate_engine.stored_quant import restore_nvfp4_tensor

    weight = restore_nvfp4_tensor(
        torch.zeros((32, 24), dtype=torch.uint8),
        torch.ones((128, 4), dtype=torch.float8_e4m3fn),
        torch.tensor(0.25, dtype=torch.float32),
        (17, 33),
        torch.bfloat16,
    )
    transported = qwen_runtime._transport_z_image_quantized_weight(
        weight, "cpu", verify_bits=True
    )
    actual = qwen_runtime._direct_fp32_dequantize_z_image_weight(transported)
    assert actual.shape == (17, 33)
    assert actual.dtype is torch.float32 and actual.is_contiguous()


@pytest.mark.parametrize(
    "failure",
    (
        "origin_flat_prepare",
        "origin_uint8_copy",
        "flat_dtype_view",
        "shape_restore",
        "scale_move",
        "bit_verify",
    ),
)
def test_mixed_qwen_raw_transport_reports_exact_failure_substage(monkeypatch, failure):
    from latentslate_engine.stored_quant import restore_global_fp8_tensor

    qdata = torch.zeros((8, 8), dtype=torch.float8_e4m3fn)
    if failure == "origin_flat_prepare":
        qdata = qdata.t()
    weight = restore_global_fp8_tensor(
        qdata,
        torch.tensor(0.25, dtype=torch.float32),
        torch.bfloat16,
    )
    if failure == "origin_uint8_copy":
        monkeypatch.setattr(
            qwen_runtime,
            "_copy_z_image_raw_bytes",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("private qdata copy detail")
            ),
        )
    elif failure == "flat_dtype_view":
        monkeypatch.setattr(
            qwen_runtime,
            "_view_z_image_flat_raw_as_dtype",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("private dtype view detail")
            ),
        )
    elif failure == "shape_restore":
        monkeypatch.setattr(
            qwen_runtime,
            "_restore_z_image_qdata_shape",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("private shape restore detail")
            ),
        )
    elif failure == "scale_move":
        monkeypatch.setattr(
            qwen_runtime,
            "_move_z_image_scale_field",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("private scale detail")
            ),
        )
    elif failure == "bit_verify":
        original_copy = qwen_runtime._copy_z_image_raw_bytes

        def corrupt_copy(raw, device, **kwargs):
            copied = original_copy(raw, device, **kwargs)
            copied.view(-1)[0] = 1
            return copied

        monkeypatch.setattr(qwen_runtime, "_copy_z_image_raw_bytes", corrupt_copy)
    active = {"stage": ""}
    with pytest.raises(RuntimeError):
        qwen_runtime._transport_z_image_quantized_weight(
            weight,
            "cpu",
            diagnostic_prefix="conditioning.preflight_fp8",
            diagnostic=lambda stage: active.__setitem__("stage", stage),
            verify_bits=True,
        )
    assert active["stage"] == f"conditioning.preflight_fp8_{failure}"


@pytest.mark.parametrize(
    "failure",
    (
        "cuda_sync",
        "uint8_allocate",
        "ordinary_uint8_copy",
        "ordinary_uint8_sync",
        "ordinary_uint8_readback",
    ),
)
def test_mixed_qwen_preflight_isolates_uint8_copy_failures(monkeypatch, failure):
    from latentslate_engine.stored_quant import restore_global_fp8_tensor

    module = qwen_runtime.ZImageFullPrecisionFP8Linear(
        restore_global_fp8_tensor(
            torch.zeros((8, 8), dtype=torch.float8_e4m3fn),
            torch.tensor(0.25, dtype=torch.float32),
            torch.bfloat16,
        )
    )
    shared_substage = {
        "cuda_sync": "sync_before",
        "uint8_allocate": "allocate",
        "ordinary_uint8_copy": "copy",
        "ordinary_uint8_sync": "sync_after",
        "ordinary_uint8_readback": "readback",
    }[failure]

    def fail_shared(_torch, _device, *, checkpoint):
        checkpoint(shared_substage)
        raise RuntimeError("private shared health detail")

    monkeypatch.setattr(
        qwen_runtime._cuda_health, "z_image_cuda_health_check", fail_shared
    )
    active = {"stage": ""}
    with pytest.raises(RuntimeError, match="private"):
        qwen_runtime._preflight_z_image_full_precision_linear(
            module,
            "cpu",
            expected_shape=(8, 8),
            diagnostic=lambda stage: active.__setitem__("stage", stage),
        )
    assert active["stage"] == f"conditioning.preflight_fp8_{failure}"


def test_mixed_qwen_byte_copy_is_exact_empty_like_then_blocking_copy(monkeypatch):
    from comfy_kitchen.tensor import QuantizedTensor

    source = torch.arange(16, dtype=torch.uint8)
    observed: list[tuple[str, object]] = []
    original_allocate = qwen_runtime._allocate_z_image_raw_bytes
    original_copy = qwen_runtime._copy_z_image_bytes_into

    def observe_allocate(flat_bytes_cpu, device):
        observed.append(("allocate", (flat_bytes_cpu.shape, device)))
        return original_allocate(flat_bytes_cpu, device)

    def observe_copy(destination, flat_bytes_cpu):
        observed.append(("copy", (destination.shape, flat_bytes_cpu.shape)))
        return original_copy(destination, flat_bytes_cpu)

    monkeypatch.setattr(qwen_runtime, "_allocate_z_image_raw_bytes", observe_allocate)
    monkeypatch.setattr(qwen_runtime, "_copy_z_image_bytes_into", observe_copy)
    destination = qwen_runtime._copy_z_image_raw_bytes(source, torch.device("cpu"))
    assert [entry[0] for entry in observed] == ["allocate", "copy"]
    assert isinstance(source, torch.Tensor) and not isinstance(source, QuantizedTensor)
    assert isinstance(destination, torch.Tensor)
    assert not isinstance(destination, QuantizedTensor)
    assert destination is not source and torch.equal(destination, source)
    allocate_source = inspect.getsource(original_allocate)
    copy_source = inspect.getsource(original_copy)
    assert "torch.empty_like(flat_bytes_cpu, device=device)" in allocate_source
    assert "destination.copy_(flat_bytes_cpu, non_blocking=False)" in copy_source
    runtime_source = Path(qwen_runtime.__file__).read_text(encoding="utf-8")
    assert ".to(device=device, non_blocking=False, copy=True)" not in runtime_source


def test_mixed_qwen_fp8_raw_transport_roundtrips_every_byte_pattern():
    from latentslate_engine.stored_quant import restore_global_fp8_tensor

    raw = torch.arange(256, dtype=torch.int16).to(torch.uint8).reshape(16, 16)
    qdata = raw.view(torch.float8_e4m3fn)
    weight = restore_global_fp8_tensor(
        qdata,
        torch.tensor(0.25, dtype=torch.float32),
        torch.bfloat16,
    )
    source_qdata = weight._qdata
    source_scale = weight.params.scale
    transported = qwen_runtime._transport_z_image_quantized_weight(
        weight, "cpu", verify_bits=True
    )
    assert torch.equal(transported.qdata.view(torch.uint8), raw)
    assert transported.qdata.dtype is torch.float8_e4m3fn
    assert transported.qdata.shape == (16, 16)
    assert transported.qdata is not source_qdata
    assert transported.scale is not source_scale
    assert weight._qdata is source_qdata
    assert weight.params.scale is source_scale
    assert torch.equal(weight._qdata.view(torch.uint8), raw)


def test_mixed_qwen_raw_transport_rejects_nonzero_source_storage_offset():
    from latentslate_engine.stored_quant import restore_global_fp8_tensor

    backing = torch.zeros((65,), dtype=torch.float8_e4m3fn)
    qdata = backing[1:].view(8, 8)
    assert qdata.is_contiguous() and qdata.storage_offset() == 1
    weight = restore_global_fp8_tensor(
        qdata,
        torch.tensor(0.25, dtype=torch.float32),
        torch.bfloat16,
    )
    active = {"stage": ""}
    with pytest.raises(RuntimeError, match="flat-byte-view compatible"):
        qwen_runtime._transport_z_image_quantized_weight(
            weight,
            "cpu",
            diagnostic_prefix="conditioning.preflight_fp8",
            diagnostic=lambda stage: active.__setitem__("stage", stage),
        )
    assert active["stage"] == "conditioning.preflight_fp8_origin_flat_prepare"


@pytest.mark.parametrize("geometry", ("nonzero_offset", "noncontiguous"))
def test_mixed_qwen_raw_transport_rejects_invalid_copied_flat_geometry(
    monkeypatch, geometry
):
    from latentslate_engine.stored_quant import restore_global_fp8_tensor

    weight = restore_global_fp8_tensor(
        torch.zeros((8, 8), dtype=torch.float8_e4m3fn),
        torch.tensor(0.25, dtype=torch.float32),
        torch.bfloat16,
    )

    def malformed_copy(raw, _device, **_kwargs):
        if geometry == "nonzero_offset":
            return torch.empty((raw.numel() + 1,), dtype=torch.uint8)[1:]
        return torch.empty((raw.numel() * 2,), dtype=torch.uint8)[::2]

    monkeypatch.setattr(qwen_runtime, "_copy_z_image_raw_bytes", malformed_copy)
    active = {"stage": ""}
    with pytest.raises(RuntimeError, match="copy changed byte geometry"):
        qwen_runtime._transport_z_image_quantized_weight(
            weight,
            "cpu",
            diagnostic_prefix="conditioning.preflight_fp8",
            diagnostic=lambda stage: active.__setitem__("stage", stage),
        )
    assert active["stage"] == "conditioning.preflight_fp8_origin_uint8_copy"


def test_mixed_qwen_fp8_flat_reinterpret_preserves_fake_cuda_geometry():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        raw_source = torch.empty((256,), device="cpu", dtype=torch.uint8)
        scale_source = torch.empty((), device="cpu", dtype=torch.float32)
        raw = qwen_runtime._copy_z_image_raw_bytes(
            raw_source, torch.device("cuda:0")
        )
        scale = qwen_runtime._move_z_image_scale_field(
            scale_source, torch.device("cuda:0")
        )
        typed_flat = qwen_runtime._view_z_image_flat_raw_as_dtype(
            raw, torch.float8_e4m3fn
        )
        qdata = qwen_runtime._restore_z_image_qdata_shape(
            typed_flat, torch.Size((16, 16))
        )
        assert typed_flat.ndim == 1 and typed_flat.shape == (256,)
        assert qdata.device == torch.device("cuda:0")
        assert qdata.dtype is torch.float8_e4m3fn
        assert qdata.shape == (16, 16)
        assert qdata.is_contiguous()
        assert scale.device == torch.device("cuda:0")
        assert scale.dtype is torch.float32 and scale.shape == torch.Size([])


def test_mixed_qwen_source_shaped_fp8_cuda_capability_seam_is_independent():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        source = torch.empty((4, 4), device="cpu", dtype=torch.float8_e4m3fn)
        viewed = qwen_runtime._probe_z_image_source_shaped_fp8_view_capability(
            source, "cuda:0"
        )
        assert viewed.device == torch.device("cuda:0")
        assert viewed.dtype is torch.float8_e4m3fn
        assert viewed.shape == (4, 4)


@pytest.mark.parametrize("kind", ("fp8", "nvfp4"))
def test_mixed_qwen_raw_transport_uses_exact_flat_call_geometry(monkeypatch, kind):
    from latentslate_engine.stored_quant import (
        restore_global_fp8_tensor,
        restore_nvfp4_tensor,
    )

    if kind == "fp8":
        weight = restore_global_fp8_tensor(
            torch.zeros((8, 8), dtype=torch.float8_e4m3fn),
            torch.tensor(0.25, dtype=torch.float32),
            torch.bfloat16,
        )
    else:
        weight = restore_nvfp4_tensor(
            torch.zeros((8, 4), dtype=torch.uint8),
            torch.ones((128, 1), dtype=torch.float8_e4m3fn),
            torch.tensor(0.25, dtype=torch.float32),
            (8, 8),
            torch.bfloat16,
        )
    observations: list[tuple[str, int, tuple[int, ...], torch.dtype]] = []
    original_copy = qwen_runtime._copy_z_image_raw_bytes
    original_dtype_view = qwen_runtime._view_z_image_flat_raw_as_dtype
    original_restore = qwen_runtime._restore_z_image_qdata_shape

    def capture_copy(raw, device, **kwargs):
        observations.append(("copy", raw.ndim, tuple(raw.shape), raw.dtype))
        assert raw.is_contiguous() and raw.storage_offset() == 0
        assert raw.stride() == (1,)
        return original_copy(raw, device, **kwargs)

    def capture_dtype_view(raw, dtype):
        observations.append(("dtype_view", raw.ndim, tuple(raw.shape), raw.dtype))
        return original_dtype_view(raw, dtype)

    def capture_restore(typed, shape):
        observations.append(("restore", typed.ndim, tuple(typed.shape), typed.dtype))
        return original_restore(typed, shape)

    monkeypatch.setattr(qwen_runtime, "_copy_z_image_raw_bytes", capture_copy)
    monkeypatch.setattr(qwen_runtime, "_view_z_image_flat_raw_as_dtype", capture_dtype_view)
    monkeypatch.setattr(qwen_runtime, "_restore_z_image_qdata_shape", capture_restore)
    transported = qwen_runtime._transport_z_image_quantized_weight(weight, "cpu")
    assert observations[0][:2] == ("copy", 1)
    assert observations[-1][0:2] == ("restore", 1)
    if kind == "fp8":
        assert [entry[0] for entry in observations] == [
            "copy",
            "dtype_view",
            "restore",
        ]
        assert observations[1][3] is torch.uint8
    else:
        assert [entry[0] for entry in observations] == ["copy", "restore"]
        assert transported.qdata.dtype is torch.uint8


def test_mixed_qwen_raw_transport_never_uses_numeric_uint8_conversion():
    source = inspect.getsource(qwen_runtime._transport_z_image_quantized_weight)
    prepare = inspect.getsource(qwen_runtime._prepare_z_image_flat_raw_bytes)
    dtype_view = inspect.getsource(qwen_runtime._view_z_image_flat_raw_as_dtype)
    assert ".view(torch.uint8)" in prepare
    assert ".reshape(-1)" in prepare
    assert ".view(dtype=storage_dtype)" in dtype_view
    assert ".to(torch.uint8)" not in source
    assert ".to(torch.uint8)" not in prepare


@pytest.mark.parametrize("failure", ("direct_fp32_dequant", "f_linear"))
def test_mixed_qwen_preflight_reports_exact_failure_substage(monkeypatch, failure):
    import comfy_kitchen

    from latentslate_engine.stored_quant import restore_global_fp8_tensor

    module = qwen_runtime.ZImageFullPrecisionFP8Linear(
        restore_global_fp8_tensor(
            torch.zeros((8, 8), dtype=torch.float8_e4m3fn),
            torch.tensor(0.25, dtype=torch.float32),
            torch.bfloat16,
        )
    )
    if failure == "direct_fp32_dequant":
        monkeypatch.setattr(
            comfy_kitchen,
            "dequantize_per_tensor_fp8",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("private dequant detail")
            ),
        )
    else:
        monkeypatch.setattr(
            qwen_runtime.F,
            "linear",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("private F.linear detail")
            ),
        )
    active = {"stage": ""}
    with pytest.raises(RuntimeError, match="private"):
        qwen_runtime._preflight_z_image_full_precision_linear(
            module,
            "cpu",
            expected_shape=(8, 8),
            diagnostic=lambda stage: active.__setitem__("stage", stage),
        )
    assert active["stage"] == f"conditioning.preflight_fp8_{failure}"


def test_mixed_qwen_real_first_preflight_restores_dispatch_counters(monkeypatch):
    from latentslate_engine.stored_quant import restore_global_fp8_tensor

    model = nn.Module()
    model.first = qwen_runtime.ZImageFullPrecisionFP8Linear(
        restore_global_fp8_tensor(
            torch.zeros((8, 8), dtype=torch.float8_e4m3fn),
            torch.tensor(0.25, dtype=torch.float32),
            torch.bfloat16,
        ),
        ordinal=0,
    )
    model._latentslate_z_image_first_linear_format = "fp8"

    def synthetic_preflight(module, *_args, **_kwargs):
        module.native_dequant_count += 1
        module.f_linear_count += 1
        module.rejected_dispatch_count += 1
        module.per_op_move_count += 1
        module.last_dispatch_error = "private"
        return {"first_linear_preflight": True}

    monkeypatch.setattr(
        qwen_runtime, "_preflight_z_image_full_precision_linear", synthetic_preflight
    )
    proof = qwen_runtime._preflight_z_image_first_linear(
        model, torch.device("cpu"), lambda: False, lambda _stage: None
    )
    assert proof == {"first_linear_preflight": True}
    assert model.first.native_dequant_count == 0
    assert model.first.f_linear_count == 0
    assert model.first.rejected_dispatch_count == 0
    assert model.first.per_op_move_count == 0
    assert model.first.last_dispatch_error is None


def test_mixed_qwen_stage_never_onloads_the_full_module_to_cuda(monkeypatch):
    class CpuMaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.ones(1), requires_grad=False)
            self.to_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def to(self, *args, **kwargs):
            self.to_calls.append((args, kwargs))
            raise AssertionError("the complete Qwen module must remain on CPU")

    model = CpuMaster()
    monkeypatch.setattr(
        qwen_runtime,
        "_preflight_z_image_first_linear",
        lambda *_args: {"first_linear_preflight": True},
    )
    monkeypatch.setattr(qwen_runtime, "z_image_mixed_dispatch_snapshot", lambda _model: {})
    stages: list[str] = []
    stage = qwen_runtime.ZImageMixedQwenStage(model, "cuda:0", diagnostic=stages.append)
    stage.onload()
    assert stages == [
        "conditioning.edge_07",
        "conditioning.edge_08",
        "conditioning.edge_09",
    ]
    assert stage._before == {}
    assert model.anchor.device.type == "cpu"
    assert model.to_calls == []
    stage.offload()
    assert stage._before is None


@pytest.mark.parametrize("edge", range(7, 10))
def test_mixed_qwen_stage_preblock_boundaries_are_cancellable(monkeypatch, edge):
    model = nn.Linear(1, 1, bias=False)
    monkeypatch.setattr(
        qwen_runtime,
        "_preflight_z_image_first_linear",
        lambda *_args: {"first_linear_preflight": True},
    )
    monkeypatch.setattr(qwen_runtime, "z_image_mixed_dispatch_snapshot", lambda _model: {})
    active = {"stage": ""}
    stage = qwen_runtime.ZImageMixedQwenStage(
        model,
        "cuda:0",
        cancelled=lambda: active["stage"] == f"conditioning.edge_{edge:02d}",
        diagnostic=lambda value: active.__setitem__("stage", value),
    )
    with pytest.raises(RuntimeError, match="conditioning canceled"):
        stage.onload()
    assert active["stage"] == f"conditioning.edge_{edge:02d}"
    stage.offload()
    assert stage._before is None
    assert next(model.parameters()).device.type == "cpu"


@pytest.mark.parametrize("edge", (10, 11))
def test_qwen_token_boundaries_are_cancellable_before_cuda_transfer(edge):
    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

    active = {"stage": ""}
    with pytest.raises(RuntimeError, match="conditioning canceled"):
        conditioning.encode_z_image_prompt(
            nn.Identity(),
            Tokenizer(),
            "private prompt",
            device="cuda:0",
            cancelled=lambda: active["stage"] == f"conditioning.edge_{edge:02d}",
            diagnostic=lambda value: active.__setitem__("stage", value),
        )
    assert active["stage"] == f"conditioning.edge_{edge:02d}"


@pytest.mark.parametrize("edge", range(12, 18))
def test_qwen_shell_preblock_boundaries_are_cancellable(edge):
    active = {"stage": ""}

    class TinyEmbedding(nn.Module):
        def set_runtime_callbacks(self, cancelled, diagnostic):
            self.cancelled = cancelled
            self.diagnostic = diagnostic

        def forward(self, ids, *, out_dtype):
            for current in (13, 14):
                self.diagnostic(f"conditioning.edge_{current:02d}")
                if self.cancelled():
                    raise RuntimeError("Z-Image Qwen conditioning canceled")
            return ids.unsqueeze(-1).to(out_dtype).expand(-1, -1, 2).clone()

    backbone = nn.Module()
    backbone.embed_tokens = TinyEmbedding()
    backbone.layers = nn.ModuleList()
    backbone.norm = nn.Identity()
    encoder = qwen_architecture.ZImageQwenTextEncoder.__new__(qwen_architecture.ZImageQwenTextEncoder)
    nn.Module.__init__(encoder)
    encoder.model = backbone
    encoder.final_norm_execution_count = 0
    with pytest.raises(RuntimeError, match="conditioning canceled"):
        encoder.forward_conditioning(
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.tensor([[1, 1]], dtype=torch.long),
            cancelled=lambda: active["stage"] == f"conditioning.edge_{edge:02d}",
            diagnostic=lambda value: active.__setitem__("stage", value),
        )
    assert active["stage"] == f"conditioning.edge_{edge:02d}"


def test_engine_qwen_shell_has_exact_raw_398_key_closure_and_no_lm_head():
    model = qwen_architecture.ZImageQwenTextEncoder(device="meta")
    state = model.state_dict()
    assert len(state) == 398
    assert set(state) == set(qwen_checkpoint.expected_qwen_weight_shapes())
    assert all(key.startswith("model.") for key in state)
    assert not any("lm_head" in key for key in state)
    assert len(model.model.layers) == 36


def test_qwen_dense_linear_casts_bf16_storage_to_rank_preserving_fp32():
    linear = qwen_runtime.ZImageQwenDenseLinear(3, 2, device="cpu", ordinal=0)
    linear.weight = nn.Parameter(torch.ones((2, 3), dtype=torch.bfloat16), requires_grad=False)
    value = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    output = linear(value)
    assert output.dtype is torch.float32
    assert output.shape == (2, 2, 2)
    assert torch.equal(output, value.sum(dim=-1, keepdim=True).expand(-1, -1, 2))
    assert linear.weight.device.type == "cpu" and linear.weight.dtype is torch.bfloat16
    assert linear.per_op_move_count == 1


def test_qwen_embedding_moves_bf16_weight_per_operation_then_returns_fp32():
    embedding = qwen_runtime.ZImageQwenEmbedding(device="meta")
    master = nn.Parameter(
        torch.arange(15, dtype=torch.bfloat16).reshape(5, 3), requires_grad=False
    )
    embedding.weight = master
    stages: list[str] = []
    embedding.set_runtime_callbacks(lambda: False, stages.append)
    output = embedding(torch.tensor([[0, 3, 4]], dtype=torch.long), out_dtype=torch.float32)
    assert output.dtype is torch.float32 and output.shape == (1, 3, 3)
    assert torch.equal(output, master[[0, 3, 4]].float().unsqueeze(0))
    assert embedding.weight is master
    assert embedding.weight.device.type == "cpu" and embedding.weight.dtype is torch.bfloat16
    assert embedding.per_op_move_count == 1
    assert stages == ["conditioning.edge_13", "conditioning.edge_14"]


def test_qwen_gqa_matches_explicit_kv_repeat_on_cpu():
    generator = torch.Generator().manual_seed(4)
    query = torch.randn((1, 4, 3, 2), generator=generator)
    key = torch.randn((1, 2, 3, 2), generator=generator)
    value = torch.randn((1, 2, 3, 2), generator=generator)
    mask = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
    actual = qwen_architecture.qwen_gqa_attention(query, key, value, mask)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key.repeat_interleave(2, dim=1),
        value.repeat_interleave(2, dim=1),
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_qwen_36_block_capture_mask_final_norm_and_cancellation_boundaries():
    masks: list[torch.Tensor] = []

    class TinyEmbedding(nn.Module):
        def forward(self, ids, *, out_dtype):
            return ids.unsqueeze(-1).to(out_dtype).expand(-1, -1, 2).clone()

    class TinyBlock(nn.Module):
        def forward(self, hidden, additive_mask, _frequencies):
            masks.append(additive_mask.clone())
            return hidden + 1

    class TinyNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden):
            self.calls += 1
            return hidden * 2

    backbone = nn.Module()
    backbone.embed_tokens = TinyEmbedding()
    backbone.layers = nn.ModuleList(TinyBlock() for _ in range(36))
    backbone.norm = TinyNorm()
    encoder = qwen_architecture.ZImageQwenTextEncoder.__new__(qwen_architecture.ZImageQwenTextEncoder)
    nn.Module.__init__(encoder)
    encoder.model = backbone
    encoder.final_norm_execution_count = 0
    ids = torch.tensor([[4, 5, 151643, 7]], dtype=torch.long)
    binary = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    stages: list[str] = []
    result = encoder.forward_conditioning(ids, binary, diagnostic=stages.append)
    expected = ids.unsqueeze(-1).float().expand(-1, -1, 2) + 35
    assert torch.equal(result, expected)
    assert backbone.norm.calls == encoder.final_norm_execution_count == 1
    assert "conditioning.block_35" in stages
    assert "conditioning.edge_18" in stages and "conditioning.edge_19" in stages
    assert stages[-1] == "conditioning.edge_20"
    floor = torch.finfo(torch.float32).min / 4
    expected_mask = torch.tensor(
        [[[[0.0, floor, floor * 2, floor * 2], [0.0, 0.0, floor * 2, floor * 2],
           [0.0, 0.0, floor, floor * 2], [0.0, 0.0, floor, floor]]]],
        dtype=torch.float32,
    )
    assert torch.equal(masks[0], expected_mask)

    active = {"stage": ""}
    with pytest.raises(RuntimeError, match="conditioning canceled"):
        encoder.forward_conditioning(
            ids,
            binary,
            diagnostic=lambda stage: active.__setitem__("stage", stage),
            cancelled=lambda: active["stage"] == "conditioning.block_17",
        )
    assert active["stage"] == "conditioning.block_17"
