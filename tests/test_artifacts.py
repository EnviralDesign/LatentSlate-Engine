from __future__ import annotations

import json
import math
import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from latentslate_engine.artifacts import probe_artifact
from latentslate_engine.resources import (
    ArtifactPrecision,
    ArtifactQuantization,
    ResourceDescriptor,
    ResourceFormat,
    ResourceInventory,
    ResourceKind,
)
from latentslate_engine.wan22_recipe import (
    Wan22I2VRecipe,
    Wan22RecipeComponent,
    build_wan22_i2v_14b_runtime_request,
    revalidate_runtime_request,
    validate_native_wan22_i2v_14b_recipe,
    validate_wan22_i2v_14b_recipe,
)


def _safetensors(path: Path, header: dict) -> Path:
    header = json.loads(json.dumps(header))
    sizes = {"U8": 1, "I8": 1, "F8_E4M3": 1, "F16": 2, "BF16": 2, "F32": 4}
    offset = 0
    for key, value in header.items():
        if key == "__metadata__":
            continue
        size = math.prod(value["shape"]) * sizes[value["dtype"]]
        value["data_offsets"] = [offset, offset + size]
        offset += size
    encoded = json.dumps(header).encode("utf-8")
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        if offset:
            stream.seek(offset - 1, 1)
            stream.write(b"\0")
    return path


def _wan_header(*, prefix: str = "", dtype: str = "F16") -> dict:
    header = {
        f"{prefix}patch_embedding.weight": {"dtype": dtype, "shape": [5120, 36, 1, 2, 2]},
        f"{prefix}head.modulation": {"dtype": "F16", "shape": [1, 2, 5120]},
        f"{prefix}head.head.weight": {"dtype": "F16", "shape": [64, 5120]},
        f"{prefix}blocks.0.cross_attn.q.weight": {"dtype": dtype, "shape": [5120, 5120]},
    }
    header.update(
        {
            f"{prefix}blocks.{index}.self_attn.q.weight": {
                "dtype": dtype,
                "shape": [5120, 5120],
            }
            for index in range(40)
        }
    )
    return header


def _umt5_header(*, dtype: str = "F16") -> dict:
    header = {
        "spiece_model": {"dtype": "U8", "shape": [4548313]},
        "encoder.final_layer_norm.weight": {"dtype": dtype if dtype == "BF16" else "F16", "shape": [4096]},
        "encoder.block.0.layer.0.SelfAttention.q.weight": {"dtype": dtype, "shape": [4096, 4096]},
        "encoder.block.0.layer.1.DenseReluDense.wi_0.scale_weight": {"dtype": "F32", "shape": [1]},
    }
    header.update(
        {
            f"encoder.block.{index}.layer.0.SelfAttention.q.scale_weight": {
                "dtype": dtype,
                "shape": [1],
            }
            for index in range(24)
        }
    )
    return header


def _resource(
    resource_id: str,
    component: str,
    path: Path,
    *,
    base_model: str = "wan22-14b-i2v-int8-convrot",
    contract: str = "comfy_quant/int8_tensorwise_convrot",
    architecture: str | None = None,
) -> ResourceDescriptor:
    if architecture is None:
        architecture = "wan2.2_i2v_14b" if component.startswith("transformer") else "umt5_xxl"
    metadata = {"architecture": architecture, "quantization_contract": contract}
    if component.endswith("high_noise"):
        metadata["noise_stage"] = "high"
    if component.endswith("low_noise"):
        metadata["noise_stage"] = "low"
    precision = ArtifactPrecision.UNKNOWN
    quantization = ArtifactQuantization.INT8
    if contract in {"comfy_quant/float8_e4m3fn", "comfy_legacy/scaled_fp8_e4m3fn"}:
        precision, quantization = ArtifactPrecision.FP8, ArtifactQuantization.NATIVE
    elif contract == "native/bf16":
        precision, quantization = ArtifactPrecision.BF16, ArtifactQuantization.NATIVE
    return ResourceDescriptor(
        id=resource_id,
        kind=ResourceKind.MODEL,
        family="wan22",
        name=component,
        relative_path=str(path),
        format=ResourceFormat.SAFETENSORS,
        size_bytes=path.stat().st_size,
        precision=precision,
        quantization=quantization,
        component=component,
        base_model=base_model,
        metadata=metadata,
    )


def _inventory_for(recipe: Wan22I2VRecipe) -> ResourceInventory:
    components = [recipe.high_noise, recipe.low_noise, recipe.text_encoder, recipe.vae]
    if recipe.pipeline_support is not None:
        components.append(recipe.pipeline_support)
    return ResourceInventory(
        resources=[component.resource for component in components],
        paths={component.resource.id: component.path for component in components},
    )


def _pipeline_support(path: Path) -> Wan22RecipeComponent:
    path.mkdir(parents=True, exist_ok=True)
    resource = ResourceDescriptor(
        id="model:wan22:pipeline-support",
        kind=ResourceKind.MODEL,
        family="wan22",
        name="pipeline support",
        relative_path=path.as_posix(),
        format=ResourceFormat.DIRECTORY,
        size_bytes=0,
        component="pipeline_support",
    )
    return Wan22RecipeComponent(resource, path)


def test_safetensors_probe_detects_strict_wan_signature_and_convrot(tmp_path: Path):
    header = _wan_header(dtype="I8")
    layers = {f"blocks.{index}.self_attn.q": {"format": "int8_tensorwise", "convrot": True} for index in range(40)}
    header["__metadata__"] = {"_quantization_metadata": json.dumps({"layers": layers})}
    for index in range(40):
        header[f"blocks.{index}.self_attn.q.weight_scale"] = {"dtype": "F32", "shape": [1]}
        header[f"blocks.{index}.self_attn.q.comfy_quant"] = {"dtype": "U8", "shape": [1]}
    path = _safetensors(tmp_path / "high.safetensors", header)

    report = probe_artifact(path)

    assert report.family_signals == ("wan22",)
    assert report.architecture_signals == ("wan22_14b_36ch_40block_out16",)
    assert report.component_signals == ("transformer",)
    assert report.quantization_contract == "comfy_quant/int8_tensorwise_convrot"
    assert report.key_prefix == ""
    assert report.key_shape_signals["patch_embedding.weight"] == "5120x36x1x2x2"
    assert report.key_shape_signals["head.modulation"] == "1x2x5120"
    assert report.key_shape_signals["head.head.weight"] == "64x5120"
    assert report.key_shape_signals["transformer_block_count"] == 40


def test_safetensors_probe_distinguishes_t2v_16_channel_wan14(tmp_path: Path):
    header = _wan_header(dtype="F8_E4M3")
    header["patch_embedding.weight"] = {"dtype": "F8_E4M3", "shape": [5120, 16, 1, 2, 2]}
    report = probe_artifact(_safetensors(tmp_path / "t2v.safetensors", header))
    assert report.architecture_signals == ("wan22_14b_16ch_40block_out16",)


def test_safetensors_probe_detects_comfy_fp8_and_legacy_scaled_fp8(tmp_path: Path):
    fp8 = _wan_header(prefix="model.diffusion_model.", dtype="F8_E4M3")
    for index in range(40):
        fp8[f"model.diffusion_model.blocks.{index}.self_attn.q.comfy_quant"] = {"dtype": "U8", "shape": [1]}
        fp8[f"model.diffusion_model.blocks.{index}.self_attn.q.weight_scale"] = {"dtype": "F32", "shape": [1]}
    fp8["model.diffusion_model.blocks.0.cross_attn.q.comfy_quant"] = {"dtype": "U8", "shape": [1]}
    fp8["model.diffusion_model.blocks.0.cross_attn.q.weight_scale"] = {"dtype": "F32", "shape": [1]}
    smoothmix = probe_artifact(_safetensors(tmp_path / "smoothmix.safetensors", fp8))
    legacy = probe_artifact(
        _safetensors(
            tmp_path / "umt5.safetensors",
            _umt5_header(dtype="F8_E4M3"),
        )
    )

    assert smoothmix.quantization_contract == "comfy_quant/float8_e4m3fn"
    assert smoothmix.key_prefix == "model.diffusion_model."
    assert smoothmix.architecture_signals == ("wan22_14b_36ch_40block_out16",)
    assert legacy.quantization_contract == "comfy_legacy/scaled_fp8_e4m3fn"
    assert legacy.component_signals == ("text_encoder",)


def test_gguf_probe_maps_q5_k_m_and_preserves_file_type(tmp_path: Path):
    values = ((b"general.architecture", b"wan"),)
    path = tmp_path / "wan.gguf"
    table = (
        b"GGUF"
        + struct.pack("<IQQ", 3, 2, 2)
        + b"".join(
            struct.pack("<Q", len(key)) + key + struct.pack("<I", 8) + struct.pack("<Q", len(value)) + value
            for key, value in values
        )
        + struct.pack("<Q", len(b"general.file_type"))
        + b"general.file_type"
        + struct.pack("<II", 4, 17)
    )
    tensor = (
        struct.pack("<Q", 8) + b"x.weight" + struct.pack("<IQQIQ", 2, 16, 16, 13, 0)
        + struct.pack("<Q", 8) + b"y.weight" + struct.pack("<IQQIQ", 2, 16, 16, 14, 192)
    )
    payload_start = (len(table + tensor) + 31) // 32 * 32
    path.write_bytes(table + tensor + b"\0" * (payload_start - len(table + tensor) + 402))

    report = probe_artifact(path)

    assert report.quantization_contract == "gguf/q5_k_m"
    assert report.metadata["general.file_type"] == 17
    assert report.architecture_signals == ()


def test_recipe_requires_explicit_vae_but_allows_role_specific_text_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    high_header = _wan_header(dtype="F8_E4M3")
    low_header = _wan_header(dtype="F8_E4M3")
    for header in (high_header, low_header):
        for index in range(40):
            header[f"blocks.{index}.self_attn.q.comfy_quant"] = {"dtype": "U8", "shape": [1]}
            header[f"blocks.{index}.self_attn.q.weight_scale"] = {"dtype": "F32", "shape": [1]}
        header["blocks.0.cross_attn.q.comfy_quant"] = {"dtype": "U8", "shape": [1]}
        header["blocks.0.cross_attn.q.weight_scale"] = {"dtype": "F32", "shape": [1]}
    high = _safetensors(tmp_path / "high.safetensors", high_header)
    low = _safetensors(tmp_path / "low.safetensors", low_header)
    text = _safetensors(tmp_path / "text.safetensors", _umt5_header(dtype="BF16"))
    vae = _safetensors(
        tmp_path / "vae.safetensors",
        {
            "decoder.middle.0.residual.0.gamma": {"dtype": "BF16", "shape": [384, 1, 1, 1]},
            "decoder.conv1.weight": {"dtype": "BF16", "shape": [384, 16, 3, 3, 3]},
            "encoder.head.2.weight": {"dtype": "BF16", "shape": [32, 384, 3, 3, 3]},
        },
    )
    support = _pipeline_support(tmp_path / "support")
    support_plan = SimpleNamespace(
        root=support.path,
        fingerprint="support:sha256:test",
        tokenizer_sha256="a" * 64,
        files=(1, 2, 3, 4, 5, 6, 7),
    )
    monkeypatch.setattr(
        "latentslate_engine.wan22_recipe._plan_pipeline_support",
        lambda _path: support_plan,
    )
    monkeypatch.setattr(
        "latentslate_engine.wan22_recipe._revalidate_pipeline_support",
        lambda _plan: True,
    )
    recipe = Wan22I2VRecipe(
        base_model="wan22-14b-i2v-int8-convrot",
        high_noise=Wan22RecipeComponent(_resource("model:wan22:high", "transformer_high_noise", high, contract="comfy_quant/float8_e4m3fn"), high),
        low_noise=Wan22RecipeComponent(_resource("model:wan22:low", "transformer_low_noise", low, contract="comfy_quant/float8_e4m3fn"), low),
        text_encoder=Wan22RecipeComponent(_resource("model:wan22:text", "text_encoder", text, contract="native/bf16"), text),
        vae=Wan22RecipeComponent(_resource("model:wan22:vae", "vae", vae, base_model="wan22-14b-i2v", contract="native/bf16", architecture="wan_vae_2_1"), vae),
        pipeline_support=support,
    )

    inventory = _inventory_for(recipe)
    result = validate_wan22_i2v_14b_recipe(recipe, inventory)

    assert result.available
    assert len(result.probes) == 4
    assert result.support_plan is support_plan
    request = build_wan22_i2v_14b_runtime_request(recipe, inventory)
    assert request.components["vae"]["path"] == str(vae)
    assert request.components["pipeline_support"]["path"] == str(support.path)
    assert revalidate_runtime_request(request)

    portable = replace(recipe, pipeline_support=None)
    portable_result = validate_wan22_i2v_14b_recipe(
        portable,
        _inventory_for(portable),
    )
    assert portable_result.available
    assert portable_result.support_plan is None

    probe_by_path = {probe.path: probe for probe in result.probes}

    class FakePlan:
        def __init__(self, path: Path, contract: str):
            self.identity = probe_by_path[path].identity
            self.artifact_contract = contract

        def require_available(self):
            return None

    def unsupported_text(_path: Path):
        raise ValueError("unsupported artifact contract 'native/bf16'")

    monkeypatch.setattr(
        "latentslate_engine.wan22_recipe._native_adapter_planners",
        lambda: {
            "transformer_high_noise": lambda path: FakePlan(
                path, "comfy_quant/float8_e4m3fn"
            ),
            "transformer_low_noise": lambda path: FakePlan(
                path, "comfy_quant/float8_e4m3fn"
            ),
            "text_encoder": unsupported_text,
            "vae": lambda path: FakePlan(path, "native/bf16"),
        },
    )
    native_result = validate_native_wan22_i2v_14b_recipe(recipe, inventory)
    assert not native_result.available
    assert any(
        "native text_encoder does not support stored contract 'native/bf16'" in error
        for error in native_result.errors
    )


def test_recipe_rejects_mismatched_transformer_architecture_signature(tmp_path: Path):
    high = _safetensors(tmp_path / "high.safetensors", _wan_header())
    low = _safetensors(tmp_path / "low.safetensors", {"blocks.0.self_attn.q.weight": {"dtype": "F16", "shape": [1]}})
    text = _safetensors(tmp_path / "text.safetensors", {"encoder.block.0.q": {"dtype": "F16", "shape": [1]}})
    vae = _safetensors(tmp_path / "vae.safetensors", {"vae.decoder.conv1.weight": {"dtype": "BF16", "shape": [1]}})
    recipe = Wan22I2VRecipe(
        "wan22-14b-i2v-int8-convrot",
        Wan22RecipeComponent(_resource("model:wan22:high", "transformer_high_noise", high), high),
        Wan22RecipeComponent(_resource("model:wan22:low", "transformer_low_noise", low), low),
        Wan22RecipeComponent(_resource("model:wan22:text", "text_encoder", text), text),
        Wan22RecipeComponent(_resource("model:wan22:vae", "vae", vae, base_model="wan22-14b-i2v", contract="native/bf16", architecture="wan_vae_2_1"), vae),
    )

    result = validate_wan22_i2v_14b_recipe(recipe, _inventory_for(recipe))

    assert not result.available
    assert any("architecture signature" in error for error in result.errors)


def test_recipe_components_are_excluded_from_ordinary_model_selection(tmp_path: Path):
    path = _safetensors(tmp_path / "high.safetensors", _wan_header())
    resource = _resource("model:wan22:high", "transformer_high_noise", path)
    inventory = ResourceInventory(resources=[resource])

    assert inventory.matching(kind=ResourceKind.MODEL, family="wan22") == []
    assert inventory.matching(kind=ResourceKind.MODEL, family="wan22", include_components=True) == [resource]


def test_arbitrary_decoder_and_encoder_keys_do_not_claim_wan_component_architectures(tmp_path: Path):
    arbitrary_decoder = probe_artifact(
        _safetensors(
            tmp_path / "decoder.safetensors",
            {"decoder.conv1.weight": {"dtype": "BF16", "shape": [4, 4, 3, 3, 3]}},
        )
    )
    arbitrary_encoder = probe_artifact(
        _safetensors(
            tmp_path / "encoder.safetensors",
            {"encoder.block.0.layer.0.SelfAttention.q.weight": {"dtype": "F16", "shape": [1]}},
        )
    )

    assert "vae" not in arbitrary_decoder.component_signals
    assert "wan_vae_2_1" not in arbitrary_decoder.architecture_signals
    assert "text_encoder" not in arbitrary_encoder.component_signals
    assert "umt5_xxl" not in arbitrary_encoder.architecture_signals


def test_safetensors_rejects_offset_mismatch_and_truncated_header(tmp_path: Path):
    malformed = tmp_path / "bad-offsets.safetensors"
    header = {"x": {"dtype": "U8", "shape": [2], "data_offsets": [0, 3]}}
    encoded = json.dumps(header).encode("utf-8")
    malformed.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0\0\0")
    with pytest.raises(ValueError, match="bounds mismatch"):
        probe_artifact(malformed)

    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(struct.pack("<Q", 16) + b"{}")
    with pytest.raises(ValueError, match="truncated"):
        probe_artifact(truncated)


def test_claimed_int8_requires_real_int8_payload_dtype(tmp_path: Path):
    header = _wan_header(dtype="F16")
    header["__metadata__"] = {
        "_quantization_metadata": json.dumps(
            {"layers": {"blocks.0.self_attn.q": {"format": "int8_tensorwise", "convrot": True}}}
        )
    }
    artifact = _safetensors(tmp_path / "f16-claim.safetensors", header)
    probe = probe_artifact(artifact)
    assert probe.quantization_contract == "native/fp16"


def test_claimed_int8_requires_quantized_weight_coverage(tmp_path: Path):
    header = _wan_header(dtype="F16")
    stem = "blocks.0.self_attn.q"
    header[f"{stem}.weight"]["dtype"] = "I8"
    header[f"{stem}.weight_scale"] = {"dtype": "F32", "shape": [1]}
    header[f"{stem}.comfy_quant"] = {"dtype": "U8", "shape": [1]}
    header["__metadata__"] = {
        "_quantization_metadata": json.dumps(
            {"layers": {stem: {"format": "int8_tensorwise", "convrot": True}}}
        )
    }

    probe = probe_artifact(_safetensors(tmp_path / "mostly-f16.safetensors", header))

    assert probe.quantization_contract is None
