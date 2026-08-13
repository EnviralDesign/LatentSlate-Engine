from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from latentslate_engine.artifacts import ArtifactIdentity
from latentslate_engine.config import Settings
from latentslate_engine.protocol import WorkflowKind
from latentslate_engine.resources import (
    ArtifactPrecision,
    ArtifactQuantization,
    ResourceDescriptor,
    ResourceFormat,
    ResourceInventory,
    ResourceKind,
    discover_resources,
)
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.tools import default_registry
from latentslate_engine.tools import wan22_native as native_tool_module
from latentslate_engine.tools.base import ExecutionPlan
from latentslate_engine.tools.wan22_native import (
    NATIVE_WAN14B_I2V_KEY,
    NativeWan14BI2VTool,
)
from latentslate_engine.variants import Wan22I2VRecipeConfig
from latentslate_engine.wan22_recipe import (
    Wan22RecipeValidation,
    Wan22RuntimeRequest,
    validate_native_wan22_i2v_14b_recipe,
    validate_wan22_i2v_14b_recipe,
)


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024 * 1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    return settings


def _support_tree(path: Path) -> Path:
    for relative in (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/spiece.model",
        "transformer/config.json",
        "transformer_2/config.json",
        "text_encoder/config.json",
        "vae/config.json",
    ):
        file = path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"{}" if file.suffix == ".json" else b"sentencepiece")
    return path


def _resource(
    resource_id: str,
    component: str,
    path: Path,
    *,
    resource_format: ResourceFormat = ResourceFormat.SAFETENSORS,
) -> ResourceDescriptor:
    metadata: dict[str, object] = {}
    precision = ArtifactPrecision.UNKNOWN
    quantization = ArtifactQuantization.UNKNOWN
    base_model = None
    if component.startswith("transformer"):
        metadata = {
            "architecture": "wan2.2_i2v_14b",
            "quantization_contract": "comfy_quant/float8_e4m3fn",
            "noise_stage": "high" if component.endswith("high_noise") else "low",
        }
        precision = ArtifactPrecision.FP8
        quantization = ArtifactQuantization.NATIVE
        base_model = "wan22-14b-i2v"
    elif component == "text_encoder":
        metadata = {
            "architecture": "umt5_xxl",
            "quantization_contract": "comfy_legacy/scaled_fp8_e4m3fn",
        }
        precision = ArtifactPrecision.FP8
        quantization = ArtifactQuantization.NATIVE
    elif component == "vae":
        metadata = {
            "architecture": "wan_vae_2_1",
            "quantization_contract": "native/bf16",
        }
        precision = ArtifactPrecision.BF16
        quantization = ArtifactQuantization.NATIVE
    return ResourceDescriptor(
        id=resource_id,
        kind=ResourceKind.MODEL,
        family="wan22",
        name=component,
        relative_path=path.as_posix(),
        format=resource_format,
        precision=precision,
        quantization=quantization,
        size_bytes=path.stat().st_size if path.is_file() else 0,
        component=component,
        base_model=base_model,
        metadata=metadata,
    )


def _inventory(tmp_path: Path) -> ResourceInventory:
    support_path = _support_tree(tmp_path / "support")
    paths = {
        "pipeline_support": support_path,
        "transformer_high_noise": tmp_path / "high.safetensors",
        "transformer_low_noise": tmp_path / "low.safetensors",
        "text_encoder": tmp_path / "text.safetensors",
        "vae": tmp_path / "vae.safetensors",
    }
    for role, path in paths.items():
        if role != "pipeline_support":
            path.write_bytes(role.encode("utf-8"))
    resources = [
        _resource(
            "model:wan22:support",
            "pipeline_support",
            support_path,
            resource_format=ResourceFormat.DIRECTORY,
        ),
        _resource("model:wan22:high", "transformer_high_noise", paths["transformer_high_noise"]),
        _resource("model:wan22:low", "transformer_low_noise", paths["transformer_low_noise"]),
        _resource("model:wan22:text", "text_encoder", paths["text_encoder"]),
        _resource("model:wan22:vae", "vae", paths["vae"]),
    ]
    return ResourceInventory(
        resources=resources,
        paths={resource.id: paths[resource.component or ""] for resource in resources},
    )


def _variant_toml() -> str:
    return """
schema_version = 1
key = "wan22.native.test"
name = "Native Wan 14B I2V Test"
family = "wan22"
base_tool = "wan22.native_image_to_video"

[recipe]
type = "wan22_i2v_14b"
base_model = "wan22-14b-i2v"
pipeline_support = "model:wan22:support"
transformer_high_noise = "model:wan22:high"
transformer_low_noise = "model:wan22:low"
text_encoder = "model:wan22:text"
vae = "model:wan22:vae"

[optimizations]
keep_pipeline_loaded = false
"""


def test_pipeline_support_is_recipe_only_directory_resource(tmp_path: Path):
    settings = _settings(tmp_path)
    support = _support_tree(settings.model_root / "wan22" / "native-support")

    inventory = discover_resources(settings)
    descriptor = next(
        resource
        for resource in inventory.resources
        if resource.family == "wan22" and resource.component == "pipeline_support"
    )

    assert inventory.path_for(descriptor.id) == support.resolve()
    assert descriptor.kind == ResourceKind.MODEL
    assert descriptor.format == ResourceFormat.DIRECTORY
    assert inventory.matching(kind=ResourceKind.MODEL, family="wan22") == []
    assert inventory.matching(
        kind=ResourceKind.MODEL,
        family="wan22",
        include_components=True,
    ) == [descriptor]


def test_dense_diffusers_directory_is_not_inferred_as_pipeline_support(
    tmp_path: Path,
):
    settings = _settings(tmp_path)
    dense = _support_tree(settings.model_root / "wan22" / "dense-pipeline")
    weight = dense / "transformer" / "diffusion_pytorch_model.safetensors"
    weight.write_bytes(b"dense")

    inventory = discover_resources(settings)
    descriptor = next(
        resource for resource in inventory.resources if inventory.path_for(resource.id) == dense
    )

    assert descriptor.component is None
    assert descriptor.format == ResourceFormat.DIFFUSERS
    assert descriptor in inventory.matching(kind=ResourceKind.MODEL, family="wan22")


def test_recipe_config_requires_every_semantic_role():
    with pytest.raises(Exception, match="pipeline_support"):
        Wan22I2VRecipeConfig(
            base_model="wan22-14b-i2v",
            transformer_high_noise="model:wan22:high",
            transformer_low_noise="model:wan22:low",
            text_encoder="model:wan22:text",
            vae="model:wan22:vae",
        )


def test_lightx_recipe_config_keeps_stage_bindings_explicit() -> None:
    config = Wan22I2VRecipeConfig(
        base_model="wan22-14b-i2v",
        pipeline_support="model:wan22:support",
        transformer_high_noise="model:wan22:high",
        transformer_low_noise="model:wan22:low",
        text_encoder="model:wan22:text",
        vae="model:wan22:vae",
        operation="comfy_i2v_lightx2v_4step",
        lora_stage_by_slot={"high_noise": "high", "low_noise": "low"},
    )

    assert config.operation == "comfy_i2v_lightx2v_4step"
    assert config.lora_stage_by_slot == {"high_noise": "high", "low_noise": "low"}


def test_recipe_fails_closed_without_pipeline_support(tmp_path: Path):
    inventory = _inventory(tmp_path)
    config = Wan22I2VRecipeConfig(
        base_model="wan22-14b-i2v",
        pipeline_support="model:wan22:support",
        transformer_high_noise="model:wan22:high",
        transformer_low_noise="model:wan22:low",
        text_encoder="model:wan22:text",
        vae="model:wan22:vae",
    )
    del config  # construction proves all five references are required by the grammar

    from latentslate_engine.wan22_recipe import Wan22I2VRecipe, Wan22RecipeComponent

    by_id = inventory.by_id()
    recipe = Wan22I2VRecipe(
        base_model="wan22-14b-i2v",
        high_noise=Wan22RecipeComponent(
            by_id["model:wan22:high"], inventory.path_for("model:wan22:high")
        ),
        low_noise=Wan22RecipeComponent(
            by_id["model:wan22:low"], inventory.path_for("model:wan22:low")
        ),
        text_encoder=Wan22RecipeComponent(
            by_id["model:wan22:text"], inventory.path_for("model:wan22:text")
        ),
        vae=Wan22RecipeComponent(by_id["model:wan22:vae"], inventory.path_for("model:wan22:vae")),
        pipeline_support=None,
    )

    generic = validate_wan22_i2v_14b_recipe(recipe, inventory)
    result = validate_native_wan22_i2v_14b_recipe(recipe, inventory)

    assert generic.available is False  # synthetic files are intentionally invalid
    assert not result.available
    assert "native Wan execution requires pipeline support" in result.errors


def test_native_recipe_variant_is_cataloged_but_hidden_base_is_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(tmp_path)
    inventory = _inventory(tmp_path / "resources")
    variant_path = settings.variants_root / "wan22" / "native-test.toml"
    variant_path.write_text(_variant_toml(), encoding="utf-8")
    support_plan = SimpleNamespace(
        root=inventory.path_for("model:wan22:support"),
        fingerprint="support:sha256:test",
        tokenizer_sha256="a" * 64,
        files=(1, 2, 3, 4, 5, 6, 7),
    )

    def valid_recipe(recipe, recipe_inventory, *, include_adapter_plans=True):
        resolved = {
            "pipeline_support": recipe.pipeline_support,
            "transformer_high_noise": recipe.high_noise,
            "transformer_low_noise": recipe.low_noise,
            "text_encoder": recipe.text_encoder,
            "vae": recipe.vae,
        }
        assert recipe_inventory is inventory
        assert include_adapter_plans is False
        return Wan22RecipeValidation(True, (), (), resolved, support_plan)

    monkeypatch.setattr(native_tool_module, "_native_runtime_availability", lambda: (True, None))
    monkeypatch.setattr("latentslate_engine.tools.discover_resources", lambda _settings: inventory)
    monkeypatch.setattr(
        "latentslate_engine.variants.validate_native_wan22_i2v_14b_recipe",
        valid_recipe,
    )

    registry = default_registry(settings, emit_warnings=False)
    repeated = default_registry(settings, emit_warnings=False)
    descriptors = {descriptor.key: descriptor for descriptor in registry.descriptors()}
    repeated_descriptors = {descriptor.key: descriptor for descriptor in repeated.descriptors()}

    assert "wan22.text_to_video" not in descriptors
    assert NATIVE_WAN14B_I2V_KEY not in descriptors
    native = descriptors["wan22.native.test"]
    assert native.available
    assert native.workflow_kind == WorkflowKind.IMAGE_TO_VIDEO
    assert native.schema_hash
    assert repeated_descriptors["wan22.native.test"].schema_hash == native.schema_hash
    assert {input_.key for input_ in native.inputs} == {
        "source_image",
        "prompt",
        "negative_prompt",
        "num_frames",
        "width",
        "height",
        "steps",
        "seed",
        "stage_policy",
        "high_guidance",
        "low_guidance",
    }
    entry = next(entry for entry in registry.variants if entry.key == "wan22.native.test")
    assert entry.recipe_type == "wan22_i2v_14b"
    assert entry.recipe_resources["pipeline_support"] == "model:wan22:support"


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("PIL") is None,
    reason="runtime request validation requires the locked runtime group",
)
def test_invalid_native_request_is_rejected_before_runtime_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from latentslate_engine.runtime.wan22_i2v_runtime import WanI2VRequest
    from latentslate_engine.runtime.wan22_native_managed import (
        ManagedNativeWanI2VRuntime,
    )

    recipe = SimpleNamespace(fingerprint="recipe:test")
    managed = ManagedNativeWanI2VRuntime(recipe)  # type: ignore[arg-type]
    monkeypatch.setattr(
        "latentslate_engine.runtime.wan22_native_managed.revalidate_runtime_request",
        lambda _request: True,
    )
    request = WanI2VRequest(
        image=None,
        prompt="move",
        num_frames=6,
        height=64,
        width=64,
        steps=4,
    )

    with pytest.raises(ValueError, match=r"4k\+1"):
        managed.generate(
            request,
            source_image_path=tmp_path / "missing.png",
            output_path=tmp_path / "output.mp4",
            device="cpu",
            fps=16,
        )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("PIL") is None,
    reason="runtime dispatch smoke requires the locked runtime group",
)
def test_native_tool_dispatches_runtime_and_atomic_serializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from PIL import Image

    import latentslate_engine.runtime.wan22_native_managed as managed_module
    from latentslate_engine.storage import Storage
    from latentslate_engine.tools.base import ToolContext

    settings = _settings(tmp_path)
    support_root = _support_tree(tmp_path / "support")
    artifact_paths = {}
    identities = {}
    components: dict[str, dict[str, str | int]] = {
        "pipeline_support": {
            "resource_id": "model:wan22:support",
            "path": str(support_root),
            "format": "directory",
            "component": "pipeline_support",
            "support_fingerprint": "support:sha256:test",
            "tokenizer_sha256": "a" * 64,
            "file_count": 7,
        }
    }
    for role in ("transformer_high_noise", "transformer_low_noise", "text_encoder", "vae"):
        path = tmp_path / f"{role}.safetensors"
        path.write_bytes(role.encode("utf-8"))
        artifact_paths[role] = path
        identity = ArtifactIdentity(path, path.stat().st_size, path.stat().st_mtime_ns, role)
        identities[role] = identity
        components[role] = {
            "resource_id": f"model:wan22:{role}",
            "path": str(path),
            "format": "safetensors",
            "component": role,
            "quantization_contract": "test",
            "size_bytes": identity.size_bytes,
            "mtime_ns": identity.mtime_ns,
            "header_sha256": identity.header_sha256,
            "schema_sha256": role,
        }
    support_plan = SimpleNamespace(
        root=support_root,
        fingerprint="support:sha256:test",
        tokenizer_sha256="a" * 64,
        files=(1, 2, 3, 4, 5, 6, 7),
    )
    recipe = Wan22RuntimeRequest(
        2,
        "wan22",
        "wan22_14b_36ch_40block_out16",
        "wan22-14b-i2v",
        components,
        identities,
        support_plan,
    )
    captured: dict[str, object] = {}

    provenance = {"sampler": "euler", "scheduler": "simple", "shift": 5.0}

    class FakeManagedRuntime:
        def __init__(self, runtime_recipe):
            captured["recipe"] = runtime_recipe

        def generate(
            self,
            request,
            *,
            source_image_path,
            output_path,
            device,
            fps,
            progress,
            cancelled,
        ):
            captured["request"] = request
            captured["source_image_path"] = source_image_path
            captured["device"] = device
            captured["fps"] = fps
            assert not cancelled()
            progress(1, 4, "high")
            output_path.write_bytes(b"mp4")
            return SimpleNamespace(
                provenance=provenance,
                worker_pid=42,
                worker_exit_code=0,
            )

        def status(self):
            return {"loaded": False}

        def unload(self):
            captured["unloaded"] = True

        def clear_cache(self):
            pass

    monkeypatch.setattr(managed_module, "ManagedNativeWanI2VRuntime", FakeManagedRuntime)

    asset_id = uuid4()
    asset_folder = settings.assets_dir / str(asset_id)
    asset_folder.mkdir(parents=True)
    source = asset_folder / "source.png"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(source)
    progress_events = []
    context = ToolContext(
        job_id=uuid4(),
        settings=settings,
        storage=Storage(settings),
        cancel_event=SimpleNamespace(is_set=lambda: False),
        progress=lambda value, message: progress_events.append((value, message)),
        execution=ExecutionPlan(
            variant_key="wan22.native.test",
            family="wan22",
            optimizations={"keep_pipeline_loaded": False},
            recipe=recipe,
        ),
    )
    RUNTIME_MANAGER.clear()
    try:
        artifacts = NativeWan14BI2VTool().run(
            context,
            {
                "source_image": {"asset_id": str(asset_id)},
                "prompt": "move the camera forward",
                "negative_prompt": "static",
                "num_frames": 5,
                "width": 64,
                "height": 64,
                "steps": 4,
                "seed": 7,
                "stage_policy": "comfy_split",
                "high_guidance": 1.0,
                "low_guidance": 1.0,
            },
        )
    finally:
        RUNTIME_MANAGER.clear()

    assert captured["recipe"] is recipe
    request = captured["request"]
    assert request.num_frames == 5
    assert request.stage_policy == "comfy_split"
    assert captured["device"] == "cuda"
    assert captured["fps"] == 16
    assert captured["source_image_path"] == source
    assert len(artifacts) == 1
    assert artifacts[0].path.read_bytes() == b"mp4"
    assert artifacts[0].metadata["recipe_fingerprint"] == recipe.fingerprint
    assert artifacts[0].metadata["fps"] == 16
    assert artifacts[0].metadata["duration_seconds"] == 5 / 16
    assert context.runtime_provenance["runtime_result"]["pipeline_warm"] is False
    assert context.runtime_provenance["runtime_result"]["execution_cache"] == {
        "supported": False,
        "hit": False,
        "mode": "fresh_disposable_process",
    }
    assert context.runtime_provenance["runtime_result"]["worker"]["terminated"] is True
    assert progress_events[-1] == (1.0, "Complete")
