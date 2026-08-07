import sys
from types import ModuleType, SimpleNamespace

from latentslate_engine.bundles import BUNDLES
from latentslate_engine.config import Settings
from latentslate_engine.protocol import InputRole, WorkflowKind
from latentslate_engine.runtime.klein import KLEIN_SIZE_PRESETS, KleinRuntime
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.tools import klein as klein_tools


def test_klein_sizes_are_valid_flux2_canvases():
    assert "512x512" in KLEIN_SIZE_PRESETS
    for name, size in KLEIN_SIZE_PRESETS.items():
        if name == "source":
            assert size.width is None
            assert size.height is None
            continue
        assert size.width is not None
        assert size.height is not None
        assert size.width % 16 == 0
        assert size.height % 16 == 0


def test_klein_tools_follow_latentslate_taxonomy():
    text4 = klein_tools.Klein4BTextToImageTool().descriptor
    edit4 = klein_tools.Klein4BImageToImageTool().descriptor
    text9 = klein_tools.KleinTextToImageTool().descriptor
    edit9 = klein_tools.KleinImageToImageTool().descriptor

    assert text4.name == "Klein 4B Text to Image"
    assert text4.workflow_kind == WorkflowKind.TEXT_TO_IMAGE
    assert edit4.name == "Klein 4B Image to Image (1-3 refs)"
    assert edit4.workflow_kind == WorkflowKind.IMAGE_TO_IMAGE
    assert text9.name == "Klein 9B Text to Image"
    assert edit9.name == "Klein 9B Image to Image (1-3 refs)"

    text4_inputs = {item.key: item for item in text4.inputs}
    text9_inputs = {item.key: item for item in text9.inputs}
    edit4_inputs = {item.key: item for item in edit4.inputs}
    edit9_inputs = {item.key: item for item in edit9.inputs}

    assert text4_inputs["size"].default == "512x512"
    assert text9_inputs["size"].default == "1024x1024"
    assert "source" not in {option.value for option in text4_inputs["size"].options}
    assert "source" in {option.value for option in edit4_inputs["size"].options}
    assert edit4_inputs["source_image"].role == InputRole.SOURCE_IMAGE
    assert edit4_inputs["reference_image_2"].required is False
    assert edit4_inputs["reference_image_3"].required is False
    assert edit9_inputs["reference_image_2"].required is False
    assert edit9_inputs["reference_image_3"].required is False


def test_klein_bundles_cover_4b_and_consumer_9b():
    bundle4 = BUNDLES["klein4b-basic"]
    assert bundle4.repo_id == "black-forest-labs/FLUX.2-klein-4B"
    assert bundle4.required_repo_ids() == {
        "black-forest-labs/FLUX.2-klein-4B"
    }

    bundle9 = BUNDLES["klein9b-basic"]
    assert bundle9.repo_id == "black-forest-labs/FLUX.2-klein-9B"
    assert bundle9.required_repo_ids() == {
        "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-9b-nvfp4",
        "Qwen/Qwen3-8B-FP8",
    }
    assert bundle9.files[0].filename == "flux-2-klein-9b-nvfp4.safetensors"
    assert "transformer/config.json" in bundle9.allow_patterns
    assert "transformer/**" not in bundle9.allow_patterns
    assert "text_encoder/**" not in bundle9.allow_patterns


def test_klein_defaults_prioritize_4b_and_keep_9b_blackwell_path(tmp_path):
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="consumer_int8",
        h3_device="cuda",
    )
    assert settings.klein4b_model_id.endswith("FLUX.2-klein-4B")
    assert settings.klein4b_profile == "bf16_model_offload"
    assert settings.klein_profile == "consumer_nvfp4"
    assert settings.klein_transformer_model_id.endswith("klein-9b-nvfp4")
    assert settings.klein_text_encoder_model_id == "Qwen/Qwen3-8B-FP8"


def test_klein_tools_share_runtime_within_variant_and_evict_between_variants(monkeypatch):
    created = []

    class FakeRuntime:
        def __init__(self, settings, variant):
            self.settings = settings
            self.variant = variant
            self.unloaded = False
            created.append(self)

        def unload(self):
            self.unloaded = True

    RUNTIME_MANAGER.clear()
    monkeypatch.setattr(klein_tools, "KleinRuntime", FakeRuntime)
    context = SimpleNamespace(
        settings=SimpleNamespace(
            klein4b_model_id="test/4b",
            klein4b_profile="bf16_model_offload",
            klein4b_device="cuda",
            klein_model_id="test/9b",
            klein_profile="consumer_nvfp4",
            klein_device="cuda",
            klein_transformer_model_id="test/transformer",
            klein_transformer_filename="transformer.safetensors",
            klein_text_encoder_model_id="test/text-encoder",
        )
    )

    text4_runtime = klein_tools.Klein4BTextToImageTool()._runtime(context)
    edit4_runtime = klein_tools.Klein4BImageToImageTool()._runtime(context)
    assert text4_runtime is edit4_runtime
    assert created == [text4_runtime]

    text9_runtime = klein_tools.KleinTextToImageTool()._runtime(context)
    edit9_runtime = klein_tools.KleinImageToImageTool()._runtime(context)
    assert text9_runtime is edit9_runtime
    assert text4_runtime.unloaded is True
    assert created == [text4_runtime, text9_runtime]
    RUNTIME_MANAGER.clear()


def test_klein_runtime_passes_one_to_three_references_to_diffusers(tmp_path, monkeypatch):
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="consumer_int8",
        h3_device="cuda",
    )
    runtime = KleinRuntime(settings, "klein4b")
    calls = []

    class FakeGenerator:
        def __init__(self, device):
            self.device = device
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class FakeImage:
        width = 512
        height = 512

        def save(self, path, format):
            assert format == "PNG"
            path.write_bytes(b"png")

    class FakePipeline:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[FakeImage()])

    fake_torch = ModuleType("torch")
    fake_torch.Generator = FakeGenerator
    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.__path__ = []
    fake_utils = ModuleType("diffusers.utils")
    fake_utils.load_image = lambda path: f"loaded:{path}"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", fake_utils)
    monkeypatch.setattr(runtime, "_load_pipeline", lambda: FakePipeline())

    paths = [tmp_path / f"ref-{index}.png" for index in range(3)]
    for path in paths:
        path.write_bytes(b"ref")
    output = tmp_path / "output.png"

    metadata = runtime.generate(
        prompt="combine the references",
        output_path=output,
        size_name="512x512",
        seed=42,
        image_paths=paths,
        progress=lambda *_: None,
        check_cancelled=lambda: None,
    )

    assert output.read_bytes() == b"png"
    assert calls[0]["image"] == [f"loaded:{path}" for path in paths]
    assert calls[0]["width"] == 512
    assert calls[0]["height"] == 512
    assert metadata["reference_count"] == 3
    assert metadata["model_variant"] == "klein4b"
