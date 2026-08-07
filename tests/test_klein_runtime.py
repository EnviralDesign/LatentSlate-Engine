from types import SimpleNamespace

from latentslate_engine.bundles import BUNDLES
from latentslate_engine.config import Settings
from latentslate_engine.protocol import InputRole, WorkflowKind
from latentslate_engine.runtime.klein import KLEIN_SIZE_PRESETS
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.tools import klein as klein_tools


def test_klein_sizes_are_valid_flux2_canvases():
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
    text = klein_tools.KleinTextToImageTool().descriptor
    edit = klein_tools.KleinImageToImageTool().descriptor

    assert text.name == "Text to Image"
    assert text.workflow_kind == WorkflowKind.TEXT_TO_IMAGE
    assert edit.name == "Image to Image"
    assert edit.workflow_kind == WorkflowKind.IMAGE_TO_IMAGE
    assert edit.inputs[1].role == InputRole.SOURCE_IMAGE
    assert text.inputs[1].default == "1024x1024"
    assert edit.inputs[2].default == "source"
    assert "source" not in {option.value for option in text.inputs[1].options}
    assert "source" in {option.value for option in edit.inputs[2].options}


def test_klein_bundle_is_consumer_composed():
    bundle = BUNDLES["klein9b-basic"]
    assert bundle.repo_id == "black-forest-labs/FLUX.2-klein-9B"
    assert bundle.required_repo_ids() == {
        "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-9b-nvfp4",
        "Qwen/Qwen3-8B-FP8",
    }
    assert bundle.files[0].filename == "flux-2-klein-9b-nvfp4.safetensors"
    assert "transformer/config.json" in bundle.allow_patterns
    assert "transformer/**" not in bundle.allow_patterns
    assert "text_encoder/**" not in bundle.allow_patterns


def test_klein_defaults_target_consumer_blackwell_stack(tmp_path):
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="consumer_int8",
        h3_device="cuda",
    )
    assert settings.klein_profile == "consumer_nvfp4"
    assert settings.klein_transformer_model_id.endswith("klein-9b-nvfp4")
    assert settings.klein_text_encoder_model_id == "Qwen/Qwen3-8B-FP8"


def test_klein_tools_share_one_runtime_for_the_same_settings(monkeypatch):
    created = []

    class FakeRuntime:
        def __init__(self, settings):
            self.settings = settings
            self.unloaded = False
            created.append(self)

        def unload(self):
            self.unloaded = True

    RUNTIME_MANAGER.clear()
    monkeypatch.setattr(klein_tools, "KleinRuntime", FakeRuntime)
    context = SimpleNamespace(
        settings=SimpleNamespace(
            klein_model_id="test/model",
            klein_profile="consumer_nvfp4",
            klein_device="cuda",
            klein_transformer_model_id="test/transformer",
            klein_transformer_filename="transformer.safetensors",
            klein_text_encoder_model_id="test/text-encoder",
        )
    )

    text_runtime = klein_tools.KleinTextToImageTool()._runtime(context)
    edit_runtime = klein_tools.KleinImageToImageTool()._runtime(context)

    assert text_runtime is edit_runtime
    assert created == [text_runtime]
    RUNTIME_MANAGER.clear()
