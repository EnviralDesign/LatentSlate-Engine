from types import SimpleNamespace

from latentslate_engine.bundles import BUNDLES
from latentslate_engine.config import Settings
from latentslate_engine.protocol import WorkflowKind
from latentslate_engine.runtime.ltx23 import (
    LTX23_GUIDANCE_SCALE,
    LTX23_MAX_FRAMES,
    LTX23_MIN_FRAMES,
    LTX23_SIZE_PRESETS,
    LTX23_STEPS,
    frames_for_duration,
)
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.tools import ltx23 as ltx23_tools


def test_ltx23_frame_counts_follow_temporal_contract():
    for duration in (1.0, 2.0, 5.0, 10.0, 20.0):
        frames = frames_for_duration(duration)
        assert frames % 8 == 1
        assert LTX23_MIN_FRAMES <= frames <= LTX23_MAX_FRAMES


def test_ltx23_tool_follows_latentslate_taxonomy():
    descriptor = ltx23_tools.LTX23TextToVideoTool().descriptor

    assert descriptor.name == "Text to Video"
    assert descriptor.key == "ltx23.text_to_video"
    assert descriptor.workflow_kind == WorkflowKind.TEXT_TO_VIDEO
    assert descriptor.inputs[1].default == "768x512"
    assert descriptor.inputs[2].default == 5.0
    assert {option.value for option in descriptor.inputs[1].options} == set(
        LTX23_SIZE_PRESETS
    )


def test_ltx23_bundle_and_defaults_use_converted_distilled_checkpoint(tmp_path):
    model_id = "diffusers/LTX-2.3-Distilled-Diffusers"
    assert BUNDLES["ltx23-basic"].repo_id == model_id
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    assert settings.ltx23_model_id == model_id
    assert settings.ltx23_profile == "bf16_sequential_offload"
    assert LTX23_STEPS == 8
    assert LTX23_GUIDANCE_SCALE == 1.0


def test_ltx23_runtime_is_reused(monkeypatch):
    created = []

    class FakeRuntime:
        def __init__(self, settings):
            self.settings = settings
            self.unloaded = False
            created.append(self)

        def unload(self):
            self.unloaded = True

    RUNTIME_MANAGER.clear()
    monkeypatch.setattr(ltx23_tools, "LTX23Runtime", FakeRuntime)
    context = SimpleNamespace(
        settings=SimpleNamespace(
            ltx23_model_id="test/model",
            ltx23_profile="bf16_sequential_offload",
            ltx23_device="cuda",
        )
    )
    tool = ltx23_tools.LTX23TextToVideoTool()

    first = tool._runtime(context)
    second = tool._runtime(context)

    assert first is second
    assert created == [first]
    RUNTIME_MANAGER.clear()
