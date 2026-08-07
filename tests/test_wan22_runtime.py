from types import SimpleNamespace

from latentslate_engine.bundles import BUNDLES
from latentslate_engine.config import Settings
from latentslate_engine.protocol import WorkflowKind
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.runtime.wan22 import (
    WAN22_MAX_FRAMES,
    WAN22_MIN_FRAMES,
    WAN22_SIZE_PRESETS,
    frames_for_duration,
)
from latentslate_engine.tools import wan22 as wan22_tools


def test_wan22_frame_counts_follow_temporal_contract():
    for duration in (1.0, 2.0, 5.0, 10.0):
        frames = frames_for_duration(duration)
        assert (frames - 1) % 4 == 0
        assert WAN22_MIN_FRAMES <= frames <= WAN22_MAX_FRAMES


def test_wan22_tool_follows_latentslate_taxonomy():
    descriptor = wan22_tools.Wan22TextToVideoTool().descriptor

    assert descriptor.name == "Text to Video"
    assert descriptor.key == "wan22.text_to_video"
    assert descriptor.workflow_kind == WorkflowKind.TEXT_TO_VIDEO
    assert descriptor.inputs[1].default == "1280x704"
    assert descriptor.inputs[2].default == 5.0
    assert {option.value for option in descriptor.inputs[1].options} == set(
        WAN22_SIZE_PRESETS
    )


def test_wan22_bundle_and_defaults_are_declared(tmp_path):
    assert (
        BUNDLES["wan22-basic"].repo_id
        == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    )
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="consumer_int8",
        h3_device="cuda",
    )
    assert settings.wan22_model_id == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert settings.wan22_profile == "bf16_sequential_offload"


def test_wan22_runtime_is_reused(monkeypatch):
    created = []

    class FakeRuntime:
        def __init__(self, settings):
            self.settings = settings
            self.unloaded = False
            created.append(self)

        def unload(self):
            self.unloaded = True

    RUNTIME_MANAGER.clear()
    monkeypatch.setattr(wan22_tools, "Wan22Runtime", FakeRuntime)
    context = SimpleNamespace(
        settings=SimpleNamespace(
            wan22_model_id="test/model",
            wan22_profile="bf16_sequential_offload",
            wan22_device="cuda",
        )
    )
    tool = wan22_tools.Wan22TextToVideoTool()

    first = tool._runtime(context)
    second = tool._runtime(context)

    assert first is second
    assert created == [first]
    RUNTIME_MANAGER.clear()
