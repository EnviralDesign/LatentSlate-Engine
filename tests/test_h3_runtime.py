from types import SimpleNamespace

from latentslate_engine.runtime.h3 import (
    H3_MAX_DURATION_SECONDS,
    H3_MAX_FRAMES,
    H3_MIN_FRAMES,
    frames_for_duration,
)
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.tools import h3 as h3_tools


def test_h3_duration_alignment_stays_inside_model_limits():
    assert frames_for_duration(5.0) == H3_MIN_FRAMES
    assert frames_for_duration(H3_MAX_DURATION_SECONDS) == H3_MAX_FRAMES
    assert frames_for_duration(15.0) == H3_MAX_FRAMES


def test_h3_frame_counts_follow_vae_contract():
    for duration in (5.0, 7.0, 10.0, 14.0, 15.0):
        frames = frames_for_duration(duration)
        assert frames % 17 == 5
        assert H3_MIN_FRAMES <= frames <= H3_MAX_FRAMES


def test_h3_tools_share_one_runtime_for_the_same_settings(monkeypatch):
    created = []

    class FakeRuntime:
        def __init__(self, settings):
            self.settings = settings
            self.unloaded = False
            created.append(self)

        def unload(self):
            self.unloaded = True

    RUNTIME_MANAGER.clear()
    monkeypatch.setattr(h3_tools, "H3Runtime", FakeRuntime)
    context = SimpleNamespace(
        settings=SimpleNamespace(
            h3_model_id="test/model",
            h3_profile="bf16_auto_offload",
            h3_device="cuda",
        )
    )

    text_runtime = h3_tools.H3TextToVideoTool()._runtime(context)
    keyframe_runtime = h3_tools.H3FirstLastFrameTool()._runtime(context)

    assert text_runtime is keyframe_runtime
    assert created == [text_runtime]
    RUNTIME_MANAGER.clear()
