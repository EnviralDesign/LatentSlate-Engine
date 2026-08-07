from latentslate_engine.runtime.h3 import (
    H3_MAX_DURATION_SECONDS,
    H3_MAX_FRAMES,
    H3_MIN_FRAMES,
    frames_for_duration,
)


def test_h3_duration_alignment_stays_inside_model_limits():
    assert frames_for_duration(5.0) == H3_MIN_FRAMES
    assert frames_for_duration(H3_MAX_DURATION_SECONDS) == H3_MAX_FRAMES
    assert frames_for_duration(15.0) == H3_MAX_FRAMES


def test_h3_frame_counts_follow_vae_contract():
    for duration in (5.0, 7.0, 10.0, 14.0, 15.0):
        frames = frames_for_duration(duration)
        assert frames % 17 == 5
        assert H3_MIN_FRAMES <= frames <= H3_MAX_FRAMES
