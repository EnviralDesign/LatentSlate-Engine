from __future__ import annotations

from pathlib import Path

import pytest
import torch

from latentslate_engine.runtime.video_output import encode_rgb_video_tensor


def test_video_output_converts_frames_lazily_and_replaces_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    observed: list[torch.Tensor] = []
    encoded_path: Path | None = None

    def fake_encode_video(video, *, fps, output_path, video_chunks_number):
        nonlocal encoded_path
        assert fps == 24
        assert video_chunks_number == 3
        encoded_path = Path(output_path)
        assert encoded_path.parent == tmp_path
        assert encoded_path.name.startswith(".output.mp4.")
        assert encoded_path.name.endswith(".tmp.mp4")
        observed.extend(chunk.clone() for chunk in video)
        encoded_path.write_bytes(b"mp4")

    monkeypatch.setattr(
        "diffusers.utils.export_utils.encode_video",
        fake_encode_video,
    )
    target = tmp_path / "output.mp4"
    video = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.bfloat16).view(1, 1, 3, 1, 1)
    video = video.expand(1, 3, 3, 2, 2).contiguous()

    encode_rgb_video_tensor(video, fps=24, output_path=target)

    assert target.read_bytes() == b"mp4"
    assert encoded_path is not None and not encoded_path.exists()
    assert len(observed) == 3
    assert [int(chunk[0, 0, 0, 0]) for chunk in observed] == [0, 128, 255]
    assert all(chunk.shape == (1, 2, 2, 3) for chunk in observed)
    assert all(chunk.dtype == torch.uint8 for chunk in observed)


def test_video_output_failure_removes_temporary_and_preserves_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "output.mp4"
    target.write_bytes(b"existing")

    def fail(video, *, fps, output_path, video_chunks_number):
        Path(output_path).write_bytes(b"partial")
        raise RuntimeError("encoder failed")

    monkeypatch.setattr("diffusers.utils.export_utils.encode_video", fail)
    with pytest.raises(RuntimeError, match="encoder failed"):
        encode_rgb_video_tensor(
            torch.zeros((1, 3, 1, 2, 2), dtype=torch.float16),
            fps=24,
            output_path=target,
        )

    assert target.read_bytes() == b"existing"
    assert list(tmp_path.glob(".*.tmp.mp4")) == []


def test_video_output_cancellation_cleans_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checks = 0

    def check_cancelled():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise KeyboardInterrupt

    def consume(video, *, fps, output_path, video_chunks_number):
        Path(output_path).write_bytes(b"partial")
        list(video)

    monkeypatch.setattr("diffusers.utils.export_utils.encode_video", consume)
    with pytest.raises(KeyboardInterrupt):
        encode_rgb_video_tensor(
            torch.zeros((1, 3, 3, 2, 2), dtype=torch.float32),
            fps=24,
            output_path=tmp_path / "output.mp4",
            check_cancelled=check_cancelled,
        )

    assert not (tmp_path / "output.mp4").exists()
    assert list(tmp_path.glob(".*.tmp.mp4")) == []


@pytest.mark.parametrize(
    "video,error",
    (
        (torch.zeros((3, 1, 2, 2)), "shape"),
        (torch.zeros((1, 3, 1, 3, 2)), "even"),
        (torch.full((1, 3, 1, 2, 2), float("nan")), "finite"),
        (torch.full((1, 3, 1, 2, 2), 1.1), "within"),
        (torch.zeros((1, 3, 1, 2, 2), dtype=torch.int8), "float16"),
    ),
)
def test_video_output_rejects_invalid_tensor(video: torch.Tensor, error: str, tmp_path: Path):
    with pytest.raises((TypeError, ValueError), match=error):
        encode_rgb_video_tensor(video, fps=24, output_path=tmp_path / "output.mp4")


@pytest.mark.parametrize("fps", (0, 241, True, 24.0))
def test_video_output_rejects_invalid_fps(fps, tmp_path: Path):
    with pytest.raises((TypeError, ValueError), match="FPS"):
        encode_rgb_video_tensor(
            torch.zeros((1, 3, 1, 2, 2)),
            fps=fps,
            output_path=tmp_path / "output.mp4",
        )
