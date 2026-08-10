"""Bounded, atomic serialization for Engine-owned RGB video tensors."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from uuid import uuid4

import torch

_SUPPORTED_VIDEO_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32}
)
_MAX_VIDEO_FRAMES = 121
_MAX_VIDEO_PIXELS_PER_FRAME = 1280 * 704
_MAX_VIDEO_FPS = 240


def encode_rgb_video_tensor(
    video: torch.Tensor,
    *,
    fps: int,
    output_path: Path,
    check_cancelled: Callable[[], None] | None = None,
) -> None:
    """Encode one normalized CPU RGB video to MP4 without a full-video uint8 copy.

    ``video`` must use the native runtime boundary ``[1, 3, T, H, W]`` with
    finite floating values in ``[-1, 1]``. Frames are converted lazily to
    ``uint8`` and handed to Diffusers' PyAV encoder one at a time. The final
    path is replaced atomically only after encoding succeeds.
    """

    frame_count, _, _ = _validate_video(video)
    _validate_fps(fps)
    target = Path(output_path)
    if target.suffix.lower() != ".mp4":
        raise ValueError("video output path must use the .mp4 extension")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp.mp4")
    try:
        from diffusers.utils.export_utils import encode_video

        encode_video(
            _uint8_frame_chunks(video, check_cancelled=check_cancelled),
            fps=fps,
            output_path=str(temporary),
            video_chunks_number=frame_count,
        )
        if check_cancelled is not None:
            check_cancelled()
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("video encoder completed without a nonempty MP4")
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _uint8_frame_chunks(
    video: torch.Tensor,
    *,
    check_cancelled: Callable[[], None] | None,
) -> Iterator[torch.Tensor]:
    frames = video[0].permute(1, 2, 3, 0)
    for frame in frames:
        if check_cancelled is not None:
            check_cancelled()
        # Work one frame at a time so a 121-frame 1280x704 result does not
        # acquire a second, full-video uint8 allocation during serialization.
        yield (
            frame.to(dtype=torch.float32)
            .add(1.0)
            .mul(127.5)
            .round()
            .clamp_(0, 255)
            .to(dtype=torch.uint8)
            .unsqueeze(0)
            .contiguous()
        )


def _validate_video(video: torch.Tensor) -> tuple[int, int, int]:
    if not isinstance(video, torch.Tensor):
        raise TypeError("video output requires a torch.Tensor")
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise ValueError("video output requires shape [1, 3, frames, height, width]")
    frame_count, height, width = map(int, video.shape[2:])
    if not 1 <= frame_count <= _MAX_VIDEO_FRAMES:
        raise ValueError(f"video output supports 1..{_MAX_VIDEO_FRAMES} frames")
    if height <= 0 or width <= 0 or height * width > _MAX_VIDEO_PIXELS_PER_FRAME:
        raise ValueError("video output exceeds the supported per-frame pixel budget")
    if height % 2 or width % 2:
        raise ValueError("MP4 output dimensions must both be even")
    if video.device.type != "cpu":
        raise ValueError("video output tensor must be CPU-resident")
    if video.dtype not in _SUPPORTED_VIDEO_DTYPES:
        raise TypeError("video output tensor must use float16, bfloat16, or float32")
    if not bool(torch.isfinite(video).all()):
        raise ValueError("video output tensor must contain only finite values")
    if bool((video < -1.0).any()) or bool((video > 1.0).any()):
        raise ValueError("video output tensor values must be within [-1, 1]")
    return frame_count, height, width


def _validate_fps(fps: int) -> None:
    if isinstance(fps, bool) or not isinstance(fps, int):
        raise TypeError("video output FPS must be an integer")
    if not 1 <= fps <= _MAX_VIDEO_FPS or not math.isfinite(float(fps)):
        raise ValueError(f"video output FPS must be between 1 and {_MAX_VIDEO_FPS}")
