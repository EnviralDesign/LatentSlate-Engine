"""Small callback seam for truthful native execution progress."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ProgressCallback = Callable[[dict[str, Any]], None]


def report_progress(
    callback: ProgressCallback | None,
    overall: float,
    label: str,
    *,
    stage_progress: float | None = None,
    detail: str | None = None,
) -> None:
    if callback is None:
        return
    stage: dict[str, Any] = {"label": label}
    if stage_progress is not None:
        stage["progress"] = stage_progress
    if detail is not None:
        stage["detail"] = detail
    callback({"progress": overall, "stage": stage})
