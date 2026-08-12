from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from ..acquisition import deployment_install as installer
from .service_types import CatalogAuthoringError


def _publish_text_file(
    stage: Path,
    destination: Path,
    *,
    replace: bool,
    label: str,
) -> tuple[str, bytes | None]:
    if installer._exists(destination):
        if not replace:
            raise CatalogAuthoringError(
                f"catalog destination already exists for {label}: {destination}",
                code="catalog_conflict",
            )
        if installer._is_reparse(destination) or not destination.is_file():
            raise CatalogAuthoringError(f"catalog destination is unsafe: {destination}")
        previous = destination.read_bytes()
        os.replace(stage, destination)
        return "replaced", previous
    installer._publish_file_no_clobber(stage, destination, label)
    return "created", None


def _rollback_text_file(
    destination: Path,
    state: tuple[str, bytes | None],
    stage_root: Path,
) -> None:
    action, previous = state
    if action == "created":
        if installer._exists(destination):
            installer._safe_unlink(destination, destination.parent)
        return
    assert previous is not None
    restore = stage_root / f"restore-{uuid4().hex}.toml"
    restore.write_bytes(previous)
    os.replace(restore, destination)


def _owned_path(root: Path, path: Path, label: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    resolved_parent = path.parent.resolve()
    try:
        resolved_parent.relative_to(resolved_root)
    except ValueError as exc:
        raise CatalogAuthoringError(f"{label} path escapes Engine-owned storage") from exc
    return resolved_parent / path.name


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
