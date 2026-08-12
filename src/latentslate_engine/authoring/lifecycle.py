from __future__ import annotations

import hashlib
from pathlib import Path

from ..config import Settings
from .models import CatalogActivation, CatalogStatus
from .service_types import ActivationAction


def catalog_disk_revision(settings: Settings) -> str:
    digest = hashlib.sha256()
    roots: list[tuple[str, Path]] = [
        ("resources", settings.resource_declarations_root),
        ("recipes", settings.recipes_root),
        ("profiles", settings.deployment_profiles_root),
        ("variants", settings.variants_root),
    ]
    roots.extend(
        (f"private-recipe-{index}", path)
        for index, path in enumerate(settings.recipe_paths, start=1)
    )
    roots.extend(
        (f"private-profile-{index}", path)
        for index, path in enumerate(settings.deployment_profile_paths, start=1)
    )
    for label, root in roots:
        digest.update(label.encode("utf-8"))
        if not root.exists():
            digest.update(b"<missing>")
            continue
        for path in sorted(root.rglob("*.toml")):
            if path.name.startswith(".") or path.is_symlink() or not path.is_file():
                continue
            try:
                relative = path.resolve().relative_to(root.resolve())
            except ValueError:
                digest.update(f"<escape:{path}>".encode())
                continue
            digest.update(relative.as_posix().encode("utf-8"))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def catalog_status(settings: Settings, loaded_revision: str) -> CatalogStatus:
    disk = catalog_disk_revision(settings)
    stale = disk != loaded_revision
    return CatalogStatus(
        loaded_revision=loaded_revision,
        disk_revision=disk,
        stale=stale,
        required_action="restart_engine" if stale else "none",
    )


def _activation(settings: Settings, action: ActivationAction) -> CatalogActivation:
    return CatalogActivation(
        published=True,
        active_in_current_process=False,
        required_action=action,
        disk_revision=catalog_disk_revision(settings),
    )
