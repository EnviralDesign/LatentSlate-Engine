from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .model_store import configured_engine_home, initialize_engine_data


def _default_home() -> Path:
    return configured_engine_home()


def _env_paths(name: str) -> tuple[Path, ...]:
    raw = os.getenv(name, "")
    paths: list[Path] = []
    seen: set[Path] = set()
    for value in raw.split(os.pathsep):
        if not value.strip():
            continue
        path = Path(value.strip()).expanduser().resolve()
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return tuple(paths)


def _catalog_roots(entries: list[tuple[str, Path]]) -> tuple[tuple[str, Path], ...]:
    roots: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, path in entries:
        resolved = path.resolve()
        if resolved in seen:
            continue
        roots.append((label, resolved))
        seen.add(resolved)
    return tuple(roots)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false, yes/no, on/off, or 1/0")


@dataclass(frozen=True, slots=True)
class Settings:
    home: Path
    token: str | None
    max_upload_bytes: int
    h3_model_id: str
    h3_profile: str
    h3_device: str
    ltx23_model_id: str = "diffusers/LTX-2.3-Distilled-Diffusers"
    ltx23_profile: str = "bf16_sequential_offload"
    ltx23_device: str = "cuda"
    wan22_model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    wan22_profile: str = "bf16_sequential_offload"
    wan22_device: str = "cuda"
    comfyui_root: Path = Path("C:/ComfyUI")
    klein4b_model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    klein4b_profile: str = "bf16_model_offload"
    klein4b_device: str = "cuda"
    klein_model_id: str = "black-forest-labs/FLUX.2-klein-9B"
    klein_profile: str = "bf16_model_offload"
    klein_device: str = "cuda"
    cache_enabled: bool = True
    cache_max_bytes: int = 2 * 1024**3
    cache_max_entries: int = 16
    recipe_paths: tuple[Path, ...] = ()
    deployment_profile_paths: tuple[Path, ...] = ()

    @classmethod
    def from_env(cls) -> Settings:
        token = os.getenv("LATENTSLATE_ENGINE_TOKEN")
        token = token.strip() if token and token.strip() else None
        return cls(
            home=_default_home(),
            token=token,
            max_upload_bytes=int(os.getenv("LATENTSLATE_ENGINE_MAX_UPLOAD_BYTES", str(2**34))),
            h3_model_id=os.getenv("LATENTSLATE_H3_MODEL", "MiniMaxAI/MiniMax-H3"),
            h3_profile=os.getenv("LATENTSLATE_H3_PROFILE", "bf16_auto_offload"),
            h3_device=os.getenv("LATENTSLATE_H3_DEVICE", "cuda"),
            ltx23_model_id=os.getenv(
                "LATENTSLATE_LTX23_MODEL",
                "diffusers/LTX-2.3-Distilled-Diffusers",
            ),
            ltx23_profile=os.getenv(
                "LATENTSLATE_LTX23_PROFILE",
                "bf16_sequential_offload",
            ),
            ltx23_device=os.getenv("LATENTSLATE_LTX23_DEVICE", "cuda"),
            wan22_model_id=os.getenv(
                "LATENTSLATE_WAN22_MODEL",
                "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            ),
            wan22_profile=os.getenv(
                "LATENTSLATE_WAN22_PROFILE",
                "bf16_sequential_offload",
            ),
            wan22_device=os.getenv("LATENTSLATE_WAN22_DEVICE", "cuda"),
            comfyui_root=Path(
                os.getenv("LATENTSLATE_COMFYUI_ROOT", "C:/ComfyUI")
            ).expanduser().resolve(),
            klein4b_model_id=os.getenv(
                "LATENTSLATE_KLEIN4B_MODEL",
                "black-forest-labs/FLUX.2-klein-4B",
            ),
            klein4b_profile=os.getenv(
                "LATENTSLATE_KLEIN4B_PROFILE",
                "bf16_model_offload",
            ),
            klein4b_device=os.getenv("LATENTSLATE_KLEIN4B_DEVICE", "cuda"),
            klein_model_id=os.getenv(
                "LATENTSLATE_KLEIN_MODEL",
                "black-forest-labs/FLUX.2-klein-9B",
            ),
            klein_profile=os.getenv(
                "LATENTSLATE_KLEIN_PROFILE",
                "bf16_model_offload",
            ),
            klein_device=os.getenv("LATENTSLATE_KLEIN_DEVICE", "cuda"),
            cache_enabled=_env_bool("LATENTSLATE_CACHE_ENABLED", True),
            cache_max_bytes=int(os.getenv("LATENTSLATE_CACHE_MAX_BYTES", str(2 * 1024**3))),
            cache_max_entries=int(os.getenv("LATENTSLATE_CACHE_MAX_ENTRIES", "16")),
            recipe_paths=_env_paths("LATENTSLATE_RECIPE_PATHS"),
            deployment_profile_paths=_env_paths(
                "LATENTSLATE_DEPLOYMENT_PROFILE_PATHS"
            ),
        )

    def ensure_directories(self) -> None:
        initialize_engine_data(self.home)

    @property
    def model_root(self) -> Path:
        return self.home / "models"

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    @property
    def lora_root(self) -> Path:
        return self.home / "loras"

    @property
    def variants_root(self) -> Path:
        """Legacy data-defined tool directory retained for compatibility."""

        return self.home / "variants"

    @property
    def recipes_root(self) -> Path:
        return self.home / "recipes"

    @property
    def deployment_profiles_root(self) -> Path:
        return self.home / "profiles"

    @property
    def resource_declarations_root(self) -> Path:
        return self.home / "resource_declarations"

    @property
    def builtin_resource_declarations_root(self) -> Path:
        return Path(__file__).resolve().parent / "builtin_resource_declarations"

    def resource_declaration_roots(self) -> tuple[tuple[str, Path], ...]:
        """Return package declarations before user-owned declarations."""

        return _catalog_roots(
            [
                ("builtin", self.builtin_resource_declarations_root),
                ("local", self.resource_declarations_root),
            ]
        )

    @property
    def builtin_recipes_root(self) -> Path:
        return Path(__file__).resolve().parent / "builtin_recipes"

    @property
    def builtin_deployment_profiles_root(self) -> Path:
        return Path(__file__).resolve().parent / "builtin_profiles"

    def recipe_catalog_roots(self) -> tuple[tuple[str, Path], ...]:
        return _catalog_roots(
            [
                ("builtin", self.builtin_recipes_root),
                ("local", self.recipes_root),
                *[
                    (f"private-{index}", path)
                    for index, path in enumerate(self.recipe_paths, start=1)
                ],
                ("variants", self.variants_root),
            ]
        )

    def deployment_profile_roots(self) -> tuple[tuple[str, Path], ...]:
        return _catalog_roots(
            [
                ("builtin", self.builtin_deployment_profiles_root),
                ("local", self.deployment_profiles_root),
                *[
                    (f"private-{index}", path)
                    for index, path in enumerate(
                        self.deployment_profile_paths,
                        start=1,
                    )
                ],
            ]
        )

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def temp_dir(self) -> Path:
        return self.home / "temp"

    @property
    def assets_dir(self) -> Path:
        return self.home / "assets"

    @property
    def jobs_dir(self) -> Path:
        return self.home / "jobs"
