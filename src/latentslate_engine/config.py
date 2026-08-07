from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_home() -> Path:
    configured = os.getenv("LATENTSLATE_ENGINE_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        root = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "LatentSlateEngine"
    return Path.home() / ".local" / "share" / "latentslate-engine"


@dataclass(frozen=True, slots=True)
class Settings:
    home: Path
    token: str | None
    max_upload_bytes: int
    h3_model_id: str
    h3_profile: str
    h3_device: str

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.getenv("LATENTSLATE_ENGINE_TOKEN")
        token = token.strip() if token and token.strip() else None
        return cls(
            home=_default_home(),
            token=token,
            max_upload_bytes=int(os.getenv("LATENTSLATE_ENGINE_MAX_UPLOAD_BYTES", str(2**34))),
            h3_model_id=os.getenv("LATENTSLATE_H3_MODEL", "MiniMaxAI/MiniMax-H3"),
            h3_profile=os.getenv("LATENTSLATE_H3_PROFILE", "consumer_int8"),
            h3_device=os.getenv("LATENTSLATE_H3_DEVICE", "cuda"),
        )

    def ensure_directories(self) -> None:
        for directory in (self.home, self.assets_dir, self.jobs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def assets_dir(self) -> Path:
        return self.home / "assets"

    @property
    def jobs_dir(self) -> Path:
        return self.home / "jobs"
