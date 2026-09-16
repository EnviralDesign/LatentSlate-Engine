"""Native-free artifact identity for LTX 2.5."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Ltx25ModelPaths:
    diffusion: Path
    text_encoder: Path
    video_vae: Path
    audio_vae: Path
    upsampler: Path
    prompt_enhancer: Path

    def available(self, operation):
        return all(
            path.is_file()
            for key, path in self.__dict__.items()
            if key != "upsampler" or operation != "flf"
        )

    @classmethod
    def from_root(cls, root: Path):
        """Canonical bootstrap destinations for the built-in recipes."""
        return cls(
            root
            / "diffusion_models/ltx25/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
            root
            / "text_encoders/ltx25/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
            root / "vae/ltx25/ltx-2.5-video-vae-bf16.safetensors",
            root / "vae/ltx25/ltx-2.5-audio-vae-bf16.safetensors",
            root
            / "latent_upscale_models/ltx25/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            root / "text_encoders/ltx25/gemma4_e2b_it_int8_convrot.safetensors",
        )


@dataclass(frozen=True)
class Ltx25Identity:
    diffusion_path: str
    text_encoder_path: str
    video_vae_path: str
    audio_vae_path: str
    upsampler_path: str | None = None
    device_index: int = 0
    prompt_enhancer_path: str | None = None
    artifact_versions: tuple[tuple[str, int, int], ...] = ()
    transformer_adapters: tuple[tuple[str, float], ...] = ()

    @classmethod
    def from_paths(cls, *, transformer_adapters=(), **paths):
        """Snapshot fixed files so replacement invalidates worker model state."""
        versions = []
        resolved = {}
        for key, value in paths.items():
            if value is None:
                resolved[key] = None
                continue
            path = Path(value).resolve(strict=True)
            if not path.is_file():
                raise ValueError("LTX model artifacts must be files")
            stat = path.stat()
            resolved[key] = str(path)
            versions.append((str(path), stat.st_size, stat.st_mtime_ns))
        adapters = []
        for value, strength in transformer_adapters:
            path = Path(value).resolve(strict=True)
            if not path.is_file():
                raise ValueError("LTX adapters must be files")
            stat = path.stat()
            versions.append((str(path), stat.st_size, stat.st_mtime_ns))
            adapters.append((str(path), float(strength)))
        return cls(
            **resolved,
            artifact_versions=tuple(versions),
            transformer_adapters=tuple(adapters),
        )
