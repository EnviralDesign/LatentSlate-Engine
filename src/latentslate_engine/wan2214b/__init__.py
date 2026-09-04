from typing import Any

__all__ = [
    "GenerationResult",
    "WanFLFRecipe",
    "WanFLFSession",
    "WanI2VRecipe",
    "WanI2VSession",
    "WanRecipe",
    "WanSession",
    "canonical_sigmas",
]


def __getattr__(name: str) -> Any:
    if name in {"WanFLFRecipe", "WanFLFSession"}:
        from .flf import WanFLFRecipe, WanFLFSession

        return {"WanFLFRecipe": WanFLFRecipe, "WanFLFSession": WanFLFSession}[name]
    if name in {"WanI2VRecipe", "WanI2VSession"}:
        from .i2v import WanI2VRecipe, WanI2VSession

        return {"WanI2VRecipe": WanI2VRecipe, "WanI2VSession": WanI2VSession}[name]
    if name in {"GenerationResult", "WanRecipe", "WanSession", "canonical_sigmas"}:
        from .pipeline import GenerationResult, WanRecipe, WanSession, canonical_sigmas

        return {
            "GenerationResult": GenerationResult,
            "WanRecipe": WanRecipe,
            "WanSession": WanSession,
            "canonical_sigmas": canonical_sigmas,
        }[name]
    raise AttributeError(name)
