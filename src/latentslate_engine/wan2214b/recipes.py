"""Family-owned capability and recipe adapter for proven Wan 2.2 T2V."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from latentslate_engine.recipe import (
    Adapter,
    Artifact,
    Capability,
    CapabilitySet,
    Recipe,
    exposed,
    fixed,
)

from .contracts import (
    MAX_PIXELS,
    MAX_SEED,
    MIN_SIDE,
    WanRecipe,
    validate_request,
)
from .timing import (
    DURATION_STEP_SECONDS,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    native_frame_count,
)

_HIGH_CHECKPOINT = Capability("high_checkpoint", "artifact")
_HIGH_ADAPTERS = Capability("high_adapters", "adapter", ordered=True)
_LOW_CHECKPOINT = Capability("low_checkpoint", "artifact")
_LOW_ADAPTERS = Capability("low_adapters", "adapter", ordered=True)
_TEXT_ENCODER = Capability("text_encoder", "artifact")
_VAE = Capability("vae", "artifact")
_NEGATIVE_PROMPT = Capability("negative_prompt", "text")
_SHIFT = Capability("shift", "number")
_STEPS = Capability("steps", "integer")
_SPLIT_STEP = Capability("split_step", "integer")
_CFG = Capability("cfg", "number")
_PROMPT = Capability("prompt", "text")
_WIDTH = Capability(
    "width",
    "integer",
    role="width",
    minimum=MIN_SIDE,
    maximum=MAX_PIXELS // MIN_SIDE,
    step=16,
)
_HEIGHT = Capability(
    "height",
    "integer",
    role="height",
    minimum=MIN_SIDE,
    maximum=MAX_PIXELS // MIN_SIDE,
    step=16,
)
_DURATION = Capability(
    "duration_seconds",
    "number",
    role="duration_seconds",
    minimum=MIN_DURATION_SECONDS,
    maximum=MAX_DURATION_SECONDS,
    step=DURATION_STEP_SECONDS,
)
_SEED = Capability("seed", "integer", role="seed", minimum=0, maximum=MAX_SEED)


def _validate_t2v_capabilities(values: Mapping[str, object]) -> None:
    for key in ("high_adapters", "low_adapters"):
        adapters = values[key]
        if not isinstance(adapters, tuple) or len(adapters) > 2:
            raise ValueError(f"{key} supports at most primary and secondary adapters")
    frames = native_frame_count(values["duration_seconds"])  # type: ignore[arg-type]
    validate_request(
        values["width"],  # type: ignore[arg-type]
        values["height"],  # type: ignore[arg-type]
        frames,
        values["seed"],  # type: ignore[arg-type]
    )


WAN2214B_T2V_CAPABILITIES = CapabilitySet(
    "wan2214b.t2v",
    (
        _HIGH_CHECKPOINT,
        _HIGH_ADAPTERS,
        _LOW_CHECKPOINT,
        _LOW_ADAPTERS,
        _TEXT_ENCODER,
        _VAE,
        _NEGATIVE_PROMPT,
        _SHIFT,
        _STEPS,
        _SPLIT_STEP,
        _CFG,
        _PROMPT,
        _WIDTH,
        _HEIGHT,
        _DURATION,
        _SEED,
    ),
    _validate_t2v_capabilities,
)


def wan2214b_t2v_recipe(
    *,
    high_checkpoint: str | Path,
    high_adapters: Sequence[Adapter],
    low_checkpoint: str | Path,
    low_adapters: Sequence[Adapter],
    text_encoder: str | Path,
    vae: str | Path,
    negative_prompt: str,
) -> Recipe:
    """Define one Wan product while retaining separate high/low ownership."""
    return Recipe(
        "wan2214b.t2v.v1",
        WAN2214B_T2V_CAPABILITIES,
        (
            fixed(_HIGH_CHECKPOINT, Artifact(high_checkpoint)),
            fixed(_HIGH_ADAPTERS, tuple(high_adapters)),
            fixed(_LOW_CHECKPOINT, Artifact(low_checkpoint)),
            fixed(_LOW_ADAPTERS, tuple(low_adapters)),
            fixed(_TEXT_ENCODER, Artifact(text_encoder)),
            fixed(_VAE, Artifact(vae)),
            fixed(_NEGATIVE_PROMPT, negative_prompt),
            fixed(_SHIFT, 5.000000000000001),
            fixed(_STEPS, 4),
            fixed(_SPLIT_STEP, 2),
            fixed(_CFG, 1.0),
            exposed(_PROMPT),
            exposed(_WIDTH, default=512),
            exposed(_HEIGHT, default=512),
            exposed(_DURATION, default=5.0),
            exposed(_SEED, default=0),
        ),
    )


def resolve_wan2214b_t2v(
    definition: Recipe, overrides: Mapping[str, object]
) -> tuple[WanRecipe, dict[str, object]]:
    """Resolve policy into the existing Wan recipe and generate arguments."""
    if definition.capabilities is not WAN2214B_T2V_CAPABILITIES:
        raise TypeError("recipe does not use the Wan 2.2 T2V capability set")
    values = definition.resolve(overrides)
    high = values["high_adapters"]
    low = values["low_adapters"]
    assert isinstance(high, tuple) and isinstance(low, tuple)
    high_primary, high_secondary = _adapter_slots(high)
    low_primary, low_secondary = _adapter_slots(low)
    frames = native_frame_count(values["duration_seconds"])  # type: ignore[arg-type]
    family_recipe = WanRecipe(
        high_checkpoint=str(values["high_checkpoint"].path),  # type: ignore[union-attr]
        high_lora=_adapter_path(high_primary),
        low_checkpoint=str(values["low_checkpoint"].path),  # type: ignore[union-attr]
        low_lora=_adapter_path(low_primary),
        text_encoder=str(values["text_encoder"].path),  # type: ignore[union-attr]
        vae=str(values["vae"].path),  # type: ignore[union-attr]
        high_secondary_lora=_adapter_path(high_secondary),
        low_secondary_lora=_adapter_path(low_secondary),
        high_lora_strength=_adapter_strength(high_primary),
        low_lora_strength=_adapter_strength(low_primary),
        high_secondary_lora_strength=_adapter_strength(high_secondary),
        low_secondary_lora_strength=_adapter_strength(low_secondary),
        shift=values["shift"],  # type: ignore[arg-type]
        steps=values["steps"],  # type: ignore[arg-type]
        split_step=values["split_step"],  # type: ignore[arg-type]
        cfg=values["cfg"],  # type: ignore[arg-type]
        width=values["width"],  # type: ignore[arg-type]
        height=values["height"],  # type: ignore[arg-type]
        frame_count=frames,
        positive=values["prompt"],  # type: ignore[arg-type]
        negative=values["negative_prompt"],  # type: ignore[arg-type]
    )
    family_recipe.validate()
    request = {
        "seed": values["seed"],
        "width": values["width"],
        "height": values["height"],
        "frame_count": frames,
        "positive_prompt": values["prompt"],
        "negative_prompt": values["negative_prompt"],
    }
    return family_recipe, request


def _adapter_slots(
    adapters: tuple[object, ...],
) -> tuple[Adapter | None, Adapter | None]:
    primary = adapters[0] if adapters else None
    secondary = adapters[1] if len(adapters) > 1 else None
    assert primary is None or isinstance(primary, Adapter)
    assert secondary is None or isinstance(secondary, Adapter)
    return primary, secondary


def _adapter_path(adapter: Adapter | None) -> str | None:
    return None if adapter is None else str(adapter.artifact.path)


def _adapter_strength(adapter: Adapter | None) -> float:
    return 1.0 if adapter is None else float(adapter.strength)
