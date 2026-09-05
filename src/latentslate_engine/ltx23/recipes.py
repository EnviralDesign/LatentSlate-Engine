"""Family-owned capabilities and product recipes for proven LTX 2.3 operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from latentslate_engine.recipe import (
    Adapter,
    Artifact,
    Capability,
    CapabilitySet,
    Field,
    Recipe,
    exposed,
    fixed,
)

from .contracts import (
    MAX_DURATION_SECONDS,
    MAX_PIXELS,
    MAX_SEED,
    MIN_DURATION_SECONDS,
    MIN_SIDE,
    Ltx23FlfIdentity,
    Ltx23I2VIdentity,
    Ltx23T2VIdentity,
    validate_ltx_request,
)

_CHECKPOINT = Capability("checkpoint", "artifact")
_TEXT_CHECKPOINT = Capability("text_checkpoint", "artifact")
_UPSAMPLER = Capability("upsampler", "artifact")
_ADAPTER_ARTIFACTS = Capability(
    "transformer_adapter_artifacts", "artifact", ordered=True
)
_ADAPTER_STRENGTHS = Capability("transformer_adapter_strengths", "number", ordered=True)
_DEVICE_INDEX = Capability("device_index", "integer")
_PROMPT = Capability("prompt", "text")
_WIDTH = Capability(
    "width",
    "integer",
    role="width",
    minimum=MIN_SIDE,
    maximum=MAX_PIXELS // MIN_SIDE,
    step=64,
)
_HEIGHT = Capability(
    "height",
    "integer",
    role="height",
    minimum=MIN_SIDE,
    maximum=MAX_PIXELS // MIN_SIDE,
    step=64,
)
_FLF_WIDTH = Capability(
    "width",
    "integer",
    role="width",
    minimum=MIN_SIDE,
    maximum=MAX_PIXELS // MIN_SIDE,
    step=32,
)
_FLF_HEIGHT = Capability(
    "height",
    "integer",
    role="height",
    minimum=MIN_SIDE,
    maximum=MAX_PIXELS // MIN_SIDE,
    step=32,
)
_DURATION = Capability(
    "duration_seconds",
    "number",
    role="duration_seconds",
    minimum=MIN_DURATION_SECONDS,
    maximum=MAX_DURATION_SECONDS,
    step=0.5,
)
_SEED = Capability("seed", "integer", role="seed", minimum=0, maximum=MAX_SEED)
_START_IMAGE = Capability("start_image", "image", role="start_image")
_END_IMAGE = Capability("end_image", "image", role="end_image")


def _validate_two_pass_capabilities(values: Mapping[str, object]) -> None:
    artifacts = values["transformer_adapter_artifacts"]
    strengths = values["transformer_adapter_strengths"]
    assert isinstance(artifacts, tuple) and isinstance(strengths, tuple)
    if len(artifacts) != len(strengths):
        raise ValueError(
            "transformer adapter artifacts and strengths must have matching order and length"
        )
    validate_ltx_request(
        values["width"],  # type: ignore[arg-type]
        values["height"],  # type: ignore[arg-type]
        values["duration_seconds"],  # type: ignore[arg-type]
        values["seed"],  # type: ignore[arg-type]
        alignment=64,
    )


LTX23_T2V_CAPABILITIES = CapabilitySet(
    "ltx23.t2v",
    (
        _CHECKPOINT,
        _TEXT_CHECKPOINT,
        _UPSAMPLER,
        _ADAPTER_ARTIFACTS,
        _ADAPTER_STRENGTHS,
        _DEVICE_INDEX,
        _PROMPT,
        _WIDTH,
        _HEIGHT,
        _DURATION,
        _SEED,
    ),
    _validate_two_pass_capabilities,
)


LTX23_I2V_CAPABILITIES = CapabilitySet(
    "ltx23.i2v",
    (*LTX23_T2V_CAPABILITIES.capabilities, _START_IMAGE),
    _validate_two_pass_capabilities,
)


def _validate_flf_capabilities(values: Mapping[str, object]) -> None:
    validate_ltx_request(
        values["width"],  # type: ignore[arg-type]
        values["height"],  # type: ignore[arg-type]
        values["duration_seconds"],  # type: ignore[arg-type]
        values["seed"],  # type: ignore[arg-type]
        alignment=32,
    )


LTX23_FLF_CAPABILITIES = CapabilitySet(
    "ltx23.flf",
    (
        _CHECKPOINT,
        _TEXT_CHECKPOINT,
        _DEVICE_INDEX,
        _PROMPT,
        _START_IMAGE,
        _END_IMAGE,
        _FLF_WIDTH,
        _FLF_HEIGHT,
        _DURATION,
        _SEED,
    ),
    _validate_flf_capabilities,
)


def _adapter_values(
    adapters: Sequence[Adapter],
) -> tuple[tuple[Artifact, ...], tuple[float, ...]]:
    return (
        tuple(adapter.artifact for adapter in adapters),
        tuple(float(adapter.strength) for adapter in adapters),
    )


def _fixed_model_fields(
    *,
    checkpoint: str | Path,
    text_checkpoint: str | Path,
    upsampler: str | Path,
    transformer_adapters: Sequence[Adapter],
    device_index: int,
) -> tuple[Field, ...]:
    artifacts, strengths = _adapter_values(transformer_adapters)
    return (
        fixed(_CHECKPOINT, Artifact(checkpoint)),
        fixed(_TEXT_CHECKPOINT, Artifact(text_checkpoint)),
        fixed(_UPSAMPLER, Artifact(upsampler)),
        fixed(_ADAPTER_ARTIFACTS, artifacts),
        fixed(_ADAPTER_STRENGTHS, strengths),
        fixed(_DEVICE_INDEX, device_index),
    )


def ltx23_t2v_recipe(
    *,
    checkpoint: str | Path,
    text_checkpoint: str | Path,
    upsampler: str | Path,
    transformer_adapters: Sequence[Adapter] = (),
    device_index: int = 0,
) -> Recipe:
    """Preserve the V1 LTX product surface over the shared T2V capabilities."""
    model_fields = _fixed_model_fields(
        checkpoint=checkpoint,
        text_checkpoint=text_checkpoint,
        upsampler=upsampler,
        transformer_adapters=transformer_adapters,
        device_index=device_index,
    )
    return Recipe(
        "ltx23.t2v.v1",
        LTX23_T2V_CAPABILITIES,
        model_fields
        + (
            exposed(_PROMPT),
            exposed(_WIDTH, default=512),
            exposed(_HEIGHT, default=512),
            exposed(_DURATION, default=5.0),
            exposed(_SEED, default=0),
        ),
    )


def ltx23_t2v_locked_recipe(
    *,
    checkpoint: str | Path,
    text_checkpoint: str | Path,
    upsampler: str | Path,
    transformer_adapters: Sequence[Adapter],
    device_index: int = 0,
) -> Recipe:
    """Define a curated LTX product exposing only prompt and seed."""
    model_fields = _fixed_model_fields(
        checkpoint=checkpoint,
        text_checkpoint=text_checkpoint,
        upsampler=upsampler,
        transformer_adapters=transformer_adapters,
        device_index=device_index,
    )
    return Recipe(
        "ltx23.t2v.locked.v1_1",
        LTX23_T2V_CAPABILITIES,
        model_fields
        + (
            exposed(_PROMPT),
            fixed(_WIDTH, 768),
            fixed(_HEIGHT, 512),
            fixed(_DURATION, 5.0),
            exposed(_SEED, default=0),
        ),
    )


def ltx23_t2v_tunable_recipe(
    *,
    checkpoint: str | Path,
    text_checkpoint: str | Path,
    upsampler: str | Path,
    transformer_adapters: Sequence[Adapter],
    device_index: int = 0,
) -> Recipe:
    """Define a developer product with bounded geometry and ordered LoRA strengths."""
    if not transformer_adapters:
        raise ValueError("tunable LTX recipe requires at least one transformer adapter")
    artifacts, strengths = _adapter_values(transformer_adapters)
    return Recipe(
        "ltx23.t2v.tunable.v1_1",
        LTX23_T2V_CAPABILITIES,
        (
            fixed(_CHECKPOINT, Artifact(checkpoint)),
            fixed(_TEXT_CHECKPOINT, Artifact(text_checkpoint)),
            fixed(_UPSAMPLER, Artifact(upsampler)),
            fixed(_ADAPTER_ARTIFACTS, artifacts),
            exposed(_ADAPTER_STRENGTHS, default=strengths, minimum=0.0, maximum=1.0),
            fixed(_DEVICE_INDEX, device_index),
            exposed(_PROMPT),
            exposed(_WIDTH, default=512, minimum=256, maximum=1024),
            exposed(_HEIGHT, default=512, minimum=256, maximum=1024),
            exposed(_DURATION, default=5.0, minimum=2.0, maximum=5.0),
            exposed(_SEED, default=0),
        ),
    )


def ltx23_i2v_recipe(
    *,
    checkpoint: str | Path,
    text_checkpoint: str | Path,
    upsampler: str | Path,
    transformer_adapters: Sequence[Adapter] = (),
    device_index: int = 0,
) -> Recipe:
    """Define one I2V product with fixed model and adapter state."""
    return Recipe(
        "ltx23.i2v.v1_1",
        LTX23_I2V_CAPABILITIES,
        _fixed_model_fields(
            checkpoint=checkpoint,
            text_checkpoint=text_checkpoint,
            upsampler=upsampler,
            transformer_adapters=transformer_adapters,
            device_index=device_index,
        )
        + (
            exposed(_PROMPT),
            exposed(_START_IMAGE),
            exposed(_WIDTH, default=512),
            exposed(_HEIGHT, default=512),
            exposed(_DURATION, default=5.0),
            exposed(_SEED, default=0),
        ),
    )


def ltx23_flf_recipe(
    *,
    checkpoint: str | Path,
    text_checkpoint: str | Path,
    device_index: int = 0,
) -> Recipe:
    """Define one LTX FLF product over the existing runtime contract."""
    return Recipe(
        "ltx23.flf.v1_1",
        LTX23_FLF_CAPABILITIES,
        (
            fixed(_CHECKPOINT, Artifact(checkpoint)),
            fixed(_TEXT_CHECKPOINT, Artifact(text_checkpoint)),
            fixed(_DEVICE_INDEX, device_index),
            exposed(_PROMPT),
            exposed(_START_IMAGE),
            exposed(_END_IMAGE),
            exposed(_FLF_WIDTH, default=512),
            exposed(_FLF_HEIGHT, default=512),
            exposed(_DURATION, default=5.0),
            exposed(_SEED, default=0),
        ),
    )


def _resolved_adapters(
    values: Mapping[str, object],
) -> tuple[str | None, float, tuple[tuple[str, float], ...]]:
    artifacts = values["transformer_adapter_artifacts"]
    strengths = values["transformer_adapter_strengths"]
    assert isinstance(artifacts, tuple) and isinstance(strengths, tuple)
    ordered_adapters = tuple(
        (str(artifact.path), float(strength))
        for artifact, strength in zip(artifacts, strengths, strict=True)
    )
    if len(ordered_adapters) == 1:
        lora_path, lora_strength = ordered_adapters[0]
        return lora_path, lora_strength, ()
    return None, 0.5, ordered_adapters


def resolve_ltx23_t2v(
    definition: Recipe, overrides: Mapping[str, object]
) -> tuple[Ltx23T2VIdentity, dict[str, object]]:
    """Resolve recipe policy into the existing LTX identity and generate arguments."""
    if definition.capabilities is not LTX23_T2V_CAPABILITIES:
        raise TypeError("recipe does not use the LTX 2.3 T2V capability set")
    values = definition.resolve(overrides)
    lora_path, lora_strength, transformer_loras = _resolved_adapters(values)
    identity = Ltx23T2VIdentity(
        checkpoint_path=str(values["checkpoint"].path),  # type: ignore[union-attr]
        text_checkpoint_path=str(values["text_checkpoint"].path),  # type: ignore[union-attr]
        transformer_lora_path=lora_path,
        upsampler_path=str(values["upsampler"].path),  # type: ignore[union-attr]
        lora_strength=lora_strength,
        device_index=values["device_index"],  # type: ignore[arg-type]
        transformer_loras=transformer_loras,
    )
    request = {
        key: values[key]
        for key in ("prompt", "width", "height", "duration_seconds", "seed")
    }
    return identity, request


def resolve_ltx23_i2v(
    definition: Recipe, overrides: Mapping[str, object]
) -> tuple[Ltx23I2VIdentity, dict[str, object]]:
    """Resolve policy into the existing I2V identity and generate arguments."""
    if definition.capabilities is not LTX23_I2V_CAPABILITIES:
        raise TypeError("recipe does not use the LTX 2.3 I2V capability set")
    values = definition.resolve(overrides)
    lora_path, lora_strength, transformer_loras = _resolved_adapters(values)
    identity = Ltx23I2VIdentity(
        checkpoint_path=str(values["checkpoint"].path),  # type: ignore[union-attr]
        text_checkpoint_path=str(values["text_checkpoint"].path),  # type: ignore[union-attr]
        transformer_lora_path=lora_path,
        upsampler_path=str(values["upsampler"].path),  # type: ignore[union-attr]
        lora_strength=lora_strength,
        device_index=values["device_index"],  # type: ignore[arg-type]
        transformer_loras=transformer_loras,
    )
    return identity, {
        "prompt": values["prompt"],
        "image_path": values["start_image"],
        "width": values["width"],
        "height": values["height"],
        "duration_seconds": values["duration_seconds"],
        "seed": values["seed"],
    }


def resolve_ltx23_flf(
    definition: Recipe, overrides: Mapping[str, object]
) -> tuple[Ltx23FlfIdentity, dict[str, object]]:
    """Resolve policy to the existing LTX FLF identity and generate arguments."""
    if definition.capabilities is not LTX23_FLF_CAPABILITIES:
        raise TypeError("recipe does not use the LTX 2.3 FLF capability set")
    values = definition.resolve(overrides)
    identity = Ltx23FlfIdentity(
        checkpoint_path=str(values["checkpoint"].path),  # type: ignore[union-attr]
        text_checkpoint_path=str(values["text_checkpoint"].path),  # type: ignore[union-attr]
        device_index=values["device_index"],  # type: ignore[arg-type]
    )
    return identity, {
        "prompt": values["prompt"],
        "first_image_path": values["start_image"],
        "last_image_path": values["end_image"],
        "width": values["width"],
        "height": values["height"],
        "duration_seconds": values["duration_seconds"],
        "seed": values["seed"],
    }
