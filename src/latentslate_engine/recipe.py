"""Small Torch-free capability, recipe policy, and caller-surface contract."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .klein9b.contracts import (
    KLEIN_ALIGNMENT,
    KLEIN_MAX_PIXELS,
    KLEIN_MAX_SEED,
    KLEIN_MIN_SIDE,
    Klein9BIdentity,
    validate_klein_dimensions,
    validate_klein_seed,
)
from .ltx23.contracts import (
    MAX_DURATION_SECONDS as LTX_MAX_DURATION_SECONDS,
)
from .ltx23.contracts import (
    MAX_PIXELS as LTX_MAX_PIXELS,
)
from .ltx23.contracts import (
    MAX_SEED as LTX_MAX_SEED,
)
from .ltx23.contracts import (
    MIN_DURATION_SECONDS as LTX_MIN_DURATION_SECONDS,
)
from .ltx23.contracts import (
    MIN_SIDE as LTX_MIN_SIDE,
)
from .ltx23.contracts import (
    Ltx23T2VIdentity,
    validate_ltx_request,
)
from .wan2214b.contracts import (
    MAX_PIXELS as WAN_MAX_PIXELS,
)
from .wan2214b.contracts import (
    MAX_SEED as WAN_MAX_SEED,
)
from .wan2214b.contracts import (
    MIN_SIDE as WAN_MIN_SIDE,
)
from .wan2214b.contracts import (
    WanRecipe,
)
from .wan2214b.contracts import (
    validate_request as validate_wan_request,
)
from .wan2214b.timing import (
    DURATION_STEP_SECONDS as WAN_DURATION_STEP_SECONDS,
)
from .wan2214b.timing import (
    MAX_DURATION_SECONDS as WAN_MAX_DURATION_SECONDS,
)
from .wan2214b.timing import (
    MIN_DURATION_SECONDS as WAN_MIN_DURATION_SECONDS,
)
from .wan2214b.timing import (
    native_frame_count,
)

_MISSING = object()
_VALUE_TYPES = frozenset(
    {
        "text",
        "number",
        "integer",
        "boolean",
        "choice",
        "image",
        "video",
        "audio",
        "artifact",
        "adapter",
    }
)


@dataclass(frozen=True)
class Artifact:
    """A recipe-selected artifact path, not a cross-family model identity."""

    path: Path

    def __init__(self, path: str | Path) -> None:
        object.__setattr__(self, "path", Path(path))


@dataclass(frozen=True)
class Adapter:
    """One ordered adapter artifact and the strength its family consumes."""

    artifact: Artifact
    strength: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.strength, bool) or not isinstance(
            self.strength, (int, float)
        ):
            raise TypeError("adapter strength must be numeric")
        if not math.isfinite(float(self.strength)):
            raise ValueError("adapter strength must be finite")


@dataclass(frozen=True)
class Capability:
    """One value genuinely accepted by a concrete family operation."""

    key: str
    value_type: str
    optional: bool = False
    ordered: bool = False
    role: str | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("capability key must not be empty")
        if self.value_type not in _VALUE_TYPES:
            raise ValueError(f"unsupported capability type: {self.value_type}")

    def normalize(self, value: object) -> object:
        if value is None:
            if not self.optional:
                raise TypeError(f"{self.key} does not accept None")
            return None
        if self.ordered:
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"{self.key} must be an ordered collection")
            return tuple(self._validate_item(item) for item in value)
        return self._validate_item(value)

    def _validate_item(self, value: object) -> object:
        valid = True
        if self.value_type == "text":
            valid = isinstance(value, str)
        elif self.value_type == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif self.value_type == "number":
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            )
        elif self.value_type == "boolean":
            valid = isinstance(value, bool)
        elif self.value_type in {"image", "video", "audio"}:
            valid = isinstance(value, (str, Path))
        elif self.value_type == "artifact":
            valid = isinstance(value, Artifact)
        elif self.value_type == "adapter":
            valid = isinstance(value, Adapter)
        if not valid:
            raise TypeError(f"{self.key} must be {self.value_type}")
        return value


@dataclass(frozen=True)
class Field:
    """Recipe policy for one capability: fixed or caller-exposed."""

    capability: Capability
    value: object = field(default=_MISSING, repr=False)
    exposed: bool = False
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float | None = None
    choices: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not self.exposed and self.value is _MISSING:
            raise ValueError(f"fixed field {self.capability.key} requires a value")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"{self.capability.key} step must be positive")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(f"{self.capability.key} minimum exceeds maximum")
        if self.value is not _MISSING:
            normalized = self.capability.normalize(self.value)
            object.__setattr__(self, "value", normalized)
            self.validate(normalized)

    def validate(self, value: object) -> None:
        if value is None:
            return
        values = value if self.capability.ordered else (value,)
        for item in values:
            if self.choices and item not in self.choices:
                raise ValueError(
                    f"{self.capability.key} must be one of {self.choices!r}"
                )
            if self.minimum is not None and item < self.minimum:  # type: ignore[operator]
                raise ValueError(
                    f"{self.capability.key} must be at least {self.minimum}"
                )
            if self.maximum is not None and item > self.maximum:  # type: ignore[operator]
                raise ValueError(
                    f"{self.capability.key} must be at most {self.maximum}"
                )
            if self.step is not None:
                anchor = self.minimum or 0
                quotient = (float(item) - float(anchor)) / float(self.step)
                if not math.isclose(quotient, round(quotient), abs_tol=1e-9):
                    raise ValueError(
                        f"{self.capability.key} must use increments of {self.step}"
                    )


def fixed(capability: Capability, value: object) -> Field:
    return Field(capability, value=value)


def exposed(
    capability: Capability,
    *,
    default: object = _MISSING,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
    choices: tuple[object, ...] = (),
) -> Field:
    return Field(
        capability,
        value=default,
        exposed=True,
        minimum=minimum,
        maximum=maximum,
        step=step,
        choices=choices,
    )


@dataclass(frozen=True)
class Recipe:
    """An in-code product definition over concrete family capabilities."""

    key: str
    fields: tuple[Field, ...]
    validate: Callable[[Mapping[str, object]], None] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        keys = tuple(item.capability.key for item in self.fields)
        if len(keys) != len(set(keys)):
            raise ValueError("recipe capability keys must be unique")

    def resolve(self, overrides: Mapping[str, object]) -> dict[str, object]:
        fields = {item.capability.key: item for item in self.fields}
        fixed_overrides = sorted(
            key for key in overrides if key in fields and not fields[key].exposed
        )
        if fixed_overrides:
            raise ValueError(
                f"recipe fields are fixed and cannot be overridden: {fixed_overrides}"
            )
        unknown = sorted(key for key in overrides if key not in fields)
        if unknown:
            raise ValueError(f"unknown recipe overrides: {unknown}")

        resolved: dict[str, object] = {}
        missing: list[str] = []
        for item in self.fields:
            key = item.capability.key
            value = overrides.get(key, item.value)
            if value is _MISSING:
                missing.append(key)
                continue
            normalized = item.capability.normalize(value)
            item.validate(normalized)
            resolved[key] = normalized
        if missing:
            raise ValueError(f"missing required recipe inputs: {sorted(missing)}")
        if self.validate is not None:
            self.validate(resolved)
        return resolved

    def surface(self) -> tuple[dict[str, object], ...]:
        result: list[dict[str, object]] = []
        for item in self.fields:
            if not item.exposed:
                continue
            capability = item.capability
            descriptor: dict[str, object] = {
                "key": capability.key,
                "type": capability.value_type,
                "required": item.value is _MISSING,
            }
            if item.value is not _MISSING:
                descriptor["default"] = _surface_value(item.value)
            if capability.optional:
                descriptor["nullable"] = True
            if capability.ordered:
                descriptor["collection"] = True
                descriptor["ordered"] = True
            if capability.role is not None:
                descriptor["role"] = capability.role
            constraints = {
                key: value
                for key, value in (
                    ("min", item.minimum),
                    ("max", item.maximum),
                    ("step", item.step),
                    ("choices", _surface_value(item.choices) if item.choices else None),
                )
                if value is not None
            }
            if constraints:
                descriptor["constraints"] = constraints
            result.append(descriptor)
        return tuple(result)


def _surface_value(value: object) -> object:
    if isinstance(value, Artifact):
        return str(value.path)
    if isinstance(value, Adapter):
        return {"artifact": str(value.artifact.path), "strength": value.strength}
    if isinstance(value, tuple):
        return [_surface_value(item) for item in value]
    return value


def ltx23_t2v_recipe(
    *,
    checkpoint: str | Path,
    text_checkpoint: str | Path,
    upsampler: str | Path,
    transformer_adapters: Sequence[Adapter] = (),
    device_index: int = 0,
) -> Recipe:
    """Define one LTX T2V product over the existing T2V operation."""

    def validate(values: Mapping[str, object]) -> None:
        validate_ltx_request(
            values["width"],  # type: ignore[arg-type]
            values["height"],  # type: ignore[arg-type]
            values["duration_seconds"],  # type: ignore[arg-type]
            values["seed"],  # type: ignore[arg-type]
            alignment=64,
        )

    return Recipe(
        "ltx23.t2v.v1",
        (
            fixed(Capability("checkpoint", "artifact"), Artifact(checkpoint)),
            fixed(Capability("text_checkpoint", "artifact"), Artifact(text_checkpoint)),
            fixed(Capability("upsampler", "artifact"), Artifact(upsampler)),
            fixed(
                Capability("transformer_adapters", "adapter", ordered=True),
                tuple(transformer_adapters),
            ),
            fixed(Capability("device_index", "integer"), device_index),
            exposed(Capability("prompt", "text")),
            exposed(
                Capability("width", "integer", role="width"),
                default=512,
                minimum=LTX_MIN_SIDE,
                maximum=LTX_MAX_PIXELS // LTX_MIN_SIDE,
                step=64,
            ),
            exposed(
                Capability("height", "integer", role="height"),
                default=512,
                minimum=LTX_MIN_SIDE,
                maximum=LTX_MAX_PIXELS // LTX_MIN_SIDE,
                step=64,
            ),
            exposed(
                Capability("duration_seconds", "number", role="duration_seconds"),
                default=5.0,
                minimum=LTX_MIN_DURATION_SECONDS,
                maximum=LTX_MAX_DURATION_SECONDS,
                step=0.5,
            ),
            exposed(
                Capability("seed", "integer", role="seed"),
                default=0,
                minimum=0,
                maximum=LTX_MAX_SEED,
            ),
        ),
        validate,
    )


def resolve_ltx23_t2v(
    definition: Recipe, overrides: Mapping[str, object]
) -> tuple[Ltx23T2VIdentity, dict[str, object]]:
    values = definition.resolve(overrides)
    adapters = values["transformer_adapters"]
    assert isinstance(adapters, tuple)
    if len(adapters) == 1:
        single = adapters[0]
        assert isinstance(single, Adapter)
        lora_path = str(single.artifact.path)
        lora_strength = float(single.strength)
        ordered_loras: tuple[tuple[str, float], ...] = ()
    else:
        lora_path = None
        lora_strength = 0.5
        ordered_loras = tuple(
            (str(adapter.artifact.path), float(adapter.strength))
            for adapter in adapters
        )
    identity = Ltx23T2VIdentity(
        checkpoint_path=str(values["checkpoint"].path),  # type: ignore[union-attr]
        text_checkpoint_path=str(values["text_checkpoint"].path),  # type: ignore[union-attr]
        transformer_lora_path=lora_path,
        upsampler_path=str(values["upsampler"].path),  # type: ignore[union-attr]
        lora_strength=lora_strength,
        device_index=values["device_index"],  # type: ignore[arg-type]
        transformer_loras=ordered_loras,
    )
    request = {
        key: values[key]
        for key in ("prompt", "width", "height", "duration_seconds", "seed")
    }
    return identity, request


def klein9b_two_image_recipe(
    *,
    diffusion: str | Path,
    text_encoder: str | Path,
    vae: str | Path,
    tokenizer: str | Path,
    loras: Sequence[str | Path] = (),
) -> Recipe:
    """Define one Klein two-image product without exposing fictional sampling knobs."""

    def validate(values: Mapping[str, object]) -> None:
        validate_klein_seed(values["seed"])  # type: ignore[arg-type]
        width = values["width"]
        height = values["height"]
        if (width is None) != (height is None):
            raise ValueError(
                "width and height must either both be provided or both omitted"
            )
        if width is not None and height is not None:
            validate_klein_dimensions(width, height)  # type: ignore[arg-type]

    return Recipe(
        "flux2_klein9b.two_image.v1",
        (
            fixed(Capability("diffusion", "artifact"), Artifact(diffusion)),
            fixed(Capability("text_encoder", "artifact"), Artifact(text_encoder)),
            fixed(Capability("vae", "artifact"), Artifact(vae)),
            fixed(Capability("tokenizer", "artifact"), Artifact(tokenizer)),
            exposed(
                Capability("loras", "artifact", ordered=True),
                default=tuple(Artifact(path) for path in loras),
            ),
            exposed(Capability("prompt", "text")),
            exposed(Capability("image_1", "image", role="start_image")),
            exposed(Capability("image_2", "image", role="end_image")),
            exposed(
                Capability("width", "integer", optional=True, role="width"),
                default=None,
                minimum=KLEIN_MIN_SIDE,
                maximum=KLEIN_MAX_PIXELS // KLEIN_MIN_SIDE,
                step=KLEIN_ALIGNMENT,
            ),
            exposed(
                Capability("height", "integer", optional=True, role="height"),
                default=None,
                minimum=KLEIN_MIN_SIDE,
                maximum=KLEIN_MAX_PIXELS // KLEIN_MIN_SIDE,
                step=KLEIN_ALIGNMENT,
            ),
            exposed(
                Capability("seed", "integer", role="seed"),
                default=0,
                minimum=0,
                maximum=KLEIN_MAX_SEED,
            ),
        ),
        validate,
    )


def resolve_klein9b_two_image(
    definition: Recipe, overrides: Mapping[str, object]
) -> tuple[Klein9BIdentity, dict[str, object]]:
    values = definition.resolve(overrides)
    identity = Klein9BIdentity.from_paths(
        values["diffusion"].path,  # type: ignore[union-attr]
        values["text_encoder"].path,  # type: ignore[union-attr]
        values["vae"].path,  # type: ignore[union-attr]
        values["tokenizer"].path,  # type: ignore[union-attr]
        loras=tuple(artifact.path for artifact in values["loras"]),  # type: ignore[union-attr]
    )
    request = {
        "prompt": values["prompt"],
        "first_image": Path(values["image_1"]),  # type: ignore[arg-type]
        "second_image": Path(values["image_2"]),  # type: ignore[arg-type]
        "seed": values["seed"],
        "width": values["width"],
        "height": values["height"],
    }
    return identity, request


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
    """Define one Wan T2V product while retaining separate high/low ownership."""

    def validate(values: Mapping[str, object]) -> None:
        for key in ("high_adapters", "low_adapters"):
            adapters = values[key]
            if not isinstance(adapters, tuple) or len(adapters) > 2:
                raise ValueError(
                    f"{key} supports at most primary and secondary adapters"
                )
        frames = native_frame_count(values["duration_seconds"])  # type: ignore[arg-type]
        validate_wan_request(
            values["width"],  # type: ignore[arg-type]
            values["height"],  # type: ignore[arg-type]
            frames,
            values["seed"],  # type: ignore[arg-type]
        )

    return Recipe(
        "wan2214b.t2v.v1",
        (
            fixed(Capability("high_checkpoint", "artifact"), Artifact(high_checkpoint)),
            fixed(
                Capability("high_adapters", "adapter", ordered=True),
                tuple(high_adapters),
            ),
            fixed(Capability("low_checkpoint", "artifact"), Artifact(low_checkpoint)),
            fixed(
                Capability("low_adapters", "adapter", ordered=True),
                tuple(low_adapters),
            ),
            fixed(Capability("text_encoder", "artifact"), Artifact(text_encoder)),
            fixed(Capability("vae", "artifact"), Artifact(vae)),
            fixed(Capability("negative_prompt", "text"), negative_prompt),
            fixed(Capability("shift", "number"), 5.000000000000001),
            fixed(Capability("steps", "integer"), 4),
            fixed(Capability("split_step", "integer"), 2),
            fixed(Capability("cfg", "number"), 1.0),
            exposed(Capability("prompt", "text")),
            exposed(
                Capability("width", "integer", role="width"),
                default=512,
                minimum=WAN_MIN_SIDE,
                maximum=WAN_MAX_PIXELS // WAN_MIN_SIDE,
                step=16,
            ),
            exposed(
                Capability("height", "integer", role="height"),
                default=512,
                minimum=WAN_MIN_SIDE,
                maximum=WAN_MAX_PIXELS // WAN_MIN_SIDE,
                step=16,
            ),
            exposed(
                Capability("duration_seconds", "number", role="duration_seconds"),
                default=5.0,
                minimum=WAN_MIN_DURATION_SECONDS,
                maximum=WAN_MAX_DURATION_SECONDS,
                step=WAN_DURATION_STEP_SECONDS,
            ),
            exposed(
                Capability("seed", "integer", role="seed"),
                default=0,
                minimum=0,
                maximum=WAN_MAX_SEED,
            ),
        ),
        validate,
    )


def resolve_wan2214b_t2v(
    definition: Recipe, overrides: Mapping[str, object]
) -> tuple[WanRecipe, dict[str, object]]:
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
