"""Inert family-neutral capability, recipe policy, and caller-surface primitives."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

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
    """One value and its inherent domain in a concrete family operation."""

    key: str
    value_type: str
    optional: bool = False
    ordered: bool = False
    role: str | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float | None = None
    choices: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("capability key must not be empty")
        if self.value_type not in _VALUE_TYPES:
            raise ValueError(f"unsupported capability type: {self.value_type}")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"{self.key} step must be positive")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(f"{self.key} minimum exceeds maximum")
        for boundary in (self.minimum, self.maximum):
            if boundary is not None:
                self._validate_item(boundary)
                self._validate_domain_item(boundary)
        for choice in self.choices:
            self._validate_item(choice)

    def normalize(self, value: object) -> object:
        if value is None:
            if not self.optional:
                raise TypeError(f"{self.key} does not accept None")
            return None
        if self.ordered:
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"{self.key} must be an ordered collection")
            normalized = tuple(self._validate_item(item) for item in value)
            for item in normalized:
                self._validate_domain_item(item)
            return normalized
        normalized = self._validate_item(value)
        self._validate_domain_item(normalized)
        return normalized

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

    def _validate_domain_item(self, value: object) -> None:
        if self.choices and value not in self.choices:
            raise ValueError(f"{self.key} must be one of {self.choices!r}")
        if self.minimum is not None and value < self.minimum:  # type: ignore[operator]
            raise ValueError(f"{self.key} must be at least {self.minimum}")
        if self.maximum is not None and value > self.maximum:  # type: ignore[operator]
            raise ValueError(f"{self.key} must be at most {self.maximum}")
        if self.step is not None:
            anchor = self.minimum if self.minimum is not None else 0
            quotient = (float(value) - float(anchor)) / float(self.step)
            if not math.isclose(quotient, round(quotient), abs_tol=1e-9):
                raise ValueError(f"{self.key} must use increments of {self.step}")


@dataclass(frozen=True)
class CapabilitySet:
    """A family's declared capabilities and cross-value operation invariants."""

    key: str
    capabilities: tuple[Capability, ...]
    validate: Callable[[Mapping[str, object]], None] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        keys = tuple(capability.key for capability in self.capabilities)
        if not self.key:
            raise ValueError("capability set key must not be empty")
        if len(keys) != len(set(keys)):
            raise ValueError("capability keys must be unique")

    def __getitem__(self, key: str) -> Capability:
        for capability in self.capabilities:
            if capability.key == key:
                return capability
        raise KeyError(key)


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
        self._validate_narrowing()
        if self.value is not _MISSING:
            normalized = self.capability.normalize(self.value)
            object.__setattr__(self, "value", normalized)
            self.validate(normalized)

    def _validate_narrowing(self) -> None:
        capability = self.capability
        if (
            self.minimum is not None
            and capability.minimum is not None
            and self.minimum < capability.minimum
        ):
            raise ValueError(
                f"{capability.key} recipe minimum cannot be lower than capability minimum"
            )
        if (
            self.maximum is not None
            and capability.maximum is not None
            and self.maximum > capability.maximum
        ):
            raise ValueError(
                f"{capability.key} recipe maximum cannot exceed capability maximum"
            )
        for boundary in (self.minimum, self.maximum):
            if boundary is not None:
                capability._validate_item(boundary)
                capability._validate_domain_item(boundary)
        for choice in self.choices:
            capability._validate_item(choice)
            capability._validate_domain_item(choice)
        if self.step is not None and capability.step is not None:
            quotient = float(self.step) / float(capability.step)
            if not math.isclose(quotient, round(quotient), abs_tol=1e-9):
                raise ValueError(
                    f"{capability.key} recipe step must preserve capability increments"
                )

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
                anchor = (
                    self.minimum
                    if self.minimum is not None
                    else self.capability.minimum
                    if self.capability.minimum is not None
                    else 0
                )
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
    """An in-code product policy over one concrete family capability set."""

    key: str
    capabilities: CapabilitySet
    fields: tuple[Field, ...]

    def __post_init__(self) -> None:
        field_keys = tuple(item.capability.key for item in self.fields)
        capability_keys = tuple(item.key for item in self.capabilities.capabilities)
        if len(field_keys) != len(set(field_keys)):
            raise ValueError("recipe capability keys must be unique")
        if set(field_keys) != set(capability_keys):
            raise ValueError("recipe must define policy for every family capability")
        declared = {id(item) for item in self.capabilities.capabilities}
        if any(id(item.capability) not in declared for item in self.fields):
            raise ValueError("recipe fields must reuse declared capability objects")

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
        if self.capabilities.validate is not None:
            self.capabilities.validate(resolved)
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
                    (
                        "min",
                        item.minimum
                        if item.minimum is not None
                        else capability.minimum,
                    ),
                    (
                        "max",
                        item.maximum
                        if item.maximum is not None
                        else capability.maximum,
                    ),
                    (
                        "step",
                        item.step if item.step is not None else capability.step,
                    ),
                    (
                        "choices",
                        _surface_value(item.choices or capability.choices)
                        if item.choices or capability.choices
                        else None,
                    ),
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
