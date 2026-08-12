from __future__ import annotations

import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any

from ..resources import ResourceDescriptor
from ..variants import VariantDefinition

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")

_RESOURCE_ORDER = (
    "id",
    "kind",
    "family",
    "name",
    "relative_path",
    "format",
    "precision",
    "quantization",
    "size_bytes",
    "base_model",
    "component",
    "description",
    "tags",
    "default_strength",
    "trigger_words",
    "config_path",
    "metadata",
    "sources",
)

_RECIPE_ORDER = (
    "schema_version",
    "id",
    "key",
    "schema_revision",
    "name",
    "description",
    "enabled",
    "family",
    "base_tool",
    "tags",
    "model",
    "recipe",
    "inputs",
    "fixed",
    "loras",
    "optimizations",
)


def render_resource_toml(resource: ResourceDescriptor) -> str:
    payload = resource.model_dump(
        mode="json",
        exclude={"available", "unavailable_reason"},
        exclude_none=True,
    )
    return _render_document("resource", _ordered(payload, _RESOURCE_ORDER))


def render_recipe_toml(definition: VariantDefinition) -> str:
    payload = definition.model_dump(
        mode="json",
        exclude_none=True,
        exclude_unset=True,
    )
    payload.setdefault("schema_version", 1)
    payload.setdefault("schema_revision", definition.schema_revision)
    payload.setdefault("enabled", definition.enabled)
    return _render_document("runnable_recipe", _ordered(payload, _RECIPE_ORDER))


def load_recipe_file(path: Path) -> VariantDefinition:
    suffix = path.suffix.casefold()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("recipe file must contain an object/table")
    payload = raw.get("runnable_recipe", raw.get("variant", raw))
    return VariantDefinition.model_validate(payload)


def _ordered(payload: dict[str, Any], preferred: tuple[str, ...]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in preferred:
        if key in payload:
            ordered[key] = payload[key]
    for key in sorted(set(payload) - set(ordered)):
        ordered[key] = payload[key]
    return ordered


def _render_document(root: str, payload: dict[str, Any]) -> str:
    lines: list[str] = []
    _render_table(lines, (root,), payload, emit_header=True)
    return "\n".join(lines).rstrip() + "\n"


def _render_table(
    lines: list[str],
    path: tuple[str, ...],
    payload: dict[str, Any],
    *,
    emit_header: bool,
) -> None:
    if emit_header:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(f"[{_table_path(path)}]")

    child_tables: list[tuple[str, dict[str, Any]]] = []
    array_tables: list[tuple[str, list[dict[str, Any]]]] = []
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, dict):
            child_tables.append((key, value))
            continue
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            array_tables.append((key, value))
            continue
        lines.append(f"{_key(key)} = {_value(value)}")

    for key, value in child_tables:
        _render_table(lines, (*path, key), value, emit_header=True)
    for key, values in array_tables:
        for value in values:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(f"[[{_table_path((*path, key))}]]")
            _render_table(lines, (*path, key), value, emit_header=False)


def _table_path(parts: tuple[str, ...]) -> str:
    return ".".join(_key(part) for part in parts)


def _key(value: str) -> str:
    return value if _BARE_KEY.fullmatch(value) else json.dumps(value, ensure_ascii=False)


def _value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("TOML output cannot contain non-finite floats")
        return repr(value)
    if isinstance(value, list) or isinstance(value, tuple):
        if any(isinstance(item, (dict, list, tuple)) for item in value):
            raise ValueError("nested arrays require an explicit TOML table shape")
        return "[" + ", ".join(_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value {type(value).__name__}")
