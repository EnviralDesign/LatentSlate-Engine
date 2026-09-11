"""Portable recipe documents compiled through the existing family-owned rules."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path, PureWindowsPath

from .artifact_sources import validate_reference
from .klein9b import authoring as klein
from .ltx23 import authoring as ltx
from .recipe import _MISSING, Adapter, Artifact, Field, Recipe, fixed
from .wan2214b import authoring as wan

OPERATIONS = {
    policy.capabilities.key: (family, policy)
    for family in (ltx, klein, wan)
    for policy in family.POLICIES
}
_CONSTRAINTS = {"minimum", "maximum", "step", "choices", "nullable"}


def validate_authoring_contract(family) -> dict[str, dict[str, str]]:
    """Prove an explicit ownership partition for every supported operation."""
    classes = {
        "caller": set(family.CALLER_INPUTS),
        "recipe": set(family.RECIPE_FIELDS),
        "artifact": set(family.ARTIFACT_SLOTS),
        "host": set(family.HOST_BINDINGS),
    }
    declared = set().union(*classes.values())
    # Metadata is shared across a family's operations; an operation may omit
    # fields another uses (e.g. LTX FLF has no upsampler). No entry may be stale
    # across the family's actual declared operation sets.
    known = {
        cap.key
        for policy in family.POLICIES
        for cap in policy.capabilities.capabilities
    }
    if stale := declared - known:
        raise ValueError(f"Stale authoring metadata: {sorted(stale)}")
    result = {}
    for policy in family.POLICIES:
        partition = {}
        for capability in policy.capabilities.capabilities:
            owners = [
                owner for owner, keys in classes.items() if capability.key in keys
            ]
            if len(owners) != 1:
                raise ValueError(
                    f"{policy.capabilities.key}.{capability.key} must have exactly one authoring owner; got {owners}"
                )
            owner = owners[0]
            if owner == "artifact" and capability.value_type not in {
                "artifact",
                "adapter",
            }:
                raise ValueError(
                    f"{capability.key} artifact slot must use an artifact or adapter capability"
                )
            if capability.value_type in {"artifact", "adapter"} and owner != "artifact":
                raise ValueError(f"{capability.key} requires artifact ownership")
            partition[capability.key] = owner
        result[policy.capabilities.key] = partition
    return result


OPERATION_OWNERSHIP = {
    key: partition
    for family in (ltx, klein, wan)
    for key, partition in validate_authoring_contract(family).items()
}


def canonical_bytes(value: object) -> bytes:
    """Canonical UTF-8 JSON; path strings are never interpreted here."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def definition_hash(document: dict) -> str:
    """Hash semantic policy independently of display name, UUID and revision."""
    return hashlib.sha256(
        canonical_bytes(
            {key: document[key] for key in ("format_version", "operation", "fields")}
        )
    ).hexdigest()


def operation_descriptors() -> list[dict]:
    """Describe inherent domains and family ownership, separate from the catalog."""
    result = []
    for key, (family, policy) in OPERATIONS.items():
        fields = []
        for capability in policy.capabilities.capabilities:
            name = capability.key
            owner = OPERATION_OWNERSHIP[key][name]
            fields.append(
                {
                    **asdict(capability),
                    "owner": owner,
                    **(
                        {"presentation": family.FIELD_PRESENTATION[name]}
                        if name in getattr(family, "FIELD_PRESENTATION", {})
                        else {}
                    ),
                    **(
                        {"artifact": family.ARTIFACT_SLOTS[name]}
                        if owner == "artifact"
                        else {}
                    ),
                }
            )
        result.append({"key": key, "fields": fields})
    return result


def local_reference(path: str) -> dict:
    return {"source": "local", "path": path}


def _encode(value: object) -> object:
    if isinstance(value, Artifact):
        return local_reference(str(value.path))
    if isinstance(value, Adapter):
        return {"artifact": _encode(value.artifact), "strength": value.strength}
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    return value


def document_from_recipe(recipe: Recipe, *, name: str, recipe_id: str) -> dict:
    """Capture a certified bound recipe; host device state is deliberately omitted."""
    ownership = OPERATION_OWNERSHIP[recipe.capabilities.key]
    fields = []
    for item in recipe.fields:
        if ownership[item.capability.key] == "host":
            continue
        field = {
            "key": item.capability.key,
            "mode": "exposed" if item.exposed else "fixed",
        }
        if item.value is not _MISSING:
            field["value"] = _encode(item.value)
        for key in sorted(_CONSTRAINTS):
            value = getattr(item, key)
            if value is not None and value != ():
                field[key] = _encode(value)
        fields.append(field)
    return {
        "format_version": 1,
        "id": recipe_id,
        "name": name,
        "operation": recipe.capabilities.key,
        "fields": fields,
    }


class DocumentError(ValueError):
    def __init__(self, path: str, message: str):
        super().__init__(message)
        self.path = path


def parse_document(value: object) -> dict:
    """Strictly parse the versioned wire format without resolving local paths."""
    if not isinstance(value, dict) or set(value) != {
        "format_version",
        "id",
        "name",
        "operation",
        "fields",
    }:
        raise DocumentError(
            "$", "Expected format_version, id, name, operation and fields only"
        )
    if type(value["format_version"]) is not int or value["format_version"] != 1:
        raise DocumentError("format_version", "Unsupported authoring document version")
    try:
        if (
            not isinstance(value["id"], str)
            or str(uuid.UUID(value["id"])) != value["id"]
        ):
            raise ValueError
    except ValueError:
        raise DocumentError("id", "Expected a canonical UUID") from None
    for key in ("name", "operation"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise DocumentError(key, "Expected a non-empty string")
    if not isinstance(value["fields"], list):
        raise DocumentError("fields", "Expected an ordered field list")
    for index, item in enumerate(value["fields"]):
        path = f"fields[{index}]"
        if (
            not isinstance(item, dict)
            or not {"key", "mode"} <= item.keys()
            or item.keys() - ({"key", "mode", "value"} | _CONSTRAINTS)
        ):
            raise DocumentError(
                path, "Expected a field key, mode, optional value and constraints"
            )
        if not isinstance(item["key"], str) or item["mode"] not in ("fixed", "exposed"):
            raise DocumentError(path, "Invalid field key or mode")
        for key in ("minimum", "maximum", "step"):
            number = item.get(key)
            if key in item and (
                type(number) not in (int, float) or not math.isfinite(number)
            ):
                raise DocumentError(f"{path}.{key}", "Expected a finite number")
        if "choices" in item and not isinstance(item["choices"], list):
            raise DocumentError(f"{path}.choices", "Expected a choice list")
        if "nullable" in item and type(item["nullable"]) is not bool:
            raise DocumentError(f"{path}.nullable", "Expected a boolean")
    try:
        canonical_bytes(value)
    except (TypeError, ValueError, UnicodeError):
        raise DocumentError("$", "Document must contain finite JSON values") from None
    return deepcopy(value)


def _reference(value: object) -> str:
    reference = validate_reference(value)
    if reference["source"] != "local":
        raise ValueError(
            "Hugging Face dependency must be materialized and localized before execution"
        )
    return reference["path"]


def _policy_reference(value: object) -> str:
    reference = validate_reference(value)
    if reference["source"] == "local":
        return reference["path"]
    # Only policy/domain validation uses this non-existent path. Execution
    # compilation rejects remote references until the host localizes them.
    return f".unmaterialized/{reference['sha256']}"


def _decode_item(capability, value, reference_path):
    if capability.value_type == "artifact":
        return Artifact(reference_path(value))
    if capability.value_type == "adapter":
        if not isinstance(value, dict) or set(value) != {"artifact", "strength"}:
            raise ValueError("Expected an adapter artifact and strength")
        return Adapter(Artifact(reference_path(value["artifact"])), value["strength"])
    return value


def _decode(capability, value, reference_path):
    if capability.ordered and value is not None:
        if not isinstance(value, list):
            raise ValueError("Expected an ordered JSON list")
        return tuple(_decode_item(capability, item, reference_path) for item in value)
    return _decode_item(capability, value, reference_path)


def _issue(stage: str, code: str, path: str, message: str, remediation: str) -> dict:
    return {
        "stage": stage,
        "severity": "error",
        "code": code,
        "path": path,
        "message": message,
        "remediation": remediation,
    }


def _compile(
    document: dict, reference_path=_policy_reference
) -> tuple[Recipe | None, list[dict]]:
    if document["operation"] not in OPERATIONS:
        return None, [
            _issue(
                "compile",
                "unknown_operation",
                "operation",
                "Unknown family operation",
                "Choose an operation from authoring introspection",
            )
        ]
    family, policy = OPERATIONS[document["operation"]]
    ownership = OPERATION_OWNERSHIP[document["operation"]]
    fields, issues = [], []
    for index, item in enumerate(document["fields"]):
        path = f"fields[{index}]"
        try:
            key = item["key"]
            capability = policy.capabilities[key]
            owner = ownership[key]
            if owner == "host":
                raise ValueError("Host state cannot be stored in recipe policy")
            if owner == "caller" and item != {"key": key, "mode": "exposed"}:
                raise ValueError(
                    "Prompt and media inputs remain caller-owned; omit stored values and constraints"
                )
            if owner == "artifact" and item["mode"] != "fixed":
                raise ValueError("Artifact selections must be fixed recipe content")
            kwargs = {
                key: tuple(value) if key == "choices" else value
                for key, value in item.items()
                if key in _CONSTRAINTS
            }
            if "value" in item:
                kwargs["value"] = _decode(capability, item["value"], reference_path)
            fields.append(
                Field(capability, exposed=item["mode"] == "exposed", **kwargs)
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            issues.append(
                _issue(
                    "compile",
                    "invalid_field_policy",
                    path,
                    str(error),
                    "Use the family domain and a compatible fixed value or exposed default",
                )
            )
    if issues:
        return None, issues
    try:
        fields.extend(
            fixed(policy.capabilities[key], family.HOST_BINDINGS[key])
            for key, owner in ownership.items()
            if owner == "host"
        )
        recipe = Recipe(document["id"], policy.capabilities, tuple(fields))
        # All non-caller policies need defaults so actual cross-field validation
        # can run. These placeholders are never persisted or used for inference.
        caller = {
            key: "authoring validation placeholder"
            for key, owner in ownership.items()
            if owner == "caller"
        }
        recipe.resolve(caller)
        return recipe, []
    except (TypeError, ValueError, OverflowError) as error:
        return None, [
            _issue(
                "compile",
                "incompatible_recipe_policy",
                "fields",
                str(error),
                "Provide compatible defaults and satisfy the family cross-field rules",
            )
        ]


def compile_document(value: object, *, policy_only: bool = False) -> Recipe:
    """Compile local execution bindings, or explicitly inspect portable policy."""
    document = parse_document(value)
    recipe, issues = _compile(
        document, _policy_reference if policy_only else _reference
    )
    if recipe is None:
        raise ValueError("; ".join(item["message"] for item in issues))
    return recipe


def artifact_dependencies(document: dict) -> list[dict]:
    """Enumerate the family-declared artifact slots of an exact document."""
    family, policy = OPERATIONS[document["operation"]]
    result = []
    for index, field in enumerate(document["fields"]):
        key = field["key"]
        if key not in family.ARTIFACT_SLOTS:
            continue
        capability = policy.capabilities[key]
        values = field["value"] if capability.ordered else [field["value"]]
        for position, value in enumerate(values):
            path = f"fields[{index}].value" + (
                f"[{position}]" if capability.ordered else ""
            )
            reference = (
                value["artifact"] if capability.value_type == "adapter" else value
            )
            result.append(
                {
                    "key": key,
                    "path": path,
                    "reference": reference,
                    "requirements": family.ARTIFACT_SLOTS[key],
                }
            )
    return result


def localize_document(value: object, resolve_artifact) -> dict:
    """Return transient host bindings while preserving the canonical input."""
    document = parse_document(value)
    paths = {}
    for dependency in artifact_dependencies(document):
        reference = validate_reference(dependency["reference"])
        if reference["source"] == "huggingface":
            if dependency["requirements"]["kind"] != "file":
                raise ValueError(
                    "Hugging Face references support file slots, not directories"
                )
            digest = reference["sha256"]
            if digest not in paths:
                paths[digest] = resolve_artifact(reference)
            path = paths[digest]
            reference.clear()
            reference.update(local_reference(str(path)))
    return document


def _resolve_artifacts(document: dict, resolve_artifact=None) -> dict:
    slots, issues = [], []
    entry = OPERATIONS.get(document["operation"])
    if entry is None:
        return {"status": "unresolved", "slots": [], "issues": []}
    family, policy = entry
    ownership = OPERATION_OWNERSHIP[document["operation"]]
    for index, item in enumerate(document["fields"]):
        key = item["key"]
        if ownership.get(key) != "artifact":
            continue
        capability = policy.capabilities[key]
        requirements = family.ARTIFACT_SLOTS[key]
        raw = item.get("value")
        values = raw if capability.ordered and isinstance(raw, list) else [raw]
        for position, value in enumerate(values):
            path = f"fields[{index}].value" + (
                f"[{position}]" if capability.ordered else ""
            )
            slot_issues = []
            reference = None
            try:
                reference = (
                    value["artifact"]
                    if capability.value_type == "adapter" and isinstance(value, dict)
                    else value
                )
                validate_reference(reference)
                if reference["source"] == "huggingface":
                    if requirements["kind"] != "file":
                        raise ValueError(
                            "This slot requires a local directory; a Hugging Face file cannot supply it"
                        )
                    if resolve_artifact is None:
                        raise ValueError(
                            "Hugging Face dependency is not materialized on this host"
                        )
                    raw_path = str(resolve_artifact(reference))
                else:
                    raw_path = _reference(reference)
                # A foreign drive/UNC path is an opaque, unresolved reference.
                # Never reinterpret it as a relative POSIX filename.
                if os.name != "nt" and PureWindowsPath(raw_path).drive:
                    raise ValueError("Windows path cannot resolve on this host")
                local = Path(raw_path)
                if not local.is_absolute():
                    raise ValueError(
                        "Artifact path must be absolute to resolve on this host"
                    )
                exists = (
                    local.is_dir()
                    if requirements["kind"] == "directory"
                    else local.is_file()
                )
                if not exists:
                    raise ValueError(
                        f"Expected {requirements['kind']} does not exist: {raw_path}"
                    )
                for companion in requirements.get("required_files", ()):
                    if not (local / companion).is_file():
                        slot_issues.append(
                            _issue(
                                "artifact",
                                "missing_companion",
                                path,
                                f"Required companion file is missing: {companion}",
                                "Select a complete family artifact directory",
                            )
                        )
            except (KeyError, ValueError, TypeError, OSError) as error:
                slot_issues.append(
                    _issue(
                        "artifact",
                        "unresolved_artifact",
                        path,
                        str(error),
                        "Materialize the pinned file, or select an existing local path of the required kind",
                    )
                )
            slots.append(
                {
                    "key": key,
                    "path": path,
                    "status": "unresolved" if slot_issues else "resolved",
                    **(
                        {"reference": reference}
                        if isinstance(reference, dict)
                        and reference.get("source") == "huggingface"
                        else {}
                    ),
                    "issues": slot_issues,
                }
            )
            issues.extend(slot_issues)
    present = {item["key"] for item in document["fields"]}
    for capability in policy.capabilities.capabilities:
        if capability.key in family.ARTIFACT_SLOTS and capability.key not in present:
            issues.append(
                _issue(
                    "artifact",
                    "missing_artifact_slot",
                    "fields",
                    f"Missing artifact slot: {capability.key}",
                    "Provide the required artifact policy",
                )
            )
    return {
        "status": "unresolved" if issues else "resolved",
        "slots": slots,
        "issues": issues,
    }


def validate_document(value: object, *, resolve_artifact=None) -> dict:
    """Keep structural, policy, dependency and untested execution results distinct."""
    try:
        document = parse_document(value)
    except DocumentError as error:
        issues = [
            _issue(
                "document",
                "invalid_document",
                error.path,
                str(error),
                "Use the versioned authoring document format",
            )
        ]
        return {
            "document_valid": False,
            "recipe_compiles": False,
            "artifact_resolution": {"status": "unresolved", "slots": [], "issues": []},
            "execution_readiness": {"status": "blocked", "backend_checked": False},
            "issues": issues,
        }
    recipe, issues = _compile(document)
    artifacts = _resolve_artifacts(document, resolve_artifact)
    blocked = recipe is None or artifacts["status"] != "resolved"
    return {
        "document_valid": True,
        "recipe_compiles": recipe is not None,
        "definition_hash": definition_hash(document),
        "caller_surface": list(recipe.surface()) if recipe is not None else None,
        "artifact_resolution": artifacts,
        "execution_readiness": {
            "status": "blocked" if blocked else "unverified",
            "backend_checked": False,
            "message": "Native execution/backend readiness was not checked. Enable a saved recipe to publish an Engine tool.",
        },
        "issues": issues + artifacts["issues"],
    }
