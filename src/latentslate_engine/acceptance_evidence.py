"""Compact, privacy-safe hardware acceptance evidence contracts."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA_VERSION = 1
CONTRACT_FINGERPRINT_VERSION = 2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROHIBITED_KEYS = frozenset(
    {
        "account_id",
        "base_url",
        "bearer",
        "credential",
        "credentials",
        "input",
        "inputs",
        "instance_id",
        "job_id",
        "password",
        "prompt",
        "request",
        "secret",
        "token",
    }
)
_SHARED_EXECUTION_FILES = (
    "src/latentslate_engine/protocol.py",
    "src/latentslate_engine/variants.py",
    "src/latentslate_engine/tools/base.py",
    "src/latentslate_engine/runtime/cache.py",
    "src/latentslate_engine/runtime/manager.py",
    "src/latentslate_engine/runtime/process_memory.py",
    "src/latentslate_engine/runtime/windows_process.py",
)
_FAMILY_PREFIXES = {
    "klein4b": ("klein",),
    "klein9b": ("klein",),
    "ltx23": ("ltx23",),
    "wan22": ("wan22", "wan5", "umt5"),
    "zimage": ("z_image",),
}


class EvidenceValidationError(ValueError):
    """One compact evidence record or claim is invalid."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def builtin_hardware_claims(root: Path | None = None) -> dict[str, dict[str, Any]]:
    root = (root or repository_root()).resolve()
    claims: dict[str, dict[str, Any]] = {}
    recipe_root = root / "src" / "latentslate_engine" / "builtin_recipes"
    for path in sorted(recipe_root.rglob("*.toml")):
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        declaration = document.get("runnable_recipe")
        if not isinstance(declaration, Mapping):
            continue
        tags = declaration.get("tags", [])
        if "hardware-proven" not in tags:
            continue
        key = declaration.get("key")
        if not isinstance(key, str) or key in claims:
            raise EvidenceValidationError(f"duplicate or invalid built-in recipe key in {path}")
        claims[key] = {
            "declaration": dict(declaration),
            "path": path,
            "resource_ids": recipe_resource_ids(declaration),
            "execution_contract_fingerprint": execution_contract_fingerprint(
                declaration, root=root
            ),
        }
    return claims


def recipe_resource_ids(declaration: Mapping[str, Any]) -> tuple[str, ...]:
    values: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str) and value.startswith(("model:", "lora:")):
            values.add(value)
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for field in ("recipe", "loras"):
        visit(declaration.get(field))
    return tuple(sorted(values))


def execution_contract_fingerprint(
    declaration: Mapping[str, Any], *, root: Path | None = None
) -> str:
    root = (root or repository_root()).resolve()
    family = declaration.get("family")
    if not isinstance(family, str) or family not in _FAMILY_PREFIXES:
        raise EvidenceValidationError(f"unsupported evidence family {family!r}")
    contract = {
        key: declaration.get(key)
        for key in (
            "key",
            "family",
            "base_tool",
            "model",
            "recipe",
            "inputs",
            "fixed",
            "loras",
            "optimizations",
        )
    }
    sources: dict[str, str] = {}
    candidates = {root / value for value in _SHARED_EXECUTION_FILES}
    candidates.update((root / "src" / "latentslate_engine" / "runtime" / "framework").rglob("*.py"))
    prefixes = _FAMILY_PREFIXES[family]
    for directory in (
        root / "src" / "latentslate_engine",
        root / "src" / "latentslate_engine" / "runtime",
        root / "src" / "latentslate_engine" / "tools",
    ):
        candidates.update(
            path
            for path in directory.glob("*.py")
            if path.stem.startswith(prefixes)
        )
    for path in sorted(candidates):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            sources[relative] = hashlib.sha256(content).hexdigest()
    payload = {
        "version": CONTRACT_FINGERPRINT_VERSION,
        "contract": contract,
        "sources": sources,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def load_evidence_records(directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvidenceValidationError(f"invalid evidence JSON {path.name}") from exc
        if not isinstance(value, dict):
            raise EvidenceValidationError(f"evidence {path.name} must be an object")
        records.append(value)
    return records


def validate_evidence_records(
    records: Iterable[Mapping[str, Any]],
    *,
    claims: Mapping[str, Mapping[str, Any]],
) -> None:
    by_id: dict[str, Mapping[str, Any]] = {}
    matching_claims: set[str] = set()
    for record in records:
        _validate_record_shape(record)
        evidence_id = str(record["evidence_id"])
        if evidence_id in by_id:
            raise EvidenceValidationError(f"duplicate evidence ID {evidence_id!r}")
        by_id[evidence_id] = record
        _reject_private_fields(record)
        recipe_key = str(record["recipe_key"])
        claim = claims.get(recipe_key)
        if claim is None:
            continue
        if record["execution_contract_fingerprint"] != claim["execution_contract_fingerprint"]:
            continue
        if tuple(record["resource_ids"]) != tuple(claim["resource_ids"]):
            raise EvidenceValidationError(f"resource identities differ for {recipe_key!r}")
        review = record["creator_review"]
        if not isinstance(review, Mapping) or review.get("state") != "accepted":
            continue
        results = record["results"]
        if (
            not isinstance(results, Mapping)
            or results.get("cold") is not True
            or any(value not in (True, "not_applicable") for value in results.values())
        ):
            raise EvidenceValidationError(
                f"accepted evidence lacks complete lifecycle results for {recipe_key!r}"
            )
        _validate_fallback_proof(record, recipe_key)
        matching_claims.add(recipe_key)
    missing = sorted(set(claims) - matching_claims)
    if missing:
        raise EvidenceValidationError(
            "Hardware-proven recipes lack matching accepted evidence: " + ", ".join(missing)
        )


def acceptance_matrix(
    records: Iterable[Mapping[str, Any]], claims: Mapping[str, Mapping[str, Any]]
) -> str:
    accepted = {
        str(record.get("recipe_key")): str(record.get("evidence_id"))
        for record in records
        if isinstance(record.get("creator_review"), Mapping)
        and record["creator_review"].get("state") == "accepted"
        and record.get("recipe_key") in claims
        and record.get("execution_contract_fingerprint")
        == claims[str(record["recipe_key"])]["execution_contract_fingerprint"]
    }
    lines = [
        "# Hardware acceptance matrix",
        "",
        "Generated from versioned compact evidence records; do not edit by hand.",
        "",
        "| Recipe | Evidence |",
        "| --- | --- |",
    ]
    lines.extend(f"| `{key}` | `{accepted.get(key, 'MISSING')}` |" for key in sorted(claims))
    return "\n".join(lines) + "\n"


def _validate_record_shape(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "evidence_id",
        "recipe_key",
        "execution_contract_fingerprint",
        "tool",
        "engine",
        "environment",
        "source_manifests",
        "resource_ids",
        "results",
        "observations",
        "creator_review",
        "known_limitations",
    }
    if set(record) != required:
        raise EvidenceValidationError("evidence record fields are not canonical")
    if record["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceValidationError("unsupported evidence schema version")
    for field in ("evidence_id", "recipe_key"):
        if not isinstance(record[field], str) or not record[field]:
            raise EvidenceValidationError(f"evidence {field} is invalid")
    fingerprint = record["execution_contract_fingerprint"]
    if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:") or not _SHA256.fullmatch(fingerprint[7:]):
        raise EvidenceValidationError("execution-contract fingerprint is invalid")
    if not isinstance(record["resource_ids"], list) or record["resource_ids"] != sorted(set(record["resource_ids"])):
        raise EvidenceValidationError("resource identities must be a sorted unique list")
    tool = record["tool"]
    if (
        not isinstance(tool, Mapping)
        or set(tool) != {"id", "schema_revision", "schema_hash"}
        or not isinstance(tool["id"], str)
        or isinstance(tool["schema_revision"], bool)
        or not isinstance(tool["schema_revision"], int)
        or tool["schema_revision"] <= 0
        or not isinstance(tool["schema_hash"], str)
        or not tool["schema_hash"].startswith("sha256:")
        or not _SHA256.fullmatch(tool["schema_hash"][7:])
    ):
        raise EvidenceValidationError("tool schema identity is invalid")
    engine = record["engine"]
    if (
        not isinstance(engine, Mapping)
        or set(engine) != {"commit", "source_states"}
        or not isinstance(engine["commit"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", engine["commit"])
    ):
        raise EvidenceValidationError("Engine source identity is invalid")
    source_states = engine["source_states"]
    if not isinstance(source_states, list) or not source_states:
        raise EvidenceValidationError("Engine source identity is invalid")
    canonical_states: list[tuple[bool, str | None]] = []
    for state in source_states:
        if (
            not isinstance(state, Mapping)
            or set(state) != {"dirty", "worktree_diff_sha256"}
            or not isinstance(state["dirty"], bool)
            or (
                state["worktree_diff_sha256"] is not None
                and (
                    not isinstance(state["worktree_diff_sha256"], str)
                    or not _SHA256.fullmatch(state["worktree_diff_sha256"])
                )
            )
            or (state["dirty"] is (state["worktree_diff_sha256"] is None))
        ):
            raise EvidenceValidationError("Engine source identity is invalid")
        canonical_states.append((state["dirty"], state["worktree_diff_sha256"]))
    if canonical_states != sorted(set(canonical_states), key=lambda value: (value[0], value[1] or "")):
        raise EvidenceValidationError("Engine source states must be sorted and unique")
    results = record["results"]
    expected_results = {
        "cold",
        "warm",
        "switching",
        "cancellation",
        "worker_exit_cleanup_recovery",
    }
    if (
        not isinstance(results, Mapping)
        or set(results) != expected_results
        or any(value not in (True, False, "not_applicable") for value in results.values())
    ):
        raise EvidenceValidationError("lifecycle results are invalid")
    sources = record["source_manifests"]
    if not isinstance(sources, list) or not sources:
        raise EvidenceValidationError("at least one source manifest identity is required")
    locations: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != {"sha256", "retained_class", "retained_location"}:
            raise EvidenceValidationError("source manifest identity is invalid")
        if not isinstance(source["sha256"], str) or not _SHA256.fullmatch(source["sha256"]):
            raise EvidenceValidationError("source manifest SHA-256 is invalid")
        location = source["retained_location"]
        if (
            not isinstance(location, str)
            or Path(location).is_absolute()
            or not location.startswith("hardware-study-runs/")
            or location in locations
        ):
            raise EvidenceValidationError("source manifest locations must be unique retained relative study paths")
        locations.add(location)


def _validate_fallback_proof(record: Mapping[str, Any], recipe_key: str) -> None:
    observations = record["observations"]
    if not isinstance(observations, Mapping):
        raise EvidenceValidationError(f"observations are invalid for {recipe_key!r}")
    hashes = observations.get("output_hashes")
    if (
        not isinstance(hashes, list)
        or not hashes
        or hashes != sorted(set(hashes))
        or any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in hashes)
    ):
        raise EvidenceValidationError(f"observed output hashes are invalid for {recipe_key!r}")
    if observations.get("fallback_claim") != "zero":
        return
    counters = observations.get("fallback_counters")
    if (
        not isinstance(counters, Mapping)
        or not counters
        or any(isinstance(value, bool) or not isinstance(value, int) or value != 0 for value in counters.values())
    ):
        raise EvidenceValidationError(
            f"zero fallback lacks observed zero counters for {recipe_key!r}"
        )


def _reject_private_fields(value: Any, *, trail: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _PROHIBITED_KEYS or normalized.endswith(("_token", "_secret", "_password")):
                raise EvidenceValidationError(
                    "prohibited private evidence field " + ".".join((*trail, str(key)))
                )
            _reject_private_fields(child, trail=(*trail, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_private_fields(child, trail=(*trail, str(index)))
    elif isinstance(value, str) and (
        re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith(("/home/", "/Users/"))
    ):
        raise EvidenceValidationError("absolute local path is prohibited in evidence")


__all__ = (
    "CONTRACT_FINGERPRINT_VERSION",
    "EVIDENCE_SCHEMA_VERSION",
    "EvidenceValidationError",
    "acceptance_matrix",
    "builtin_hardware_claims",
    "execution_contract_fingerprint",
    "load_evidence_records",
    "recipe_resource_ids",
    "validate_evidence_records",
)
