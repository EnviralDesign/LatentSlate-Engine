#!/usr/bin/env python3
"""Extract and validate compact acceptance records from hardware-study manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from latentslate_engine.acceptance_evidence import (
    EVIDENCE_SCHEMA_VERSION,
    EvidenceValidationError,
    acceptance_matrix,
    builtin_hardware_claims,
    load_evidence_records,
    repository_root,
    validate_evidence_records,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--write-matrix", action="store_true")
    extract = commands.add_parser("extract")
    extract.add_argument("--manifest", action="append", type=Path, required=True)
    extract.add_argument("--recipe", required=True)
    extract.add_argument("--evidence-id", required=True)
    extract.add_argument("--review-state", choices=("pending", "accepted", "rejected"), default="pending")
    extract.add_argument("--reviewer", default="unreviewed")
    extract.add_argument("--reviewed-at")
    for name in ("switching", "cancellation", "worker-exit-cleanup-recovery"):
        extract.add_argument(
            f"--{name}",
            choices=("passed", "failed", "not-observed", "not-applicable"),
            default="not-observed",
        )
    extract.add_argument("--known-limitation", action="append", default=[])
    return value


def _load_manifest(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise EvidenceValidationError("source manifests must remain inside the repository") from exc
    if not relative.startswith("hardware-study-runs/") or resolved.name != "manifest.json":
        raise EvidenceValidationError("source must be a retained hardware-study manifest")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format") != "latentslate-hardware-study-v2":
        raise EvidenceValidationError(f"unsupported hardware-study manifest {relative}")
    return value, {
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "retained_class": "local-ignored",
        "retained_location": relative,
    }


def _descriptor(manifests: list[Mapping[str, Any]], recipe_key: str) -> Mapping[str, Any]:
    matches = [
        selected["descriptor"]
        for manifest in manifests
        for selected in manifest.get("selected", [])
        if isinstance(selected, Mapping)
        and isinstance(selected.get("descriptor"), Mapping)
        and selected["descriptor"].get("key") == recipe_key
    ]
    if not matches:
        raise EvidenceValidationError(f"recipe {recipe_key!r} is absent from source manifests")
    identities = {
        (value.get("id"), value.get("schema_revision"), value.get("schema_hash"))
        for value in matches
    }
    if len(identities) != 1:
        raise EvidenceValidationError("tool identity drifted across source manifests")
    return matches[0]


def _fallback_counters(value: Any, result: dict[str, int]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if "fallback" in normalized and isinstance(child, int) and not isinstance(child, bool):
                result[str(key)] = max(result.get(str(key), child), child)
            else:
                _fallback_counters(child, result)
    elif isinstance(value, list):
        for child in value:
            _fallback_counters(child, result)


def extract(args: argparse.Namespace) -> int:
    root = repository_root()
    claims = builtin_hardware_claims(root)
    claim = claims.get(args.recipe)
    if claim is None:
        raise EvidenceValidationError("only current Hardware-proven built-ins can be extracted")
    loaded = [_load_manifest(path, root) for path in args.manifest]
    manifests = [value for value, _source in loaded]
    fingerprints = {
        value
        for manifest in manifests
        if (
            value := manifest.get("execution_contract_fingerprints", {}).get(
                args.recipe
            )
        )
    }
    expected = claim["execution_contract_fingerprint"]
    if fingerprints != {expected}:
        raise EvidenceValidationError(
            "source manifests do not identify the current execution contract; rerun the study"
        )
    descriptor = _descriptor(manifests, args.recipe)
    successful = [
        run
        for manifest in manifests
        for run in manifest.get("runs", [])
        if run.get("recipe_key") == args.recipe
        and isinstance(run.get("job"), Mapping)
        and run["job"].get("status") == "succeeded"
    ]
    if not successful:
        raise EvidenceValidationError("accepted evidence requires at least one successful API run")
    pipeline_states = {
        run.get("observed_state", {}).get("pipeline_warm") for run in successful
    }
    proven_cold_runs = {
        (reset.get("recipe_key"), reset.get("repeat_index"))
        for manifest in manifests
        for reset in manifest.get("runtime_resets", [])
        if reset.get("cold_precondition_proven") is True
    }
    cold_observed = False in pipeline_states or any(
        run.get("expected_state") == "runtime_cold"
        and (run.get("recipe_key"), run.get("repeat_index")) in proven_cold_runs
        for run in successful
    )
    output_hashes = sorted(
        {
            digest
            for manifest in manifests
            for summary in manifest.get("measurement_summary", [])
            if summary.get("recipe_key") == args.recipe
            for digest in summary.get("artifact_hashes", [])
            if isinstance(digest, str)
        }
    )
    counters: dict[str, int] = {}
    for run in successful:
        _fallback_counters(run.get("job", {}).get("provenance", {}), counters)
    environments = [
        manifest.get("runtime_environment", {})
        for manifest in manifests
        if isinstance(manifest.get("runtime_environment"), Mapping)
    ]
    gpu_names = sorted(
        {
            str(device.get("name"))
            for run in successful
            for device in run.get("gpu_summary", [])
            if device.get("name")
        }
    )
    engine_sources = [manifest.get("engine_source") for manifest in manifests]
    if any(not isinstance(source, Mapping) for source in engine_sources):
        raise EvidenceValidationError("source manifests must identify exact Engine source states")
    commits = {source.get("commit") for source in engine_sources}
    if len(commits) != 1:
        raise EvidenceValidationError("source manifests must share one Engine commit")
    source_states = sorted(
        {
            (bool(source.get("dirty")), source.get("worktree_diff_sha256"))
            for source in engine_sources
        },
        key=lambda value: (value[0], value[1] or ""),
    )
    environment = environments[0] if environments else {}
    record = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "evidence_id": args.evidence_id,
        "recipe_key": args.recipe,
        "execution_contract_fingerprint": expected,
        "tool": {
            "id": descriptor.get("id"),
            "schema_revision": descriptor.get("schema_revision"),
            "schema_hash": descriptor.get("schema_hash"),
        },
        "engine": {
            "commit": next(iter(commits)),
            "source_states": [
                {"dirty": dirty, "worktree_diff_sha256": digest}
                for dirty, digest in source_states
            ],
        },
        "environment": {
            "hardware_class": ", ".join(gpu_names) or "unrecorded",
            "os": environment.get("os"),
            "python": environment.get("python"),
            "torch": environment.get("torch"),
            "cuda": environment.get("cuda"),
            "comfy_kitchen": environment.get("comfy_kitchen"),
        },
        "source_manifests": [source for _manifest, source in loaded],
        "resource_ids": list(claim["resource_ids"]),
        "results": {
            "cold": cold_observed,
            "warm": (
                "not_applicable"
                if claim["declaration"].get("recipe", {}).get("type") == "wan5_kitchen"
                else True in pipeline_states
            ),
            "switching": _lifecycle_result(args.switching),
            "cancellation": _lifecycle_result(args.cancellation),
            "worker_exit_cleanup_recovery": _lifecycle_result(
                args.worker_exit_cleanup_recovery
            ),
        },
        "observations": {
            "fallback_claim": "zero" if counters and all(value == 0 for value in counters.values()) else "unobserved",
            "fallback_counters": counters,
            "output_hashes": output_hashes,
        },
        "creator_review": {
            "state": args.review_state,
            "reviewer": args.reviewer,
            "reviewed_at": args.reviewed_at,
        },
        "known_limitations": args.known_limitation,
    }
    output = root / "evidence" / "acceptance" / f"{args.evidence_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output.relative_to(root).as_posix())
    return 0


def _lifecycle_result(value: str) -> bool | str:
    if value == "passed":
        return True
    if value == "failed":
        return False
    return "not_applicable" if value == "not-applicable" else False


def validate(*, write_matrix: bool) -> int:
    root = repository_root()
    claims = builtin_hardware_claims(root)
    records = load_evidence_records(root / "evidence" / "acceptance")
    validate_evidence_records(records, claims=claims)
    generated = acceptance_matrix(records, claims)
    matrix_path = root / "docs" / "HARDWARE_ACCEPTANCE_MATRIX.md"
    if write_matrix:
        matrix_path.write_text(generated, encoding="utf-8")
    elif not matrix_path.is_file() or matrix_path.read_text(encoding="utf-8") != generated:
        raise EvidenceValidationError(
            "hardware acceptance matrix drifted; run acceptance-evidence.py validate --write-matrix"
        )
    print(f"validated {len(records)} evidence records for {len(claims)} claims")
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.command == "extract":
        return extract(args)
    return validate(write_matrix=args.write_matrix)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceValidationError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"acceptance-evidence: {exc}") from exc
