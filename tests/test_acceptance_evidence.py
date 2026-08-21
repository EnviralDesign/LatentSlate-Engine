from __future__ import annotations

from copy import deepcopy

import pytest

from latentslate_engine.acceptance_evidence import (
    EvidenceValidationError,
    acceptance_matrix,
    builtin_hardware_claims,
    validate_evidence_records,
)


def _record(recipe_key: str, claim: dict, *, evidence_id: str) -> dict:
    return {
        "schema_version": 1,
        "evidence_id": evidence_id,
        "recipe_key": recipe_key,
        "execution_contract_fingerprint": claim["execution_contract_fingerprint"],
        "tool": {
            "id": "00000000-0000-0000-0000-000000000001",
            "schema_revision": 1,
            "schema_hash": "sha256:" + "1" * 64,
        },
        "engine": {
            "commit": "1" * 40,
            "source_states": [
                {"dirty": False, "worktree_diff_sha256": None},
            ],
        },
        "environment": {
            "hardware_class": "test-gpu",
            "os": "test-os",
            "python": "3.12",
            "torch": "test",
            "cuda": "test",
            "comfy_kitchen": "test",
        },
        "source_manifests": [
            {
                "sha256": "2" * 64,
                "retained_class": "local-ignored",
                "retained_location": f"hardware-study-runs/{evidence_id}/manifest.json",
            }
        ],
        "resource_ids": list(claim["resource_ids"]),
        "results": {
            "cold": True,
            "warm": True,
            "switching": True,
            "cancellation": True,
            "worker_exit_cleanup_recovery": True,
        },
        "observations": {
            "fallback_claim": "zero",
            "fallback_counters": {"dense_fallback_count": 0},
            "output_hashes": ["3" * 64],
        },
        "creator_review": {
            "state": "accepted",
            "reviewer": "creator",
            "reviewed_at": "2026-08-20",
        },
        "known_limitations": [],
    }


def test_every_hardware_claim_requires_current_accepted_evidence():
    claims = builtin_hardware_claims()
    records = [
        _record(key, claim, evidence_id=f"test-{index:02d}")
        for index, (key, claim) in enumerate(sorted(claims.items()), start=1)
    ]

    validate_evidence_records(records, claims=claims)

    stale = deepcopy(records)
    stale[0]["execution_contract_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(EvidenceValidationError, match="lack matching accepted evidence"):
        validate_evidence_records(stale, claims=claims)


def test_evidence_rejects_duplicate_ids_private_fields_and_unobserved_zero_fallback():
    recipe_key, claim = next(iter(sorted(builtin_hardware_claims().items())))
    base = _record(recipe_key, claim, evidence_id="duplicate")

    with pytest.raises(EvidenceValidationError, match="duplicate evidence ID"):
        validate_evidence_records((base, deepcopy(base)), claims={recipe_key: claim})

    private = deepcopy(base)
    private["environment"]["token"] = "private"
    with pytest.raises(EvidenceValidationError, match="prohibited private evidence field"):
        validate_evidence_records((private,), claims={recipe_key: claim})

    unobserved = deepcopy(base)
    unobserved["observations"]["fallback_counters"] = {}
    with pytest.raises(EvidenceValidationError, match="zero fallback lacks observed"):
        validate_evidence_records((unobserved,), claims={recipe_key: claim})


def test_acceptance_matrix_is_deterministic_and_marks_missing_claims():
    claims = builtin_hardware_claims()
    recipe_key, claim = next(iter(sorted(claims.items())))
    matrix = acceptance_matrix(
        (_record(recipe_key, claim, evidence_id="one-record"),), claims
    )

    assert f"| `{recipe_key}` | `one-record` |" in matrix
    assert "`MISSING`" in matrix
