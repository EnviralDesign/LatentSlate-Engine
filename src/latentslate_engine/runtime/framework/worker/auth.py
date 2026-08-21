"""Canonical envelope encoding and fixed authentication primitives."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from typing import Any


def canonical_json(value: Mapping[str, Any]) -> bytes:
    """Encode one bounded protocol object deterministically.

    Protocol envelopes deliberately reject non-finite floats: JSON spellings for
    them are non-standard and do not belong in an authenticated request.
    """

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def hmac_sha256(value: Mapping[str, Any], secret: bytes) -> str:
    """Authenticate the canonical bytes of ``value`` with HMAC-SHA256."""

    if not isinstance(secret, bytes):
        raise TypeError("worker protocol secret must be bytes")
    return hmac.new(secret, canonical_json(value), hashlib.sha256).hexdigest()


def result_hmac_sha256(
    value: Mapping[str, Any],
    secret: bytes,
    *,
    binding_field: str = "result_binding",
) -> str:
    """Authenticate a result envelope without its self-referential binding."""

    if not binding_field or not isinstance(binding_field, str):
        raise ValueError("worker result binding field must be non-empty")
    return hmac_sha256(
        {key: item for key, item in value.items() if key != binding_field},
        secret,
    )


def sha256_fingerprint(value: Mapping[str, Any]) -> str:
    """Fingerprint a canonical envelope without claiming authentication."""

    return hashlib.sha256(canonical_json(value)).hexdigest()
