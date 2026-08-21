from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest

from latentslate_engine.runtime.framework.worker import (
    WorkerJsonFileError,
    atomic_write_json,
    canonical_json,
    hmac_sha256,
    read_bounded_json,
    result_hmac_sha256,
    sha256_fingerprint,
)


def test_canonical_json_has_exact_stable_bytes_and_rejects_nan():
    value = {"unicode": "café", "nested": {"b": 2, "a": 1}, "enabled": True}
    expected = b'{"enabled":true,"nested":{"a":1,"b":2},"unicode":"caf\\u00e9"}'

    assert canonical_json(value) == expected
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json({"value": float("nan")})


def test_hmac_and_legacy_fingerprint_bind_exact_canonical_bytes():
    value = {"schema_version": 1, "request": {"seed": 42}}
    secret = bytes(range(32))
    encoded = canonical_json(value)

    assert hmac_sha256(value, secret) == hmac.new(
        secret, encoded, hashlib.sha256
    ).hexdigest()
    assert sha256_fingerprint(value) == hashlib.sha256(encoded).hexdigest()
    assert hmac_sha256(value, b"short") == hmac.new(
        b"short", encoded, hashlib.sha256
    ).hexdigest()
    with pytest.raises(TypeError, match="must be bytes"):
        hmac_sha256(value, "not-bytes")  # type: ignore[arg-type]


def test_result_hmac_excludes_only_the_binding_field():
    secret = bytes(range(32))
    result = {
        "schema_version": 1,
        "ok": True,
        "request_binding": "request",
        "metadata": {"count": 3},
        "result_binding": "untrusted",
    }
    expected = dict(result)
    expected.pop("result_binding")

    assert result_hmac_sha256(result, secret) == hmac_sha256(expected, secret)
    changed = {**result, "metadata": {"count": 4}}
    assert result_hmac_sha256(changed, secret) != result_hmac_sha256(result, secret)


def test_atomic_json_publication_matches_canonical_bytes(tmp_path: Path):
    target = tmp_path / "request.json"
    value = {"schema_version": 1, "message": "café"}

    atomic_write_json(target, value)

    assert target.read_bytes() == canonical_json(value)
    assert read_bounded_json(target, maximum_bytes=1024) == value
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("maximum", [False, 0, -1])
def test_bounded_json_rejects_invalid_bounds(tmp_path: Path, maximum: int):
    target = tmp_path / "request.json"
    target.write_text("{}", encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        read_bounded_json(target, maximum_bytes=maximum)


def test_bounded_json_rejects_missing_empty_oversized_and_invalid_files(tmp_path: Path):
    target = tmp_path / "request.json"
    with pytest.raises(WorkerJsonFileError, match="missing"):
        read_bounded_json(target, maximum_bytes=8)

    target.touch()
    with pytest.raises(WorkerJsonFileError, match="empty"):
        read_bounded_json(target, maximum_bytes=8)

    target.write_bytes(b"{}" * 5)
    with pytest.raises(WorkerJsonFileError, match="exceeds"):
        read_bounded_json(target, maximum_bytes=8)

    target.write_bytes(b"not-json")
    with pytest.raises(json.JSONDecodeError):
        read_bounded_json(target, maximum_bytes=8)
