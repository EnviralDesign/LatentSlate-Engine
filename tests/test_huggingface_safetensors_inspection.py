from __future__ import annotations

import json
import struct
from types import SimpleNamespace
from typing import Any

import pytest

from latentslate_engine.authoring import inspection_huggingface as huggingface
from latentslate_engine.authoring.inspection_artifacts import _parse_safetensors_bytes
from latentslate_engine.authoring.models import ResourceInspectRequest


def _header(*, lora: bool) -> bytes:
    payload: dict[str, Any] = {"__metadata__": {"ss_base_model_version": "flux2_klein_9b"}}
    if lora:
        for index in range(112):
            payload[f"transformer.blocks.{index}.lora_A.weight"] = {
                "dtype": "BF16",
                "shape": [8, 8],
                "data_offsets": [0, 128],
            }
            payload[f"transformer.blocks.{index}.lora_B.weight"] = {
                "dtype": "BF16",
                "shape": [8, 8],
                "data_offsets": [128, 256],
            }
    else:
        payload["transformer.blocks.0.weight"] = {
            "dtype": "BF16",
            "shape": [8, 8],
            "data_offsets": [0, 128],
        }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


class _Api:
    def model_info(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            sha="bd49995e" + "0" * 32,
            siblings=[
                SimpleNamespace(
                    rfilename="70sSciFiKlein9b.safetensors",
                    size=82_866_712,
                    lfs={"sha256": "a" * 64, "size": 82_866_712},
                )
            ],
        )


def test_huggingface_exact_safetensors_header_populates_lora_recommendations() -> None:
    calls: list[tuple[str, str | None]] = []

    def probe(url: str, token: str | None) -> tuple[bytes | None, str | None]:
        calls.append((url, token))
        return _header(lora=True), None

    result = huggingface.inspect_huggingface(
        ResourceInspectRequest(source="hf://Kutches/Kl4b/70sSciFiKlein9b.safetensors"),
        hf_api_factory=lambda **_kwargs: _Api(),
        probe_safetensors_header=probe,
    )

    assert result.facts.size_bytes == 82_866_712
    assert result.facts.sha256 == "a" * 64
    assert result.exact_source is not None
    assert result.exact_source.repo_id == "Kutches/Kl4b"
    assert result.exact_source.revision == "bd49995e" + "0" * 32
    assert result.exact_source.filename == "70sSciFiKlein9b.safetensors"
    assert result.exact_source.sha256 == "a" * 64
    assert result.facts.safetensors is not None
    assert result.facts.safetensors.tensor_count == 224
    assert result.facts.safetensors.truncated is False
    assert all(
        key.endswith((".lora_A.weight", ".lora_B.weight"))
        for key in result.facts.safetensors.tensor_keys
    )
    assert result.facts.safetensors.metadata["ss_base_model_version"] == "flux2_klein_9b"
    assert result.detected["lora_tensor_pairs"] is True
    assert result.detected["base_model"] == "black-forest-labs/FLUX.2-klein-9B"
    assert result.recommended["component"] == "lora"
    assert result.recommended["family"] == "klein9b"
    assert result.recommended["base_model"] == "black-forest-labs/FLUX.2-klein-9B"
    assert calls[0][0].endswith(
        "/Kutches/Kl4b/resolve/bd49995e00000000000000000000000000000000/70sSciFiKlein9b.safetensors"
    )


def test_huggingface_header_probe_rejects_malformed_and_oversize_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        huggingface,
        "_read_huggingface_range",
        lambda *_args: (b"short", None),
    )
    assert huggingface._probe_safetensors_header("https://huggingface.co/a", None)[0] is None

    monkeypatch.setattr(
        huggingface,
        "_read_huggingface_range",
        lambda *_args: (struct.pack("<Q", 8 * 1024 * 1024 + 1), None),
    )
    raw, warning = huggingface._probe_safetensors_header("https://huggingface.co/a", None)
    assert raw is None
    assert warning is not None and "exceeds" in warning


def test_huggingface_header_probe_reads_only_declared_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    header = _header(lora=True)
    ranges: list[str] = []

    def read_range(_url: str, _token: str | None, byte_range: str) -> tuple[bytes, None]:
        ranges.append(byte_range)
        return (header[:8] if byte_range == "bytes=0-7" else header), None

    monkeypatch.setattr(huggingface, "_read_huggingface_range", read_range)
    raw, warning = huggingface._probe_safetensors_header("https://huggingface.co/a", None)
    assert raw == header
    assert warning is None
    assert ranges == ["bytes=0-7", f"bytes=0-{len(header) - 1}"]


@pytest.mark.parametrize(
    "entry",
    [
        {"dtype": "BF16", "shape": ["invalid"], "data_offsets": [0, 2]},
        {"dtype": 16, "shape": [1], "data_offsets": [0, 2]},
        {"dtype": "BF16", "shape": [1], "data_offsets": [2, 0]},
    ],
)
def test_safetensors_parser_fails_soft_for_malformed_tensor_entries(
    entry: dict[str, object],
) -> None:
    encoded = json.dumps({"weight": entry}, separators=(",", ":")).encode("utf-8")
    assert _parse_safetensors_bytes(struct.pack("<Q", len(encoded)) + encoded) is None


def test_huggingface_non_lora_header_remains_ambiguous() -> None:
    result = huggingface.inspect_huggingface(
        ResourceInspectRequest(source="hf://Kutches/Kl4b/70sSciFiKlein9b.safetensors"),
        hf_api_factory=lambda **_kwargs: _Api(),
        probe_safetensors_header=lambda *_args: (_header(lora=False), None),
    )
    assert "component" not in result.recommended
    assert "lora_tensor_pairs" not in result.detected
    assert result.recommended["family"] == "klein9b"


class _Response:
    def __init__(self, status: int, headers: dict[str, str], body: bytes = b"header") -> None:
        self._status = status
        self.headers = headers
        self._body = body

    def getcode(self) -> int:
        return self._status

    def read(self, size: int) -> bytes:
        return self._body[:size]

    def close(self) -> None:
        pass


def test_huggingface_header_redirect_strips_token_from_cdn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, str]] = []
    responses = iter(
        [
            _Response(302, {"Location": "https://cdn.example.invalid/header"}),
            _Response(206, {}),
        ]
    )

    def open_request(request: Any, *, timeout: int) -> _Response:
        requests.append({key.casefold(): value for key, value in request.header_items()})
        return next(responses)

    monkeypatch.setattr(huggingface, "_hf_urlopen", open_request)
    raw, warning = huggingface._read_huggingface_range(
        "https://huggingface.co/example/model/resolve/rev/model.safetensors",
        "secret",
        "bytes=0-7",
    )
    assert raw == b"header"
    assert warning is None
    assert requests[0]["authorization"] == "Bearer secret"
    assert "authorization" not in requests[1]
