"""Exact Civitai version/file selection; download state never enters recipes."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

import httpx

from .artifact_sources import SHA256, positive_id, validate_reference


def civitai_locator(value: object, *, pin: bool = False) -> dict:
    """Accept a version ID or an explicit modelVersionId model-page URL."""
    if not isinstance(value, dict):
        raise TypeError("Provide a Civitai model version ID or model-page URL")
    if pin:
        if set(value) != {"model_version_id", "file_id"}:
            raise ValueError("Pin requires an exact model_version_id and file_id")
        return {key: positive_id(value[key]) for key in ("model_version_id", "file_id")}
    if set(value) == {"model_version_id"}:
        return {"model_version_id": positive_id(value["model_version_id"])}
    if set(value) != {"url"} or not isinstance(value["url"], str):
        raise ValueError("Provide a Civitai model version ID or model-page URL")
    url = urlsplit(value["url"].strip())
    versions = parse_qs(url.query).get("modelVersionId", [])
    if (
        url.scheme != "https"
        or url.netloc not in {"civitai.com", "www.civitai.com"}
        or not re.fullmatch(r"/models/[1-9][0-9]*(?:/[^/]+)?/?", url.path)
        or len(versions) != 1
        or not re.fullmatch(r"[1-9][0-9]*", versions[0])
    ):
        raise ValueError("Use a Civitai model-page URL containing modelVersionId")
    return {"model_version_id": positive_id(int(versions[0]))}


def _failure(error: httpx.HTTPError, operation: str) -> ValueError:
    status = getattr(getattr(error, "response", None), "status_code", None)
    advice = (
        "Host authentication is required or lacks access; check CIVITAI_TOKEN."
        if status in {401, 403}
        else "Check the version/file and host authentication, then retry."
    )
    return ValueError(
        f"Civitai {operation} failed{f' (HTTP {status})' if status else ''}. {advice}"
    )


def _digest(file: dict) -> str | None:
    digest = file.get("hashes", {}).get("SHA256")
    if digest is None:
        return None
    if not isinstance(digest, str) or not SHA256.fullmatch(digest.lower()):
        raise ValueError("Civitai returned invalid SHA-256 metadata")
    return digest.lower()


def _text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(c for c in value if ord(c) >= 32)[:500]


@dataclass(frozen=True)
class CivitaiFile:
    model_version_id: int
    file_id: int
    sha256: str | None
    location: str
    # sizeKB is rounded display metadata, not an exact byte-length guarantee.
    size: None = None

    def reference(self, digest: str) -> dict:
        return validate_reference(
            {
                "source": "civitai",
                "model_version_id": self.model_version_id,
                "file_id": self.file_id,
                "sha256": digest,
            }
        )


class CivitaiSource:
    """Civitai metadata and streaming with host-only Bearer authentication."""

    @staticmethod
    def authentication_configured() -> bool:
        return bool(os.environ.get("CIVITAI_TOKEN", "").strip())

    @staticmethod
    def _headers(url: str) -> dict:
        token = os.environ.get("CIVITAI_TOKEN", "").strip()
        return (
            {"Authorization": "Bearer " + token}
            if token and urlsplit(url).netloc == "civitai.com"
            else {}
        )

    def _version(self, version_id: int) -> dict:
        url = f"https://civitai.com/api/v1/model-versions/{positive_id(version_id)}"
        try:
            response = httpx.get(
                url, headers=self._headers(url), timeout=15, follow_redirects=True
            )
            response.raise_for_status()
            value = response.json()
        except httpx.HTTPError as error:
            raise _failure(error, "metadata request") from None
        except ValueError:
            raise ValueError("Civitai returned invalid version metadata") from None
        if (
            not isinstance(value, dict)
            or type(value.get("id")) is not int
            or value["id"] != version_id
            or not isinstance(value.get("files"), list)
        ):
            raise ValueError("Civitai did not return the requested version and files")
        return value

    def inspect(self, locator: dict) -> dict:
        version = self._version(locator["model_version_id"])
        files = []
        for file in version["files"]:
            size = file.get("sizeKB")
            size_bytes = (
                round(size * 1024)
                if type(size) in {int, float} and math.isfinite(size) and size >= 0
                else None
            )
            files.append(
                {
                    "file_id": positive_id(file.get("id")),
                    "name": _text(file.get("name")),
                    "size_bytes_estimate": size_bytes,
                    "type": _text(file.get("type")),
                    "format": _text(file.get("metadata", {}).get("format")),
                    "primary": file.get("primary") is True,
                    "sha256": _digest(file),
                }
            )
        return {
            "model_version_id": version["id"],
            "model_name": _text(version.get("model", {}).get("name")),
            "version_name": _text(version.get("name")),
            "files": files,
        }

    def describe(self, locator: dict) -> CivitaiFile:
        version = self._version(locator["model_version_id"])
        matches = [
            file for file in version["files"] if file.get("id") == locator["file_id"]
        ]
        if len(matches) != 1:
            raise ValueError(
                "Selected Civitai file is not present in the requested version"
            )
        file = matches[0]
        location = file.get("downloadUrl")
        if not isinstance(location, str) or not location:
            raise ValueError("Selected Civitai file has no download URL")
        url = urlsplit(location)
        if (
            url.scheme != "https"
            or not url.netloc
            or url.username
            or url.password
            or any(key.lower() == "token" for key in parse_qs(url.query))
        ):
            raise ValueError("Civitai returned an unsupported file download URL")
        return CivitaiFile(
            version["id"], positive_id(file["id"]), _digest(file), location
        )

    def download(self, file: CivitaiFile, write, cancel) -> None:
        headers = {**self._headers(file.location), "Accept-Encoding": "identity"}
        try:
            # httpx removes Authorization on redirects to unrelated origins.
            with httpx.stream(
                "GET", file.location, headers=headers, follow_redirects=True, timeout=15
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(1024 * 1024):
                    cancel()
                    write(chunk)
        except httpx.HTTPError as error:
            raise _failure(error, "download") from None
