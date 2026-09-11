"""Portable artifact identities and the Hugging Face acquisition boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

import httpx

SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
REPO_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}\Z")


def validate_repo(repo: object) -> str:
    """Accept model repository IDs, never URLs or filesystem paths."""
    if not isinstance(repo, str):
        raise TypeError("Hugging Face repo must be owner/model")
    parts = repo.split("/")
    if len(parts) != 2 or any(
        not REPO_PART.fullmatch(part)
        or ".." in part
        or "--" in part
        or part.endswith((".", ".git"))
        for part in parts
    ):
        raise ValueError("Hugging Face repo must be owner/model")
    return repo


def validate_file(filename: object) -> str:
    """Keep a repository-relative POSIX file name portable and unambiguous."""
    if (
        not isinstance(filename, str)
        or not filename
        or any(ord(c) < 32 for c in filename)
        or any(c in filename for c in "\\:?#")
        or any(part in {"", ".", ".."} for part in filename.split("/"))
    ):
        raise ValueError("Hugging Face file must be a relative repository file path")
    return filename


def validate_reference(value: object) -> dict:
    """Validate identity without network access, host paths, or credentials."""
    if not isinstance(value, dict):
        raise TypeError("Expected an artifact reference")
    if value.get("source") == "local":
        if (
            set(value) != {"source", "path"}
            or not isinstance(value["path"], str)
            or not value["path"]
            or "\x00" in value["path"]
        ):
            raise ValueError("Expected a local artifact reference with source and path")
    elif value.get("source") == "huggingface":
        if set(value) != {"source", "repo", "revision", "file", "sha256"}:
            raise ValueError(
                "Hugging Face reference requires repo, revision, file and sha256 only"
            )
        validate_repo(value["repo"])
        validate_file(value["file"])
        if not isinstance(value["revision"], str) or not COMMIT.fullmatch(
            value["revision"]
        ):
            raise ValueError(
                "Hugging Face revision must be an immutable 40-character commit SHA"
            )
        if not isinstance(value["sha256"], str) or not SHA256.fullmatch(
            value["sha256"]
        ):
            raise ValueError(
                "Artifact sha256 must be 64 lowercase hexadecimal characters"
            )
    else:
        raise ValueError("Artifact source must be local or huggingface")
    return value


def hf_locator(value: object) -> dict:
    """Normalize friendly file URLs or repo/file/revision input before pinning."""
    if not isinstance(value, dict):
        raise TypeError("Provide a Hugging Face file URL or repo, file and revision")
    if set(value) == {"url"} and isinstance(value["url"], str):
        url = urlsplit(value["url"].strip())
        parts = url.path.strip("/").split("/")
        if (
            url.scheme != "https"
            or url.netloc != "huggingface.co"
            or len(parts) < 5
            or parts[2] not in {"blob", "resolve"}
        ):
            raise ValueError(
                "Use a https://huggingface.co/owner/model/blob/revision/file URL"
            )
        value = {
            "repo": "/".join(parts[:2]),
            "revision": unquote(parts[3]),
            "file": unquote("/".join(parts[4:])),
        }
    elif not {"repo", "file"} <= value.keys() or value.keys() - {
        "repo",
        "file",
        "revision",
    }:
        raise ValueError("Provide repo, file and optional revision, or one file URL")
    repo, filename = validate_repo(value.get("repo")), validate_file(value.get("file"))
    revision = value.get("revision", "main")
    if (
        not isinstance(revision, str)
        or not revision.strip()
        or len(revision) > 200
        or any(ord(c) < 33 for c in revision)
        or any(c in revision for c in "\\?#")
        or any(part in {"", ".", ".."} for part in revision.split("/"))
    ):
        raise ValueError("Provide a non-empty Hugging Face revision")
    return {"repo": repo, "revision": revision, "file": filename}


@dataclass(frozen=True)
class HfFile:
    repo: str
    revision: str
    file: str
    sha256: str | None
    size: int | None
    location: str

    def reference(self, digest: str) -> dict:
        return validate_reference(
            {
                "source": "huggingface",
                "repo": self.repo,
                "revision": self.revision,
                "file": self.file,
                "sha256": digest,
            }
        )


class HuggingFaceSource:
    """Official Hub metadata; bounded streaming into a caller-owned sink."""

    @staticmethod
    def authentication_configured() -> bool:
        try:
            from huggingface_hub.utils import build_hf_headers

            return "authorization" in build_hf_headers()
        except ImportError:
            return False

    def describe(self, locator: dict) -> HfFile:
        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url
        except ImportError:
            raise ValueError(
                "Hugging Face support requires huggingface_hub; install the documented Engine dependency"
            ) from None
        try:
            metadata = get_hf_file_metadata(
                hf_hub_url(
                    locator["repo"],
                    locator["file"],
                    revision=locator["revision"],
                    endpoint="https://huggingface.co",
                ),
                timeout=10,
            )
        except (httpx.HTTPError, OSError, ValueError) as error:
            # Hub errors can contain authenticated/signed URLs. Never return them.
            status = getattr(getattr(error, "response", None), "status_code", None)
            raise ValueError(
                f"Hugging Face metadata request failed{f' (HTTP {status})' if status else ''}. Check the source and host authentication."
            ) from None
        if not metadata.commit_hash or not COMMIT.fullmatch(metadata.commit_hash):
            raise ValueError("Hugging Face did not return an immutable commit")
        if (
            COMMIT.fullmatch(locator["revision"])
            and metadata.commit_hash != locator["revision"]
        ):
            raise ValueError("Hugging Face returned a different commit than requested")
        if not COMMIT.fullmatch(locator["revision"]):
            # Re-read at the immutable commit so a branch move cannot change the
            # bytes fetched between friendly resolution and the actual download.
            return self.describe({**locator, "revision": metadata.commit_hash})
        # The official Hub reports the content SHA-256 as the ETag for LFS files;
        # ordinary Git blob ETags are SHA-1 and cannot stand in for SHA-256.
        digest = (
            metadata.etag if metadata.etag and SHA256.fullmatch(metadata.etag) else None
        )
        return HfFile(
            locator["repo"],
            metadata.commit_hash,
            locator["file"],
            digest,
            metadata.size,
            metadata.location,
        )

    def download(self, file: HfFile, write, cancel) -> None:
        # Only Hub-origin requests receive HF credentials. httpx also strips
        # authorization on redirects to a different origin (e.g. signed CDNs).
        headers = {}
        if urlsplit(file.location).netloc == "huggingface.co":
            from huggingface_hub.utils import build_hf_headers

            headers = build_hf_headers()
        headers["Accept-Encoding"] = "identity"
        try:
            with httpx.stream(
                "GET", file.location, headers=headers, follow_redirects=True, timeout=10
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(1024 * 1024):
                    cancel()
                    write(chunk)
        except httpx.HTTPError as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            raise ValueError(
                f"Hugging Face download failed{f' (HTTP {status})' if status else ''}. Retry or check host authentication."
            ) from None
