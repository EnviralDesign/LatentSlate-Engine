#!/usr/bin/env python3
"""Run small, opt-in generation studies through the public Engine HTTP API.

This is deliberately not a pytest test or a benchmark framework.  It exercises
the same catalog, asset, job, polling, cancellation, and artifact routes used by
LatentSlate and records enough evidence to reproduce a one-off or small A/B run.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import dotenv_values

TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class StudyError(RuntimeError):
    """An actionable harness or Engine protocol failure."""


class StudyTimeoutError(StudyError):
    """A timed-out job with the best known cancellation/final state."""

    def __init__(
        self,
        message: str,
        *,
        cancellation_response: dict[str, Any] | None = None,
        final_job: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.cancellation_response = cancellation_response
        self.final_job = final_job


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent credentials from following an Engine response to another origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def json_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def key_values(values: list[str], *, parse_json: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key.strip():
            raise StudyError(f"expected KEY=VALUE, got {value!r}")
        key = key.strip()
        if key in result:
            raise StudyError(f"duplicate value for {key!r}")
        result[key] = json_value(raw) if parse_json else raw
    return result


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return cleaned or "result"


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class StudyClient:
    def __init__(self, base_url: str, token: str | None, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.parsed = urllib.parse.urlparse(self.base_url)
        if (
            self.parsed.scheme != "http"
            or self.parsed.hostname not in LOOPBACK_HOSTS
            or self.parsed.username is not None
            or self.parsed.password is not None
            or self.parsed.query
            or self.parsed.fragment
        ):
            raise StudyError("--base-url must be a local HTTP URL such as http://127.0.0.1:8765")
        self.token = token
        self.timeout = timeout
        self.opener = urllib.request.build_opener(NoRedirectHandler)

    def headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def request_json(self, method: str, path: str, payload: Any | None = None) -> Any:
        url = urllib.parse.urljoin(self.base_url + "/", path.lstrip("/"))
        body = None
        headers = self.headers()
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise StudyError(f"{method} {path} returned HTTP {exc.code}: {detail}") from exc
        except OSError as exc:
            raise StudyError(f"{method} {path} failed: {exc}") from exc

    def upload(self, path: Path) -> dict[str, Any]:
        boundary = f"latentslate-study-{uuid4().hex}"
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        upload_name = safe_name(path.name)
        disposition = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{upload_name}"\r\nContent-Type: {content_type}\r\n\r\n'
        ).encode()
        ending = f"\r\n--{boundary}--\r\n".encode("ascii")
        connection = http.client.HTTPConnection(
            self.parsed.hostname,
            self.parsed.port or 80,
            timeout=self.timeout,
        )
        route = (self.parsed.path.rstrip("/") + "/v1/assets") or "/v1/assets"
        try:
            connection.putrequest("POST", route)
            for key, value in self.headers().items():
                connection.putheader(key, value)
            connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
            connection.putheader("Content-Length", str(len(disposition) + path.stat().st_size + len(ending)))
            connection.endheaders()
            connection.send(disposition)
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    connection.send(chunk)
            connection.send(ending)
            response = connection.getresponse()
            body = response.read()
            if response.status != 201:
                raise StudyError(
                    f"POST /v1/assets returned HTTP {response.status}: "
                    f"{body.decode('utf-8', errors='replace')}"
                )
            return json.loads(body)
        except OSError as exc:
            raise StudyError(f"uploading {path} failed: {exc}") from exc
        finally:
            connection.close()

    def download(self, path: str, destination: Path) -> dict[str, Any]:
        url = urllib.parse.urljoin(self.base_url + "/", path.lstrip("/"))
        request = urllib.request.Request(url, headers=self.headers(), method="GET")
        digest = hashlib.sha256()
        size = 0
        try:
            with (
                self.opener.open(request, timeout=self.timeout) as response,
                destination.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        except (OSError, urllib.error.HTTPError) as exc:
            destination.unlink(missing_ok=True)
            raise StudyError(f"downloading {path} failed: {exc}") from exc
        return {"path": str(destination), "size_bytes": size, "sha256": digest.hexdigest()}


class GpuSampler:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
        self._thread.start()

    def close(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 2))
        return self.samples

    def _run(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=max(5.0, self.interval),
                )
                rows = []
                for line in completed.stdout.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) == 5:
                        rows.append(
                            {
                                "index": int(fields[0]),
                                "name": fields[1],
                                "memory_used_mib": int(fields[2]),
                                "memory_total_mib": int(fields[3]),
                                "utilization_gpu_percent": int(fields[4]),
                            }
                        )
                self.samples.append({"at": utc_now(), "devices": rows})
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                self.samples.append({"at": utc_now(), "error": str(exc)})
                return
            self._stop.wait(self.interval)


def descriptor_inputs(descriptor: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["key"]: item for item in descriptor.get("inputs", [])}


def validate_input_value(descriptor_key: str, definition: dict[str, Any], value: Any) -> None:
    key = definition["key"]
    values = value if definition.get("multiple") else [value]
    if definition.get("multiple") and not isinstance(value, list):
        raise StudyError(f"{descriptor_key} input {key!r} must be a list")
    input_type = definition["type"]
    choices = {option["value"] for option in definition.get("options", [])}
    ui = definition.get("ui") or {}
    for item in values:
        valid = True
        if input_type == "text":
            valid = isinstance(item, str)
        elif input_type == "number":
            valid = (
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(item)
            )
        elif input_type == "integer":
            valid = not isinstance(item, bool) and isinstance(item, int)
        elif input_type == "boolean":
            valid = isinstance(item, bool)
        elif input_type == "choice":
            valid = isinstance(item, str) and item in choices
        elif input_type in {"image", "video", "audio"}:
            valid = (
                isinstance(item, dict)
                and item.get("type") == "asset"
                and isinstance(item.get("asset_id"), str)
            )
        elif input_type == "resource":
            valid = isinstance(item, str) and bool(item.strip())
        else:
            raise StudyError(
                f"{descriptor_key} input {key!r} has unsupported type {input_type!r}"
            )
        if not valid:
            expected = f"one of {sorted(choices)}" if input_type == "choice" else input_type
            raise StudyError(f"{descriptor_key} input {key!r} must be {expected}")
        if input_type in {"number", "integer"}:
            if ui.get("min") is not None and item < ui["min"]:
                raise StudyError(f"{descriptor_key} input {key!r} is below {ui['min']}")
            if ui.get("max") is not None and item > ui["max"]:
                raise StudyError(f"{descriptor_key} input {key!r} is above {ui['max']}")


def effective_inputs(
    descriptor: dict[str, Any],
    shared: dict[str, Any],
    assets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    definitions = descriptor_inputs(descriptor)
    supplied = {**shared, **assets}
    unknown = sorted(set(supplied) - set(definitions))
    if unknown:
        raise StudyError(f"{descriptor['key']} does not accept inputs: {', '.join(unknown)}")
    result: dict[str, Any] = {}
    missing: list[str] = []
    for key, definition in definitions.items():
        if key in supplied:
            result[key] = supplied[key]
        elif definition.get("default") is not None:
            result[key] = definition["default"]
        elif definition.get("required"):
            missing.append(key)
    if missing:
        raise StudyError(f"{descriptor['key']} is missing required inputs: {', '.join(missing)}")
    for key, value in result.items():
        validate_input_value(descriptor["key"], definitions[key], value)
    return result


def engine_token() -> str | None:
    token = os.getenv("LATENTSLATE_ENGINE_TOKEN")
    if token:
        return token
    repository_env = Path(__file__).resolve().parents[1] / ".env"
    if repository_env.is_file():
        value = dotenv_values(repository_env).get("LATENTSLATE_ENGINE_TOKEN")
        if isinstance(value, str) and value:
            return value
    return None


def append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def await_job(
    client: StudyClient,
    job_id: str,
    *,
    timeout: float,
    poll_interval: float,
    cancellation_grace: float,
    events_path: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    previous: tuple[Any, Any, Any] | None = None
    while True:
        job = client.request_json("GET", f"/v1/jobs/{job_id}")
        state = (job.get("status"), job.get("progress"), job.get("message"))
        if state != previous:
            append_event(
                events_path,
                {"at": utc_now(), "kind": "job_state", "job_id": job_id, "state": state},
            )
            previous = state
        if job.get("status") in TERMINAL_STATUSES:
            return job
        if time.monotonic() >= deadline:
            try:
                cancellation = client.request_json("DELETE", f"/v1/jobs/{job_id}")
                append_event(
                    events_path,
                    {
                        "at": utc_now(),
                        "kind": "cancellation_requested",
                        "job_id": job_id,
                        "response": cancellation,
                    },
                )
            except StudyError as exc:
                append_event(
                    events_path,
                    {
                        "at": utc_now(),
                        "kind": "cancellation_request_failed",
                        "job_id": job_id,
                        "error": str(exc),
                    },
                )
                raise StudyTimeoutError(
                    f"job {job_id} exceeded {timeout:.1f}s; cancellation is unconfirmed "
                    f"because the request failed: {exc}"
                ) from exc
            cancellation_deadline = time.monotonic() + cancellation_grace
            while time.monotonic() < cancellation_deadline:
                try:
                    canceled_job = client.request_json("GET", f"/v1/jobs/{job_id}")
                except StudyError as exc:
                    raise StudyTimeoutError(
                        f"job {job_id} exceeded {timeout:.1f}s; cancellation is unconfirmed "
                        f"because status polling failed: {exc}",
                        cancellation_response=cancellation,
                    ) from exc
                if canceled_job.get("status") in TERMINAL_STATUSES:
                    append_event(
                        events_path,
                        {
                            "at": utc_now(),
                            "kind": "cancellation_terminal",
                            "job_id": job_id,
                            "status": canceled_job.get("status"),
                            "job": canceled_job,
                        },
                    )
                    status = canceled_job.get("status")
                    outcome = (
                        "cancellation confirmed"
                        if status == "canceled"
                        else f"job reached terminal state {status!r} after cancellation request"
                    )
                    raise StudyTimeoutError(
                        f"job {job_id} exceeded {timeout:.1f}s; {outcome}",
                        cancellation_response=cancellation,
                        final_job=canceled_job,
                    )
                time.sleep(poll_interval)
            raise StudyTimeoutError(
                f"job {job_id} exceeded {timeout:.1f}s; cancellation remains unconfirmed "
                f"after {cancellation_grace:.1f}s",
                cancellation_response=cancellation,
            )
        time.sleep(poll_interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic, opt-in Engine generation studies over HTTP."
    )
    parser.add_argument("--recipe", action="append", required=True, help="Recipe key; repeat in execution order")
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--input", action="append", default=[], metavar="KEY=JSON")
    parser.add_argument("--asset", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=3600.0, help="Per-job timeout in seconds")
    parser.add_argument("--http-timeout", type=float, default=300.0, help="Individual HTTP I/O timeout")
    parser.add_argument("--cancellation-grace", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.timeout <= 0
        or args.http_timeout <= 0
        or args.cancellation_grace <= 0
        or args.poll_interval <= 0
        or args.gpu_sample_interval <= 0
    ):
        raise StudyError("timeout and sampling intervals must be positive")
    run_dir = args.run_dir or Path("hardware-study-runs") / datetime.now(UTC).strftime(
        "%Y%m%d-%H%M%S"
    )
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.jsonl"
    client = StudyClient(args.base_url, engine_token(), args.http_timeout)
    shared_inputs = key_values(args.input, parse_json=True)
    asset_paths = {
        key: Path(value).expanduser().resolve()
        for key, value in key_values(args.asset, parse_json=False).items()
    }
    if args.prompt is not None:
        if "prompt" in shared_inputs:
            raise StudyError("use either --prompt or --input prompt=..., not both")
        shared_inputs["prompt"] = args.prompt
    if args.seed is not None:
        if "seed" in shared_inputs:
            raise StudyError("use either --seed or --input seed=..., not both")
        shared_inputs["seed"] = args.seed
    for key, path in asset_paths.items():
        if not path.is_file():
            raise StudyError(f"asset {key!r} is not a file: {path}")

    health = client.request_json("GET", "/v1/health")
    catalog = client.request_json("GET", "/v1/catalog")
    recipes = client.request_json("GET", "/v1/recipes")
    catalog_by_key = {item["key"]: item for item in catalog.get("tools", [])}
    recipes_by_key = {item["key"]: item for item in recipes.get("recipes", [])}
    selected: list[dict[str, Any]] = []
    if len(set(args.recipe)) != len(args.recipe):
        raise StudyError("duplicate --recipe keys are not allowed")
    for key in args.recipe:
        recipe = recipes_by_key.get(key)
        descriptor = catalog_by_key.get(key)
        if recipe is None or descriptor is None:
            raise StudyError(f"unknown recipe {key!r}")
        if not recipe.get("available") or not descriptor.get("available"):
            reason = recipe.get("unavailable_reason") or descriptor.get("unavailable_reason")
            raise StudyError(f"recipe {key!r} is unavailable: {reason}")
        selected.append({"recipe": recipe, "descriptor": descriptor})

    preflight_asset_values = {
        key: {"type": "asset", "asset_id": "00000000-0000-0000-0000-000000000000"}
        for key in asset_paths
    }
    planned = [
        {
            "recipe_key": item["descriptor"]["key"],
            "inputs": effective_inputs(item["descriptor"], shared_inputs, preflight_asset_values),
        }
        for item in selected
    ]
    manifest: dict[str, Any] = {
        "format": "latentslate-hardware-study-v1",
        "created_at": utc_now(),
        "base_url": args.base_url,
        "health": health,
        "engine_version": catalog.get("engine_version"),
        "protocol_version": catalog.get("protocol_version"),
        "execution_order": "recipe_major",
        "repeat": args.repeat,
        "selected": selected,
        "planned_requests": planned,
        "assets": [],
        "runs": [],
        "notes": [
            "GPU samples are device-wide and may include other processes.",
            "The first run is not guaranteed cold unless the Engine process was freshly started.",
        ],
    }
    atomic_json(run_dir / "manifest.json", manifest)
    if args.preflight_only:
        print(f"Preflight complete: {run_dir / 'manifest.json'}")
        return 0

    uploaded: dict[str, dict[str, Any]] = {}
    for key, path in asset_paths.items():
        response = client.upload(path)
        uploaded[key] = {"type": "asset", "asset_id": response["id"]}
        manifest["assets"].append(
            {"input": key, "source": str(path), "upload": response}
        )
    atomic_json(run_dir / "manifest.json", manifest)

    for selection in selected:
        descriptor = selection["descriptor"]
        inputs = effective_inputs(descriptor, shared_inputs, uploaded)
        for repeat_index in range(1, args.repeat + 1):
            run_index = len(manifest["runs"]) + 1
            label = f"{run_index:02d}-{safe_name(descriptor['key'])}-r{repeat_index}"
            request_payload = {
                "tool_id": descriptor["id"],
                "schema_revision": descriptor["schema_revision"],
                "schema_hash": descriptor["schema_hash"],
                "inputs": inputs,
            }
            record: dict[str, Any] = {
                "index": run_index,
                "label": label,
                "recipe_key": descriptor["key"],
                "repeat_index": repeat_index,
                "request": request_payload,
                "submitted_at": utc_now(),
            }
            sampler = GpuSampler(args.gpu_sample_interval)
            started = time.monotonic()
            current_job_id: str | None = None
            try:
                sampler.start()
                created = client.request_json("POST", "/v1/jobs", request_payload)
                current_job_id = created["id"]
                record["job_id"] = current_job_id
                job = await_job(
                    client,
                    current_job_id,
                    timeout=args.timeout,
                    poll_interval=args.poll_interval,
                    cancellation_grace=args.cancellation_grace,
                    events_path=events_path,
                )
                record["job"] = job
                if job["status"] != "succeeded":
                    raise StudyError(
                        f"job {current_job_id} ended as {job['status']}: {job.get('error')}"
                    )
                downloads = []
                for artifact_index, artifact in enumerate(job.get("artifacts", []), start=1):
                    filename = f"{label}-a{artifact_index}-{safe_name(artifact['filename'])}"
                    downloaded = client.download(
                        artifact["download_url"], run_dir / filename
                    )
                    downloads.append({"artifact": artifact, "download": downloaded})
                record["artifacts"] = downloads
            except KeyboardInterrupt:
                if current_job_id is not None:
                    try:
                        client.request_json("DELETE", f"/v1/jobs/{current_job_id}")
                    except StudyError:
                        pass
                record["error"] = "interrupted; cancellation requested"
                raise
            except StudyError as exc:
                record["error"] = str(exc)
                if isinstance(exc, StudyTimeoutError):
                    record["timeout"] = {
                        "cancellation_response": exc.cancellation_response,
                        "final_job": exc.final_job,
                    }
                print(f"FAILED {label}: {exc}", file=sys.stderr)
            finally:
                record["completed_at"] = utc_now()
                record["client_elapsed_seconds"] = round(time.monotonic() - started, 6)
                record["gpu_samples"] = sampler.close()
                try:
                    record["runtime_after"] = client.request_json("GET", "/v1/runtime")
                except StudyError as exc:
                    record["runtime_after_error"] = str(exc)
                manifest["runs"].append(record)
                atomic_json(run_dir / "manifest.json", manifest)
            if "error" in record:
                return 1
            print(f"Complete {label} ({record['client_elapsed_seconds']:.1f}s)")

    print(f"Study complete: {run_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("hardware-study: interrupted; cancellation requested", file=sys.stderr)
        raise SystemExit(130) from None
    except StudyError as exc:
        print(f"hardware-study: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
