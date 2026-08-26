#!/usr/bin/env python3
"""No-dependency helper used by both bootstrap entrypoints around locked uv sync."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_policy_module():
    """Load the stdlib-only policy without importing Engine's package initializer.

    Selection is deliberately runnable before a project environment exists.
    Importing ``latentslate_engine`` would execute its dotenv/cache setup and
    require third-party dependencies, defeating that fresh-bootstrap boundary.
    """

    module_path = ROOT / "src" / "latentslate_engine" / "runtime_selection.py"
    spec = importlib.util.spec_from_file_location("latentslate_runtime_selection", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load bootstrap policy from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_policy = _load_policy_module()
RuntimeSelectionError = _policy.RuntimeSelectionError
choose_runtime = _policy.choose_runtime
facts_as_dict = _policy.facts_as_dict
probe_hardware = _policy.probe_hardware
selection_payload = _policy.selection_payload
supports_kitchen_capability = _policy.supports_kitchen_capability


def _emit(payload: dict[str, Any], code: int = 0) -> None:
    print(json.dumps(payload, separators=(",", ":")))
    raise SystemExit(code)


def _device_capability_failure(capability: tuple[int, int]) -> dict[str, Any] | None:
    """Return the explicit validation error for an unsupported actual device 0."""

    if supports_kitchen_capability(capability):
        return None
    return {
        "ok": False,
        "error_code": "torch_device_capability_unsupported",
        "message": (
            "PyTorch default CUDA device 0 has compute capability "
            f"{capability[0]}.{capability[1]}; Comfy Kitchen requires SM >= 7.5."
        ),
    }


def select(mode: str) -> None:
    facts = probe_hardware()
    try:
        selection = choose_runtime(mode, facts)  # type: ignore[arg-type]
    except RuntimeSelectionError as exc:
        _emit({"ok": False, "error_code": "selection_unavailable", "message": str(exc)}, 2)
    payload = selection_payload(selection, facts)
    payload["ok"] = True
    payload["facts"] = facts_as_dict(facts)
    _emit(payload)


def validate(tier: str) -> None:
    if tier == "protocol":
        _emit(
            {
                "ok": True,
                "tier": tier,
                "torch": None,
                "kitchen": {"state": "not-installed", "backends": [], "capabilities": {}},
            }
        )
    if importlib.util.find_spec("torch") is None:
        _emit({"ok": False, "error_code": "torch_import_failed", "message": "PyTorch is absent."}, 3)
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - this is an installer boundary
        _emit(
            {
                "ok": False,
                "error_code": "torch_import_failed",
                "message": f"PyTorch import failed: {exc}",
            },
            3,
        )

    compiled_cuda = getattr(torch.version, "cuda", None)
    if not torch.cuda.is_available():
        _emit(
            {
                "ok": False,
                "error_code": "torch_cuda_unavailable",
                "message": "PyTorch did not report a CUDA device.",
                "torch": {"version": torch.__version__, "compiled_cuda": compiled_cuda},
            },
            3,
        )
    capability = torch.cuda.get_device_capability(0)
    if failure := _device_capability_failure(capability):
        _emit(failure, 3)
    if tier == "nvidia-cu130" and not str(compiled_cuda or "").startswith("13."):
        _emit(
            {
                "ok": False,
                "error_code": "torch_cuda_mismatch",
                "message": f"Expected a CUDA 13 PyTorch wheel; detected CUDA {compiled_cuda!r}.",
            },
            3,
        )
    if tier == "nvidia-cu128" and not str(compiled_cuda or "").startswith("12.8"):
        _emit(
            {
                "ok": False,
                "error_code": "torch_cuda_mismatch",
                "message": f"Expected a CUDA 12.8 PyTorch wheel; detected CUDA {compiled_cuda!r}.",
            },
            3,
        )
    try:
        aimdo_version = importlib.metadata.version("comfy-aimdo")
        if aimdo_version != "0.4.15":
            _emit(
                {
                    "ok": False,
                    "error_code": "aimdo_version_mismatch",
                    "message": f"Expected comfy-aimdo 0.4.15; detected {aimdo_version!r}.",
                },
                3,
            )
    except importlib.metadata.PackageNotFoundError:
        _emit(
            {
                "ok": False,
                "error_code": "aimdo_metadata_missing",
                "message": "Standalone comfy-aimdo 0.4.15 is absent from the NVIDIA tier.",
            },
            3,
        )
    try:
        import comfy_kitchen as kitchen
        from comfy_kitchen.tensor import (
            TensorCoreFP8Layout,
            TensorCoreMXFP8Layout,
            TensorCoreNVFP4Layout,
        )

        reported_backends = kitchen.list_backends()
        if isinstance(reported_backends, dict):
            backends = sorted(str(name) for name, value in reported_backends.items() if value)
        else:
            backends = sorted(str(name) for name in reported_backends)
        cuda_backend = "cuda" in {backend.lower() for backend in backends}
        if tier == "nvidia-cu130" and not cuda_backend:
            _emit(
                {
                    "ok": False,
                    "error_code": "kitchen_cuda_backend_unavailable",
                    "message": "Comfy Kitchen did not report its CUDA backend for the CUDA 13 tier.",
                },
                3,
            )
        kitchen_payload = {
            "state": "ready",
            "backends": backends,
            "cuda_backend": cuda_backend,
            "capabilities": {
                "fp8": bool(cuda_backend and capability >= (8, 9) and TensorCoreFP8Layout),
                "nvfp4": bool(cuda_backend and capability >= (10, 0) and TensorCoreNVFP4Layout),
                "mxfp8": bool(cuda_backend and capability >= (10, 0) and TensorCoreMXFP8Layout),
            },
        }
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - validation classifies this narrowly
        _emit(
            {
                "ok": False,
                "error_code": "kitchen_import_failed",
                "message": f"Comfy Kitchen import/probe failed: {exc}",
            },
            3,
        )
    _emit(
        {
            "ok": True,
            "tier": tier,
            "torch": {
                "version": torch.__version__,
                "compiled_cuda": compiled_cuda,
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
            },
            "kitchen": kitchen_payload,
            "aimdo": {
                "state": "installed-not-initialized",
                "version": aimdo_version,
                "scope": "persistent-gpu-child-only",
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    command = parser.add_subparsers(dest="command", required=True)
    select_parser = command.add_parser("select")
    select_parser.add_argument("--mode", choices=("auto", "cu130", "cu128", "protocol"), default="auto")
    validate_parser = command.add_parser("validate")
    validate_parser.add_argument("--tier", choices=("nvidia-cu130", "nvidia-cu128", "protocol"), required=True)
    args = parser.parse_args()
    if args.command == "select":
        select(args.mode)
    validate(args.tier)


if __name__ == "__main__":
    main()
