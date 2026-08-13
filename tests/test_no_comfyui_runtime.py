from __future__ import annotations

import ast
import re
from pathlib import Path

_FORBIDDEN_RUNTIME_MARKERS = (
    "LATENTSLATE_COMFYUI_ROOT",
    "comfyui_root",
    "comfyui_loopback",
    "comfyui_disposable_worker",
    '"/system_stats"',
    '"/object_info"',
    '"/prompt"',
    '"/history/',
    "ComfyUI worker",
    "Comfy worker",
)
_FORBIDDEN_LITERAL_PATTERNS = (
    re.compile(r"comfyui", re.IGNORECASE),
    re.compile(r"/(?:prompt|object_info|history|system_stats|interrupt|view)(?:/|$)", re.IGNORECASE),
    re.compile(r"(?:127\.0\.0\.1|localhost):8188", re.IGNORECASE),
)


def test_engine_has_no_comfyui_execution_backend() -> None:
    """ComfyUI is source evidence; only Comfy Kitchen may execute in Engine."""

    root = Path(__file__).resolve().parents[1]
    runtime_files = [
        *sorted((root / "src" / "latentslate_engine").rglob("*.py")),
        *sorted((root / "scripts").glob("*.py")),
    ]
    violations: list[str] = []
    for path in runtime_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = (node.module,)
            for module in imported:
                if module.casefold().startswith("comfy") and not module.casefold().startswith(
                    "comfy_kitchen"
                ):
                    violations.append(f"{path.relative_to(root)}: forbidden import {module}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for pattern in _FORBIDDEN_LITERAL_PATTERNS:
                    if pattern.search(node.value):
                        violations.append(
                            f"{path.relative_to(root)}:{getattr(node, 'lineno', '?')}: "
                            f"forbidden runtime literal {node.value!r}"
                        )
        for marker in _FORBIDDEN_RUNTIME_MARKERS:
            if marker in source:
                violations.append(f"{path.relative_to(root)}: {marker}")
    assert not violations, "ComfyUI execution backend is forbidden:\n" + "\n".join(violations)


def test_active_docs_do_not_advertise_comfyui_as_an_execution_fallback() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    docs.extend(sorted((root / "docs" / "model-roadmaps").glob("*.md")))
    docs = [path for path in docs if path.name != "COMFY_ENGINE_POLICY.md"]
    forbidden = (
        "user-owned comfy fallback",
        "user-owned comfy workflow",
        "generic comfy fallback",
        "comfyui execution fallback",
        "comfy-backed recipes stage",
    )
    violations = []
    for path in docs:
        source = path.read_text(encoding="utf-8").casefold()
        for marker in forbidden:
            if marker in source:
                violations.append(f"{path.relative_to(root)}: {marker}")
        for pattern in (
            r"(?:user-owned|generic)\s+comfy",
            r"comfy(?:ui)?\s+(?:provider|runtime|server|worker|executor|fallback)",
        ):
            if re.search(pattern, source):
                violations.append(f"{path.relative_to(root)}: /{pattern}/")
    assert not violations, "ComfyUI fallback language is forbidden:\n" + "\n".join(violations)
