from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src" / "latentslate_engine" / "runtime"
FRAMEWORK = RUNTIME / "framework"
Z_QWEN_ARCHITECTURE = RUNTIME / "z_image_qwen_architecture.py"
Z_QWEN_CHECKPOINT = RUNTIME / "z_image_qwen_checkpoint.py"
Z_QWEN_RUNTIME = RUNTIME / "z_image_qwen_runtime.py"
Z_IMAGE_RECIPE = ROOT / "src" / "latentslate_engine" / "z_image_turbo_recipe.py"
SHARED_RUNTIME_PATHS = (
    ROOT / "src" / "latentslate_engine" / "stored_quant.py",
    RUNTIME / "cache.py",
    RUNTIME / "manager.py",
    RUNTIME / "process_memory.py",
    RUNTIME / "residency_policy.py",
    RUNTIME / "windows_process.py",
)

MODEL_MARKERS = ("klein", "ltx", "wan", "z_image", "zimage")

# These are Phase 0 debt, not approved architecture. Entries identify the
# migration that deletes them, and the test prevents the list from growing.
GRANDFATHERED_PRIVATE_IMPORTS: set[tuple[str, str, str, str]] = set()

# Public names can still violate the dependency direction when they expose a
# model-named implementation. Keep this ledger exact and shrinking until the
# shared stored-execution tranche removes the remaining compatibility imports.
GRANDFATHERED_MODEL_NAMED_IMPORTS: set[tuple[str, str, str, str]] = set()


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _relative_module(node: ast.ImportFrom) -> str:
    return node.module or ""


def _model_family(module_or_file: str) -> str | None:
    stem = Path(module_or_file).stem
    return next((marker for marker in MODEL_MARKERS if stem.startswith(marker)), None)


def test_private_cross_family_import_debt_is_fixed_and_cannot_grow():
    observed: set[tuple[str, str, str, str]] = set()
    phases = {
        (file_name, module, symbol): phase
        for file_name, module, symbol, phase in GRANDFATHERED_PRIVATE_IMPORTS
    }
    for path in sorted(RUNTIME.glob("*.py")):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            module = _relative_module(node)
            for alias in node.names:
                if not alias.name.startswith("_"):
                    continue
                key = (path.name, module, alias.name)
                if key in phases:
                    observed.add((*key, phases[key]))
                    continue
                source_family = next(
                    (marker for marker in MODEL_MARKERS if path.stem.startswith(marker)),
                    None,
                )
                target_family = next(
                    (marker for marker in MODEL_MARKERS if module.startswith(marker)),
                    None,
                )
                if source_family != target_family and target_family is not None:
                    raise AssertionError(
                        f"new private cross-family import: {path.name} -> "
                        f"{module}.{alias.name}"
                    )
    assert observed == GRANDFATHERED_PRIVATE_IMPORTS


def test_model_named_cross_family_import_debt_is_exact_and_cannot_grow():
    observed: set[tuple[str, str, str, str]] = set()
    phases = {
        (file_name, module, symbol): phase
        for file_name, module, symbol, phase in GRANDFATHERED_MODEL_NAMED_IMPORTS
    }
    for path in sorted(RUNTIME.glob("*.py")):
        source_family = _model_family(path.name)
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            if node.module is None:
                imports = ((alias.name, "<module>") for alias in node.names)
            else:
                imports = ((node.module, alias.name) for alias in node.names)
            for module, symbol in imports:
                target_family = _model_family(module)
                if target_family is None or target_family == source_family:
                    continue
                key = (path.name, module, symbol)
                if key in phases:
                    observed.add((*key, phases[key]))
                    continue
                raise AssertionError(
                    f"new model-named cross-family import: {path.name} -> "
                    f"{module}.{symbol}"
                )
    assert observed == GRANDFATHERED_MODEL_NAMED_IMPORTS


def test_framework_is_model_neutral_and_not_a_dynamic_loader():
    if not FRAMEWORK.exists():
        return
    for path in sorted(FRAMEWORK.rglob("*.py")):
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = (node.module or "").lower()
                assert not any(marker in module for marker in MODEL_MARKERS), (
                    f"framework imports model family in {path}: {module}"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.lower()
                    assert not any(marker in module for marker in MODEL_MARKERS), (
                        f"framework imports model family in {path}: {module}"
                    )
            elif isinstance(node, ast.Call):
                name = _call_name(node.func)
                assert name not in {
                    "importlib.import_module",
                    "importlib.util.spec_from_file_location",
                    "runpy.run_module",
                    "runpy.run_path",
                }, f"framework contains a dynamic Python loader in {path}: {name}"
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.lower()
                assert not any(marker in value for marker in MODEL_MARKERS), (
                    f"framework contains model-named protocol/policy text in {path}"
                )


def test_existing_shared_runtime_capabilities_remain_model_neutral():
    for path in SHARED_RUNTIME_PATHS:
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.lower()
                assert not any(marker in value for marker in MODEL_MARKERS), (
                    f"shared runtime capability contains model-specific text in {path}"
                )


def test_z_image_qwen_split_has_one_way_responsibility_edges_and_no_facade():
    assert not (RUNTIME / "z_image_mixed_qwen.py").exists()
    architecture = _tree(Z_QWEN_ARCHITECTURE)
    checkpoint = _tree(Z_QWEN_CHECKPOINT)
    runtime = _tree(Z_QWEN_RUNTIME)
    recipe = _tree(Z_IMAGE_RECIPE)

    architecture_imports = {
        node.module or ""
        for node in ast.walk(architecture)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(architecture)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert all(
        module == "__future__"
        or module in {"collections.abc", "contextlib", "typing"}
        or module == "torch"
        or module.startswith("torch.")
        for module in architecture_imports
    )

    checkpoint_imports = {
        node.module or ""
        for node in ast.walk(checkpoint)
        if isinstance(node, ast.ImportFrom)
    }
    runtime_imports = {
        node.module or "" for node in ast.walk(runtime) if isinstance(node, ast.ImportFrom)
    }
    recipe_imports = {
        node.module or "" for node in ast.walk(recipe) if isinstance(node, ast.ImportFrom)
    }
    assert "z_image_turbo_recipe" not in " ".join(checkpoint_imports)
    assert "z_image_qwen_runtime" not in " ".join(checkpoint_imports)
    assert "z_image_qwen_architecture" in " ".join(checkpoint_imports)
    assert "z_image_qwen_checkpoint" in " ".join(recipe_imports)
    assert "z_image_qwen_architecture" not in " ".join(recipe_imports)
    assert "z_image_qwen_runtime" not in " ".join(recipe_imports)
    assert "z_image_qwen_architecture" in " ".join(runtime_imports)
    assert "z_image_qwen_checkpoint" in " ".join(runtime_imports)


def test_engine_has_no_comfyui_import_or_launch_surface():
    source_root = ROOT / "src" / "latentslate_engine"
    violations: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = _tree(path)
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            for module in imported:
                lowered = module.lower()
                if lowered == "comfy" or lowered.startswith("comfy.") or "comfyui" in lowered:
                    violations.append(f"{path.relative_to(ROOT)} imports {module}")
            if isinstance(node, ast.Call) and _call_name(node.func) in {
                "subprocess.Popen",
                "subprocess.run",
                "subprocess.call",
                "subprocess.check_call",
                "subprocess.check_output",
            }:
                literals = " ".join(
                    value
                    for child in ast.walk(node)
                    if isinstance(child, ast.Constant) and isinstance((value := child.value), str)
                ).lower()
                if "comfyui" in literals or "main.py --listen" in literals:
                    violations.append(f"{path.relative_to(ROOT)} launches ComfyUI")
    assert violations == []


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""
