from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src" / "latentslate_engine" / "runtime"
FRAMEWORK = RUNTIME / "framework"
Z_QWEN_ARCHITECTURE = RUNTIME / "z_image_qwen_architecture.py"
Z_QWEN_CHECKPOINT = RUNTIME / "z_image_qwen_checkpoint.py"
Z_QWEN_RUNTIME = RUNTIME / "z_image_qwen_runtime.py"
Z_IMAGE_RECIPE = ROOT / "src" / "latentslate_engine" / "z_image_turbo_recipe.py"
WAN_NATIVE_MANAGED = RUNTIME / "wan22_native_managed.py"
WAN_NATIVE_WORKER = RUNTIME / "wan22_native_worker.py"
LTX_MANAGED = RUNTIME / "ltx23_managed.py"
LTX_WORKER = RUNTIME / "ltx23_worker.py"
LTX_KITCHEN_MANAGED = RUNTIME / "ltx23_kitchen_managed.py"
LTX_KITCHEN_WORKER = RUNTIME / "ltx23_kitchen_worker.py"
WAN_PROMPT_PARENT = RUNTIME / "wan22.py"
WAN_PROMPT_WORKER = RUNTIME / "wan22_prompt_worker.py"
VARIANTS = ROOT / "src" / "latentslate_engine" / "variants.py"
SHARED_RUNTIME_PATHS = (
    ROOT / "src" / "latentslate_engine" / "stored_quant.py",
    RUNTIME / "cache.py",
    RUNTIME / "manager.py",
    RUNTIME / "process_memory.py",
    RUNTIME / "residency_policy.py",
    RUNTIME / "windows_process.py",
)

FAMILY_PREFIXES = {
    "klein": ("klein",),
    "ltx": ("ltx",),
    "wan": ("wan",),
    "zimage": ("z_image", "zimage"),
    "h3": ("h3",),
    "qwen": ("qwen",),
    "krea": ("krea",),
    "ideogram": ("ideogram",),
    "sdxl": ("sdxl",),
}
MODEL_MARKERS = tuple(
    prefix for prefixes in FAMILY_PREFIXES.values() for prefix in prefixes
)
_MODEL_NAME = re.compile(
    r"(?<![a-z0-9])(?:klein|ltx|wan|z[_-]?image|zimage|h3|qwen|krea|ideogram|sdxl)(?![a-z0-9])"
)

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
    normalized = module_or_file.replace("\\", "/")
    leaf = (
        Path(normalized).stem
        if normalized.endswith(".py")
        else normalized.rsplit("/", 1)[-1].rsplit(".", 1)[-1]
    )
    for family, prefixes in FAMILY_PREFIXES.items():
        if any(leaf.startswith(prefix) for prefix in prefixes):
            return family
    return None


def _contains_model_name(value: str) -> bool:
    return _MODEL_NAME.search(value.lower()) is not None


def _imported_module(node: ast.ImportFrom, alias: ast.alias) -> str:
    return node.module or alias.name


def test_private_cross_family_import_debt_is_fixed_and_cannot_grow():
    observed: set[tuple[str, str, str, str]] = set()
    phases = {
        (file_name, module, symbol): phase
        for file_name, module, symbol, phase in GRANDFATHERED_PRIVATE_IMPORTS
    }
    for path in sorted(RUNTIME.rglob("*.py")):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = _relative_module(node)
            for alias in node.names:
                if not alias.name.startswith("_"):
                    continue
                key = (path.name, module, alias.name)
                if key in phases:
                    observed.add((*key, phases[key]))
                    continue
                source_family = _model_family(path.stem)
                target_family = _model_family(_imported_module(node, alias))
                if source_family != target_family and target_family is not None:
                    raise AssertionError(
                        f"new private cross-family import: {path.name} -> {module}.{alias.name}"
                    )
    assert observed == GRANDFATHERED_PRIVATE_IMPORTS


def test_model_named_cross_family_import_debt_is_exact_and_cannot_grow():
    observed: set[tuple[str, str, str, str]] = set()
    phases = {
        (file_name, module, symbol): phase
        for file_name, module, symbol, phase in GRANDFATHERED_MODEL_NAMED_IMPORTS
    }
    for path in sorted(RUNTIME.rglob("*.py")):
        source_family = _model_family(path.name)
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ImportFrom):
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
                    f"new model-named cross-family import: {path.name} -> {module}.{symbol}"
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
                assert _model_family(module) is None, (
                    f"framework imports model family in {path}: {module}"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.lower()
                    assert _model_family(module) is None, (
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
                assert not _contains_model_name(value), (
                    f"framework contains model-named protocol/policy text in {path}"
                )


def test_existing_shared_runtime_capabilities_remain_model_neutral():
    for path in SHARED_RUNTIME_PATHS:
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.lower()
                assert not _contains_model_name(value), (
                    f"shared runtime capability contains model-specific text in {path}"
                )


def test_production_has_no_dynamic_recipe_selected_execution_surface():
    source_root = ROOT / "src" / "latentslate_engine"
    forbidden_builtins = {"__import__", "eval", "exec"}
    forbidden_calls = {
        "importlib.import_module",
        "importlib.util.spec_from_file_location",
        "importlib.metadata.entry_points",
        "pkg_resources.iter_entry_points",
        "pkg_resources.load_entry_point",
        "runpy.run_module",
        "runpy.run_path",
    }
    violations: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if name in forbidden_calls or (
                isinstance(node.func, ast.Name) and name in forbidden_builtins
            ):
                violations.append(
                    f"{path.relative_to(ROOT)} calls {name}"
                )
    assert violations == []


def test_z_image_qwen_split_has_one_way_responsibility_edges_and_no_facade():
    assert not (RUNTIME / "z_image_mixed_qwen.py").exists()
    architecture = _tree(Z_QWEN_ARCHITECTURE)
    checkpoint = _tree(Z_QWEN_CHECKPOINT)
    runtime = _tree(Z_QWEN_RUNTIME)
    recipe = _tree(Z_IMAGE_RECIPE)

    architecture_imports = {
        node.module or "" for node in ast.walk(architecture) if isinstance(node, ast.ImportFrom)
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
        node.module or "" for node in ast.walk(checkpoint) if isinstance(node, ast.ImportFrom)
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


def test_native_wan_has_one_shared_persistent_supervisor_path():
    managed = _tree(WAN_NATIVE_MANAGED)
    worker = _tree(WAN_NATIVE_WORKER)
    managed_names = {node.name for node in ast.walk(managed) if isinstance(node, ast.FunctionDef)}
    assert "_DisposableNativeWanI2VRuntime" not in WAN_NATIVE_MANAGED.read_text(encoding="utf-8")
    assert "PersistentWorkerSupervisor" in WAN_NATIVE_MANAGED.read_text(encoding="utf-8")
    assert "run_persistent_child" in WAN_NATIVE_WORKER.read_text(encoding="utf-8")
    assert "_spawn_session" not in managed_names
    for tree, path in ((managed, WAN_NATIVE_MANAGED), (worker, WAN_NATIVE_WORKER)):
        calls = {_call_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        assert "subprocess.Popen" not in calls, f"model-owned process launch returned in {path}"
        assert "hmac.new" not in calls, f"model-owned HMAC implementation returned in {path}"
        assert "atomic_write_json" not in calls, f"model-owned atomic IPC returned in {path}"
        assert "append_bounded_jsonl" not in calls, f"model-owned JSONL transport returned in {path}"


def test_active_ltx_and_prompt_workers_use_only_shared_worker_control_planes():
    expectations = (
        (LTX_MANAGED, "DisposableWorkerSupervisor"),
        (LTX_WORKER, "run_disposable_child"),
        (LTX_KITCHEN_MANAGED, "PersistentWorkerSupervisor"),
        (LTX_KITCHEN_WORKER, "run_persistent_child"),
        (WAN_PROMPT_PARENT, "DisposableWorkerSupervisor"),
        (WAN_PROMPT_WORKER, "run_disposable_child"),
    )
    forbidden = {
        "subprocess.Popen",
        "DisposableProcessTree",
        "drain_bounded_jsonl",
        "append_bounded_jsonl",
    }
    for path, required in expectations:
        source = path.read_text(encoding="utf-8")
        tree = _tree(path)
        calls = {_call_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        assert required in source, f"{path} is not using {required}"
        assert calls.isdisjoint(forbidden), (
            f"model-owned worker infrastructure returned in {path}: "
            f"{sorted(calls.intersection(forbidden))}"
        )


def test_recipe_dispatch_is_a_closed_static_registry_without_type_conditionals():
    source = VARIANTS.read_text(encoding="utf-8")
    tree = _tree(VARIANTS)
    assert "_RECIPE_HANDLER_BY_CONFIG" in source
    assert "_RECIPE_HANDLER_BY_RECIPE" in source
    assert "_RECIPE_HANDLER_BY_TYPE_NAME" in source
    variant_definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VariantDefinition"
    )
    recipe_field = next(
        node
        for node in variant_definition.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "recipe"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id.endswith("RecipeConfig")
        for node in ast.walk(recipe_field.annotation)
    ), "new recipe types must not require editing a central Pydantic union"
    assert not any(
        isinstance(node, ast.Call)
        and _call_name(node.func) == "isinstance"
        and any(
            isinstance(child, ast.Name) and "Recipe" in child.id
            for argument in node.args[1:]
            for child in ast.walk(argument)
        )
        for node in ast.walk(tree)
    ), "recipe dispatch must remain registry-driven rather than type-conditional"


def test_z_worker_protocol_has_one_parent_child_authority():
    for path in (RUNTIME / "z_image_turbo_managed.py", RUNTIME / "z_image_turbo_worker.py"):
        source = path.read_text(encoding="utf-8")
        assert "z_image_worker_protocol" in source
    protocol_names = {
        "CUDA_ERROR_CODES",
        "CUDA_HEALTH_PHASES",
        "QWEN_FAILURE_STAGES",
        "FAILURE_STAGES",
        "FAILURE_LOCATIONS",
        "SAFE_EXCEPTION_NAMES",
    }
    for path in (RUNTIME / "z_image_turbo_managed.py", RUNTIME / "z_image_turbo_worker.py"):
        assigned = {
            target.id
            for node in ast.walk(_tree(path))
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
            if isinstance(target, ast.Name)
        }
        assert assigned.isdisjoint(protocol_names), f"duplicated Z protocol authority in {path}"


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""
