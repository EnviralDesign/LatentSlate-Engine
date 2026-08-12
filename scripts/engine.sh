#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
python_path="$repo_root/.venv/bin/python"
if [[ ! -x "$python_path" ]]; then
  echo "The Engine environment is missing. Run ./scripts/bootstrap.sh first." >&2
  exit 1
fi

engine_home="$("$python_path" -c 'from latentslate_engine.model_store import configured_engine_home; print(configured_engine_home())')"
state_path="$engine_home/runtime/runtime-selection.json"
if [[ ! -f "$state_path" ]]; then
  echo "No runtime selection state found. Run ./scripts/bootstrap.sh first." >&2
  exit 1
fi
tier="$("$python_path" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_tier"])' "$state_path")"
case "$tier" in
  nvidia-cu130|nvidia-cu128|protocol) ;;
  *) echo "Runtime selection state has an invalid tier: $tier" >&2; exit 1 ;;
esac

uv_args=(run --locked --extra "$tier")
if [[ "$tier" != protocol ]]; then
  uv_args+=(--group runtime)
fi
uv_args+=(latentslate-engine "$@")
cd "$repo_root"
exec uv "${uv_args[@]}"
