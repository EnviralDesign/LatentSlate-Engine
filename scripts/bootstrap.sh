#!/usr/bin/env bash
set -euo pipefail

backend="Auto"
usage() {
  cat <<'EOF'
Usage: ./scripts/bootstrap.sh [--backend auto|cu130|cu128|protocol]

Auto is the default. It selects the highest compatible locked Engine runtime,
then validates Torch/CUDA/Comfy Kitchen. A CUDA 13 -> CUDA 12.8 retry occurs
only for a classified CUDA/Kitchen backend validation error.
EOF
}

while (($#)); do
  case "$1" in
    --backend)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      backend="$2"
      shift 2
      ;;
    --backend=*)
      backend="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

backend="${backend,,}"
case "$backend" in
  auto|cu130|cu128|protocol) ;;
  *) usage >&2; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
policy_script="$script_dir/runtime_bootstrap.py"
cd "$repo_root"

if [[ -e .env || -L .env ]]; then
  echo "Preserving existing .env."
elif [[ -f .env.example ]]; then
  if (set -o noclobber; cat .env.example > .env) 2>/dev/null; then
    echo "Created .env from .env.example. Add HF_TOKEN or CIVITAI_TOKEN only when a deployment plan requires it."
  else
    echo "Preserving existing .env."
  fi
fi

json_field() {
  local payload="$1"
  local field="$2"
  uv run --python 3.12 --no-project python -c \
    'import json, sys; value = json.load(sys.stdin); print(value[sys.argv[1]] if value.get(sys.argv[1]) is not None else "")' \
    "$field" <<<"$payload"
}

if ! selection_raw="$(uv run --python 3.12 --no-project "$policy_script" select --mode "$backend")"; then
  message="$(json_field "$selection_raw" message 2>/dev/null || true)"
  echo "Runtime selection failed: ${message:-$selection_raw}" >&2
  exit 1
fi

selected_tier="$(json_field "$selection_raw" selected_tier)"
preferred_tier="$(json_field "$selection_raw" preferred_tier)"
reason="$(json_field "$selection_raw" reason)"
impact="$(json_field "$selection_raw" impact)"
printf 'Runtime selection\n  Preferred: %s\n  Selected:  %s\n  Reason:    %s\n  Impact:    %s\n' \
  "$preferred_tier" "$selected_tier" "$reason" "$impact"

locked_sync() {
  local tier="$1"
  local uv_args=(sync --locked --extra "$tier")
  if [[ "$tier" != protocol ]]; then
    uv_args+=(--group runtime)
  fi
  if ! uv "${uv_args[@]}"; then
    echo "Locked uv sync failed for $tier. This is not a compatibility fallback; fix the install failure and retry." >&2
    exit 1
  fi
}

validate() {
  local tier="$1"
  if validation_raw="$("$python_path" "$policy_script" validate --tier "$tier")"; then
    return 0
  fi
  return 1
}

locked_sync "$selected_tier"
python_path="$repo_root/.venv/bin/python"
if [[ ! -x "$python_path" ]]; then
  echo "Locked uv sync completed without creating the expected project Python at $python_path." >&2
  exit 1
fi

if ! validate "$selected_tier"; then
  error_code="$(json_field "$validation_raw" error_code 2>/dev/null || true)"
  error_message="$(json_field "$validation_raw" message 2>/dev/null || true)"
  if [[ "$backend" == auto && "$selected_tier" == nvidia-cu130 ]] && \
    [[ "$error_code" == torch_cuda_unavailable || "$error_code" == torch_cuda_mismatch || "$error_code" == kitchen_cuda_backend_unavailable ]]; then
    echo "CUDA 13 validation fallback: $error_message"
    selection_raw="$("$python_path" -c '
import json, sys
payload = json.loads(sys.stdin.read())
payload["selected_tier"] = "nvidia-cu128"
payload["reason"] = "CUDA 13 validation failed with a classified CUDA/Kitchen compatibility error."
payload["impact"] = "CUDA 13 Kitchen acceleration is unavailable; CUDA 12.8 compatibility recipes remain available."
payload["fallback_reason"] = sys.argv[1]
print(json.dumps(payload, separators=(",", ":")))
' "$error_message" <<<"$selection_raw")"
    selected_tier="nvidia-cu128"
    locked_sync "$selected_tier"
    if ! validate "$selected_tier"; then
      error_message="$(json_field "$validation_raw" message 2>/dev/null || true)"
      echo "Runtime validation failed for $selected_tier: ${error_message:-$validation_raw}" >&2
      exit 1
    fi
  else
    echo "Runtime validation failed for $selected_tier: ${error_message:-$validation_raw}" >&2
    exit 1
  fi
fi

"$python_path" -m latentslate_engine data init
engine_home="$("$python_path" -c 'from latentslate_engine.model_store import configured_engine_home; print(configured_engine_home())')"
state_path="$engine_home/runtime/runtime-selection.json"
mkdir -p -- "$(dirname -- "$state_path")"
"$python_path" -c '
import json, pathlib, sys
selection = json.loads(sys.argv[1])
selection["validation"] = json.loads(sys.argv[2])
path = pathlib.Path(sys.argv[3])
path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
' "$selection_raw" "$validation_raw" "$state_path"

printf 'Runtime ready\n  Selected: %s\n' "$selected_tier"
fallback_reason="$(json_field "$selection_raw" fallback_reason 2>/dev/null || true)"
if [[ -n "$fallback_reason" ]]; then
  printf '  Fallback: %s\n' "$fallback_reason"
fi
printf '  State:    %s\n' "$state_path"
if "$python_path" -m latentslate_engine doctor; then
  doctor_exit_code=0
else
  doctor_exit_code=$?
fi
if [[ "$selected_tier" != protocol && "$doctor_exit_code" -ne 0 ]]; then
  echo "Doctor failed after GPU runtime bootstrap; inspect the reported runtime drift or CUDA error." >&2
  exit "$doctor_exit_code"
fi
if [[ "$selected_tier" == protocol && "$doctor_exit_code" -ne 0 ]]; then
  echo "Doctor correctly reports protocol-only as not inference-ready."
fi
echo "Next: ./scripts/engine.sh recipes list"
