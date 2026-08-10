#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

if [[ -e .env || -L .env ]]; then
  echo "Preserving existing .env."
elif [[ -f .env.example ]]; then
  if (set -o noclobber; cat .env.example > .env) 2>/dev/null; then
    echo "Created .env from .env.example. Add HF_TOKEN there for gated models."
  else
    echo "Preserving existing .env."
  fi
fi

uv sync
uv run latentslate-engine data init
uv run latentslate-engine doctor
