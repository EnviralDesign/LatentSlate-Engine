#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add HF_TOKEN there for gated models."
fi

uv sync
uv run latentslate-engine doctor
