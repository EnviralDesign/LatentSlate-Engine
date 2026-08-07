#!/usr/bin/env bash
set -euo pipefail
uv sync --extra h3 --extra klein
uv run latentslate-engine bundles install h3-basic
