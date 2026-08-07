$ErrorActionPreference = "Stop"

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Add HF_TOKEN there for gated models."
}

uv sync
uv run latentslate-engine doctor
