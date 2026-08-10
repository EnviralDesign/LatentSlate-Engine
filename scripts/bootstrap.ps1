$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EnvPath = Join-Path $RepoRoot ".env"
$EnvExamplePath = Join-Path $RepoRoot ".env.example"

Push-Location $RepoRoot
try {
    if (Test-Path -LiteralPath $EnvPath) {
        Write-Host "Preserving existing .env."
    }
    elseif (Test-Path -LiteralPath $EnvExamplePath -PathType Leaf) {
        try {
            [System.IO.File]::Copy($EnvExamplePath, $EnvPath, $false)
            Write-Host "Created .env from .env.example. Add HF_TOKEN there for gated models."
        }
        catch [System.IO.IOException] {
            if (Test-Path -LiteralPath $EnvPath) {
                Write-Host "Preserving existing .env."
            }
            else {
                throw
            }
        }
    }

    uv sync
    uv run latentslate-engine data init
    uv run latentslate-engine doctor
}
finally {
    Pop-Location
}
