[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$EngineArguments
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonPath = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "The Engine environment is missing. Run .\scripts\bootstrap.ps1 first."
}

# Resolve Engine home through the same Python settings path as bootstrap/doctor.
# This honors an ignored .env file as well as a process-level override without
# duplicating dotenv parsing in PowerShell.
$EngineHome = (& $PythonPath -c "from latentslate_engine.model_store import configured_engine_home; print(configured_engine_home())").Trim()
if ($LASTEXITCODE -ne 0 -or -not $EngineHome) {
    throw "Could not resolve LATENTSLATE_ENGINE_HOME from the bootstrapped environment."
}
$StatePath = Join-Path $EngineHome "runtime\runtime-selection.json"
if (-not (Test-Path -LiteralPath $StatePath)) {
    throw "No runtime selection state found. Run .\scripts\bootstrap.ps1 first."
}
$State = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
$UvArguments = @("run", "--locked", "--extra", $State.selected_tier)
if ($State.selected_tier -ne "protocol") {
    $UvArguments += @("--group", "runtime")
}
$UvArguments += @("latentslate-engine") + $EngineArguments
Push-Location $RepoRoot
try {
    & uv @UvArguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
