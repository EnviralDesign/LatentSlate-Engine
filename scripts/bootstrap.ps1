[CmdletBinding()]
param(
    [ValidateSet("Auto", "Cu130", "Cu128", "Protocol")]
    [string]$Backend = "Auto"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EnvPath = Join-Path $RepoRoot ".env"
$EnvExamplePath = Join-Path $RepoRoot ".env.example"
$PolicyScript = Join-Path $PSScriptRoot "runtime_bootstrap.py"

function Invoke-LockedSync([pscustomobject]$Selection) {
    $UvArguments = @("sync", "--locked", "--extra", $Selection.selected_tier)
    if ($Selection.selected_tier -ne "protocol") {
        $UvArguments += @("--group", "runtime")
    }
    & uv @UvArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Locked uv sync failed for $($Selection.selected_tier). This is not a compatibility fallback; fix the install failure and retry."
    }
}

function Invoke-Validation([string]$PythonPath, [string]$Tier) {
    $Raw = & $PythonPath $PolicyScript validate --tier $Tier
    $ExitCode = $LASTEXITCODE
    try {
        $Payload = $Raw | ConvertFrom-Json
    }
    catch {
        throw "Runtime validation produced unreadable output for $Tier. This is not a compatibility fallback."
    }
    return [pscustomobject]@{ ExitCode = $ExitCode; Payload = $Payload }
}

Push-Location $RepoRoot
try {
    if (Test-Path -LiteralPath $EnvPath) {
        Write-Host "Preserving existing .env."
    }
    elseif (Test-Path -LiteralPath $EnvExamplePath -PathType Leaf) {
        try {
            [System.IO.File]::Copy($EnvExamplePath, $EnvPath, $false)
            Write-Host "Created .env from .env.example. Add HF_TOKEN or CIVITAI_TOKEN only when a deployment plan requires it."
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

    # Use the checked-in Engine interpreter target for selection. This can be
    # uv-managed on a fresh machine; unsupported Python never selects a GPU tier.
    $SelectionRaw = & uv run --python 3.12 --no-project $PolicyScript select --mode $Backend.ToLowerInvariant()
    if ($LASTEXITCODE -ne 0) {
        $Failure = $SelectionRaw | ConvertFrom-Json
        throw $Failure.message
    }
    $Selection = $SelectionRaw | ConvertFrom-Json
    Write-Host "Runtime selection"
    Write-Host "  Preferred: $($Selection.preferred_tier)"
    Write-Host "  Selected:  $($Selection.selected_tier)"
    Write-Host "  Reason:    $($Selection.reason)"
    Write-Host "  Impact:    $($Selection.impact)"

    Invoke-LockedSync $Selection
    $PythonPath = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        throw "Locked uv sync completed without creating the expected project Python at $PythonPath."
    }
    $Validation = Invoke-Validation $PythonPath $Selection.selected_tier

    # Only a validated CUDA/Kitchen incompatibility permits one automatic downgrade.
    # Resolver, network, lock, hash, and generic import failures remain hard errors.
    $FallbackCodes = @(
        "torch_cuda_unavailable",
        "torch_cuda_mismatch",
        "kitchen_cuda_backend_unavailable"
    )
    if (
        $Backend -eq "Auto" -and
        $Selection.selected_tier -eq "nvidia-cu130" -and
        $Validation.ExitCode -ne 0 -and
        $FallbackCodes -contains $Validation.Payload.error_code
    ) {
        $FallbackReason = $Validation.Payload.message
        Write-Host "CUDA 13 validation fallback: $FallbackReason"
        $Selection.selected_tier = "nvidia-cu128"
        $Selection.reason = "CUDA 13 validation failed with a classified CUDA/Kitchen compatibility error."
        $Selection.impact = "CUDA 13 Kitchen acceleration is unavailable; CUDA 12.8 compatibility recipes remain available."
        $Selection.fallback_reason = $FallbackReason
        Invoke-LockedSync $Selection
        $Validation = Invoke-Validation $PythonPath $Selection.selected_tier
    }
    if ($Validation.ExitCode -ne 0) {
        throw "Runtime validation failed for $($Selection.selected_tier): $($Validation.Payload.message)"
    }

    & $PythonPath -m latentslate_engine data init
    if ($LASTEXITCODE -ne 0) {
        throw "Engine data initialization failed after runtime validation."
    }
    $EngineHome = (& $PythonPath -c "from latentslate_engine.model_store import configured_engine_home; print(configured_engine_home())").Trim()
    $StateDirectory = Join-Path $EngineHome "runtime"
    New-Item -ItemType Directory -Force -Path $StateDirectory | Out-Null
    $Selection | Add-Member -NotePropertyName validation -NotePropertyValue $Validation.Payload -Force
    $StatePath = Join-Path $StateDirectory "runtime-selection.json"
    # PowerShell 5.1's Set-Content -Encoding utf8 emits a BOM; write a
    # BOM-less UTF-8 record that is portable to Python and Bash readers.
    [System.IO.File]::WriteAllText(
        $StatePath,
        (($Selection | ConvertTo-Json -Depth 8) + [Environment]::NewLine),
        [System.Text.UTF8Encoding]::new($false)
    )

    Write-Host "Runtime ready"
    Write-Host "  Selected: $($Selection.selected_tier)"
    if ($Selection.fallback_reason) {
        Write-Host "  Fallback: $($Selection.fallback_reason)"
    }
    Write-Host "  State:    $StatePath"
    & $PythonPath -m latentslate_engine doctor
    $DoctorExitCode = $LASTEXITCODE
    if ($Selection.selected_tier -ne "protocol" -and $DoctorExitCode -ne 0) {
        throw "Doctor failed after GPU runtime bootstrap; inspect the reported runtime drift or CUDA error."
    }
    if ($Selection.selected_tier -eq "protocol" -and $DoctorExitCode -ne 0) {
        Write-Host "Doctor correctly reports protocol-only as not inference-ready."
    }
    Write-Host "Next: .\scripts\engine.ps1 recipes list"
}
finally {
    Pop-Location
}
