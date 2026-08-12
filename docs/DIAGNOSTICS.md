# Engine Diagnostics

Run the preflight after bootstrap, before downloading a recipe closure or attempting
the first GPU job:

```powershell
.\scripts\engine.ps1 doctor
```

```bash
./scripts/engine.sh doctor
```

For automation or a remote bootstrap script:

```powershell
.\scripts\engine.ps1 doctor --json
```

```bash
./scripts/engine.sh doctor --json
```

The command does not load a model or make a generation request. It reports:

- Python, operating system, and Engine version;
- system RAM, the resolved `LATENTSLATE_ENGINE_HOME`, model root, and free disk space;
- the recorded bootstrap selection mode, preferred and selected runtime tier, and
  any visible fallback reason;
- PyTorch/CUDA availability, actual Torch CUDA build, GPU names, compute capability,
  and VRAM;
- Comfy Kitchen availability, qualified backend, and hardware-gated capabilities;
- installed versions of the H3, LTX 2.3, Wan 2.2, and Klein runtime dependencies;
- the configured default execution mode for every model family;
- local Hugging Face authentication presence without printing the token;
- legacy compatibility-bundle cache status;
- actionable warnings for known V0 constraints, including H3 host-memory pressure
  and unvalidated 16 GB LTX/Wan offload paths.

The report's readiness is a runtime preflight. It does not mean model artifacts are
installed or that a recipe is runnable. Use `.\scripts\engine.ps1 recipes list` to
inspect recipe availability. Missing legacy bundles are warnings rather than hard
failures because recipe installation acquires the exact resource closure separately.
Gated repositories still require prior authentication and accepted terms.

The preferred setup command is `.\scripts\bootstrap.ps1` on Windows or
`./scripts/bootstrap.sh` on Linux. Auto mode selects the highest compatible locked
tier (`nvidia-cu130`, then `nvidia-cu128`, then `protocol`) and persists the result
below the Engine data root. An automatic fallback is reported; it is never hidden.
Use `scripts\engine.ps1` on Windows or `./scripts/engine.sh` on Linux for later
commands so uv retains the recorded tier.

The report deliberately avoids network calls and never prints credentials.
