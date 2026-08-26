# Engine Diagnostics

Run the preflight after bootstrap, before downloading a recipe closure or attempting
the first GPU job:

```powershell
.\scripts\engine.ps1 doctor
.\scripts\engine.ps1 doctor --json
```

```bash
./scripts/engine.sh doctor
./scripts/engine.sh doctor --json
```

The command does not load a model or make a generation request. It reports Python,
operating system, Engine version, RAM/disk, Engine home, selected runtime tier and
fallback reason, Torch/CUDA/GPU capability, Comfy Kitchen availability and qualified
direct backends, standalone comfy-aimdo package metadata, family runtime dependencies,
authentication presence, and actionable warnings. Doctor never initializes AIMDO;
that remains a persistent GPU-child responsibility.

This diagnostic follows [COMFY_ENGINE_POLICY.md](./COMFY_ENGINE_POLICY.md):
ComfyUI is not an Engine dependency and must not appear as a readiness requirement,
process, server, graph executor, plugin host, or fallback. Only Comfy Kitchen and
standalone low-level comfy-aimdo are allowed direct dependencies from the Comfy
ecosystem.

Readiness is a runtime preflight. It does not mean model artifacts are installed or a
recipe is runnable. Recipe availability still requires the complete typed closure,
Engine-owned loader/orchestration, direct Kitchen/native capability where applicable,
license/auth gates, and truthful operation support.

The preferred setup command is `.\scripts\bootstrap.ps1` on Windows or
`./scripts/bootstrap.sh` on Linux. Auto mode selects the highest compatible locked tier
and persists it below Engine home. An automatic fallback is reported, never hidden.
The report avoids network calls and never prints credentials.
