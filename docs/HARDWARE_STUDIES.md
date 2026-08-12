# Opt-in hardware generation studies

`scripts/hardware-study.py` runs small, manually requested generation studies
through the same public HTTP routes used by LatentSlate. It is intentionally
outside `tests/`, is never run by ordinary pytest or CI, and does not import
Engine internals.

Use it for one-off hardware smoke tests and small deterministic recipe
comparisons—not broad benchmarking.

## Start the Engine

Use the recorded runtime tier and keep the server console visible so failures and
tracebacks remain observable:

```powershell
.\scripts\engine.ps1 serve
```

In another terminal, preflight a request without uploading assets or creating a
job:

```powershell
uv run --no-sync python scripts\hardware-study.py `
  --recipe flux2-klein-4b.image-to-image.comfy-distilled-fp8 `
  --prompt "Change the bag color to blue." `
  --seed 43301611940728 `
  --asset source_image=C:\path\source.png `
  --input width=1024 --input height=1024 `
  --preflight-only
```

Then remove `--preflight-only` to generate. The script uploads each source once,
submits jobs sequentially, polls to a terminal state, downloads every artifact,
and writes `manifest.json` plus state changes in `events.jsonl` beneath a new
ignored `hardware-study-runs/<timestamp>/` directory. The client only accepts a
loopback Engine URL and reads `LATENTSLATE_ENGINE_TOKEN` from the process or the
repository `.env`; credentials are never written to the manifest.

## Small A/B runs

Repeat `--recipe` in the desired order. Recipes run recipe-major so later repeats
can expose warm behavior without switching the active runtime between every job.
For measurements rather than exploratory smoke tests, reset and prove the runtime
state before each recipe:

```powershell
uv run --no-sync python scripts\hardware-study.py `
  --recipe flux2-klein-4b.image-to-image.comfy-distilled-fp8 `
  --recipe flux2-klein-4b.image-to-image.comfy-base-fp8 `
  --prompt "Change the bag color to blue." `
  --seed 43301611940728 `
  --asset source_image=C:\path\source.png `
  --input width=1024 --input height=1024 `
  --repeat 4 `
  --reset-runtime-before-recipe `
  --assert-runtime-state `
  --assert-deterministic
```

With `--cold-repeats 1`, this produces one proven `runtime_cold` observation followed
by three proven `pipeline_warm_cache_warm` observations for each recipe. The family
benchmark scenarios use six jobs per recipe: three independently reset cold trials,
then three warm/cache-hit trials on the final loaded runtime. The reset endpoint unloads
and evicts Engine runtime wrappers and their prompt/reference caches. It does not
flush Windows filesystem caches, restart Python, or reboot the host, so the harness
never mislabels it `process_cold`.

Explicit matching dimensions are important when comparing these Klein recipes:
the official Distilled workflow scales references toward one megapixel, while the
Base workflow otherwise inherits the source canvas.

Use additional ordered references with:

```powershell
--asset reference_image_2=C:\path\reference2.png `
--asset reference_image_3=C:\path\reference3.png
```

Generic family-specific values are accepted as JSON literals through repeated
`--input KEY=VALUE` options. Values that are not valid JSON are treated as plain
strings.

## Klein 4B acceptance scenarios

`scripts/klein4b-generation-tests.py` is the first family-level acceptance suite.
It keeps fixed 1024×1024 dimensions, prompts, and seed while composing the generic
HTTP harness into named manual scenarios:

- `t2i-smoke` / `i2i-smoke`: one runtime-reset, state-asserted Recommended NVFP4 generation;
- `t2i-warm` / `i2i-warm`: one proven runtime-cold call plus three proven warm,
  identical-input calls to the Recommended recipe;
- `t2i-benchmark` / `i2i-benchmark`: three independently reset runtime-cold trials
  plus three warm/cache-hit trials per recipe. Klein 4B covers Distilled NVFP4,
  Distilled FP8, and its runnable BF16 Reference; Klein 9B covers NVFP4 and FP8
  because its exact BF16 Reference honestly exceeds the 16 GB workstation;
- `t2i-switch` / `i2i-switch`: Recommended → Fallback → Recommended, retaining the
  same Engine process so runtime eviction and reconstruction are exercised;
- `t2i-family` / `i2i-family`: one pass across every current operation-compatible
  Klein recipe, including the BF16 Reference path.

Start Engine separately, then run for example:

```powershell
uv run --no-sync python scripts\klein4b-generation-tests.py t2i-warm

uv run --no-sync python scripts\klein4b-generation-tests.py t2i-benchmark

uv run --no-sync python scripts\klein4b-generation-tests.py i2i-switch `
  --source-image C:\path\source.png
```

Add `--preflight-only` to prove current catalog/schema/input compatibility without
creating a GPU job. Family sweeps are best effort on the operator's workstation:
the runner records a failure and stops by default, while `--keep-going` attempts
later sub-runs. A BF16 Reference failure caused by local RAM/VRAM limits is an
acceptance result, not a reason to weaken the reference recipe or burden routine CI.

## Klein 9B acceptance scenarios

`scripts/klein9b-generation-tests.py` mirrors the same small scenario grammar for
the ordinary Distilled 9B ladder: Recommended first-party NVFP4, first-party FP8
Fallback, and complete BF16 Reference. Base and KV are deliberately absent.

```powershell
uv run --no-sync python scripts\klein9b-generation-tests.py t2i-smoke

uv run --no-sync python scripts\klein9b-generation-tests.py t2i-benchmark

uv run --no-sync python scripts\klein9b-generation-tests.py i2i-switch `
  --source-image C:\path\source.png
```

Use `--preflight-only` before installing or loading weights to prove the live
catalog/schema/input contract. The `*-warm`, `*-switch`, and `*-family` scenarios
have the same meaning as 4B. The BF16 family sweep remains best effort on the local
16 GB/64 GB workstation; keep its failure in the manifest rather than substituting
a smaller or quantized artifact for the reference.

## Evidence and limits

The v2 manifest records exact catalog descriptors and schema identities, effective
requests, upload identities, Engine job responses, provenance, artifact metadata,
download hashes, lifecycle preconditions, observed pipeline/cache state, client and
server timestamps, and device-wide `nvidia-smi` samples. Its measurement summary
separates server queue/execution time, submit-to-terminal observation, artifact
download, total client time, and cold/warm distributions. It also records whether
identical repeated outputs were byte-deterministic.

- GPU samples include other processes and are labeled accordingly; they are not
  exact per-process allocator peaks.
- `runtime_cold` is proven by `DELETE /v1/runtime` plus an empty manager status
  immediately before submission. `process_cold` requires a freshly started Engine
  and is never inferred. OS filesystem-cache state is outside both labels and must
  be reported separately if it matters to a study.
- Jobs are in memory. Keep the Engine running until polling and artifact download
  complete.
- On timeout the harness requests cancellation and waits up to the configured
  cancellation grace for a terminal job state. It explicitly reports when
  cancellation cannot be confirmed. Ctrl+C requests cancellation for the active
  job before exiting.
- Keep comparisons within the same lineage and operation. A four-step Distilled
  edit and a twenty-step Base edit answer different product questions even when
  they use the same prompt, seed, and source.
