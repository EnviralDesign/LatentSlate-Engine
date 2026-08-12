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

Repeat `--recipe` in the desired order. Recipes run recipe-major so repeats two
and three can expose warm behavior without switching the active runtime between
every job:

```powershell
uv run --no-sync python scripts\hardware-study.py `
  --recipe flux2-klein-4b.image-to-image.comfy-distilled-fp8 `
  --recipe flux2-klein-4b.image-to-image.comfy-base-fp8 `
  --prompt "Change the bag color to blue." `
  --seed 43301611940728 `
  --asset source_image=C:\path\source.png `
  --input width=1024 --input height=1024 `
  --repeat 2
```

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

## Evidence and limits

The manifest records exact catalog descriptors and schema identities, effective
requests, upload identities, Engine job responses, provenance, artifact metadata,
download hashes, client and server timestamps, runtime status, and device-wide
`nvidia-smi` samples.

- GPU samples include other processes and are labeled accordingly; they are not
  exact per-process allocator peaks.
- A first run is only truly cold when the Engine was freshly started. Clearing
  prompt/media caches does not unload every runtime.
- Jobs are in memory. Keep the Engine running until polling and artifact download
  complete.
- On timeout the harness requests cancellation and waits up to the configured
  cancellation grace for a terminal job state. It explicitly reports when
  cancellation cannot be confirmed. Ctrl+C requests cancellation for the active
  job before exiting.
- Keep comparisons within the same lineage and operation. A four-step Distilled
  edit and a twenty-step Base edit answer different product questions even when
  they use the same prompt, seed, and source.
