# Engine Diagnostics

Run the preflight before downloading large bundles or attempting the first GPU job:

```bash
uv run latentslate-engine doctor
```

For automation or a remote bootstrap script:

```bash
uv run latentslate-engine doctor --json
```

The command does not load a model or make a generation request. It reports:

- Python, operating system, and Engine version;
- system RAM, the resolved `LATENTSLATE_ENGINE_HOME`, model root, and free disk space;
- PyTorch/CUDA availability, GPU names, compute capability, and VRAM;
- installed versions of the H3, LTX 2.3, Wan 2.2, and Klein runtime dependencies;
- the configured default execution mode for every model family;
- local Hugging Face authentication presence without printing the token;
- legacy compatibility-bundle cache status;
- actionable warnings for known V0 constraints, including H3 host-memory pressure
  and unvalidated 16 GB LTX/Wan offload paths.

The report's readiness is a runtime preflight: CUDA is available, at least one
model family has a complete dependency set, and no configuration error such as
an invalid or retired execution mode is present. It does not mean model artifacts
are installed or that a recipe is runnable. Use `latentslate-engine recipes list`
to inspect recipe availability. Missing legacy bundles are warnings rather than
hard failures because deployment profiles install the exact resource closure
separately. Gated repositories still require prior authentication and accepted
terms.

The report deliberately avoids network calls and never prints credentials.
