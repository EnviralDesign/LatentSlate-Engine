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
- configured runtime profiles for every model family;
- local Hugging Face authentication presence without printing the token;
- canonical bundle cache status;
- actionable warnings for known V0 constraints, including H3 host-memory pressure
  and unvalidated 16 GB LTX/Wan offload paths.

The report is ready for inference when CUDA is available, at least one model
family has a complete dependency set, and no configuration error such as an
invalid or retired runtime profile is present. Missing bundles are warnings rather
than hard failures because they can be installed separately. Gated repositories
still require prior authentication and accepted terms.

The report deliberately avoids network calls and never prints credentials.
