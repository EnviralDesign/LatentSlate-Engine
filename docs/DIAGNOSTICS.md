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
- system RAM and free disk space near `LATENTSLATE_ENGINE_HOME`;
- PyTorch/CUDA availability, GPU names, compute capability, and VRAM;
- installed versions of the H3 and Klein runtime dependencies;
- configured H3, Klein 4B, and Klein 9B runtime profiles;
- local Hugging Face authentication presence without printing the token;
- canonical bundle cache status;
- actionable warnings for known V0 constraints, including H3 host-memory pressure
  and native-Windows Klein 9B ModelOpt uncertainty.

The process exits successfully when CUDA is available and at least one model
family has a complete dependency set. Missing bundles are warnings rather than
hard failures because Hugging Face can still download them on first use. Klein 9B
repositories require prior authentication and accepted terms; Klein 4B is the
simpler first local image test.

The report deliberately avoids network calls and never prints credentials.
