# Compact acceptance evidence

`acceptance/*.json` contains bounded, privacy-safe records extracted from ignored
`hardware-study-runs/**/manifest.json` files. Records identify the exact
execution-contract fingerprint, tool schema, source state, resource closure,
environment class, lifecycle results, observed dispatch/fallback facts, output
hashes, creator review, and limitations.

The compact records never retain prompts, submitted inputs, media, credentials,
absolute paths, account/instance/job identifiers, or full logs. The original
manifests and generated media remain local and ignored.

Create records only with manifests produced by the current study harness:

```powershell
.venv\Scripts\python.exe scripts\acceptance-evidence.py extract `
  --manifest hardware-study-runs\<study>\manifest.json `
  --recipe <recipe-key> --evidence-id <stable-id>
```

After creator review, validate records and regenerate the checked matrix:

```powershell
.venv\Scripts\python.exe scripts\acceptance-evidence.py validate --write-matrix
```

Validation fails when a Hardware-proven recipe lacks a current accepted record,
resource identities drift, a zero-fallback claim lacks observed zero counters,
IDs collide, or a record contains private/path-shaped data. Hardware-proven is
an execution/lifecycle proof level; it does not imply Recommended tier or pixel
parity with ComfyUI.
