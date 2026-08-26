# Third-party notices

LatentSlate Engine is GPLv3. The greenfield rebuild now contains an initial LTX 2.3
runtime implementation and direct/adapted use of the pinned upstream projects below.
There is not yet a committed dependency lockfile on this branch.

Preserve the applicable license and attribution requirements as implementation evolves:

- **comfy-aimdo 0.4.15** — GNU General Public License, version 3.
  Source: <https://github.com/Comfy-Org/comfy-aimdo>.
- **comfy-kitchen 0.2.31** — Apache License 2.0.
  Source: <https://github.com/Comfy-Org/comfy-kitchen>.
- **ComfyUI v0.34.0**, commit
  `12d5279438bfefc058a269eae805ceab6047777f` — GNU General Public License,
  version 3. ComfyUI is a source/behavior reference, not an Engine runtime
  dependency. Narrow source adaptations must retain appropriate provenance and
  attribution.
  Source: <https://github.com/Comfy-Org/ComfyUI>.

  The AIMDO-backed safetensors mapping in
  `src/latentslate_engine/ltx23/checkpoint.py` is narrowly adapted from
  `comfy.utils.load_safetensors` at that commit.

  The LTX AV transformer modules under `src/latentslate_engine/ltx23/` are
  narrow adaptations of ComfyUI's pinned `comfy.ldm.lightricks` sources. Their
  Comfy runtime call sites are replaced with direct Torch, AIMDO, and Kitchen
  primitives.

Update this notice when additional adapted source or third-party dependencies enter the
tracked implementation.
