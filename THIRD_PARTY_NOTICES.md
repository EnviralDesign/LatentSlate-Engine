# Third-party notices

LatentSlate Engine is GPLv3. This greenfield branch currently contains no runtime
implementation or dependency lockfile.

The first LTX 2.3 implementation is expected to use or adapt from the following
pinned upstream projects. Preserve their applicable license and attribution
requirements as implementation is added:

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

Update this notice when actual dependencies or adapted source enter the tracked
implementation.
