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

  The Krea Turbo family under `src/latentslate_engine/krea2/` narrowly adapts
  `comfy/ldm/krea2/model.py`, Flux positional embeddings, Qwen3-VL/llama text
  inference and token sampling, mixed-precision operations, latent normalization,
  Euler sampling, and the single-frame path of `comfy/ldm/wan/vae.py` from that
  same commit. The Wan VAE originates with the Alibaba Wan Team (2024–2025).
  Comfy operation source includes copyright (C) 2024 Stability AI. Runtime
  dependencies are direct Torch/AIMDO/Kitchen primitives, not ComfyUI.
  Krea model weights are separately governed by the Krea 2 Community License;
  they are not included in this repository.

Update this notice when additional adapted source or third-party dependencies enter the
tracked implementation.
