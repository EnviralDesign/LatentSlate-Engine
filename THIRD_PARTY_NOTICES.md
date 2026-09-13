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

  Qwen Image Edit 2511 preprocessing and single-frame VAE encoding under
  `src/latentslate_engine/qwen2511/` adapt the pinned `nodes_flux.py`,
  `nodes_qwen.py`, `comfy/utils.py` and `comfy/ldm/wan/vae.py` paths.
  Its Qwen 2.5 VL visual and language conditioning adapt
  `comfy/text_encoders/qwen_vl.py`, `llama.py`, `qwen_image.py`, and
  the scaled FP8 full-precision linear operation from `comfy/ops.py`.
  Its diffusion transformer adapts `comfy/ldm/qwen_image/model.py`,
  Lightricks timestep embeddings, and Flux position math; the Qwen-Image
  upstream model source is Apache-2.0 licensed. Curated sampling follows
  `model_sampling.py`, `samplers.py`, `nodes_cfg.py`, and Wan21 normalization.
  The identical Qwen Image decoder, Torch attention dispatch and mapped
  checkpoint primitives used by Krea and Qwen reside in `qwen_image_vae.py`,
  `torch_attention.py` and `mapped_checkpoint.py` under the Engine package.

Update this notice when additional adapted source or third-party dependencies enter the
tracked implementation.
