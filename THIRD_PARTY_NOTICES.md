# Third-party notices

LatentSlate Engine is GPLv3. The greenfield rebuild now contains an initial LTX 2.3
runtime implementation and direct/adapted use of the pinned upstream projects below.
There is not yet a committed dependency lockfile on this branch.

Preserve the applicable license and attribution requirements as implementation evolves:

- **LTX 2.5:** `src/latentslate_engine/ltx25/` narrowly adapts
  ComfyUI commit `1a14b82e7339176357d627c41a262543a6a1356b` (GPL-3.0):
  `comfy/ldm/lightricks/vae/na_diffusion_decoder.py`, per-channel normalization,
  video tile placement/feathering from `comfy/utils.py` and `comfy/sd.py`,
  text-only Gemma 4 and dual projection from `comfy/text_encoders/{gemma4,llama,lt}.py`,
  enhancer vision and prompt/image-token preparation from `comfy_extras/nodes_textgen.py`
  and `comfy/text_encoders/gemma4.py`,
  and rectified-flow Euler ancestral sampling from `comfy/k_diffusion/sampling.py`.
  INT8 activation fusion in the shared LTX operations follows `comfy/ops.py`.
  Direct Torch and comfy-kitchen operations replace Comfy runtime dependencies.
  Source: <https://github.com/Comfy-Org/ComfyUI>.

- **Z-Image Turbo source adaptations:** `src/latentslate_engine/zimage/`
  narrowly adapts ComfyUI commit `1a14b82e7339176357d627c41a262543a6a1356b`
  (GPL-3.0): `comfy/ldm/lumina/model.py`, Qwen3 text conditioning,
  Flux autoencoder decoding, flow scheduling and RES multistep sampling,
  and quantized LoRA patch arithmetic. It uses Torch, comfy-kitchen 0.2.34
  and comfy-aimdo 0.4.15 directly; ComfyUI is not a runtime dependency.
  Source: <https://github.com/Comfy-Org/ComfyUI>.

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

  The Ideogram v4 family under `src/latentslate_engine/ideogram4/` adapts
  ComfyUI commit `1a14b82e7339176357d627c41a262543a6a1356b` (GPL-3.0):
  `comfy/ldm/ideogram4/model.py`, the text-only Qwen3-VL path in
  `comfy/text_encoders/{ideogram4,qwen3vl,llama}.py`, the Flux2 decoder in
  `comfy/ldm/{models/autoencoder,modules/diffusionmodules/model}.py`,
  decoder key conversion in `comfy/diffusers_convert.py`, and the Ideogram
  schedule/dual-model Euler path in `comfy_extras/nodes_ideogram4.py`,
  `nodes_custom_sampler.py`, `comfy/model_sampling.py` and
  `comfy/k_diffusion/sampling.py`. Transformer adapter mapping and quantized
  patch rounding adapt `comfy/lora.py`, `comfy/ops.py`, `comfy/quant_ops.py`
  and `comfy/float.py` from the same commit. Sampling activation and stage scratch
  release follow `comfy/model_patcher.py`, `comfy/model_management.py` and
  `execution.py`. Direct Torch/AIMDO/Kitchen calls replace
  Comfy runtime dependencies. Model weights are downloaded separately under
  their upstream terms and are not included here.

  The SDXL family under `src/latentslate_engine/sdxl/` adapts ComfyUI
  commit `1a14b82e7339176357d627c41a262543a6a1356b` (GPL-3.0):
  the base UNet in `comfy/ldm/modules/diffusionmodules/openaimodel.py`,
  transformer attention in `comfy/ldm/modules/attention.py`, CLIP tokenization,
  weighting and text inference in `comfy/{sd1_clip,sdxl_clip,clip_model}.py`,
  the decoder in `comfy/ldm/modules/diffusionmodules/model.py`, SDXL size and
  latent conditioning in `comfy/{model_base,latent_formats}.py`, and ordinary
  epsilon schedules and Euler/DPM++ 2M sampling in `comfy/model_sampling.py`,
  `comfy/samplers.py` and `comfy/k_diffusion/sampling.py`. Direct Torch calls
  replace the graph and model-management runtime. Weights are not included.

Update this notice when additional adapted source or third-party dependencies enter the
tracked implementation.
