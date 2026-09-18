# LatentSlate Engine

LatentSlate Engine is a local inference service for LatentSlate, with a versioned
HTTP catalog, media uploads, asynchronous generation, and downloadable artifacts.

## Model-family roadmap

This is a rough scope tracker, not a release schedule. **Implemented** means the
listed operations are available; it does not imply every variant is supported or
that further compatibility work is finished. **In progress** means active work;
**Planned** means agreed work not yet started; **Candidate** means under discussion.

| Family | Operations / intended scope | Status |
|---|---|---|
| LTX 2.3 | Text-to-video, image-to-video, first/last-frame video | Implemented |
| FLUX.2 Klein 9B | Text-to-image, image editing with 1–3 references | Implemented |
| Wan 2.2 14B Turbo | Text-to-video, image-to-video, first/last-frame video | Implemented |
| Krea 2 Turbo | Text-to-image, optional prompt enhancement | Implemented |
| Qwen Image Edit 2511 | Image editing with one to three input images | Implemented |
| Qwen MetaView | Single-image novel-view synthesis with camera controls | Implemented |
| Ideogram v4 | Text-to-image, INT8/NVFP4, transformer LoRAs, single/dual-model guidance and structured spatial prompts | Implemented |
| Z Image Turbo | Text-to-image, INT8 ConvRot baseline | Implemented |
| SDXL | Text-to-image, ordinary checkpoints, positive/negative prompts, optional VAE override; no refiner | Implemented |
| LTX 2.5 | Text-to-video, image-to-video, first/last-frame video, optional prompt enhancement | Implemented |
| MiniMax H3 | Text-to-video, image-to-video, multimodal reference-to-video, synchronized audio, optional turbo | Implemented |

Additional families and priorities will be added as they are agreed. Hosted API
providers belong in LatentSlate's provider roadmap; inclusion here does not imply
that downloadable weights or a native Engine implementation are available.

## Current implementation

The runtime contains eleven model families: LTX 2.3 under
`src/latentslate_engine/ltx23/`, FLUX.2 Klein 9B under
`src/latentslate_engine/klein9b/`, and Wan 2.2 14B turbo under
`src/latentslate_engine/wan2214b/`, Krea 2 Turbo under
`src/latentslate_engine/krea2/`, and Qwen Image Edit 2511 under
`src/latentslate_engine/qwen2511/`, plus Z-Image Turbo under
`src/latentslate_engine/zimage/`, and Ideogram v4 under
`src/latentslate_engine/ideogram4/`, and SDXL under
`src/latentslate_engine/sdxl/`, plus LTX 2.5 under
`src/latentslate_engine/ltx25/`, and MiniMax H3 under
`src/latentslate_engine/h3/`, plus MetaView under `src/latentslate_engine/metaview/`.
Their first evidence-earned shared request
invariants are described in `docs/ENGINE_ARCHITECTURE.md`; inference, lifecycle,
cache, and artifact ownership otherwise remain family-local. The serving/API layer
now exposes the three stable LTX 2.3 tools, the proven Klein 9B text-to-image
and two-image tools, the three accepted Wan video operations, Krea text-to-image,
Qwen editing, Z-Image, Ideogram and SDXL text-to-image, and the three LTX 2.5
video operations, plus H3 text, image and multimodal reference generation to
LatentSlate. Qwen MetaView Novel View is a built-in under Qwen, with pinned
Hugging Face sources included in bootstrap and Recipe Studio downloads. Adapter controls are available
where supported, with local artifact folder filters.

MetaView requires the Depth Anything 3 inference core at commit
`3fe327a6abe2e5db95b54444ea95463dbfef5610`, plus its geometry preprocessing
dependencies. In the tested Torch 2.11/CUDA 13 environment:

```powershell
uv pip install --python .venv/Scripts/python.exe --no-deps "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3fe327a6abe2e5db95b54444ea95463dbfef5610"
uv pip install --python .venv/Scripts/python.exe einops==0.8.2 omegaconf==2.3.0 opencv-python==4.13.0.92 scipy==1.17.1 addict==2.4.0 imageio==2.37.2
uv pip install --python .venv/Scripts/python.exe --no-deps torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
```

The core uses the existing Torch environment. Its demo, visualization and export
dependency bundle is unnecessary. Model artifacts remain recipe-selected and
outside the repository; see the MetaView section of `docs/ENGINE_CONTRACT.md`.

Krea prompt enhancement is off by default and exposed as a caller toggle. Recipe
Studio can fix it on/off or expose it with a chosen default. It reuses Krea's text
encoder; disabling it encodes the original prompt, followed by any recipe suffix.

LTX 2.5 prompt enhancement also defaults off and is exposed by the built-ins,
but uses a separate model. A recipe that exposes enhancement or fixes it on must
bind that model; a recipe with enhancement fixed off may omit it. Acquire it
through the normal recipe download flow before generation. Image-conditioned
enhancement uses the first input image. Output FPS defaults to 24 and accepts
integers; frame counts follow the model's eight-frame grid plus one.

Qwen Image Edit 2511 is also available through the Engine catalog and job API
with one to three logical input images. The builtin remains FP8mixed; authored
Recipes also certify the official BF16 and INT8 ConvRot files, without adapters.
The curated FP8mixed checkpoint plus the pinned Lightx2v four-step Lightning
adapter at strength 1 is accepted with explicit residency-dependent output
variability; see [`docs/CANONICAL_PARITY_CERTIFICATION.md`](docs/CANONICAL_PARITY_CERTIFICATION.md).
See the Qwen section of
[`docs/ENGINE_CONTRACT.md`](docs/ENGINE_CONTRACT.md) for model bindings and inputs.

For Hugging Face artifact pinning/materialization, install the lightweight
official client in the Engine environment: `python -m pip install huggingface_hub==1.27.0`.
Public files need no token; private/gated repositories use the host's `HF_TOKEN`
or normal Hub login. In Recipe Studio, choose **Hugging Face** on a file slot,
choose **Check source**, apply the online reference, then **Download missing files**. Import never starts
downloads automatically. See the authoring contract below for cache and task
semantics. Inference dependencies are unchanged.

Civitai files use the same cache and explicit download flow. Choose **Civitai**,
paste the link copied from the Download button, enter a model-version ID, or use
a model-page URL containing `modelVersionId`. A download link with `fileId`
selects that exact variant; otherwise inspect the available files and choose one
before pinning. The API also accepts explicit `model_version_id` and `file_id`.
Public files can work anonymously; authenticated
downloads use the host's `CIVITAI_TOKEN` as a Bearer header. No additional client
dependency is needed, and source tokens never enter recipe JSON or the browser.

**Manage downloads** in Recipe Studio prepares multiple saved recipes together.
Checkboxes start from enabled recipes and can be overridden without changing
publication or recipe definitions. The preview counts shared files once, uses
official bootstrap sources for built-ins, and separates missing local references
from downloads. **Download now** shows file and byte progress; cancellation keeps
completed files and **Retry remaining** rechecks what is still missing. Preview
checks presence without hashing large existing files; acquired files are verified.

## Bootstrap built-in models

From an installed Engine Python environment, preview the official dependencies:

```powershell
python -m latentslate_engine.bootstrap plan --home M:\LatentSlateEngineData
python -m latentslate_engine.bootstrap install --home M:\LatentSlateEngineData
```

Use `--family ltx23`, `--family flux2_klein9b`, `--family wan2214b`,
`--family krea2`, `--family qwen2511`, `--family zimage`, `--family ideogram4`,
`--family sdxl`, or `--family ltx25`
to select families (repeatable).
Planning is offline and does not download. `plan --verify` hashes existing
files; installation always verifies and refuses to overwrite conflicting files.
Re-running installation reuses verified assets and needs no network when complete.

`src/latentslate_engine/builtin-assets.json` freezes official source revisions,
SHA-256 digests, lengths and destination paths. It includes only the assets used
by the built-in defaults, including tokenizer files and Klein's encoder config.
Weights and most support files come from Hugging Face; the Qwen/Krea/Z-Image/Ideogram tokenizer
files come from a pinned official Comfy source revision to preserve tokenization.
No community/custom artifacts are selected by bootstrap.

Bootstrap downloads each built-in weight directly to its canonical model path.
Recipes sharing that weight resolve to the same file, without filesystem links
or a second cache copy. Small tokenizer support files are ordinary copies where
separate tokenizer directories require them. Other downloaded artifacts live
once under `artifacts/sha256/`. Local recipe references use their original paths;
bootstrap never imports or copies those files. No storage cleanup is performed.

Set `LATENTSLATE_ENGINE_HOME` to the same home when starting the service. Explicit
family path overrides (`LATENTSLATE_KLEIN9B_VAE`, `LATENTSLATE_WAN_MODEL_ROOT`,
`LATENTSLATE_KREA2_MODEL_ROOT`, `LATENTSLATE_QWEN2511_MODEL_ROOT`) still take
precedence; remove those overrides if the service should use the bootstrapped
defaults. Existing user recipes retain their explicit references.

Source credentials are host configuration: set `HF_TOKEN` and `CIVITAI_TOKEN`
in the Engine process environment or its gitignored `.env`, then restart Engine
through the Process Manager. Normal Hugging Face login is also supported. Recipe
Studio reports whether authentication is configured; it does not store source
keys in recipes or the browser. Gated downloads require the account to have
accepted the upstream repository's access terms. Do not put tokens in URLs.

Start with:

- [`AGENTS.md`](AGENTS.md)
- [`docs/GREENFIELD_RESET.md`](docs/GREENFIELD_RESET.md)
- [`docs/ENGINE_ARCHITECTURE.md`](docs/ENGINE_ARCHITECTURE.md)
- [`docs/ENGINE_CONTRACT.md`](docs/ENGINE_CONTRACT.md)
- [`docs/COMFY_REFERENCE.md`](docs/COMFY_REFERENCE.md)
- [`docs/LTX23_TARGET.md`](docs/LTX23_TARGET.md)
- [`docs/KLEIN9B_TARGET.md`](docs/KLEIN9B_TARGET.md)
- [`docs/WAN2214B_TARGET.md`](docs/WAN2214B_TARGET.md)
- [`docs/CANONICAL_PARITY_CERTIFICATION.md`](docs/CANONICAL_PARITY_CERTIFICATION.md)

The pre-reset implementation remains recoverable at the annotated Git tag
`ltx23-pre-greenfield-reset-2026-08-26`
(`86419a7b943a2dcd9a172c817aafb3f05728331d`). It is a historical checkpoint,
not the architecture for this rebuild.
