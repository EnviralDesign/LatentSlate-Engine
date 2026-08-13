# Native Wan 2.2 14B image-to-video variants

The native 14B path is a distinct `image_to_video` implementation. It does not
replace or reinterpret the existing `wan22.text_to_video` TI2V-5B tool.

A native tool appears only when one variant names all five validated local roles:

- `pipeline_support`: the exact scheduler/tokenizer/config directory used by the
  proven runtime;
- `transformer_high_noise`;
- `transformer_low_noise`;
- `text_encoder`;
- `vae`.

The support directory is a recipe-only component resource. It is never offered as
a generic model selector and is validated by `plan_wan_i2v_support`, not by the
SafeTensors/GGUF artifact probe. The other four roles remain exact artifact
contracts. Catalog construction fails closed when any role, header, contract,
noise-stage binding, support file, or inventory-owned path is missing or changed.
The Engine never downloads, converts, quantizes, or substitutes a dense model for
this path.

## Built-in Comfy-Org FP8 recipe

`wan-2-2-14b-i2v.image-to-video.comfy-org-fp8` packages the workstation-proven five-role topology
as a cataloged recipe and `wan22-14b-i2v-fp8` as its lean deployment profile. It
uses the exact high/low FP8 files, UMT5 FP8 file, BF16 VAE file, and a filtered
official support directory. Cataloging does not acquire any artifact; the recipe
is available only after every declared local artifact passes the existing support,
SafeTensors-header, and native-adapter checks.

The support directory has no `ResourceSource`: the relevant upstream snapshot
contains additional transformer/text-encoder weights and cannot serve as an exact
acquisition description for the filtered 529,069,044-byte directory. Its upstream
provenance records the actual cache commit
`596658fd9ca6b7b71d5057529bbf319ecbc61d74`, correcting the older local sidecar
revision `596658fd9ca6b7b71d5057522ac03f1f7246e520`. Accordingly, the deployment
plan is useful for exact local closure/size reporting but is deliberately not
remote-provisionable or an installer.

## Resource metadata

A support directory with the exact native support layout is inferred as
`component = "pipeline_support"`. Explicit metadata is also supported:

```toml
# models/wan22/my-support/.latentslate-resource.toml
id = "model:wan22:my-support"
name = "My Wan 14B I2V support"
family = "wan22"
component = "pipeline_support"
format = "directory"
base_model = "wan22-14b-i2v"
```

High/low, UMT5, and VAE artifacts continue to use the component metadata and
stored contracts documented in `WAN22.md`.

## Variant example

```toml
schema_version = 1
key = "wan22.native.production"
name = "Wan 2.2 14B Image to Video"
family = "wan22"
base_tool = "wan22.native_image_to_video"

[recipe]
type = "wan22_i2v_14b"
base_model = "wan22-14b-i2v"
pipeline_support = "model:wan22:my-support"
transformer_high_noise = "model:wan22:smoothmix-high"
transformer_low_noise = "model:wan22:smoothmix-low"
text_encoder = "model:wan22:umt5-convrot"
vae = "model:wan22:wan21-vae"

[optimizations]
keep_pipeline_loaded = false
```

The resulting LatentSlate tool exposes:

- one source image;
- positive and negative prompts;
- frame count (`4k+1`, up to 121);
- width and height (multiples of 16, each up to 1280, total area no larger than
  1280×704);
- steps;
- seed;
- `expert_split` or `diffusers_boundary` stage policy;
- independent high-noise and low-noise guidance.

The default 20-step Comfy split uses guidance `3.5` for both stages. The
guidance-`1` four-step path remains a separate future LoRA preset; it is not
silently applied while native Wan LoRA support is unavailable.

## Reference and acceleration policy

ComfyUI is the primary behavioral reference for artifact topology, stored
quantization, high/low execution, LoRA semantics, and component residency. The
Engine reimplements those lessons around its own recipe, identity, lifecycle, and
protocol boundaries; it does not bundle or reproduce ComfyUI's GPL implementation.

[Wan2GP](https://github.com/deepbeepmeep/Wan2GP) is a secondary optimization
reference. The implementation audited at commit
`7e45fe7e21105807b43f6285827d9ebb5fa72906` expresses its fast Wan video paths as
data-driven accelerator-LoRA profiles: step count, solver and flow shift, guidance
phases, high/low switch thresholds, and phase-specific LoRA multipliers travel as
one profile. The current tree does not define general video modes literally named
`draft` or `super-draft`; LatentSlate should only use labels like those after an
exact artifact-backed preset has passed output and residency validation.

The resulting Engine policy is deliberately composable and fail-closed:

- accelerator LoRAs and their scheduler/guidance/stage schedule must become one
  immutable variant preset, not a generic "fewer steps" toggle;
- TeaCache is not enabled for the proven dual-transformer Wan 2.2 I2V path;
- MagCache remains an experiment because its thresholds are calibrated by model
  and resolution and need visual acceptance on the exact Engine runtime;
- alternate attention, compilation, and additional caching remain independent
  capabilities and are exposed only after their own stored-tensor and output proof;
- cache state must never cross a guidance-phase or high/low model boundary.

Wan2GP is distributed under the
[WanGP Community License 2.0](https://github.com/deepbeepmeep/Wan2GP/blob/main/LICENSE.txt),
which restricts embedding the implementation in paid products, while its
[`mmgp` dependency](https://github.com/deepbeepmeep/mmgp) is GPL-3.0. LatentSlate
Engine therefore uses both as behavioral research only unless separate compatible
permission is established. No Wan2GP or `mmgp` source is copied, linked, vendored,
or required at runtime.

The runtime contract remains conversion-free. `NativeWanI2VRuntime` owns UMT5,
first-frame VAE conditioning, explicit high/low stage residency, scheduler work,
and decode. The tool serializes the returned CPU RGB tensor exclusively through
`encode_rgb_video_tensor`, which converts one frame at a time and publishes the
MP4 atomically.

## Residency and failure behavior

Recipe fingerprints include the support fingerprint and every selected resource
identity. Matching variants reuse one managed native runtime. A changed support
file or artifact identity invalidates execution before model load. Cancellation,
serialization failure, or runtime failure evicts the managed wrapper; a successful
variant with `keep_pipeline_loaded = false` unloads its materialized CPU models
after the job.

The hidden curated base is intentionally absent from `/v1/catalog`. Only an
available data-defined recipe variant is cataloged, so users never see a native
14B tool that lacks the exact local components required to execute it.


A directory is inferred as `pipeline_support` only when all required support files
exist **and no model-weight or weight-index file is present anywhere beneath it**.
A complete dense Diffusers pipeline therefore remains an ordinary model resource.
An explicitly tagged support directory may also contain the runtime's VAE weights;
the support planner still opens only its fixed, bounded config/tokenizer files, while
the recipe binds the separately declared VAE artifact used for execution.

Input validation (including the `4k+1` frame contract and total canvas-area bound)
runs before native model materialization, so a malformed request cannot allocate the
high/low transformers merely to fail later.


The hidden native base does not import Torch, Diffusers, or Comfy Kitchen during a
normal Engine startup when no native recipe variant exists. Runtime dependency
checks occur only when a variant actually selects that base.
