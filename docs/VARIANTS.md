# Model resources and tool variants

LatentSlate Engine keeps its public catalog opinionated while allowing hands-on local
iteration through file-drop resources and TOML variant definitions.

This foundation separates three concerns:

1. **Resources** are model or LoRA files/directories discovered beneath the Engine data root.
2. **Variants** reshape one curated base tool into a different LatentSlate-facing schema.
3. **Runtime capabilities** decide whether a requested model override, LoRA stack, or
   optimization is actually implemented by that model-family adapter.

A variant that requests an unimplemented runtime feature remains inspectable but is
reported unavailable. The Engine does not silently ignore requested optimizations.

## Data layout

```text
LatentSlateEngineData/
├── models/{h3,ltx23,wan22,klein4b,klein9b,custom}/
├── loras/{h3,ltx23,wan22,klein4b,klein9b,custom}/
└── variants/{h3,ltx23,wan22,klein4b,klein9b,custom}/
```

A Diffusers directory containing `model_index.json`, or a supported single model file
(`.safetensors`, `.gguf`, `.ckpt`, `.pt`, `.pth`, or `.bin`), is discovered after an
Engine restart. LoRA files are discovered recursively from their family folder.

Discovered paths are resolved and checked against their owned `models/` or `loras/`
root. A symlink or metadata file that escapes that root is rejected and reported as a
catalog error.

Optional sidecar metadata uses TOML. For a file named `cinematic.safetensors`, place
`cinematic.toml` or `cinematic.safetensors.toml` beside it. A model directory can contain
`.latentslate-model.toml` or `.latentslate-resource.toml`.

```toml
name = "Cinematic movement"
description = "A locally trained motion LoRA"
tags = ["motion", "cinematic"]
trigger_words = ["cinematic movement"]
default_strength = 0.8
```

The Engine derives concise stable IDs from the family-relative path, for example:

```text
model:klein4b:black-forest-labs--flux.2-klein-4b
lora:klein4b:cinematic
```

An explicit lowercase `id` can be declared in sidecar metadata when a path-independent
identity is needed. Run `latentslate-engine resources list` to inspect the IDs available
to variants.

## Minimal variant

A variant wraps one curated Engine tool. When `[inputs]` entries are present, only those
base inputs are exposed. Putting a scalar base input in `[fixed]` removes it from the
LatentSlate schema while still supplying it at execution.

Optional media inputs do not need false sentinels: simply omit them from the variant.
Media inputs cannot be fixed in TOML; expose them or omit them when optional.

```toml
schema_version = 1
key = "klein4b.edit.one_ref"
name = "Klein 4B Image to Image"
family = "klein4b"
base_tool = "flux2_klein4b.image_to_image"
schema_revision = 1

[inputs.prompt]
label = "Prompt"

[inputs.source_image]
label = "Input Image"

[inputs.size]
default = "512x512"
options = ["512x512", "768x768"]

[inputs.seed]
```

Fixed values are validated against the base input type, choice options, and numeric
bounds while the catalog is built. An input cannot be both fixed and exposed, and two
variant inputs cannot target the same base input.

Only fixed keys that are actual base-tool inputs are forwarded by the foundation today.
Additional fixed keys are reserved as runtime parameters and require the family runtime
to advertise `runtime_parameters` support.

## Model, LoRA, and optimization kit

The grammar represents:

- a fixed model resource;
- an exposed model picker constrained by glob-style `allowed` patterns;
- fixed LoRAs;
- optional or required LoRA pickers;
- fixed LoRA strengths;
- selectively exposed strength sliders;
- fixed or exposed creator-facing base-tool parameters;
- composable acceleration and memory-policy requests.

```toml
schema_version = 1
key = "klein4b.edit.cinematic"
name = "Klein 4B Cinematic Edit"
family = "klein4b"
base_tool = "flux2_klein4b.image_to_image"

[model]
resource = "model:klein4b:black-forest-labs--flux.2-klein-4b"

[[loras]]
slot = "cinematic"
resource = "lora:klein4b:cinematic"
strength = 0.8

[[loras]]
slot = "style"
exposed = true
allowed = ["*style*"]
parameter_key = "style_lora"
label = "Style"
strength_exposed = true
strength_key = "style_strength"
strength = 0.7

[optimizations]
attention = "sage_hub"
offload = "group_block"
quantization = "int8"
vae_tiling = "on"
cache = "first_block"
group_offload_blocks = 1
group_offload_use_stream = true
```

A required LoRA selector never includes `none`; if no compatible LoRA exists, the tool
is unavailable. Optional selectors may resolve to `none`. Model selectors are likewise
unavailable when their allowed set resolves to no resources.

A variant's `family` must exactly match the curated base tool's declared family.
Runtime capabilities are mode-aware: an adapter must advertise each concrete model
format, LoRA format, attention backend, offload mode, quantization mode, cache mode,
and compile policy that it actually implements. Broad feature flags are not accepted.

Model repositories that contain sharded component weights (for example a Qwen text
encoder with `model.safetensors.index.json`) are grouped as one component resource.
They remain visible in `/v1/resources` but are excluded from general model selectors.
Allowlist matching is explicitly case-insensitive and slash-normalized on every OS.

GGUF is intentionally stricter than other formats: `quantization = "gguf"` requires
one fixed GGUF model resource. GGUF never appears in an exposed general model picker.

Supported vocabulary is validated strictly:

- attention: `inherit`, `auto`, `native`, `flash`, `flash_hub`, `flash3_hub`,
  `flash4_hub`, `sage`, `sage_hub`, `xformers`, `sol`
- offload: `inherit`, `auto`, `none`, `model`, `sequential`, `group_block`, `group_leaf`
- quantization: `inherit`, `native`, `bf16`, `fp16`, `fp8`, `int8`, `nvfp4`, `gguf`
- VAE tiling/slicing: `inherit`, `auto`, `on`, `off`
- cache: `inherit`, `none`, `prompt`, `media`, `first_block`, `tea`, `easy`

Dependent settings are also checked. For example, group-offload fields require a group
offload mode, compile-specific fields require `compile = true`, and `quantization =
"gguf"` requires a fixed GGUF model resource.

The vocabulary is intentionally broader than the first implemented runtime adapters.
Each curated base tool advertises the features it can honor. Until a family adapter
advertises and implements a feature, a requesting variant is visible but unavailable
with a specific reason.

## Catalog and validation

Use these commands after adding or editing files:

```bash
latentslate-engine resources list
latentslate-engine variants list
latentslate-engine variants validate
```

The HTTP equivalents are:

```text
GET /v1/resources
GET /v1/variants
GET /v1/catalog
```

Enabled variants that pass grammar validation become normal tools in `/v1/catalog`, so
LatentSlate uses its existing schema and job path. Disabled variants remain visible in
`/v1/variants` for authoring observability but do not appear as executable tools.

Resource and variant authoring errors cause `variants validate` to exit nonzero. Runtime
capability gaps do not count as authoring errors; they appear as unavailable tools so the
adapter work can land separately from the grammar.
