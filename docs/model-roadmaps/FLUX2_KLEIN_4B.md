# FLUX.2 Klein 4B optimization roadmap

Last reviewed: **2026-08-11**  
Target workstation: **Windows 11, RTX 5080 16 GB (SM120), 64 GB RAM, Python 3.12**

## Executive decision

The local Klein 4B optimization ladder remains deliberately small:

1. **Reference:** matching first-party BFL BF16 transformer and the same operation-specific
   Comfy component closure.
2. **Recommended incumbent:** matching first-party BFL FP8 transformer through Engine's
   existing stored-weight path.
3. **Next challenger:** matching first-party BFL NVFP4 transformer, with native
   Comfy Kitchen CUDA dispatch proved rather than inferred.
4. **One optional later experiment:** the single published community **Distilled 4B
   INT8 ConvRot** file, only after NVFP4 is accepted or rejected.

Do not add tensorwise INT8, W4A8, W4A4, MXFP8, GGUF, Nunchaku, or another runtime
stack to this tranche. They remain deferred because none is needed to answer the
current product question: whether official NVFP4 materially improves the existing
official FP8 path on an RTX 5080, and whether one native-Kitchen ConvRot artifact
adds anything after that result is known.

The most important parity finding is negative: at official workflow-template commit
[`96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb`](https://github.com/Comfy-Org/workflow_templates/tree/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb),
**no checked-in Klein 4B workflow selects an NVFP4 or INT8 ConvRot transformer**.
Implementation must therefore preserve a matching official Base or Distilled graph and
replace only the transformer. That derived graph is an Engine qualification graph, not
an "official Comfy NVFP4 workflow."

## Evidence labels

- **Verified:** directly stated or encoded by BFL, the official Comfy workflow
  repository, ComfyUI, Comfy Kitchen, or Engine at the pinned revision.
- **External measurement:** a publisher or community timing/memory/quality claim;
  motivation only, never Engine acceptance evidence.
- **Inference:** the product decision in this roadmap, to be validated on the target
  workstation.

## Lineage and operation boundaries

| Lineage / operation | Official Comfy parity settings | Reference boundary |
| --- | --- | --- |
| Distilled T2I | Euler; `Flux2Scheduler`; 4 steps; CFG 1; 1024×1024; Qwen3 4B; full Flux2 VAE | Compare BF16, FP8, NVFP4, and optional ConvRot with only the transformer changed |
| Distilled edit | Same 4-step/CFG-1 schedule; one active reference; a disabled two-reference example; references scaled to one megapixel with nearest-exact | Test one and two references separately; do not call Engine's broader 1–3 reference support official workflow parity |
| Base T2I | Euler; `Flux2Scheduler`; 20 steps; CFG 5; 1024×1024; Qwen3 4B; Base operation shell | Separate Base ladder; never compare against Distilled |
| Base edit | Same 20-step/CFG-5 schedule; one active reference; a disabled two-reference example; full-encoder/small-decoder VAE | Highest-priority matching reference gap before judging Base NVFP4 |

The pinned official graphs are:

- [`image_flux2_klein_text_to_image.json`](https://github.com/Comfy-Org/workflow_templates/blob/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb/templates/image_flux2_klein_text_to_image.json)
  — separate Base and Distilled T2I subgraphs.
- [`image_flux2_klein_image_edit_4b_distilled.json`](https://github.com/Comfy-Org/workflow_templates/blob/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb/templates/image_flux2_klein_image_edit_4b_distilled.json)
  — Distilled FP8 edit; one active reference and a disabled two-reference example.
- [`image_flux2_klein_image_edit_4b_base.json`](https://github.com/Comfy-Org/workflow_templates/blob/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb/templates/image_flux2_klein_image_edit_4b_base.json)
  — Base FP8 edit; one active reference and a disabled two-reference example.

All use core `UNETLoader` / Load Diffusion Model semantics. No community custom node is
part of the parity contract.

## Exact first-party transformer artifacts

All verified BFL 4B model pages identify the line as Apache-2.0 and publicly readable
at review time. Engine's existing Distilled BF16 resource still carries
`requires_auth = true`; that catalog flag should be rechecked during implementation,
but this roadmap does not change resources.

| Line / representation | Exact first-party artifact | Immutable identity | Role and disposition |
| --- | --- | --- | --- |
| Distilled BF16 | [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/flux-2-klein-4b.safetensors), `flux-2-klein-4b.safetensors` | revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`; 7,751,105,712 bytes; SHA-256 `ec3d4e733a771f61c052fb4856c48b336c55eaf2c65487c2a1faeb9bbda7a343` | Matching high-precision reference |
| Distilled FP8 | [`black-forest-labs/FLUX.2-klein-4B-fp8`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B-fp8/blob/5b4408e59397a4a37ccb46afe426d8ed86379441/flux-2-klein-4b-fp8.safetensors), `flux-2-klein-4b-fp8.safetensors` | revision `5b4408e59397a4a37ccb46afe426d8ed86379441`; 4,070,624,520 bytes; SHA-256 `97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6` | Recommended incumbent; Engine header contract `comfy_quant/float8_e4m3fn_global` |
| Distilled NVFP4 | [`black-forest-labs/FLUX.2-klein-4b-nvfp4`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4/blob/286fd2fbb83294d929d5be472620826c28e6085b/flux-2-klein-4b-nvfp4.safetensors), `flux-2-klein-4b-nvfp4.safetensors` | release `286fd2fbb83294d929d5be472620826c28e6085b`; 2,460,413,488 bytes; SHA-256 `d8c5007b6a3bbbdfd38538bbcef5101a55dfde81894f58d2e3c8701cdef3542b`; Xet `6a9e32b8dbe085988e6bc9125053bf1270e1beab9a2ecdd042e1527971e5a7ff` | **Next Experimental challenger** |
| Base BF16 | [`black-forest-labs/FLUX.2-klein-base-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/blob/e1a7c4a3dec9992738f809f2213129bc568630c2/flux-2-klein-base-4b.safetensors), `flux-2-klein-base-4b.safetensors` | release `e1a7c4a3dec9992738f809f2213129bc568630c2`; 7,751,105,712 bytes; SHA-256 `9c5fed22b76baea749d88fc2abe3ad53245e7b21a0d353a762665eea00043b92` | Missing matching Engine Base edit reference |
| Base FP8 | [`black-forest-labs/FLUX.2-klein-base-4B-fp8`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B-fp8/blob/103db268c10d4d3921101b46057671f9ac460da6/flux-2-klein-base-4b-fp8.safetensors), `flux-2-klein-base-4b-fp8.safetensors` | revision `103db268c10d4d3921101b46057671f9ac460da6`; 4,089,498,488 bytes; SHA-256 `44bab3a86fe98b85d21dd2a4729ebdc3ae51fb8a39f76e457e18c724219e6840` | Recommended Base edit incumbent |
| Base NVFP4 | [`black-forest-labs/FLUX.2-klein-base-4b-nvfp4`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-nvfp4/blob/24aaf6182c9fa1ad33aceaee87cf933e5f2d15a8/flux-2-klein-base-4b-nvfp4.safetensors), `flux-2-klein-base-4b-nvfp4.safetensors` | release `24aaf6182c9fa1ad33aceaee87cf933e5f2d15a8`; 2,487,544,776 bytes; SHA-256 `f66faefe951b8eefe0ff3d4082b393fb6dc76d9a34861bd9f4ac54dc3ef381ab`; Xet `e666e592741c9807cb3014b654ab08713fb232f9e482f852f38dc2b0514ea562` | Experimental only after Base BF16 reference exists |

BFL labels the files NVFP4 but does not publish a complete per-layer SafeTensors
header/schema inventory on the model card. Filename, size, and SHA prove byte identity;
they do **not** prove that Engine can materialize every layer. Header-only inspection
must confirm the actual `comfy_quant` marker, packed tensor dtypes, scale tensors,
original shapes, and any pre-quantization scale before a resource contract is authored.

## Exact component closures

A valid A/B changes one transformer and nothing else.

| Closure | Transformer | Text encoder | VAE | Support shell | Declared bytes |
| --- | --- | --- | --- | --- | ---: |
| Distilled FP8 incumbent | Distilled FP8 above | `qwen_3_4b.safetensors`, BF16, 8,044,982,048 bytes, revision `d24c4cf2a0cd98a42f23467e27e3d76ee9438b8e`, SHA `6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a` | `flux2-vae.safetensors`, FP32, 336,213,556 bytes, revision `03d6521e6f6a47396b3f951cbea50f7e6c2f482e`, SHA `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5` | Distilled config/tokenizer/scheduler shell, 15,886,276 bytes at `e7b7dc27...` | 12,467,706,400 |
| Distilled NVFP4 challenger | Distilled NVFP4 above | **same** | **same** | **same** | 10,857,495,368 |
| Distilled ConvRot optional | Community ConvRot below | **same** | **same** | **same** | 12,471,517,664 |
| Base FP8 incumbent | Base FP8 above | same Qwen3 4B BF16 | `full_encoder_small_decoder.safetensors`, FP32, 249,519,092 bytes, revision `a3efc24f613ef42d9428af62fdbd6f5fd8856c4a`, SHA `ea4273f02d1fafbf8e1d1c2cf6018ed8748652eb0bf34f2dd91171f16f15ab62` | Base shell, 15,886,242 bytes at `a3b4f4849157f664bdbc776fd7453c2783562f4d` | 12,399,885,870 |
| Base NVFP4 challenger | Base NVFP4 above | **same** | **same** | **same** | 10,797,932,158 |

The closure totals are storage/download accounting, not VRAM predictions. NVFP4
removes roughly 1.6 GB from these complete closures because the unchanged Qwen encoder
dominates much of the footprint.

## Official workflow parity and the missing NVFP4 graph

No `nvfp4` reference occurs in the pinned official Klein templates. Consequently:

- Distilled NVFP4 T2I must clone the Distilled T2I subgraph: core `UNETLoader`,
  Qwen3 4B, full Flux2 VAE, Euler, four-step `Flux2Scheduler`, CFG 1, 1024².
- Distilled NVFP4 edit must clone the pinned Distilled edit graph and preserve its
  single-reference active path and disabled two-reference example.
- Base NVFP4 edit must clone the pinned Base edit graph: Base support, small decoder,
  Euler, 20 steps, CFG 5, 1024².
- A graph that changes VAE, encoder precision, scheduler, reference preprocessing,
  guidance, dimensions, or loader node is not an NVFP4-versus-FP8 qualification.

The first Engine NVFP4 path should be Distilled T2I, then Distilled one-reference edit.
Base edit follows only after the matching componentized Base BF16 reference is present.

## Current ComfyUI → Comfy Kitchen NVFP4 path

The primary source trail is:

1. Official `UNETLoader` loads the standalone SafeTensors state dictionary.
2. Current ComfyUI
   [`comfy/ops.py`](https://github.com/Comfy-Org/ComfyUI/blob/27bca654eb9a70237d93f56a6ea336ab55f8925d/comfy/ops.py)
   reads each layer's `comfy_quant` metadata and requests the registered quantized
   layout rather than casting the stored weight to a dense dtype.
3. Current ComfyUI
   [`comfy/quant_ops.py`](https://github.com/Comfy-Org/ComfyUI/blob/27bca654eb9a70237d93f56a6ea336ab55f8925d/comfy/quant_ops.py)
   maps `format = "nvfp4"` to packed `torch.uint8`,
   `TensorCoreNVFP4Layout`, group size 16, and recognized scale inputs
   `weight_scale`, `weight_scale_2`, `input_scale`, and `pre_quant_scale`.
4. ComfyUI disables the Kitchen CUDA backend when `torch.version.cuda < 13`.
5. Engine's pinned
   [`comfy-kitchen` 0.2.28](https://github.com/Comfy-Org/comfy-kitchen/tree/75aa2ab6f9f45575205489b9593cf9fe01a57028)
   already implements `TensorCoreNVFP4Layout`: E2M1 packed weights, one FP32
   tensor scale, per-block scale, 16×16 padding, and `scaled_mm_nvfp4`.
6. `TensorCoreNVFP4Layout` requires SM ≥ 10.0 for accelerated matmul. The RTX 5080
   is SM120. The CUDA wheel must include the `cublas` extra; Engine's cu130 extra
   already declares `comfy-kitchen[cublas]==0.2.28`.
7. Unsupported shapes/transposition, missing quantized operands, or a kernel exception
   can dequantize and execute ordinary Torch. A successful image is therefore not
   proof of native NVFP4.

Current Kitchen main is 0.2.30 at
[`83df5cada08aa9253f0ff8c56c70b174eac28336`](https://github.com/Comfy-Org/comfy-kitchen/tree/83df5cada08aa9253f0ff8c56c70b174eac28336).
The required NVFP4 and ConvRot kernels are already present in Engine's 0.2.28 pin, so
an upgrade is not automatically required. Implementation must still run a focused
0.2.28-versus-current source regression review rather than assuming identical edge
behavior.

### Native-dispatch proof

An NVFP4 acceptance record must show all of the following:

- Engine selected `nvidia-cu130`; Torch reports CUDA 13.x.
- the actual CUDA device is the RTX 5080 and capability is `(12, 0)`;
- Kitchen reports the CUDA backend available with `scaled_mm_nvfp4`;
- every inspected NVFP4 layer materializes as a Kitchen quantized tensor with the
  expected packed/scale contract;
- dispatch instrumentation or a focused profiler trace records CUDA
  `scaled_mm_nvfp4` calls during denoising;
- zero unsupported-transpose, missing-operand, kernel-failure, or dequantization
  fallback events occur in the timed region;
- no dense BF16/FP16 transformer copy is produced as a hidden load-time fallback.

Kitchen's backend banner is necessary but insufficient: the layout itself catches
some unsupported cases and can fall back after startup.

## INT8 ConvRot: exact optional candidate

Only one 4B artifact qualifies for the optional research rung:

- publisher: community account `supermind`;
- repository:
  [`supermind/int8_convrot_models`](https://huggingface.co/supermind/int8_convrot_models);
- file:
  [`diffusion_models/flux/flux-2-klein-4b_int8_convrot.safetensors`](https://huggingface.co/supermind/int8_convrot_models/blob/0f9fc3a4a78dd992ec144a0767ee6e19fba6319a/diffusion_models/flux/flux-2-klein-4b_int8_convrot.safetensors);
- immutable upload commit:
  [`0f9fc3a4a78dd992ec144a0767ee6e19fba6319a`](https://huggingface.co/supermind/int8_convrot_models/commit/0f9fc3a4a78dd992ec144a0767ee6e19fba6319a);
- 4,074,435,784 bytes;
- SHA-256 `f2ba417fcd5bb674b3d674bef42f1179244a00ef5dc721318a2972464dfbbb07`.

The publisher maps it to “Flux-2 Klein 4B,” not Base, and the filename contains no
`base`; treat it as **Distilled only**. No Base counterpart was identified. The
repository metadata says `license: other`, while its model card says each artifact
inherits the original model license. That disagreement must be resolved before
redistribution or a built-in automatic download.

This is **community weight evidence**, not BFL or Comfy-Org weight evidence. Its
technical provenance is nevertheless concrete: the publisher says almost all files
were converted from original BF16 with Comfy-Org's
[`quant_int8_convrot.py`](https://github.com/Comfy-Org/comfy-model-tools/blob/1fe341bb8a4e46f161a978b5faa2412d8c39c768/quant_int8_convrot.py)
and `--mseclip`. That official script:

- quantizes selected attention/FFN linear weights and passes other tensors through;
- upcasts the source weight to FP32 for conversion;
- applies group-wise Hadamard rotation;
- chooses 256/64/16 group sizes according to divisibility;
- stores per-channel INT8 scales; and
- writes `comfy_quant` metadata equivalent to
  `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":...}`.

No official workflow template selects this file. Qualification must use the unchanged
Distilled FP8 T2I/edit graph and core `UNETLoader`.

Engine's pinned Kitchen 0.2.28 already supports the corresponding
`TensorWiseINT8Layout`: dynamically row-quantized activations, ConvRot-aware
per-channel weight scales, SM ≥ 7.5, and the CUDA `int8_linear` path. Native proof
requires observed CUDA `int8_linear` dispatch with `convrot=true`, the stored group
size, and zero dequantized `torch.nn.functional.linear` fallback in the timed region.

The ConvRot file is essentially the same size as official FP8. Storage is not a
reason to add it. It remains one optional experiment only if its exact header passes
validation and NVFP4 leaves a creator-visible gap.

## Current Engine truth at `6e29afc`

- Package-owned Distilled FP8 T2I and Base FP8 I2I recipes and exact component
  declarations exist.
- The stored adapter recognizes only the 4B global E4M3 FP8 contract. It does not
  recognize NVFP4 or INT8 ConvRot.
- Stored execution is fixed to native attention, Engine-owned staged residency, no
  `torch.compile`, and no LoRA switching.
- The public Klein tool accepts one to three references, but official 4B templates
  prove one active reference and show two only in a disabled example.
- Distilled complete-folder BF16 exists; a matching componentized Base BF16 edit
  recipe is still missing.
- Engine pins Kitchen 0.2.28; cu130 includes the `cublas` extra required for the native
  NVFP4 path.
- No NVFP4/ConvRot resource declaration, header planner, materializer, recipe, or
  runtime provenance field exists.

## Opinionated status matrix

| Path | Status | Reason |
| --- | --- | --- |
| Matching BFL BF16 | **Reference** | Source of truth within the same Base/Distilled operation |
| Existing official BFL FP8 | **Recommended** | Exact Engine path and official workflow parity already exist |
| Distilled BFL NVFP4 | **Experimental — next** | First-party bytes and native Kitchen kernel exist; exact header/materializer and hardware proof are missing |
| Base BFL NVFP4 | **Experimental — blocked** | Same technical opportunity, but no matching Base BF16 component reference yet |
| Community Distilled INT8 ConvRot | **Experimental — optional after NVFP4** | Official conversion tool and native Kitchen path, but community bytes, license metadata conflict, no official graph, no Base pair |
| Tensorwise INT8 without ConvRot | **Deferred** | Does not answer a product need beyond the selected ConvRot experiment |
| MXFP8, W4A8, W4A4, GGUF, Nunchaku | **Rejected from this tranche** | Additional formats/loaders are not justified before the first-party NVFP4 decision |
| Runtime quantization/conversion | **Rejected** | Violates Engine's stored-artifact policy |

## Implementation-ready gates

### Artifact and header gate

For each candidate:

1. Lock repository, revision, filename, byte count, SHA-256, and source license.
2. Read only the SafeTensors header and reject:
   unknown `comfy_quant` formats; missing/duplicate scales; unexpected dense copies;
   malformed original shapes; unsupported rank; inconsistent scale/group metadata;
   tensor names outside the matching BF16 schema; or a lineage mismatch.
3. Persist a deterministic schema fingerprint covering names, shapes, storage dtypes,
   quant metadata, and scale tensors.
4. Revalidate file size/mtime/hash and schema immediately before materialization.
5. Never quantize, rotate, repack, or save converted weights during normal execution.

NVFP4 additionally requires packed `uint8`, the actual `nvfp4` marker, supported
scale tensors, group size 16, and shapes the Kitchen layout can pad/materialize.
ConvRot requires `int8_tensorwise`, `convrot=true`, valid per-channel scales, and a
supported persisted group size for every quantized linear.

### Loader and residency gate

- Extend a 4B stored-component planner by exact layout, not a generic
  `quantization=` switch.
- Preserve Kitchen quantized tensors through module assignment; fail if a requested
  layer becomes a dense parameter.
- Keep the existing staged policy: prompt encoder first, release/offload it before
  transformer residency, release transformer before VAE decode where possible.
- Record loaded and peak-resident bytes by component and detect accidental duplicate
  host or device copies.
- Poison and evict the runtime after any materialization, CUDA, cancellation, or
  fallback-integrity failure.

### Deterministic A/B gate

Use the pinned official parity graph and hold constant:

- lineage, operation, prompt/negative conditioning, ordered references and hashes;
- seed, effective width/height, scheduler, Euler sampler, steps, CFG;
- Qwen encoder, VAE, support shell, reference scaling, cache policy, and residency;
- Torch, CUDA, Kitchen, driver, and Engine commit.

Minimum pairs:

1. Distilled T2I: BF16 → FP8 → NVFP4.
2. Distilled one-reference edit: BF16 → FP8 → NVFP4.
3. Distilled two-reference edit: BF16 → FP8 → NVFP4, preserving the official disabled
   example's topology when activated.
4. Base one-reference edit: BF16 → FP8 → NVFP4, **only after** the Base BF16
   component reference exists.
5. Optional ConvRot: Distilled FP8 versus accepted NVFP4 versus ConvRot; never compare
   it only against BF16.

### Lifecycle and evidence gate

Use the opt-in API harness defined in [README](./README.md). Record one cold run and
one or two warm runs initially, with:

- load/parse/materialize, text encode, denoise, VAE decode, save, and total timing;
- peak allocated/reserved VRAM, process RSS, system commit/RAM, and transfer bytes;
- quant header summary, layout counts, selected backend counts, and fallback counts;
- output file hashes plus creator-reviewed comparisons;
- prompt/reference cache hit state;
- cancellation during model materialization, denoising, and decode, followed by a
  clean recovery job;
- warm reuse, changed prompt, changed reference set, and explicit teardown to baseline.

Stop immediately on a header mismatch, dense fallback, eager/dequantized hot path,
OOM, NaN/black output, corrupted output, poisoned post-cancel reuse, or obvious
creator-visible regression.

## Promotion threshold

NVFP4 replaces FP8 only when it preserves accepted output quality and provides at
least one material benefit:

- normally **20–25% or more warm end-to-end improvement**;
- a similarly meaningful cold/load or stability improvement; or
- memory savings that unlock a valuable operation/resolution/reference count that
  FP8 cannot run.

A single-digit kernel gain or storage reduction alone is not enough.

ConvRot must beat the **best accepted first-party path**, not merely BF16. Because its
file is the same size as FP8 and adds community provenance risk, reject it after one
bounded experiment unless it produces a clear creator-visible frontier improvement.

## Ordered next actions

1. Add the matching componentized Base BF16 edit reference.
2. Implement header-only NVFP4 validation for the exact Distilled and Base files.
3. Add a layout-specific 4B NVFP4 materializer using Kitchen 0.2.28 and fail closed
   before any timing run.
4. Run the small manual API harness for Distilled T2I and one-reference edit on cu130.
5. Add the two-reference Distilled case after single-reference correctness.
6. Evaluate Base NVFP4 only after the Base BF16 reference passes.
7. Run the single community ConvRot experiment only if NVFP4 leaves a material gap.
8. Stop; do not widen the format matrix without a new creator requirement.

## Source conflicts and implementation blockers

1. **No official NVFP4 or ConvRot workflow JSON.** Parity must be derived from official
   FP8/BF16 graphs and labeled accordingly.
2. **BFL NVFP4 header metadata is not published in the card.** Exact layout
   compatibility remains unverified until header inspection.
3. **Base BF16 reference missing in Engine.** Base NVFP4 quality cannot yet be judged
   scientifically.
4. **Engine adapter is FP8-only.** Kitchen kernels already exist, but Engine has no
   NVFP4/ConvRot planner or materializer.
5. **Kitchen version seam.** Engine pins 0.2.28 while current source is 0.2.30; the
   required kernels exist in 0.2.28, but regression behavior must be reviewed.
6. **ConvRot licensing metadata conflicts.** Repository metadata says `other`; the card
   says inherited source licenses.
7. **Reference-count scope differs.** Engine exposes 1–3 references; official templates
   prove one active reference and only demonstrate two in disabled examples.
8. **BFL auth metadata conflict.** Current public 4B pages are Apache-2.0/readable while
   Engine's Distilled BF16 declaration still requests authentication.

## Primary sources

- [BFL FLUX.2 Klein 4B Distilled BF16](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [BFL FLUX.2 Klein Base 4B BF16](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
- [BFL Distilled FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B-fp8)
- [BFL Base FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B-fp8)
- [BFL Distilled NVFP4](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4)
- [BFL Base NVFP4](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-nvfp4)
- [Official Comfy workflow templates at `96a8cab`](https://github.com/Comfy-Org/workflow_templates/tree/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb)
- [ComfyUI quantized loading at `27bca65`](https://github.com/Comfy-Org/ComfyUI/blob/27bca654eb9a70237d93f56a6ea336ab55f8925d/comfy/quant_ops.py)
- [Comfy Kitchen 0.2.28](https://github.com/Comfy-Org/comfy-kitchen/tree/75aa2ab6f9f45575205489b9593cf9fe01a57028)
- [Official ConvRot conversion tool](https://github.com/Comfy-Org/comfy-model-tools/blob/1fe341bb8a4e46f161a978b5faa2412d8c39c768/quant_int8_convrot.py)
- [Community ConvRot collection at the audited commit](https://huggingface.co/supermind/int8_convrot_models/commit/0f9fc3a4a78dd992ec144a0767ee6e19fba6319a)
- [Engine source at `6e29afc`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/6e29afc907e397f4e57bb02cdcec43b24af9455d)
