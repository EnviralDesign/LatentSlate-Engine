# FLUX.2 Klein 9B roadmap

Last reviewed: **2026-08-11**  
Target workstation: **Windows 11, RTX 5080 16 GB (SM120), 64 GB RAM, Python 3.12**

## Executive decision

FLUX.2 Klein 9B remains a high-value next local tranche, but it is not one
interchangeable model and it is not yet a 16 GB product path.

Keep three lines separate:

1. **Distilled 9B:** four-step text-to-image and ordinary one/multi-reference editing.
   This is the next Engine tranche.
2. **Base 9B:** undistilled foundation line. It has its own schedule, references, and
   training/fine-tuning value; defer its quantized product path.
3. **9B-KV:** a Distilled-derived editing line with reference K/V reuse. It is a
   separate repeated-reference experiment, not a generic faster 9B default.

The small product ladder is:

- matching first-party Distilled BF16 as reference;
- official first-party Distilled FP8 as the incumbent candidate;
- official Distilled NVFP4 as **one later challenger only if FP8 leaves a measured
  memory or performance gap**.

KV receives a separate BF16-versus-FP8 repeated-reference ladder after ordinary
Distilled editing is stable. No first-party KV NVFP4 artifact was verified. Community
9B ConvRot, GGUF, W4, Nunchaku, mixed-precision encoders, and other format branches are
outside this tranche.

Feasibility on RTX 5080 16 GB plus 64 GB RAM remains unproven. The raw FP8 component
closures are below 19 GB, but load-time copies, the Qwen encoder, activations, prompt
and media state, Windows commit charge, and decode buffers can still fail the host or
device envelope.

## Evidence labels

- **Verified:** directly stated or encoded by BFL, the official Comfy workflow
  repository, ComfyUI, Comfy Kitchen, or Engine at a pinned revision.
- **Corroborated identity:** an exact public LFS/Xet pointer matching the official
  filename when the gated first-party repository did not expose an anonymous immutable
  file commit. It is not an implementation lock.
- **External measurement:** publisher speed/VRAM/quality claims, not Engine results.
- **Inference:** the product judgment in this roadmap.

## Lineage and operation boundaries

| Line | Operations | Canonical parity | Product treatment |
| --- | --- | --- | --- |
| Distilled 9B | T2I; ordinary single/multi-reference edit | Four steps, CFG 1 for the checked-in edit graph | **Next local tranche** |
| Base 9B | T2I; ordinary editing; foundation/fine-tuning | Official Comfy graphs use 20 steps, CFG 5; BFL Diffusers examples may use different full-model settings | Separate line, deferred after Distilled |
| 9B-KV | Repeated-reference editing; model also retains broader capabilities | Four steps, CFG 1; references cached after step 0 | Separate experiment after ordinary edit |

Never use a Distilled result as a Base reference, a Base result as a Distilled
reference, or an ordinary 9B result as a KV cache-lifecycle result.

All first-party 9B lines are gated under the **FLUX Non-Commercial License**. BFL
requires inference filters or manual review for 9B use. Product/distribution and
moderation requirements must be reviewed before a built-in recipe is shipped.

## Official Comfy workflow parity

Primary workflow source:
[`Comfy-Org/workflow_templates@96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb`](https://github.com/Comfy-Org/workflow_templates/tree/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb).

| Pinned workflow | Selected transformer and components | Operation/settings | Reference semantics |
| --- | --- | --- | --- |
| [`image_flux2_text_to_image_9b.json`](https://github.com/Comfy-Org/workflow_templates/blob/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb/templates/image_flux2_text_to_image_9b.json) | **Base FP8** `flux-2-klein-base-9b-fp8.safetensors`; `qwen_3_8b_fp8mixed.safetensors`; `full_encoder_small_decoder.safetensors`; core `UNETLoader` | T2I; Euler; `Flux2Scheduler`; 20 steps; CFG 5; 1024×1024 | No references |
| [`image_flux2_klein_image_edit_9b_distilled.json`](https://github.com/Comfy-Org/workflow_templates/blob/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb/templates/image_flux2_klein_image_edit_9b_distilled.json) | Distilled FP8 `flux-2-klein-9b-fp8.safetensors`; Qwen 8B FP8-mixed; small decoder; core `UNETLoader` | Ordinary edit; Euler; 4 steps; CFG 1; 1024×1024 | One active reference; disabled two-reference example |
| [`image_flux2_klein_image_edit_9b_base.json`](https://github.com/Comfy-Org/workflow_templates/blob/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb/templates/image_flux2_klein_image_edit_9b_base.json) | Base FP8 `flux-2-klein-base-9b-fp8.safetensors`; Qwen 8B FP8-mixed; small decoder; core `UNETLoader` | Ordinary Base edit; Euler; 20 steps; CFG 5; 1024×1024 | One active reference; disabled two-reference example |
| [`image_flux2_klein_9b_kv_image_edit.json`](https://github.com/Comfy-Org/workflow_templates/blob/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb/templates/image_flux2_klein_9b_kv_image_edit.json) | KV FP8 `flux-2-klein-9b-kv-fp8.safetensors`; Qwen 8B FP8-mixed; **full** `flux2-vae.safetensors`; core `UNETLoader` plus core `FluxKVCache` | KV edit; Euler; 4 steps; CFG 1; 1024×1024 | Two active reference images |

Important negative findings:

- The pinned 9B T2I JSON selects **Base**, not Distilled.
- Distilled FP8 appears in template download guidance, but no dedicated checked-in
  Distilled 9B T2I graph was identified at this commit. A Distilled T2I Engine graph
  must preserve the four-step Distilled operation contract and be labeled derived.
- No checked-in 9B NVFP4 graph was identified.
- The ordinary Distilled/Base edit templates prove one active reference and show two
  only in disabled examples. The KV graph proves two active references.
- All selected loaders are core nodes; community custom nodes are not parity evidence.

### KV cache semantics

Current core
[`FluxKVCache`](https://github.com/Comfy-Org/ComfyUI/blob/27bca654eb9a70237d93f56a6ea336ab55f8925d/comfy_extras/nodes_flux.py)
is explicitly experimental. It computes reference K/V on the first applicable model
call, reuses cached K/V on later denoising calls, and removes the reference tokens from
subsequent model input. The checked-in behavior uses the `index_timestep_zero` reference
method.

This is model-object state, not an automatically safe cross-job cache. An Engine KV
implementation needs an explicit cache key over transformer identity, ordered reference
hashes, preprocessing, dimensions, and relevant model configuration; changed
references, model, dimensions, or operation must invalidate it. First-generation and
reuse-generation timings must be reported separately.

## First-party and Comfy artifact surface

### Transformer artifacts

| Line / representation | Repository and exact file | Identity status | Disposition |
| --- | --- | --- | --- |
| Distilled BF16 | [`black-forest-labs/FLUX.2-klein-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B), complete Diffusers repository plus standalone `flux-2-klein-9b.safetensors` (~18.2 GB) | First-party, gated NCL. Anonymous audit could not obtain a first-party immutable file commit/byte pointer. Public matching pointers report SHA-256 `0975d6b77b5f510b99547d6724a208e36527df654e8f6134f59ece3f9f30da58`; re-resolve after authenticated acceptance. | **Reference for Distilled** |
| Base BF16 | [`black-forest-labs/FLUX.2-klein-base-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B), complete Diffusers repository plus `flux-2-klein-base-9b.safetensors` (~18.2 GB) | First-party, gated NCL. Public matching pointers report SHA-256 `4a54fad7f5f741b99eee217198daac20b8d8e515e2a1f5b064fd51cf074f95bd`; authenticated immutable lock still required. | **Reference for Base only** |
| KV BF16 | [`black-forest-labs/FLUX.2-klein-9b-kv`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv), complete Diffusers repository plus `flux-2-klein-9b-kv.safetensors` | First-party, gated NCL. Exact anonymous pointer/revision was not resolved in this audit. | **Reference for KV only** |
| Distilled FP8 | [`black-forest-labs/FLUX.2-klein-9b-fp8`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8), `flux-2-klein-9b-fp8.safetensors` | First-party gated artifact. Corroborated pointer: 9,433,061,528 bytes; SHA-256 `865ba09f5b4c3cbd3468a4bd3acb9fcb2f8740c54317482f0bcd4ed1d3655cee`. Official immutable file commit must be re-resolved with credentials. | **Experimental incumbent candidate** |
| Base FP8 | [`black-forest-labs/FLUX.2-klein-base-9b-fp8`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8), `flux-2-klein-base-9b-fp8.safetensors` | First-party gated artifact. Corroborated pointer: 9,567,278,472 bytes; SHA-256 `a9f5028c24a7a96f4f45beb883aad287d9bccc246227a6803edc898ddda42cf4`. Official immutable file commit still required. | **Deferred Base line** |
| KV FP8 | [`black-forest-labs/FLUX.2-klein-9b-kv-fp8`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv-fp8/blob/a4d584032d9be4310b40531bc76b6b8398eba2c5/flux-2-klein-9b-kv-fp8.safetensors), `flux-2-klein-9b-kv-fp8.safetensors` | First-party upload commit `a4d584032d9be4310b40531bc76b6b8398eba2c5`; 9,818,935,984 bytes; SHA-256 `33f7da5625a00798349a719742999d3c7dd20c1a7eda14663922c363640728f1`; Xet `b8763ddd83d92fb7592fdb30153fd46521dc7b610e919e20159825964f4711c7` | **Separate KV experiment** |
| Distilled NVFP4 | [`black-forest-labs/FLUX.2-klein-9b-nvfp4`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4), `flux-2-klein-9b-nvfp4.safetensors` | First-party gated artifact. Corroborated pointer: 5,760,960,048 bytes; SHA-256 `5c72214496dd278f721a112e1bd1585fffed487bc0831c894bcbf30d12e9ee48`. Official immutable file commit still required. | **One later challenger** |
| Base NVFP4 | [`black-forest-labs/FLUX.2-klein-base-9b-nvfp4`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-nvfp4), `flux-2-klein-base-9b-nvfp4.safetensors` | First-party gated artifact. Corroborated pointer: 5,809,193,808 bytes; SHA-256 `730d6bdbd5069cd4cd263cfdc4801d0d06ca14457b903baa5d953a8c2f9e84c9`. Official immutable file commit still required. | **Deferred Base line** |
| KV NVFP4 | No first-party artifact verified | Community mixed artifacts exist but are outside this tranche | **Rejected from current ladder** |

Corroborated identities are useful for detecting source drift, but they must not appear
in a production resource declaration until an authenticated first-party revision
resolves to the same filename, bytes, and SHA.

### Shared Comfy components

The ordinary Base/Distilled edit and Base T2I graphs use:

- [`qwen_3_8b_fp8mixed.safetensors`](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/f5fad177c9453d2ee329cdd272418127cdfbce92/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors):
  Comfy-Org upload commit `f5fad177c9453d2ee329cdd272418127cdfbce92`;
  8,664,848,742 bytes; SHA-256
  `abad16806e0cbabc54e0325d6565847443fe396d5f0be38bb3cd3fe75a1201d6`.
- [`full_encoder_small_decoder.safetensors`](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/blob/a3efc24f613ef42d9428af62fdbd6f5fd8856c4a/full_encoder_small_decoder.safetensors):
  249,519,092 bytes; SHA-256
  `ea4273f02d1fafbf8e1d1c2cf6018ed8748652eb0bf34f2dd91171f16f15ab62`.
- a matching tokenizer/config/scheduler shell that still needs an exact Engine
  acquisition manifest.

The KV graph instead uses the same Qwen FP8-mixed encoder plus the full
`flux2-vae.safetensors` selected by the template. For a production lock, use the exact
template-resolved VAE origin/revision; do not substitute the ordinary small decoder.

Minimum weight footprints before support metadata:

| Candidate closure | Transformer + Qwen + VAE | Bytes |
| --- | --- | ---: |
| Distilled FP8 ordinary T2I/edit | Distilled FP8 + Qwen FP8-mixed + small decoder | 18,347,429,362 |
| Base FP8 ordinary T2I/edit | Base FP8 + Qwen FP8-mixed + small decoder | 18,481,646,306 |
| KV FP8 edit | KV FP8 + Qwen FP8-mixed + full Flux2 VAE | 18,819,998,282 |
| Distilled NVFP4 ordinary T2I/edit | Distilled NVFP4 + Qwen FP8-mixed + small decoder | 14,675,327,882 |
| Base NVFP4 ordinary T2I/edit | Base NVFP4 + Qwen FP8-mixed + small decoder | 14,723,561,642 |

These are storage/weight-closure figures, not peak RAM or VRAM. The official Comfy
repository also publishes BF16 and FP4-mixed Qwen variants. The first tranche must
use the exact FP8-mixed encoder selected by the parity graphs rather than introducing
an encoder-format A/B at the same time.

## Current Comfy/Kitchen runtime support

The relevant official source is current ComfyUI
[`27bca654eb9a70237d93f56a6ea336ab55f8925d`](https://github.com/Comfy-Org/ComfyUI/tree/27bca654eb9a70237d93f56a6ea336ab55f8925d)
and Engine's pinned Kitchen
[`0.2.28`](https://github.com/Comfy-Org/comfy-kitchen/tree/75aa2ab6f9f45575205489b9593cf9fe01a57028).

- Core `UNETLoader` plus Comfy's quantized state-dict path can materialize registered
  FP8/NVFP4/INT8 layouts without a custom node.
- Kitchen 0.2.28 already provides native FP8, NVFP4, and INT8 primitives. The next 9B
  tranche does not require a new external runtime merely because the model is larger.
- Engine's implementation is the blocker: its standalone stored planner is explicitly
  restricted to Klein 4B global E4M3 FP8 and rejects standalone 9B SafeTensors.
- The generic Engine 9B path loads a complete Diffusers repository and always applies
  Distilled four-step/CFG-1 semantics. It cannot truthfully load Base or KV.
- Current ComfyUI disables Kitchen CUDA when Torch reports CUDA below 13. The target
  qualification tier should therefore remain cu130, even though an individual format's
  hardware floor may be lower.
- NVFP4 requires SM ≥ 10.0 and native `scaled_mm_nvfp4`; the RTX 5080 is SM120.
- A transformer that fits on disk or even in VRAM does not prove that the encoder,
  activations, reference latents, cache, and decoder fit together.

## Realistic 16 GB / 64 GB staging hypothesis

This is an implementation hypothesis, not proof:

1. Validate all resource identities and headers without loading tensors.
2. Load/execute Qwen on CPU or a bounded staged device policy; materialize prompt
   conditioning, then release its device residency before transformer denoising.
3. Keep the stored transformer CPU-backed and onload only the required layer/block
   residency under a deterministic policy; avoid a second dense or quantized host copy.
4. Keep prompt and reference caches explicitly byte-bounded.
5. Release transformer residency before VAE decode.
6. Record Windows process commit and system commit in addition to RSS; temporary
   SafeTensors/module assignment copies can exceed 64 GB even when the unique weights
   do not.
7. Treat any paging-thrash path whose warm time is not creator-usable as failed, even
   if it eventually produces an image.

The first proof target is Distilled FP8 T2I at 1024², followed by one-reference
Distilled edit. Do not start with KV, Base, three references, or NVFP4.

## Current Engine truth at `6e29afc`

- Direct Klein 9B T2I and I2I tools exist.
- The direct path is a complete BF16 Diffusers folder with Distilled four-step/CFG-1
  semantics and model offload/cuda profiles.
- `klein9b-basic` names the first-party Distilled repository but has no immutable
  revision.
- No package-owned 9B recipe, resource declaration, deployment profile, or exact
  component closure exists.
- Standalone SafeTensors execution, stored FP8/NVFP4 materialization, Base semantics,
  KV semantics, and `FluxKVCache` lifecycle are absent.
- I2I remains unaccepted on the target workstation.
- Engine pins Kitchen 0.2.28, which contains the needed low-bit kernels; the missing
  work is Engine schema/materialization/staging/provenance, not format availability.
- No accepted output, cold/warm timing, cancellation, teardown, or 16 GB/64 GB memory
  record exists.

**Proof level: Direct tool only; Distilled BF16 structure implemented, target-hardware
qualification incomplete.**

## Opinionated status matrix

| Path | Status | Reason |
| --- | --- | --- |
| Authenticated, immutable Distilled BF16 repository | **Reference** | Matching first-party source for ordinary Distilled T2I/edit |
| Bounded Distilled BF16 Engine recipe | **Experimental correctness path** | Required to prove operation and staging before quantized promotion |
| Official Distilled FP8 + exact Comfy closure | **Experimental incumbent candidate** | Smallest useful stored-weight next tranche |
| Official Distilled NVFP4 | **Deferred next challenger** | Consider only after FP8 is native, accepted, and still leaves a material gap |
| Base BF16 | **Reference for Base only** | Separate operation/foundation line |
| Base FP8/NVFP4 | **Deferred** | No current product need justifies a second 9B line |
| KV BF16/FP8 | **Separate Experimental line** | Valuable only for repeated-reference editing with explicit cache lifecycle |
| Community KV NVFP4 / ConvRot / GGUF / W4 / Nunchaku | **Rejected from this tranche** | Not needed before first-party ordinary FP8 proof |
| Hosted BFL API | **Fallback** | Cloud access path, not local artifact qualification |
| Recommended local 9B path | **None** | Feasibility, license, acquisition, and output proof remain incomplete |

## Minimum prerequisites for the next Engine tranche

### Resource prerequisites

1. Accept the BFL gate and resolve an immutable Distilled BF16 repository revision,
   complete file inventory, sizes, and hashes.
2. Resolve the official Distilled FP8 file at an immutable first-party revision and
   verify it matches the corroborated 9,433,061,528-byte SHA.
3. Pin Qwen 8B FP8-mixed, small decoder, tokenizer/config/scheduler shell, and every
   acquisition credential/term.
4. Represent BF16 and FP8 as separate exact recipes; no runtime dtype conversion.
5. Add an authenticated license/filter product decision before built-in distribution.

### Loader/runtime prerequisites

1. Add a 9B architecture/schema fingerprint and standalone SafeTensors planner.
2. Materialize only the exact official FP8 layout through Kitchen quantized tensors;
   fail on unknown metadata, unsupported layers, dense copies, or fallback.
3. Implement deterministic staged encoder → transformer → decoder residency with
   byte accounting and poisoned-runtime eviction.
4. Preserve separate T2I and ordinary edit operation contracts. Support one reference
   first; add the official two-reference topology only after one-reference acceptance.
5. Record backend/layout/fallback provenance in every output.
6. Keep Base and KV code paths absent until the ordinary Distilled path passes.

### Harness prerequisites

Use the manual non-CI API harness in [README](./README.md):

- deterministic prompt/asset/seed/settings;
- one recipe or an ordered BF16→FP8 A/B sequence;
- one cold plus one or two warm runs initially;
- public job submission and polling;
- output hashes plus timing/memory/runtime provenance;
- one cancellation point and a required clean recovery job;
- immediate stop on mismatch, fallback, OOM, corrupted output, or poisoned reuse.

This is sufficient for the tranche; do not build a benchmark service.

## Qualification ladders

### Ordinary Distilled T2I/edit

1. **Reference:** exact authenticated Distilled BF16 closure.
2. **Incumbent candidate:** exact official Distilled FP8 closure using the same
   operation, Qwen, VAE, support, prompt/assets, and schedule.
3. **One later challenger:** Distilled NVFP4 only if FP8 cannot meet the target
   envelope or leaves a material measured opportunity.

Minimum progression:

- T2I 1024²;
- one-reference edit;
- two-reference edit matching the disabled official topology when activated;
- varied dimensions only after 1024² acceptance.

### Repeated-reference KV editing

Deferred until ordinary edit passes.

1. KV BF16 reference.
2. KV FP8 candidate.
3. No NVFP4 or community format in this tranche.

Measure separately:

- first generation, when reference K/V is created;
- repeated prompts with identical ordered references;
- changed-reference invalidation;
- cancellation during cache creation and reuse;
- teardown and memory return.

Do not report a cached second job as model-wide speedup.

## Acceptance and material-win rules

Use fixed creator-reviewed cases for typography, fine texture, photorealism, geometry,
identity-preserving edits, object/style transfer, and minimal-change edits. Hold
constant lineage, operation, prompt, negative conditioning, ordered assets, seed,
dimensions, Euler sampler, `Flux2Scheduler`, steps, CFG, Qwen, VAE, support, and
residency policy.

Record phase time, cold/warm state, VRAM allocated/reserved, process RSS, Windows
commit/system RAM, disk/PCIe transfer, backend dispatch, fallback counts, output hashes,
cancellation, reuse, and teardown.

FP8 becomes Recommended only if it completes an accepted creator workload reliably
inside the target envelope. A later NVFP4 loader requires accepted quality plus
normally a **20–25% warm end-to-end win**, a comparable cold/stability improvement,
or a creator-relevant workload that FP8 cannot run. Vendor family claims and storage
savings alone do not pass.

## Ordered next actions

1. Resolve authenticated immutable Distilled BF16 and FP8 source locks.
2. Pin the exact Comfy Qwen/small-decoder/support closure.
3. Implement a bounded Distilled BF16 recipe and prove T2I staging on 16 GB/64 GB.
4. Diagnose and accept one-reference BF16 editing.
5. Add one exact 9B FP8 stored planner/materializer and prove native dispatch.
6. Run the small manual BF16→FP8 API A/B.
7. Add two-reference ordinary editing only after one-reference stability.
8. Evaluate KV BF16/FP8 only for repeated-reference creator demand.
9. Consider Distilled NVFP4 last and only once.
10. Stop; do not add community or additional quantization branches.

## Source conflicts and blockers

1. **Gated revision gap:** several first-party 9B files expose model cards but not
   anonymous immutable file commits. Corroborated LFS/Xet identities must be re-locked
   after authenticated access.
2. **T2I graph mismatch:** the checked-in official 9B T2I graph selects Base FP8;
   no dedicated selected Distilled T2I graph was found at the pinned commit.
3. **Base schedule context:** official Comfy uses 20 steps/CFG 5, while BFL Diffusers
   examples may show different full-model settings. Use the Comfy JSON for Comfy-aligned
   parity; do not claim one universal Base schedule.
4. **Reference-count scope:** ordinary graphs prove one active reference and show two
   disabled; KV proves two active references.
5. **KV VAE differs:** the KV graph uses the full Flux2 VAE while ordinary 9B edit uses
   the small decoder.
6. **Mutable Comfy acquisition URLs:** templates use `resolve/main`; implementation
   must resolve the actual repository and immutable revision.
7. **VRAM claims conflict by context:** BFL cards report about 29 GB, while other BFL
   overview numbers and community reports use different residency assumptions. Do not
   reconcile them without one harness.
8. **Engine stored-loader block:** the current adapter rejects standalone 9B weights.
9. **License/moderation block:** NCL gate and filter/manual-review obligations need a
   product decision.
10. **Feasibility block:** no 16 GB/64 GB output, lifecycle, or paging-thrash proof exists.

## Explicit non-goals

- No Base quantized product tranche.
- No KV as a generic recommended path.
- No 9B ConvRot, GGUF, MXFP8, W4A8, W4A4, Nunchaku, or runtime conversion.
- No encoder-format shootout; use the FP8-mixed Qwen selected by official graphs.
- No three-reference claim until an official or deliberately product-authored topology
  is separately accepted.
- No commercial/local availability claim before NCL review.
- No benchmark framework or CI inference.

## Primary sources

- [BFL Distilled 9B BF16](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
- [BFL Base 9B BF16](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)
- [BFL 9B-KV BF16](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv)
- [BFL Distilled FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8)
- [BFL Base FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8)
- [BFL KV FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv-fp8)
- [BFL Distilled NVFP4](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4)
- [BFL Base NVFP4](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-nvfp4)
- [Official Comfy workflow templates at `96a8cab`](https://github.com/Comfy-Org/workflow_templates/tree/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb)
- [Official Comfy 9B encoder/VAE repository](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b)
- [ComfyUI current quantized loading](https://github.com/Comfy-Org/ComfyUI/tree/27bca654eb9a70237d93f56a6ea336ab55f8925d)
- [Comfy Kitchen 0.2.28](https://github.com/Comfy-Org/comfy-kitchen/tree/75aa2ab6f9f45575205489b9593cf9fe01a57028)
- [Engine source at `6e29afc`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/6e29afc907e397f4e57bb02cdcec43b24af9455d)
