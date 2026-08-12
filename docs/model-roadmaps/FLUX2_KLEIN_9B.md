# FLUX.2 Klein 9B roadmap

Last reviewed: **2026-08-12**
Target workstation: **Windows 11, RTX 5080 16 GB (SM120), 64 GB RAM, Python 3.12**

## Executive decision

FLUX.2 Klein 9B now has a complete ordinary-Distilled Engine ladder that mirrors
the useful 4B product surface. It is still not one interchangeable model; the
quantized ordinary paths are workstation-output accepted, while complete BF16
Reference remains a source-of-truth contract that exceeds this 16 GB GPU envelope.

Keep three lines separate:

1. **Distilled 9B:** four-step text-to-image and ordinary one/multi-reference editing.
   Its Recommended/Fallback recipes have passed the initial workstation output
   suite; cancellation, recovery, and multi-reference lifecycle acceptance remain.
2. **Base 9B:** undistilled foundation line. It has its own schedule, references, and
   training/fine-tuning value; defer its quantized product path.
3. **9B-KV:** a Distilled-derived editing line with reference K/V reuse. It is a
   separate repeated-reference experiment, not a generic faster 9B default.

The opinionated ordinary-Distilled ladder is:

- matching first-party Distilled BF16 as **Reference**;
- official first-party Distilled NVFP4 as **Recommended** on qualified Blackwell;
- official first-party Distilled FP8 as the non-Blackwell **Fallback**.

KV receives a separate BF16-versus-FP8 repeated-reference ladder after ordinary
Distilled editing is stable. No first-party KV NVFP4 artifact was verified. Community
9B ConvRot, GGUF, W4, Nunchaku, mixed-precision encoders, and other format branches are
outside this tranche.

The recipes, immutable acquisition locks, exact header contracts, stored transformer
materializers, mixed Qwen3-8B loader, native Kitchen micro-dispatch, staging, and
catalog/profile closure are implemented. End-to-end feasibility on RTX 5080 16 GB
plus 64 GB RAM is accepted for the ordinary FP8/NVFP4 paths. NVFP4 stores about 14.7 GB of model payload across
the transformer, mixed Qwen, and small decoder; FP8 stores about 18.35 GB. Sequential
component residency is therefore essential even though no single stage needs all
payload resident on CUDA at once.

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
| Distilled 9B | T2I; ordinary single/multi-reference edit | Four steps, CFG 1 for the checked-in edit graph | **NVFP4/FP8 initial acceptance complete; lifecycle/multi-reference pending** |
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
| Distilled FP8 | [`black-forest-labs/FLUX.2-klein-9b-fp8`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8), `flux-2-klein-9b-fp8.safetensors` | First-party gated artifact pinned at `902d9d510b51533e07729f19211414a3648b77d2`; 9,433,061,528 bytes; SHA-256 `865ba09f5b4c3cbd3468a4bd3acb9fcb2f8740c54317482f0bcd4ed1d3655cee`; exact Engine schema `c25cec50…`. | **Fallback accepted on RTX 5080** |
| Base FP8 | [`black-forest-labs/FLUX.2-klein-base-9b-fp8`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8), `flux-2-klein-base-9b-fp8.safetensors` | First-party gated artifact. Corroborated pointer: 9,567,278,472 bytes; SHA-256 `a9f5028c24a7a96f4f45beb883aad287d9bccc246227a6803edc898ddda42cf4`. Official immutable file commit still required. | **Deferred Base line** |
| KV FP8 | [`black-forest-labs/FLUX.2-klein-9b-kv-fp8`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv-fp8/blob/a4d584032d9be4310b40531bc76b6b8398eba2c5/flux-2-klein-9b-kv-fp8.safetensors), `flux-2-klein-9b-kv-fp8.safetensors` | First-party upload commit `a4d584032d9be4310b40531bc76b6b8398eba2c5`; 9,818,935,984 bytes; SHA-256 `33f7da5625a00798349a719742999d3c7dd20c1a7eda14663922c363640728f1`; Xet `b8763ddd83d92fb7592fdb30153fd46521dc7b610e919e20159825964f4711c7` | **Separate KV experiment** |
| Distilled NVFP4 | [`black-forest-labs/FLUX.2-klein-9b-nvfp4`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4), `flux-2-klein-9b-nvfp4.safetensors` | First-party gated artifact pinned at `e882f64f6aa086fcf8915a7763550e05af10ef13`; 5,760,960,048 bytes; SHA-256 `5c72214496dd278f721a112e1bd1585fffed487bc0831c894bcbf30d12e9ee48`; exact Engine schema `a222d48e…`. | **Recommended accepted on RTX 5080** |
| Base NVFP4 | [`black-forest-labs/FLUX.2-klein-base-9b-nvfp4`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-nvfp4), `flux-2-klein-base-9b-nvfp4.safetensors` | First-party gated artifact. Corroborated pointer: 5,809,193,808 bytes; SHA-256 `730d6bdbd5069cd4cd263cfdc4801d0d06ca14457b903baa5d953a8c2f9e84c9`. Official immutable file commit still required. | **Deferred Base line** |
| KV NVFP4 | No first-party artifact verified. Community lead: [`ApacheOne/FLUX.2-klein-9b-kv-nvfp4_mixed`](https://huggingface.co/ApacheOne/FLUX.2-klein-9b-kv-nvfp4_mixed/tree/1c119e68f2741d0ad46ff56940ca54f622af1a24), including an [all-eligible-attention mixed NVFP4 file](https://huggingface.co/ApacheOne/FLUX.2-klein-9b-kv-nvfp4_mixed/blob/1c119e68f2741d0ad46ff56940ca54f622af1a24/flux2-klein-9b-kv-nvfp4.safetensors) and a [BF16 text-attention variant](https://huggingface.co/ApacheOne/FLUX.2-klein-9b-kv-nvfp4_mixed/blob/1c119e68f2741d0ad46ff56940ca54f622af1a24/flux2-klein-9b-kv-nvfp4_txtattnBF16.safetensors). | Community NCL conversion with a small adoption signal. Its author says it works in ComfyUI and with ordinary Klein 9B LoRAs, but also explicitly reports incomplete support and VRAM/RAM spikes. Header-only inspection found a coherent mixed NVFP4 SafeTensors layout; no Engine materialization, output, memory, LoRA, or quality proof exists. | **Watch-list only; not recipe-grade** |

Corroborated identities are useful for detecting source drift, but they must not appear
in a production resource declaration until an authenticated first-party revision
resolves to the same filename, bytes, and SHA.

### Shared Comfy components

The ordinary Base/Distilled edit and Base T2I graphs use:

- [`qwen_3_8b_fp8mixed.safetensors`](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/23fbc8aa8b621f29f2249cd1bd9c47e5d0eebd83/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors):
  pinned Comfy-Org revision `23fbc8aa8b621f29f2249cd1bd9c47e5d0eebd83`;
  8,664,848,742 bytes; SHA-256
  `abad16806e0cbabc54e0325d6565847443fe396d5f0be38bb3cd3fe75a1201d6`;
  exact mixed layout: 141 global FP8 Linear layers, 85 packed NVFP4 Linear
  layers, and 172 dense BF16 tensors, with only tied `lm_head.weight` omitted.
- [`full_encoder_small_decoder.safetensors`](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/blob/a3efc24f613ef42d9428af62fdbd6f5fd8856c4a/full_encoder_small_decoder.safetensors):
  249,519,092 bytes; SHA-256
  `ea4273f02d1fafbf8e1d1c2cf6018ed8748652eb0bf34f2dd91171f16f15ab62`.
- a matching 15,886,279-byte tokenizer/config/scheduler shell pinned from the
  first-party Distilled repository at `92196c8e11f7b6cf2b7493e037d8c5345c559216`.

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
- Engine's stored planner is parameterized by the exact 4B/9B architecture and accepts
  only typed, exact 9B FP8/NVFP4 recipe components; ad-hoc standalone 9B overrides
  remain rejected so incomplete component closures cannot masquerade as runnable.
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

The first proof target is Distilled NVFP4 T2I at 1024², followed by one-reference
Distilled edit, warm reuse, and the matching FP8 cases. Do not start with KV, Base,
or three references.

## Current Engine truth

- Package recipes expose ordinary Distilled T2I and one-to-three-reference I2I for
  first-party NVFP4, first-party FP8, and complete first-party BF16.
- `klein9b-image` deduplicates the two quantized representations across T2I/I2I and
  shares the exact Comfy mixed Qwen3-8B, BFL small decoder, and weight-free support
  shell. `klein9b-reference-bf16-image` is a separate source-of-truth profile.
- The transformer planner is parameterized by the exact 9B Diffusers architecture;
  both FP8 and NVFP4 materializers consume the original SafeTensors payload directly.
- The mixed Qwen loader proves every source/target/sidecar, reconstructs 141 FP8 and
  85 NVFP4 Kitchen tensors without a converted copy, explicitly stages the component,
  and requires positive native-dispatch deltas from every quantized Linear.
- NVFP4 transformer execution likewise bypasses eager/dequant fallback and requires
  measured native-dispatch deltas after every successful generation.
- I2I reuses the mandatory partial, persistent-subset transformer residency policy
  developed for 4B. Qwen is offloaded and allocator-cleaned before transformer
  residency; the VAE is released before denoising and the transformer before decode.
- Structural catalog/runtime tests and native RTX 5080 micro-kernels pass. The full
  9B ordinary closure is installed; fixed-seed public-API 1024² NVFP4/FP8 T2I/I2I,
  warm reuse, and switching records exist. Cancellation and multi-reference coverage
  remain separate follow-up work.
- Base and KV semantics remain absent by design; the community KV-NVFP4 lead is only
  a backburner roadmap item.

**Proof level: ordinary FP8 and NVFP4 output-qualified on the RTX 5080 through the
public API at 1024². The complete BF16 Reference closure is installed and remains
available, but its 15.9 GiB workstation attempt OOMs during a 1.16 GiB allocation;
that result is retained as an honest reference-limit record rather than changing the
source-of-truth contract.**

## Opinionated status matrix

| Path | Status | Reason |
| --- | --- | --- |
| Authenticated, immutable Distilled BF16 repository | **Reference** | Matching first-party source for ordinary Distilled T2I/edit |
| Bounded Distilled BF16 Engine recipe | **Reference** | Exact complete-folder source-of-truth recipe/profile is cataloged |
| Official Distilled FP8 + exact Comfy closure | **Fallback** | 1024² public-API T2I/I2I and NVFP4↔FP8 switching accepted on RTX 5080 |
| Official Distilled NVFP4 | **Recommended on Blackwell** | 1024² public-API T2I/I2I, warm reuse, switching, and native dispatch accepted on RTX 5080 |
| Base BF16 | **Reference for Base only** | Separate operation/foundation line |
| Base FP8/NVFP4 | **Deferred** | No current product need justifies a second 9B line |
| KV BF16/FP8 | **Separate Experimental line** | Valuable only for repeated-reference editing with explicit cache lifecycle |
| Community KV NVFP4 / ConvRot / GGUF / W4 / Nunchaku | **Backburner** | Not needed before the first-party ordinary ladder passes acceptance |
| Hosted BFL API | **Fallback** | Cloud access path, not local artifact qualification |
| Recommended local 9B path | **Distilled NVFP4** | Opinionated Blackwell default; accepted for the initial RTX 5080 ordinary Distilled suite |

## Remaining acceptance work

### Resource prerequisites

1. Retain the FLUX NCL/filter/manual-review decision as an explicit product gate.
2. Re-run acquisition validation whenever a pinned artifact is revised; the accepted
   RTX 5080 install resolved the exact declared FP8/NVFP4 bytes and SHA-256 values.

### Loader/runtime prerequisites

1. Preserve the accepted deterministic public-API 1024² T2I and one-reference I2I
   records for NVFP4 and FP8, including mixed-Qwen/transformer native dispatch,
   residency accounting, output dimensions, seed, and four-step schedule.
2. Add cancellation and clean-recovery coverage before treating the ladder as fully
   lifecycle-qualified.
3. Activate the official disabled two-reference topology, then exercise the Engine's
   deliberate third-reference extension only after one-reference acceptance.
4. Re-attempt BF16 reference output only on hardware with enough VRAM/RAM headroom;
   the first 15.9 GiB attempt recorded an expected OOM without weakening the closure.
5. Keep Base and KV code paths absent until their separately scoped work begins.

### Harness prerequisites

Use the manual non-CI API harness in [Hardware studies](../HARDWARE_STUDIES.md):

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
2. **Recommended on Blackwell:** exact official Distilled NVFP4 closure using the
   same operation, Qwen, VAE, support, prompt/assets, and schedule.
3. **Fallback:** exact official Distilled FP8 closure for the broader supported
   hardware tier.

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

NVFP4 retains the opinionated Recommended tier only if it completes the accepted
creator workload reliably inside the target envelope and remains creator-equivalent
to FP8/BF16 in held-constant studies. FP8 retains Fallback when it is dependable on
its supported hardware tier. If end-to-end evidence contradicts either judgment,
change the tier rather than hiding the result.

## Ordered next actions

1. Add cancellation and clean-recovery cases to the public API harness.
2. Exercise two ordered references, then the deliberate three-reference Engine extension.
3. Creator-review the accepted held-seed FP8/NVFP4 outputs before making a quality claim.
4. Re-attempt BF16 only on a larger-VRAM reference workstation.
5. Update proof status from saved harness manifests; fix surfaced bugs before adding
   formats or model lines.
6. Stop. Base and KV—including the community KV-NVFP4 candidate—stay backburnered.

## Source conflicts and blockers

1. **Gated acquisition:** the first-party 9B artifacts are immutably pinned, but local
   installation still requires accepting BFL terms and a valid Hugging Face token.
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
8. **Hardware acceptance block:** exact stored loaders exist, but no full public-API
   9B output or lifecycle proof has run on the target workstation.
9. **License/moderation block:** NCL gate and filter/manual-review obligations need a
   product decision.
10. **Feasibility block:** no 16 GB/64 GB output, lifecycle, or paging-thrash proof exists.

## Explicit non-goals

- No Base quantized product tranche.
- No KV as a generic recommended path.
- No 9B ConvRot, GGUF, MXFP8, W4A8, W4A4, Nunchaku, or runtime conversion.
- No encoder-format shootout; use the FP8-mixed Qwen selected by official graphs.
- The third reference remains an explicit Engine extension, not an official Comfy
  parity claim; accept it only after the official one/two-reference topology passes.
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
- [ApacheOne community KV NVFP4 mixed conversion (unqualified watch-list)](https://huggingface.co/ApacheOne/FLUX.2-klein-9b-kv-nvfp4_mixed/tree/1c119e68f2741d0ad46ff56940ca54f622af1a24)
- [Official Comfy workflow templates at `96a8cab`](https://github.com/Comfy-Org/workflow_templates/tree/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb)
- [Official Comfy 9B encoder/VAE repository](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b)
- [ComfyUI current quantized loading](https://github.com/Comfy-Org/ComfyUI/tree/27bca654eb9a70237d93f56a6ea336ab55f8925d)
- [Comfy Kitchen 0.2.28](https://github.com/Comfy-Org/comfy-kitchen/tree/75aa2ab6f9f45575205489b9593cf9fe01a57028)
- [LatentSlate Engine](https://github.com/EnviralDesign/LatentSlate-Engine)
