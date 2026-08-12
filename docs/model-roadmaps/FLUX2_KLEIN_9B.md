# FLUX.2 Klein 9B roadmap

Last reviewed: **2026-08-11**  
Target workstation: **Windows 11, RTX 5080 16 GB (SM120), Python 3.12**

## Executive decision

FLUX.2 Klein 9B is a high-value research target but **not a current 16 GB product
default**. Keep each of its three materially different lines separate:

1. **Distilled 9B** — unified T2I and one-to-many-reference editing in four steps.
2. **Base 9B** — undistilled foundation model, normally 50 steps with CFG, intended
   for maximum flexibility and fine-tuning.
3. **9B-KV** — a Distilled editing variant that caches reference-image K/V state after
   the first denoising step; its benefit is specific to repeated reference editing.

The current Engine direct tool is a complete-folder **Distilled BF16** path. It is not
an opinionated recipe, has no immutable upstream revision, has no 9B stored-quantized
loader, and has not completed 16 GB T2I/I2I acceptance. Do not expand into a format zoo.
The next useful work is to pin the exact Distilled source, prove the offload boundary,
and then qualify the official BFL FP8 artifact. Treat KV FP8 as a separate editing
experiment only after ordinary 9B editing is stable.

## Evidence labels

- **Verified** — stated by a first-party model card, repository, vendor document, or
  the Engine source at the pinned audit commit.
- **Vendor measurement** — an upstream performance claim; not an Engine result.
- **Inference** — a roadmap judgment that must be validated on the target workstation.

## Scope and lineages

| Line | Operations | Canonical settings | Comparison boundary |
| --- | --- | --- | --- |
| Distilled 9B | T2I, single-reference edit, multi-reference edit | 4 inference steps; unified 9B flow model plus Qwen3 8B text embedder | Compare only against Distilled 9B artifacts with the same references and schedule |
| Base 9B | T2I and editing; training/fine-tuning foundation | BFL Diffusers example: 50 steps, guidance 4.0, 1024×1024 | Base is not a high-precision substitute for Distilled and must have its own ladder |
| 9B-KV | Repeated single/multi-reference editing; T2I capability remains present | 4-step Distilled schedule; reference K/V calculated on step 0 and reused on steps 1–3 | Compare ordinary 9B and KV only on repeated-edit workloads using identical references |

Verified sources: [Distilled model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B),
[Base model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B), and
[KV model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv).
All three use the **FLUX Non-Commercial License** and are gated on Hugging Face. A
LatentSlate distribution or commercial-product decision therefore needs explicit
license review before implementation.

## Published artifacts and topology

| Artifact | Role / format | Published state | Size / memory evidence | Disposition |
| --- | --- | --- | --- | --- |
| [`black-forest-labs/FLUX.2-klein-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | Complete Distilled Diffusers BF16 repository | First-party, gated | HF card says the complete model fits in roughly **29 GB VRAM**; BFL's product page lists **19.6 GB**. These are conflicting measurement contexts, not interchangeable facts. | **Reference** for Distilled |
| [`black-forest-labs/FLUX.2-klein-base-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) | Complete Base Diffusers BF16 repository | First-party, gated | BFL product page lists **21.7 GB**; the Base card demonstrates CPU offload rather than a 16 GB resident path. | **Reference** for Base |
| [`black-forest-labs/FLUX.2-klein-9b-kv`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv) | Complete KV-enabled BF16 repository | First-party, gated | HF card says roughly **29 GB VRAM**. | **Reference** for KV editing |
| [`black-forest-labs/FLUX.2-klein-9b-fp8`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) | Official Distilled FP8 artifact | First-party, gated | BFL says its FP8 family can provide up to 1.6× speed and 40% less VRAM on RTX 5080/5090; this is a vendor family benchmark, not an Engine 9B result. | **Experimental challenger** |
| [`black-forest-labs/FLUX.2-klein-base-9b-fp8`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8) | Official Base FP8 artifact | First-party, gated | Must be benchmarked against Base BF16, not Distilled. | **Deferred** until a Base product need exists |
| [`black-forest-labs/FLUX.2-klein-9b-kv-fp8`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv-fp8) | Official KV FP8 artifact | First-party, gated | Model card reports up to **2.5×** for multi-reference editing from KV reuse and roughly 29 GB for the complete pipeline; the exact local closure and residency must still be measured. | **Experimental**, separate repeated-edit track |
| [`black-forest-labs/FLUX.2-klein-9b-nvfp4`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4) | Official Distilled NVFP4 | First-party, gated; Blackwell-oriented | BFL family claim: up to 2.7× speed and 55% lower VRAM on RTX 5080/5090. No Engine 9B proof. | **Deferred** until FP8 loader/offload is stable |
| [`black-forest-labs/FLUX.2-klein-base-9b-nvfp4`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-nvfp4) | Official Base NVFP4 | First-party, gated | Same lineage restriction as Base FP8. | **Deferred** |
| Community GGUF, INT8, ConvRot, MXFP8, Nunchaku, and mixed W4 paths | Various | Real artifacts exist in the ecosystem | Kernel or file existence does not establish a Klein 9B production path. | **Rejected for the first ladder** |

The vendor speed/VRAM claims come from BFL's
[FLUX.2 Klein technical post](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence),
which explicitly says the quantized-family benchmarks used RTX 5080/5090 at 1024².
They are useful motivation, not acceptance evidence.

## Current Engine truth at `2ba5709`

- **Direct tool only.** Engine registers Klein 9B T2I and I2I tools in
  [`tools/__init__.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/tools/__init__.py)
  and defines the complete-folder path in
  [`tools/klein.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/tools/klein.py).
- **Distilled-only runtime semantics.** The generic native Klein runtime uses the
  four-step, guidance-1 schedule in
  [`runtime/klein.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/runtime/klein.py).
  It is not a Base or KV path.
- **Acquisition is not reproducible enough.** `klein9b-basic` in
  [`bundles.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/bundles.py)
  names the official repository but does **not** pin a revision. That is a verified
  gap, not an immutable artifact contract.
- **No package-owned recipe or resource declaration.** There is no Klein 9B directory
  under `builtin_recipes`, no exact component closure, and no installed-artifact
  schema for FP8/NVFP4/KV.
- **Proof level: direct tool only.** Repository status says Klein 9B I2I still needs
  hands-on diagnosis. No target-class cold/warm/VRAM/output record is checked in.

## Opinionated status matrix

| Path | Status | Why |
| --- | --- | --- |
| Distilled BF16 complete repository | **Reference** | Highest-precision first-party source for the currently implemented Distilled operation |
| Base BF16 complete repository | **Reference** | Required only for Base training/generation/edit qualification |
| KV BF16 complete repository | **Reference** | Required for a truthful KV editing comparison |
| Current Engine Distilled BF16 direct tool | **Experimental** | Code exists, but acquisition, 16 GB acceptance, and recipe closure are incomplete |
| Official Distilled FP8 | **Experimental** | Best next production-format candidate once an exact stored layout and offload design are proved |
| Official KV FP8 | **Experimental** | High-value only for iterative multi-reference editing; separate cache lifecycle acceptance required |
| Official NVFP4 | **Deferred** | Blackwell upside is plausible, but a second low-bit loader is premature before FP8 is stable |
| Base quantized artifacts | **Deferred** | Base is a different product/training line and has no current Engine need |
| GGUF / community W4 / INT8 / ConvRot zoo | **Rejected** | No creator-visible need justifies multiple loaders before first-party FP8 qualification |
| Hosted BFL API | **Fallback** | Useful for access or larger hardware avoidance, but it is a cloud-provider path, not local artifact qualification |

## Small qualification ladder

### Distilled T2I and ordinary editing

1. **Reference:** exact pinned BFL Distilled BF16 repository, four steps, guidance 1.
2. **Incumbent candidate:** the same BF16 artifact through a bounded Engine recipe with
   explicit encoder/transformer/VAE residency and no runtime conversion.
3. **Challenger:** exact official Distilled FP8 artifact with the same components,
   schedule, prompt/reference set, and offload policy.

Do not add NVFP4 unless FP8 either cannot meet the 16 GB envelope or leaves a measured
material opportunity.

### Repeated-reference editing

1. **Reference:** ordinary Distilled BF16 editing with fixed references.
2. **KV reference:** BF16 KV artifact, same prompts/seeds/references and four steps.
3. **Challenger:** official KV FP8.

Measure the first generation separately from reference-reuse generations. A cached
second job is not comparable to an uncached first job.

## Model-specific acceptance

Use the shared harness in [README](./README.md), plus:

- T2I corpus: typography, fine texture, identity-free photorealism, geometry, and
  long-prompt composition at 1024².
- Editing corpus: one, two, and three references; identity preservation; object/style
  transfer; text replacement; and a no-op/minimal-change case.
- KV corpus: at least five prompt variations over the same reference set, then a
  changed-reference job that must invalidate the cache correctly.
- Record host-RAM and PCIe transfer peaks because a 16 GB path will necessarily rely
  on staged residency or offload.
- Verify cancellation during text encoding, transformer load, denoising, KV-cache
  creation, cache-reuse generation, and VAE decode.
- Require a clean follow-up after cache invalidation, cancellation, and operation
  switches between T2I and I2I.

A quantized 9B loader must produce either a 20–25% end-to-end warm win or make an
accepted workload run within the 16 GB envelope that the BF16 path cannot. BFL's
published family benchmark does not satisfy this gate by itself.

## Hard gaps and source conflicts

1. **VRAM conflict:** BFL's product page lists Distilled 9B at 19.6 GB and Base at
   21.7 GB, while the HF cards say roughly 29 GB for complete 9B/KV pipelines. The
   likely explanation is different component residency/offload, but that is an
   inference. Engine must reproduce both cold and steady-state memory under one harness.
2. **No immutable Engine source:** the current 9B bundle follows a mutable HF main.
3. **License gate:** all 9B lines use FLUX NCL; local distribution and commercial use
   need a reviewed product decision.
4. **No matching Base or KV Engine references:** the generic Distilled direct path
   cannot qualify Base FP8/NVFP4 or KV behavior.
5. **No stored-format contract:** official FP8/NVFP4 file presence does not prove the
   current Klein 4B stored adapter can load 9B topology.
6. **16 GB feasibility is unproved:** a transformer that fits after quantization does
   not imply the Qwen3 8B encoder, VAE, activations, and caches fit together.

## Ordered next actions

1. Pin an immutable revision and exact file manifest for the Distilled BF16 repository;
   add no new runtime behavior in the roadmap change itself.
2. Run a header-only component/schema audit and document transformer, Qwen3 8B, VAE,
   tokenizer, and scheduler roles and sizes.
3. Prove Distilled BF16 T2I on 16 GB with explicit staged residency, then diagnose
   one-reference I2I before multi-reference work.
4. Promote the bounded BF16 path into a package recipe only after output, cancel,
   reuse, and teardown acceptance.
5. Inspect the official Distilled FP8 header/layout and implement one exact stored
   contract if it can reuse Engine's quantized-tensor infrastructure without runtime
   conversion.
6. Evaluate KV BF16/FP8 only after ordinary edit is stable and only on repeated-reference
   creator workflows.
7. Revisit NVFP4 last, on a runtime that proves native SM120 dispatch.

## Explicit non-goals

- Do not use Distilled BF16 as a Base or KV quality reference.
- Do not recommend 9B merely because BFL publishes an RTX 5080 benchmark.
- Do not add generic `weight_dtype` casting or runtime quantization.
- Do not implement GGUF, Nunchaku, ConvRot, MXFP8, AWQ, or arbitrary community W4
  loaders in the first 9B milestone.
- Do not promise local commercial availability before FLUX NCL review.
- Do not call a cached KV warm run a model-wide speedup.

## Primary sources

- BFL model overview and vendor comparison: <https://bfl.ai/models/flux-2-klein>
- BFL technical post and quantized-family claims: <https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence>
- Distilled BF16: <https://huggingface.co/black-forest-labs/FLUX.2-klein-9B>
- Base BF16: <https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B>
- KV BF16: <https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv>
- Distilled FP8: <https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8>
- Base FP8: <https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8>
- KV FP8: <https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv-fp8>
- Distilled NVFP4: <https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4>
- Base NVFP4: <https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-nvfp4>
