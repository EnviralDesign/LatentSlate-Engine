# Krea 2 roadmap

Last reviewed: **2026-08-13**
Target workstation: **Windows 11, RTX 5080 16 GB (SM120), Python 3.12**

## Executive decision

Krea 2 is **not a FLUX.2 model**. It is a 12–13B-class text-to-image family
trained from scratch by Krea and released as two coordinated checkpoints:

- **Krea 2 Raw** — undistilled foundation checkpoint for post-training and LoRA
  training; 52 steps and CFG 3.5 in Krea's recommended example.
- **Krea 2 Turbo** — post-trained/distilled production checkpoint for ordinary
  inference; 8 steps, CFG 0, `mu=1.15`, and 1K–2K output support.

For LatentSlate, Raw is a **training reference**, not the inference default. Turbo is
the only first product candidate. Engine currently has no Krea family, loader,
recipe, or artifact declaration. Turbo BF16 remains the source-of-truth reference,
but the first implementation candidate is now the exact pinned INT8 workflow topology
documented below rather than a generic quantization project. Its saved default is a
bounded three-artifact closure; the
configured Darkbrush LoRA becomes a separate four-file variant only when enabled.
Do not add community FP8/GGUF formats before Turbo itself demonstrates enough creator
value to justify the family.

The Krea 2 Community License is a product gate: commercial use under the community
license is limited to organizations below **US$1 million trailing-twelve-month
company-wide revenue**, deployers must implement appropriate content filtering, and
larger commercial use requires an enterprise license. That obligation belongs in the
recipe decision, not only in legal notes.

## Evidence labels

- **Verified** — stated by an official model card, repository, license, or the Engine source at the audited commit.
- **Publisher measurement** — an upstream speed, memory, or quality claim; not an Engine result.
- **Inference** — a roadmap product judgment requiring target-workstation validation.

## Scope and lineages

| Line | Intended use | Verified default | Comparison boundary |
| --- | --- | --- | --- |
| Raw | LoRA training, fine-tuning, post-training, foundation research | 52 steps, CFG 3.5, trained up to 1K | Compare Raw only to Raw derivatives and training workflows |
| Turbo | Creator-facing T2I inference; Raw-trained LoRAs may be applied | 8 steps, CFG 0, `mu=1.15`; 1K–2K, dimensions padded to multiples of 16 | Compare Turbo precision/runtime variants with the same prompt, LoRA, seed, and schedule |

Verified source: the official Krea repository at immutable commit
[`db3984fbc6e13b34c0064990fc2d95ac64d00058`](https://github.com/krea-ai/krea-2/tree/db3984fbc6e13b34c0064990fc2d95ac64d00058).
Krea explicitly says “TRAIN on Raw and RUN on Turbo.”

## Published artifacts and topology

| Artifact | Role / format | Exact evidence | Disposition |
| --- | --- | --- | --- |
| [`krea/Krea-2-Raw/raw.safetensors`](https://huggingface.co/krea/Krea-2-Raw/blob/main/raw.safetensors) | Raw single-file checkpoint; mixed BF16/F32 metadata | **26.3 GB**; SHA-256 `f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7` | **Reference** for Raw/training only |
| [`krea/Krea-2-Turbo/turbo.safetensors`](https://huggingface.co/krea/Krea-2-Turbo/blob/68e6019eebd5040b612105903a0c366c35cac757/turbo.safetensors) | Turbo single-file checkpoint; official inference artifact | **26.3 GB**; SHA-256 `78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3` | **Reference** and first Experimental candidate |
| Official repository inference shell | Loads Raw/Turbo by environment path; official examples; Comfy, Diffusers, Fal, and SGLang entry points are named | Code is available at the immutable GitHub commit above | Required implementation reference |
| Community FP8 / mixed FP8 / GGUF derivatives | Post-release third-party quantizations | Real artifacts exist, but provenance, layer protection, and end-to-end Krea 2 quality evidence vary | **Deferred** |
| Runtime cast from Turbo BF16 to FP8 | Conversion at load or execution time | Technically possible in generic runtimes | **Rejected**: violates Engine's stored-artifact rule |
| Hosted Krea/Fal API | Hosted T2I | First-party/partner access, different product boundary | **Fallback** |

The Hugging Face repository heads are mutable. The Turbo file link above uses a
specific file revision; any Engine implementation must also pin the repository
revision and all auxiliary configuration/tokenizer files rather than record only the
large checkpoint hash.

### Verified official INT8 workflow closure

The pinned [official Comfy workflow template](https://github.com/Comfy-Org/workflow_templates/blob/2b7f823136606344f0bccce249898d771b809aa1/templates/image_krea2_turbo_t2i_int8.json)
is the current concrete Turbo execution reference. The graph configures
`krea2_darkbrush.safetensors` through `LoraLoaderModelOnly` at strength **0.8**, but
its saved `enable_lora` switch is **false** and selects the base transformer branch.
Saved-default parity is therefore the **three-file** closure. The four-file path is a
separate explicitly enabled Darkbrush variant; preserve the wired literal `0.8` when
qualifying it. Configured-but-disabled is not active execution. The saved workflow
also fixes **8 steps, CFG 1, Euler/simple, and prompt enhancement enabled**. Those
settings are its operation contract and must not be replaced with the publisher BF16
example's CFG 0 / `mu=1.15` settings.

| Role | Immutable source | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Transformer | [`Comfy-Org/Krea-2`, `diffusion_models/krea2_turbo_int8_convrot.safetensors`, `6b1d7191d84d5ded74d83a1a98211dad0ac8ae25`](https://huggingface.co/Comfy-Org/Krea-2/blob/6b1d7191d84d5ded74d83a1a98211dad0ac8ae25/diffusion_models/krea2_turbo_int8_convrot.safetensors) | 13,492,686,496 | `8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1` |
| Text/vision encoder | [`Comfy-Org/Krea-2`, `text_encoders/qwen3vl_4b_fp8_scaled.safetensors`, `4aa0eed112bd2780ceea37583edbdcd2df6c2c09`](https://huggingface.co/Comfy-Org/Krea-2/blob/4aa0eed112bd2780ceea37583edbdcd2df6c2c09/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) | 5,242,467,968 | `54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094` |
| VAE | [`Comfy-Org/Krea-2`, `vae/qwen_image_vae.safetensors`, `a0a28f7e5b645c950ad56fc2e45bfd3e0044c06e`](https://huggingface.co/Comfy-Org/Krea-2/blob/a0a28f7e5b645c950ad56fc2e45bfd3e0044c06e/vae/qwen_image_vae.safetensors) | 253,806,246 | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` |
| Optional Darkbrush LoRA | [`Comfy-Org/Krea-2`, `loras/krea2_darkbrush.safetensors`, `b5a1dcd1574c1d256408cbb5ae46a67b225481e6`](https://huggingface.co/Comfy-Org/Krea-2/blob/b5a1dcd1574c1d256408cbb5ae46a67b225481e6/loras/krea2_darkbrush.safetensors) | 469,291,992 | `f47c4316dd93af66e0518c93b582f459571d4925b519133770c73a52cd5db7c6` |

Saved-default parity is `13,492,686,496 + 5,242,467,968 + 253,806,246 =
18,988,960,710` bytes. The explicitly enabled Darkbrush variant adds `469,291,992`
bytes, for **19,458,252,702 bytes**. These identities were rechecked through Hugging Face's
immutable file metadata on 2026-08-13; no model payload was downloaded.

## License and distribution gate

The official
[Krea 2 Community License v1](https://github.com/krea-ai/krea-2/blob/db3984fbc6e13b34c0064990fc2d95ac64d00058/docs/KREA-2-COMMUNITY-LICENSE)
contains several operational requirements relevant to a local Engine recipe:

- commercial use under the community license requires company-wide annual revenue
  below **US$1,000,000** on a trailing-twelve-month basis;
- reaching that threshold requires stopping community-license commercial use and
  obtaining an enterprise license;
- redistribution has license, naming, notice, and derivative-attribution conditions;
- deployments must implement reasonable content-filter measures;
- the agreement is revocable and contains acceptable-use and provenance obligations.

This roadmap does not provide legal advice. It records that a public download is not
sufficient product clearance.

## Current Engine truth at `2ba5709`

- **Not implemented.** `model_store.py` and the tool registry contain no `krea2`
  family or operation:
  [`model_store.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/model_store.py),
  [`tools/__init__.py`](https://github.com/EnviralDesign/LatentSlate-Engine/blob/2ba57095796ca6e13285afd23da3582383d82df9/src/latentslate_engine/tools/__init__.py).
- **No recipe, resource, or bundle.** No exact Krea artifact is cataloged or
  downloadable through Engine.
- **No proof.** There are no Engine schema tests, header plans, target-hardware jobs,
  or creator acceptance results.
- **Runtime compatibility is unverified.** Upstream says ComfyUI and Diffusers support
  Krea 2, but that does not prove compatibility with Engine's pinned Diffusers commit
  or its lifecycle/offload model.

## Opinionated status matrix

| Path | Status | Why |
| --- | --- | --- |
| Raw official checkpoint | **Reference** | Source of truth for Raw training/LoRA work, not the ordinary generator |
| Turbo official checkpoint | **Reference** | First-party source for the creator-facing inference line |
| Exact Turbo BF16 Engine path | **Reference** | First-party source of truth for the creator-facing inference line |
| Pinned INT8 saved default | **Experimental** | Exact three-file Engine-native base topology; Darkbrush is configured but disabled; requires lifecycle and native-dispatch proof |
| Pinned INT8 + enabled Darkbrush | **Experimental variant** | Exact four-file Engine-native variant with fixed `0.8` LoRA; separately qualify output and dispatch |
| Raw-trained LoRA on Turbo | **Deferred** | Upstream-recommended ecosystem feature, but only after the base Turbo recipe is accepted |
| Community stored FP8 | **Deferred** | Could help a 16 GB workstation, but no candidate is yet important enough to precede BF16 qualification |
| GGUF / arbitrary INT4 / Nunchaku / community ConvRot | **Rejected** | Multiple unpinned loaders would front-load maintenance before creator value is known; this does not reject the pinned official INT8 ConvRot topology above |
| Raw as an ordinary inference default | **Rejected** | Krea explicitly recommends Turbo for inference |
| Hosted API | **Fallback** | Useful for access and comparison, but not local artifact qualification |
| Recommended local path | **None** | No Engine result exists yet |

## Small qualification ladder

1. **Reference:** exact official Turbo BF16 checkpoint, 8 steps, CFG 0,
   `mu=1.15`, fixed resolution and seed.
2. **Incumbent candidate:** the same stored artifact through a bounded Engine loader
   with explicit component residency and no conversion.
3. **Official challenger:** the exact three-file saved-default INT8 ConvRot topology
   after its headers, lifecycle, and native dispatch are proven. Qualify the enabled
   Darkbrush `0.8` four-file variant separately.
4. **Optional challenger:** one pinned community stored-FP8 Turbo artifact only after
   the first two paths are accepted and only if it plausibly changes 16 GB viability.

Raw gets a separate, smaller research check only if LatentSlate adds model training:
verify a tiny Raw-trained LoRA applies to Turbo as upstream claims. Do not mix that
with inference-format acceptance.

## Model-specific acceptance

Use the shared harness in [README](./README.md), plus a Krea-specific corpus:

- aesthetic/editorial photography, stylized illustration, abstract design, fashion,
  material studies, cinematic scenes, typography, and long-form art direction;
- 1024², 1536×1024, 1024×1536, and 2048² where memory allows;
- at least one prompt that stresses diversity across several seeds, because Raw/Turbo
  post-training may change diversity even when prompt alignment remains good;
- optional Raw-trained LoRA compatibility only after plain Turbo acceptance.

Measure prompt encode, checkpoint load, eight denoise steps, VAE decode, peak VRAM,
peak RAM, and PCIe transfer. A 26.3 GB single file cannot be assumed viable on 16 GB
merely because CPU offload exists. Cancellation must be tested during load, encode,
denoise, and decode, followed by a clean second job.

A community low-bit loader requires the normal 20–25% material win or a previously
unrunnable accepted workload. Storage reduction by itself is insufficient.

## Hard gaps and source conflicts

1. **Family naming error corrected:** Krea 2 is trained from scratch and is not
   FLUX.2 Krea. The portfolio calls it Krea 2.
2. **16 GB behavior is unknown:** the 26.3 GB Turbo file implies offload, but no
   first-party target-workstation VRAM/RAM/cold-time result is available here.
3. **License/product gap:** revenue threshold, content filtering, redistribution, and
   enterprise-license requirements must be reviewed before a built-in downloader.
4. **Auxiliary closure gap:** the single-file checkpoint is not enough to define a
   reproducible runtime; configs, tokenizer/encoder expectations, scheduler, and
   safety/provenance behavior need a pinned manifest.
5. **Engine implementation gap:** an immutable official workflow, three-file saved
   default, and four-file enabled-LoRA variant are now pinned above, but Engine has
   not yet proved its Krea loader, staged
   residency, exact prompt-enhancement behavior, or native INT8/FP8 dispatch.
6. **No accepted quant candidate:** community formats are real, but none has sufficient
   first-party provenance or Engine evidence to be the first implementation.

## Ordered next actions

1. Decide whether the Krea Community License and content-filter requirements are
   acceptable for a built-in local model family.
2. Build a header-only manifest for the three saved-default artifacts; separately
   validate the optional Darkbrush file and full target map for its enabled variant.
3. Implement one bounded saved-default INT8 Turbo T2I path with explicit
   encoder/transformer/VAE residency and no runtime weight conversion.
4. Run the graph-default eight-step case on the RTX 5080, then compare it with the
   four-file Darkbrush-at-0.8 variant using the same prompt and seed.
5. Add the Turbo BF16 Reference only after its complete immutable support closure is
   pinned and adequate-hardware/offload qualification is possible.
6. Promote neither path until cancellation, reuse, teardown, RAM, cold-load, and
   native-dispatch acceptance succeed. Defer Raw and user LoRA tooling.

## Explicit non-goals

- Do not describe Krea 2 as FLUX.2 or assume FLUX loaders apply.
- Do not make Raw the normal creator-facing inference path.
- Do not download both 26.3 GB checkpoints in a default recipe.
- Do not add runtime FP8 casting, generic weight-dtype switches, or unpinned community
  quantizations.
- Do not implement training/LoRA management as part of initial T2I support.
- Do not treat permissive-sounding marketing text as a substitute for the license.

## Primary sources

- Official code and defaults, immutable commit:
  <https://github.com/krea-ai/krea-2/tree/db3984fbc6e13b34c0064990fc2d95ac64d00058>
- Official Community License, immutable commit:
  <https://github.com/krea-ai/krea-2/blob/db3984fbc6e13b34c0064990fc2d95ac64d00058/docs/KREA-2-COMMUNITY-LICENSE>
- Raw model card: <https://huggingface.co/krea/Krea-2-Raw>
- Raw checkpoint: <https://huggingface.co/krea/Krea-2-Raw/blob/main/raw.safetensors>
- Turbo model card: <https://huggingface.co/krea/Krea-2-Turbo>
- Turbo checkpoint at a specific revision:
  <https://huggingface.co/krea/Krea-2-Turbo/blob/68e6019eebd5040b612105903a0c366c35cac757/turbo.safetensors>
