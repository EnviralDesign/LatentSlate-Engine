# Source-pin and authority audit

Last authority audit: **2026-08-13**

This ledger records immutable objects resolved during the authority audit, classifies accepted versus research pins, and lists links intentionally left mutable. It does not replace the per-family authority maps or artifact tables.

## Pin interpretation

- **Accepted baseline:** exact upstream/runtime source used by retained Engine acceptance evidence.
- **Current research:** immutable snapshot used to compile the next implementation packet.
- **Authoring baseline:** source/package used by recipe/template ingestion rather than family execution.
- **Historical alternate:** immutable older source retained to explain prior results.
- **Mutable discovery:** landing page, gated model card, or tutorial used to locate material; never an acquisition lock.

Do not unify these categories merely because one commit is newer.

## Independently resolved GitHub objects

Every object below and every changed immutable path under it was fetched through GitHub before publication.

| Repository / role | Resolved commit | Classification |
| --- | --- | --- |
| LatentSlate Engine audit base | [`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c) | Engine truth audited by this branch |
| Comfy workflow templates | [`1206ea94470a5b66948f1758a8feea5b00801ed1`](https://github.com/Comfy-Org/workflow_templates/tree/1206ea94470a5b66948f1758a8feea5b00801ed1) | recipe-authoring baseline, package `0.1.37` |
| Comfy workflow templates | [`96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb`](https://github.com/Comfy-Org/workflow_templates/tree/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb) | Klein accepted baseline |
| Comfy workflow templates | [`2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1) | current portfolio research and Wan 14 active accepted-template baseline; Wan 14 package `0.1.42` |
| ComfyUI | [`27bca654eb9a70237d93f56a6ea336ab55f8925d`](https://github.com/Comfy-Org/ComfyUI/tree/27bca654eb9a70237d93f56a6ea336ab55f8925d) | Klein accepted node/runtime baseline |
| ComfyUI | [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541) | current portfolio research baseline |
| ComfyUI | [`eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f`](https://github.com/Comfy-Org/ComfyUI/tree/eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f) | Wan 5 accepted executable worker baseline |
| Comfy Kitchen v0.2.28 | [`75aa2ab6f9f45575205489b9593cf9fe01a57028`](https://github.com/Comfy-Org/comfy-kitchen/tree/75aa2ab6f9f45575205489b9593cf9fe01a57028) | Klein accepted dispatch baseline |
| Comfy Kitchen research snapshot | [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4) | current portfolio research baseline |
| comfy-model-tools ConvRot source | [`1fe341bb8a4e46f161a978b5faa2412d8c39c768`](https://github.com/Comfy-Org/comfy-model-tools/tree/1fe341bb8a4e46f161a978b5faa2412d8c39c768) | conversion/header research only |
| ComfyUI examples | [`f9431bb000ce792094ff345446e22cac1ea6cef3`](https://github.com/comfyanonymous/ComfyUI_examples/tree/f9431bb000ce792094ff345446e22cac1ea6cef3) | Wan 5 accepted workflow source and older Wan 14 historical evidence |
| Wan 2.2 publisher source | [`42bf4cfaa384bc21833865abc2f9e6c0e67233dc`](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc) | weight/architecture authority |
| LTX-2 publisher source | [`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`](https://github.com/Lightricks/LTX-2/tree/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca) | LTX 2.5 architecture/package authority |
| Krea 2 publisher source | [`db3984fbc6e13b34c0064990fc2d95ac64d00058`](https://github.com/krea-ai/krea-2/tree/db3984fbc6e13b34c0064990fc2d95ac64d00058) | Krea architecture/license authority |
| Ideogram 4 publisher source | [`990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2`](https://github.com/ideogram-oss/ideogram4/tree/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2) | Ideogram architecture/prompt authority |
| MiniMax H3 publisher source | [`fa6891ff7cdaaa03fa4497e89ac64ff169219acf`](https://github.com/MiniMax-AI/MiniMax-H3/tree/fa6891ff7cdaaa03fa4497e89ac64ff169219acf) | H3 architecture/operation authority |
| Z-Image publisher source | [`26f23eda626ffadda020b04ff79488e1d72004cd`](https://github.com/Tongyi-MAI/Z-Image/tree/26f23eda626ffadda020b04ff79488e1d72004cd) | Z-Image architecture/lineage authority |

## Resolved ledger contradictions

The prior ledger said the Krea, Ideogram, and MiniMax GitHub objects were unresolved. Independent API resolution established:

- `krea-ai/krea-2@db3984fbc6e13b34c0064990fc2d95ac64d00058` exists, and `docs/KREA-2-COMMUNITY-LICENSE` exists at that commit.
- `ideogram-oss/ideogram4@990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2` exists, and `docs/model_architecture.md`, `docs/prompting.md`, and `docs/inference.md` resolve at it.
- `MiniMax-AI/MiniMax-H3@fa6891ff7cdaaa03fa4497e89ac64ff169219acf` exists, and `README.md` resolves at it.

The family roadmaps were correct to treat those GitHub objects as immutable. The old unresolved warning was stale and is removed rather than chosen over the independently fetched objects.

## Workflow pin roles

| Family/surface | Accepted baseline | Current research | Notes |
| --- | --- | --- | --- |
| Recipe authoring | `1206ea...`, package `0.1.37` | family-specific | authoring ingestion is not runtime acceptance |
| Klein 4B/9B | `96a8cab...`, ComfyUI `27bca...`, Kitchen `0.2.28@75aa...` | `2b7f...` / `725e...` / `78e...` for new research | accepted results retain the older exact stack |
| Wan 14B | `2b7f...`, package `0.1.42` | same until deliberately advanced | older `f9431bb...` shift-8 examples are historical alternate evidence |
| Wan 5B | examples `f9431bb...`, executable ComfyUI `eb4a...` | later templates may be studied separately | accepted worker provenance must not be rewritten to `725e...` merely because it is newer |
| Krea/Qwen/Ideogram/LTX/H3/Z research | `2b7f...` raw workflows | same at this audit | none are accepted merely because the research pin is exact |

## Closure checks repeated in this audit

- Saved-default switches determine active closure. Krea’s `enable_lora=false` keeps Darkbrush configured but disabled; the saved base graph is three files, while the enabled 0.8 mode is four files.
- Qwen 2511’s saved Lightning switch is off; the saved INT8 mode is 40 steps/CFG 4 and three fixed files. The four-step/CFG-1 LoRA mode is a separate four-file contract.
- LTX 2.5 T2V actively selects six resources; its FLF sibling selects a different graph and must not substitute for T2V.
- Wan 14B T2V, I2V, and FLF are distinct operation graphs even when components overlap.
- Ideogram requires both conditional and unconditional branches plus encoder and VAE.
- Z-Image Turbo INT8 is a three-file T2I graph; first-party architecture source does not create an edit operation.

## Mutable links intentionally retained

The following link classes remain mutable because they are discovery, gating, or legal landing pages rather than immutable execution locks:

- Hugging Face model-card/repository landing pages used to describe publisher claims, request access, or discover files.
- Official Comfy tutorials used as explanatory discovery material when an immutable raw workflow is cited alongside them.
- Hosted product/API documentation whose current service contract cannot be represented by a Git commit.
- License landing pages where the gated source serves the operative text; roadmaps avoid unsupported legal conclusions and require product/legal review.

A mutable link is never the only support for an exact artifact identity, saved graph, node schema, or dispatch claim.

## Changed-link audit rule

For every immutable link added or changed in this branch:

1. resolve the complete commit object;
2. resolve the exact file/directory path at that commit;
3. reject shortened or guessed suffixes;
4. preserve accepted historical pins instead of replacing them with a newer research pin;
5. label any remaining mutable link with its discovery/gating rationale.
