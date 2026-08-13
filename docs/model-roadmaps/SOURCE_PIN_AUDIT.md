# Source-pin and authority audit

Last authority audit: **2026-08-13**

This ledger is subordinate to
[COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md). Every ComfyUI reference below is a
research or historical source pin, never a deployable dependency.

## Pin interpretation

- **Accepted behavioral baseline:** workflow and ComfyUI source used to derive an
  accepted Engine-native runtime.
- **Accepted Kitchen baseline:** Kitchen source/version called directly by Engine.
- **Current research:** source for a pending clean-room implementation.
- **Authoring baseline:** workflow package/source used for evidence ingestion.
- **Historical prototype/alternate:** retained to explain prior observations; not
  current acceptance.
- **Mutable discovery:** landing, access, tutorial, hosted-service, or legal page.

Do not replace an accepted historical pin merely because a newer research pin exists.

## Resolved immutable objects

| Repository / role | Commit | Classification |
| --- | --- | --- |
| Engine policy baseline | [`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2) | current tracked Engine truth |
| workflow templates | [`1206ea94470a5b66948f1758a8feea5b00801ed1`](https://github.com/Comfy-Org/workflow_templates/tree/1206ea94470a5b66948f1758a8feea5b00801ed1) | authoring, package `0.1.37` |
| workflow templates | [`96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb`](https://github.com/Comfy-Org/workflow_templates/tree/96a8cab7fa7b4c201910cd59cdd94dcc3c2d2deb) | Klein accepted behavior |
| workflow templates | [`2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1) | portfolio research; Wan 14 behavior, package `0.1.42` |
| ComfyUI source | [`27bca654eb9a70237d93f56a6ea336ab55f8925d`](https://github.com/Comfy-Org/ComfyUI/tree/27bca654eb9a70237d93f56a6ea336ab55f8925d) | Klein accepted behavioral source only |
| ComfyUI source | [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541) | current research only |
| ComfyUI source | [`eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f`](https://github.com/Comfy-Org/ComfyUI/tree/eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f) | Wan 5 historical nonconforming prototype source |
| Kitchen | [`75aa2ab6f9f45575205489b9593cf9fe01a57028`](https://github.com/Comfy-Org/comfy-kitchen/tree/75aa2ab6f9f45575205489b9593cf9fe01a57028) | Klein accepted direct Kitchen `0.2.28` |
| Kitchen | [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4) | current research |
| model tools | [`1fe341bb8a4e46f161a978b5faa2412d8c39c768`](https://github.com/Comfy-Org/comfy-model-tools/tree/1fe341bb8a4e46f161a978b5faa2412d8c39c768) | ConvRot/header research only |
| examples | [`f9431bb000ce792094ff345446e22cac1ea6cef3`](https://github.com/comfyanonymous/ComfyUI_examples/tree/f9431bb000ce792094ff345446e22cac1ea6cef3) | Wan 5 behavior/artifact contract and older Wan 14 history |
| Wan publisher | [`42bf4cfaa384bc21833865abc2f9e6c0e67233dc`](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc) | architecture/lineage |
| LTX publisher | [`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`](https://github.com/Lightricks/LTX-2/tree/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca) | architecture/package |
| Krea publisher | [`db3984fbc6e13b34c0064990fc2d95ac64d00058`](https://github.com/krea-ai/krea-2/tree/db3984fbc6e13b34c0064990fc2d95ac64d00058) | architecture/license |
| Ideogram publisher | [`990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2`](https://github.com/ideogram-oss/ideogram4/tree/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2) | architecture/prompt |
| MiniMax publisher | [`fa6891ff7cdaaa03fa4497e89ac64ff169219acf`](https://github.com/MiniMax-AI/MiniMax-H3/tree/fa6891ff7cdaaa03fa4497e89ac64ff169219acf) | architecture/operation |
| Z-Image publisher | [`26f23eda626ffadda020b04ff79488e1d72004cd`](https://github.com/Tongyi-MAI/Z-Image/tree/26f23eda626ffadda020b04ff79488e1d72004cd) | architecture/lineage |

Krea’s license path, Ideogram’s architecture/prompt/inference paths, and MiniMax’s
README were independently resolved at the listed commits. The former unresolved
warning was stale.

## Architecture interpretation

- Klein accepted results are Engine-native stored/Kitchen results derived from the
  pinned workflow and ComfyUI source; no external-UI execution is implied.
- Wan 14 accepted results are Engine-native operation-specific expert runtimes derived
  from the pinned official workflows; Engine owns workers, lifecycle, and output.
- Wan 5 `f9431bb...` workflows and `eb4a...` source remain valuable behavior/artifact
  evidence, but any prototype that executed ComfyUI is nonconforming and does not
  establish current Engine acceptance.
- LTX 2.3 optimized Engine-native work is uncommitted/in progress and has no runnable
  or accepted claim; LTX 2.5 optimized workflows remain research contracts pending
  Engine-native implementation.
- All future quantized paths must name the Kitchen version actually called directly by
  Engine; a central research Kitchen pin cannot be substituted into retained evidence.

## Closure checks retained

- Krea saved `enable_lora=false`: three-file base; enabled Darkbrush at 0.8 is a
  separate four-file mode.
- Qwen saved Lightning off: three-file 40-step/CFG-4 standard; four-file
  four-step/CFG-1 mode is separate.
- LTX 2.5 T2V selects six resources; FLF is a different graph.
- Wan 14 T2V/I2V/FLF are distinct operation contracts.
- Ideogram requires conditional and unconditional branches, encoder, and VAE.
- Z-Image Turbo INT8 is a three-file T2I contract.

## Mutable links

Mutable publisher/model-card, gated-access, tutorial, hosted-service, and legal pages
may remain for discovery or access. They are never the sole authority for artifact
identity, saved behavior, node semantics, or dispatch.

## Changed-link rule

Resolve full commit objects and exact paths; reject guessed suffixes; preserve accepted
historical pins; label every remaining mutable link by purpose.
