# Source pin and closure audit

Last corrected: **2026-08-12**

This ledger records the factual correction pass applied after the first roadmap tranche. It is not a substitute for the family roadmaps; it documents the cross-file source-pin rules, invalid references that were removed, and closure-completeness checks that were repeated across the portfolio.

## Verified repository pins

The following full 40-character GitHub commits were resolved through the GitHub API before being used as immutable links:

| Repository | Verified commit |
| --- | --- |
| LatentSlate Engine architecture baseline | `b2481702d7b888a8553a4ce8b3302258a7a1fd96` |
| LatentSlate Engine Wan 5B reconciliation source | `f59c3970d7ca72d63533f9eb37d8f0dcc91b2810` |
| Comfy workflow templates | `2b7f823136606344f0bccce249898d771b809aa1` |
| ComfyUI audited source | `725e6ec60621c6f001af04769173e7dbb3c53541` |
| Comfy Kitchen | `78e6dd22fe4ebe7bde5062e050a045dc3a244ee4` |
| comfy-model-tools ConvRot source | `1fe341bb8a4e46f161a978b5faa2412d8c39c768` |
| ComfyUI examples | `f9431bb000ce792094ff345446e22cac1ea6cef3` |
| Wan 2.2 source | `42bf4cfaa384bc21833865abc2f9e6c0e67233dc` |
| LTX-2 source | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |
| MiniMax H3 source | `fa6891ff7cdaaa03fa4497e89ac64ff169219acf` |
| Ideogram 4 source | `990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2` |
| Krea 2 source/license snapshot | `db3984fbc6e13b34c0064990fc2d95ac64d00058` |

## Invalid pins corrected

| Invalid or guessed reference | Correct treatment |
| --- | --- |
| `725e6ecf9f11561da664cae996e0ab27ed7bfc6c` | replaced with existing ComfyUI commit `725e6ec60621c6f001af04769173e7dbb3c53541` |
| `9816d220021ab526e2cc1700a68b68d1b72d961c` | removed; replaced with existing Comfy Kitchen commit `78e6dd22fe4ebe7bde5062e050a045dc3a244ee4` |
| `1fe341001c27e8fe7e0450e8ce7fd3333d97c34c` | replaced with existing comfy-model-tools commit `1fe341bb8a4e46f161a978b5faa2412d8c39c768` |
| `5ff76a42c5f8fa15a8b18cde8c96cb3d2052b1ce` | removed; the audited ComfyUI examples snapshot is `f9431bb000ce792094ff345446e22cac1ea6cef3` |
| short SHA prefixes expanded with unverified suffixes | removed; links now use an existing full commit or an explicitly mutable discovery page |
| gated or mutable Hugging Face `main` pages presented as immutable | changed to mutable discovery links unless exact file revision, filename, bytes, and SHA-256 were independently resolved |

A mutable model card or repository page is allowed for discovery, license, or publisher-intent evidence only when the text labels that limitation. It is never an acquisition lock.

## Workflow closure rule

An official parity closure contains every artifact actively wired by the pinned workflow, including fixed LoRAs, prompt-assistant models, unconditional branches, VAEs, audio components, and stage-specific weights. Notes that merely list optional downloads are not enough; the active node connections and switch defaults decide the saved graph closure.

The correction pass fetched and inspected the pinned workflow JSON for the family graphs cited by the roadmaps, including the Klein 4B/9B graphs, Krea 2 INT8 T2I and style-reference graphs, Qwen Image Edit 2511 standard/INT8 graphs, Ideogram 4 standard/INT8 graphs, Z-Image Base/Turbo graphs, Wan 14B examples and FLF graph, LTX 2.3/2.5 FLF graphs, and MiniMax H3 T2V/I2V/R2V graphs.

## Krea 2 correction

The pinned Krea INT8 T2I workflow actively applies `krea2_darkbrush.safetensors` through `LoraLoaderModelOnly` at strength `0.8`. Exact official parity is therefore a four-file closure:

| Artifact | Bytes |
| --- | ---: |
| `krea2_turbo_int8_convrot.safetensors` | 13,492,686,496 |
| `qwen3vl_4b_fp8_scaled.safetensors` | 5,242,467,968 |
| `qwen_image_vae.safetensors` | 253,806,246 |
| `krea2_darkbrush.safetensors` | 469,291,992 |
| **Official four-file total** | **19,458,252,702** |

The LoRA-disabled subtotal is `18,988,960,710` bytes. That three-file path is retained only as a deliberate Alternate for isolating base Turbo behavior and is explicitly marked non-parity.

## Verification policy

Before a roadmap can call a source immutable or a closure exact:

1. the GitHub commit or blob path must resolve;
2. a Hugging Face artifact must have a concrete revision, exact filename, byte count, and SHA-256/LFS identity when available;
3. every actively wired workflow artifact must appear in the closure and arithmetic;
4. unresolved gated identities remain blockers rather than inferred values;
5. relative Markdown targets and heading anchors must resolve;
6. `git diff --check` must be clean, with no trailing-space hard breaks;
7. changed paths must remain inside `docs/model-roadmaps/**` for this branch.
