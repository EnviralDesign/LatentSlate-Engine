# Wan 2.2 TI2V 5B roadmap

Last authority audit: **2026-08-13**

Engine source audited: [`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

Follow the shared [authority policy and implementation preflight](./README.md).

## Authority map

| Surface | Authority |
| --- | --- |
| Weights/architecture/lineage/license | Wan publisher source [`42bf4cfaa384bc21833865abc2f9e6c0e67233dc`](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc), official model cards, exact artifacts |
| Saved topology/defaults | accepted [`text_to_video_wan22_5B.json`](https://github.com/comfyanonymous/ComfyUI_examples/blob/f9431bb000ce792094ff345446e22cac1ea6cef3/wan22/text_to_video_wan22_5B.json) and [`image_to_video_wan22_5B.json`](https://github.com/comfyanonymous/ComfyUI_examples/blob/f9431bb000ce792094ff345446e22cac1ea6cef3/wan22/image_to_video_wan22_5B.json) at `f9431bb...` |
| Node/dispatch schema | accepted executable ComfyUI [`eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f`](https://github.com/Comfy-Org/ComfyUI/tree/eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f); Kitchen applies only to future native low-bit challengers |
| Acceptance/tier | Engine public-API artifacts, worker/graph provenance, source preprocessing, switching, cancellation/recovery, observed streams, memory, and creator review |

A newer research checkout does not rewrite accepted worker provenance. Mutable `master` workflow links are not used.

## Product decision

Two distinct accepted practical operations share one exact split closure:

- T2V has no image field;
- I2V requires exactly one source image.

Matching dense BF16 is **Reference**. The official split Comfy path is the accepted RTX 5080 **Fallback**. There is no Recommended native path and no value in reimplementing the accepted worker merely to rename it native.

## Accepted graph and closure

Active components:

| Role | Bytes | SHA-256 |
| --- | ---: | --- |
| FP16 transformer | 9,999,658,848 | `456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e` |
| scaled-FP8 UMT5 | 6,735,906,897 | `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68` |
| Wan 2.2 VAE | 1,409,400,960 | `e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156` |
| **total** | **18,144,966,705** | — |

Saved defaults: 30 steps, CFG 5, `uni_pc`/`simple`, shift 8, denoise 1, 24 fps. I2V connects one source through `Wan22ImageToVideoLatent`; T2V does not.

The dense complete Reference is 34,203,021,834 bytes at revision `b8fff7315c768468a5333511427288870b2e9635`. Its native 50-step topology is not settings-equivalent to the 30-step Comfy path.

## Engine proof preserved

Current main contains distinct T2V/I2V recipes, typed three-role closure, exact component/operation fingerprints, an isolated pinned worker with custom nodes disabled, and 1280-by-704 / 121-frame / 24-fps public-API outputs for both operations.

It also preserves T2V-to-I2V-to-T2V switching, cancellation/eviction/recovery, observed memory release, exact I2V source hash and center-crop/VAE-anchor provenance, and one fail-closed model-only LoRA with exact schema/rank and zero unmapped-key warnings.

Execution-cache byte equality is cache reuse, not an independent warm stochastic job. Accepted results establish narrow operational Fallback status, not broad quality superiority or BF16 equivalence.

Proof level: **Hardware-proven Fallback** for T2V and required-image I2V.

## Challenger and next work

Any FP8/NVFP4/other challenger starts from the same normalized operation graph or declares a separate topology. A ModelOpt NVFP4 path is its own Blackwell loader contract; no credible TI2V 5B ConvRot file was established; Turbo/Lightning/GGUF descendants are separate lineages or schedules. Kitchen authority applies only when native stored low-bit dispatch is claimed.

Next: broaden creator evidence for motion, people, animals, products, camera moves, transitions, signs/text, negative prompts, source identity, prompt/image balance, temporal texture, and freezing. Use changed-seed warm execution, exact pins, cancellation/recovery, observed streams, memory, hashes, and creator review.

Compare one practical challenger only for a measured need. Run a settings-equivalent dense BF16 study on high-memory hardware only when creator comparison justifies cost. Add LoRAs only for explicit demand through the existing exact gate.

Stop on graph drift, optional-image ambiguity, hidden worker fallback, false availability, assumed fps/frames, or cancellation without observed cleanup.
