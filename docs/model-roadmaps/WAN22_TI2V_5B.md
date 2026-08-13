# Wan 2.2 TI2V 5B roadmap

Last authority audit: **2026-08-13**

Engine policy baseline:
[`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2)

Follow [COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md).

## Authority map

| Surface | Authority |
| --- | --- |
| weights/architecture/license | Wan publisher source [`42bf4cfaa384bc21833865abc2f9e6c0e67233dc`](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc), dense Reference, exact split artifacts |
| saved topology/defaults | pinned [`text_to_video_wan22_5B.json`](https://github.com/comfyanonymous/ComfyUI_examples/blob/f9431bb000ce792094ff345446e22cac1ea6cef3/wan22/text_to_video_wan22_5B.json), blob `25dc2512aec510be1d569226aa8598c42b9e0fbe`; and [`image_to_video_wan22_5B.json`](https://github.com/comfyanonymous/ComfyUI_examples/blob/f9431bb000ce792094ff345446e22cac1ea6cef3/wan22/image_to_video_wan22_5B.json), blob `6160b103fd0f752719aa7360961d7ba3cec89e34` |
| node behavior / dispatch | ComfyUI source [`eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f`](https://github.com/Comfy-Org/ComfyUI/tree/eb4a7b4fcfcedba4aba66b7297de4137ce0e1b2f) is historical research; future conforming runtime calls Kitchen/native primitives directly |
| acceptance/tier | only a conforming Engine-native public-API implementation may own acceptance |

## Architecture correction

T2V and required-image I2V share an exact three-file source contract, but Engine must
not execute these workflows through ComfyUI.

Any prior prototype that launched/imported ComfyUI or submitted a graph is
**nonconforming historical evidence**. Its artifacts, settings, crop/anchor
observations, and produced media may inform independent fixtures, but it does not make
the current split path runnable, Hardware-proven, or Fallback under Engine policy.

Current conforming status: **Engine-native CPU/source candidate complete;
catalog execution remains gated pending target-hardware output acceptance**. The candidate owns typed T2V and
required-first-frame I2V recipes, exact resource validation, direct Kitchen dispatch
for the stored-FP8 UMT5 encoder, staged transformer/VAE residency, a fresh disposable
worker per job, cancellation, atomic MP4 publication, and observed output provenance.
It imports, launches, and submits nothing to ComfyUI.

## Exact split contract

| Resource identity | Bytes | SHA-256 |
| --- | ---: | --- |
| `Comfy-Org/Wan_2.2_ComfyUI_Repackaged@fb1388adc906ab39ffc26ee40e96b22886b56bc4` / `split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors` | 9,999,658,848 | `456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e` |
| `Comfy-Org/Wan_2.1_ComfyUI_repackaged@06e001fc51048fb03433a6fb25334de7836704a5` / `split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6,735,906,897 | `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68` |
| `Comfy-Org/Wan_2.2_ComfyUI_Repackaged@fb1388adc906ab39ffc26ee40e96b22886b56bc4` / `split_files/vae/wan2.2_vae.safetensors` | 1,409,400,960 | `e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156` |
The runnable recipe closure also includes a nine-file, 21,458,979-byte support shell
containing only the exact Diffusers configs and tokenizer files. The full installed
four-resource closure is **18,166,425,684 bytes** under
`models/wan22/ti2v-5b/`; every file was acquired through the ordinary Engine installer
and independently verified to have one filesystem link.

Saved behavior: 30 executed steps, CFG 5, `uni_pc`/`simple`, shift 8, denoise 1,
24 fps. The source requests a 31-step simple grid and removes the penultimate sigma
for UniPC. Its saved 41-frame value is a short preview setting; the same workflow
labels 121 frames as optimal, so Engine deliberately uses 121 as the product default
and fingerprints that deviation.
T2V accepts no image. I2V requires one image and preserves exact preprocessing and
first-latent anchoring.

Dense Reference: 34,203,021,834 bytes at
`b8fff7315c768468a5333511427288870b2e9635`; its 50-step contract is not
settings-equivalent.

## Landed Engine-native implementation

1. `wan-2-2-5b-ti2v.text-to-video.engine-stored-mixed` owns T2V with no image.
2. `wan-2-2-5b-ti2v.image-to-video.engine-stored-mixed` owns I2V with one required
   image; operation/tool mismatch is rejected while compiling the catalog.
3. Both bind the exact FP16 transformer and VAE, stored-FP8 UMT5, and bounded support
   closure. No role is optional or substituted.
4. The worker revalidates source/header/schema/plan identities before heavy imports,
   then runs direct Kitchen text matmuls with every one of 168 stored modules observed
   and zero fallback calls.
5. The 30 executed sigmas are independently pinned to the shifted 31-step source
   grid after UniPC's penultimate-sigma removal. The current Engine candidate maps
   `bh1`, order 3, CFG 5, and flow shift 8 onto Diffusers' UniPC implementation;
   target-GPU output acceptance must still validate that solver substitution.
6. The parent owns no tensors. A fresh Windows Job Object worker owns prompt encoding,
   guide preprocessing, denoising, decode, MP4 validation, cancellation, cleanup, and
   exact provenance.

## Remaining acceptance work

Run the two recipes through the public Engine API during the paired RTX 5080 session:
cold success, cancellation during observed generation, fresh-worker recovery,
single-worker VRAM sampling, native-dispatch evidence, deterministic fixed-seed replay,
and creator review. Until that evidence lands, both recipes remain **Experimental**;
the catalog intentionally reports them unavailable, preventing an unaccepted path
from being advertised as runnable;
the dense BF16 recipe remains **Reference** and is reserved for a batched rental study.

A future LoRA mode is separate and must be requalified in the Engine-native runtime;
it is not part of the present implementation.

Stop on ComfyUI dependency, optional-image ambiguity, graph drift, incomplete mapping,
hidden fallback, false availability, assumed metadata, or prototype evidence being
promoted as current acceptance.
