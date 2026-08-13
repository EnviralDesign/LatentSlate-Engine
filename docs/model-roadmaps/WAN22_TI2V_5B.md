# Wan 2.2 TI2V 5B roadmap

Last authority audit: **2026-08-13**

Engine source audited:
[`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

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

Current conforming status: **artifact/workflow contract known; Engine-native runtime
not implemented or accepted**.

## Exact split contract

| Resource identity | Bytes | SHA-256 |
| --- | ---: | --- |
| `Comfy-Org/Wan_2.2_ComfyUI_Repackaged@fb1388adc906ab39ffc26ee40e96b22886b56bc4` / `split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors` | 9,999,658,848 | `456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e` |
| `Comfy-Org/Wan_2.1_ComfyUI_repackaged@06e001fc51048fb03433a6fb25334de7836704a5` / `split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6,735,906,897 | `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68` |
| `Comfy-Org/Wan_2.2_ComfyUI_Repackaged@fb1388adc906ab39ffc26ee40e96b22886b56bc4` / `split_files/vae/wan2.2_vae.safetensors` | 1,409,400,960 | `e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156` |
| total | 18,144,966,705 | — |

Saved behavior: 30 steps, CFG 5, `uni_pc`/`simple`, shift 8, denoise 1, 24 fps.
T2V accepts no image. I2V requires one image and preserves exact preprocessing and
first-latent anchoring.

Dense Reference: 34,203,021,834 bytes at
`b8fff7315c768468a5333511427288870b2e9635`; its 50-step contract is not
settings-equivalent.

## Required Engine-native implementation

1. normalize both workflows into independent fixtures;
2. author typed three-role resources and distinct T2V/I2V requests;
3. implement Engine-owned prompt encoding, image preprocessing, materialization,
   denoising, VAE decode, output, cancellation, and provenance;
4. call Kitchen directly for the scaled-FP8 encoder/native fast paths;
5. use Engine-owned disposable workers and no ComfyUI dependency;
6. prove exact dispatch, zero fallback, observed streams, switching, cancellation,
   recovery, memory, and creator quality.

A future LoRA mode is separate and must be requalified in the Engine-native runtime.

Stop on ComfyUI dependency, optional-image ambiguity, graph drift, incomplete mapping,
hidden fallback, false availability, assumed metadata, or prototype evidence being
promoted as current acceptance.
