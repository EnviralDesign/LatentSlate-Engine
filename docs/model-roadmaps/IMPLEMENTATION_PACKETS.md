# Cross-family implementation packets

Last authority audit: **2026-08-13**

Engine source audited:
[`bde267f5f5b772f52e5b43a394de11b28465459c`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/bde267f5f5b772f52e5b43a394de11b28465459c)

Every packet is subordinate to
[COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md) and the
[roadmap preflight](./README.md#implementation-agent-preflight).

No packet may import, launch, proxy, require, or submit work to ComfyUI. Official
workflows and ComfyUI source are test authorities only. Engine implements typed native
orchestration and calls Kitchen directly inside Engine-owned workers.

## Priority packets

| Priority | Packet | Existing reuse | Stop conditions |
| ---: | --- | --- | --- |
| 1 | Wan 14B: broaden I2V/T2V/FLF creator and switching evidence | accepted Engine-native expert runtimes, Engine workers, direct stored/Kitchen dispatch | lost expert identity, fallback, unobserved cleanup |
| 2 | Klein 4B/9B: cancellation and clean recovery | golden Engine-native stored/Kitchen runtimes | fallback, stale caches, poisoned recovery |
| 3 | Klein 4B/9B: official two-reference behavior | normalized disabled example, ordered Engine cache | order/preprocessing drift |
| 4 | Wan 5B: build conforming Engine-native split T2V/I2V runtime | exact three-file artifact contract and `f9431bb...` behavioral evidence | any ComfyUI runtime dependency, incomplete map, unsupported dispatch |
| 5 | Wan 5B: target acceptance | packet 4 | previous nonconforming prototype evidence may not substitute |
| 6 | LTX 2.3 T2V/I2V: Engine-native Dev-FP8 + Distilled-LoRA runtime | BF16 source contract, Engine A/V/mux/lifecycle seams | gated closure, hidden conversion, A/V drift |
| 7 | LTX 2.3 FLF: Engine-native Distilled-FP8 endpoint runtime | packet 6 infrastructure, ordered endpoints | T2V/I2V lineage substituted for FLF |
| 8 | Z-Image Turbo: Engine-native INT8 T2I | Klein materializers, mixed-Qwen, direct Kitchen | unknown layout or fallback |
| 9 | H3: authenticate optimized FL2VA closure | BF16 CPU/source validator | gated identities or FL2VA/Ref2VA mixing |
| 10 | H3: Engine-native optimized T2VA/endpoints | packet 9 | no direct Kitchen/native proof or incomplete A/V closure |
| 11 | LTX 2.5: exact six-role T2V authority packet | LTX A/V and Engine-worker seams | gated identity/support gaps |
| 12 | LTX 2.5: Engine-native Kitchen-backed T2V | packet 11 | any ComfyUI process/graph dependency, missing enhancer/upscaler/audio |
| 13 | Qwen 2511: ordered-input standard INT8 runtime | Engine media schema, ConvRot, Qwen encoder | ambiguous order/count or fallback |
| 14 | Qwen 2511: fixed Lightning mode | packet 13 and exact LoRA | standard/Lightning switches not atomic |
| 15 | Krea 2: license and normalized saved-default packet | ConvRot and Qwen3-VL research | license rejection or switch ambiguity |
| 16 | Krea 2: Engine-native saved-default INT8 runtime | packet 15 | any ComfyUI dependency or incomplete direct dispatch |
| 17 | Ideogram 4: JSON and composite-license contract | Engine request/resource layers | missing branch or hidden expansion |
| 18 | Ideogram 4: Engine-native dual-branch INT8 runtime | packet 17, direct Kitchen | branch substitution or fallback |
| 19 | SDXL: product-value and embedded-graph extraction | Engine Diffusers lifecycle | no concrete Engine-owned value |
| 20 | Dense video Reference campaign on Vast | CPU/source-pinned Wan/LTX/H3 closures | local settings substituted for full Reference |

## Packet entry contract

Each agent receives one operation, publisher authority, exact workflow/blob,
ComfyUI source revision for behavioral study, Kitchen/header authority, complete
artifact closure, current Engine proof level, likely modules, tests, acceptance cases,
exclusions, and stop conditions.

The agent returns the normalized contract and independent fixtures before coding.

## Universal stop conditions

Stop when:

- immutable workflow/source/artifact/license facts cannot be resolved;
- active and disabled branches cannot be distinguished;
- tensor maps, scales, sidecars, aliases, or fixed-LoRA targets are incomplete;
- direct Kitchen/native dispatch cannot be proven;
- a ComfyUI import/process/server/graph/plugin/workspace/folder dependency is proposed;
- output metadata is unknown or assumed;
- cancellation leaves an Engine worker, allocation, temp output, or poisoned cache;
- cataloged work is treated as runnable;
- dense Reference settings are weakened to fit local hardware.
