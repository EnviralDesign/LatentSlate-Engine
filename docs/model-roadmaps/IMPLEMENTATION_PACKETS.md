# Cross-family implementation packets

Last authority audit: **2026-08-13**

Engine policy baseline:
[`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2)

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
| 4 | Wan 5B: paired target acceptance for landed Engine-native T2V/I2V | exact four-resource closure, direct-Kitchen CPU/source runtime, disposable worker | previous nonconforming prototype evidence may not substitute; dispatch/output/lifecycle must be observed |
| 5 | LTX 2.3: paired target acceptance for landed Dev T2V/I2V and Distilled FLF | exact stored closures, direct-Kitchen CPU/source runtimes, ordered endpoints | hidden conversion, A/V drift, operation substitution, unobserved cleanup |
| 6 | Z-Image Turbo: Engine-native INT8 T2I | Klein materializers, mixed-Qwen, direct Kitchen | unknown layout or fallback |
| 7 | H3: authenticate optimized FL2VA closure | BF16 CPU/source validator | gated identities or FL2VA/Ref2VA mixing |
| 8 | H3: Engine-native optimized T2VA/endpoints | packet 7 | no direct Kitchen/native proof or incomplete A/V closure |
| 9 | LTX 2.5: exact six-role T2V authority packet | LTX A/V and Engine-worker seams | gated identity/support gaps |
| 10 | LTX 2.5: Engine-native Kitchen-backed T2V | packet 9 | any ComfyUI process/graph dependency, missing enhancer/upscaler/audio |
| 11 | Qwen 2511: ordered-input standard INT8 runtime | Engine media schema, ConvRot, Qwen encoder | ambiguous order/count or fallback |
| 12 | Qwen 2511: fixed Lightning mode | packet 11 and exact LoRA | standard/Lightning switches not atomic |
| 13 | Krea 2: license and normalized saved-default packet | ConvRot and Qwen3-VL research | license rejection or switch ambiguity |
| 14 | Krea 2: Engine-native saved-default INT8 runtime | packet 13 | any ComfyUI dependency or incomplete direct dispatch |
| 15 | Ideogram 4: JSON and composite-license contract | Engine request/resource layers | missing branch or hidden expansion |
| 16 | Ideogram 4: Engine-native dual-branch INT8 runtime | packet 15, direct Kitchen | branch substitution or fallback |
| 17 | SDXL: product-value and embedded-graph extraction | Engine Diffusers lifecycle | no concrete Engine-owned value |
| 18 | Dense video Reference campaign on Vast | CPU/source-pinned Wan/LTX/H3 closures | local settings substituted for full Reference |

LTX 2.3 and Wan 5B optimized Engine-native CPU/source implementations are landed.
They remain Experimental until the paired RTX acceptance packets above are complete.

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
