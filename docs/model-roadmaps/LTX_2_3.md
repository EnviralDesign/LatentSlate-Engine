# LTX 2.3 roadmap

Last authority audit: **2026-08-15**

Engine policy baseline:
[`b1def580cf835356f57a82d46b17055d05a215a2`](https://github.com/EnviralDesign/LatentSlate-Engine/tree/b1def580cf835356f57a82d46b17055d05a215a2)

Follow [COMFY_ENGINE_POLICY.md](../COMFY_ENGINE_POLICY.md).

## Authority map

| Surface | Authority |
| --- | --- |
| weights/architecture/license | Lightricks sources and Engine’s immutable BF16 closure at upstream `432e0d3c2d1769aaa4d295f9243f7062bf6b47ee` |
| saved topology/defaults | official T2V, I2V, and FLF workflows at [`2b7f823136606344f0bccce249898d771b809aa1`](https://github.com/Comfy-Org/workflow_templates/tree/2b7f823136606344f0bccce249898d771b809aa1) |
| node behavior / dispatch | ComfyUI source [`725e6ec60621c6f001af04769173e7dbb3c53541`](https://github.com/Comfy-Org/ComfyUI/tree/725e6ec60621c6f001af04769173e7dbb3c53541) for research only; Kitchen [`78e6dd22fe4ebe7bde5062e050a045dc3a244ee4`](https://github.com/Comfy-Org/comfy-kitchen/tree/78e6dd22fe4ebe7bde5062e050a045dc3a244ee4) for direct Engine dispatch |
| acceptance/tier | all three optimized operations are narrow Hardware-proven Recommended through LatentSlate-originated Engine public-API A/V, exact dispatch, and lifecycle evidence; BF16 remains structural only |

## Decision

Current main owns a strong native BF16 structural Reference, not a practical local
product path. One bounded RTX 5080 OOM is enough; dense output belongs on Vast.

Practical source contracts:

- T2V and first-frame I2V: Dev FP8 plus fixed Distilled LoRA;
- FLF: Distilled FP8 with ordered endpoints, a separate graph/transformer line.

The pinned workflow widgets default to **1280×720**. Engine deliberately exposes
**768×512** as the initial 16 GB acceptance preset: Dev's two-stage Diffusers path
requires the final dimensions to be divisible by 64, while 720 is not. Both values
are carried in variant provenance; this is an explicit compatibility/footprint
deviation, not an inferred workflow default.

All three Engine-native optimized operations are **narrow Hardware-proven
Recommended**: Dev T2V, strict first-frame Dev I2V, and ordered-endpoint Distilled
FLF. Recommended is operation-local here: FLF is its own typed operation, not a quality
alternate to T2V/I2V. The saved workflow contracts remain the topology/default oracle,
but no ComfyUI process or graph participates and no pixel/latent parity is claimed.

## BF16 truth

The exact 50-file closure totals **94,977,693,482 bytes** and includes transformer,
encoder/tokenizer, connectors, scheduler, video/audio VAEs, vocoder, and configs. It
fixes 24 fps, 8 steps/CFG 1, `8n+1` frames, aligned dimensions, synchronized 48 kHz
stereo, typed operations, and Engine disposable-worker cancellation boundaries.

Proof: **Cataloged / structurally validated Reference, one bounded local OOM**.

## Optimized implementation truth

The three operations bind their exact workflow revision and raw JSON SHA-256, all
active A/V resources, operation-specific fixed LoRAs (none for FLF), stored-FP8/NVFP4
header contracts, additive LoRA targets, direct Kitchen dispatch, Engine-owned typed
orchestration, disposable workers, cancellation, mux, output probing, and provenance.
The ordinary optimized
profile is **72,529,224,527 bytes** across seven canonical resources; each installed
file has one NTFS identity and acquisition rejects staged multiply-linked files.

The Pro/Comfy-derived source timing contract is preserved directly in Engine: 129
video frames at 24 fps produce 5.375 seconds of H.264 plus 48 kHz stereo AAC. The
accepted audio proof is `N=129`, latent `L=134`, mel `M=533`, decoded `255840`, target
`258000`, pad `2160`, policy `source_derived_exact_duration_v1`. No workflow is
submitted to ComfyUI. T2V/I2V and FLF must not share the wrong lineage.

| Operation | LatentSlate app job / Engine job | Output proof |
| --- | --- | --- |
| Dev T2V cold | `625a798e-5afc-4393-a5b1-ef5bb9f38d2f` / `7a03e1f2-cf83-4a19-986a-4612ddad4624` | 1,331,572 B; SHA-256 `2ddccdf05f4167c5c558a96eb593b9ced6cead703ffc0156ca3adbd33dc3a711` |
| Dev T2V recovery | `ddcca150-cadc-4394-9aec-3765e0c6d96d` / `f7548f73-7599-4585-9560-1e8dcf49f69e` | fresh worker; 1,225,703 B; SHA-256 `f5bdf109d0676ccd96f2a6afb22e4d6e13fe64cc992b6abdf1a18c3225617634` |
| Dev I2V, strict one start image | `94dde246-912b-4468-81c3-f27653f14d7a` / `02bb5e57-f342-4a49-b085-a1d1cdf46602` | 1,126,917 B; SHA-256 `7a0d847c3d9cf5c48faa221ec3d95e29fc12f8cfe4370f60139e77be15a5142a` |
| Distilled FLF, strict ordered endpoints | `cabb3a3a-64a6-4c93-9961-97b4d8b88926` / `263f2b6a-3843-4605-9937-5397961712b8` | 801,222 B; SHA-256 `0728298a9bef42468cd4d6cc433f67acdc92da5b79c45eb5e195f00fc9448ff7`; intentionally no LoRAs |

Dev T2V/I2V recorded exact Kitchen `1496/1496` module closure with `16456` native
calls; Distilled FLF recorded `1462/1462` with `11696` native calls. Reject/fallback
counts were zero. Cancel job `d6b22fe5-e06e-499b-98dd-55bec97cb2ae` /
`78134062-206e-4501-b191-e2efad9e80d5` stopped at 54%, emitted no artifact, and left
the output tree empty before the fresh-worker T2V recovery.

## Next slices

1. broaden prompt, guide, endpoint, and switching coverage while retaining exact A/V,
   dispatch, and lifecycle proof;
2. optional batched dense BF16 Vast comparison, without calling it locally accepted;
3. no new 2.3 feature without specific compatibility value.

Stop on ComfyUI dependency, partial closure, lineage
substitution, hidden conversion/fallback, assumed A/V metadata, or unobserved cleanup.
