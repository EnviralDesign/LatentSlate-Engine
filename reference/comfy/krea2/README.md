# Krea 2 Turbo T2I oracle

Captured 2026-09-13 on the existing RTX 5080 PyTorch baseline, ComfyUI
`12d5279438bfefc058a269eae805ceab6047777f` (0.34.0), Torch 2.11.0+cu130,
Python 3.13.12, comfy-aimdo 0.4.15, comfy-kitchen 0.2.31, frontend 1.51.9.
This is reference evidence, not a claim of native Engine support.

`curated-t2i.json` is the current `image_krea2_turbo_t2i` template from
Comfy-Org/workflow_templates commit `5ec2c667b540389b74a3442947ab2cdfe6e0c59e`.
The installed gallery copy and freshly fetched upstream file both hash to
`d04b37d8345b1bf6aa247ef182a4454e0221730cf353379af27b20e932e9f11f`.
The API fixture was exported by the real frontend `app.graphToPrompt()` after
changing only four model selector paths to the installed M-drive hierarchy.
The MCP/CLI frontend converter did not preserve outer promoted subgraph values
(seed and thinking differed); its first successful smoke is excluded. Execute
the frontend-exported `turbo-t2i-api.json` through comfy-local instead.

## Effective behavior

- Original detailed hand/martini-glass prompt is in the API fixture; the exact
  Qwen expansion is `expanded-prompt.txt`. Enhancement is on, thinking off,
  text seed 0, maximum 512 tokens, sampling temperature .7/top-k 64/top-p .95/
  min-p .05/repetition penalty 1.05/presence penalty 0.
- Image seed 594361197674106; Euler/simple, 8 steps, CFG 1, denoise 1.
  The negative branch is zeroed and CFG 1 executes conditional-only denoising.
- Resolution selector 1 megapixel, multiple 8: square 1024×1024; second fixture
  changes only aspect to 16:9 and produces 1368×768. Patch padding/cropping
  must preserve this non-multiple-of-16 output width.
- LoRA toggle is false: the optional darkbrush file is installed and hashed
  but the lazy branch does not load it. Enabled template trigger concatenation
  occurs **after** prompt enhancement; its current trigger/strength are fixture
  values, not a generic rule for all official LoRA cards.
- Text generation executes BF16. Conditioning executes FP32, taps layers
  2,5,8,…,35 into [1,12,184,2560], strips 34 prefix tokens and flattens to
  [1,150,30720]. Model input conditioning is then cast to BF16.
- Noise is CPU FP32 [1,16,1,128,128]. The nine measured sigma values, first
  model input/output, sampled latent, decoder input/output hashes and numerical
  summaries are in `boundary-summary.json`. The model uses BF16 activations,
  mixed FP8/full-precision matmul flags and dynamic AIMDO loading.
- VAE receives FP32 [1,16,1,128,128] and returns FP32
  [1,1,1024,1024,3] on CPU. SaveImage clamps/scales to 8-bit RGB PNG.

## Measurements and limitations

Uninstrumented cold execution: 58.509 seconds. Five warm executions changed
only image seed by +1 through +5, proving sampler execution despite text cache:
25.017, 24.101, 81.950, 20.260, 12.532 seconds; median **24.101 seconds**.
The large variance is unresolved; do not discard the slow run or use this
initial campaign as the final performance gate. Rebaseline equivalently before
the native comparison. Full per-run metrics are retained alongside this file.

Resource sampling every ~50 ms used total-device NVML dedicated memory and
the sum of process-tree working sets. Peaks were 16,411,357,184 bytes GPU
(15.285 GiB) and 21,368,778,752 bytes process RAM (19.901 GiB). Total-device
VRAM includes desktop/other occupants, so paired comparisons need equivalent
external occupancy. These are not Torch reserved-memory measurements.

A second fresh-process canonical run with temporary boundary capture produced
exactly the same PNG bytes and RGB pixels (MAE/RMSE/max error all zero).
The widescreen case completed as well. `output-manifest.json` records locations,
geometry and hashes; generated images/tensor captures are deliberately excluded
from Git. Local tensors and detailed histories live in
`reference/local/krea2/oracle/`. Temporary Comfy runtime instrumentation was
removed after capture; final performance must run after a clean restart.

Artifact sources, revisions, sizes and SHA256 are in `artifacts.json`. The
official weights use the Krea 2 Community License; source-code licensing is
separate. Primary authorities: [Krea](https://github.com/krea-ai/krea-2),
[Comfy weights](https://huggingface.co/Comfy-Org/Krea-2), and
[Comfy tutorial](https://docs.comfy.org/tutorials/image/krea/krea-2).
