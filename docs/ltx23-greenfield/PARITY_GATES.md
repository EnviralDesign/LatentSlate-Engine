# LTX 2.3 parity gates

The routine parity canvas is 512 by 512. The final-quality canvas is 768 by
768. A 1280 by 704 canvas is not an acceptance gate.

Use this exact T2V prompt for the fixed comparison:

> “A tiny silver wind-up bird flutters across a sunlit workshop table, its metal
> wings clicking softly as the camera follows at eye level.”

Pinned Comfy 512 baseline:

- Cold: 141.709 s
- Warm: 44.157 s
- RAM peak: 60.484 / 61.393 GiB
- GPU peak: 15,097 / 15,074 MiB

The initial parity objective is approximately no more than 10 percent slower
than pinned Comfy, with no materially worse RAM/VRAM behavior and a correct,
complete media contract.

Establish and prove paths in this order:

1. T2V
2. I2V
3. FLF

I2V and FLF parity baselines are **TO BE ESTABLISHED** from pinned Comfy. The
reset contains no benchmark evidence or prior measurement implementation.
