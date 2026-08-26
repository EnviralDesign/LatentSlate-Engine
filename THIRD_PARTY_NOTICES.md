# Third-party notices

LatentSlate Engine preserves the license notices of its direct runtime
dependencies. Dependency packages remain separately copyrighted and licensed
by their respective authors.

- **comfy-aimdo 0.4.15** — Copyright its contributors; GNU General Public
  License, version 3. Source: <https://github.com/Comfy-Org/comfy-aimdo>.
- **Comfy Kitchen 0.2.31** — Copyright its contributors; Apache License 2.0.
  Source: <https://github.com/Comfy-Org/ComfyUI-Kitchen>.
- **ComfyUI v0.34.0** — Copyright its contributors; GNU General Public License,
  version 3. The LTX 2.3 AV direct residency and Gemma text substrates narrowly
  adapt code from commit `12d5279438bfefc058a269eae805ceab6047777f`
  (`model_patcher.py`, `ops.py`, `model_prefetch.py`, `memory_management.py`, and
  `text_encoders/llama.py`) while replacing ComfyUI graph/global-manager policy
  with Engine's authenticated model context.
  Source: <https://github.com/Comfy-Org/ComfyUI>.

The complete dependency graph and exact pinned versions are recorded in
`uv.lock`. This notice does not replace any license file distributed inside a
dependency wheel.
