# ComfyUI comparison baselines

These bounded JSON records capture observed reference behavior from selected
ComfyUI workflows for Engine parity work. They retain workflow and
output hashes, normalized graph facts, environment versions, timing, and
machine-wide memory observations. They omit prompts, absolute paths,
credentials, full logs, and job identifiers.

A baseline is not Engine acceptance. It is an external behavioral oracle used
to compare topology, defaults, output metadata, execution cost, and optional
optimization modes. Engine remains a typed native runtime and never submits
these graphs.

For frontend workflows containing subgraphs, preserve the source UI JSON. When
the MCP/comfy-cli converter cannot resolve promoted inputs, use a disposable
repaired UI copy for one successful reference run, capture the exact flat API
prompt emitted by ComfyUI history, and use that flat prompt for controlled MCP
repetitions. Record both source and derived hashes.
