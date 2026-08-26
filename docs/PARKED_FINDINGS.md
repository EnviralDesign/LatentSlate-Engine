# Parked Findings — Explicit Approval Required

This file records evidence-backed observations that may matter in a later,
explicitly authorized task.  Nothing listed here is implementation scope,
a requirement, or permission to change the Engine, fixtures, or acceptance
criteria.  Re-evaluate each item against the active goal before acting on it.

## LTX 2.3 temporal-frame boundary

The canonical 30 fps T2V API fixture requests 5 seconds through 151 temporal
positions (`5 * 30 + 1`).  LTX's empty-video-latent path maps this to 19 latent
time positions using `((length - 1) // 8) + 1`; decoding therefore produces
`8 * (19 - 1) + 1 = 145` video frames, or 4.833 seconds at 30 fps.

If a later product requirement demands an exact or at-least-five-second
delivered MP4, decide and validate an intentional valid `8n + 1` frame-count
policy (including whether to round up) against the pinned Comfy reference.
