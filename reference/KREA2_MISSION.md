# Native Krea 2 mission

## Objective and protected state

Deliver measured native Krea 2 Turbo text-to-image through ordinary Engine
Recipe/catalog/jobs and generic LatentSlate image/version/provenance contracts.
Preserve the original checkouts and unrelated work. Push feature branches only.
RAW/training, style-image/reference conditioning and unrelated architecture are
outside this milestone. The user-designated Home Lab conversation reviews the
final candidate; a review does not authorize a push to main.

## Current phase: complete at the reviewed feature-candidate boundary

Implementation and all local/live acceptance are complete. Product source is
`fc134b21ad87edcbbc1af56462b6b820ab0cde68`; subsequent packet commits contain
evidence only. The [evidence packet](comfy/krea2/README.md) is the current result,
including all measurements, exceptions, compatibility limits and screenshots.

| Repository | Feature branch | Source SHA | Isolated checkout |
|---|---|---|---|
| EnviralDesign/LatentSlate-Engine | codex/krea2-native | fc134b21ad87edcbbc1af56462b6b820ab0cde68 | C:/repos/LatentSlate-Engine-Krea2 |
| EnviralDesign/LatentSlate | codex/krea2-native | 2e312d804694d745687fdcf86f669298be7a73ea | C:/repos/LatentSlate-Krea2 |

Engine started from `8c6e0b6f175074dd96fac75bc60e75546bf9f785`.
Desktop requires no Rust change. Both feature refs are pushed. Original Engine
uncommitted authoring/Klein work is preserved. The separately requested Klein
FP8 adapter scale fix was verified in the original checkout and retained in
feature commit `fd886a9029cad28608d18002167d9e7233835d37`.

## Oracle and result

Frozen Comfy 0.34.0 commit `12d5279438bfefc058a269eae805ceab6047777f`,
curated template commit `5ec2c667b540389b74a3442947ab2cdfe6e0c59e`.
Exact model sources/hashes, frontend API export and boundary captures are in
`comfy/krea2/`; full local traces and generated corpus remain ignored under
`reference/local/krea2/`.

Baseline matches every measured coarse boundary and RGB pixels across thirteen
geometries. BF16, INT8 ConvRot, NVFP4, MXFP8 and community W4A8 match their own
encoding's reference. Every alternate passes its measured resource gates.
The FP8 primary warm result remains a numerical FAIL (+15.26%); Home Lab
explicitly accepted the narrow Windows/RTX5080/frozen-fixture timing exception.
INT8's initial single-cold failure is retained alongside its passing three-start
median. No alternate inherits the FP8 exception.

Three official style LoRAs pass the twenty-case state/strength/order matrix.
FP8 LoRAs are mechanically validated and deterministic, with explicitly reviewed
untouched-Comfy pixel differences due to resident requantization. BF16 with
Darkbrush matches exactly. INT8/NVFP4/MXFP8/W4A8 adapters are unsupported.

Live ordered-adapter Recipe Studio creation, reordering, pinned HF acquisition,
export/import, ordinary jobs, immutable revision/schema provenance, restart and
actual worker replacement/release all pass. Generic desktop generation produced
two persisted image versions and a reopened timeline preview. Original managed
Engine definition and desktop project/provider state are restored.

## Acceptance ledger

- [x] Curated local Comfy oracle and official installed/hash-verified artifacts.
- [x] Native family, identity/request, capability/policy, built-in and lifecycle.
- [x] Frozen boundary/output parity and reviewed time/RAM/VRAM results.
- [x] Prior built-in/catalog contracts preserved; ordinary jobs and provenance.
- [x] Recipe Studio/local and pinned sources/materialization/export/import.
- [x] Generic desktop image/version/preview/persisted-provenance and layouts.
- [x] Alternate formats and three credible LoRAs; unsupported modes explicit.
- [x] Windows Engine: 441 tests and 27 subtests pass.
- [x] Native-free CI passes on Linux, macOS and Windows.
- [x] Desktop fmt/check, 228 tests (one ignored), release build/stage/deploy pass.
- [x] Complete public evidence packet and pushed feature source checkpoints.
- [x] Clean evidence checkpoint pushed; independent Home Lab review accepted.

## Final review and stopping point

Home Lab completed its read-only review on 2026-09-13. Reviewed evidence head:
`33f2f53efb23897a6fb0835607ecf743e44da2b1`; tested product source remains
`fc134b21ad87edcbbc1af56462b6b820ab0cde68`. The review found no concrete
acceptance blocker and accepted the feature candidate for guarded later
canonization. It independently inspected pushed source, ancestry and portable
CI; local GPU measurements and worktree cleanliness remain Codex's verified
evidence. See [verification.json](comfy/krea2/verification.json).

Review conversation:
https://chatgpt.com/g/g-p-6776fc5af4b4819194c3b58779fb0a7c/c/6a971736-5f74-83ea-8dc6-b6d013796058

This final status commit changes only evidence records. No experiment, source
change or additional scope is required. Main remains untouched. A later main
push requires Lucas's authorization and fresh ancestry/cleanliness checks; it
must account for the separately authorized Klein scale fix in the Engine stack.
