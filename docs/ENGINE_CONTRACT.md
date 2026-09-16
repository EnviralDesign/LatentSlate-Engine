# LatentSlate ↔ Engine product contract

This document records the external contract the Engine rebuild must preserve. It
does not prescribe internal architecture.

If this document conflicts with current LatentSlate, current LatentSlate source
is authoritative. The primary client implementation is:

`EnviralDesign/LatentSlate/src/providers/latentslate_engine.rs`

## Connection

Default local Engine URL:

`http://127.0.0.1:8765`

Bearer authentication is optional.

When `LATENTSLATE_ENGINE_TOKEN` is set, every `/v1/` route requires the same
bearer token, including catalog, uploads, polling, artifacts, cancellation, and
runtime release.

## HTTP surface currently consumed by LatentSlate

### `GET /v1/health`

Basic live connection test.

### `GET /v1/catalog`

Returns Engine version/protocol information and a list of tools.

LatentSlate consumes, at minimum, tool:

- `id`
- `key`
- `schema_revision`
- `schema_hash`
- `name`
- optional `description`
- `workflow_kind`
- output media type
- input descriptors
- optional canvas information
- optional timing information
- availability/unavailable reason

Supported input types currently include text, number, integer, boolean, choice,
image, video, and audio.

### `POST /v1/assets`

Multipart field: `file`.

Returns an asset containing a UUID `id`.

For media-valued job inputs, LatentSlate submits:

`{"type": "asset", "asset_id": "<uuid>"}`

rather than a shared-filesystem path.

### `POST /v1/jobs`

Request shape:

```json
{
  "tool_id": "<uuid>",
  "schema_revision": 2,
  "schema_hash": "<hash>",
  "inputs": {}
}
```

Returns a job object.

### `GET /v1/jobs/{id}`

Job statuses consumed by LatentSlate:

- `queued`
- `running`
- `succeeded`
- `failed`
- `canceled`

Jobs retain the existing overall `progress` value in the range 0..1. They may
also expose an additive `stage` object with a human-readable `label`, optional
determinate `progress` in the range 0..1, and optional short `detail` such as
`Step 2 of 4`. These values report completed native boundaries or sampling
iterations; they are not elapsed-time estimates. Clients that only consume the
existing status/progress/artifact fields remain compatible.

Current family reporting follows executed recipe boundaries: LTX reports
text/source/endpoint conditioning, first/second-pass or guided sampler steps,
spatial refinement where present, video/audio decode, audio synthesis, and MP4
encoding; Klein reports text/reference conditioning, model preparation, its
four distilled sampler steps, VAE decode, and PNG encoding; Wan reports
text/source/endpoint conditioning, high-noise steps 1–2, low-noise steps 1–2,
VAE decode, and MP4 encoding.

Successful jobs expose artifacts. An artifact requires:

- `role`
- `filename`
- `download_url`

The primary artifact is preferred when `role == "primary"`. Current LTX and
Wan tools publish one MP4 primary artifact and current Klein tools publish one
PNG primary artifact.

### `DELETE /v1/jobs/{id}`

Requests cancellation. Cancellation acknowledgement does not imply that native
GPU work is already quiescent; Engine owns safe termination semantics.

### `GET <artifact.download_url>`

Downloads generated media bytes.

### `DELETE /v1/runtime`

Explicitly releases active Engine runtime/model resources.

Release succeeds only while the native runtime is idle. A concurrent generation
returns `409` rather than releasing state that is still in use; LatentSlate
already prevents resource release while its generation queue is active.

## Stable LTX 2.3 public identities

These identities were extracted once from the historical Engine checkpoint and
are preserved here so the new implementation does not need to inspect the old
runtime.

### Text to Video

- ID: `46bdb57c-3b19-5397-8949-4e20ffe757c9`
- key: `ltx23.text_to_video`
- schema revision: `2`
- workflow kind: `text_to_video`
- output: video
- schema hash: `sha256:94f9397a5ff16d5101e81f62396c5c744f045799bcdbdf961b036ee8f0ac2c78`

Inputs:

- `prompt`
- `width`
- `height`
- `duration_seconds`
- `seed`

### Image to Video

- ID: `5d6e2d6f-216c-5f35-a4ec-1565d6e56ee7`
- key: `ltx23.image_to_video`
- schema revision: `3`
- workflow kind: `image_to_video`
- output: video
- schema hash: `sha256:be3be547dd665155e162d51a5bea089cfcb0da66116c6e58c1766af04679bb24`

Inputs:

- `prompt`
- `start_image`
- `width`
- `height`
- `duration_seconds`
- `seed`

### First/Last Frame to Video

- ID: `1a8f9c0b-410e-56e4-90de-23bcb9d644ca`
- key: `ltx23.first_last_frame_to_video`
- schema revision: `3`
- workflow kind: `first_frame_last_frame_video`
- output: video
- schema hash: `sha256:b58e76368b442ca723a0e2679db3b5b011870c4eeaba223704192d1190d9de1c`

Inputs:

- `prompt`
- `start_image`
- `end_image`
- `width`
- `height`
- `duration_seconds`
- `seed`

Both frame inputs are required for the FLF public operation.

Revision 3 declares `"image_dimensions": "match_output_canvas"` on I2V's
`start_image` and both FLF image descriptors. This optional input-level semantic
constraint means the uploaded image's pixel dimensions must equal the submitted
width/height. Absence imposes no source/target dimension relation. It is included
in schema hashing, unlike presentation-only timing. The rule was already enforced
by admission; publishing it lets consumers diagnose mismatches before execution.
UUIDs, input keys, and request values are unchanged. Revision-2 submissions must
refresh catalog identity; source bytes and stored inputs must not be rewritten.

Each LTX video tool also advertises additive timing metadata. The excerpt below
shows one row of the full 19-row `output_frame_counts` mapping:

```json
{
  "timing": {
    "fps": {"mode": "fixed", "value": 30.0},
    "duration_seconds": {
      "min": 1.0, "max": 10.0, "step": 0.5,
      "output_frame_counts": [{"duration_seconds": 1.0, "frame_count": 25}]
    }
  }
}
```

Timing metadata is presentation/output metadata and is deliberately excluded
from the request-schema hash. Adding it therefore does not change these three
LTX hashes. The `fps.mode` field leaves room for future fixed, editable, or
enumerated cadence metadata without changing the current request contract.

The optional mapping relates nominal request values to encoded frame counts;
output duration is frame count divided by fixed FPS. LTX preserves its native
temporal lattice: 1.0 requested seconds yields 25 frames, and 5.0 yields 145.
Consumers must retain nominal requests separately from actual media extent and
must not reverse-copy delivered duration into the request. An unmatched row has
no prediction. Without a mapping, fixed-FPS tools such as Wan retain the ordinary
duration-times-FPS output rule. No request revisions or hashes change.

The catalog advertises the established LTX product domain: T2V and I2V use
64-pixel width/height alignment, FLF uses 32-pixel alignment, every side is at
least 64 pixels, the maximum canvas area is 942,080 pixels, and duration is
1.0–10.0 seconds in 0.5-second increments. The dependent pixel-area rule and
unsigned 64-bit seed range remain server-authoritative.

Uploaded image assets are session-local request content with UUID identity.
Current LatentSlate sends imported still-image bytes without pixel resampling,
so I2V and FLF source images must already match the requested width and height.

Job submission returns promptly. One bounded FIFO queue feeds one active GPU
worker. Cancellation of a running native call is acknowledged immediately but
the job remains nonterminal until that call is quiescent; its output is then
discarded and never exposed as an artifact.

## Stable FLUX.2 Klein 9B public identities

These identities were created once for the first public Klein service surface
and are now stable product identities.

### Text to Image

- ID: `e7dcbbde-d58f-4354-ad36-b684b5c236f3`
- key: `flux2_klein9b.text_to_image`
- schema revision: `1`
- workflow kind: `text_to_image`
- output: image
- schema hash: `sha256:2e94d609c2db43e883da19fb0c73faa1bef7f3459c916760079f7cedd212c6b3`

Inputs:

- `prompt`
- `width`
- `height`
- `seed`

### Two-Image to Image

- ID: `a7489e73-3bb9-4bb9-888f-fa592c8f4430`
- key: `flux2_klein9b.two_image_to_image`
- schema revision: `1`
- workflow kind: `image_to_image`
- output: image
- schema hash: `sha256:d756bc62e593edd29f3c2c909f3c92fd22d10cb2fb44a2b51bdd93afdb605ed8`

Inputs:

- `prompt`
- `image_1`
- `image_2`
- `width`
- `height`
- `seed`

Both image inputs are required and ordered. The service preserves their uploaded
bytes; it does not require either source to match the requested target canvas.
EXIF transpose, RGB conversion, independent one-megapixel scaling, slot-specific
interpolation, and centered VAE-grid cropping remain owned by the accepted Klein
runtime.

Both tools require explicit target dimensions on a 16-pixel grid. Each side is
at least 256 pixels, area is at most 1,048,576 pixels, aspect ratio is at most
4:1, and seed is an unsigned 64-bit integer. Two-image source dimensions are
independent of this target geometry.

Availability is evaluated per family. Missing Klein artifacts do not disable
the three LTX tools, and missing LTX artifacts do not disable the two Klein
tools. Submission independently rejects an unavailable tool.

## Stable Wan 2.2 14B turbo public identities

These identities expose the three accepted Wan operations without adding
service-only recipe controls.

Revision 2 is a deliberate request-contract migration from public
`frame_count` to public `duration_seconds`. The UUIDs and keys are unchanged;
revision-1 submissions are stale and must refresh the catalog rather than
translating the old field client-side.

### Text to Video

- ID: `34e57585-95a3-4bb6-b3de-fca5dd924ba6`
- key: `wan2214b_turbo.text_to_video`
- schema revision: `2`
- workflow kind: `text_to_video`
- output: video
- schema hash: `sha256:4556b1e1b1ae9483ce25f2a90b45f0a3b709bff6e46b34b0b835507f81ef4f8e`

Inputs: `prompt`, `width`, `height`, `duration_seconds`, and `seed`.

### Image to Video

- ID: `aac35e26-08e7-400b-bf9b-dc389809ddd5`
- key: `wan2214b_turbo.image_to_video`
- schema revision: `2`
- workflow kind: `image_to_video`
- output: video
- schema hash: `sha256:8c2c935669909fa6e010369137025cbffff321e4789b2966a31d761303d48426`

Inputs: `prompt`, `start_image`, `width`, `height`, `duration_seconds`, and
`seed`.

### First/Last Frame to Video

- ID: `d0c202bf-7dd5-4df8-b116-f7633dc94cfe`
- key: `wan2214b_turbo.first_last_frame_to_video`
- schema revision: `2`
- workflow kind: `first_frame_last_frame_video`
- output: video
- schema hash: `sha256:9cf28f66f4a51f1631f4f527d26081bf72ba9644d453b1e6f65b34acbcf5601a`

Inputs: `prompt`, `start_image`, `end_image`, `width`, `height`,
`duration_seconds`, and `seed`. Both endpoint images are required and ordered.

All three tools use the accepted fixed 16 fps recipe. Target width and height
are on a 16-pixel grid, each side is at least 480 pixels, area is at most
921,600 pixels, and aspect ratio is at most 16:9. Duration is 1.0–5.0 seconds
in 0.25-second increments. Seed is an unsigned 64-bit integer. Internally Wan
derives `duration_seconds * 16 + 1` native samples, preserving its `4n+1`
lattice. The terminal sample remains present through conditioning, sampling,
and decode, then the family-owned writer emits the first `duration_seconds *
16` frames at 16 fps. A 5.0-second request therefore decodes 81 native samples
and publishes 80 display frames with an exact 5.0-second duration. This
half-open boundary prevents duplicate endpoint display across loops and chained
FLF segments.

Each Wan tool advertises:

```json
{
  "timing": {
    "fps": {"mode": "fixed", "value": 16.0},
    "duration_seconds": {"min": 1.0, "max": 5.0, "step": 0.25}
  }
}
```

I2V and FLF preserve uploaded source bytes and accept source dimensions
independent of the target canvas; Wan owns their center-crop/resize and causal
conditioning semantics.

Wan availability is operation-specific: T2V requires its high/low checkpoint
and LoRA pair, while I2V and FLF require the corresponding shared image-video
pair. Both groups also require the accepted UMT5 encoder and Wan VAE. Missing
Wan artifacts do not affect the five LTX/Klein tools.

## Krea 2 Turbo text to image

- ID: `fbdce87a-02cb-546e-98a3-4d268d35025b`
- key: `krea2_turbo.text_to_image`
- schema revision: `1`
- workflow kind: `text_to_image`; output: image
- inputs: `prompt`, `width`, `height`, `seed`

The native product performs automatic prompt enhancement followed by eight fixed
Euler/simple Turbo steps. Steps, guidance, and enhancement settings are not caller
controls. Canvas dimensions use an eight-pixel grid, each side is 256–2048 pixels,
area is at most 1,055,040 pixels, and aspect ratio is at most 4:1. The area includes
the rounded one-megapixel selector outputs, including 840 × 1256. All eight
curated aspect pairs and five freeform boundary cases match the frozen Comfy
pixels (`reference/comfy/krea2/geometry-parity.json`). Performance acceptance is
tracked separately in `reference/KREA2_MISSION.md`.

The built-in binds diffusion, text encoder, VAE, and a tokenizer directory. Set
`LATENTSLATE_KREA2_MODEL_ROOT` to a model tree containing:

- `diffusion_models/krea2/krea2_turbo_fp8_scaled.safetensors`
- `text_encoders/krea2/qwen3vl_4b_fp8_scaled.safetensors`
- `vae/qwen/qwen_image_vae.safetensors`
- `text_encoders/krea2/tokenizer/{vocab.json,merges.txt,tokenizer_config.json}`

The default root is `<engine home>/models`. User recipes can bind other local or
pinned remote model files through ordinary authoring; tokenizer directories are
local bindings. Krea uses the normal serialized family worker and `/v1/jobs`,
and does not change previous built-in identities or hashes.

User recipes mechanically support the measured Turbo BF16, scaled FP8, INT8
ConvRot, NVFP4, MXFP8 and community W4A8 encodings. The built-in stays on scaled
FP8. Ordered adapters with fixed strengths are validated on BF16 and FP8 only;
other encodings reject adapters. A recipe's `prompt_suffix` is appended after
enhancement, for style triggers. BF16 adapter pixels match the frozen reference;
FP8 adapters use deterministic immutable-base patching and carry the documented
Comfy residency/requantization caveat. These are measured artifact-specific
compatibility claims, not certification of arbitrary checkpoints or LoRAs.
See `reference/comfy/krea2/` for hashes, comparisons and resource measurements.

## Qwen Image Edit 2511

The curated non-Lightning edit operation is `qwen2511.edit`, tool ID
`b89fecef-a923-5108-8100-c49b7f469cdc`, schema revision 1. It accepts uploaded
`image_1`, independently optional `image_2` and `image_3`, `prompt` and `seed`.
Optional images may be omitted or null. There are no caller width/height fields:
image 1 determines the output canvas, and sparse image 1 + image 3 preserves
those logical roles. Success publishes one PNG through the ordinary job API.

The builtin resolves files below `<Engine home>/models` or the host's
`LATENTSLATE_QWEN2511_MODEL_ROOT`:

- `diffusion_models/qwen/qwen_image_edit_2511_fp8mixed.safetensors`
- `text_encoders/qwen/qwen_2.5_vl_7b_fp8_scaled.safetensors`
- `vae/qwen/qwen_image_vae.safetensors`
- `text_encoders/qwen/tokenizer/{vocab.json,merges.txt,tokenizer_config.json}`

Missing files disable this tool without disabling other families. Authored
Recipes use existing artifact selection and immutable revision admission;
checkpoint selection is independent of the curated builtin's file binding.
Certification covers the curated FP8mixed composition and the official BF16 and
INT8 ConvRot diffusion files, with fixed 40-step Euler / simple sampling, CFG 4,
shift 3.1, and no adapters. Alternate files use the authored Recipe's existing
`diffusion` artifact slot; changing representation preserves the public request
schema and changes the Recipe definition hash. Each representation matches its
own pinned Comfy reference. See `reference/comfy/qwen2511/formats/acceptance.json`
for exact artifact identities, service checks and measured resource limits.

Qwen job status retains the admitted Recipe identity. Successful `execution`
metadata includes diffusion/text/VAE/tokenizer SHA-256 and byte sizes, ordered
logical input slots with content identities, prompt, seed, effective settings
and output dimensions. File encoding is not inferred from a name or default.
Canceled jobs publish neither an artifact nor successful execution metadata.
Runtime release uses the existing `/v1/runtime` endpoint and exits the worker.

## Recipe authoring V0

The `/v1/authoring` API uses the same bearer boundary. Saved user recipes can
be enabled as ordinary catalog tools; authoring validation itself does not
probe native execution.

| Method | Path under `/v1/authoring` | Result |
| --- | --- | --- |
| GET | `/operations` | Nine family operation descriptors, inherent domains and field ownership |
| GET | `/builtins`, `/builtins/{key}` | Immutable certified definitions |
| POST | `/builtins/{key}/duplicate` | New UUID, revision 1; body `{}` or `{"name":"My recipe"}` |
| POST | `/validate` | Layered validation of a canonical document, without saving |
| GET | `/recipes`, `/recipes/{uuid}` | User recipes at their current heads |
| POST | `/recipes` | Create a canonical document with a new client-supplied UUID |
| PUT | `/recipes/{uuid}` | Save `{"base_revision":1,"document":{...}}`; stale heads return 409 |
| DELETE | `/recipes/{uuid}` | Permanently remove a user recipe, all revisions and publication state |
| GET | `/recipes/{uuid}/revisions`, `/recipes/{uuid}/revisions/{number}` | Immutable revision history |
| GET, PUT | `/recipes/{uuid}/publication` | Inspect or set host publication with `{"enabled":true}` |
| GET | `/recipes/{uuid}/export` | Download the current user head as one canonical JSON document |
| POST | `/imports/preview` | Validate `{"document":{...}}` and classify local UUID conflicts without writing |
| POST | `/imports` | Explicitly import `{"document":{...},"as_copy":false}`; copies require `as_copy:true` |
| GET, POST | `/roots` | List registered model folders; add an existing absolute directory with `{"path":"...","name":"optional"}` |
| DELETE | `/roots/{uuid}` | Unregister a folder without deleting files or changing recipes |
| POST | `/artifacts/refresh` | Rebuild the in-memory index of registered folders |
| GET | `/artifacts/search?operation=...&field=...&q=...&limit=50` | Contextual file/directory candidates, local references and structural checks |

A canonical document contains exactly `format_version` (currently `1`), `id`
(canonical UUID), `name`, `operation` (from introspection), and an ordered `fields`
array. Each field has `key`, `mode` (`fixed` or `exposed`), an optional `value`,
and optional `minimum`, `maximum`, `step`, `choices`, and `nullable` constraints.
Fixed policy requires a value. Exposed recipe parameters require defaults so
the existing family cross-field validator can check a concrete configuration.
Prompt/media fields are caller-owned: exposed without stored values or
constraints. Host state such as `device_index` is omitted.

Artifact values use `{"source":"local","path":"opaque local path string"}`.
Ordered artifact collections remain arrays. Wan adapter entries contain
`{"artifact":{"source":"local","path":"..."},"strength":1.0}`; high and low
phases remain separate. LTX retains parallel artifact/strength arrays, and
Klein LoRAs carry per-artifact strengths, like Wan adapters. Legacy artifact-only
LoRAs retain strength 1; strength changes are part of Klein model identity.
Artifact selection is fixed recipe content;
eligible recipe parameters may be fixed or exposed within the family domain.

Canonical UTF-8 JSON sorts object keys, uses compact separators, rejects
non-finite numbers, and preserves array order and exact string values. Saving,
loading and hashing never resolve, normalize, or rewrite artifact path strings.
`definition_hash` is SHA-256 of the canonical `format_version`, `operation`, and
`fields` object. It excludes display name, UUID, revision and timestamps, and is
independent of the execution catalog's request-schema hash.

Revision records include `parent_revision` (null for the first publication).
Published history follows this linear chain from head. Interrupted writes can
leave unused numbers, but their files never become readable revisions merely
because a later save publishes a higher number.

Validation reports `document_valid` (structural parsing), `recipe_compiles`
(existing family `CapabilitySet`/`Field`/`Recipe` rules), independent per-slot
`artifact_resolution`, and `execution_readiness`. Compilation checks defaults
and family cross-field rules; it does not promise every combination of later
caller overrides will pass. Existing request-time validation remains necessary.
Artifact resolution checks absolute paths on this host, file/directory kind,
and known companion files such as Klein tokenizer support. Missing or foreign
paths remain valid, storable recipe content but resolve as `unresolved`.

Execution readiness is only `blocked` or `unverified`, with `backend_checked`
always false. Complete local dependencies produce `unverified`, never `ready`.
Issues include stage, severity, code, document path, message and remediation;
policy errors do not suppress independently checkable missing-artifact issues.
Invalid structure/policy cannot be saved (422); unresolved dependencies can.

### Local artifact library and Recipe Studio

`/authoring` serves a static, dependency-free browser UI from this Engine. It
uses the operation descriptors and authoring API for built-in duplication,
artifact selection, fixed/exposed parameter editing, validation and explicit
revision saves. Built-ins remain read-only; caller inputs are informational,
and host bindings are omitted. A stale save preserves the draft and offers an
explicit reload of the current head. Current browsers with source-aware JSON
and `JSON.rawJSON` preserve the full unsigned 64-bit seed domain.

Operation descriptors carry family-owned `field_groups` and optional field
presentation labels. The `collection` layout groups ordered fields into shared
rows; the first field supplies the heading and section. Add, remove and reorder
act on every member together. LTX T2V/I2V use this for LoRA files and strengths,
including strength exposure and constraints within the same card. Layout metadata
does not enter saved recipes, hashes, or the execution schema; the existing
parallel fields and family validation remain authoritative.

The static page is public; all `/v1/authoring` requests retain the existing
optional bearer protection. The UI uses its serving origin and keeps an entered
token only in browser-tab session storage. No token is embedded in page assets
or saved in Engine authoring state.

Registered roots have stable UUIDs, optional display names and host-normalized
absolute paths. A missing folder stays registered and reports unavailable.
Roots aid discovery; they are not an artifact access boundary. Manual absolute
paths outside roots remain supported, and selecting a candidate stores its
returned local reference without further rewriting the recipe path.

Search uses family-owned artifact slot kinds and required companion files.
Results include root, relative/absolute paths, kind and structural checks;
overlapping roots are deduplicated. Matching ranks filename/path substrings
and abbreviated subsequences. The index builds lazily, is reused while typing,
and refreshes explicitly or when registered roots/availability change. Added
or removed files require refresh; displayed candidates are checked against
current local structure. These checks do not inspect tensors or establish
model architecture compatibility.

Search also returns folder facets with IDs, root/path labels, ancestors and counts
across all query matches, before folder filtering or the result limit. Repeated
`folder` query parameters select these IDs: candidates must belong to at least one
selected directory or its descendants. Same-named folders at different paths stay
distinct; a directory candidate includes itself. Recipe Studio keeps removable
path chips below the search box while the query changes. Choosing a subfolder
replaces its selected ancestor; separate branches can be searched together.
Clearing the chips restores full search.

### Pinned remote files and explicit materialization

File slots also accept portable remote references:

```json
{"source":"huggingface","repo":"owner/model","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","file":"folder/model.safetensors","sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}
```

```json
{"source":"civitai","model_version_id":9208,"file_id":8955,"sha256":"c74b4e810b030f6b75fde959e2db678c268d07115b85356d3c0138ba5eb42340"}
```

Civitai version/file IDs are positive integers. Only those IDs and SHA-256 are
canonical; names, filenames, URLs, API responses and authentication remain host
state. Reported uppercase SHA-256 values normalize to lowercase before pinning.

The revision is a full immutable 40-character commit; SHA-256 is 64 lowercase
hexadecimal characters. Mutable revisions, credentials, download URLs and cache
paths are not canonical fields. Directory slots retain local folder references.
An unmaterialized pinned file is valid policy with an unresolved dependency.

| Method | Path under `/v1/authoring` | Result |
| --- | --- | --- |
| GET | `/sources/huggingface` | Only `authentication_configured`; no token value |
| POST | `/sources/huggingface/pin` | Start a task from `{"url":"https://huggingface.co/owner/model/blob/main/file"}` or `{"repo":"owner/model","file":"file","revision":"main"}` |
| GET | `/sources/civitai` | Only `authentication_configured`; no token value |
| POST | `/sources/civitai/inspect` | Inspect `{"model_version_id":9208}` or a `{"url":"https://civitai.com/models/7808?modelVersionId=9208"}` model page; return selectable file metadata |
| POST | `/sources/civitai/pin` | Start a pin task from exact `{"model_version_id":9208,"file_id":8955}` |
| POST | `/materializations/plan` | Plan `{"documents":[...]}` using exact canonical definitions |
| POST | `/materializations` | Explicitly start acquisition for the same document envelope |
| GET | `/materializations/{id}` | Poll pin/materialization state, current-file byte progress, result or error |
| DELETE | `/materializations/{id}` | Request cancellation; poll until terminal |

Start returns 202 and a task ID. One artifact task runs at a time; concurrent
starts return 409. Tasks transition from `running` to `succeeded`, `failed` or
`canceled`. The most recent 32 task records are retained in memory; restart loses
tasks, but keeps verified cache files. Plans accept 1–32 documents and at most
256 unique dependencies, with consumers, local/cached/missing/unresolved states,
known download bytes and an explicit count of unknown sizes. Planning does not
download or automatically resolve mutable source identities.

Pinning uses official Hub metadata, then re-reads at the immutable commit. A
trusted content SHA-256 can pin without download; an ordinary Git SHA-1 ETag
cannot, so that pin task streams the file and computes SHA-256. Authentication
uses normal `huggingface_hub` host semantics (`HF_TOKEN`, saved Hub login and
`HF_HUB_DISABLE_IMPLICIT_TOKEN`); public sources work without authentication.

Civitai inspection displays version/file names, IDs, approximate size, type/format,
primary status and SHA availability. Pinning re-fetches the version and selects
the exact file ID. Missing SHA-256 triggers the existing download-to-hash task;
other hash algorithms are never substituted. Acquisition re-fetches metadata,
rejects changed SHA-256, and verifies downloaded bytes before publication. The
file-specific metadata download URL is transient. `sizeKB` is approximate display
metadata, so it is not used as an exact byte-length assertion.

`CIVITAI_TOKEN` is optional host state, sent only as a Bearer header to Civitai.
Cross-origin redirects receive no Civitai credentials; tokens are never added
to URLs. Authentication failures return sanitized diagnostics without response
bodies or signed URLs. An unauthenticated host can still import a valid canonical
reference, even when that file will require authentication to download.

Verified content is shared at
`ENGINE_HOME/artifacts/sha256/{first-two-digest-characters}/{digest}/blob`.
Filenames, repositories and source services do not affect cache identity: HF and
Civitai references with the same SHA share one dependency/cache object in either
acquisition order. Writes stream through temporary files, verify the digest and
exact size when available, fsync, then atomically
publish. Cancellation/failure never publishes partial content. Corrupt entries
can be repaired by explicit materialization; local files are never copied.
Cache checks rehash after restart or file-stat changes; normal execution forces
digest verification again once per unique remote dependency.

Import and export preserve canonical references exactly. Materializing neither
saves a recipe revision nor enables a tool. An enabled recipe remains in the
ordinary catalog as unavailable until its dependencies resolve, then becomes
available under the same identity. Admission and execution create transient
local bindings above `Recipe`; runtime `Artifact(Path)` remains local-only, and
accepted job snapshots/provenance retain the canonical document and hash.

### Canonical document interchange

Export downloads the existing canonical document only: no revision envelope,
timestamps or history. Recipe Studio exports saved user heads; unsaved edits
must be saved first, and built-ins must be duplicated before export.

Import preview returns the parsed document, layered validation and a status:
`new`, `identical`, `conflict`, `builtin`, or `invalid`. It does not write files.
A new UUID imports as revision 1 with that UUID intact. An identical canonical
document at the same UUID is already present and writes no revision; comparison
includes name and all document content, not just the semantic definition hash.
A differing same-UUID document or reserved built-in ID returns a conflict unless
the caller explicitly requests a copy. Copies receive a new UUID and revision 1,
preserving name, operation, fields, exact path strings and definition hash.
Imports never append to an existing recipe; the store rechecks identity on
commit and its existing atomic create guard rejects concurrent collisions.

Missing or foreign paths remain storable and unresolved. Invalid structure or
policy is previewable with diagnostics but cannot be imported. Browser file
selection stages multiple independent JSON documents, each with its own explicit
import action and result. The original JSON text reaches the Engine parser:
browser reserialization must not round uint64 values or change `1.0` into `1`,
which would change canonical bytes and definition hashes. Export similarly
downloads the Engine's exact canonical text.

### User recipe publication and execution

New, duplicated and imported recipes start disabled. Enable state and request
schema lineage live in host metadata, outside canonical JSON and definition
hashes. Enabling publishes the current head; saving an enabled recipe advances
its catalog entry automatically. Disabling removes it from new submissions.

User tool IDs are deterministic UUIDv5 values in a separate namespace, derived
from the recipe UUID, and stay stable through edits and renames. The eight
built-in tool IDs, schema revisions and hashes remain unchanged. Each enabled
entry carries `recipe: {id, revision, definition_hash}` in addition to ordinary
`schema_revision` and `schema_hash`. Missing dependencies leave an enabled tool
visible with `available:false` and an explanatory `unavailable_reason`.

Schema lineage starts at 1 on first enable and advances only when the public
request contract changes: exposed inputs, defaults and constraints, canvas,
timing, workflow or output. Hidden model paths, strengths and names can advance
the recipe revision without changing the schema. At startup, previously published
heads also reconcile schema lineage against the current Engine projection,
including disabled recipes, without changing immutable recipe revisions.
Fixed dimensions appear as
`canvas.fixed_width` / `fixed_height`; fixed duration adds `mode:"fixed"` and
`value` to `timing.duration_seconds`. LTX frame mappings contain only reachable
durations. Klein reference-derived null dimensions do not invent fixed sizes.
Ordered exposed values retain `collection:true`, `ordered:true`, array defaults
and constraints. Consumers must support these shapes or fail closed.

`POST /v1/jobs` for a user tool requires its exact recipe block and schema
revision/hash alongside `tool_id` and `inputs`. Stale metadata returns 409;
unavailable dependencies return 503. Admission resolves caller inputs through
the compiled family recipe and captures its immutable revision before queueing.
Later edits, disabling or deletion cannot change an accepted job. User job status retains
its accepted tool, schema and recipe provenance. Qwen also retains its builtin
Recipe identity and successful execution metadata as described above; existing
built-in job JSON is otherwise unchanged.
Uploaded media and generated artifacts use the ordinary service endpoints.

`DELETE /v1/authoring/recipes/{recipe_id}` permanently removes a user recipe's
entire local revision history and head/publication state. Recipe Studio requires
confirmation and does not offer deletion for built-ins; their reserved UUIDs are
also rejected by the API. Deleted tools disappear from the catalog and reject new
submissions. Existing client references become missing on refresh, without a
replacement tool. Model files, accepted jobs and stored generated-version
provenance are untouched. Re-importing the same UUID creates a fresh disabled
recipe at revision 1. There is no archive or trash.

There is no recipe pack/history archive, automatic local-file copying, or
directory watcher. Remote acquisition is explicit through the authoring APIs above.

## Boundary

Stable external IDs and input keys are product identities.

Internal recipe types, class names, storage structures, runtime objects,
diagnostic schemas, and historical Engine implementation details are not.

## Z-Image Turbo text-to-image

`zimage_turbo.text_to_image` is the built-in image tool; its authoring operation
is `zimage.t2i` and its recipe policy is `zimage.turbo.t2i.v1`. The default binds
the official INT8 ConvRot diffusion checkpoint, mixed-FP8 Qwen3 text encoder,
Flux AE decoder and tokenizer. Install its six pinned dependencies with
`python -m latentslate_engine.bootstrap install --home <engine-home> --family zimage`.
Built-in source references are returned with the authoring document; installation
verifies sizes and SHA-256 digests and reuses ordinary canonical model files.

Inputs are `prompt`, unsigned 64-bit `seed`, `width` and `height`. Canvas sides
are multiples of 16, at least 256, with at most 1,048,576 pixels and a 4:1 aspect
ratio. Default canvas is 1024 square. Width, height and seed may be fixed or
exposed by a recipe. Output is one PNG. The reference sampling policy uses eight
RES multistep steps, the simple flow schedule with shift 3, and CFG 1.

Authored recipes may replace the diffusion artifact with the official NVFP4
representation and bind up to two ordered ordinary transformer LoRAs with finite
strengths from -2 to 2. Supported LoRA tensors use paired `lora_A.weight` and
`lora_B.weight` factors, optional alpha, and native or Diffusers transformer
projection names. Unsupported tensor forms fail validation at load time.

One isolated GPU worker owns the current model identity. Seed-only requests
reuse model and conditioning state; a changed prompt invalidates conditioning.
Changing a model artifact or adapter composition releases prior model state
before loading the replacement. `DELETE /v1/runtime` exits the worker and releases
its native state. Engine does not import or run the Comfy graph executor.

## Ideogram v4 text-to-image

`ideogram4.text_to_image` exposes one PNG output through the ordinary image job
contract. Its authoring operation is `ideogram4.t2i` and policy is
`ideogram4.t2i.v1`. Inputs are `prompt`, unsigned 64-bit `seed`, `width` and
`height`; default dimensions are 1024 square, aligned to 16, with minimum side
256, maximum 1,056,768 pixels (including the reference's rounded 1 MP widescreen
preset) and maximum aspect ratio 4:1. Recipes can fix or
expose dimensions and seed. The conditional and negative diffusion checkpoints
are separate fixed bindings, alongside text encoder, tokenizer and VAE.

Custom recipes support ordinary INT8, INT8 ConvRot and mixed NVFP4/FP8
transformer files according to their quantization metadata. Up to two ordered
native transformer LoRAs can be fixed in the recipe, with strengths from -2 to 2;
the same composition applies to both transformers. Zero strength applies no
update. Changing a checkpoint, adapter or strength invalidates the loaded state;
seed-only requests retain it. Saved recipes without an adapter field retain an
empty composition.

For single-transformer checkpoints trained for unguided inference, set
`negative_diffusion` to null. This runs only the conditional transformer with
CFG 1, retaining the 20-step Euler schedule. It does not load a placeholder
negative model. The official built-in retains its required negative checkpoint
and dual-model guidance.

Install the seven pinned official dependencies with bootstrap
`--family ideogram4`. Recipe Studio exposes their source references and includes
both transformers in Manage downloads. Generation uses existing canonical
files; it does not silently download missing weights.

The baseline follows the executed official Comfy INT8 template: 20 Euler steps,
logit-normal schedule with mu 0 and std 1.75, CFG 7 changing to 3 at sigma <= 0.3.
Its negative transformer receives zeroed text features. This differs from the
text-free negative pass described in the template's note; Engine preserves the
actual connected graph behavior for reproducible comparison.

`prompt` remains a string and is encoded without JSON rewriting or automatic
prompt enhancement. Callers can provide serialized structured captions including
spatial boxes. The official caption format uses `compositional_deconstruction`
with `background` and `elements`; optional boxes are integer coordinates
`[y_min, x_min, y_max, x_max]` on a 0–1000 grid. Keep canvas dimensions outside
the caption. See the [official prompting guide](https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/docs/prompting.md)
for style and element fields. Plain text is passed through like native Comfy;
it is not automatically converted into the structured format used in training.
Dedicated spatial prompting UI is separate from this Engine contract.

One isolated worker retains both transformers and the last prompt's
conditioning. Seed changes reuse them; prompt changes re-encode conditioning.
Artifact identity changes release previous state. No adapter capability is
advertised by this baseline.


## SDXL text-to-image

`sdxl.text_to_image` uses the `sdxl.t2i` authoring operation and `sdxl.t2i.v1`
policy. Its ordinary epsilon checkpoint supplies the base UNet, CLIP-L/G and
VAE. Set the optional `vae` binding to override the embedded decoder; a null
binding uses the checkpoint VAE. Refiner, LoRA, ControlNet, textual inversion,
EDM and v-prediction variants are outside this operation.

Inputs are `prompt`, `negative_prompt` (default empty), `width`, `height`,
unsigned 64-bit `seed`, `steps` (1–100), `cfg` (1–20), `sampler`
(`euler`, `euler_ancestral`, `dpmpp_2m`) and `scheduler` (`normal`, `karras`).
The official no-refiner template supplies defaults: 1024 square, 25 steps,
CFG 7 and DPM++ 2M Karras. Canvas sides align to 8 pixels, minimum 256,
maximum 1,048,576 pixels and 4:1 aspect ratio. Recipe authors can fix or expose
negative prompt and generation controls; positive prompt remains caller input.

Bootstrap `--family sdxl` installs the pinned official SDXL Base checkpoint and
three CLIP tokenizer files from Hugging Face. Recipe Studio displays these
sources and includes them in Manage downloads. Community checkpoint selection
belongs to the user and does not alter built-in defaults.

The isolated worker retains the UNet, decoder and last positive/negative prompt
conditioning. Seed and sampler changes reuse weights; either prompt changing
invalidates conditioning. A checkpoint, tokenizer or VAE identity change
releases the prior state. `DELETE /v1/runtime` exits the worker.

## LTX 2.5 video

`ltx25.t2v`, `ltx25.i2v` and `ltx25.flf` provide text-to-video,
image-to-video and ordered first/last-frame generation with audio. Their recipe
policies use the corresponding `.v1` suffix. All accept `prompt`, `width`,
`height`, `duration_seconds`, integer `fps`, unsigned 64-bit `seed` and
`prompt_enhancement`. Image operations require `start_image`; FLF also requires
`end_image`.

Defaults are 512 square, five requested seconds, 24 FPS, seed zero and enhancement
off. Duration is 1–10 seconds and FPS is 1–120. Frame count is
`floor(duration_seconds * fps / 8) * 8 + 1`; report the resulting duration from
that frame count and FPS. Canvas sides align to 64 pixels for T2V/I2V and 32 for
FLF. The curated T2V/I2V path samples at half resolution, spatially upscales,
then refines; FLF samples directly with ordered guide frames.

Recipes fix diffusion, text encoder, video VAE and audio VAE files. T2V/I2V also
require the spatial upsampler. Ordered transformer adapter files are recipe-owned;
their matching strengths can be fixed or exposed. Bootstrap `--family ltx25`
installs canonical pinned Hugging Face assets for the built-ins.

The separate prompt enhancer is required whenever enhancement is exposed or
fixed on, including an exposed toggle whose default is off. A fixed-off recipe
may omit it. Files are acquired through normal authoring downloads, never by a
generation request. I2V/FLF enhancement consumes the first image. Progress reports
model loading and generated enhancement tokens before video generation.

The isolated worker retains transformer/text weights and the most recent prompt
and image conditioning. Seed-only changes reuse them. Prompt or image-content
changes invalidate the corresponding conditioning; FLF image order is significant.
Model and adapter identity changes replace the worker. `DELETE /v1/runtime`
releases it.

## MiniMax H3 video (in progress)

`h3.t2v`, `h3.i2v` and `h3.r2v` expose native video with synchronized stereo
audio. Policies use the corresponding `.v1` suffix. T2V and I2V share the FL2VA
backbone; reference generation uses Ref2VA. Bootstrap `--family h3` installs
eight pinned Hugging Face dependencies into ordinary canonical files, including
both backbones, the text encoder, both decoders and tokenizer companions.
The built-in authoring documents expose those sources through normal downloads.

Callers submit `prompt`, unsigned 64-bit `seed`, explicit integer `width` and
`height`, and `duration_seconds`. Canvas sides must be multiples of 32 and at
least 32 pixels. Engine rejects invalid dimensions before media decoding or GPU
work; it never adapts the requested output canvas. FPS is fixed at 24. Duration
targets span 5/24 to 362/24 seconds and resolve upward to the model's `17n+5`
frame grid after rounding the target frame count. The catalog declares that grid
so frontends can present realizable durations. Defaults are 864×480 and 124
frames. I2V's required `start_image` must already match the output canvas.

Reference inputs are optional numbered slots: `reference_image_1` through `9`,
`reference_video_1` through `3`, and `reference_audio_1` through `3`. Each video
can have an explicit index-paired `reference_video_audio_N` input; passing the
same uploaded video asset there includes its soundtrack. Uploads use the
existing asset endpoint. The catalog uses the supported `custom` workflow kind
for this multimodal operation, with typed image/video/audio inputs. Reference
image preparation is separate from output sizing and follows the recipe's
fixed `reference_image_size` choice (`match` or `max`).

The worker retains one model identity and the most recent conditioning. Seed
changes reuse both; prompt, content, ordered reference roles and relevant
geometry changes invalidate conditioning. Switching backbones replaces the
worker. `DELETE /v1/runtime` releases its native state. Baseline generation uses
20 RES multistep steps; optional turbo/adapters, broader compatibility and final
performance certification are still pending.
