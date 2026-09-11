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

## Recipe authoring V0

The separate `/v1/authoring` API uses the same bearer boundary. It does not add
tools to `/v1/catalog`, enable user-recipe jobs, or probe native execution.

| Method | Path under `/v1/authoring` | Result |
| --- | --- | --- |
| GET | `/operations` | Eight family operation descriptors, inherent domains and field ownership |
| GET | `/builtins`, `/builtins/{key}` | Immutable certified definitions |
| POST | `/builtins/{key}/duplicate` | New UUID, revision 1; body `{}` or `{"name":"My recipe"}` |
| POST | `/validate` | Layered validation of a canonical document, without saving |
| GET | `/recipes`, `/recipes/{uuid}` | User recipes at their current heads |
| POST | `/recipes` | Create a canonical document with a new client-supplied UUID |
| PUT | `/recipes/{uuid}` | Save `{"base_revision":1,"document":{...}}`; stale heads return 409 |
| GET | `/recipes/{uuid}/revisions`, `/recipes/{uuid}/revisions/{number}` | Immutable revision history |
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
Klein retains artifact-only LoRAs. Artifact selection is fixed recipe content;
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

There is no recipe deletion, enable/publication, recipe pack or history archive,
artifact acquisition/copying, directory watcher, or user-recipe execution in V0.

## Boundary

Stable external IDs and input keys are product identities.

Internal recipe types, class names, storage structures, runtime objects,
diagnostic schemas, and historical Engine implementation details are not.
