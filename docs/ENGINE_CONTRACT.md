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

Jobs may expose overall `progress` in the range 0..1 and a low-volume phase
message.

Successful jobs expose artifacts. An artifact requires:

- `role`
- `filename`
- `download_url`

The primary artifact is preferred when `role == "primary"`. Current LTX tools
publish one MP4 primary artifact and current Klein tools publish one PNG primary
artifact.

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
- schema revision: `2`
- workflow kind: `image_to_video`
- output: video
- schema hash: `sha256:8364fcc55ec44ae780d49d9c9404768c81a5680783106934f9a17bd990be7efa`

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
- schema revision: `2`
- workflow kind: `first_frame_last_frame_video`
- output: video
- schema hash: `sha256:aa624d8d8fe060dcc39c15623e4b4b07eb405305051ebdd5fd2caf8368d8acd9`

Inputs:

- `prompt`
- `start_image`
- `end_image`
- `width`
- `height`
- `duration_seconds`
- `seed`

Both frame inputs are required for the FLF public operation.

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

## Boundary

Stable external IDs and input keys are product identities.

Internal recipe types, class names, storage structures, runtime objects,
diagnostic schemas, and historical Engine implementation details are not.
