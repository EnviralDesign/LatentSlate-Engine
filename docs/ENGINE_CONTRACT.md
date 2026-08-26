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

The primary artifact is preferred when `role == "primary"`.

### `DELETE /v1/jobs/{id}`

Requests cancellation. Cancellation acknowledgement does not imply that native
GPU work is already quiescent; Engine owns safe termination semantics.

### `GET <artifact.download_url>`

Downloads generated media bytes.

### `DELETE /v1/runtime`

Explicitly releases active Engine runtime/model resources.

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

Inputs:

- `prompt`
- `start_image`
- `end_image`
- `width`
- `height`
- `duration_seconds`
- `seed`

Both frame inputs are required for the FLF public operation.

## Boundary

Stable external IDs and input keys are product identities.

Internal recipe types, class names, storage structures, runtime objects,
diagnostic schemas, and historical Engine implementation details are not.
