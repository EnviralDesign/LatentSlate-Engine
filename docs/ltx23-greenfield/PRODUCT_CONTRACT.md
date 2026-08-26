# LTX 2.3 product contract

LTX 2.3 must eventually support text-to-video (T2V), image-to-video (I2V), and
first/last-frame video (FLF). No implementation exists in this reset.

| Capability | Stable UUID | Product key | Operation | Required inputs |
| --- | --- | --- | --- | --- |
| T2V | `46bdb57c-3b19-5397-8949-4e20ffe757c9` | `ltx23.text_to_video` | `ltx23_dev_t2v` | `prompt`, `width`, `height`, `duration_seconds`, `seed` |
| I2V | `5d6e2d6f-216c-5f35-a4ec-1565d6e56ee7` | `ltx23.image_to_video` | `ltx23_dev_i2v` | `prompt`, `start_image`, `width`, `height`, `duration_seconds`, `seed` |
| FLF | `1a8f9c0b-410e-56e4-90de-23bcb9d644ca` | `ltx23.first_last_frame_to_video` | `ltx23_distilled_flf` | `prompt`, `start_image`, `end_image`, `width`, `height`, `duration_seconds`, `seed` |

The eventual catalog must preserve these six recipe identifiers:

- `ltx-2-3.text-to-video.kitchen-dev-fp8`
- `ltx-2-3.image-to-video.kitchen-dev-fp8`
- `ltx-2-3.first-last-frame-to-video.kitchen-distilled-fp8`
- `ltx-2-3.text-to-video.native-distilled-bf16`
- `ltx-2-3.image-to-video.native-distilled-bf16`
- `ltx-2-3.first-last-frame-to-video.native-distilled-bf16`

The eventual runtime must preserve warm useful state when a model and recipe are
unchanged, strictly purge it when either changes, isolate GPU-worker
cancellation, and produce complete MP4 output.

The fixed parity target is a five-second request at 25 fps: 121 effective
frames, 4.84 seconds of media, H.264 video, and AAC 48 kHz stereo audio.
