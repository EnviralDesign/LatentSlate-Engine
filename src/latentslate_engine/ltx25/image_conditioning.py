"""Image preprocessing from the curated LTX 2.5 I2V workflow."""

from io import BytesIO
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image


def preprocess_source_image(path: str | Path) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    if width > height:
        resized_width = 1536
        resized_height = round(height * (1536 / width))
    else:
        resized_height = 1536
        resized_width = round(width * (1536 / height))
    image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    pixels = np.asarray(image).astype(np.float32) / 255.0
    pixels = pixels[: (pixels.shape[0] // 2) * 2, : (pixels.shape[1] // 2) * 2]
    return _compress(pixels)


def _compress(pixels: np.ndarray) -> torch.Tensor:
    image_tensor = torch.from_numpy(pixels)
    image_array = (image_tensor * 255.0).byte().numpy()

    with BytesIO() as output_file:
        container = av.open(output_file, "w", format="mp4")
        try:
            stream = container.add_stream(
                "libx264", rate=1, options={"crf": "18", "preset": "veryfast"}
            )
            stream.height = image_array.shape[0]
            stream.width = image_array.shape[1]
            frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(
                format="yuv420p"
            )
            container.mux(stream.encode(frame))
            container.mux(stream.encode())
        finally:
            container.close()
        video_bytes = output_file.getvalue()

    with BytesIO(video_bytes) as video_file:
        container = av.open(video_file)
        try:
            stream = next(item for item in container.streams if item.type == "video")
            decoded = next(container.decode(stream)).to_ndarray(format="rgb24")
        finally:
            container.close()
    return (
        torch.from_numpy(decoded.astype(np.float32) / 255.0).movedim(-1, 0).unsqueeze(0)
    )


def preprocess_guide_image(path: str | Path, width: int, height: int) -> torch.Tensor:
    """Fit a FLF guide before the template's CRF18 compression."""
    with Image.open(path) as source:
        pixels = np.asarray(source.convert("RGB")).astype(np.float32) / 255.0
    image = torch.from_numpy(pixels).movedim(-1, 0).unsqueeze(0)
    old_height, old_width = image.shape[-2:]
    old_aspect, new_aspect = old_width / old_height, width / height
    x = (
        round((old_width - old_width * new_aspect / old_aspect) / 2)
        if old_aspect > new_aspect
        else 0
    )
    y = (
        round((old_height - old_height * old_aspect / new_aspect) / 2)
        if old_aspect < new_aspect
        else 0
    )
    image = image[:, :, y : old_height - y, x : old_width - x]
    image = torch.nn.functional.interpolate(
        image, size=(height, width), mode="nearest-exact"
    )
    return _compress(image.squeeze(0).movedim(0, -1).numpy())
