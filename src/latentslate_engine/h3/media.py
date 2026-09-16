"""H3 reference-media preparation; it never selects the output canvas.

Media decode/resize follows ComfyUI 1a14b82e nodes_minimax_h3.py,
nodes_audio.py and VideoFromFile (GPL-3.0). The default Hann resampler is
adapted from TorchAudio 2.11 functional.py (BSD-2-Clause; see notices).
"""

import math

import av
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def resample_audio(waveform, sample_rate):
    """Match torchaudio.functional.resample's default kernel at 32 kHz."""
    if sample_rate <= 0:
        raise ValueError("H3 reference audio requires a positive sample rate")
    if sample_rate == 32000:
        return waveform
    divisor = math.gcd(sample_rate, 32000)
    source, target = sample_rate // divisor, 32000 // divisor
    base = min(source, target) * 0.99
    width = math.ceil(6 * source / base)
    index = (
        torch.arange(
            -width, width + source, device=waveform.device, dtype=waveform.dtype
        )[None, None]
        / source
    )
    time = (
        torch.arange(0, -target, -1, device=waveform.device, dtype=waveform.dtype)[
            :, None, None
        ]
        / target
        + index
    )
    time *= base
    time.clamp_(-6, 6)
    window = torch.cos(time * math.pi / 6 / 2) ** 2
    time *= math.pi
    kernel = torch.where(time == 0, torch.tensor(1.0).to(time), time.sin() / time)
    kernel *= window * (base / source)
    shape = waveform.shape
    flat = waveform.view(-1, shape[-1])
    padded = F.pad(flat, (width, width + source))
    result = (
        F.conv1d(padded[:, None], kernel, stride=source)
        .transpose(1, 2)
        .reshape(flat.shape[0], -1)
    )
    length = torch.ceil(torch.as_tensor(target * shape[-1] / source)).long()
    return result[..., :length].view(*shape[:-1], -1)


def load_audio(path, *, soundtrack=False):
    """Read an explicit audio input or a video's last decodable soundtrack."""
    with av.open(str(path)) as container:
        if soundtrack:
            stream = next(
                (
                    s
                    for s in reversed(container.streams.audio)
                    if s.codec_context is not None
                ),
                None,
            )
        else:
            stream = container.streams.audio[0] if container.streams.audio else None
        if stream is None:
            if soundtrack:
                return None
            raise ValueError("H3 reference audio has no decodable audio stream")
        sample_rate, channels = stream.codec_context.sample_rate, stream.channels
        frames = []
        resampler = av.AudioResampler(format="fltp") if soundtrack else None
        for frame in container.decode(stream):
            decoded = resampler.resample(frame) if resampler else [frame]
            for part in decoded:
                value = torch.from_numpy(part.to_ndarray())
                if value.shape[0] != channels:
                    value = value.view(-1, channels).t()
                if soundtrack and not frames:
                    skip = max(0, int(-part.pts * stream.time_base * sample_rate))
                    value = value[..., skip:]
                    if not value.shape[-1]:
                        continue
                frames.append(value)
        if not frames:
            raise ValueError("H3 reference audio contains no decoded samples")
        waveform = torch.cat(frames, dim=1)
        if waveform.dtype == torch.int16:
            waveform = waveform.float() / 2**15
        elif waveform.dtype == torch.int32:
            waveform = waveform.float() / 2**31
        elif not waveform.is_floating_point():
            raise ValueError("Unsupported H3 reference audio PCM representation")
        waveform = resample_audio(waveform.unsqueeze(0), sample_rate)
        if channels == 1:
            waveform = waveform.repeat(1, 2, 1)
        return waveform[:, :2]


def reference_video_size(width, height):
    """The oracle's reference-only 768-short-edge/area-bounded preparation."""
    ratio = width / height
    nominal_w, nominal_h = (768 * ratio, 768) if ratio >= 1 else (768, 768 / ratio)
    if nominal_w * nominal_h > 768 * 1344:
        scale = math.sqrt(768 * 1344 / (nominal_w * nominal_h))
        nominal_w, nominal_h = nominal_w * scale, nominal_h * scale
    target_w, target_h = (max(32, round(v / 32) * 32) for v in (nominal_w, nominal_h))
    if width * height < target_w * target_h:
        target_w, target_h = (max(32, round(v / 32) * 32) for v in (width, height))
    return target_w, target_h


def load_video(path, frame_limit):
    """Decode and normalize an ordered reference clip, without changing output size."""
    images = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("H3 reference video has no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        align_graph = None
        for frame in container.decode(stream):
            if frame.format.name in (
                "yuvj420p",
                "yuvj422p",
                "yuvj444p",
                "rgb24",
                "rgba",
                "pal8",
            ):
                pixels = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
            else:
                if frame.width % 32:
                    if align_graph is None:
                        pad_w, pad_h = (
                            (v + 31) // 32 * 32 for v in (frame.width, frame.height)
                        )
                        graph = av.filter.Graph()
                        source = graph.add_buffer(
                            width=frame.width,
                            height=frame.height,
                            format=frame.format.name,
                            time_base=stream.time_base,
                        )
                        pad = graph.add("pad", f"{pad_w}:{pad_h}:0:0")
                        fill = graph.add(
                            "fillborders",
                            f"left=0:right={pad_w - frame.width}:top=0:bottom={pad_h - frame.height}:mode=smear",
                        )
                        sink = graph.add("buffersink")
                        source.link_to(pad)
                        pad.link_to(fill)
                        fill.link_to(sink)
                        graph.configure()
                        align_graph = (graph, source, sink)
                    align_graph[1].push(frame)
                    pixels = (
                        align_graph[2]
                        .pull()
                        .to_ndarray(format="gbrpf32le")[: frame.height, : frame.width]
                    )
                else:
                    pixels = frame.to_ndarray(format="gbrpf32le")
            rotation = round(frame.rotation // 90) % 4 if frame.rotation else 0
            if rotation:
                pixels = np.rot90(pixels, k=rotation, axes=(0, 1)).copy()
            target_size = reference_video_size(pixels.shape[1], pixels.shape[0])
            image = Image.fromarray(
                np.clip(255.0 * pixels, 0, 255).astype(np.uint8)
            ).resize(target_size, Image.Resampling.LANCZOS)
            images.append(torch.from_numpy(np.array(image).astype(np.float32) / 255.0))
            if len(images) == frame_limit:
                break
        if len(images) < 5:
            raise ValueError("H3 reference videos need at least five frames")
        count = (len(images) - 5) // 17 * 17 + 5
        return torch.stack(images[:count])
