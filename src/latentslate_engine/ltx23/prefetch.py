"""LTX transformer block transfers ordered like pinned Comfy's VBAR path."""

from __future__ import annotations

import torch


def make_prefetch_queue(blocks, device, _options):
    if torch.device(device).type != "cuda":
        return None
    return {
        "entries": [None, *blocks, None],
        "streams": (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)),
        "host_buffers": blocks[0]._latentslate_host_buffers,
        "stream_index": 0,
    }


def prefetch_queue_pop(queue, device, _block) -> None:
    if queue is None:
        return

    consumed = queue["entries"].pop(0)
    current = torch.cuda.current_stream(device)
    if consumed is not None:
        stream, block = consumed
        stream.wait_stream(current)
        block._latentslate_release()

    next_block = queue["entries"][0]
    if next_block is None:
        return

    stream = queue["streams"][queue["stream_index"]]
    host_buffer = queue["host_buffers"][queue["stream_index"]]
    queue["stream_index"] = (queue["stream_index"] + 1) % len(queue["streams"])
    stream.wait_stream(current)
    next_block._latentslate_prepare(stream, host_buffer)
    current.wait_stream(stream)
    queue["entries"][0] = (stream, next_block)
