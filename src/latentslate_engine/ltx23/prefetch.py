"""LTX transformer block transfers ordered like pinned Comfy's VBAR path."""

from __future__ import annotations

import torch


def make_prefetch_queue(blocks, device, _options):
    if torch.device(device).type != "cuda":
        return None
    if _options.get("latentslate_pipeline_prefetch", False):
        return {
            "pipeline": True,
            "blocks": blocks,
            "streams": (
                torch.cuda.Stream(device=device),
                torch.cuda.Stream(device=device),
            ),
            "buffers": blocks[0]._latentslate_host_buffers,
            "index": -1,
        }
    return {
        "entries": [None, *blocks, None],
        "streams": (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)),
        "host_buffers": blocks[0]._latentslate_host_buffers,
        "stream_index": 0,
    }


def prefetch_queue_pop(queue, device, _block) -> None:
    if queue is None:
        return
    if queue.get("pipeline", False):
        current = torch.cuda.current_stream(device)
        next_index = queue["index"] + 1
        if next_index == 0:
            stream = queue["streams"][0]
            stream.wait_stream(current)
            queue["blocks"][0]._latentslate_prepare(stream, queue["buffers"][0])
            current.wait_stream(stream)
            if len(queue["blocks"]) > 1:
                stream = queue["streams"][1]
                queue["blocks"][1]._latentslate_prepare(stream, queue["buffers"][1])
            queue["index"] = 0
            return

        previous = queue["blocks"][queue["index"]]
        if next_index < len(queue["blocks"]):
            ready_stream = queue["streams"][next_index % 2]
            current.wait_stream(ready_stream)
        previous._latentslate_release()
        following_index = next_index + 1
        if following_index < len(queue["blocks"]):
            stream = queue["streams"][following_index % 2]
            stream.wait_stream(current)
            queue["blocks"][following_index]._latentslate_prepare(
                stream, queue["buffers"][following_index % 2]
            )
        queue["index"] = next_index
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
