"""Block residency hooks for the pinned LTX transformer call order."""

from __future__ import annotations


def make_prefetch_queue(_blocks, _device, _options):
    return {"active": None}


def prefetch_queue_pop(queue, _device, _block) -> None:
    if queue is None:
        return
    previous = queue["active"]
    if previous is _block:
        return
    if previous is not None:
        previous._latentslate_release()
    if _block is not None:
        _block._latentslate_prepare()
    queue["active"] = _block
