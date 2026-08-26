"""Temporary direct execution hooks for the pinned transformer call order."""

from __future__ import annotations


def make_prefetch_queue(blocks, _device, _options):
    return iter(blocks)


def prefetch_queue_pop(queue, _device, _block) -> None:
    if queue is not None:
        next(queue, None)
