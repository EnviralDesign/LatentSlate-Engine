from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from latentslate_engine.runtime.ltx23_video_vae_aimdo import (
    LTX23VideoVAEAimdoState,
)


class _ToyVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Larger than Comfy's pinned 16 KiB force-load boundary.
        self.projection = nn.Linear(128, 128)

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


def test_video_vae_owner_scopes_direct_leaves_and_restores_methods() -> None:
    vae = _ToyVAE().eval()
    owner = LTX23VideoVAEAimdoState(vae, "cpu")
    before = owner.diagnostics()
    value = torch.randn((1, 128))

    with pytest.raises(RuntimeError, match="outside encode/decode scope"):
        vae.projection(value)

    first = vae.encode(value)
    second = vae.encode(value)
    decoded = vae.decode(value)
    assert first.shape == second.shape == decoded.shape == (1, 128)
    assert all(parameter.device.type == "cpu" for parameter in vae.parameters())

    proof = owner.diagnostics()
    assert proof["operation_calls"] == {"encode": 2, "decode": 1}
    assert proof["leaf_bind_calls"] == 3
    assert proof["whole_module_move_calls"] == 0
    assert proof["active_scope"] is None
    assert proof["operation_seconds"]["encode"] >= 0.0
    delta = owner.diagnostics_delta(before)
    assert delta["operation_calls"] == {"encode": 2, "decode": 1}
    assert delta["whole_module_move_calls"] == 0
    assert delta["full_module_moves"] is False

    owner.close()
    assert "encode" not in vars(vae)
    assert "decode" not in vars(vae)
    assert vae.encode(value).shape == (1, 128)


def test_kitchen_video_vae_path_has_no_blanket_device_move() -> None:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "latentslate_engine"
        / "runtime"
        / "ltx23_kitchen.py"
    ).read_text(encoding="utf-8")
    assert '_move_module(c["video_vae"]' not in source
    assert "_move_module(vae" not in source
    assert '"video_vae_residency": {' in source
    assert "video_vae_residency.diagnostics_delta(video_vae_before)" in source
