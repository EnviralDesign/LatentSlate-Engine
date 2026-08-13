from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from latentslate_engine.lora import ConfiguredLora
from latentslate_engine.runtime.wan22_stored_lora import (
    WanStoredLoraLinear,
    apply_wan_stage_loras,
    plan_wan_stored_lora,
)
from latentslate_engine.tools.base import LoraExecution
from latentslate_engine.wan22_recipe import _build_stage_loras


class _Target(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_k = nn.Linear(3, 4, bias=False)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn2 = _Target()


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block()])


def _lora(path: Path, *, strength: float = 1.0) -> LoraExecution:
    save_file(
        {
            "diffusion_model.blocks.0.cross_attn.k.lora_down.weight": torch.ones((2, 3)),
            "diffusion_model.blocks.0.cross_attn.k.lora_up.weight": torch.ones((4, 2)),
            # The pinned official LightX files encode all alpha scalars as I64.
            "diffusion_model.blocks.0.cross_attn.k.alpha": torch.tensor(2, dtype=torch.int64),
        },
        str(path),
    )
    return LoraExecution("high_noise", "lora:wan22:test", path, strength)


def test_wan_lora_maps_comfy_stage_target_and_adds_without_replacing_base(tmp_path: Path) -> None:
    transformer = _Transformer()
    base = transformer.blocks[0].attn2.to_k
    item = _lora(tmp_path / "adapter.safetensors")

    result = apply_wan_stage_loras(transformer, (item,))

    wrapped = transformer.blocks[0].attn2.to_k
    assert isinstance(wrapped, WanStoredLoraLinear)
    assert wrapped.base is base
    assert result["active"] == [item.resource_id]
    assert result["weights"] == [1.0]
    output = wrapped(torch.ones((1, 3)))
    assert output.shape == (1, 4)
    assert wrapped.lora_dispatch_count == 1


def test_wan_zero_strength_is_not_opened_or_installed(tmp_path: Path) -> None:
    transformer = _Transformer()
    missing = LoraExecution("high_noise", "lora:wan22:disabled", tmp_path / "missing.safetensors", 0.0)

    result = apply_wan_stage_loras(transformer, (missing,))

    assert result["active"] == []
    assert type(transformer.blocks[0].attn2.to_k) is nn.Linear


def test_wan_lora_rejects_an_unmapped_tensor(tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    save_file(
        {
            "diffusion_model.blocks.0.unknown.lora_down.weight": torch.ones((2, 3)),
            "diffusion_model.blocks.0.unknown.lora_up.weight": torch.ones((4, 2)),
        },
        str(path),
    )

    with pytest.raises(ValueError, match="unsupported"):
        plan_wan_stored_lora(_Transformer(), path)


def test_stage_stack_keeps_disabled_slot_configured_but_unresolved(tmp_path: Path) -> None:
    item = _lora(tmp_path / "active.safetensors")
    active, configured = _build_stage_loras(
        (item,),
        (
            ConfiguredLora("high_noise", item.resource_id, 1.0, True),
            ConfiguredLora("low_noise", "lora:wan22:not-installed", 0.0, False),
        ),
        {"high_noise": "high", "low_noise": "low"},
    )

    assert [entry.stage for entry in active] == ["high"]
    assert [entry["slot"] for entry in configured] == ["high_noise", "low_noise"]
    assert configured[1]["active"] is False
    assert configured[1]["resource_reference"] == "lora:wan22:not-installed"


def test_stage_stack_rejects_active_strength_that_differs_from_configured_slot(
    tmp_path: Path,
) -> None:
    item = _lora(tmp_path / "active.safetensors", strength=0.5)

    with pytest.raises(ValueError, match="does not match"):
        _build_stage_loras(
            (item,),
            (ConfiguredLora("high_noise", item.resource_id, 1.0, True),),
            {"high_noise": "high"},
        )
