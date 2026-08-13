from __future__ import annotations

import os
from pathlib import Path

import pytest

from latentslate_engine.runtime.wan5_kitchen_contracts import (
    WAN5_TRANSFORMER,
    WAN5_TRANSFORMER_CONFIG,
    WAN5_VAE,
    WAN5_VAE_CONFIG,
    plan_wan5_stored_artifact,
)


def test_exact_artifact_and_shell_contracts_are_pinned() -> None:
    assert WAN5_TRANSFORMER.filename == "wan2.2_ti2v_5B_fp16.safetensors"
    assert WAN5_TRANSFORMER.size_bytes == 9_999_658_848
    assert WAN5_TRANSFORMER.tensor_count == 825
    assert WAN5_TRANSFORMER.source_sha256 == (
        "456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e"
    )
    assert WAN5_VAE.filename == "wan2.2_vae.safetensors"
    assert WAN5_VAE.size_bytes == 1_409_400_960
    assert WAN5_VAE.tensor_count == 196
    assert WAN5_TRANSFORMER_CONFIG["num_layers"] == 30
    assert WAN5_TRANSFORMER_CONFIG["in_channels"] == 48
    assert WAN5_VAE_CONFIG["z_dim"] == 48


@pytest.mark.skipif(
    os.environ.get("LATENTSLATE_WAN5_REAL_HEADERS") != "1",
    reason="set LATENTSLATE_WAN5_REAL_HEADERS=1 for installed Wan 5B proof",
)
@pytest.mark.parametrize(
    ("relative", "contract", "count"),
    [
        ("wan2.2_ti2v_5B_fp16.safetensors", WAN5_TRANSFORMER, 825),
        ("wan2.2_vae.safetensors", WAN5_VAE, 196),
    ],
)
def test_installed_headers_are_exact_shell_closures(relative, contract, count) -> None:
    root = Path(os.environ["LATENTSLATE_WAN5_HOME"]) / "models" / "wan22" / "ti2v-5b"
    plan = plan_wan5_stored_artifact(root / relative, contract)

    assert plan.available, plan.errors
    assert len(plan.source_to_target) == count
    assert len(set(plan.source_to_target.values())) == count
    assert plan.identity.size_bytes == contract.size_bytes
    assert plan.identity.header_sha256 == contract.header_sha256
