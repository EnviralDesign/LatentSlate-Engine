from __future__ import annotations

from pathlib import Path

from latentslate_engine.config import Settings
from latentslate_engine.tools import default_registry


def _settings(home: Path) -> Settings:
    settings = Settings(
        home=home,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    return settings


def test_t2v_support_closure_is_exact_and_excludes_checkpoint_weights(tmp_path: Path) -> None:
    registry = default_registry(_settings(tmp_path), emit_warnings=False)
    resources = {resource.id: resource for resource in registry.resources.resources}
    support = resources["model:wan22:wan22-14b-t2v-official-support"]
    source = support.sources[0]
    files = support.metadata["upstream_snapshot"]["files"]
    assert support.size_bytes == 529_069_135
    assert source.revision == "5be7df9619b54f4e2667b2755bc6a756675b5cd7"
    assert source.is_exact()
    assert len(files) == len(source.allow_patterns) == 12
    assert sum(item["size_bytes"] for item in files) == support.size_bytes
    assert [item["path"] for item in files] == list(source.allow_patterns)
    assert not any(
        path.startswith("transformer/") and not path.endswith("config.json")
        or path.startswith("transformer_2/") and not path.endswith("config.json")
        or path.startswith("text_encoder/") and not path.endswith("config.json")
        for path in source.allow_patterns
    )

    high = resources[
        "model:wan22:comfy-org-wan22-14b-t2v-fp8/"
        "split_files/diffusion_models/wan2.2_t2v_high_noise_14b_fp8_scaled"
    ]
    low = resources[
        "model:wan22:comfy-org-wan22-14b-t2v-fp8/"
        "split_files/diffusion_models/wan2.2_t2v_low_noise_14b_fp8_scaled"
    ]
    assert {high.sources[0].sha256, low.sources[0].sha256} == {
        "cad711ae211c8b23455ec68cd6a190a33a3d874234a77eb57266d73f8f0e6c9f",
        "e71b96d7c82e638694c5e7fb98fac4bfb0e4ddc5fbbb4b1df40da8f0f1278a97",
    }
    assert high.size_bytes == low.size_bytes == 14_293_923_632
    entry = next(
        item
        for item in registry.variants
        if item.key == "wan-2-2-14b-t2v.text-to-video.comfy-org-fp8"
    )
    assert entry.base_tool == "wan22.native_text_to_video"
    assert entry.recipe_type == "wan22_t2v_14b"
    assert entry.tags == [
        "builtin",
        "experimental",
        "wan2.2",
        "t2v",
        "14b",
        "comfy-org",
        "fp8",
        "native-stored-weights",
    ]
