from __future__ import annotations

from pathlib import Path

import torch

from latentslate_engine.klein9b.model import KleinTransformer
from latentslate_engine.klein9b.runtime import (
    KLEIN_PROMPT_TEMPLATE,
    Klein9BIdentity,
    Klein9BRuntime,
    _sigmas,
    _unpack_latent,
)


def _identity(root: Path, suffix: str = "") -> Klein9BIdentity:
    paths = []
    for name in ("diffusion", "text", "vae"):
        path = root / f"{name}{suffix}.safetensors"
        path.write_bytes(name.encode())
        paths.append(path)
    tokenizer = root / f"tokenizer{suffix}"
    tokenizer.mkdir()
    for name in (
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        (tokenizer / name).write_text(name)
    config = root / "text_encoder" / "config.json"
    config.parent.mkdir(exist_ok=True)
    config.write_text("config")
    return Klein9BIdentity.from_paths(*paths, tokenizer)


def test_exact_identity_reuses_state_and_change_releases_it(tmp_path: Path) -> None:
    first = _identity(tmp_path)
    second = _identity(tmp_path, "-changed")
    runtime = Klein9BRuntime(device="cpu")

    assert runtime.ensure_identity(first) is False
    transformer = object()
    vae = object()
    conditioning = ("prompt", torch.zeros(1))
    runtime.transformer = transformer  # type: ignore[assignment]
    runtime.vae = vae  # type: ignore[assignment]
    runtime.conditioning = conditioning

    assert runtime.ensure_identity(first) is True
    assert runtime.transformer is transformer
    assert runtime.vae is vae
    assert runtime.conditioning is conditioning

    assert runtime.ensure_identity(second) is False
    assert runtime.identity == second
    assert runtime.transformer is None
    assert runtime.vae is None
    assert runtime.conditioning is None


def test_consumed_tokenizer_and_text_config_are_identity_inputs(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path)
    arguments = (
        identity.diffusion.path,
        identity.text_encoder.path,
        identity.vae.path,
        identity.tokenizer,
    )

    (identity.tokenizer / "tokenizer_config.json").write_text("changed tokenizer")
    tokenizer_changed = Klein9BIdentity.from_paths(*arguments)
    assert tokenizer_changed != identity

    config_path = identity.tokenizer.parent / "text_encoder" / "config.json"
    config_path.write_text("changed text encoder config")
    assert Klein9BIdentity.from_paths(*arguments) != tokenizer_changed


def test_packed_latent_becomes_flux2_vae_shape() -> None:
    packed = torch.arange(128 * 2 * 3).reshape(1, 128, 2, 3)
    unpacked = _unpack_latent(packed)
    assert unpacked.shape == (1, 32, 4, 6)
    assert torch.equal(unpacked[:, :, 0::2, 0::2], packed[:, 0::4])


def test_canonical_schedule_matches_pinned_flux2_scheduler() -> None:
    schedule = _sigmas(4, torch.device("cpu"))
    expected = torch.tensor(
        [1.0, 0.9622337222099304, 0.8946577906608582, 0.7389686703681946, 0.0]
    )
    torch.testing.assert_close(schedule, expected, rtol=0, atol=1e-7)


def test_klein_prompt_template_is_pinned() -> None:
    assert KLEIN_PROMPT_TEMPLATE.format("prompt") == (
        "<|im_start|>user\nprompt<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_transformer_schema_matches_canonical_checkpoint_shape() -> None:
    state = KleinTransformer().state_dict()
    assert len(state) == 201
    assert state["img_in.weight"].shape == (4096, 128)
    assert state["txt_in.weight"].shape == (4096, 12288)
    assert state["double_blocks.7.img_attn.qkv.weight"].shape == (12288, 4096)
    assert state["single_blocks.23.linear1.weight"].shape == (36864, 4096)
    assert state["final_layer.linear.weight"].shape == (128, 4096)
