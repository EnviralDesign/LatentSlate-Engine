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
from latentslate_engine.klein9b.two_image import (
    Klein9BTwoImageRuntime,
    ReferenceCacheEntry,
    SourceImageIdentity,
    _scale_to_one_megapixel,
    _sigmas_for_dimensions,
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


def test_two_image_schedule_matches_pinned_flux2_scheduler() -> None:
    schedule = _sigmas_for_dimensions(4, 1237, 847, torch.device("cpu"))
    expected = torch.tensor([1.0, 0.9673759937, 0.9081227183, 0.7671545148, 0.0])
    torch.testing.assert_close(schedule, expected, rtol=0, atol=1e-7)


def test_two_image_scaling_matches_canonical_dimensions() -> None:
    first = torch.zeros((1, 630, 920, 3))
    second = torch.zeros((1, 512, 512, 3))
    assert _scale_to_one_megapixel(first, "nearest-exact").shape == (
        1,
        847,
        1237,
        3,
    )
    assert _scale_to_one_megapixel(second, "lanczos").shape == (
        1,
        1024,
        1024,
        3,
    )


def test_source_image_identity_includes_content_hash(tmp_path: Path) -> None:
    image = tmp_path / "source.png"
    image.write_bytes(b"first")
    first = SourceImageIdentity.from_path(image)
    image.write_bytes(b"other")
    second = SourceImageIdentity.from_path(image)
    assert first.sha256 != second.sha256
    assert first != second


def test_two_image_identity_change_clears_reference_state(tmp_path: Path) -> None:
    first = _identity(tmp_path)
    second = _identity(tmp_path, "-changed")
    runtime = Klein9BTwoImageRuntime(device="cpu")
    runtime.ensure_identity(first)
    runtime.references = [object(), object()]  # type: ignore[list-item]

    assert runtime.ensure_identity(second) is False
    assert runtime.references == [None, None]
    assert runtime.transformer is None
    assert runtime.vae is None
    assert runtime.conditioning is None


def test_two_image_reference_slots_invalidate_independently_and_on_swap(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    runtime = Klein9BTwoImageRuntime(device="cpu")
    runtime.vae = object()  # type: ignore[assignment]
    monkeypatch.setattr(
        "latentslate_engine.klein9b.two_image._load_rgb",
        lambda _path: torch.zeros((1, 16, 16, 3)),
    )
    monkeypatch.setattr(
        "latentslate_engine.klein9b.two_image._scale_to_one_megapixel",
        lambda image, _method: image,
    )
    monkeypatch.setattr(
        "latentslate_engine.klein9b.two_image._encode_reference",
        lambda _vae, _pixels, _device: torch.zeros((1, 128, 1, 1)),
    )

    runtime._reference(0, first, "nearest-exact")
    runtime._reference(1, second, "lanczos")
    first.write_bytes(b"first changed")
    assert runtime._reference(0, first, "nearest-exact")[1] is False
    assert runtime._reference(1, second, "lanczos")[1] is True

    second.write_bytes(b"second changed")
    assert runtime._reference(0, first, "nearest-exact")[1] is True
    assert runtime._reference(1, second, "lanczos")[1] is False

    assert runtime._reference(0, second, "nearest-exact")[1] is False
    assert runtime._reference(1, first, "lanczos")[1] is False


def test_prompt_change_reencodes_text_but_reuses_references(
    tmp_path: Path, monkeypatch
) -> None:
    identity = _identity(tmp_path)
    runtime = Klein9BTwoImageRuntime(device="cpu")
    runtime.identity = identity
    runtime.conditioning = ("old prompt", torch.zeros((1, 1, 12288)))

    class Transformer:
        def __call__(self, latent, *_args):
            return torch.zeros_like(latent)

    class BatchNorm:
        running_mean = torch.zeros(128)
        running_var = torch.ones(128)

    class Vae:
        bn = BatchNorm()

        def decode(self, latent, return_dict=False):
            return (torch.zeros((1, 3, latent.shape[2] * 8, latent.shape[3] * 8)),)

    runtime.transformer = Transformer()  # type: ignore[assignment]
    runtime.vae = Vae()  # type: ignore[assignment]
    monkeypatch.setattr(
        "latentslate_engine.klein9b.two_image._encode_prompt",
        lambda *_args: torch.ones((1, 1, 12288)),
    )
    source_identity = SourceImageIdentity(tmp_path / "source.png", "hash")
    entry = ReferenceCacheEntry(source_identity, torch.zeros((1, 128, 1, 1)), 16, 16)
    monkeypatch.setattr(runtime, "_reference", lambda *_args: (entry, True))

    result = runtime.generate_two_image(
        identity,
        "new prompt",
        tmp_path / "first.png",
        tmp_path / "second.png",
        42,
        tmp_path / "output.png",
    )

    assert result.conditioning_reused is False
    assert result.reference_reused == (True, True)
    assert runtime.conditioning is not None
    assert runtime.conditioning[0] == "new prompt"
