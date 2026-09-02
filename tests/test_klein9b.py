from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

import latentslate_engine.klein9b.two_image as klein_two_image
from latentslate_engine.klein9b.model import KleinTransformer, Linear
from latentslate_engine.klein9b.runtime import (
    KLEIN_ALIGNMENT,
    KLEIN_MAX_ASPECT,
    KLEIN_MAX_PIXELS,
    KLEIN_MAX_SEED,
    KLEIN_MIN_SIDE,
    KLEIN_PROMPT_TEMPLATE,
    ArtifactIdentity,
    Klein9BIdentity,
    Klein9BRuntime,
    _apply_loras,
    _sigmas,
    _sigmas_for_dimensions,
    _unpack_latent,
    validate_klein_request,
)
from latentslate_engine.klein9b.two_image import (
    Klein9BTwoImageRuntime,
    ReferenceCacheEntry,
    SourceImageIdentity,
    _one_megapixel_dimensions,
    _scale_to_one_megapixel,
    _target_geometry,
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


def test_complete_product_geometry_lattice_matches_recovered_domain() -> None:
    for width in range(KLEIN_MIN_SIDE, 2048 + KLEIN_ALIGNMENT, KLEIN_ALIGNMENT):
        for height in range(
            KLEIN_MIN_SIDE, 2048 + KLEIN_ALIGNMENT, KLEIN_ALIGNMENT
        ):
            accepted = (
                width * height <= KLEIN_MAX_PIXELS
                and max(width, height) <= min(width, height) * KLEIN_MAX_ASPECT
            )
            if accepted:
                validate_klein_request(width, height, KLEIN_MAX_SEED)
            else:
                with pytest.raises(ValueError):
                    validate_klein_request(width, height, 0)


@pytest.mark.parametrize(
    ("width", "height", "seed"),
    [
        (240, 1024, 0),
        (256, 1008, -1),
        (257, 1024, 0),
        (1024, 1040, 0),
        (256, 1040, 0),
        (256, 1024, KLEIN_MAX_SEED + 1),
    ],
)
def test_unsupported_product_requests_are_rejected(
    width: int, height: int, seed: int
) -> None:
    with pytest.raises(ValueError):
        validate_klein_request(width, height, seed)


def test_product_requests_require_integer_types() -> None:
    with pytest.raises(TypeError):
        validate_klein_request(512.0, 512, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate_klein_request(512, 512, True)


def test_t2i_geometry_propagates_without_invalidating_warm_conditioning(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path)
    runtime = Klein9BRuntime(device="cpu")
    runtime.identity = identity
    runtime.conditioning = ("prompt", torch.zeros((1, 1, 12288)))
    observed_shapes: list[tuple[int, ...]] = []

    class Transformer:
        def __call__(self, latent, *_args):
            observed_shapes.append(tuple(latent.shape))
            return torch.zeros_like(latent)

    class BatchNorm:
        running_mean = torch.zeros(128)
        running_var = torch.ones(128)

    class Vae:
        bn = BatchNorm()

        def decode(self, latent, return_dict=False):
            return (torch.zeros((1, 3, latent.shape[2] * 8, latent.shape[3] * 8)),)

    transformer = Transformer()
    runtime.transformer = transformer  # type: ignore[assignment]
    runtime.vae = Vae()  # type: ignore[assignment]

    landscape = runtime.generate(
        identity,
        "prompt",
        42,
        tmp_path / "landscape.png",
        width=1024,
        height=512,
    )
    portrait = runtime.generate(
        identity,
        "prompt",
        43,
        tmp_path / "portrait.png",
        width=512,
        height=1024,
    )

    assert landscape.conditioning_reused is True
    assert portrait.conditioning_reused is True
    assert runtime.transformer is transformer
    assert observed_shapes[:4] == [(1, 128, 32, 64)] * 4
    assert observed_shapes[4:] == [(1, 128, 64, 32)] * 4
    with Image.open(landscape.output) as image:
        assert image.size == (1024, 512)
    with Image.open(portrait.output) as image:
        assert image.size == (512, 1024)


def test_invalid_t2i_request_does_not_switch_model_identity(tmp_path: Path) -> None:
    first = _identity(tmp_path)
    second = _identity(tmp_path, "-changed")
    runtime = Klein9BRuntime(device="cpu")
    runtime.identity = first
    transformer = object()
    runtime.transformer = transformer  # type: ignore[assignment]

    with pytest.raises(ValueError):
        runtime.generate(
            second,
            "prompt",
            42,
            tmp_path / "output.png",
            width=257,
            height=512,
        )

    assert runtime.identity == first
    assert runtime.transformer is transformer


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


def test_raw_fp8_linear_uses_the_compute_dtype_when_no_scale_is_present() -> None:
    linear = Linear(2, 1, device="cpu")
    linear.weight = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0]], dtype=torch.float8_e4m3fn), requires_grad=False
    )

    output = linear(torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16))

    torch.testing.assert_close(
        output, torch.tensor([[11.0]], dtype=torch.bfloat16), rtol=0, atol=0
    )


def test_linear_releases_an_unprepared_dynamic_weight_after_its_forward() -> None:
    class DynamicWeight:
        def __init__(self) -> None:
            self.materialized = 0
            self.unpinned = 0

        def materialize(self, _: int) -> torch.Tensor:
            self.materialized += 1
            return torch.tensor([[1.0, 2.0]])

        def unpin(self, _: int) -> None:
            self.unpinned += 1

    linear = Linear(2, 1, device="cpu")
    dynamic_weight = DynamicWeight()
    linear.bind_dynamic_weight(dynamic_weight, 0)

    output = linear(torch.tensor([[3.0, 4.0]]))

    torch.testing.assert_close(output, torch.tensor([[11.0]]))
    assert dynamic_weight.materialized == 1
    assert dynamic_weight.unpinned == 1


def test_linear_applies_observed_lora_and_lokr_updates() -> None:
    linear = Linear(2, 2, device="cpu")
    linear.weight = torch.nn.Parameter(torch.zeros((2, 2)), requires_grad=False)
    linear.add_weight_update(
        "lora", torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0, 4.0]])
    )
    linear.add_weight_update(
        "lokr", torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0]])
    )

    output = linear(torch.tensor([[1.0, 2.0]]))

    torch.testing.assert_close(output, torch.tensor([[36.0, 77.0]]))


@pytest.mark.parametrize("include_alpha", [False, True])
def test_direct_lokr_accepts_optional_alpha_metadata(
    tmp_path: Path, include_alpha: bool
) -> None:
    checkpoint = tmp_path / "direct-lokr.safetensors"
    tensors = {
        "diffusion_model.target.lokr_w1": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "diffusion_model.target.lokr_w2": torch.tensor([[0.5]]),
    }
    if include_alpha:
        tensors["diffusion_model.target.alpha"] = torch.tensor(8.0)
    save_file(tensors, str(checkpoint))

    transformer = torch.nn.Module()
    target = Linear(2, 2, device="cpu")
    transformer.add_module("target", target)

    _apply_loras(
        transformer,  # type: ignore[arg-type]
        (ArtifactIdentity.from_path(checkpoint),),
        torch.device("cpu"),
    )

    assert len(target.weight_updates) == 1
    kind, first, second = target.weight_updates[0]
    assert kind == "lokr"
    torch.testing.assert_close(first, tensors["diffusion_model.target.lokr_w1"])
    torch.testing.assert_close(second, tensors["diffusion_model.target.lokr_w2"])


def test_ordered_lora_identity_change_releases_model_state(tmp_path: Path) -> None:
    base = _identity(tmp_path)
    first_lora = tmp_path / "first-lora.safetensors"
    second_lora = tmp_path / "second-lora.safetensors"
    first_lora.write_bytes(b"first")
    second_lora.write_bytes(b"second")
    arguments = (
        base.diffusion.path,
        base.text_encoder.path,
        base.vae.path,
        base.tokenizer,
    )
    first = Klein9BIdentity.from_paths(*arguments, loras=(first_lora, second_lora))
    reversed_order = Klein9BIdentity.from_paths(
        *arguments, loras=(second_lora, first_lora)
    )
    runtime = Klein9BRuntime(device="cpu")
    runtime.identity = first
    runtime.transformer = object()  # type: ignore[assignment]

    assert runtime.ensure_identity(reversed_order) is False
    assert runtime.transformer is None


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
    assert _one_megapixel_dimensions(920, 630) == (1237, 847)


def test_two_image_target_geometry_preserves_source_mode_and_allows_explicit_canvas(
) -> None:
    assert _target_geometry(1237, 847, None, None) == (1232, 832, 1237, 847)
    assert _target_geometry(1237, 847, 512, 1024) == (512, 1024, 512, 1024)
    with pytest.raises(ValueError):
        _target_geometry(1237, 847, 512, None)


def test_source_image_identity_includes_content_hash(tmp_path: Path) -> None:
    image = tmp_path / "source.png"
    image.write_bytes(b"first")
    first = SourceImageIdentity.from_path(image)
    image.write_bytes(b"other")
    second = SourceImageIdentity.from_path(image)
    assert first.sha256 != second.sha256
    assert first != second


def test_two_image_reference_reuses_same_content_at_different_path(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.png"
    duplicate = tmp_path / "duplicate.png"
    first.write_bytes(b"same content")
    duplicate.write_bytes(first.read_bytes())
    runtime = Klein9BTwoImageRuntime(device="cpu")
    entry = ReferenceCacheEntry(
        SourceImageIdentity.from_path(first),
        torch.zeros((1, 128, 1, 1)),
        16,
        16,
    )
    runtime.references[0] = entry

    cached, reused = runtime._reference(0, duplicate, "nearest-exact")

    assert reused is True
    assert cached is entry


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
    source_identity = SourceImageIdentity(tmp_path / "source.png", 1, "hash")
    entry = ReferenceCacheEntry(source_identity, torch.zeros((1, 128, 1, 1)), 16, 16)
    monkeypatch.setattr(runtime, "_reference", lambda *_args: (entry, True))
    Image.new("RGB", (512, 512)).save(tmp_path / "first.png")
    Image.new("RGB", (512, 512)).save(tmp_path / "second.png")

    result = runtime.generate_two_image(
        identity,
        "new prompt",
        tmp_path / "first.png",
        tmp_path / "second.png",
        42,
        tmp_path / "output.png",
        width=512,
        height=256,
    )

    assert result.conditioning_reused is False
    assert result.reference_reused == (True, True)
    assert runtime.conditioning is not None
    assert runtime.conditioning[0] == "new prompt"
    with Image.open(result.output) as image:
        assert image.size == (512, 256)


def test_one_image_uses_one_reference_and_applies_loras(
    tmp_path: Path, monkeypatch
) -> None:
    lora = tmp_path / "adapter.safetensors"
    lora.write_bytes(b"adapter")
    identity = replace(
        _identity(tmp_path), loras=(ArtifactIdentity.from_path(lora),)
    )
    image = tmp_path / "source.png"
    Image.new("RGB", (512, 512)).save(image)
    runtime = Klein9BTwoImageRuntime(device="cpu")
    runtime.identity = identity
    runtime.conditioning = ("prompt", torch.zeros((1, 1, 12288)))

    class Transformer:
        reference_latents: tuple[torch.Tensor, ...] | None = None

        def __call__(self, latent, _timestep, _context, _mask, reference_latents):
            self.reference_latents = reference_latents
            return torch.zeros_like(latent)

    class BatchNorm:
        running_mean = torch.zeros(128)
        running_var = torch.ones(128)

    class Vae:
        bn = BatchNorm()

        def decode(self, latent, return_dict=False):
            return (torch.zeros((1, 3, latent.shape[2] * 8, latent.shape[3] * 8)),)

    transformer = Transformer()
    runtime.vae = Vae()  # type: ignore[assignment]
    reference = ReferenceCacheEntry(
        SourceImageIdentity.from_path(image), torch.zeros((1, 128, 1, 1)), 16, 16
    )
    monkeypatch.setattr(runtime, "_reference", lambda *_args: (reference, True))
    monkeypatch.setattr(klein_two_image, "_load_transformer", lambda *_args: transformer)
    applied: list[tuple[object, tuple[ArtifactIdentity, ...], torch.device]] = []
    monkeypatch.setattr(
        klein_two_image,
        "_apply_loras",
        lambda model, loras, device: applied.append((model, loras, device)),
    )

    result = runtime.generate_one_image(
        identity,
        "prompt",
        image,
        42,
        tmp_path / "output.png",
        width=256,
        height=256,
    )

    assert transformer.reference_latents is not None
    assert len(transformer.reference_latents) == 1
    assert transformer.reference_latents[0] is reference.latent
    assert result.reference_reused == (True,)
    assert applied == [(transformer, identity.loras, torch.device("cpu"))]
