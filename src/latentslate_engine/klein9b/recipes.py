"""Family-owned capability and recipe adapter for Klein 9B two-image."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from latentslate_engine.recipe import (
    Artifact,
    Capability,
    CapabilitySet,
    Recipe,
    exposed,
    fixed,
)

from .contracts import (
    KLEIN_ALIGNMENT,
    KLEIN_MAX_PIXELS,
    KLEIN_MAX_SEED,
    KLEIN_MIN_SIDE,
    Klein9BIdentity,
    validate_klein_dimensions,
    validate_klein_seed,
)

_DIFFUSION = Capability("diffusion", "artifact")
_TEXT_ENCODER = Capability("text_encoder", "artifact")
_VAE = Capability("vae", "artifact")
_TOKENIZER = Capability("tokenizer", "artifact")
_LORAS = Capability("loras", "artifact", ordered=True)
_PROMPT = Capability("prompt", "text")
_IMAGE_1 = Capability("image_1", "image", role="start_image")
_IMAGE_2 = Capability("image_2", "image", role="end_image")
_WIDTH = Capability(
    "width",
    "integer",
    optional=True,
    role="width",
    minimum=KLEIN_MIN_SIDE,
    maximum=KLEIN_MAX_PIXELS // KLEIN_MIN_SIDE,
    step=KLEIN_ALIGNMENT,
)
_HEIGHT = Capability(
    "height",
    "integer",
    optional=True,
    role="height",
    minimum=KLEIN_MIN_SIDE,
    maximum=KLEIN_MAX_PIXELS // KLEIN_MIN_SIDE,
    step=KLEIN_ALIGNMENT,
)
_SEED = Capability("seed", "integer", role="seed", minimum=0, maximum=KLEIN_MAX_SEED)


def _validate_two_image_capabilities(values: Mapping[str, object]) -> None:
    validate_klein_seed(values["seed"])  # type: ignore[arg-type]
    width = values["width"]
    height = values["height"]
    if (width is None) != (height is None):
        raise ValueError(
            "width and height must either both be provided or both omitted"
        )
    if width is not None and height is not None:
        validate_klein_dimensions(width, height)  # type: ignore[arg-type]


KLEIN9B_TWO_IMAGE_CAPABILITIES = CapabilitySet(
    "flux2_klein9b.two_image",
    (
        _DIFFUSION,
        _TEXT_ENCODER,
        _VAE,
        _TOKENIZER,
        _LORAS,
        _PROMPT,
        _IMAGE_1,
        _IMAGE_2,
        _WIDTH,
        _HEIGHT,
        _SEED,
    ),
    _validate_two_image_capabilities,
)


def klein9b_two_image_recipe(
    *,
    diffusion: str | Path,
    text_encoder: str | Path,
    vae: str | Path,
    tokenizer: str | Path,
    loras: Sequence[str | Path] = (),
) -> Recipe:
    """Define one Klein product without exposing fictional sampling knobs."""
    return Recipe(
        "flux2_klein9b.two_image.v1",
        KLEIN9B_TWO_IMAGE_CAPABILITIES,
        (
            fixed(_DIFFUSION, Artifact(diffusion)),
            fixed(_TEXT_ENCODER, Artifact(text_encoder)),
            fixed(_VAE, Artifact(vae)),
            fixed(_TOKENIZER, Artifact(tokenizer)),
            exposed(_LORAS, default=tuple(Artifact(path) for path in loras)),
            exposed(_PROMPT),
            exposed(_IMAGE_1),
            exposed(_IMAGE_2),
            exposed(_WIDTH, default=None),
            exposed(_HEIGHT, default=None),
            exposed(_SEED, default=0),
        ),
    )


def resolve_klein9b_two_image(
    definition: Recipe, overrides: Mapping[str, object]
) -> tuple[Klein9BIdentity, dict[str, object]]:
    """Resolve policy into the existing Klein identity and generate arguments."""
    if definition.capabilities is not KLEIN9B_TWO_IMAGE_CAPABILITIES:
        raise TypeError("recipe does not use the Klein 9B two-image capability set")
    values = definition.resolve(overrides)
    identity = Klein9BIdentity.from_paths(
        values["diffusion"].path,  # type: ignore[union-attr]
        values["text_encoder"].path,  # type: ignore[union-attr]
        values["vae"].path,  # type: ignore[union-attr]
        values["tokenizer"].path,  # type: ignore[union-attr]
        loras=tuple(artifact.path for artifact in values["loras"]),  # type: ignore[union-attr]
    )
    request = {
        "prompt": values["prompt"],
        "first_image": Path(values["image_1"]),  # type: ignore[arg-type]
        "second_image": Path(values["image_2"]),  # type: ignore[arg-type]
        "seed": values["seed"],
        "width": values["width"],
        "height": values["height"],
    }
    return identity, request
