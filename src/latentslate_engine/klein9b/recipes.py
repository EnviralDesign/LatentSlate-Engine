"""Family-owned Klein 9B capabilities, products and native recipe adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from latentslate_engine.recipe import (
    Artifact,
    Capability,
    CapabilitySet,
    ProductPolicy,
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
_T2I_WIDTH = Capability(
    "width",
    "integer",
    role="width",
    minimum=KLEIN_MIN_SIDE,
    maximum=KLEIN_MAX_PIXELS // KLEIN_MIN_SIDE,
    step=KLEIN_ALIGNMENT,
)
_T2I_HEIGHT = Capability(
    "height",
    "integer",
    role="height",
    minimum=KLEIN_MIN_SIDE,
    maximum=KLEIN_MAX_PIXELS // KLEIN_MIN_SIDE,
    step=KLEIN_ALIGNMENT,
)


def _validate_t2i_capabilities(values: Mapping[str, object]) -> None:
    validate_klein_seed(values["seed"])  # type: ignore[arg-type]
    validate_klein_dimensions(values["width"], values["height"])  # type: ignore[arg-type]


KLEIN9B_T2I_CAPABILITIES = CapabilitySet(
    "flux2_klein9b.t2i",
    (
        _DIFFUSION,
        _TEXT_ENCODER,
        _VAE,
        _TOKENIZER,
        _LORAS,
        _PROMPT,
        _T2I_WIDTH,
        _T2I_HEIGHT,
        _SEED,
    ),
    _validate_t2i_capabilities,
)


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


KLEIN9B_T2I_POLICY = ProductPolicy(
    "flux2_klein9b.t2i.v1",
    KLEIN9B_T2I_CAPABILITIES,
    (
        fixed(_LORAS, ()),
        exposed(_PROMPT),
        exposed(_T2I_WIDTH, default=768),
        exposed(_T2I_HEIGHT, default=768),
        exposed(_SEED, default=0),
    ),
)

KLEIN9B_TWO_IMAGE_EXPLICIT_POLICY = ProductPolicy(
    "flux2_klein9b.two_image.explicit.v1",
    KLEIN9B_TWO_IMAGE_CAPABILITIES,
    (
        fixed(_LORAS, ()),
        exposed(_PROMPT),
        exposed(_IMAGE_1),
        exposed(_IMAGE_2),
        exposed(_WIDTH, default=768, nullable=False),
        exposed(_HEIGHT, default=768, nullable=False),
        exposed(_SEED, default=0),
    ),
)


def klein9b_t2i_recipe(
    *,
    diffusion: str | Path,
    text_encoder: str | Path,
    vae: str | Path,
    tokenizer: str | Path,
) -> Recipe:
    """Bind the no-LoRA T2I product with concrete output geometry."""
    return KLEIN9B_T2I_POLICY.bind(
        {
            "diffusion": Artifact(diffusion),
            "text_encoder": Artifact(text_encoder),
            "vae": Artifact(vae),
            "tokenizer": Artifact(tokenizer),
        }
    )


def klein9b_two_image_explicit_recipe(
    *,
    diffusion: str | Path,
    text_encoder: str | Path,
    vae: str | Path,
    tokenizer: str | Path,
) -> Recipe:
    """Bind the no-LoRA two-image product with explicit output geometry."""
    return KLEIN9B_TWO_IMAGE_EXPLICIT_POLICY.bind(
        {
            "diffusion": Artifact(diffusion),
            "text_encoder": Artifact(text_encoder),
            "vae": Artifact(vae),
            "tokenizer": Artifact(tokenizer),
        }
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


def _klein_identity(values: Mapping[str, object]) -> Klein9BIdentity:
    return Klein9BIdentity.from_paths(
        values["diffusion"].path,  # type: ignore[union-attr]
        values["text_encoder"].path,  # type: ignore[union-attr]
        values["vae"].path,  # type: ignore[union-attr]
        values["tokenizer"].path,  # type: ignore[union-attr]
        loras=tuple(artifact.path for artifact in values["loras"]),  # type: ignore[union-attr]
    )


def resolve_klein9b_fixed_identity(definition: Recipe) -> Klein9BIdentity:
    """Resolve fixed Klein model state once, before caller inputs are available."""
    if (
        definition.capabilities is not KLEIN9B_T2I_CAPABILITIES
        and definition.capabilities is not KLEIN9B_TWO_IMAGE_CAPABILITIES
    ):
        raise TypeError("recipe does not use a Klein 9B capability set")
    fields = {field.capability.key: field for field in definition.fields}
    values = {}
    for key in ("diffusion", "text_encoder", "vae", "tokenizer", "loras"):
        field = fields[key]
        if field.exposed:
            raise ValueError(f"pre-request Klein identity requires fixed {key}")
        values[key] = field.value
    return _klein_identity(values)


def _t2i_request(values: Mapping[str, object]) -> dict[str, object]:
    return {key: values[key] for key in ("prompt", "seed", "width", "height")}


def _two_image_request(values: Mapping[str, object]) -> dict[str, object]:
    return {
        "prompt": values["prompt"],
        "first_image": Path(values["image_1"]),  # type: ignore[arg-type]
        "second_image": Path(values["image_2"]),  # type: ignore[arg-type]
        "seed": values["seed"],
        "width": values["width"],
        "height": values["height"],
    }


def resolve_klein9b_t2i(
    definition: Recipe, overrides: Mapping[str, object]
) -> tuple[Klein9BIdentity, dict[str, object]]:
    """Resolve T2I policy into the shared identity and native generate arguments."""
    if definition.capabilities is not KLEIN9B_T2I_CAPABILITIES:
        raise TypeError("recipe does not use the Klein 9B T2I capability set")
    values = definition.resolve(overrides)
    return _klein_identity(values), _t2i_request(values)


def resolve_klein9b_t2i_request(
    definition: Recipe, overrides: Mapping[str, object]
) -> dict[str, object]:
    """Resolve caller state without reading model identity metadata."""
    if definition.capabilities is not KLEIN9B_T2I_CAPABILITIES:
        raise TypeError("recipe does not use the Klein 9B T2I capability set")
    return _t2i_request(definition.resolve(overrides))


def resolve_klein9b_two_image(
    definition: Recipe, overrides: Mapping[str, object]
) -> tuple[Klein9BIdentity, dict[str, object]]:
    """Resolve policy into the existing Klein identity and generate arguments."""
    if definition.capabilities is not KLEIN9B_TWO_IMAGE_CAPABILITIES:
        raise TypeError("recipe does not use the Klein 9B two-image capability set")
    values = definition.resolve(overrides)
    return _klein_identity(values), _two_image_request(values)


def resolve_klein9b_two_image_request(
    definition: Recipe, overrides: Mapping[str, object]
) -> dict[str, object]:
    """Resolve ordered caller references/geometry without model metadata IO."""
    if definition.capabilities is not KLEIN9B_TWO_IMAGE_CAPABILITIES:
        raise TypeError("recipe does not use the Klein 9B two-image capability set")
    return _two_image_request(definition.resolve(overrides))
