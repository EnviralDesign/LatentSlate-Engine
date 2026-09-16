"""Ideogram v4 capability policy and native request compilation."""

from latentslate_engine.recipe import (
    Adapter,
    Artifact,
    Capability,
    CapabilitySet,
    ProductPolicy,
    exposed,
)
from latentslate_engine.validation import MAX_U64

from .contracts import (
    ALIGNMENT,
    MIN_SIDE,
    Ideogram4Identity,
    validate_adapters,
    validate_request,
)


def _validate(values):
    validate_request(values["width"], values["height"], values["seed"])
    validate_adapters(tuple((a.artifact.path, a.strength) for a in values["adapters"]))
    if not values["prompt"].strip():
        raise ValueError("Ideogram v4 prompt must be nonempty text")


_DIFFUSION = Capability("diffusion", "artifact")
_TEXT_ENCODER = Capability("text_encoder", "artifact")
_VAE = Capability("vae", "artifact")
_TOKENIZER = Capability("tokenizer", "artifact")
_NEGATIVE_DIFFUSION = Capability("negative_diffusion", "artifact", optional=True)
_ADAPTERS = Capability("adapters", "adapter", ordered=True)
_PROMPT = Capability("prompt", "text")
_WIDTH = Capability(
    "width", "integer", role="width", minimum=MIN_SIDE, maximum=2048, step=ALIGNMENT
)
_HEIGHT = Capability(
    "height", "integer", role="height", minimum=MIN_SIDE, maximum=2048, step=ALIGNMENT
)
_SEED = Capability("seed", "integer", role="seed", minimum=0, maximum=MAX_U64)

IDEOGRAM4_T2I_CAPABILITIES = CapabilitySet(
    "ideogram4.t2i",
    (
        _DIFFUSION,
        _TEXT_ENCODER,
        _VAE,
        _TOKENIZER,
        _NEGATIVE_DIFFUSION,
        _ADAPTERS,
        _PROMPT,
        _WIDTH,
        _HEIGHT,
        _SEED,
    ),
    _validate,
)
IDEOGRAM4_T2I_POLICY = ProductPolicy(
    "ideogram4.t2i.v1",
    IDEOGRAM4_T2I_CAPABILITIES,
    (
        exposed(_PROMPT),
        exposed(_WIDTH, default=1024),
        exposed(_HEIGHT, default=1024),
        exposed(_SEED, default=0),
    ),
)


def ideogram4_t2i_recipe(
    *,
    diffusion,
    negative_diffusion,
    text_encoder,
    vae,
    tokenizer,
    adapters: tuple[Adapter, ...] = (),
):
    """Bind the model components to the ordinary fixed/exposed recipe policy."""
    return IDEOGRAM4_T2I_POLICY.bind(
        {
            "diffusion": Artifact(diffusion),
            "text_encoder": Artifact(text_encoder),
            "vae": Artifact(vae),
            "tokenizer": Artifact(tokenizer),
            "negative_diffusion": None
            if negative_diffusion is None
            else Artifact(negative_diffusion),
            "adapters": adapters,
        }
    )


def resolve_ideogram4_fixed_identity(definition):
    """Resolve state-bearing fields before loading the native family runtime."""
    if definition.capabilities is not IDEOGRAM4_T2I_CAPABILITIES:
        raise TypeError("recipe does not use Ideogram v4 T2I capabilities")
    fields = {field.capability.key: field for field in definition.fields}
    values = {}
    for key in ("diffusion", "negative_diffusion", "text_encoder", "vae", "tokenizer"):
        if fields[key].exposed:
            raise ValueError(f"pre-request Ideogram v4 identity requires fixed {key}")
        values[key] = None if fields[key].value is None else fields[key].value.path
    if fields["adapters"].exposed:
        raise ValueError("pre-request Ideogram v4 identity requires fixed adapters")
    values["adapters"] = tuple(
        (a.artifact.path, a.strength) for a in fields["adapters"].value
    )
    return Ideogram4Identity.from_paths(**values)


def resolve_ideogram4_request(definition, overrides):
    """Resolve caller values through the recipe's fixed/exposed policy."""
    if definition.capabilities is not IDEOGRAM4_T2I_CAPABILITIES:
        raise TypeError("recipe does not use Ideogram v4 T2I capabilities")
    values = definition.resolve(overrides)
    return {key: values[key] for key in ("prompt", "width", "height", "seed")}
