"""Krea-owned capability, certified product and native request compilation."""

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
    Krea2Identity,
    validate_adapters,
    validate_request,
)


def _validate(values):
    validate_request(values["width"], values["height"], values["seed"])
    validate_adapters(
        tuple((adapter.artifact.path, adapter.strength) for adapter in values["adapters"])
    )
    if not values["prompt"].strip():
        raise ValueError("Krea prompt must be nonempty text")


_DIFFUSION = Capability("diffusion", "artifact")
_TEXT_ENCODER = Capability("text_encoder", "artifact")
_VAE = Capability("vae", "artifact")
_TOKENIZER = Capability("tokenizer", "artifact")
_ADAPTERS = Capability("adapters", "adapter", ordered=True)
_PROMPT_SUFFIX = Capability("prompt_suffix", "text")
_PROMPT = Capability("prompt", "text")
_WIDTH = Capability(
    "width", "integer", role="width", minimum=MIN_SIDE, maximum=2048, step=ALIGNMENT
)
_HEIGHT = Capability(
    "height", "integer", role="height", minimum=MIN_SIDE, maximum=2048, step=ALIGNMENT
)
_SEED = Capability("seed", "integer", role="seed", minimum=0, maximum=MAX_U64)

KREA2_T2I_CAPABILITIES = CapabilitySet(
    "krea2.t2i",
    (
        _DIFFUSION, _TEXT_ENCODER, _VAE, _TOKENIZER, _ADAPTERS,
        _PROMPT_SUFFIX, _PROMPT, _WIDTH, _HEIGHT, _SEED,
    ),
    _validate,
)
KREA2_T2I_POLICY = ProductPolicy(
    "krea2.turbo.t2i.v1",
    KREA2_T2I_CAPABILITIES,
    (
        exposed(_PROMPT),
        exposed(_WIDTH, default=1024),
        exposed(_HEIGHT, default=1024),
        exposed(_SEED, default=0),
    ),
)


def krea2_t2i_recipe(
    *, diffusion, text_encoder, vae, tokenizer,
    adapters: tuple[Adapter, ...] = (), prompt_suffix: str = "",
):
    """Bind Turbo artifacts and fixed post-enhancement prompt text."""
    return KREA2_T2I_POLICY.bind(
        {
            "diffusion": Artifact(diffusion),
            "text_encoder": Artifact(text_encoder),
            "vae": Artifact(vae),
            "tokenizer": Artifact(tokenizer),
            "adapters": adapters,
            "prompt_suffix": prompt_suffix,
        }
    )


def resolve_krea2_fixed_identity(definition):
    """Resolve fixed loaded state independently from per-generation values."""
    if definition.capabilities is not KREA2_T2I_CAPABILITIES:
        raise TypeError("recipe does not use Krea Turbo T2I capabilities")
    fields = {field.capability.key: field for field in definition.fields}
    values = {}
    for key in ("diffusion", "text_encoder", "vae", "tokenizer"):
        if fields[key].exposed:
            raise ValueError(f"pre-request Krea identity requires fixed {key}")
        values[key] = fields[key].value.path
    if fields["adapters"].exposed:
        raise ValueError("pre-request Krea identity requires fixed adapters")
    values["adapters"] = tuple(
        (adapter.artifact.path, adapter.strength)
        for adapter in fields["adapters"].value
    )
    return Krea2Identity.from_paths(**values)


def resolve_krea2_request(definition, overrides):
    """Resolve caller inputs through the recipe's fixed/exposed policy."""
    if definition.capabilities is not KREA2_T2I_CAPABILITIES:
        raise TypeError("recipe does not use Krea Turbo T2I capabilities")
    values = definition.resolve(overrides)
    return {
        key: values[key]
        for key in ("prompt", "width", "height", "seed", "prompt_suffix")
    }
