"""Krea-owned capability, certified product and native request compilation."""

from latentslate_engine.recipe import (
    Artifact,
    Capability,
    CapabilitySet,
    ProductPolicy,
    exposed,
)
from latentslate_engine.validation import MAX_U64

from .contracts import ALIGNMENT, MIN_SIDE, Krea2Identity, validate_request


def _validate(values):
    validate_request(values["width"], values["height"], values["seed"])
    if not values["prompt"].strip():
        raise ValueError("Krea prompt must be nonempty text")


_DIFFUSION = Capability("diffusion", "artifact")
_TEXT_ENCODER = Capability("text_encoder", "artifact")
_VAE = Capability("vae", "artifact")
_TOKENIZER = Capability("tokenizer", "artifact")
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
    (_DIFFUSION, _TEXT_ENCODER, _VAE, _TOKENIZER, _PROMPT, _WIDTH, _HEIGHT, _SEED),
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


def krea2_t2i_recipe(*, diffusion, text_encoder, vae, tokenizer):
    """Bind the certified enhanced-prompt, eight-step Turbo image product."""
    return KREA2_T2I_POLICY.bind(
        {
            "diffusion": Artifact(diffusion),
            "text_encoder": Artifact(text_encoder),
            "vae": Artifact(vae),
            "tokenizer": Artifact(tokenizer),
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
    return Krea2Identity.from_paths(**values)


def resolve_krea2_request(definition, overrides):
    """Resolve caller inputs through the recipe's fixed/exposed policy."""
    if definition.capabilities is not KREA2_T2I_CAPABILITIES:
        raise TypeError("recipe does not use Krea Turbo T2I capabilities")
    values = definition.resolve(overrides)
    return {key: values[key] for key in ("prompt", "width", "height", "seed")}
