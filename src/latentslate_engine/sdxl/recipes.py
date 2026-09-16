"""Ordinary SDXL checkpoint recipes and caller controls."""

from latentslate_engine.recipe import (
    Artifact,
    Capability,
    CapabilitySet,
    ProductPolicy,
    exposed,
)
from latentslate_engine.validation import MAX_U64

from .contracts import ALIGNMENT, MIN_SIDE, SDXLIdentity, validate_request


def _validate(values):
    validate_request(values["width"], values["height"], values["seed"])
    if not values["prompt"].strip():
        raise ValueError("SDXL prompt must be nonempty text")


_CHECKPOINT = Capability("checkpoint", "artifact")
_VAE = Capability("vae", "artifact", optional=True)
_TOKENIZER = Capability("tokenizer", "artifact")
_PROMPT = Capability("prompt", "text")
_NEGATIVE = Capability("negative_prompt", "text")
_WIDTH = Capability(
    "width", "integer", role="width", minimum=MIN_SIDE, maximum=2048, step=ALIGNMENT
)
_HEIGHT = Capability(
    "height", "integer", role="height", minimum=MIN_SIDE, maximum=2048, step=ALIGNMENT
)
_SEED = Capability("seed", "integer", role="seed", minimum=0, maximum=MAX_U64)
_STEPS = Capability("steps", "integer", minimum=1, maximum=100, step=1)
_CFG = Capability("cfg", "number", minimum=1.0, maximum=20.0)
_SAMPLER = Capability(
    "sampler", "choice", choices=("euler", "euler_ancestral", "dpmpp_2m")
)
_SCHEDULER = Capability("scheduler", "choice", choices=("normal", "karras"))
SDXL_T2I_CAPABILITIES = CapabilitySet(
    "sdxl.t2i",
    (
        _CHECKPOINT,
        _VAE,
        _TOKENIZER,
        _PROMPT,
        _NEGATIVE,
        _WIDTH,
        _HEIGHT,
        _SEED,
        _STEPS,
        _CFG,
        _SAMPLER,
        _SCHEDULER,
    ),
    _validate,
)
SDXL_T2I_POLICY = ProductPolicy(
    "sdxl.t2i.v1",
    SDXL_T2I_CAPABILITIES,
    (
        exposed(_PROMPT),
        exposed(_NEGATIVE, default=""),
        exposed(_WIDTH, default=1024),
        exposed(_HEIGHT, default=1024),
        exposed(_SEED, default=0),
        exposed(_STEPS, default=25),
        exposed(_CFG, default=7.0),
        exposed(_SAMPLER, default="dpmpp_2m"),
        exposed(_SCHEDULER, default="karras"),
    ),
)


def sdxl_t2i_recipe(*, checkpoint, tokenizer, vae=None):
    """Bind a single-file checkpoint with embedded CLIP and optional VAE override."""
    return SDXL_T2I_POLICY.bind(
        {
            "checkpoint": Artifact(checkpoint),
            "tokenizer": Artifact(tokenizer),
            "vae": None if vae is None else Artifact(vae),
        }
    )


def resolve_sdxl_fixed_identity(definition):
    if definition.capabilities is not SDXL_T2I_CAPABILITIES:
        raise TypeError("recipe does not use SDXL T2I capabilities")
    fields = {field.capability.key: field for field in definition.fields}
    values = {}
    for key in ("checkpoint", "vae", "tokenizer"):
        if fields[key].exposed:
            raise ValueError(f"pre-request SDXL identity requires fixed {key}")
        values[key] = None if fields[key].value is None else fields[key].value.path
    return SDXLIdentity.from_paths(**values)


def resolve_sdxl_request(definition, overrides):
    if definition.capabilities is not SDXL_T2I_CAPABILITIES:
        raise TypeError("recipe does not use SDXL T2I capabilities")
    values = definition.resolve(overrides)
    return {
        key: value
        for key, value in values.items()
        if key not in {"checkpoint", "vae", "tokenizer"}
    }
