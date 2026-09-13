"""The curated non-Lightning Qwen edit recipe, using the existing Recipe seam."""

from latentslate_engine.recipe import Artifact, Capability, CapabilitySet, ProductPolicy, exposed, fixed
from latentslate_engine.validation import MAX_U64
from .contracts import Qwen2511Identity, validate_sampling

_DIFFUSION = Capability("diffusion", "artifact")
_TEXT_ENCODER = Capability("text_encoder", "artifact")
_VAE = Capability("vae", "artifact")
_TOKENIZER = Capability("tokenizer", "artifact")
_PROMPT = Capability("prompt", "text")
_IMAGE_1 = Capability("image_1", "image")
_IMAGE_2 = Capability("image_2", "image", optional=True)
_IMAGE_3 = Capability("image_3", "image", optional=True)
_SEED = Capability("seed", "integer", role="seed", minimum=0, maximum=MAX_U64)
_STEPS = Capability("steps", "integer", choices=(40,))
_CFG = Capability("cfg", "number", choices=(4.0,))
_SHIFT = Capability("shift", "number", choices=(3.1,))


def _validate(values):
    validate_sampling(values["steps"], values["cfg"], values["shift"])


QWEN2511_EDIT_CAPABILITIES = CapabilitySet(
    "qwen2511.edit",
    (_DIFFUSION, _TEXT_ENCODER, _VAE, _TOKENIZER, _PROMPT,
     _IMAGE_1, _IMAGE_2, _IMAGE_3, _SEED, _STEPS, _CFG, _SHIFT),
    _validate,
)

# Euler/simple at full denoise, post-CFG norm strength one, index_timestep_zero,
# and genuine multimodal empty negative text are intrinsic to this curated core.
QWEN2511_EDIT_POLICY = ProductPolicy(
    "qwen2511.edit.curated.v1", QWEN2511_EDIT_CAPABILITIES,
    (exposed(_PROMPT), exposed(_IMAGE_1), exposed(_IMAGE_2, default=None),
     exposed(_IMAGE_3, default=None), exposed(_SEED, default=0),
     fixed(_STEPS, 40), fixed(_CFG, 4.0), fixed(_SHIFT, 3.1)),
)


def qwen2511_edit_recipe(*, diffusion, text_encoder, vae, tokenizer):
    """Bind the curated model selection with only image, prompt and seed inputs."""
    return QWEN2511_EDIT_POLICY.bind({
        "diffusion": Artifact(diffusion), "text_encoder": Artifact(text_encoder),
        "vae": Artifact(vae), "tokenizer": Artifact(tokenizer),
    })


def resolve_qwen2511_fixed_identity(definition):
    """Resolve the fixed model identity before per-request input processing."""
    if definition.capabilities is not QWEN2511_EDIT_CAPABILITIES:
        raise TypeError("recipe does not use Qwen 2511 edit capabilities")
    fields = {field.capability.key: field for field in definition.fields}
    values = {}
    for key in ("diffusion", "text_encoder", "vae", "tokenizer"):
        if fields[key].exposed:
            raise ValueError(f"pre-request Qwen identity requires fixed {key}")
        values[key] = fields[key].value.path
    return Qwen2511Identity.from_paths(**values)


def resolve_qwen2511_request(definition, overrides):
    """Compile exposed request values and fixed sampling choices for the native core."""
    if definition.capabilities is not QWEN2511_EDIT_CAPABILITIES:
        raise TypeError("recipe does not use Qwen 2511 edit capabilities")
    values = definition.resolve(overrides)
    return {key: values[key] for key in (
        "prompt", "image_1", "image_2", "image_3", "seed", "steps", "cfg", "shift",
    )}
