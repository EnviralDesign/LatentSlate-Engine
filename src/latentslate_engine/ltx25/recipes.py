"""LTX 2.5 recipe capabilities and request compilation, without GPU imports."""

from latentslate_engine.recipe import (
    Artifact,
    Capability,
    CapabilitySet,
    ProductPolicy,
    exposed,
    fixed,
)
from latentslate_engine.validation import MAX_U64

from .contracts import Ltx25Identity

_ARTIFACTS = tuple(
    Capability(key, "artifact")
    for key in (
        "diffusion",
        "text_encoder",
        "video_vae",
        "audio_vae",
    )
)
_UPSAMPLER = Capability("upsampler", "artifact")
_ENHANCER = Capability("prompt_enhancer", "artifact", optional=True)
_ENHANCE = Capability("prompt_enhancement", "boolean")
_ADAPTERS = Capability("transformer_adapter_artifacts", "artifact", ordered=True)
_STRENGTHS = Capability("transformer_adapter_strengths", "number", ordered=True)
_PROMPT = Capability("prompt", "text")
_DURATION = Capability(
    "duration_seconds", "number", role="duration_seconds", minimum=1, maximum=10
)
_FPS = Capability("fps", "integer", role="fps", minimum=1, maximum=120)
_SEED = Capability("seed", "integer", role="seed", minimum=0, maximum=MAX_U64)
_START = Capability("start_image", "image", role="start_image")
_END = Capability("end_image", "image", role="end_image")


def _validate(values):
    if len(values["transformer_adapter_artifacts"]) != len(
        values["transformer_adapter_strengths"]
    ):
        raise ValueError(
            "Transformer adapter artifacts and strengths must have matching length"
        )
    if not values["prompt"].strip():
        raise ValueError("LTX prompt must be nonempty text")
    if values["prompt_enhancement"] and values["prompt_enhancer"] is None:
        raise ValueError("Prompt enhancement requires its model artifact")


def _capabilities(operation):
    alignment = 32 if operation == "flf" else 64
    images = (
        ()
        if operation == "t2v"
        else (_START,)
        if operation == "i2v"
        else (_START, _END)
    )
    return CapabilitySet(
        f"ltx25.{operation}",
        (
            *_ARTIFACTS,
            *((_UPSAMPLER,) if operation != "flf" else ()),
            _ENHANCER,
            _ENHANCE,
            _ADAPTERS,
            _STRENGTHS,
            _PROMPT,
            Capability(
                "width", "integer", role="width", minimum=alignment, step=alignment
            ),
            Capability(
                "height", "integer", role="height", minimum=alignment, step=alignment
            ),
            _DURATION,
            _FPS,
            _SEED,
            *images,
        ),
        _validate,
    )


LTX25_T2V_CAPABILITIES = _capabilities("t2v")
LTX25_I2V_CAPABILITIES = _capabilities("i2v")
LTX25_FLF_CAPABILITIES = _capabilities("flf")
CAPABILITIES = {
    "t2v": LTX25_T2V_CAPABILITIES,
    "i2v": LTX25_I2V_CAPABILITIES,
    "flf": LTX25_FLF_CAPABILITIES,
}


def _policy(operation, capabilities):
    by_key = {item.key: item for item in capabilities.capabilities}
    fields = (
        exposed(_PROMPT),
        exposed(_ENHANCE, default=False),
        fixed(_ADAPTERS, ()),
        fixed(_STRENGTHS, ()),
        exposed(by_key["width"], default=512),
        exposed(by_key["height"], default=512),
        exposed(_DURATION, default=5.0),
        exposed(_FPS, default=24),
        exposed(_SEED, default=0),
    )
    fields += tuple(
        exposed(by_key[key]) for key in ("start_image", "end_image") if key in by_key
    )
    return ProductPolicy(f"ltx25.{operation}.v1", capabilities, fields)


POLICIES = {op: _policy(op, caps) for op, caps in CAPABILITIES.items()}


def validate_enhancer_requirement(definition):
    """Exposed or fixed-on enhancement requires an explicit fixed model binding."""
    fields = {item.capability.key: item for item in definition.fields}
    enhancement, artifact = fields["prompt_enhancement"], fields["prompt_enhancer"]
    if artifact.exposed:
        raise ValueError("The prompt enhancer artifact must be fixed by the recipe")
    if (enhancement.exposed or enhancement.value) and artifact.value is None:
        raise ValueError(
            "Exposed or enabled prompt enhancement requires its model artifact"
        )


def ltx25_recipe(
    operation,
    *,
    diffusion,
    text_encoder,
    video_vae,
    audio_vae,
    prompt_enhancer,
    upsampler=None,
):
    """Bind the curated default-off, exposed enhancement recipe."""
    bindings = {
        "diffusion": Artifact(diffusion),
        "text_encoder": Artifact(text_encoder),
        "video_vae": Artifact(video_vae),
        "audio_vae": Artifact(audio_vae),
        "prompt_enhancer": Artifact(prompt_enhancer) if prompt_enhancer else None,
    }
    if operation != "flf":
        if upsampler is None:
            raise ValueError("Two-stage LTX recipes require a spatial upsampler")
        bindings["upsampler"] = Artifact(upsampler)
    recipe = POLICIES[operation].bind(bindings)
    validate_enhancer_requirement(recipe)
    return recipe


def resolve_ltx25_request(definition, overrides):
    """Compile duration and ordered image roles into the native request."""
    validate_enhancer_requirement(definition)
    values = definition.resolve(overrides)
    request = {
        key: values[key]
        for key in ("prompt", "width", "height", "fps", "seed", "prompt_enhancement")
    }
    request["frame_count"] = (
        int(values["duration_seconds"] * values["fps"]) // 8 * 8 + 1
    )
    for source, target in (
        ("start_image", "image_path"),
        ("end_image", "last_image_path"),
    ):
        if source in values:
            request[target] = values[source]
    return request


def resolve_ltx25_identity(definition, inputs=None):
    """Resolve fixed artifact state before launching or reusing a worker."""
    if not any(definition.capabilities is caps for caps in CAPABILITIES.values()):
        raise TypeError("Recipe does not use LTX 2.5 capabilities")
    validate_enhancer_requirement(definition)
    paths = {}
    fields = {item.capability.key: item for item in definition.fields}
    adapters = fields["transformer_adapter_artifacts"]
    strengths = fields["transformer_adapter_strengths"]
    if adapters.exposed:
        raise ValueError("LTX transformer artifacts must be fixed by the recipe")
    strength_values = strengths.capability.normalize(
        (inputs or {}).get(strengths.capability.key, strengths.value)
        if strengths.exposed
        else strengths.value
    )
    strengths.validate(strength_values)
    transformer_adapters = tuple(
        (artifact.path, strength)
        for artifact, strength in zip(adapters.value, strength_values, strict=True)
    )
    for field in definition.fields:
        if field.capability.key == "transformer_adapter_artifacts":
            continue
        if field.capability.value_type != "artifact":
            continue
        if field.exposed:
            raise ValueError("LTX model artifacts must be fixed by the recipe")
        paths[field.capability.key + "_path"] = (
            None if field.value is None else field.value.path
        )
    return Ltx25Identity.from_paths(transformer_adapters=transformer_adapters, **paths)
