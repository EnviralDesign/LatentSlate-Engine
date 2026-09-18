"""Recipe-selectable artifacts and explicit camera controls for novel views."""

from latentslate_engine.recipe import Artifact, Capability, CapabilitySet, ProductPolicy, exposed
from latentslate_engine.validation import MAX_U64
from .contracts import MetaViewIdentity, MIN_SIDE, ALIGNMENT, validate_request

ARTIFACTS = ("diffusion", "text_encoder", "vae", "tokenizer", "geometry_model", "depth_model")
_IMAGE = Capability("image", "image")
_WIDTH = Capability("width", "integer", role="width", minimum=MIN_SIDE, step=ALIGNMENT)
_HEIGHT = Capability("height", "integer", role="height", minimum=MIN_SIDE, step=ALIGNMENT)
_SEED = Capability("seed", "integer", role="seed", minimum=0, maximum=MAX_U64)
_YAW = Capability("yaw", "number", minimum=-180, maximum=180)
_PITCH = Capability("pitch", "number", minimum=-90, maximum=90)
_RADIUS = Capability("radius", "number", optional=True, minimum=0)


def _validate(values):
    validate_request(*(values[key] for key in ("width", "height", "seed", "yaw", "pitch", "radius")))


METAVIEW_CAPABILITIES = CapabilitySet("metaview.novel_view", (
    *(Capability(key, "artifact") for key in ARTIFACTS),
    _IMAGE, _WIDTH, _HEIGHT, _SEED, _YAW, _PITCH, _RADIUS,
), _validate)

METAVIEW_POLICY = ProductPolicy("metaview.novel_view.v1", METAVIEW_CAPABILITIES, (
    exposed(_IMAGE), exposed(_WIDTH, default=960), exposed(_HEIGHT, default=528),
    exposed(_SEED, default=0), exposed(_YAW, default=0.0), exposed(_PITCH, default=0.0),
    # Zero is the explicit automatic-depth sentinel. Keep the field optional
    # at the request boundary, but expose a concrete numeric default so
    # consumers do not need nullable scalar-input support.
    exposed(_RADIUS, default=0, nullable=False),
))


def metaview_recipe(**artifacts):
    return METAVIEW_POLICY.bind({key: Artifact(value) for key, value in artifacts.items()})


def resolve_identity(definition):
    if definition.capabilities is not METAVIEW_CAPABILITIES:
        raise TypeError("recipe does not use MetaView capabilities")
    fields = {field.capability.key: field for field in definition.fields}
    if any(fields[key].exposed for key in ARTIFACTS):
        raise ValueError("MetaView artifacts must be fixed by the recipe")
    return MetaViewIdentity.from_paths(**{key: fields[key].value.path for key in ARTIFACTS})


def resolve_request(definition, inputs):
    values = definition.resolve(inputs)
    return {key: values[key] for key in ("image", "width", "height", "seed", "yaw", "pitch", "radius")}
