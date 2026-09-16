"""H3 recipe capabilities and explicit request compilation, without GPU imports."""

from latentslate_engine.recipe import (
    Artifact,
    Capability,
    CapabilitySet,
    ProductPolicy,
    exposed,
    fixed,
)
from latentslate_engine.validation import MAX_U64

from .contracts import (
    ALIGNMENT,
    FRAME_RATE,
    MIN_SIDE,
    H3Identity,
    validate_adapters,
    validate_canvas,
)

ARTIFACT_KEYS = ("diffusion", "text_encoder", "video_vae", "audio_vae", "tokenizer")
REFERENCE_KEYS = {
    kind: tuple(f"reference_{kind}_{index}" for index in range(1, count + 1))
    for kind, count in (("image", 9), ("video", 3), ("video_audio", 3), ("audio", 3))
}
_ARTIFACTS = tuple(Capability(key, "artifact") for key in ARTIFACT_KEYS)
_ADAPTERS = Capability("adapters", "adapter", ordered=True)
_PROMPT = Capability("prompt", "text")
_WIDTH = Capability("width", "integer", role="width", minimum=MIN_SIDE, step=ALIGNMENT)
_HEIGHT = Capability(
    "height", "integer", role="height", minimum=MIN_SIDE, step=ALIGNMENT
)
_FPS = Capability("fps", "integer", role="fps", choices=(FRAME_RATE,))
# The curated model's documented trained span ends at 362 frames (about 15s).
_DURATION = Capability(
    "duration_seconds",
    "number",
    role="duration_seconds",
    minimum=5 / FRAME_RATE,
    maximum=362 / FRAME_RATE,
)
_SEED = Capability("seed", "integer", role="seed", minimum=0, maximum=MAX_U64)
_START = Capability("start_image", "image", role="start_image")
_REFERENCE_SIZE = Capability("reference_image_size", "choice", choices=("match", "max"))
_REFERENCES = tuple(
    Capability(key, "audio" if kind == "video_audio" else kind, optional=True)
    for kind, keys in REFERENCE_KEYS.items()
    for key in keys
)


def _validate(values):
    validate_canvas(values["width"], values["height"])
    validate_adapters(
        (adapter.artifact.path, adapter.strength) for adapter in values["adapters"]
    )
    if not values["prompt"].strip():
        raise ValueError("H3 prompt must be nonempty text")
    for video, audio in zip(REFERENCE_KEYS["video"], REFERENCE_KEYS["video_audio"]):
        if values.get(audio) is not None and values.get(video) is None:
            raise ValueError(f"{audio} requires its paired {video}")


def _policy(operation):
    media = (
        (_START,) if operation == "i2v" else _REFERENCES if operation == "r2v" else ()
    )
    capabilities = CapabilitySet(
        f"h3.{operation}",
        (
            *_ARTIFACTS,
            _ADAPTERS,
            _PROMPT,
            _WIDTH,
            _HEIGHT,
            _DURATION,
            _FPS,
            _SEED,
            *((_REFERENCE_SIZE,) if operation == "r2v" else ()),
            *media,
        ),
        _validate,
    )
    return ProductPolicy(
        f"h3.{operation}.v1",
        capabilities,
        (
            exposed(_PROMPT),
            exposed(_WIDTH, default=864),
            exposed(_HEIGHT, default=480),
            exposed(_DURATION, default=124 / FRAME_RATE),
            fixed(_FPS, FRAME_RATE),
            exposed(_SEED, default=0),
            *((fixed(_REFERENCE_SIZE, "match"),) if operation == "r2v" else ()),
            *(
                exposed(item, default=None) if item.optional else exposed(item)
                for item in media
            ),
        ),
    )


POLICIES = {operation: _policy(operation) for operation in ("t2v", "i2v", "r2v")}


def h3_recipe(operation, *, adapters=(), **paths):
    """Bind one canonical operation's fixed artifact selection."""
    return POLICIES[operation].bind(
        {**{key: Artifact(value) for key, value in paths.items()}, "adapters": adapters}
    )


def resolve_h3_identity(definition):
    """Resolve fixed artifact state independently of per-request media."""
    fields = {field.capability.key: field for field in definition.fields}
    for key in ARTIFACT_KEYS:
        if fields[key].exposed:
            raise ValueError(f"H3 {key} artifact must be fixed by the recipe")
    if fields["adapters"].exposed:
        raise ValueError("H3 adapters must be fixed by the recipe")
    return H3Identity.from_paths(
        **{key: fields[key].value.path for key in ARTIFACT_KEYS},
        adapters=tuple(
            (adapter.artifact.path, adapter.strength)
            for adapter in fields["adapters"].value
        ),
    )


def resolve_h3_request(definition, overrides):
    """Preserve explicit canvas dimensions and the reference's ordered media roles."""
    values = definition.resolve(overrides)
    request = {key: values[key] for key in ("prompt", "width", "height", "fps", "seed")}
    frames = max(5, round(values["duration_seconds"] * FRAME_RATE))
    request["frame_count"] = frames + (5 - frames % 17) % 17
    if "start_image" in values:
        request["image_path"] = values["start_image"]
    if "reference_image_size" in values:
        request["reference_image_size"] = values["reference_image_size"]
        for kind in ("image", "audio"):
            request[f"reference_{kind}_paths"] = tuple(
                values[key] for key in REFERENCE_KEYS[kind] if values[key] is not None
            )
        pairs = [
            (values[video], values[audio])
            for video, audio in zip(
                REFERENCE_KEYS["video"], REFERENCE_KEYS["video_audio"]
            )
            if values[video] is not None
        ]
        request["reference_video_paths"] = tuple(video for video, _ in pairs)
        request["reference_video_audio_paths"] = tuple(audio for _, audio in pairs)
    return request
