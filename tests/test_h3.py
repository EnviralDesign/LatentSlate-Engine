"""Portable H3 request invariants; no model files or GPU required."""

import pytest

from latentslate_engine.h3.contracts import (
    H3Identity,
    validate_canvas,
    validate_request,
)


@pytest.mark.parametrize("width,height", [(864, 480), (480, 864), (768, 768)])
def test_explicit_aligned_canvas_is_accepted(width, height):
    validate_canvas(width, height)


@pytest.mark.parametrize("width,height", [(865, 480), (864, 481), (0, 480), (-32, 480)])
def test_invalid_canvas_is_rejected_instead_of_resized(width, height):
    with pytest.raises(ValueError, match="H3 width and height"):
        validate_canvas(width, height)


@pytest.mark.parametrize("width,height", [(864.0, 480), (864, "480"), (True, 480)])
def test_dimensions_require_explicit_integers(width, height):
    with pytest.raises(TypeError, match="integers"):
        validate_canvas(width, height)


@pytest.mark.parametrize("frames", [5, 22, 124, 243])
def test_explicit_frame_grid_is_accepted(frames):
    validate_request(864, 480, frames, 42)


@pytest.mark.parametrize("frames", [1, 120, 125, 124.0, True])
def test_invalid_frame_count_is_rejected_instead_of_rounded(frames):
    with pytest.raises(ValueError, match="17n\\+5"):
        validate_request(864, 480, frames, 42)


@pytest.mark.parametrize("fps", [24.0, 23.976, 30, True])
def test_only_supported_integer_fps_is_accepted(fps):
    with pytest.raises(ValueError, match="24 integer FPS"):
        validate_request(864, 480, 124, 42, fps)


@pytest.mark.native
def test_runtime_rejects_bad_canvas_before_opening_artifacts():
    from latentslate_engine.h3.runtime import H3Runtime

    runtime = H3Runtime(
        H3Identity(
            diffusion="missing-diffusion",
            text_encoder="missing-text",
            video_vae="missing-video-vae",
            audio_vae="missing-audio-vae",
            tokenizer="missing-tokenizer",
        )
    )
    with pytest.raises(ValueError, match="multiples of 32"):
        runtime.generate(
            "Synthetic prompt", 865, 480, 124, reference_video_paths=["missing-video"]
        )


@pytest.mark.native
@pytest.mark.parametrize(
    "inputs,error",
    [
        ({"reference_video_paths": ["missing"] * 4}, "at most three"),
        ({"reference_audio_paths": ["missing"] * 4}, "at most three"),
        (
            {
                "reference_video_paths": ["missing"],
                "reference_video_audio_paths": [None, None],
            },
            "pair by index",
        ),
    ],
)
def test_reference_cardinality_rejected_before_opening_inputs(inputs, error):
    from latentslate_engine.h3.runtime import H3Runtime

    runtime = H3Runtime(
        H3Identity("missing", "missing", "missing", "missing", "missing")
    )
    with pytest.raises(ValueError, match=error):
        runtime.generate("Synthetic prompt", 64, 64, 22, **inputs)


@pytest.mark.native
def test_runtime_rejects_mismatched_image_before_opening_artifacts(tmp_path):
    from PIL import Image

    from latentslate_engine.h3.runtime import H3Runtime

    path = tmp_path / "synthetic.png"
    Image.new("RGB", (96, 64)).save(path)
    runtime = H3Runtime(
        H3Identity("missing", "missing", "missing", "missing", "missing")
    )
    with pytest.raises(ValueError, match="must match the 64x64 output canvas"):
        runtime.generate("Synthetic prompt", 64, 64, 22, image_path=path)


def test_identity_changes_on_artifact_replacement_not_argument_order(tmp_path):
    artifact = tmp_path / "synthetic.safetensors"
    artifact.write_bytes(b"first")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    for name in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        (tokenizer / name).write_text("synthetic")
    paths = dict.fromkeys(
        ("diffusion", "text_encoder", "video_vae", "audio_vae"), artifact
    )
    paths["tokenizer"] = tokenizer
    first = H3Identity.from_paths(**paths)
    assert first == H3Identity.from_paths(**dict(reversed(paths.items())))
    artifact.write_bytes(b"replacement")
    assert first != H3Identity.from_paths(**paths)
    adapter = tmp_path / "synthetic-adapter.safetensors"
    adapter.write_bytes(b"synthetic factors")
    adapted = H3Identity.from_paths(**paths, adapters=[(adapter, 1.0)])
    assert adapted != H3Identity.from_paths(**paths)
    assert adapted != H3Identity.from_paths(**paths, adapters=[(adapter, 0.5)])
    adapter.write_bytes(b"replacement factors")
    assert adapted != H3Identity.from_paths(**paths, adapters=[(adapter, 1.0)])


@pytest.mark.parametrize("operation", ["t2v", "i2v", "r2v"])
def test_recipe_preserves_explicit_canvas_and_publishes_alignment(tmp_path, operation):
    from latentslate_engine.h3.contracts import H3ModelPaths
    from latentslate_engine.h3.recipes import h3_recipe, resolve_h3_request

    recipe = h3_recipe(
        operation, **H3ModelPaths.from_root(tmp_path).bindings(operation)
    )
    inputs = {"prompt": "Synthetic prompt", "width": 480, "height": 864}
    if operation == "i2v":
        inputs["start_image"] = "synthetic.png"
    request = resolve_h3_request(recipe, inputs)
    assert (
        request["width"],
        request["height"],
        request["frame_count"],
        request["fps"],
    ) == (480, 864, 124, 24)
    dimensions = [
        item for item in recipe.surface() if item["key"] in {"width", "height"}
    ]
    assert all(item["constraints"]["step"] == 32 for item in dimensions)
    with pytest.raises(ValueError, match="increments of 32"):
        resolve_h3_request(recipe, {**inputs, "width": 481})


def test_recipe_preserves_video_soundtrack_pairing_when_slots_are_empty(tmp_path):
    from latentslate_engine.h3.contracts import H3ModelPaths
    from latentslate_engine.h3.recipes import h3_recipe, resolve_h3_request

    recipe = h3_recipe("r2v", **H3ModelPaths.from_root(tmp_path).bindings("r2v"))
    inputs = {
        "prompt": "Synthetic prompt",
        "reference_video_2": "first.mp4",
        "reference_video_3": "second.mp4",
        "reference_video_audio_3": "paired.wav",
        "reference_audio_1": "standalone.wav",
    }
    request = resolve_h3_request(recipe, inputs)
    assert request["reference_video_paths"] == ("first.mp4", "second.mp4")
    assert request["reference_video_audio_paths"] == (None, "paired.wav")
    assert request["reference_audio_paths"] == ("standalone.wav",)
    with pytest.raises(ValueError, match="requires its paired"):
        resolve_h3_request(
            recipe, {**inputs, "reference_video_audio_1": "unpaired.wav"}
        )


def test_reference_catalog_preserves_pairing_and_help_in_authored_recipes(tmp_path):
    from uuid import uuid4

    from latentslate_engine.authoring import document_from_recipe
    from latentslate_engine.catalog import H3_IDS, TOOLS_BY_ID, user_request_schema
    from latentslate_engine.h3.contracts import H3ModelPaths
    from latentslate_engine.h3.recipes import h3_recipe, resolve_h3_request

    recipe = h3_recipe("r2v", **H3ModelPaths.from_root(tmp_path).bindings("r2v"))
    document = document_from_recipe(recipe, name="Synthetic", recipe_id=str(uuid4()))
    for schema in (TOOLS_BY_ID[H3_IDS["r2v"]], user_request_schema(document)):
        assert schema["workflow_kind"] == "reference_to_video"
        fields = {item["key"]: item for item in schema["inputs"]}
        for index in (1, 2, 3):
            soundtrack = fields[f"reference_video_audio_{index}"]
            assert soundtrack["paired_video_input"] == f"reference_video_{index}"
            assert soundtrack["type"] == "audio"
            assert soundtrack.get("nullable")
            assert soundtrack["prompt_reference_token"] == "<Audio {index}>"
        assert (
            fields["reference_image_3"]["prompt_reference_token"] == "<Picture {index}>"
        )
        assert (
            fields["reference_video_2"]["prompt_reference_token"] == "<Video {index}>"
        )
        audio_order = [
            item["key"]
            for item in schema["inputs"]
            if item.get("prompt_reference_token") == "<Audio {index}>"
        ]
        assert audio_order == [
            *(f"reference_video_audio_{i}" for i in (1, 2, 3)),
            *(f"reference_audio_{i}" for i in (1, 2, 3)),
        ]
        assert "<Picture N>" in fields["prompt"]["description"]
        assert "soundtracks first" in fields["prompt"]["description"]
        assert all(
            "image_dimensions" not in item
            for item in fields.values()
            if item["type"] == "image"
        )
    prompt = "Use <Picture 1>, <Video 1> and <Audio 2>."
    inputs = {
        "prompt": prompt,
        "reference_image_3": "synthetic.png",
        "reference_video_2": "synthetic.mp4",
        "reference_video_audio_2": "synthetic.mp4",
        "reference_audio_3": "synthetic.wav",
    }
    request = resolve_h3_request(recipe, inputs)
    assert request["prompt"] == prompt
    assert request["reference_image_paths"] == ("synthetic.png",)
    assert request["reference_video_paths"] == ("synthetic.mp4",)
    assert request["reference_video_audio_paths"] == ("synthetic.mp4",)
    assert request["reference_audio_paths"] == ("synthetic.wav",)
    assert (
        resolve_h3_request(recipe, {**inputs, "reference_image_3": None})["prompt"]
        == prompt
    )


@pytest.mark.parametrize("operation,steps", [("t2v", 8), ("i2v", 8), ("r2v", 4)])
def test_turbo_couples_adapter_identity_and_operation_schedule(
    tmp_path, operation, steps
):
    from latentslate_engine.h3.contracts import H3ModelPaths
    from latentslate_engine.h3.recipes import (
        h3_recipe,
        resolve_h3_identity,
        resolve_h3_request,
    )
    from latentslate_engine.recipe import Adapter, Artifact

    paths = H3ModelPaths.from_root(tmp_path).bindings(operation)
    for key, path in paths.items():
        files = (
            [
                path / name
                for name in ("vocab.json", "merges.txt", "tokenizer_config.json")
            ]
            if key == "tokenizer"
            else [path]
        )
        for file in files:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"synthetic")
    adapter = tmp_path / "synthetic-adapter.safetensors"
    adapter.write_bytes(b"synthetic")
    recipe = h3_recipe(operation, adapters=(Adapter(Artifact(adapter), 0.5),), **paths)
    inputs = {"prompt": "Synthetic prompt"}
    if operation == "i2v":
        inputs["start_image"] = "synthetic.png"
    base = resolve_h3_identity(recipe, inputs)
    assert resolve_h3_request(recipe, inputs)["steps"] == 20
    turbo = {**inputs, "turbo": True}
    active = resolve_h3_identity(recipe, turbo)
    assert active.adapters == (
        (str(paths["turbo_adapter"].resolve()), 1.0),
        *base.adapters,
    )
    assert active != base
    assert resolve_h3_request(recipe, turbo)["steps"] == steps
    assert resolve_h3_identity(recipe, {**inputs, "turbo": False}) == base


@pytest.mark.native
def test_encoded_video_preserves_canvas_timing_and_stereo_audio(tmp_path):
    import av
    import torch

    from latentslate_engine.h3.runtime import H3Output

    output = H3Output(torch.full((1, 5, 32, 64, 3), 0.5), torch.zeros(1, 2, 6400))
    path = tmp_path / "synthetic.mp4"
    output.save_mp4(path)
    with av.open(str(path)) as media:
        video, audio = media.streams.video[0], media.streams.audio[0]
        assert (video.width, video.height, video.average_rate) == (64, 32, 24)
        assert video.codec_context.color_trc == 13
        assert (audio.sample_rate, audio.channels) == (32000, 2)
        assert float(audio.duration * audio.time_base) == pytest.approx(0.2)
        assert len(list(media.decode(video))) == 5


@pytest.mark.native
def test_audio_decode_limits_loudness_without_amplifying_quiet_audio():
    import torch

    from latentslate_engine.h3.runtime import H3Runtime

    waveform = torch.tensor([[[-0.5, 0.5] * 8] * 2])

    class Decoder:
        def to(self, device):
            return self

        def decode(self, latent):
            return waveform.clone()

        def cpu(self):
            return self

    runtime = H3Runtime(
        H3Identity("missing", "missing", "missing", "missing", "missing")
    )
    runtime.device = torch.device("cpu")
    runtime.audio_vae = Decoder()
    decoded = runtime._decode_audio(torch.zeros(1))
    assert decoded.std().item() == pytest.approx(0.2)
    assert torch.equal(torch.sign(decoded), torch.sign(waveform))
    waveform *= 0.1
    assert torch.equal(runtime._decode_audio(torch.zeros(1)), waveform)
