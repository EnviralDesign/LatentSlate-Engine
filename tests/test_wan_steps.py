"""Exercise real Wan sampling loops with lightweight CPU model boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from latentslate_engine.wan2214b import flf, i2v, pipeline, recipes

pytestmark = pytest.mark.native

OPERATIONS = (
    (pipeline, pipeline.WanSession, pipeline.WanRecipe, ()),
    (i2v, i2v.WanI2VSession, i2v.WanI2VRecipe, ("first.png",)),
    (flf, flf.WanFLFSession, flf.WanFLFRecipe, ("first.png", "last.png")),
)


@pytest.mark.parametrize("module,session_type,recipe_type,sources", OPERATIONS)
@pytest.mark.parametrize(
    "steps,shift",
    [(steps, 5.000000000000001) for steps in (3, 4, 6, 8)] + [(4, 4.0), (4, 6.0)],
)
def test_actual_sampling_loops(
    monkeypatch, tmp_path, module, session_type, recipe_type, sources, steps, shift
):
    # Bypass CUDA construction, not generate(): this first probes native mechanics
    # independently of the public recipe validator and trained model quality.
    artifact = tmp_path / "identity-only.bin"
    artifact.write_bytes(b"identity")
    recipe = recipe_type(
        **dict.fromkeys(
            (
                "high_checkpoint",
                "low_checkpoint",
                "high_lora",
                "low_lora",
                "text_encoder",
                "vae",
            ),
            str(artifact),
        ),
        steps=steps,
        shift=shift,
        width=480,
        height=480,
        frame_count=17,
    )
    session = object.__new__(session_type)
    session.recipe = recipe
    session._identity = recipe.identity
    session._alive = True
    session.device = torch.device("cpu")
    calls = []
    sampler_inputs = []
    transitions = []

    def weights(phase):
        return SimpleNamespace(
            phase=phase,
            activate=lambda *a, **k: transitions.append((phase, "activate")),
            deactivate=lambda: transitions.append((phase, "deactivate")),
        )

    def transformer(weights):
        def forward(x, timestep, context):
            calls.append((weights.phase, timestep.item(), x.shape[1]))
            sampler_inputs.append(x[0, 0, 0, 0, 0].item())
            return torch.ones_like(x[:, :16])

        return forward

    noise = torch.zeros(1, 16, 5, 2, 2)
    session.high_weights = weights("high")
    session.low_weights = weights("low")
    session._ensure_conditioning = lambda *a: (torch.zeros(1, 1, 4096), None)
    session._ensure_image_conditioning = lambda *a: None
    session._ensure_flf_conditioning = lambda *a: None
    decoded_latents = []

    def decode(x):
        decoded_latents.append(x.clone())
        return torch.zeros(1, 3, 17, 2, 2)

    session._vae = SimpleNamespace(decode=decode)
    monkeypatch.setattr(module, "cpu_noise", lambda *a: noise.clone())
    monkeypatch.setattr(module, "WanT2VTransformer", transformer)
    if sources:
        monkeypatch.setattr(
            module, "_model_conditioning", lambda *a: torch.zeros(1, 20, 5, 2, 2)
        )
    monkeypatch.setattr(
        module,
        "save_half_open_video",
        lambda images, *a: pipeline.half_open_delivery(images),
    )
    progress = []
    result = session.generate(*sources, tmp_path / "out.mp4", progress=progress.append)
    sigmas = pipeline.canonical_sigmas(recipe.shift, steps)
    assert len(sigmas) == steps + 1
    assert sigmas[0] == 1 and sigmas[-1] == 0
    assert torch.all(sigmas[:-1] > sigmas[1:])
    if shift != 5.000000000000001:
        assert not torch.equal(
            sigmas, pipeline.canonical_sigmas(5.000000000000001, steps)
        )
    assert [phase for phase, _, _ in calls] == ["high"] * 2 + ["low"] * (steps - 2)
    assert [t for _, t, _ in calls] == pytest.approx((sigmas[:-1] * 1000).tolist())
    assert sampler_inputs == pytest.approx((sigmas[:-1] - 1).to(torch.float16).tolist())
    assert [channels for _, _, channels in calls] == [36 if sources else 16] * steps
    assert transitions == [
        (p, action) for p in ("high", "low") for action in ("activate", "deactivate")
    ]
    expected = pipeline.process_latent_out(noise - 1).to(torch.bfloat16)
    torch.testing.assert_close(decoded_latents[0], expected)
    for label in ("High-noise sampling", "Low-noise sampling"):
        stages = [
            event["stage"] for event in progress if event["stage"]["label"] == label
        ]
        assert stages[0]["progress"] == 0 and stages[-1]["progress"] == 1
    assert (result.frame_count, result.fps, result.duration) == (16, 16, 1)


@pytest.mark.parametrize(
    "operation,session_type",
    (
        ("t2v", pipeline.WanSession),
        ("i2v", i2v.WanI2VSession),
        ("flf", flf.WanFLFSession),
    ),
)
@pytest.mark.parametrize(
    "key,baseline_value,valid,invalid",
    [
        ("steps", 4, (3, 4, 6, 8), (2, 9, True, 4.5)),
        (
            "shift",
            5.000000000000001,
            (4.0, 4.5, 5.0, 5.000000000000001, 5.5, 6.0),
            (3.5, 6.5, 4.25, True, "5", float("nan"), float("inf"), -float("inf")),
        ),
    ],
)
def test_native_resolution_and_sampling_changes_replace_session(
    monkeypatch, tmp_path, operation, session_type, key, baseline_value, valid, invalid
):
    artifact = tmp_path / "identity-only.bin"
    artifact.write_bytes(b"identity")
    definition = getattr(recipes, f"wan2214b_{operation}_recipe")(
        **dict.fromkeys(
            ("high_checkpoint", "low_checkpoint", "text_encoder", "vae"), artifact
        ),
        high_adapters=(),
        low_adapters=(),
        negative_prompt="",
    )
    inputs = {"prompt": "test"}
    if operation != "t2v":
        inputs["start_image"] = tmp_path / "first.png"
    if operation == "flf":
        inputs["end_image"] = tmp_path / "last.png"
    resolver = getattr(recipes, f"resolve_wan2214b_{operation}")
    baseline, _ = resolver(definition, inputs)
    assert getattr(baseline, key) == baseline_value
    resolved = {}
    for value in valid:
        changed = replace(
            definition,
            fields=tuple(
                replace(f, value=value) if f.capability.key == key else f
                for f in definition.fields
            ),
        )
        native, _ = resolver(changed, inputs)
        assert getattr(native, key) == value
        assert (native.identity == baseline.identity) == (value == baseline_value)
        resolved[value] = native
    for value in invalid:
        with pytest.raises((ValueError, TypeError)):
            replace(baseline, **{key: value}).validate()
        with pytest.raises((ValueError, TypeError)):
            definition.capabilities[key].normalize(value)

    def init(session, recipe, device):
        session.recipe = recipe
        session._identity = recipe.identity
        session._alive = True
        session.device = device
        for attr in (
            "_conditioning",
            "_conditioning_key",
            "_image_conditioning",
            "_flf_conditioning",
            "_vae",
            "high_weights",
            "low_weights",
            "text_weights",
        ):
            setattr(session, attr, object())

    monkeypatch.setattr(session_type, "__init__", init)
    session = session_type(baseline, torch.device("cpu"))
    assert session.replaced(resolved[baseline_value]) is session
    changed = session.replaced(resolved[6])
    assert changed is not session and changed.identity == resolved[6].identity
    assert not session._alive
    for attr in (
        "_conditioning",
        "_conditioning_key",
        "_vae",
        "high_weights",
        "low_weights",
        "text_weights",
    ):
        assert getattr(session, attr) is None
    if operation != "t2v":
        assert (
            getattr(
                session,
                "_image_conditioning" if operation == "i2v" else "_flf_conditioning",
            )
            is None
        )
