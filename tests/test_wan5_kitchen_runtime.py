from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from latentslate_engine.runtime.wan5_kitchen import (
    WAN5_MAX_PIXELS,
    WAN5_UNIPC_TERMINAL_SIGMA,
    WAN5_VAE_COMPUT_DTYPE,
    Wan5KitchenGeneration,
    Wan5SimpleUniPCScheduler,
    _build_pipeline,
    _encode_mp4,
    _endpoint_identity,
    _load_image,
    _matches_execution_device,
    _require_finite_frames,
    _require_finite_tensor,
    _validate_mp4,
    _Wan5GuideStageGuard,
    _Wan5ResidencyTransition,
    _Wan5TransformerOutputGuard,
    validate_wan5_kitchen_generation,
)


def test_scheduler_matches_pinned_simple_grid_and_unipc_bh1() -> None:
    scheduler = Wan5SimpleUniPCScheduler.build()
    scheduler.set_timesteps(30, device="cpu")

    # Independent transcription of the pinned source's 1000-entry shifted
    # ModelSamplingDiscreteFlow grid followed by simple_scheduler(31), then
    # the pinned UniPC penultimate-sigma removal.
    training = np.arange(1, 1001, dtype=np.float64) / 1000
    shifted = 8.0 * training / (1.0 + 7.0 * training)
    source_simple_31 = [shifted[-(1 + int(step * 1000 / 31))] for step in range(31)]
    source_simple_31.append(0.0)
    source_unipc = source_simple_31[:-2] + source_simple_31[-1:]
    expected = np.asarray(source_unipc[:-1], dtype=np.float64)
    assert scheduler.config.solver_type == "bh1"
    assert scheduler.config.solver_order == 3
    assert scheduler.config.lower_order_final is True
    assert scheduler.config.prediction_type == "flow_prediction"
    assert scheduler.config.flow_shift == 8.0
    assert scheduler.timesteps.shape == (30,)
    assert scheduler.sigmas.shape == (31,)
    assert scheduler.sigmas[-1].item() == pytest.approx(WAN5_UNIPC_TERMINAL_SIGMA)
    assert np.allclose(scheduler.sigmas[:-1].numpy(), expected, rtol=0, atol=2e-7)
    assert scheduler.timesteps.dtype == torch.float32
    assert scheduler.timesteps[0].item() == 1000.0


@pytest.mark.parametrize("velocity", [0.0, 0.25, -0.5])
def test_scheduler_pinned_bh1_terminal_is_finite_for_normalized_flow_fixture(
    velocity: float,
) -> None:
    """The former flow-space bh1 update deterministically became NaN at step 30."""

    scheduler = Wan5SimpleUniPCScheduler.build()
    scheduler.set_timesteps(30, device="cpu")
    sample = torch.full((1, 2, 2, 2, 2), 0.125, dtype=torch.float32)
    for timestep in scheduler.timesteps:
        sample = scheduler.step(
            torch.full_like(sample, velocity), timestep, sample, return_dict=False
        )[0]

    assert torch.isfinite(sample).all()
    assert torch.allclose(
        sample,
        torch.full_like(sample, 0.125 - velocity * (1 - WAN5_UNIPC_TERMINAL_SIGMA)),
        atol=2e-5,
        rtol=0,
    )


def test_scheduler_nonlinear_golden_locks_all_corrector_updates() -> None:
    scheduler = Wan5SimpleUniPCScheduler.build()
    scheduler.set_timesteps(30, device="cpu")
    sample = torch.tensor([[[[[0.125]]]]], dtype=torch.float32)
    history = []
    for timestep in scheduler.timesteps:
        velocity = torch.tanh(sample * 0.7 + timestep.float() / 1000 * 0.4)
        velocity += 0.05 * torch.sin(sample * 3.0)
        sample = scheduler.step(velocity, timestep, sample, return_dict=False)[0]
        history.append(sample.item())

    assert history == pytest.approx(
        [
            0.1230635867,
            0.1210355684,
            0.1188910678,
            0.1165537164,
            0.1141545996,
            0.1116135046,
            0.1089175418,
            0.1059601307,
            0.1029039100,
            0.0996439829,
            0.0961597636,
            0.0923073068,
            0.0882929415,
            0.0839738399,
            0.0793157294,
            0.0741153806,
            0.0686417669,
            0.0626918301,
            0.0562065914,
            0.0488871485,
            0.0410997309,
            0.0325490087,
            0.0231468268,
            0.0124675175,
            0.0010887893,
            -0.0113130771,
            -0.0246278811,
            -0.0389183313,
            -0.0501663797,
            -0.0857830346,
        ],
        abs=2e-7,
        rel=0,
    )


def test_transformer_guard_reports_exact_pass_and_promotes_before_cfg() -> None:
    class _Handle:
        def remove(self):
            pass

    class _Transformer:
        def register_forward_hook(self, hook):
            self.hook = hook
            return _Handle()

    transformer = _Transformer()
    guard = _Wan5TransformerOutputGuard(transformer)
    guard.attach()
    cond = transformer.hook(None, (), (torch.tensor([40_000], dtype=torch.float16),))[0]
    uncond = transformer.hook(None, (), (torch.tensor([-40_000], dtype=torch.float16),))[0]

    assert cond.dtype == torch.float32
    assert uncond.dtype == torch.float32
    assert torch.isfinite(uncond + 5 * (cond - uncond)).all()
    with pytest.raises(
        ValueError,
        match="transformer_noise_prediction contains non-finite.*step 2.*conditional",
    ):
        transformer.hook(None, (), (torch.tensor([float("nan")], dtype=torch.float16),))


def test_finite_boundaries_reject_nan_without_clamping() -> None:
    with pytest.raises(ValueError, match="text_conditioning contains non-finite"):
        _require_finite_tensor(torch.tensor([float("nan")]), "text_conditioning")
    with pytest.raises(ValueError, match="decoded frames contain non-finite"):
        _require_finite_frames(np.array([[[[float("inf"), 0.0, 0.0]]]], dtype=np.float32))


def test_decode_residency_uses_pinned_float32_vae_compute(monkeypatch: pytest.MonkeyPatch) -> None:
    import latentslate_engine.runtime.wan5_kitchen as kitchen

    calls: list[tuple[object, dict[str, object]]] = []

    class _Module:
        def to(self, *args, **kwargs):
            calls.append((args, kwargs))
            return self

    monkeypatch.setattr(kitchen, "_empty_cuda", lambda: None)
    transition = _Wan5ResidencyTransition(
        _Module(), _Module(), operation="wan5_t2v", device=torch.device("cuda")
    )
    transition.prepare_initial_residency()
    transition.prepare_decode()

    assert calls == [
        ((), {"device": torch.device("cuda"), "dtype": torch.float16}),
        (("cpu",), {}),
        ((), {"device": torch.device("cuda"), "dtype": WAN5_VAE_COMPUT_DTYPE}),
    ]


def test_i2v_guide_stage_never_overlaps_transformer_and_fp32_vae_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import latentslate_engine.runtime.wan5_kitchen as kitchen

    calls: list[str] = []
    snapshots: list[tuple[str, str]] = []

    class _Handle:
        def remove(self):
            calls.append("remove_hook")

    class _Module:
        def __init__(self, name: str):
            self.name = name
            self.device = "cpu"

        def to(self, *args, **kwargs):
            target = args[0] if args else kwargs.get("device", self.device)
            self.device = str(target)
            calls.append(f"{self.name}:{self.device}")
            snapshots.append((transformer.device, vae.device))
            return self

        def register_forward_pre_hook(self, hook, prepend=False):
            assert prepend is True
            self.hook = hook
            return _Handle()

    transformer, vae = _Module("transformer"), _Module("vae")
    monkeypatch.setattr(kitchen, "_empty_cuda", lambda: calls.append("empty_cuda"))
    transition = _Wan5ResidencyTransition(
        transformer, vae, operation="wan5_i2v", device=torch.device("cuda")
    )

    transition.prepare_initial_residency()
    transition.attach()
    transformer.hook(transformer, ())

    assert calls == [
        "transformer:cpu",
        "vae:cuda",
        "vae:cpu",
        "empty_cuda",
        "transformer:cuda",
        "remove_hook",
    ]
    assert ("cuda", "cuda") not in snapshots
    assert transition.provenance()["guide_vae_offloaded_before_transformer_onload"] is True


def test_i2v_pipeline_binds_cuda_execution_while_transformer_is_staged_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import WanImageToVideoPipeline

    required, _optional = WanImageToVideoPipeline._get_signature_keys(WanImageToVideoPipeline)
    assert required.index("transformer") < required.index("vae")

    class _Pipeline:
        def __init__(self, **components):
            self.components = components

    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.WanImageToVideoPipeline = _Pipeline
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    pipe = _build_pipeline(
        "wan5_i2v",
        object(),
        object(),
        object(),
        execution_device=torch.device("cuda"),
    )

    assert pipe._execution_device == torch.device("cuda")


def test_i2v_guide_guard_publishes_exact_stage_and_checks_encode_residency() -> None:
    stages: list[str | None] = []

    class _Processor:
        def preprocess(self, value):
            return value

    class _Vae:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.ones(1))

        def parameters(self):
            return iter((self.weight,))

        def encode(self, value):
            return value

    pipe = SimpleNamespace(video_processor=_Processor())
    vae = _Vae()
    guard = _Wan5GuideStageGuard(
        pipe,
        vae,
        device=torch.device("cpu"),
        progress=lambda _progress, message: stages.append(message),
    )
    guard.attach()

    value = pipe.video_processor.preprocess(torch.ones(1, dtype=WAN5_VAE_COMPUT_DTYPE))
    assert vae.encode(value) is value
    assert stages == [
        "Preprocessing Wan 2.2 guide image",
        "Encoding Wan 2.2 guide image",
        "Prepared Wan 2.2 guide latent",
    ]
    with pytest.raises(RuntimeError, match="encode residency"):
        vae.encode(torch.ones(1, dtype=torch.float16))
    guard.detach()


def test_indexless_cuda_matches_only_current_concrete_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    assert _matches_execution_device(torch.device("cuda"), torch.device("cuda"))
    assert _matches_execution_device(torch.device("cuda:0"), torch.device("cuda"))
    assert not _matches_execution_device(torch.device("cuda:1"), torch.device("cuda"))
    assert not _matches_execution_device(torch.device("cpu"), torch.device("cuda"))
    assert _matches_execution_device(torch.device("cuda:1"), torch.device("cuda:1"))
    assert not _matches_execution_device(torch.device("cuda:0"), torch.device("cuda:1"))


@pytest.mark.parametrize("operation", ["wan5_t2v", "wan5_i2v"])
def test_generation_contract_accepts_exact_operations(tmp_path: Path, operation: str) -> None:
    start = tmp_path / "start.png"
    start.write_bytes(b"image")
    generation = Wan5KitchenGeneration(
        operation=operation,
        prompt="A slow camera move through morning fog",
        width=1280,
        height=704,
        num_frames=121,
        seed=7,
        output_path=tmp_path / "output.mp4",
        staging_output_path=tmp_path / ".output.worker-staging.mp4",
        start_image_path=start if operation == "wan5_i2v" else None,
        start_image_identity=_endpoint_identity(start) if operation == "wan5_i2v" else None,
    )

    validate_wan5_kitchen_generation(generation, operation)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("width", 1279, "/32"),
        ("height", 736, "pixel area"),
        ("num_frames", 120, r"4k\+1"),
        ("num_frames", 21, "25..121"),
        ("seed", -1, "nonnegative"),
    ],
)
def test_generation_contract_rejects_invalid_work_before_loading(
    tmp_path: Path, field: str, value: int, message: str
) -> None:
    values = {
        "operation": "wan5_t2v",
        "prompt": "prompt",
        "width": 1280,
        "height": 704,
        "num_frames": 121,
        "seed": 0,
        "output_path": tmp_path / "output.mp4",
        "staging_output_path": tmp_path / ".output.worker-staging.mp4",
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        validate_wan5_kitchen_generation(Wan5KitchenGeneration(**values), "wan5_t2v")


def test_pixel_bound_is_the_pinned_default_area() -> None:
    assert WAN5_MAX_PIXELS == 1280 * 704


def test_output_validation_rejects_non_mp4_container(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output.mp4"
    output.write_bytes(b"container")

    class _Container:
        format = SimpleNamespace(name="matroska,webm")
        streams = SimpleNamespace(
            video=[
                    SimpleNamespace(
                        average_rate=24,
                        time_base=Fraction(1, 12288),
                        duration=12800,
                        codec_context=SimpleNamespace(name="h264", width=1280, height=704),
                )
            ],
            audio=[],
        )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def decode(self, _stream):
            return [object()] * 25

    fake_av = ModuleType("av")
    fake_av.open = lambda _path: _Container()
    monkeypatch.setitem(sys.modules, "av", fake_av)
    generation = Wan5KitchenGeneration(
        operation="wan5_t2v",
        prompt="prompt",
        width=1280,
        height=704,
        num_frames=25,
        seed=0,
        output_path=output,
        staging_output_path=tmp_path / ".output.worker-staging.mp4",
    )

    with pytest.raises(RuntimeError, match="stream facts"):
        _validate_mp4(output, generation)


def test_output_encoding_uses_engine_pyav_not_diffusers_imageio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wan output must not depend on Diffusers' optional imageio exporter."""

    encoded: list[object] = []

    class _Stream:
        width = 0
        height = 0
        pix_fmt = ""

        def encode(self, frame=None):
            encoded.append(frame)
            return ()

    class _Container:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def add_stream(self, codec, rate):
            assert (codec, rate) == ("libx264", 24)
            self.stream = _Stream()
            return self.stream

        def mux(self, packet):
            raise AssertionError(f"unexpected packet: {packet!r}")

    class _Frame:
        def __init__(self, pixels, format):
            self.pixels = pixels
            self.format = format
            self.pts = None

    fake_av = ModuleType("av")
    fake_av.open = lambda _path, _mode: _Container()
    fake_av.VideoFrame = SimpleNamespace(from_ndarray=lambda pixels, format: _Frame(pixels, format))
    monkeypatch.setitem(sys.modules, "av", fake_av)
    # The current runtime intentionally has no imageio package.  A module that
    # fails if touched makes this regression independent of that environment.
    monkeypatch.setitem(sys.modules, "imageio", None)

    _encode_mp4(
        np.zeros((2, 32, 32, 3), dtype=np.float32),
        tmp_path / "output.mp4",
        fps=24,
        check_cancelled=lambda: None,
    )

    assert len(encoded) == 3  # two frames plus the final encoder flush


def test_real_pyav_output_reports_observed_stream_timing(tmp_path: Path) -> None:
    output = tmp_path / "output.mp4"
    generation = Wan5KitchenGeneration(
        operation="wan5_t2v",
        prompt="prompt",
        width=32,
        height=32,
        num_frames=25,
        seed=0,
        output_path=output,
        staging_output_path=tmp_path / ".output.worker-staging.mp4",
    )
    _encode_mp4(
        np.zeros((25, 32, 32, 3), dtype=np.float32),
        output,
        fps=24,
        check_cancelled=lambda: None,
    )

    observed = _validate_mp4(output, generation)

    time_base = observed["time_base"]
    assert isinstance(time_base, dict)
    seconds = Fraction(time_base["numerator"], time_base["denominator"]) * observed["duration"]
    assert observed["frame_count"] == 25
    assert observed["fps"] == 24.0
    assert observed["duration_seconds"] == pytest.approx(float(seconds))
    assert observed["duration_seconds"] == pytest.approx(25 / 24)


def test_image_identity_is_rechecked_immediately_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "start.png"
    image.write_bytes(b"first")
    identity = _endpoint_identity(image)
    image.write_bytes(b"changed")
    opened = False

    fake_pil = ModuleType("PIL")

    class _Image:
        @staticmethod
        def open(_path):
            nonlocal opened
            opened = True
            raise AssertionError("opened a changed endpoint")

    fake_pil.Image = _Image
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    with pytest.raises(ValueError, match="changed immediately"):
        _load_image(image, identity)
    assert opened is False
