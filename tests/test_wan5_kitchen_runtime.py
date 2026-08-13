from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from latentslate_engine.runtime.wan5_kitchen import (
    WAN5_MAX_PIXELS,
    Wan5KitchenGeneration,
    Wan5SimpleUniPCScheduler,
    _endpoint_identity,
    _load_image,
    _validate_mp4,
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
    expected = [shifted[-(1 + int(step * 1000 / 31))] for step in range(30)]
    expected[0] -= 1e-6  # Diffusers' documented UniPC log-alpha guard.

    assert scheduler.config.solver_type == "bh1"
    assert scheduler.config.solver_order == 3
    assert scheduler.config.lower_order_final is True
    assert scheduler.config.prediction_type == "flow_prediction"
    assert scheduler.config.flow_shift == 8.0
    assert scheduler.timesteps.shape == (30,)
    assert scheduler.sigmas.shape == (31,)
    assert scheduler.sigmas[-1].item() == 0
    assert np.allclose(scheduler.sigmas[:-1].numpy(), expected, rtol=0, atol=2e-7)


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
