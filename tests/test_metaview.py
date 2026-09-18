"""Synthetic coverage for the novel-view request and conditioning lifecycle."""

from dataclasses import replace
from types import SimpleNamespace

import torch
from PIL import Image



def test_camera_orbits_center_and_automatic_radius_uses_source_depth():
    from latentslate_engine.metaview.conditioning import camera_conditioning

    source = {"depth": torch.stack((torch.zeros(4, 4), torch.full((4, 4), 3.0)))}
    for radius in (None, 0, 3):
        value, actual = camera_conditioning(source, 30, -15, radius)
        assert actual == 3
        torch.testing.assert_close(value["viewmats"][1], torch.eye(4))
        center = torch.tensor([0., 0., 3., 1.])
        torch.testing.assert_close(value["viewmats"][0] @ center, center)
    baseline, _ = camera_conditioning(source, 0, 0, None)
    torch.testing.assert_close(baseline["viewmats"], torch.eye(4).expand(2, 4, 4))


def test_source_pose_seed_reuse_and_identity_reset(monkeypatch, tmp_path):
    from latentslate_engine.metaview import runtime as module
    from latentslate_engine.metaview.contracts import MetaViewIdentity

    calls = []

    class VAE:
        def encode(self, pixels):
            calls.append("encode")
            return torch.zeros(1, 16, 1, pixels.shape[1] // 8, pixels.shape[2] // 8)

        def decode(self, latent):
            return torch.zeros(1, 3, 1, latent.shape[-2] * 8, latent.shape[-1] * 8)

    class Text:
        def __init__(self, *args):
            pass

        def encode(self, *args, numbered=True):
            assert not numbered
            calls.append("text")
            return torch.zeros(1, 2, 3584)

        def offload(self):
            pass

        def close(self):
            calls.append("close_text")

    class Weights:
        def __init__(self, *args, **kwargs):
            calls.append("load")

        def close(self):
            calls.append("close")

    def geometry(*args):
        calls.append("geometry")
        return {"depth": torch.ones(2, 4, 4)}

    monkeypatch.setattr(module, "source_geometry", geometry)
    monkeypatch.setattr(module, "load_vae", lambda *args: VAE())
    monkeypatch.setattr(module, "QwenTextEncoder", Text)
    monkeypatch.setattr(module, "MetaViewDiT", lambda **kwargs: torch.nn.Identity())
    monkeypatch.setattr(module, "QwenWeights", Weights)
    monkeypatch.setattr(module, "sample", lambda model, positive, reference, *args: reference)
    artifact = SimpleNamespace(path="synthetic")
    identity = MetaViewIdentity(artifact, artifact, artifact, artifact, artifact, tmp_path, ())
    source = tmp_path / "source.png"
    Image.new("RGB", (256, 256), "red").save(source)
    request = dict(image=source, width=256, height=256, seed=0, yaw=0, pitch=0,
                   output=tmp_path / "out.png")
    runtime = module.MetaViewRuntime("cpu")
    try:
        first = runtime.generate(identity, **request)
        repeat = runtime.generate(identity, **{**request, "seed": 1, "yaw": 30})
        assert not first.models_reused and not first.source_reused
        assert repeat.models_reused and repeat.source_reused
        assert calls.count("geometry") == calls.count("encode") == calls.count("text") == 1
        Image.new("RGB", (256, 256), "blue").save(source)
        changed = runtime.generate(identity, **request)
        assert not changed.source_reused and not changed.models_reused
        assert calls.count("geometry") == 2
        other = replace(identity, diffusion=SimpleNamespace(path="alternate"))
        switched = runtime.generate(other, **request)
        restored = runtime.generate(identity, **request)
        assert not switched.models_reused and not restored.models_reused
        assert calls.count("geometry") == 4
    finally:
        runtime.close()
    assert runtime.identity is None and runtime.source is None
