"""Klein product/native oracles and shadow policy experiment; no service wiring."""

from dataclasses import replace

import pytest
from test_recipe import _file, _klein_paths

from latentslate_engine.klein9b.contracts import Klein9BIdentity
from latentslate_engine.klein9b.recipes import (
    KLEIN9B_T2I_CAPABILITIES,
    KLEIN9B_T2I_POLICY,
    KLEIN9B_TWO_IMAGE_CAPABILITIES,
    KLEIN9B_TWO_IMAGE_EXPLICIT_POLICY,
    klein9b_t2i_recipe,
    klein9b_two_image_explicit_recipe,
    klein9b_two_image_recipe,
    resolve_klein9b_t2i,
    resolve_klein9b_two_image,
)
from latentslate_engine.recipe import Artifact


@pytest.fixture
def klein_paths(tmp_path):
    return dict(
        zip(
            ("diffusion", "text_encoder", "vae", "tokenizer"),
            _klein_paths(tmp_path),
            strict=True,
        )
    )


def test_t2i_reuses_only_matching_two_image_capability_domains():
    shared = (
        "diffusion",
        "text_encoder",
        "vae",
        "tokenizer",
        "loras",
        "prompt",
        "seed",
    )
    assert {cap.key for cap in KLEIN9B_T2I_CAPABILITIES.capabilities} == {
        *shared,
        "width",
        "height",
    }
    for key in shared:
        assert KLEIN9B_T2I_CAPABILITIES[key] is KLEIN9B_TWO_IMAGE_CAPABILITIES[key]
    for key in ("width", "height"):
        concrete = KLEIN9B_T2I_CAPABILITIES[key]
        optional = KLEIN9B_TWO_IMAGE_CAPABILITIES[key]
        assert concrete is not optional
        assert replace(concrete, optional=True) == optional
        with pytest.raises(TypeError, match="does not accept None"):
            concrete.normalize(None)
        assert optional.normalize(None) is None


def test_klein_policies_import_without_configuration_or_artifacts():
    import subprocess
    import sys

    script = """
import builtins
import io
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from latentslate_engine.recipe import Artifact, ProductPolicy, Recipe
from latentslate_engine.klein9b.contracts import ArtifactIdentity, Klein9BIdentity

os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
os.environ.pop('LATENTSLATE_ENGINE_HOME', None)
os.environ.pop('LATENTSLATE_KLEIN9B_VAE', None)
environment = dict(os.environ)
def forbidden(*args, **kwargs):
    raise AssertionError('unbound Klein policy cannot inspect artifacts or configure')
original_import = builtins.__import__
def pure_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'diffusers', 'transformers', 'dotenv'} or name == 'latentslate_engine.service':
        forbidden()
    return original_import(name, *args, **kwargs)

# Python source/bytecode loading is allowed; application file IO is not.
with ExitStack() as stack:
    for name in ('open', 'read_text', 'read_bytes', 'stat', 'exists', 'is_file', 'is_dir', 'resolve'):
        stack.enter_context(patch.object(Path, name, forbidden))
    for name in ('open', 'stat', 'listdir', 'scandir'):
        stack.enter_context(patch.object(os, name, forbidden))
    stack.enter_context(patch.object(builtins, 'open', forbidden))
    stack.enter_context(patch.object(io, 'open', forbidden))
    stack.enter_context(patch.object(builtins, '__import__', pure_import))
    for cls in (Artifact, ArtifactIdentity, Klein9BIdentity):
        stack.enter_context(patch.object(cls, '__init__', forbidden))
    stack.enter_context(patch.object(ProductPolicy, 'bind', forbidden))
    stack.enter_context(patch.object(Recipe, '__post_init__', forbidden))
    from latentslate_engine.klein9b.recipes import KLEIN9B_T2I_POLICY, KLEIN9B_TWO_IMAGE_EXPLICIT_POLICY
    assert [x['key'] for x in KLEIN9B_T2I_POLICY.surface()] == ['prompt', 'width', 'height', 'seed']
    assert [x['key'] for x in KLEIN9B_TWO_IMAGE_EXPLICIT_POLICY.surface()] == ['prompt', 'image_1', 'image_2', 'width', 'height', 'seed']
    assert dict(os.environ) == environment
    for name in ('torch', 'diffusers', 'transformers', 'dotenv', 'latentslate_engine.service',
                 'latentslate_engine.klein9b.runtime', 'latentslate_engine.klein9b.two_image'):
        assert name not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("two_image", (False, True))
def test_explicit_products_surface_binding_and_native_identity(
    klein_paths, tmp_path, two_image
):
    policy = KLEIN9B_TWO_IMAGE_EXPLICIT_POLICY if two_image else KLEIN9B_T2I_POLICY
    builder = klein9b_two_image_explicit_recipe if two_image else klein9b_t2i_recipe
    resolver = resolve_klein9b_two_image if two_image else resolve_klein9b_t2i
    image_surface = (
        (
            {
                "key": "image_1",
                "type": "image",
                "required": True,
                "role": "start_image",
            },
            {"key": "image_2", "type": "image", "required": True, "role": "end_image"},
        )
        if two_image
        else ()
    )
    assert policy.surface() == (
        {"key": "prompt", "type": "text", "required": True},
        *image_surface,
        *(
            {
                "key": key,
                "type": "integer",
                "required": False,
                "default": 768,
                "role": key,
                "constraints": {"min": 256, "max": 4096, "step": 16},
            }
            for key in ("width", "height")
        ),
        {
            "key": "seed",
            "type": "integer",
            "required": False,
            "default": 0,
            "role": "seed",
            "constraints": {"min": 0, "max": 2**64 - 1},
        },
    )
    bindings = {key: Artifact(path) for key, path in klein_paths.items()}
    definition = builder(**klein_paths)
    assert definition == policy.bind(bindings)
    assert definition.surface() == policy.surface()
    inputs = {"prompt": "Two subjects"}
    if two_image:
        inputs.update(
            image_1=tmp_path / "unopened-first.png",
            image_2=tmp_path / "unopened-second.png",
        )
    values = definition.resolve(inputs)
    assert values == {
        **bindings,
        "loras": (),
        **inputs,
        "width": 768,
        "height": 768,
        "seed": 0,
    }
    expected_identity = Klein9BIdentity.from_paths(**klein_paths)
    assert expected_identity.loras == ()
    assert expected_identity.recipe == "flux2-klein-9b-distilled-t2i-768-v1"
    assert len(expected_identity.tokenizer_files) == 5
    for changes in (
        {},
        {"width": 2048, "height": 512, "seed": 2**64 - 1},
        {"prompt": "Changed", "width": 512, "height": 1024, "seed": 17},
    ):
        identity, request = resolver(definition, {**inputs, **changes})
        assert identity == expected_identity
        expected = {
            "prompt": inputs["prompt"],
            "seed": 0,
            "width": 768,
            "height": 768,
            **changes,
        }
        if two_image:
            expected.update(
                first_image=inputs["image_1"], second_image=inputs["image_2"]
            )
        assert request == expected
    for changes in ({"width": None}, {"height": None}, {"width": None, "height": None}):
        with pytest.raises(TypeError, match="does not accept None"):
            definition.resolve({**inputs, **changes})
    for changes in (
        {"width": 240},
        {"height": 257},
        {"width": 1024, "height": 1040},
        {"width": 1280, "height": 256},
        {"seed": -1},
        {"seed": 2**64},
    ):
        with pytest.raises(ValueError):
            definition.resolve({**inputs, **changes})
    with pytest.raises(ValueError, match="fixed.*loras"):
        definition.resolve({**inputs, "loras": ()})
    with pytest.raises(ValueError, match="policy-fixed.*loras"):
        policy.bind({**bindings, "loras": ()})
    with pytest.raises(ValueError, match="missing required"):
        definition.resolve({})
    other = (
        klein9b_t2i_recipe(**klein_paths)
        if two_image
        else klein9b_two_image_explicit_recipe(**klein_paths)
    )
    with pytest.raises(TypeError, match="capability set"):
        resolver(other, inputs)


def test_flexible_two_image_complete_pre_policy_oracle(klein_paths, tmp_path):
    """Captured before changing Klein recipes at 876f15d."""
    loras = (_file(tmp_path, "first-lora"), _file(tmp_path, "second-lora"))
    definition = klein9b_two_image_recipe(**klein_paths, loras=loras)
    assert definition.key == "flux2_klein9b.two_image.v1"
    assert definition.surface() == (
        {
            "key": "loras",
            "type": "artifact",
            "required": False,
            "default": [str(path) for path in loras],
            "collection": True,
            "ordered": True,
        },
        {"key": "prompt", "type": "text", "required": True},
        {"key": "image_1", "type": "image", "required": True, "role": "start_image"},
        {"key": "image_2", "type": "image", "required": True, "role": "end_image"},
        *(
            {
                "key": key,
                "type": "integer",
                "required": False,
                "default": None,
                "nullable": True,
                "role": key,
                "constraints": {"min": 256, "max": 4096, "step": 16},
            }
            for key in ("width", "height")
        ),
        {
            "key": "seed",
            "type": "integer",
            "required": False,
            "default": 0,
            "role": "seed",
            "constraints": {"min": 0, "max": 2**64 - 1},
        },
    )
    inputs = {
        "prompt": "Two subjects",
        "image_1": tmp_path / "unopened-first.png",
        "image_2": tmp_path / "unopened-second.png",
    }
    expected_identity = Klein9BIdentity.from_paths(**klein_paths, loras=loras)
    for geometry in (
        {},
        {"width": None, "height": None},
        {"width": 512, "height": 1024},
    ):
        identity, request = resolve_klein9b_two_image(
            definition, {**inputs, **geometry}
        )
        assert identity == expected_identity
        assert request == {
            "prompt": inputs["prompt"],
            "first_image": inputs["image_1"],
            "second_image": inputs["image_2"],
            "seed": 0,
            "width": geometry.get("width"),
            "height": geometry.get("height"),
        }
    for geometry in (
        {"width": 512},
        {"height": 512},
        {"width": None, "height": 512},
        {"width": 512, "height": None},
    ):
        with pytest.raises(ValueError, match="both be provided or both omitted"):
            definition.resolve({**inputs, **geometry})
    reversed_identity, request = resolve_klein9b_two_image(
        definition,
        {
            **inputs,
            "image_1": inputs["image_2"],
            "image_2": inputs["image_1"],
            "loras": tuple(Artifact(path) for path in reversed(loras)),
        },
    )
    assert reversed_identity == replace(
        expected_identity, loras=tuple(reversed(expected_identity.loras))
    )
    assert reversed_identity != expected_identity
    assert request["first_image"] == inputs["image_2"]
    assert request["second_image"] == inputs["image_1"]


def test_flexible_recipe_auto_geometry_is_owned_by_first_reference(
    klein_paths, tmp_path
):
    from PIL import Image

    from latentslate_engine.klein9b.two_image import (
        _source_scaled_dimensions,
        _target_geometry,
    )

    first, second = tmp_path / "landscape.png", tmp_path / "square.png"
    Image.new("RGB", (920, 630)).save(first)
    Image.new("RGB", (512, 512)).save(second)
    definition = klein9b_two_image_recipe(**klein_paths)
    for images, expected in (
        ((first, second), (1232, 832, 1237, 847)),
        ((second, first), (1024, 1024, 1024, 1024)),
    ):
        _, request = resolve_klein9b_two_image(
            definition,
            {"prompt": "Two subjects", "image_1": images[0], "image_2": images[1]},
        )
        assert (
            _target_geometry(
                *_source_scaled_dimensions(request["first_image"]),
                request["width"],
                request["height"],
            )
            == expected
        )


def test_current_klein_worker_identity_requests_and_shared_runtime(
    klein_paths, tmp_path, monkeypatch
):
    """Direct service calls at 876f15d, with real CPU lifecycle and bounded models."""
    import inspect
    from types import SimpleNamespace

    import torch
    from PIL import Image

    from latentslate_engine import service
    from latentslate_engine.klein9b import runtime as native
    from latentslate_engine.klein9b import two_image

    paths = service.KleinModelPaths(**klein_paths)
    expected_identity = Klein9BIdentity.from_paths(**klein_paths)
    first, second, duplicate = (
        tmp_path / name for name in ("first.png", "second.png", "copy.png")
    )
    Image.new("RGB", (16, 16), "red").save(first)
    Image.new("RGB", (16, 16), "blue").save(second)
    duplicate.write_bytes(first.read_bytes())
    common = {"prompt": "Two subjects", "width": 768, "height": 768, "seed": 0}
    image_inputs = {**common, "image_1": first, "image_2": second}
    jobs = [
        ("klein_t2i", common),
        ("klein_two_image", image_inputs),
        ("klein_t2i", {**common, "width": 256, "height": 512, "seed": 19}),
        ("klein_two_image", {**image_inputs, "image_1": duplicate}),
        ("klein_two_image", {**image_inputs, "image_1": second}),
        ("klein_two_image", {**image_inputs, "image_1": second, "image_2": first}),
        ("klein_t2i", {**common, "prompt": "Changed prompt"}),
        (
            "klein_two_image",
            {
                **image_inputs,
                "prompt": "Changed prompt",
                "image_1": second,
                "image_2": first,
            },
        ),
    ]
    calls, results, events, messages, instances, methods = [], [], [], [], [], []
    loads = []

    class Transformer:
        def __call__(self, latent, *args):
            return torch.zeros_like(latent)

    class Vae:
        bn = SimpleNamespace(running_mean=torch.zeros(128), running_var=torch.ones(128))

        def decode(self, latent, return_dict=False):
            return (torch.zeros((1, 3, latent.shape[2] * 8, latent.shape[3] * 8)),)

    def load_transformer(*args):
        loads.append("transformer")
        return Transformer()

    def load_vae(*args):
        loads.append("vae")
        return Vae()

    for module in (native, two_image):
        monkeypatch.setattr(module, "_load_transformer", load_transformer)
        monkeypatch.setattr(module, "_load_vae", load_vae)
        monkeypatch.setattr(
            module, "_encode_prompt", lambda *args: torch.zeros((1, 1, 12288))
        )

    def scale(image, method):
        methods.append(method)
        return image

    monkeypatch.setattr(two_image, "_scale_to_one_megapixel", scale)
    monkeypatch.setattr(
        two_image, "_encode_reference", lambda *args: torch.zeros((1, 128, 1, 1))
    )

    original_two_image_generate = two_image.Klein9BTwoImageRuntime.generate_two_image

    class CapturedRuntime(two_image.Klein9BTwoImageRuntime):
        def __init__(self):
            super().__init__(device="cpu")
            instances.append(self)
            events.append("construct")

        def capture(self, method, *args, **kwargs):
            arguments = dict(
                inspect.signature(method).bind(self, *args, **kwargs).arguments
            )
            del arguments["self"]
            identity = arguments.pop("identity")
            output = arguments.pop("output")
            assert callable(arguments.pop("progress"))
            assert identity == expected_identity
            calls.append((identity, arguments, output))
            result = method(self, *args, **kwargs)
            results.append(result)
            return result

        def generate(self, *args, **kwargs):
            return self.capture(native.Klein9BRuntime.generate, *args, **kwargs)

        def generate_two_image(self, *args, **kwargs):
            return self.capture(original_two_image_generate, *args, **kwargs)

        def close(self):
            events.append("close")
            super().close()

    monkeypatch.setattr(two_image, "Klein9BTwoImageRuntime", CapturedRuntime)
    pending = iter(
        [
            *(
                {
                    "type": "generate",
                    "operation": operation,
                    "inputs": inputs,
                    "output_path": tmp_path / f"{index}.png",
                }
                for index, (operation, inputs) in enumerate(jobs)
            ),
            {"type": "close"},
        ]
    )

    class Connection:
        def recv(self):
            events.append("receive")
            return next(pending)

        def send(self, message):
            messages.append(message)

        def close(self):
            events.append("connection-close")

    service._klein_worker_main(paths, Connection())
    assert len(instances) == 1
    assert len(calls) == len(jobs) == len(results)
    assert loads == ["transformer", "vae"]
    assert events == [
        "construct",
        "receive",
        "close",
        *(["receive"] * len(jobs)),
        "close",
        "connection-close",
    ]
    # The initial ensure_identity clears the cold object; operation changes never do.
    assert [r.models_reused for r in results] == [False] + [True] * 7
    assert [r.conditioning_reused for r in results] == [
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
    ]
    assert [results[n].reference_reused for n in (1, 3, 4, 5, 7)] == [
        (False, False),
        (True, True),
        (False, True),
        (True, False),
        (True, True),
    ]
    assert methods == ["nearest-exact", "lanczos", "nearest-exact", "lanczos"]
    for index, ((operation, inputs), (identity, request, output)) in enumerate(
        zip(jobs, calls, strict=True)
    ):
        expected_request = {
            key: inputs[key] for key in ("prompt", "seed", "width", "height")
        }
        if operation == "klein_two_image":
            expected_request.update(
                first_image=inputs["image_1"], second_image=inputs["image_2"]
            )
        assert request == expected_request
        assert output == tmp_path / f"{index}.png"
        with Image.open(output) as image:
            assert image.size == (inputs["width"], inputs["height"])
    wire_results = [item for item in messages if item["type"] == "result"]
    assert len(wire_results) == len(jobs)
    for wire, result in zip(wire_results, results, strict=True):
        details = {
            "models_reused": result.models_reused,
            "conditioning_reused": result.conditioning_reused,
        }
        if hasattr(result, "reference_reused"):
            details["reference_reused"] = result.reference_reused
        assert wire == {"type": "result", "ok": True, "details": details}
    assert sum(
        item.get("event", {}).get("progress") == 1.0 for item in messages
    ) == len(jobs)

    # Shadow-only execution: feed proposed resolvers to the same native methods.
    # The real service above still constructs its identity and request explicitly.
    baseline_calls, baseline_results = list(calls), list(results)
    calls.clear()
    results.clear()
    t2i = klein9b_t2i_recipe(**klein_paths)
    editing = klein9b_two_image_explicit_recipe(**klein_paths)
    with CapturedRuntime() as shadow:
        for index, (operation, inputs) in enumerate(jobs):
            if operation == "klein_t2i":
                identity, request = resolve_klein9b_t2i(t2i, inputs)
                generate = shadow.generate
            else:
                identity, request = resolve_klein9b_two_image(editing, inputs)
                generate = shadow.generate_two_image
            generate(
                identity=identity,
                **request,
                output=tmp_path / f"{index}.png",
                progress=lambda event: None,
            )
    assert calls == baseline_calls
    assert len(results) == len(baseline_results)
    for baseline, shadow in zip(baseline_results, results, strict=True):
        assert replace(shadow, elapsed_seconds=baseline.elapsed_seconds) == baseline
