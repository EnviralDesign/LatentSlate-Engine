"""Native-boundary parity oracles captured before recipe-backed execution."""

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest

from latentslate_engine import service
from latentslate_engine.identity import FileContentIdentity
from latentslate_engine.progress import report_progress


@pytest.fixture
def recipe_calls(monkeypatch):
    from latentslate_engine.ltx23 import recipes as ltx
    from latentslate_engine.recipe import ProductPolicy
    from latentslate_engine.wan2214b import recipes as wan

    bindings, resolutions = [], []
    bind = ProductPolicy.bind

    def record_bind(policy, values):
        result = bind(policy, values)
        bindings.append((policy, values, result))
        return result

    monkeypatch.setattr(ProductPolicy, "bind", record_bind)
    for module, name in (
        (ltx, "resolve_ltx23_i2v"),
        (ltx, "resolve_ltx23_t2v"),
        (ltx, "resolve_ltx23_flf"),
        (wan, "resolve_wan2214b_flf"),
        (wan, "resolve_wan2214b_t2v"),
        (wan, "resolve_wan2214b_i2v"),
    ):
        original = getattr(module, name)

        def record_resolve(definition, inputs, resolve=original):
            result = resolve(definition, inputs)
            resolutions.append((definition, inputs, result))
            return result

        monkeypatch.setattr(module, name, record_resolve)
    return bindings, resolutions


@pytest.mark.parametrize("operation", ("i2v", "t2v", "flf"))
def test_ltx_worker_native_identity_requests_and_lifecycle(
    tmp_path, monkeypatch, recipe_calls, operation
):
    from latentslate_engine.ltx23 import flf, i2v, t2v

    paths = service.LtxModelPaths(
        *(tmp_path / name for name in ("dev", "distilled", "text", "lora", "up"))
    )
    if operation == "t2v":
        expected_identity = t2v.Ltx23T2VIdentity(
            checkpoint_path=str(paths.dev_checkpoint),
            text_checkpoint_path=str(paths.text_checkpoint),
            transformer_lora_path=str(paths.transformer_lora),
            upsampler_path=str(paths.upsampler),
            lora_strength=0.5,
            device_index=0,
            transformer_loras=(),
        )
    elif operation == "flf":
        expected_identity = flf.Ltx23FlfIdentity(
            checkpoint_path=str(paths.distilled_checkpoint),
            text_checkpoint_path=str(paths.text_checkpoint),
            device_index=0,
        )
    else:
        expected_identity = i2v.Ltx23I2VIdentity(
            checkpoint_path=str(paths.dev_checkpoint),
            text_checkpoint_path=str(paths.text_checkpoint),
            transformer_lora_path=str(paths.transformer_lora),
            upsampler_path=str(paths.upsampler),
            lora_strength=0.5,
            device_index=0,
            transformer_loras=(),
        )
    inputs = [
        {
            "prompt": "first",
            "start_image": tmp_path / "first.png",
            "width": 512,
            "height": 512,
            "duration_seconds": 1.0,
            "seed": 0,
        },
        {
            "prompt": "second",
            "start_image": tmp_path / "second.png",
            "width": 768,
            "height": 512,
            "duration_seconds": 2.5,
            "seed": 42,
        },
    ]
    if operation == "t2v":
        for item in inputs:
            del item["start_image"]
    elif operation == "flf":
        for item in inputs:
            item["end_image"] = tmp_path / "last.png"
        inputs[1]["width"] = 544  # FLF retains its distinct 32-pixel lattice.
        inputs.append(
            {
                **inputs[1],
                "start_image": inputs[1]["end_image"],
                "end_image": inputs[1]["start_image"],
            }
        )
    calls, events, outputs, messages, instances = [], [], [], [], []

    class NativeRuntime:
        def __init__(self, identity):
            assert identity == expected_identity
            instances.append(self)
            events.append("construct")

        def generate(self, *, progress, **request):
            calls.append(request)
            report_progress(progress, 0.5, "Sampling")
            return SimpleNamespace(save_mp4=outputs.append)

        def close(self):
            events.append("close")

    pending = iter(
        [
            *(
                {
                    "type": "generate",
                    "inputs": item,
                    "output_path": tmp_path / f"{n}.mp4",
                }
                for n, item in enumerate(inputs)
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

    if operation == "t2v":
        monkeypatch.setattr(t2v, "Ltx23T2VRuntime", NativeRuntime)
    elif operation == "flf":
        monkeypatch.setattr(flf, "Ltx23FlfRuntime", NativeRuntime)
    else:
        monkeypatch.setattr(i2v, "Ltx23I2VRuntime", NativeRuntime)
    service._ltx_worker_main(operation, paths, Connection())
    expected_requests = []
    for item in inputs:
        request = {
            "prompt": item["prompt"],
            "width": item["width"],
            "height": item["height"],
            "duration_seconds": item["duration_seconds"],
            "seed": item["seed"],
        }
        if operation == "i2v":
            request["image_path"] = item["start_image"]
        elif operation == "flf":
            request["first_image_path"] = item["start_image"]
            request["last_image_path"] = item["end_image"]
        expected_requests.append(request)
    assert calls == expected_requests
    assert len(instances) == 1
    assert events == [
        "construct",
        *(["receive"] * (len(inputs) + 1)),
        "close",
        "connection-close",
    ]
    assert outputs == [tmp_path / f"{index}.mp4" for index in range(len(inputs))]
    expected_messages = []
    for _ in inputs:

        def callback(event):
            expected_messages.append({"type": "progress", "event": event})

        report_progress(callback, 0.5, "Sampling")
        report_progress(callback, 0.95, "Artifact encoding")
        report_progress(callback, 1.0, "Artifact encoding", stage_progress=1.0)
        expected_messages.append({"type": "result", "ok": True})
    assert messages == expected_messages
    from latentslate_engine.ltx23.recipes import (
        LTX23_FLF_POLICY,
        LTX23_I2V_POLICY,
        LTX23_T2V_POLICY,
    )
    from latentslate_engine.recipe import Artifact

    bindings, resolutions = recipe_calls
    assert len(bindings) == 1
    policy, bound, definition = bindings[0]
    assert (
        policy
        is {"t2v": LTX23_T2V_POLICY, "i2v": LTX23_I2V_POLICY, "flf": LTX23_FLF_POLICY}[
            operation
        ]
    )
    expected_bindings = {
        "checkpoint": Artifact(
            paths.distilled_checkpoint if operation == "flf" else paths.dev_checkpoint
        ),
        "text_checkpoint": Artifact(paths.text_checkpoint),
        "device_index": 0,
    }
    if operation != "flf":
        expected_bindings.update(
            {
                "upsampler": Artifact(paths.upsampler),
                "transformer_adapter_artifacts": (Artifact(paths.transformer_lora),),
                "transformer_adapter_strengths": (0.5,),
            }
        )
    assert bound == expected_bindings
    assert len(resolutions) == len(inputs)
    for resolution, item, call in zip(resolutions, inputs, calls, strict=True):
        assert resolution == (definition, item, (expected_identity, call))


def test_wan_flf_native_recipe_requests_and_reuse(tmp_path, monkeypatch, recipe_calls):
    from latentslate_engine.wan2214b import flf

    paths = service.WanModelPaths(
        *(tmp_path / field.name for field in fields(service.WanModelPaths))
    )
    for path in vars(paths).values():
        path.write_bytes(path.name.encode())
    previous_recipe = flf.WanFLFRecipe(
        high_checkpoint=str(paths.i2v_high_checkpoint),
        high_lora=str(paths.i2v_high_lora),
        low_checkpoint=str(paths.i2v_low_checkpoint),
        low_lora=str(paths.i2v_low_lora),
        text_encoder=str(paths.text_encoder),
        vae=str(paths.vae),
    )
    first, last = tmp_path / "first.png", tmp_path / "last.png"
    first.write_bytes(b"first")
    last.write_bytes(b"last")
    base = {
        "prompt": previous_recipe.positive,
        "start_image": first,
        "end_image": last,
        "width": 512,
        "height": 512,
        "duration_seconds": 5.0,
        "frame_count": 81,
        "seed": 0,
    }
    requests = [
        base,
        {**base, "seed": 1},
        {**base, "prompt": "changed"},
        {**base, "start_image": last, "end_image": first},
        {
            **base,
            "width": 480,
            "height": 496,
            "duration_seconds": 1.25,
            "frame_count": 21,
            "seed": 42,
        },
    ]
    calls, sessions, progress_events = [], [], []

    class NativeSession:
        def __init__(self, recipe):
            assert recipe == previous_recipe
            assert recipe.identity == previous_recipe.identity
            self.recipe = recipe
            self._conditioning = self._conditioning_key = self._flf_conditioning = None
            self.destroyed = False
            sessions.append(self)

        def generate(
            self,
            first_path,
            last_path,
            output_path,
            *,
            seed,
            width,
            height,
            frame_count,
            positive_prompt,
            negative_prompt=None,
            progress=None,
        ):
            effective_negative = (
                self.recipe.negative if negative_prompt is None else negative_prompt
            )
            calls.append(
                {
                    "first_path": first_path,
                    "last_path": last_path,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "frame_count": frame_count,
                    "positive_prompt": positive_prompt,
                    "negative_prompt": effective_negative,
                }
            )
            assert output_path == tmp_path / "out.mp4"
            # Compare all effective recipe state, including request defaults.
            assert replace(
                self.recipe,
                positive=positive_prompt,
                width=width,
                height=height,
                frame_count=frame_count,
            ) == replace(
                previous_recipe,
                positive=positive_prompt,
                width=width,
                height=height,
                frame_count=frame_count,
            )
            self._conditioning = object()
            self._conditioning_key = (positive_prompt, effective_negative)
            self._flf_conditioning = SimpleNamespace(
                identity=flf.OrderedSourceIdentity(
                    FileContentIdentity.from_path(first_path),
                    FileContentIdentity.from_path(last_path),
                    width,
                    height,
                    frame_count,
                )
            )
            report_progress(progress, 1.0, "Artifact encoding", stage_progress=1.0)
            return SimpleNamespace(timings={"total": 1.0})

        def destroy(self):
            self.destroyed = True

    monkeypatch.setattr(flf, "WanFLFSession", NativeSession)
    runtime = service._WanFamilyRuntime(paths)
    details = [
        runtime.generate("wan_flf", item, tmp_path / "out.mp4", progress_events.append)
        for item in requests
    ]
    assert calls == [
        {
            "first_path": item["start_image"],
            "last_path": item["end_image"],
            "seed": item["seed"],
            "width": item["width"],
            "height": item["height"],
            "frame_count": item["frame_count"],
            "positive_prompt": item["prompt"],
            "negative_prompt": previous_recipe.negative,
        }
        for item in requests
    ]
    assert [item["session_reused"] for item in details] == [
        False,
        True,
        True,
        True,
        True,
    ]
    assert [item["conditioning_reused"] for item in details] == [
        False,
        True,
        False,
        False,
        True,
    ]
    assert [item["image_conditioning_reused"] for item in details] == [
        False,
        True,
        True,
        False,
        False,
    ]
    assert all(item["timings"] == {"total": 1.0} for item in details)
    assert len(progress_events) == len(requests)
    assert len(sessions) == 1
    runtime.close()
    assert sessions[0].destroyed
    from latentslate_engine.recipe import Adapter, Artifact
    from latentslate_engine.wan2214b.recipes import WAN2214B_FLF_POLICY

    bindings, resolutions = recipe_calls
    assert len(bindings) == 1
    policy, bound, definition = bindings[0]
    assert policy is WAN2214B_FLF_POLICY
    assert bound == {
        "high_checkpoint": Artifact(paths.i2v_high_checkpoint),
        "high_adapters": (Adapter(Artifact(paths.i2v_high_lora), 1.0),),
        "low_checkpoint": Artifact(paths.i2v_low_checkpoint),
        "low_adapters": (Adapter(Artifact(paths.i2v_low_lora), 1.0),),
        "text_encoder": Artifact(paths.text_encoder),
        "vae": Artifact(paths.vae),
        "negative_prompt": previous_recipe.negative,
    }
    # First real request supplies the constructor recipe; every call resolves again.
    assert len(resolutions) == len(requests) + 1
    assert resolutions[0][2][0] == previous_recipe
    for resolution, item, call in zip(resolutions[1:], requests, calls, strict=True):
        resolved_definition, overrides, (native, request) = resolution
        assert resolved_definition is definition
        assert overrides == {
            key: value for key, value in item.items() if key != "frame_count"
        }
        assert native == replace(
            previous_recipe,
            positive=item["prompt"],
            width=item["width"],
            height=item["height"],
            frame_count=item["frame_count"],
        )
        assert native.identity == previous_recipe.identity
        assert request == call


@pytest.mark.parametrize("operation", ("wan_flf", "wan_t2v", "wan_i2v"))
def test_wan_admitted_duration_lattice_and_first_request_defaults(
    tmp_path, monkeypatch, operation
):
    import io

    from fastapi import UploadFile
    from test_service import FakeRuntime, _job_body, _png, _wan_paths

    from latentslate_engine.wan2214b import flf, i2v, pipeline

    paths = _wan_paths(tmp_path / "models")
    paths.text_encoder.parent.mkdir()
    for path in vars(paths).values():
        path.write_bytes(path.name.encode())
    if operation == "wan_t2v":
        old_recipe = pipeline.WanRecipe(
            high_checkpoint=str(paths.t2v_high_checkpoint),
            high_lora=str(paths.t2v_high_lora),
            low_checkpoint=str(paths.t2v_low_checkpoint),
            low_lora=str(paths.t2v_low_lora),
            text_encoder=str(paths.text_encoder),
            vae=str(paths.vae),
        )
        tool_id = service.WAN_T2V_ID
    elif operation == "wan_i2v":
        old_recipe = i2v.WanI2VRecipe(
            high_checkpoint=str(paths.i2v_high_checkpoint),
            high_lora=str(paths.i2v_high_lora),
            low_checkpoint=str(paths.i2v_low_checkpoint),
            low_lora=str(paths.i2v_low_lora),
            text_encoder=str(paths.text_encoder),
            vae=str(paths.vae),
        )
        tool_id = service.WAN_I2V_ID
    else:
        old_recipe = flf.WanFLFRecipe(
            high_checkpoint=str(paths.i2v_high_checkpoint),
            high_lora=str(paths.i2v_high_lora),
            low_checkpoint=str(paths.i2v_low_checkpoint),
            low_lora=str(paths.i2v_low_lora),
            text_encoder=str(paths.text_encoder),
            vae=str(paths.vae),
        )
        tool_id = service.WAN_FLF_ID
    sessions, calls = [], []

    class NativeSession:
        def __init__(self, recipe):
            assert recipe == replace(
                old_recipe,
                positive="A small test scene",
                width=480,
                height=480,
                frame_count=17,
            )
            assert recipe.identity == old_recipe.identity
            self.recipe = recipe
            self._conditioning = self._conditioning_key = self._flf_conditioning = None
            self._image_conditioning = None
            sessions.append(self)

        def generate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(timings={})

        def destroy(self):
            pass

    monkeypatch.setattr(flf, "WanFLFSession", NativeSession)
    monkeypatch.setattr(pipeline, "WanSession", NativeSession)
    monkeypatch.setattr(i2v, "WanI2VSession", NativeSession)
    admission = service.EngineService(tmp_path / "service", FakeRuntime())
    runtime = service._WanFamilyRuntime(paths)
    try:
        asset = admission.store_asset(
            UploadFile(file=io.BytesIO(_png()), filename="source.png")
        )
        reference = {"type": "asset", "asset_id": str(asset.id)}
        images = {} if operation == "wan_t2v" else {"start_image": reference}
        if operation == "wan_flf":
            images["end_image"] = reference
        for quarter in range(4, 21):
            duration = quarter / 4
            operation, inputs, _ = admission._validate_job(
                _job_body(
                    tool_id,
                    duration_seconds=duration,
                    **images,
                )
            )
            details = runtime.generate(operation, inputs, tmp_path / "out.mp4")
            assert calls[-1]["frame_count"] == inputs["frame_count"] == quarter * 4 + 1
            assert inputs["duration_seconds"] == duration
            if operation == "wan_flf":
                assert calls[-1]["first_path"] == calls[-1]["last_path"] == asset.path
            elif operation == "wan_i2v":
                assert calls[-1]["source_path"] == asset.path
            assert calls[-1]["negative_prompt"] == old_recipe.negative
            assert details["session_reused"] is (quarter != 4)
        assert len(sessions) == 1
    finally:
        runtime.close()
        admission.close()


@pytest.mark.parametrize("operation", ("wan_t2v", "wan_i2v"))
def test_wan_t2v_i2v_native_parity(tmp_path, monkeypatch, operation, recipe_calls):
    """Captured against the direct service constructors at 56d6fb1."""
    from latentslate_engine.wan2214b import i2v, pipeline

    paths = service.WanModelPaths(
        *(tmp_path / field.name for field in fields(service.WanModelPaths))
    )
    for path in vars(paths).values():
        path.write_bytes(path.name.encode())
    if operation == "wan_t2v":
        old_recipe = pipeline.WanRecipe(
            high_checkpoint=str(paths.t2v_high_checkpoint),
            high_lora=str(paths.t2v_high_lora),
            low_checkpoint=str(paths.t2v_low_checkpoint),
            low_lora=str(paths.t2v_low_lora),
            text_encoder=str(paths.text_encoder),
            vae=str(paths.vae),
        )
    else:
        old_recipe = i2v.WanI2VRecipe(
            high_checkpoint=str(paths.i2v_high_checkpoint),
            high_lora=str(paths.i2v_high_lora),
            low_checkpoint=str(paths.i2v_low_checkpoint),
            low_lora=str(paths.i2v_low_lora),
            text_encoder=str(paths.text_encoder),
            vae=str(paths.vae),
        )
    assert (
        old_recipe.high_lora_strength
        == old_recipe.low_lora_strength
        == 1.0000000000000002
    )
    assert old_recipe.high_secondary_lora is old_recipe.low_secondary_lora is None
    source, copy, other = (
        tmp_path / name for name in ("source.png", "copy.png", "other.png")
    )
    source.write_bytes(b"source")
    copy.write_bytes(b"source")
    other.write_bytes(b"other")
    base = {
        "prompt": old_recipe.positive,
        "width": 512,
        "height": 512,
        "duration_seconds": 5.0,
        "frame_count": 81,
        "seed": 0,
    }
    if operation == "wan_i2v":
        base["start_image"] = source
    requests = [
        base,
        {**base, "seed": 19},
        {**base, "prompt": "changed"},
        {**base, "width": 480},
        {**base, "height": 496},
        {**base, "duration_seconds": 1.25, "frame_count": 21},
    ]
    if operation == "wan_i2v":
        requests += [
            base,
            {**base, "start_image": copy},
            {**base, "start_image": other},
        ]
    calls, sessions, progress_events = [], [], []

    class CapturedSession:
        def __init__(self, recipe):
            assert recipe == old_recipe
            assert recipe.identity == old_recipe.identity
            self.recipe = recipe
            self._conditioning = self._conditioning_key = self._image_conditioning = (
                None
            )
            self.destroyed = False
            sessions.append(self)

        def capture(
            self,
            output_path,
            source_path=None,
            *,
            seed,
            width,
            height,
            frame_count,
            positive_prompt,
            negative_prompt=None,
            progress=None,
        ):
            negative = (
                self.recipe.negative if negative_prompt is None else negative_prompt
            )
            request = {
                "seed": seed,
                "width": width,
                "height": height,
                "frame_count": frame_count,
                "positive_prompt": positive_prompt,
                "negative_prompt": negative,
            }
            if operation == "wan_i2v":
                request["source_path"] = source_path
                self._image_conditioning = SimpleNamespace(
                    identity=i2v.ImageConditioningIdentity(
                        FileContentIdentity.from_path(source_path),
                        width,
                        height,
                        frame_count,
                    )
                )
            assert output_path == tmp_path / "out.mp4"
            assert replace(
                self.recipe,
                width=width,
                height=height,
                frame_count=frame_count,
                positive=positive_prompt,
                negative=negative,
            ) == replace(
                old_recipe,
                width=width,
                height=height,
                frame_count=frame_count,
                positive=positive_prompt,
            )
            self._conditioning = object()
            self._conditioning_key = (positive_prompt, negative)
            calls.append(request)
            report_progress(progress, 1.0, "Artifact encoding", stage_progress=1.0)
            return SimpleNamespace(timings={"total": 1.0})

        def destroy(self):
            self.destroyed = True

    class T2VSession(CapturedSession):
        def generate(self, output_path, **kwargs):
            return self.capture(output_path, **kwargs)

    class I2VSession(CapturedSession):
        def generate(self, source_path, output_path, **kwargs):
            return self.capture(output_path, source_path, **kwargs)

    monkeypatch.setattr(pipeline, "WanSession", T2VSession)
    monkeypatch.setattr(i2v, "WanI2VSession", I2VSession)
    runtime = service._WanFamilyRuntime(paths)
    assert sessions == []  # No native session before the first real request.
    details = [
        runtime.generate(operation, item, tmp_path / "out.mp4", progress_events.append)
        for item in requests
    ]
    expected = []
    for item in requests:
        request = {
            "seed": item["seed"],
            "width": item["width"],
            "height": item["height"],
            "frame_count": item["frame_count"],
            "positive_prompt": item["prompt"],
            "negative_prompt": old_recipe.negative,
        }
        if operation == "wan_i2v":
            request["source_path"] = item["start_image"]
        expected.append(request)
    assert calls == expected
    assert len(sessions) == 1
    assert [item["session_reused"] for item in details] == [False] + [True] * (
        len(requests) - 1
    )
    assert [item["conditioning_reused"] for item in details[:6]] == [
        False,
        True,
        False,
        False,
        True,
        True,
    ]
    if operation == "wan_i2v":
        assert [item["image_conditioning_reused"] for item in details] == [
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
        ]
    else:
        assert all("image_conditioning_reused" not in item for item in details)
    assert all(item["timings"] == {"total": 1.0} for item in details)
    expected_progress = []
    report_progress(
        expected_progress.append, 1.0, "Artifact encoding", stage_progress=1.0
    )
    assert progress_events == expected_progress * len(requests)
    runtime.close()
    assert sessions[0].destroyed
    from latentslate_engine.recipe import Adapter, Artifact
    from latentslate_engine.wan2214b.recipes import (
        WAN2214B_I2V_POLICY,
        WAN2214B_T2V_POLICY,
    )

    bindings, resolutions = recipe_calls
    assert len(bindings) == 1
    policy, bound, definition = bindings[0]
    assert policy is (
        WAN2214B_T2V_POLICY if operation == "wan_t2v" else WAN2214B_I2V_POLICY
    )
    assert bound == {
        "high_checkpoint": Artifact(old_recipe.high_checkpoint),
        "high_adapters": (
            Adapter(Artifact(old_recipe.high_lora), old_recipe.high_lora_strength),
        ),
        "low_checkpoint": Artifact(old_recipe.low_checkpoint),
        "low_adapters": (
            Adapter(Artifact(old_recipe.low_lora), old_recipe.low_lora_strength),
        ),
        "text_encoder": Artifact(paths.text_encoder),
        "vae": Artifact(paths.vae),
        "negative_prompt": old_recipe.negative,
    }
    assert len(resolutions) == len(requests) + 1
    assert resolutions[0][2][0] == old_recipe
    for resolution, item, call in zip(resolutions[1:], requests, calls, strict=True):
        resolved_definition, overrides, (native, request) = resolution
        assert resolved_definition is definition
        assert overrides == {
            key: value for key, value in item.items() if key != "frame_count"
        }
        assert native == replace(
            old_recipe,
            positive=item["prompt"],
            width=item["width"],
            height=item["height"],
            frame_count=item["frame_count"],
        )
        assert native.identity == old_recipe.identity
        assert request == call
