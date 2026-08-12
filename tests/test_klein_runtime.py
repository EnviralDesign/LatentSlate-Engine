import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from latentslate_engine.bundles import BUNDLES
from latentslate_engine.config import Settings
from latentslate_engine.protocol import InputRole, WorkflowKind
from latentslate_engine.runtime.klein import KleinRuntime, resolve_klein_runtime_plan
from latentslate_engine.runtime.manager import RUNTIME_MANAGER
from latentslate_engine.tools import klein as klein_tools


def test_klein_dimension_contract_aligns_explicit_canvases():
    explicit = KleinRuntime._resolve_dimensions(
        width=513,
        height=519,
        image_paths=[],
    )
    assert (explicit.width, explicit.height) == (512, 512)

    try:
        KleinRuntime._resolve_dimensions(width=512, height=None, image_paths=[])
    except ValueError as exc:
        assert "provided together" in str(exc)
    else:
        raise AssertionError("partial explicit Klein dimensions were accepted")

    try:
        KleinRuntime._resolve_dimensions(width=1024, height=1040, image_paths=[])
    except ValueError as exc:
        assert "pixel budget" in str(exc)
    else:
        raise AssertionError("over-budget Klein dimensions were accepted")


def test_klein_source_sizing_uses_exif_oriented_visible_canvas_then_floors(tmp_path):
    from PIL import Image

    source = tmp_path / "source.png"
    image = Image.new("RGB", (517, 513))
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90° clockwise: visible canvas becomes 513x517.
    image.save(source, exif=exif)
    dimensions = KleinRuntime._resolve_dimensions(
        width=None,
        height=None,
        image_paths=[source],
    )
    assert dimensions.metadata() == {
        "requested_dimensions": {"width": 513, "height": 517},
        "effective_dimensions": {"width": 512, "height": 512},
    }


def test_klein_distilled_reference_preprocessing_matches_one_mp_ordered_contract(tmp_path):
    from PIL import Image

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (400, 200), (255, 0, 0)).save(first)
    Image.new("RGB", (200, 400), (0, 0, 255)).save(second)

    dimensions = KleinRuntime._resolve_dimensions(
        width=None,
        height=None,
        image_paths=[first, second],
        scale_references_to_one_mp=True,
    )
    first_size = KleinRuntime._one_megapixel_size(400, 200)
    second_size = KleinRuntime._one_megapixel_size(200, 400)
    assert dimensions.requested_width == first_size[0]
    assert dimensions.requested_height == first_size[1]
    assert dimensions.width % 16 == dimensions.height % 16 == 0

    ordered = [
        KleinRuntime._scale_reference_to_one_megapixel(image)
        for image in (
            Image.open(first),
            Image.open(second),
        )
    ]
    assert [image.size for image in ordered] == [first_size, second_size]
    assert ordered[0].getpixel((0, 0)) == (255, 0, 0)
    assert ordered[1].getpixel((0, 0)) == (0, 0, 255)


def test_klein_distilled_recipe_schedule_is_euler_support_four_step_guidance_one():
    plan = SimpleNamespace(
        pipeline_parameters=(
            ("recipe_fingerprint", "fixture"),
            ("recipe_mode", "distilled"),
            ("steps", 4),
            ("guidance_scale", 1.0),
        )
    )
    assert KleinRuntime._schedule(plan) == {
        "mode": "distilled",
        "steps": 4,
        "guidance_scale": 1.0,
    }


def test_all_klein_i2i_modes_require_partial_transformer_residency():
    assert KleinRuntime._requires_partial_residency([Path("base-or-distilled-reference.png")])
    assert KleinRuntime._requires_partial_residency(
        [Path("first.png"), Path("second.png"), Path("third.png")]
    )
    assert not KleinRuntime._requires_partial_residency([])


def test_klein_tools_follow_latentslate_taxonomy():
    text4 = klein_tools.Klein4BTextToImageTool().descriptor
    edit4 = klein_tools.Klein4BImageToImageTool().descriptor
    text9 = klein_tools.KleinTextToImageTool().descriptor
    edit9 = klein_tools.KleinImageToImageTool().descriptor

    assert text4.name == "Klein 4B Text to Image"
    assert text4.workflow_kind == WorkflowKind.TEXT_TO_IMAGE
    assert edit4.name == "Klein 4B Image to Image (1-3 refs)"
    assert edit4.workflow_kind == WorkflowKind.IMAGE_TO_IMAGE
    assert text9.name == "Klein 9B Text to Image"
    assert edit9.name == "Klein 9B Image to Image (1-3 refs)"

    text4_inputs = {item.key: item for item in text4.inputs}
    text9_inputs = {item.key: item for item in text9.inputs}
    edit4_inputs = {item.key: item for item in edit4.inputs}
    edit9_inputs = {item.key: item for item in edit9.inputs}

    assert "size" not in text4_inputs
    assert (text4_inputs["width"].default, text4_inputs["height"].default) == (512, 512)
    assert (text9_inputs["width"].default, text9_inputs["height"].default) == (1024, 1024)
    assert edit4_inputs["width"].default is None
    assert edit4_inputs["height"].default is None
    assert text4_inputs["width"].role == InputRole.WIDTH
    assert text4_inputs["height"].role == InputRole.HEIGHT
    assert edit4_inputs["source_image"].role == InputRole.SOURCE_IMAGE
    assert edit4_inputs["reference_image_2"].required is False
    assert edit4_inputs["reference_image_3"].required is False
    assert edit9_inputs["reference_image_2"].required is False
    assert edit9_inputs["reference_image_3"].required is False


def test_klein_bundles_require_complete_self_contained_repositories():
    bundle4 = BUNDLES["klein4b-basic"]
    assert bundle4.repo_id == "black-forest-labs/FLUX.2-klein-4B"
    assert bundle4.required_repo_ids() == {"black-forest-labs/FLUX.2-klein-4B"}

    bundle9 = BUNDLES["klein9b-basic"]
    assert bundle9.repo_id == "black-forest-labs/FLUX.2-klein-9B"
    assert bundle9.required_repo_ids() == {"black-forest-labs/FLUX.2-klein-9B"}
    assert bundle9.files == ()
    assert bundle9.allow_patterns == ()


def test_klein_defaults_prioritize_native_bf16_profiles(tmp_path):
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    assert settings.klein4b_model_id.endswith("FLUX.2-klein-4B")
    assert settings.klein4b_profile == "bf16_model_offload"
    assert settings.klein_profile == "bf16_model_offload"


def test_klein_tools_share_runtime_within_variant_and_evict_between_variants(
    tmp_path,
    monkeypatch,
):
    created = []
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    for family, repository in (
        ("klein4b", "black-forest-labs--FLUX.2-klein-4B"),
        ("klein9b", "black-forest-labs--FLUX.2-klein-9B"),
    ):
        path = settings.model_root / family / repository
        path.mkdir(parents=True)
        (path / "model_index.json").write_text("{}", encoding="utf-8")
    transformer_repo = settings.model_root / "klein9b" / "black-forest-labs--FLUX.2-klein-9b-nvfp4"
    transformer_repo.mkdir(parents=True)
    (transformer_repo / "flux-2-klein-9b-nvfp4.safetensors").write_bytes(b"nvfp4")
    text_encoder_repo = settings.model_root / "klein9b" / "Qwen--Qwen3-8B-FP8"
    text_encoder_repo.mkdir(parents=True)
    (text_encoder_repo / "config.json").write_text("{}", encoding="utf-8")
    (text_encoder_repo / "model.safetensors").write_bytes(b"qwen")

    class FakeRuntime:
        def __init__(self, settings, variant, plan):
            self.settings = settings
            self.variant = variant
            self.plan = plan
            self.unloaded = False
            created.append(self)

        def unload(self):
            self.unloaded = True

    RUNTIME_MANAGER.clear()
    monkeypatch.setattr(klein_tools, "KleinRuntime", FakeRuntime)
    context = SimpleNamespace(settings=settings)
    plan4 = resolve_klein_runtime_plan(settings, "klein4b", None)
    plan9 = resolve_klein_runtime_plan(settings, "klein9b", None)

    text4_runtime = klein_tools.Klein4BTextToImageTool()._runtime(context, plan4)
    edit4_runtime = klein_tools.Klein4BImageToImageTool()._runtime(context, plan4)
    assert text4_runtime is edit4_runtime
    assert created == [text4_runtime]

    text9_runtime = klein_tools.KleinTextToImageTool()._runtime(context, plan9)
    edit9_runtime = klein_tools.KleinImageToImageTool()._runtime(context, plan9)
    assert text9_runtime is edit9_runtime
    assert text4_runtime.unloaded is True
    assert created == [text4_runtime, text9_runtime]
    RUNTIME_MANAGER.clear()


def test_klein_runtime_passes_one_to_three_references_to_diffusers(tmp_path, monkeypatch):
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    model = settings.model_root / "klein4b" / "black-forest-labs--FLUX.2-klein-4B"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    plan = resolve_klein_runtime_plan(settings, "klein4b", None)
    runtime = KleinRuntime(settings, "klein4b", plan)
    calls = []

    class FakeGenerator:
        def __init__(self, device):
            self.device = device
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class FakeImage:
        width = 512
        height = 512

        def save(self, path, format):
            assert format == "PNG"
            path.write_bytes(b"png")

    class FakePipeline:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[FakeImage()])

    fake_torch = ModuleType("torch")
    fake_torch.Generator = FakeGenerator
    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.__path__ = []
    fake_utils = ModuleType("diffusers.utils")
    fake_utils.load_image = lambda path: f"loaded:{path}"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", fake_utils)
    monkeypatch.setattr(runtime, "_load_pipeline", lambda: FakePipeline())

    paths = [tmp_path / f"ref-{index}.png" for index in range(3)]
    for path in paths:
        path.write_bytes(b"ref")
    output = tmp_path / "output.png"

    metadata = runtime.generate(
        plan=plan,
        prompt="combine the references",
        output_path=output,
        width=512,
        height=512,
        seed=42,
        image_paths=paths,
        progress=lambda *_: None,
        check_cancelled=lambda: None,
    )

    assert output.read_bytes() == b"png"
    assert calls[0]["image"] == [f"loaded:{path}" for path in paths]
    assert calls[0]["width"] == 512
    assert calls[0]["height"] == 512
    assert metadata["requested_dimensions"] == {"width": 512, "height": 512}
    assert metadata["effective_dimensions"] == {"width": 512, "height": 512}
    assert metadata["reference_count"] == 3
    assert metadata["model_variant"] == "klein4b"


def test_klein_runtime_omitted_i2i_dimensions_keep_kwargs_omitted_but_report_floor(
    tmp_path,
    monkeypatch,
):
    from PIL import Image

    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    model = settings.model_root / "klein4b" / "black-forest-labs--FLUX.2-klein-4B"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    plan = resolve_klein_runtime_plan(settings, "klein4b", None)
    runtime = KleinRuntime(settings, "klein4b", plan)
    calls = []

    class FakeGenerator:
        def __init__(self, *, device):
            assert device == "cpu"

        def manual_seed(self, _seed):
            return self

    class FakeImage:
        width = 512
        height = 512

        def save(self, path, format):
            assert format == "PNG"
            path.write_bytes(b"png")

    class FakePipeline:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[FakeImage()])

    source = tmp_path / "source.png"
    Image.new("RGB", (513, 517)).save(source)
    fake_torch = ModuleType("torch")
    fake_torch.Generator = FakeGenerator
    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.__path__ = []
    fake_utils = ModuleType("diffusers.utils")
    fake_utils.load_image = lambda path: f"loaded:{path}"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", fake_utils)
    monkeypatch.setattr(runtime, "_load_pipeline", lambda: FakePipeline())

    metadata = runtime.generate(
        plan=plan,
        prompt="edit",
        output_path=tmp_path / "output.png",
        width=None,
        height=None,
        seed=0,
        image_paths=[source],
        progress=lambda *_: None,
        check_cancelled=lambda: None,
    )

    assert "width" not in calls[0]
    assert "height" not in calls[0]
    assert metadata["requested_dimensions"] == {"width": 513, "height": 517}
    assert metadata["effective_dimensions"] == {"width": 512, "height": 512}
    assert (metadata["width"], metadata["height"]) == (512, 512)


def test_klein_runtime_rejects_over_budget_dimensions_before_loading_pipeline(tmp_path, monkeypatch):
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    model = settings.model_root / "klein4b" / "black-forest-labs--FLUX.2-klein-4B"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    plan = resolve_klein_runtime_plan(settings, "klein4b", None)
    runtime = KleinRuntime(settings, "klein4b", plan)
    monkeypatch.setattr(
        runtime,
        "_load_pipeline",
        lambda: (_ for _ in ()).throw(AssertionError("pipeline loaded")),
    )

    try:
        runtime.generate(
            plan=plan,
            prompt="x",
            output_path=tmp_path / "no-output.png",
            width=1024,
            height=1040,
            seed=0,
            image_paths=[],
            progress=lambda *_: None,
            check_cancelled=lambda: None,
        )
    except ValueError as exc:
        assert "pixel budget" in str(exc)
    else:
        raise AssertionError("over-budget dimensions reached the pipeline")


def test_klein_reference_encode_offloads_bf16_vae_before_transformer_phase(
    tmp_path, monkeypatch
):
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    model = settings.model_root / "klein4b" / "black-forest-labs--FLUX.2-klein-4B"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    runtime = KleinRuntime(
        settings,
        "klein4b",
        resolve_klein_runtime_plan(settings, "klein4b", None),
    )
    events: list[str] = []

    class Hook:
        def __init__(self, model, name):
            self.model = model
            self.name = name

        def offload(self):
            events.append(f"{self.name}_offload")

    class Transformer:
        def forward(self):
            events.append("transformer_forward")

    from accelerate import hooks as accelerate_hooks

    vae = object()
    pipe = SimpleNamespace(
        vae=vae,
        transformer=Transformer(),
        # Keep the VAE out of position one to prove selection is by module
        # identity, not by the Diffusers offload sequence's hook index.
        _all_hooks=[Hook(object(), "transformer"), Hook(vae, "vae")],
    )
    monkeypatch.setattr(accelerate_hooks, "UserCpuOffloadHook", Hook)
    monkeypatch.setattr(
        runtime,
        "_prepare_image_latents_cached",
        lambda *args: events.append("vae_encode") or ("latents", "ids"),
    )
    runtime._install_reference_cache(pipe)

    assert pipe.prepare_image_latents([], 1, None, "cpu", None) == ("latents", "ids")
    pipe.transformer.forward()
    assert events == ["vae_encode", "vae_offload", "transformer_forward"]


def test_klein_reference_encode_preserves_stored_vae_offload(tmp_path, monkeypatch):
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    model = settings.model_root / "klein4b" / "black-forest-labs--FLUX.2-klein-4B"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    runtime = KleinRuntime(
        settings,
        "klein4b",
        resolve_klein_runtime_plan(settings, "klein4b", None),
    )
    events: list[str] = []

    class Hook:
        def offload(self):
            events.append("vae_offload")

    runtime._active_plan = SimpleNamespace(
        model_format="safetensors",
        quantization="fp8",
        offload="staged",
    )
    runtime._dense_offload_hooks = {"vae": Hook()}
    pipe = SimpleNamespace()
    monkeypatch.setattr(
        runtime,
        "_prepare_image_latents_cached",
        lambda *args: events.append("vae_encode") or ("latents", "ids"),
    )
    runtime._install_reference_cache(pipe)

    assert pipe.prepare_image_latents([], 1, None, "cpu", None) == ("latents", "ids")
    assert events == ["vae_encode", "vae_offload"]


def test_klein_standard_model_offload_keeps_t2i_hook_sequence(tmp_path, monkeypatch):
    settings = Settings(
        home=tmp_path,
        token=None,
        max_upload_bytes=1024,
        h3_model_id="unused",
        h3_profile="bf16_auto_offload",
        h3_device="cuda",
    )
    settings.ensure_directories()
    model = settings.model_root / "klein4b" / "black-forest-labs--FLUX.2-klein-4B"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    plan = resolve_klein_runtime_plan(settings, "klein4b", None)
    runtime = KleinRuntime(settings, "klein4b", plan)
    pipe = SimpleNamespace(model_cpu_offload_seq="text_encoder->transformer->vae")
    seen_sequences: list[str] = []

    monkeypatch.setattr(runtime, "_load_standard", lambda _: pipe)
    monkeypatch.setattr(
        "latentslate_engine.runtime.klein.apply_pipeline_kit",
        lambda pipeline, _: seen_sequences.append(pipeline.model_cpu_offload_seq) or {},
    )

    assert runtime._load_pipeline() is pipe
    assert seen_sequences == ["text_encoder->transformer->vae"]
