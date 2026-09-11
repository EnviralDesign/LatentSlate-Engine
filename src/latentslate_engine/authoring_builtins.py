"""Inert copies of the eight certified service configurations for authoring."""

from uuid import NAMESPACE_URL, uuid5

from .authoring import document_from_recipe
from .klein9b.recipes import klein9b_t2i_recipe, klein9b_two_image_explicit_recipe
from .ltx23.recipes import ltx23_flf_recipe, ltx23_i2v_recipe, ltx23_t2v_recipe
from .recipe import Adapter, Artifact
from .wan2214b.contracts import NEGATIVE_PROMPT
from .wan2214b.recipes import (
    wan2214b_flf_recipe,
    wan2214b_i2v_recipe,
    wan2214b_t2v_recipe,
)

# Certified I2V/FLF prompt differs from T2V. Keep this inert; the native equality
# test protects the copy without importing a GPU execution module in authoring.
_WAN_IMAGE_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def builtin_documents(ltx, klein, wan) -> dict[str, dict]:
    """Use host-selected paths and existing product factories without a runtime."""
    two_pass = {
        "checkpoint": ltx.dev_checkpoint,
        "text_checkpoint": ltx.text_checkpoint,
        "upsampler": ltx.upsampler,
        "transformer_adapters": (Adapter(Artifact(ltx.transformer_lora), 0.5),),
        "device_index": 0,
    }
    klein_bindings = {
        key: getattr(klein, key)
        for key in ("diffusion", "text_encoder", "vae", "tokenizer")
    }
    recipes = [
        ("LTX 2.3 Text to Video", ltx23_t2v_recipe(**two_pass)),
        ("LTX 2.3 Image to Video", ltx23_i2v_recipe(**two_pass)),
        (
            "LTX 2.3 First/Last Frame",
            ltx23_flf_recipe(
                checkpoint=ltx.distilled_checkpoint, text_checkpoint=ltx.text_checkpoint
            ),
        ),
        ("Klein 9B Text to Image", klein9b_t2i_recipe(**klein_bindings)),
        ("Klein 9B Two Image", klein9b_two_image_explicit_recipe(**klein_bindings)),
    ]
    for name, factory, prefix, strength, negative in (
        (
            "Wan 2.2 Text to Video",
            wan2214b_t2v_recipe,
            "t2v",
            1.0000000000000002,
            NEGATIVE_PROMPT,
        ),
        (
            "Wan 2.2 Image to Video",
            wan2214b_i2v_recipe,
            "i2v",
            1.0000000000000002,
            _WAN_IMAGE_NEGATIVE,
        ),
        (
            "Wan 2.2 First/Last Frame",
            wan2214b_flf_recipe,
            "i2v",
            1.0,
            _WAN_IMAGE_NEGATIVE,
        ),
    ):
        recipes.append(
            (
                name,
                factory(
                    high_checkpoint=getattr(wan, f"{prefix}_high_checkpoint"),
                    high_adapters=(
                        Adapter(
                            Artifact(getattr(wan, f"{prefix}_high_lora")), strength
                        ),
                    ),
                    low_checkpoint=getattr(wan, f"{prefix}_low_checkpoint"),
                    low_adapters=(
                        Adapter(Artifact(getattr(wan, f"{prefix}_low_lora")), strength),
                    ),
                    text_encoder=wan.text_encoder,
                    vae=wan.vae,
                    negative_prompt=negative,
                ),
            )
        )
    return {
        recipe.key: document_from_recipe(
            recipe,
            name=name,
            recipe_id=str(
                uuid5(NAMESPACE_URL, f"latentslate:authoring:builtin:{recipe.key}")
            ),
        )
        for name, recipe in recipes
    }
