"""Portable LTX 2.5 recipe policy checks using synthetic artifact paths."""

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from latentslate_engine.authoring import compile_document, document_from_recipe
from latentslate_engine.ltx25.contracts import Ltx25Identity
from latentslate_engine.ltx25.recipes import (
    ltx25_recipe,
    resolve_ltx25_identity,
    resolve_ltx25_request,
    validate_enhancer_requirement,
)
from latentslate_engine.recipe import Artifact, exposed, fixed


def recipe(operation="t2v"):
    return ltx25_recipe(
        operation,
        diffusion="synthetic-diffusion",
        text_encoder="synthetic-text",
        video_vae="synthetic-video-vae",
        audio_vae="synthetic-audio-vae",
        prompt_enhancer="synthetic-enhancer",
        upsampler="synthetic-upscaler",
    )


class Ltx25RecipeTests(unittest.TestCase):
    def test_adapter_strength_is_part_of_worker_identity(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.bin"
            path.write_bytes(b"synthetic")
            original = recipe()
            fields = []
            for item in original.fields:
                key = item.capability.key
                if key == "transformer_adapter_artifacts":
                    item = fixed(item.capability, (Artifact(path),))
                elif key == "transformer_adapter_strengths":
                    item = exposed(item.capability, default=(1.0,))
                elif item.capability.value_type == "artifact":
                    item = fixed(item.capability, Artifact(path))
                fields.append(item)
            definition = replace(original, fields=tuple(fields))
            first = resolve_ltx25_identity(definition)
            zero = resolve_ltx25_identity(
                definition, {"transformer_adapter_strengths": (0.0,)}
            )
            self.assertNotEqual(first, zero)
            self.assertEqual(zero.transformer_adapters, ((str(path.resolve()), 0.0),))
            with self.assertRaises(ValueError):
                resolve_ltx25_identity(
                    definition, {"transformer_adapter_strengths": ()}
                )

    def test_authoring_round_trip_enforces_enhancer_requirement(self):
        for operation in ("t2v", "i2v", "flf"):
            document = document_from_recipe(
                recipe(operation), name="Synthetic recipe", recipe_id=str(uuid4())
            )
            compiled = compile_document(document)
            self.assertEqual(compiled.capabilities.key, f"ltx25.{operation}")
            for field in document["fields"]:
                if field["key"] == "prompt_enhancer":
                    field["value"] = None
            with self.assertRaisesRegex(ValueError, "requires its model"):
                compile_document(document)
            for field in document["fields"]:
                if field["key"] == "prompt_enhancement":
                    field["mode"] = "fixed"
                    field["value"] = False
            compile_document(document)

    def test_replaced_artifact_changes_identity(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.bin"
            path.write_bytes(b"first")
            paths = dict.fromkeys(
                (
                    "diffusion_path",
                    "text_encoder_path",
                    "video_vae_path",
                    "audio_vae_path",
                ),
                path,
            )
            first = Ltx25Identity.from_paths(**paths)
            self.assertEqual(first, Ltx25Identity.from_paths(**paths))
            path.write_bytes(b"replacement")
            self.assertNotEqual(first, Ltx25Identity.from_paths(**paths))

    def test_exposed_enhancement_requires_model_even_with_default_off(self):
        original = recipe()
        missing = replace(
            original,
            fields=tuple(
                fixed(item.capability, None)
                if item.capability.key == "prompt_enhancer"
                else item
                for item in original.fields
            ),
        )
        with self.assertRaisesRegex(ValueError, "requires its model"):
            resolve_ltx25_request(missing, {"prompt": "Synthetic prompt"})
        request = resolve_ltx25_request(original, {"prompt": "Synthetic prompt"})
        self.assertFalse(request["prompt_enhancement"])
        self.assertEqual((request["fps"], request["frame_count"]), (24, 121))

    def test_fixed_off_can_omit_model_but_fixed_on_cannot(self):
        original = recipe()
        for enabled in (False, True):
            changed = replace(
                original,
                fields=tuple(
                    fixed(item.capability, None)
                    if item.capability.key == "prompt_enhancer"
                    else fixed(item.capability, enabled)
                    if item.capability.key == "prompt_enhancement"
                    else item
                    for item in original.fields
                ),
            )
            if enabled:
                with self.assertRaises(ValueError):
                    validate_enhancer_requirement(changed)
            else:
                validate_enhancer_requirement(changed)
                with self.assertRaisesRegex(ValueError, "fixed"):
                    resolve_ltx25_request(
                        changed, {"prompt": "Synthetic", "prompt_enhancement": True}
                    )

    def test_flf_roles_and_frame_alignment(self):
        request = resolve_ltx25_request(
            recipe("flf"),
            {
                "prompt": "Synthetic",
                "start_image": "first.png",
                "end_image": "last.png",
                "duration_seconds": 2.5,
                "fps": 30,
            },
        )
        self.assertEqual(request["image_path"], "first.png")
        self.assertEqual(request["last_image_path"], "last.png")
        self.assertEqual(request["frame_count"], 73)
        with self.assertRaises(TypeError):
            resolve_ltx25_request(recipe(), {"prompt": "Synthetic", "fps": 23.976})


if __name__ == "__main__":
    unittest.main()
