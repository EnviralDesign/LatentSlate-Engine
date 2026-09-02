import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from PIL import Image

from latentslate_engine.ltx23.i2v import (
    Ltx23I2VIdentity,
    Ltx23I2VRuntime,
    _conditioned_video_latent,
)


class _Closable:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _TextEncoder(_Closable):
    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []

    def encode(self, prompt: str) -> torch.Tensor:
        self.prompts.append(prompt)
        return torch.tensor([len(self.prompts)])


class _Transformer(_Closable):
    device_index = "cpu"


class _Upsampler(_Closable):
    def upsample(self, latents: torch.Tensor) -> torch.Tensor:
        return latents.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)


class _VideoDecoder(_Closable):
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        frames = latents.shape[2] * 8 - 7
        return torch.zeros((1, 3, frames, latents.shape[3] * 32, latents.shape[4] * 32))


class _AudioDecoder(_Closable):
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, 2, latents.shape[2] * 4 - 3, 64))


class _Vocoder(_Closable):
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, 2, mel.shape[3] * 480))


def identity() -> Ltx23I2VIdentity:
    return Ltx23I2VIdentity(
        checkpoint_path="checkpoint.safetensors",
        text_checkpoint_path="text.safetensors",
        transformer_lora_path="lora.safetensors",
        upsampler_path="upsampler.safetensors",
    )


class Ltx23I2VRuntimeTests(unittest.TestCase):
    def test_adapter_order_is_part_of_the_model_identity(self) -> None:
        runtime = Ltx23I2VRuntime(identity())
        transformer = _Closable()
        runtime._transformer = transformer

        replacement = runtime.replace_identity(
            replace(
                identity(),
                transformer_loras=(
                    ("first.safetensors", 0.5),
                    ("second.safetensors", 0.5),
                ),
            )
        )

        self.assertIsNot(replacement, runtime)
        self.assertEqual(transformer.close_calls, 1)

    def test_same_identity_retains_context_and_changed_identity_closes_it(self) -> None:
        runtime = Ltx23I2VRuntime(identity())
        text_encoder = _Closable()
        transformer = _Closable()
        vocoder = _Closable()
        runtime._text_encoder = text_encoder
        runtime._transformer = transformer
        runtime._vocoder = vocoder
        runtime._prompt_cache = ("prompt", torch.empty(0))
        runtime._source_cache = (
            b"source",
            512,
            512,
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
        )

        self.assertIs(runtime.replace_identity(identity()), runtime)
        self.assertEqual(text_encoder.close_calls, 0)
        self.assertEqual(transformer.close_calls, 0)
        self.assertEqual(vocoder.close_calls, 0)

        replacement = runtime.replace_identity(
            replace(identity(), checkpoint_path="other.safetensors")
        )
        self.assertIsNot(replacement, runtime)
        self.assertEqual(text_encoder.close_calls, 1)
        self.assertEqual(transformer.close_calls, 1)
        self.assertEqual(vocoder.close_calls, 1)
        self.assertIsNone(runtime._text_encoder)
        self.assertIsNone(runtime._transformer)
        self.assertIsNone(runtime._vocoder)
        self.assertIsNone(runtime._prompt_cache)
        self.assertIsNone(runtime._source_cache)

    def test_prompt_cache_is_independent_from_source_cache(self) -> None:
        runtime = Ltx23I2VRuntime(identity())
        text_encoder = _TextEncoder()
        runtime._text_encoder = text_encoder
        source_cache = (
            b"source",
            512,
            512,
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
        )
        runtime._source_cache = source_cache

        first = runtime._encode_prompt("first")
        self.assertIs(runtime._encode_prompt("first"), first)
        self.assertIsNot(runtime._encode_prompt("second"), first)
        self.assertEqual(text_encoder.prompts, ["first", "second"])
        self.assertIs(runtime._source_cache, source_cache)

    def test_source_cache_uses_content_identity_not_path(self) -> None:
        runtime = Ltx23I2VRuntime(identity())
        preprocess_calls: list[tuple[Path, int, int]] = []
        encode_sizes: list[tuple[int, int]] = []

        class _Encoder:
            def __init__(self, _checkpoint: str) -> None:
                pass

            def encode(self, source: torch.Tensor) -> torch.Tensor:
                encode_sizes.append(tuple(source.shape[-2:]))
                return torch.full((1, 128, 1, 1, 1), float(source.shape[-1]))

            def close(self) -> None:
                pass

        def preprocess(path: str | Path, width: int, height: int) -> torch.Tensor:
            preprocess_calls.append((Path(path), width, height))
            return torch.zeros((1, 3, height, width))

        with TemporaryDirectory() as directory:
            first = Path(directory, "first.png")
            alias = Path(directory, "alias.png")
            changed = Path(directory, "changed.png")
            first.write_bytes(b"same")
            alias.write_bytes(b"same")
            changed.write_bytes(b"changed")
            with (
                patch("latentslate_engine.ltx23.i2v.Ltx23VideoEncoder", _Encoder),
                patch(
                    "latentslate_engine.ltx23.i2v._preprocess_source_image",
                    preprocess,
                ),
            ):
                low, _ = runtime._encode_source(first, 512, 512)
                alias_low, _ = runtime._encode_source(alias, 512, 512)
                runtime._encode_source(changed, 512, 512)
                runtime._encode_source(first, 768, 512)

        self.assertIs(alias_low, low)
        self.assertEqual(len(preprocess_calls), 3)
        self.assertEqual(
            encode_sizes,
            [
                (256, 256),
                (512, 512),
                (256, 256),
                (512, 512),
                (256, 384),
                (512, 768),
            ],
        )

    def test_normalized_source_dimensions_must_match_request(self) -> None:
        runtime = Ltx23I2VRuntime(identity())
        with TemporaryDirectory() as directory:
            source = Path(directory, "source.png")
            Image.new("RGB", (64, 64)).save(source)
            with (
                patch.object(runtime, "_encode_prompt") as encode_prompt,
                self.assertRaisesRegex(ValueError, "must be 128x64"),
            ):
                runtime.generate("prompt", source, width=128, height=64)
        encode_prompt.assert_not_called()

    def test_product_image_latents_and_masks(self) -> None:
        low, low_mask = _conditioned_video_latent(
            torch.ones((1, 128, 1, 8, 12)),
            768,
            512,
            6,
            0.7,
            "cpu",
        )
        self.assertEqual(tuple(low.shape), (1, 128, 6, 8, 12))
        self.assertTrue(torch.equal(low[:, :, :1], torch.ones_like(low[:, :, :1])))
        self.assertEqual(torch.count_nonzero(low[:, :, 1:]).item(), 0)
        self.assertTrue(
            torch.allclose(low_mask[:, :, :1], torch.full_like(low_mask[:, :, :1], 0.3))
        )
        self.assertTrue(
            torch.equal(low_mask[:, :, 1:], torch.ones_like(low_mask[:, :, 1:]))
        )

        canonical, canonical_mask = _conditioned_video_latent(
            torch.ones((1, 128, 1, 8, 8)),
            512,
            512,
            19,
            1.0,
            "cpu",
        )
        self.assertEqual(tuple(canonical.shape), (1, 128, 19, 8, 8))
        self.assertEqual(torch.count_nonzero(canonical_mask[:, :, :1]).item(), 0)
        self.assertTrue(
            torch.equal(
                canonical_mask[:, :, 1:],
                torch.ones_like(canonical_mask[:, :, 1:]),
            )
        )

    def test_public_seed_controls_coarse_pass_and_refinement_stays_fixed(self) -> None:
        runtime = Ltx23I2VRuntime(identity())
        runtime._transformer = _Transformer()
        runtime._prompt_cache = ("prompt", torch.zeros((1, 1)))
        runtime._vocoder = _Vocoder()
        low = torch.zeros((1, 128, 1, 1, 1))
        full = torch.zeros((1, 128, 1, 2, 2))
        observed_seeds: list[int] = []

        def fake_noise(seed: int, latents: list[torch.Tensor]) -> list[torch.Tensor]:
            observed_seeds.append(seed)
            return [torch.zeros_like(latent) for latent in latents]

        with (
            patch.object(runtime, "_encode_source", return_value=(low, full)),
            patch("latentslate_engine.ltx23.i2v.nested_noise", side_effect=fake_noise),
            patch(
                "latentslate_engine.ltx23.i2v.euler_sample_masked",
                side_effect=lambda _model, _condition, latents, *_args, **_kwargs: (
                    latents
                ),
            ),
            patch(
                "latentslate_engine.ltx23.i2v.Ltx23SpatialUpsampler",
                return_value=_Upsampler(),
            ),
            patch(
                "latentslate_engine.ltx23.i2v.Ltx23VideoDecoder",
                return_value=_VideoDecoder(),
            ),
            patch(
                "latentslate_engine.ltx23.i2v.Ltx23AudioMelDecoder",
                return_value=_AudioDecoder(),
            ),
        ):
            result = runtime.generate(
                "prompt",
                "normalized.png",
                width=64,
                height=64,
                duration_seconds=1.0,
                seed=777,
            )

        self.assertEqual(observed_seeds, [777, 42])
        self.assertEqual(tuple(result.frames.shape), (1, 25, 64, 64, 3))


if __name__ == "__main__":
    unittest.main()
