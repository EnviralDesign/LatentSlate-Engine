import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

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


def identity() -> Ltx23I2VIdentity:
    return Ltx23I2VIdentity(
        checkpoint_path="checkpoint.safetensors",
        text_checkpoint_path="text.safetensors",
        transformer_lora_path="lora.safetensors",
        upsampler_path="upsampler.safetensors",
    )


class Ltx23I2VRuntimeTests(unittest.TestCase):
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
        preprocess_calls: list[Path] = []
        encode_sizes: list[int] = []

        class _Encoder:
            def __init__(self, _checkpoint: str) -> None:
                pass

            def encode(self, source: torch.Tensor) -> torch.Tensor:
                encode_sizes.append(source.shape[-1])
                return torch.full((1, 128, 1, 1, 1), float(source.shape[-1]))

            def close(self) -> None:
                pass

        def preprocess(path: str | Path) -> torch.Tensor:
            preprocess_calls.append(Path(path))
            return torch.zeros((1, 3, 512, 512))

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
                low, _ = runtime._encode_source(first)
                alias_low, _ = runtime._encode_source(alias)
                runtime._encode_source(changed)

        self.assertIs(alias_low, low)
        self.assertEqual(len(preprocess_calls), 2)
        self.assertEqual(encode_sizes, [256, 512, 256, 512])

    def test_canonical_image_latents_and_masks(self) -> None:
        low, low_mask = _conditioned_video_latent(
            torch.ones((1, 128, 1, 8, 8)), 256, 0.7, "cpu"
        )
        self.assertEqual(tuple(low.shape), (1, 128, 19, 8, 8))
        self.assertTrue(torch.equal(low[:, :, :1], torch.ones_like(low[:, :, :1])))
        self.assertEqual(torch.count_nonzero(low[:, :, 1:]).item(), 0)
        self.assertTrue(
            torch.allclose(low_mask[:, :, :1], torch.full_like(low_mask[:, :, :1], 0.3))
        )
        self.assertTrue(
            torch.equal(low_mask[:, :, 1:], torch.ones_like(low_mask[:, :, 1:]))
        )

        full, full_mask = _conditioned_video_latent(
            torch.ones((1, 128, 1, 16, 16)), 512, 1.0, "cpu"
        )
        self.assertEqual(tuple(full.shape), (1, 128, 19, 16, 16))
        self.assertEqual(torch.count_nonzero(full_mask[:, :, :1]).item(), 0)
        self.assertTrue(
            torch.equal(full_mask[:, :, 1:], torch.ones_like(full_mask[:, :, 1:]))
        )


if __name__ == "__main__":
    unittest.main()
