from dataclasses import replace
import unittest

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
        runtime._text_encoder = text_encoder
        runtime._transformer = transformer
        runtime._prompt_cache = ("prompt", torch.empty(0))
        runtime._source_cache = (b"source", torch.empty(0), torch.empty(0))

        self.assertIs(runtime.replace_identity(identity()), runtime)
        self.assertEqual(text_encoder.close_calls, 0)
        self.assertEqual(transformer.close_calls, 0)

        replacement = runtime.replace_identity(
            replace(identity(), checkpoint_path="other.safetensors")
        )
        self.assertIsNot(replacement, runtime)
        self.assertEqual(text_encoder.close_calls, 1)
        self.assertEqual(transformer.close_calls, 1)
        self.assertIsNone(runtime._text_encoder)
        self.assertIsNone(runtime._transformer)
        self.assertIsNone(runtime._prompt_cache)
        self.assertIsNone(runtime._source_cache)

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
        self.assertTrue(torch.equal(low_mask[:, :, 1:], torch.ones_like(low_mask[:, :, 1:])))

        full, full_mask = _conditioned_video_latent(
            torch.ones((1, 128, 1, 16, 16)), 512, 1.0, "cpu"
        )
        self.assertEqual(tuple(full.shape), (1, 128, 19, 16, 16))
        self.assertEqual(torch.count_nonzero(full_mask[:, :, :1]).item(), 0)
        self.assertTrue(torch.equal(full_mask[:, :, 1:], torch.ones_like(full_mask[:, :, 1:])))


if __name__ == "__main__":
    unittest.main()
