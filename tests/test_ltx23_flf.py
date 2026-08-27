import unittest
from dataclasses import replace

import torch

from latentslate_engine.ltx23.flf import (
    Ltx23FlfIdentity,
    Ltx23FlfRuntime,
    _guided_video_latent,
)


class _Closable:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def identity() -> Ltx23FlfIdentity:
    return Ltx23FlfIdentity(
        checkpoint_path="checkpoint.safetensors",
        text_checkpoint_path="text.safetensors",
    )


class Ltx23FlfRuntimeTests(unittest.TestCase):
    def test_same_identity_retains_context_and_changed_identity_purges_it(self) -> None:
        runtime = Ltx23FlfRuntime(identity())
        text_encoder = _Closable()
        transformer = _Closable()
        runtime._text_encoder = text_encoder
        runtime._transformer = transformer
        runtime._prompt_cache = ("prompt", torch.empty(0))
        runtime._guide_cache = (b"first", b"last", torch.empty(0), torch.empty(0))

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
        self.assertIsNone(runtime._guide_cache)

    def test_canonical_guides_are_appended_masked_and_temporally_placed(self) -> None:
        first = torch.full((1, 128, 1, 16, 16), 1.0)
        last = torch.full((1, 128, 1, 16, 16), 2.0)
        latent, mask, keyframes, entries = _guided_video_latent(first, last, "cpu")

        self.assertEqual(tuple(latent.shape), (1, 128, 21, 16, 16))
        self.assertEqual(torch.count_nonzero(latent[:, :, :19]).item(), 0)
        self.assertTrue(torch.equal(latent[:, :, 19:20], first))
        self.assertTrue(torch.equal(latent[:, :, 20:21], last))
        self.assertTrue(torch.equal(mask[:, :, :19], torch.ones_like(mask[:, :, :19])))
        self.assertTrue(
            torch.allclose(mask[:, :, 19:], torch.full_like(mask[:, :, 19:], 0.3))
        )
        self.assertEqual(tuple(keyframes.shape), (1, 3, 512, 2))
        self.assertEqual(torch.unique(keyframes[:, 0, :256, 0]).tolist(), [0])
        self.assertEqual(torch.unique(keyframes[:, 0, :256, 1]).tolist(), [1])
        self.assertEqual(torch.unique(keyframes[:, 0, 256:, 0]).tolist(), [144])
        self.assertEqual(torch.unique(keyframes[:, 0, 256:, 1]).tolist(), [145])
        self.assertEqual([entry["strength"] for entry in entries], [0.7, 0.7])
        self.assertEqual([entry["pre_filter_count"] for entry in entries], [256, 256])


if __name__ == "__main__":
    unittest.main()
