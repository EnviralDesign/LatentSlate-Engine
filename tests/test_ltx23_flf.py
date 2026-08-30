import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from latentslate_engine.ltx23.flf import (
    Ltx23FlfIdentity,
    Ltx23FlfOutput,
    Ltx23FlfRuntime,
    _guided_video_latent,
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
        video_decoder = _Closable()
        audio_decoder = _Closable()
        vocoder = _Closable()
        runtime._text_encoder = text_encoder
        runtime._transformer = transformer
        runtime._video_decoder = video_decoder
        runtime._audio_decoder = audio_decoder
        runtime._vocoder = vocoder
        runtime._prompt_cache = ("prompt", torch.empty(0))
        runtime._guide_cache = (b"first", b"last", torch.empty(0), torch.empty(0))

        self.assertIs(runtime.replace_identity(identity()), runtime)
        self.assertEqual(text_encoder.close_calls, 0)
        self.assertEqual(transformer.close_calls, 0)
        self.assertEqual(video_decoder.close_calls, 0)
        self.assertEqual(audio_decoder.close_calls, 0)
        self.assertEqual(vocoder.close_calls, 0)

        replacement = runtime.replace_identity(
            replace(identity(), checkpoint_path="other.safetensors")
        )
        self.assertIsNot(replacement, runtime)
        self.assertEqual(text_encoder.close_calls, 1)
        self.assertEqual(transformer.close_calls, 1)
        self.assertEqual(video_decoder.close_calls, 1)
        self.assertEqual(audio_decoder.close_calls, 1)
        self.assertEqual(vocoder.close_calls, 1)
        self.assertIsNone(runtime._text_encoder)
        self.assertIsNone(runtime._transformer)
        self.assertIsNone(runtime._video_decoder)
        self.assertIsNone(runtime._audio_decoder)
        self.assertIsNone(runtime._vocoder)
        self.assertIsNone(runtime._prompt_cache)
        self.assertIsNone(runtime._guide_cache)

    def test_prompt_cache_does_not_invalidate_guides(self) -> None:
        runtime = Ltx23FlfRuntime(identity())
        text_encoder = _TextEncoder()
        runtime._text_encoder = text_encoder
        guide_cache = (b"first", b"last", torch.empty(0), torch.empty(0))
        runtime._guide_cache = guide_cache

        first = runtime._encode_prompt("first")
        self.assertIs(runtime._encode_prompt("first"), first)
        self.assertIsNot(runtime._encode_prompt("second"), first)
        self.assertEqual(text_encoder.prompts, ["first", "second"])
        self.assertIs(runtime._guide_cache, guide_cache)

    def test_guide_cache_is_content_derived_and_role_ordered(self) -> None:
        runtime = Ltx23FlfRuntime(identity())
        preprocess_calls: list[Path] = []

        class _Encoder:
            def __init__(self, _checkpoint: str) -> None:
                pass

            def encode(self, source: torch.Tensor) -> torch.Tensor:
                return source

            def close(self) -> None:
                pass

        def preprocess(path: str | Path) -> torch.Tensor:
            preprocess_calls.append(Path(path))
            return torch.tensor([float(len(preprocess_calls))])

        with TemporaryDirectory() as directory:
            first = Path(directory, "first.png")
            first_alias = Path(directory, "first-alias.png")
            last = Path(directory, "last.png")
            last_alias = Path(directory, "last-alias.png")
            first.write_bytes(b"first")
            first_alias.write_bytes(b"first")
            last.write_bytes(b"last")
            last_alias.write_bytes(b"last")
            with (
                patch("latentslate_engine.ltx23.flf.Ltx23VideoEncoder", _Encoder),
                patch("latentslate_engine.ltx23.flf._preprocess_guide", preprocess),
            ):
                original = runtime._encode_guides(first, last)
                aliases = runtime._encode_guides(first_alias, last_alias)
                swapped = runtime._encode_guides(last, first)

        self.assertIs(aliases[0], original[0])
        self.assertIs(aliases[1], original[1])
        self.assertEqual(len(preprocess_calls), 4)
        self.assertFalse(torch.equal(swapped[0], original[0]))

    def test_flf_writer_rejects_noncanonical_media(self) -> None:
        output = Ltx23FlfOutput(
            frames=torch.empty((1, 1, 1, 1, 3)),
            waveform=torch.empty((1, 2, 1)),
        )
        with self.assertRaisesRegex(ValueError, "512x512"):
            output.save_mp4("unused.mp4")

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
