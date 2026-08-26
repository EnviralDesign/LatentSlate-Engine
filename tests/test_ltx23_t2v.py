from dataclasses import replace
import unittest

import torch

from latentslate_engine.ltx23.t2v import Ltx23T2VIdentity, Ltx23T2VOutput, Ltx23T2VRuntime


class _Closable:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def identity() -> Ltx23T2VIdentity:
    return Ltx23T2VIdentity(
        checkpoint_path="checkpoint.safetensors",
        text_checkpoint_path="text.safetensors",
        transformer_lora_path="lora.safetensors",
        upsampler_path="upsampler.safetensors",
    )


class Ltx23T2VRuntimeTests(unittest.TestCase):
    def test_same_identity_retains_context_and_changed_identity_closes_it(self) -> None:
        runtime = Ltx23T2VRuntime(identity())
        text_encoder = _Closable()
        transformer = _Closable()
        runtime._text_encoder = text_encoder
        runtime._transformer = transformer

        self.assertIs(runtime.replace_identity(identity()), runtime)
        self.assertEqual(text_encoder.close_calls, 0)
        self.assertEqual(transformer.close_calls, 0)

        replacement = runtime.replace_identity(replace(identity(), checkpoint_path="other.safetensors"))
        self.assertIsNot(replacement, runtime)
        self.assertEqual(text_encoder.close_calls, 1)
        self.assertEqual(transformer.close_calls, 1)
        self.assertIsNone(runtime._text_encoder)
        self.assertIsNone(runtime._transformer)

    def test_media_writer_rejects_noncanonical_frame_shape_before_writing(self) -> None:
        output = Ltx23T2VOutput(
            frames=torch.empty_strided((1, 144, 512, 512, 3), (0, 0, 0, 0, 0)),
            waveform=torch.empty((1, 2, 1)),
        )

        with self.assertRaisesRegex(ValueError, r"\[1, 145, 512, 512, 3\]"):
            output.save_mp4("should-not-be-created.mp4")


if __name__ == "__main__":
    unittest.main()
