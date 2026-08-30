import unittest
from dataclasses import replace

import torch

from latentslate_engine.ltx23.lora import Ltx23TransformerLora
from latentslate_engine.ltx23.sampling import canonical_empty_latents
from latentslate_engine.ltx23.t2v import (
    Ltx23T2VIdentity,
    Ltx23T2VOutput,
    Ltx23T2VRuntime,
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
        runtime._prompt_cache = ("prompt", torch.empty(0))

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

    def test_same_prompt_reuses_conditioning_and_changed_prompt_recomputes(
        self,
    ) -> None:
        runtime = Ltx23T2VRuntime(identity())
        text_encoder = _TextEncoder()
        runtime._text_encoder = text_encoder

        first = runtime._encode_prompt("first")
        self.assertIs(runtime._encode_prompt("first"), first)
        second = runtime._encode_prompt("second")

        self.assertEqual(text_encoder.prompts, ["first", "second"])
        self.assertIsNot(second, first)

    def test_media_writer_rejects_noncanonical_frame_shape_before_writing(self) -> None:
        output = Ltx23T2VOutput(
            frames=torch.empty_strided((1, 144, 512, 512, 3), (0, 0, 0, 0, 0)),
            waveform=torch.empty((1, 2, 1)),
        )

        with self.assertRaisesRegex(ValueError, "512x512 or 768x768"):
            output.save_mp4("should-not-be-created.mp4")

    def test_canonical_gate_latent_sizes(self) -> None:
        self.assertEqual(
            tuple(canonical_empty_latents("cpu", 512)[0].shape), (1, 128, 19, 8, 8)
        )
        self.assertEqual(
            tuple(canonical_empty_latents("cpu", 768)[0].shape), (1, 128, 19, 12, 12)
        )

    def test_lora_only_mutates_disposable_weights(self) -> None:
        lora = object.__new__(Ltx23TransformerLora)
        lora.strength = 0.5
        lora._names = frozenset({"layer.lora_A.weight"})
        down = torch.tensor([[1.0, 2.0]])
        up = torch.tensor([[3.0], [4.0]])

        persistent = torch.ones((2, 2))
        merged = lora.apply("model.layer", persistent, (down, up))
        self.assertTrue(torch.equal(persistent, torch.ones((2, 2))))
        self.assertIsNot(merged, persistent)

        disposable = torch.ones((2, 2))
        merged = lora.apply(
            "model.layer", disposable, (down, up), disposable_weight=True
        )
        self.assertIs(merged, disposable)
        self.assertFalse(torch.equal(disposable, torch.ones((2, 2))))


if __name__ == "__main__":
    unittest.main()
