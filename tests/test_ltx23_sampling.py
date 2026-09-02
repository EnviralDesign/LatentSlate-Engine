import unittest

import torch

from latentslate_engine.ltx23.sampling import euler_sample, euler_sample_masked


class _Patchifier:
    @staticmethod
    def patchify(value: torch.Tensor) -> tuple[torch.Tensor]:
        return (value,)


class _Model:
    patchifier = _Patchifier()
    a_patchifier = _Patchifier()

    @staticmethod
    def preprocess_text_embeds(
        condition: torch.Tensor, *, unprocessed: bool
    ) -> torch.Tensor:
        assert unprocessed
        return condition

    @staticmethod
    def __call__(
        streams: list[torch.Tensor], *_args: object, **_kwargs: object
    ) -> list[torch.Tensor]:
        return [torch.zeros_like(stream) for stream in streams]


class _Transformer:
    model = _Model()


class Ltx23SamplingTests(unittest.TestCase):
    def test_euler_returns_comfy_const_inverse_noise_scaling(self) -> None:
        latents = [torch.zeros((1, 1, 1)), torch.zeros((1, 1, 1))]
        noise = [torch.tensor([[[2.0]]]), torch.tensor([[[3.0]]])]

        result = euler_sample(
            _Transformer(),
            torch.zeros((1, 1, 1)),
            latents,
            noise,
            (1.0, 0.5),
            frame_rate=30,
        )

        torch.testing.assert_close(result[0], torch.tensor([[[4.0]]]))
        torch.testing.assert_close(result[1], torch.tensor([[[6.0]]]))

    def test_masked_euler_returns_comfy_const_inverse_noise_scaling(self) -> None:
        latents = [torch.zeros((1, 1, 1, 1, 1)), torch.zeros((1, 1, 1, 1))]
        noise = [torch.full_like(latents[0], 2.0), torch.full_like(latents[1], 3.0)]
        masks = [torch.ones_like(latents[0]), torch.ones_like(latents[1])]

        result = euler_sample_masked(
            _Transformer(),
            torch.zeros((1, 1, 1)),
            latents,
            noise,
            masks,
            (1.0, 0.5),
            frame_rate=30,
        )

        torch.testing.assert_close(result[0], torch.full_like(latents[0], 4.0))
        torch.testing.assert_close(result[1], torch.full_like(latents[1], 6.0))


if __name__ == "__main__":
    unittest.main()
