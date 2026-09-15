import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from latentslate_engine.ltx23.lora import Ltx23TransformerLora, Ltx23TransformerLoras
from latentslate_engine.ltx23.sampling import (
    MAX_SEED,
    empty_av_latents,
    ltx_temporal_shapes,
    validate_ltx_request,
)
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


class _Lora:
    def __init__(self, multiplier: float, offset: float) -> None:
        self.multiplier = multiplier
        self.offset = offset

    def has_weight(self, prefix: str) -> bool:
        return prefix == "layer"

    def apply(
        self,
        prefix: str,
        weight: torch.Tensor,
        staged: object = None,
        disposable_weight: bool = False,
    ) -> torch.Tensor:
        del prefix, staged
        if disposable_weight:
            return weight.mul_(self.multiplier).add_(self.offset)
        return weight * self.multiplier + self.offset


def identity() -> Ltx23T2VIdentity:
    return Ltx23T2VIdentity(
        checkpoint_path="checkpoint.safetensors",
        text_checkpoint_path="text.safetensors",
        transformer_lora_path="lora.safetensors",
        upsampler_path="upsampler.safetensors",
    )


class Ltx23T2VRuntimeTests(unittest.TestCase):
    def test_adapter_order_is_part_of_the_model_identity(self) -> None:
        runtime = Ltx23T2VRuntime(identity())
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

    def test_multi_lora_applies_in_identity_order(self) -> None:
        loras = object.__new__(Ltx23TransformerLoras)
        loras.loras = (_Lora(2.0, 1.0), _Lora(3.0, 4.0))

        weight = torch.zeros(1)
        self.assertTrue(torch.equal(loras.apply("layer", weight), torch.tensor([7.0])))
        self.assertTrue(torch.equal(weight, torch.zeros(1)))

        disposable = torch.zeros(1)
        self.assertIs(loras.apply("layer", disposable, disposable_weight=True), disposable)
        self.assertTrue(torch.equal(disposable, torch.tensor([7.0])))

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

    def test_media_writer_rejects_invalid_frame_shape_before_writing(self) -> None:
        output = Ltx23T2VOutput(
            frames=torch.empty((1, 144, 32, 512, 3)),
            waveform=torch.empty((1, 2, 1)),
        )

        with self.assertRaisesRegex(ValueError, "at least 64"):
            output.save_mp4("should-not-be-created.mp4")

    def test_request_validation_accepts_product_domain_boundaries(self) -> None:
        for width, height, duration, seed in (
            (64, 64, 1.0, 0),
            (512, 512, 5.0, 810138461690240),
            (1280, 704, 10.0, MAX_SEED),
            (704, 1280, 1.5, 1),
            (14720, 64, 5.0, 1),
        ):
            validate_ltx_request(width, height, duration, seed, alignment=64)

    def test_request_validation_rejects_illegal_values_without_snapping(self) -> None:
        invalid = (
            (63, 64, 5.0, 0, "at least 64"),
            (96, 64, 5.0, 0, "divisible by 64"),
            (14784, 64, 5.0, 0, "must not exceed"),
            (512, 512, 0.5, 0, "between 1.0 and 10.0"),
            (512, 512, 10.5, 0, "between 1.0 and 10.0"),
            (512, 512, 5.0, -1, "between 0"),
            (512, 512, 5.0, MAX_SEED + 1, "between 0"),
        )
        for width, height, duration, seed, message in invalid:
            with (
                self.subTest(width=width, height=height, duration=duration, seed=seed),
                self.assertRaisesRegex(ValueError, message),
            ):
                validate_ltx_request(width, height, duration, seed, alignment=64)

    def test_canonical_and_nonsquare_latent_sizes(self) -> None:
        self.assertEqual(
            tuple(
                empty_av_latents(512, 512, 5.0, spatial_divisor=64, device="cpu")[
                    0
                ].shape
            ),
            (1, 128, 20, 8, 8),
        )
        self.assertEqual(
            tuple(
                empty_av_latents(768, 768, 5.0, spatial_divisor=64, device="cpu")[
                    0
                ].shape
            ),
            (1, 128, 20, 12, 12),
        )
        landscape = empty_av_latents(1280, 704, 5.0, spatial_divisor=64, device="cpu")
        portrait = empty_av_latents(704, 1280, 5.0, spatial_divisor=64, device="cpu")
        self.assertEqual(tuple(landscape[0].shape), (1, 128, 20, 11, 20))
        self.assertEqual(tuple(portrait[0].shape), (1, 128, 20, 20, 11))
        self.assertEqual(tuple(landscape[1].shape), (1, 8, 127, 16))

    def test_duration_snaps_to_nearest_legal_temporal_shapes(self) -> None:
        expected = {
            1.0: (33, 5, 33, 28),
            1.5: (49, 7, 49, 41),
            5.0: (153, 20, 153, 127),
            10.0: (297, 38, 297, 248),
        }
        for duration, shapes in expected.items():
            self.assertEqual(ltx_temporal_shapes(duration), shapes)
            latents = empty_av_latents(
                512, 512, duration, spatial_divisor=64, device="cpu"
            )
            self.assertEqual(latents[0].shape[2], shapes[1])
            self.assertEqual(latents[1].shape[2], shapes[3])

    def test_integer_fps_and_duration_choose_nearest_legal_frame_count(self) -> None:
        for fps in (1, 12, 24, 25, 30, 48, 60, 120):
            legal = [n for n in range(1, 1201, 8) if 1 <= n / fps <= 10]
            for duration in (1.0, 1.25, 4.91, 5.0, 9.99, 10.0):
                validate_ltx_request(512, 512, duration, 0, alignment=64, fps=fps)
                requested, latent, decoded, audio = ltx_temporal_shapes(duration, fps)
                best = min(legal, key=lambda n: (abs(n / fps - duration), -n))
                self.assertEqual(decoded, best)
                self.assertEqual(requested, decoded)
                self.assertEqual(latent * 8 - 7, decoded)
                self.assertEqual(audio, round(decoded / fps * 25))

    def test_fractional_fps_is_rejected(self) -> None:
        for fps in (23.976, 29.97, 59.94):
            with self.assertRaisesRegex(ValueError, "whole number"):
                validate_ltx_request(512, 512, 5.0, 0, alignment=64, fps=fps)

    def test_flf_alignment_is_the_only_geometry_policy_difference(self) -> None:
        validate_ltx_request(96, 64, 5.0, 0, alignment=32)
        with self.assertRaisesRegex(ValueError, "divisible by 64"):
            validate_ltx_request(96, 64, 5.0, 0, alignment=64)

    def test_public_seed_controls_coarse_pass_and_refinement_stays_fixed(self) -> None:
        runtime = Ltx23T2VRuntime(identity())
        runtime._transformer = _Transformer()
        runtime._prompt_cache = ("prompt", torch.zeros((1, 1)))
        observed_seeds: list[int] = []

        def fake_noise(seed: int, latents: list[torch.Tensor]) -> list[torch.Tensor]:
            observed_seeds.append(seed)
            return [torch.zeros_like(latent) for latent in latents]

        with (
            patch("latentslate_engine.ltx23.t2v.nested_noise", side_effect=fake_noise),
            patch(
                "latentslate_engine.ltx23.t2v.euler_sample",
                side_effect=lambda _model, _condition, latents, *_args, **_kwargs: (
                    latents
                ),
            ),
            patch(
                "latentslate_engine.ltx23.t2v.Ltx23SpatialUpsampler",
                return_value=_Upsampler(),
            ),
            patch(
                "latentslate_engine.ltx23.t2v.Ltx23VideoDecoder",
                return_value=_VideoDecoder(),
            ),
            patch(
                "latentslate_engine.ltx23.t2v.Ltx23AudioMelDecoder",
                return_value=_AudioDecoder(),
            ),
            patch(
                "latentslate_engine.ltx23.t2v.Ltx23AudioVocoder",
                return_value=_Vocoder(),
            ),
        ):
            result = runtime.generate(
                "prompt", width=64, height=64, duration_seconds=1.0, seed=1234
            )

        self.assertEqual(observed_seeds, [1234, 42])
        self.assertEqual(tuple(result.frames.shape), (1, 33, 64, 64, 3))

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


def test_integer_fps_mux_keeps_audio_and_video_duration(tmp_path):
    import av
    import math

    for fps in (24, 25, 60):
        frames = ltx_temporal_shapes(1.0, fps)[2]
        duration = frames / fps
        # The audio latent grid can end slightly before or after the video.
        for tail in (-960, 960):
            output = Ltx23T2VOutput(
                frames=torch.zeros((1, frames, 64, 64, 3)),
                waveform=torch.zeros((1, 2, math.ceil(duration * 48000) + tail)),
                frame_rate=fps,
            )
            destination = tmp_path / f"{fps}-{tail}.mp4"
            output.save_mp4(destination)
            with av.open(str(destination)) as container:
                video, audio = container.streams.video[0], container.streams.audio[0]
                assert abs(float(video.average_rate) - fps) < 0.00001
                assert video.frames == frames
                assert abs(float(video.duration * video.time_base) - duration) < 0.001
                assert abs(float(audio.duration * audio.time_base) - duration) < 0.002
