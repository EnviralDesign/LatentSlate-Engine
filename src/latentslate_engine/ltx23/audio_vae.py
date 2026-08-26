"""Canonical LTX 2.3 audio-latent decoder.

The installed Diffusers LTX2 audio autoencoder has the same decoder tensor
layout and numerical behavior as the pinned Comfy implementation.  This module
uses that decoder directly and retains only the fixture's normalization seam.
"""

from __future__ import annotations

import json

import torch
from diffusers import AutoencoderKLLTX2Audio

from .checkpoint import Ltx23Checkpoint


_CANONICAL_AUDIO_SHAPE = (1, 8, 126, 16)


class Ltx23AudioMelDecoder:
    """Decode the fixture's normalized audio latent into 16 kHz stereo mel bins."""

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        checkpoint = Ltx23Checkpoint(checkpoint_path)
        config = json.loads(checkpoint.metadata["config"])["audio_vae"]
        params = config["model"]["params"]
        decoder_config = params["ddconfig"]

        with torch.device("meta"):
            autoencoder = AutoencoderKLLTX2Audio(
                base_channels=decoder_config["ch"],
                output_channels=decoder_config["out_ch"],
                ch_mult=tuple(decoder_config["ch_mult"]),
                num_res_blocks=decoder_config["num_res_blocks"],
                attn_resolutions=tuple(decoder_config["attn_resolutions"]),
                in_channels=decoder_config["in_channels"],
                resolution=decoder_config["resolution"],
                latent_channels=decoder_config["z_channels"],
                norm_type=decoder_config["norm_type"],
                causality_axis=decoder_config["causality_axis"],
                dropout=decoder_config["dropout"],
                mid_block_add_attention=decoder_config["mid_block_add_attention"],
                sample_rate=params["sampling_rate"],
                mel_hop_length=config["preprocessing"]["stft"]["hop_length"],
                is_causal=True,
                mel_bins=decoder_config["mel_bins"],
                double_z=decoder_config["double_z"],
            )

        state = {
            name.removeprefix("audio_vae."): checkpoint.tensor(name)
            for name in checkpoint.tensor_names
            if name.startswith("audio_vae.")
        }
        state["latents_mean"] = state.pop("per_channel_statistics.mean-of-means")
        state["latents_std"] = state.pop("per_channel_statistics.std-of-means")
        incompatible = autoencoder.load_state_dict(state, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys or len(state) != 102:
            raise ValueError("unexpected pinned LTX 2.3 audio VAE state")

        # T2V never encodes user audio.  Retaining only the validated decoder
        # avoids pinning the unused encoder in this operation-local runtime.
        self.decoder = autoencoder.decoder.to(device=device, dtype=torch.bfloat16).eval()
        self._mean = autoencoder.latents_mean.to(device=device, dtype=torch.bfloat16).view(1, 8, 1, 16)
        self._std = autoencoder.latents_std.to(device=device, dtype=torch.bfloat16).view(1, 8, 1, 16)

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if tuple(latents.shape) != _CANONICAL_AUDIO_SHAPE:
            raise ValueError("the canonical T2V audio decoder expects [1, 8, 126, 16]")
        x = latents.to(device=self._mean.device, dtype=torch.bfloat16)
        return self.decoder(x * self._std + self._mean)

    def close(self) -> None:
        self.decoder = None
        self._mean = None
        self._std = None
        torch.cuda.empty_cache()
