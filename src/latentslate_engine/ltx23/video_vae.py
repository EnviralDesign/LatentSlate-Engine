"""Fixture-specific LTX 2.3 causal video VAE paths.

This is a narrow adaptation of the pinned Comfy LTX causal VAE.  It keeps the
source decoder's temporal cache and recursive between-block chunking, and the
single-frame encoder path required by canonical I2V, without importing ComfyUI.
"""

from __future__ import annotations

import math
import threading
from typing import Optional

import torch
from torch import nn

from .checkpoint import Ltx23Checkpoint


_MIN_VRAM_FOR_CHUNK_SCALING = 6 * 1024**3
_MAX_VRAM_FOR_CHUNK_SCALING = 24 * 1024**3
_MIN_CHUNK_SIZE = 32 * 1024**2
_MAX_CHUNK_SIZE = 128 * 1024**2
_DECODER_BLOCKS = (
    ("res_x", {"num_layers": 4}),
    ("compress_space", {"multiplier": 2}),
    ("res_x", {"num_layers": 6}),
    ("compress_time", {"multiplier": 2}),
    ("res_x", {"num_layers": 4}),
    ("compress_all", {"multiplier": 1}),
    ("res_x", {"num_layers": 2}),
    ("compress_all", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
)
_ENCODER_BLOCKS = (
    ("res_x", {"num_layers": 4}),
    ("compress_space_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 6}),
    ("compress_time_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 4}),
    ("compress_all_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
    ("compress_all_res", {"multiplier": 1}),
    ("res_x", {"num_layers": 2}),
)


def _cat_if_needed(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=dim)


def _split2(tensor: torch.Tensor, split_point: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.split(tensor, [split_point, tensor.shape[dim] - split_point], dim=dim)


def _add_exchange_cache(
    destination: Optional[torch.Tensor],
    cache_in: Optional[torch.Tensor],
    new_input: torch.Tensor,
    dim: int = 2,
) -> Optional[torch.Tensor]:
    if destination is not None:
        if cache_in is not None:
            cache_to_destination = min(destination.shape[dim], cache_in.shape[dim])
            lead_destination, destination = _split2(destination, cache_to_destination, dim=dim)
            lead_source, cache_in = _split2(cache_in, cache_to_destination, dim=dim)
            lead_destination.add_(lead_source)
        body, new_input = _split2(new_input, destination.shape[dim], dim)
        destination.add_(body)
    if cache_in is None:
        return new_input
    return _cat_if_needed([cache_in, new_input], dim=dim)


def _mark_conv3d_ended(module: nn.Module) -> None:
    thread_id = threading.get_ident()
    for nested in module.modules():
        if isinstance(nested, _CausalConv3d):
            cached, _ = nested.temporal_cache_state.get(thread_id, (None, False))
            nested.temporal_cache_state[thread_id] = (cached, True)


def _clear_temporal_cache(module: nn.Module) -> None:
    thread_id = threading.get_ident()
    for nested in module.modules():
        if hasattr(nested, "temporal_cache_state"):
            nested.temporal_cache_state.pop(thread_id, None)


def _get_max_chunk_size(device: torch.device) -> int:
    total_memory = torch.cuda.get_device_properties(device).total_memory
    if total_memory <= _MIN_VRAM_FOR_CHUNK_SCALING:
        return _MIN_CHUNK_SIZE
    if total_memory >= _MAX_VRAM_FOR_CHUNK_SCALING:
        return _MAX_CHUNK_SIZE
    interpolation = (total_memory - _MIN_VRAM_FOR_CHUNK_SCALING) / (
        _MAX_VRAM_FOR_CHUNK_SCALING - _MIN_VRAM_FOR_CHUNK_SCALING
    )
    return int(_MIN_CHUNK_SIZE + interpolation * (_MAX_CHUNK_SIZE - _MIN_CHUNK_SIZE))


class _PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class _CausalConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int | tuple[int, int, int] = 1) -> None:
        super().__init__()
        time_stride = stride if isinstance(stride, int) else stride[0]
        self.out_channels = out_channels
        self.time_stride = time_stride
        self.time_kernel_size = 3
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            dilation=(1, 1, 1),
            padding=(0, 1, 1),
            padding_mode="zeros",
        )
        self.temporal_cache_state: dict[int, tuple[Optional[torch.Tensor], bool]] = {}

    def _empty_output(self, x: torch.Tensor) -> torch.Tensor:
        height = (x.shape[3] + 2 * self.conv.padding[1] - self.conv.kernel_size[1]) // self.conv.stride[1] + 1
        width = (x.shape[4] + 2 * self.conv.padding[2] - self.conv.kernel_size[2]) // self.conv.stride[2] + 1
        return x.new_empty((x.shape[0], self.out_channels, 0, height, width))

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        thread_id = threading.get_ident()
        cached, is_end = self.temporal_cache_state.get(thread_id, (None, False))
        if cached is None:
            padding_length = self.time_kernel_size - 1
            if not causal:
                padding_length //= 2
            if x.shape[2] == 0:
                return self._empty_output(x)
            cached = x[:, :, :1].repeat((1, 1, padding_length, 1, 1))

        pieces = [cached, x]
        if is_end and not causal:
            pieces.append(x[:, :, -1:].repeat((1, 1, (self.time_kernel_size - 1) // 2, 1, 1)))
        input_length = sum(piece.shape[2] for piece in pieces)
        cache_length = (self.time_kernel_size - self.time_stride) + (
            (input_length - self.time_kernel_size) % self.time_stride
        )

        needs_caching = not is_end
        if needs_caching and cache_length == 0:
            self.temporal_cache_state[thread_id] = (x[:, :, :0], False)
            needs_caching = False
        if needs_caching and x.shape[2] >= cache_length:
            needs_caching = False
            self.temporal_cache_state[thread_id] = (x[:, :, -cache_length:], False)

        x = torch.cat(pieces, dim=2)
        if needs_caching:
            self.temporal_cache_state[thread_id] = (x[:, :, -cache_length:], False)
        elif is_end:
            self.temporal_cache_state[thread_id] = (None, True)

        return self.conv(x) if x.shape[2] >= self.time_kernel_size else self._empty_output(x)


class _ResnetBlock3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = _PixelNorm()
        self.non_linearity = nn.SiLU()
        self.conv1 = _CausalConv3d(channels, channels)
        self.norm2 = _PixelNorm()
        self.dropout = nn.Dropout(0.0)
        self.conv2 = _CausalConv3d(channels, channels)
        self.conv_shortcut = nn.Identity()
        self.norm3 = nn.Identity()
        self.temporal_cache_state: dict[int, Optional[torch.Tensor]] = {}

    def forward(self, input_tensor: torch.Tensor, causal: bool = True) -> torch.Tensor:
        hidden_states = self.non_linearity(self.norm1(input_tensor))
        hidden_states = self.conv1(hidden_states, causal=causal)
        hidden_states = self.non_linearity(self.norm2(hidden_states))
        hidden_states = self.conv2(self.dropout(hidden_states), causal=causal)
        input_tensor = self.conv_shortcut(self.norm3(input_tensor))
        thread_id = threading.get_ident()
        cached = self.temporal_cache_state.get(thread_id)
        cached = _add_exchange_cache(hidden_states, cached, input_tensor)
        self.temporal_cache_state[thread_id] = cached
        return hidden_states


class _UNetMidBlock3d(nn.Module):
    def __init__(self, channels: int, num_layers: int) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList(_ResnetBlock3d(channels) for _ in range(num_layers))

    def forward(self, hidden_states: torch.Tensor, causal: bool = True) -> torch.Tensor:
        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states, causal=causal)
        return hidden_states


def _patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, channels, frames, height, width = x.shape
    if height % patch_size or width % patch_size:
        raise ValueError("LTX video encoder input must align to its spatial patch size")
    x = x.reshape(
        batch,
        channels,
        frames,
        1,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    return x.permute(0, 1, 3, 7, 5, 2, 4, 6).reshape(
        batch,
        channels * patch_size * patch_size,
        frames,
        height // patch_size,
        width // patch_size,
    )


def _space_to_depth(x: torch.Tensor, stride: tuple[int, int, int]) -> torch.Tensor:
    batch, channels, frames, height, width = x.shape
    temporal, vertical, horizontal = stride
    x = x.reshape(
        batch,
        channels,
        frames // temporal,
        temporal,
        height // vertical,
        vertical,
        width // horizontal,
        horizontal,
    )
    return x.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(
        batch,
        channels * temporal * vertical * horizontal,
        frames // temporal,
        height // vertical,
        width // horizontal,
    )


class _SpaceToDepthDownsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels
        self.conv = _CausalConv3d(in_channels, out_channels // math.prod(stride))
        self.temporal_cache_state: dict[
            int,
            tuple[
                Optional[torch.Tensor],
                bool,
                Optional[torch.Tensor],
                Optional[torch.Tensor],
            ],
        ] = {}

    def forward(self, x: torch.Tensor, causal: bool = True) -> Optional[torch.Tensor]:
        thread_id = threading.get_ident()
        cached, pad_first, cached_x, cached_input = self.temporal_cache_state.get(
            thread_id, (None, True, None, None)
        )
        if cached_input is not None:
            x = _cat_if_needed([cached_input, x], dim=2)
            cached_input = None

        if self.stride[0] == 2 and pad_first:
            x = torch.cat((x[:, :, :1], x), dim=2)
            pad_first = False
        if x.shape[2] < self.stride[0]:
            cached_input = x
            self.temporal_cache_state[thread_id] = (
                cached,
                pad_first,
                cached_x,
                cached_input,
            )
            return None

        residual = _space_to_depth(x, self.stride)
        residual = residual.reshape(
            residual.shape[0],
            residual.shape[1] // self.group_size,
            self.group_size,
            *residual.shape[2:],
        ).mean(dim=2)

        x = self.conv(x, causal=causal)
        if self.stride[0] == 2 and x.shape[2] == 1:
            if cached_x is not None:
                x = _cat_if_needed([cached_x, x], dim=2)
                cached_x = None
            else:
                cached_x = x
                x = None
        if x is not None:
            x = _space_to_depth(x, self.stride)

        cached = _add_exchange_cache(x, cached, residual)
        self.temporal_cache_state[thread_id] = (
            cached,
            pad_first,
            cached_x,
            cached_input,
        )
        return x


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = 128
        self.patch_size = 4
        self.conv_in = _CausalConv3d(3 * self.patch_size * self.patch_size, channels)
        self.down_blocks = nn.ModuleList()
        for block_name, params in _ENCODER_BLOCKS:
            if block_name == "res_x":
                self.down_blocks.append(_UNetMidBlock3d(channels, params["num_layers"]))
                continue
            multiplier = params["multiplier"]
            out_channels = channels * multiplier
            stride = {
                "compress_space_res": (1, 2, 2),
                "compress_time_res": (2, 1, 1),
                "compress_all_res": (2, 2, 2),
            }[block_name]
            self.down_blocks.append(
                _SpaceToDepthDownsample(channels, out_channels, stride)
            )
            channels = out_channels
        self.conv_norm_out = _PixelNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = _CausalConv3d(channels, 129)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        try:
            _mark_conv3d_ended(self)
            sample = self.conv_in(_patchify(sample, self.patch_size), causal=True)
            for block in self.down_blocks:
                sample = block(sample, causal=True)
                if sample is None:
                    raise RuntimeError("canonical single-frame LTX encode produced no latent")
            sample = self.conv_out(
                self.conv_act(self.conv_norm_out(sample)), causal=True
            )
            return sample[:, :128]
        finally:
            _clear_temporal_cache(self)


class _DepthToSpaceUpsample(nn.Module):
    def __init__(self, in_channels: int, stride: tuple[int, int, int], reduction: int) -> None:
        super().__init__()
        self.stride = stride
        self.out_channels = math.prod(stride) * in_channels // reduction
        self.conv = _CausalConv3d(in_channels, self.out_channels)
        self.residual = False
        self.temporal_cache_state: dict[int, tuple[Optional[torch.Tensor], bool, bool]] = {}

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        thread_id = threading.get_ident()
        cached, drop_first_conv, drop_first_res = self.temporal_cache_state.get(thread_id, (None, True, True))
        y = self.conv(x, causal=causal)
        batch, channels, frames, height, width = y.shape
        temporal, vertical, horizontal = self.stride
        y = y.reshape(batch, channels // (temporal * vertical * horizontal), temporal, vertical, horizontal, frames, height, width)
        y = y.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(
            batch, channels // (temporal * vertical * horizontal), frames * temporal, height * vertical, width * horizontal
        )
        if temporal == 2 and y.shape[2] > 0 and drop_first_conv:
            y = y[:, :, 1:]
            drop_first_conv = False
        self.temporal_cache_state[thread_id] = (None, drop_first_conv, False)
        return y


def _unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, channels, frames, height, width = x.shape
    x = x.reshape(batch, channels // (patch_size * patch_size), patch_size, patch_size, frames, height, width)
    return x.permute(0, 1, 4, 5, 3, 6, 2).reshape(
        batch, channels // (patch_size * patch_size), frames, height * patch_size, width * patch_size
    )


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = 1024
        self.patch_size = 4
        self.causal = False
        self.conv_in = _CausalConv3d(128, channels)
        self.up_blocks = nn.ModuleList()
        for block_name, params in reversed(_DECODER_BLOCKS):
            if block_name == "res_x":
                self.up_blocks.append(_UNetMidBlock3d(channels, params["num_layers"]))
            elif block_name == "compress_space":
                reduction = params["multiplier"]
                self.up_blocks.append(_DepthToSpaceUpsample(channels, (1, 2, 2), reduction))
                channels //= reduction
            elif block_name == "compress_time":
                reduction = params["multiplier"]
                self.up_blocks.append(_DepthToSpaceUpsample(channels, (2, 1, 1), reduction))
                channels //= reduction
            elif block_name == "compress_all":
                reduction = params["multiplier"]
                self.up_blocks.append(_DepthToSpaceUpsample(channels, (2, 2, 2), reduction))
                channels //= reduction
            else:  # pragma: no cover - the fixed source descriptor is exhaustive.
                raise ValueError(f"unsupported LTX decoder block: {block_name}")
        self.conv_norm_out = _PixelNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = _CausalConv3d(channels, 3 * self.patch_size * self.patch_size)

    @staticmethod
    def _output_shape(input_shape: torch.Size) -> tuple[int, int, int, int, int]:
        return input_shape[0], 3, input_shape[2] * 8 - 7, input_shape[3] * 32, input_shape[4] * 32

    def _run_up(
        self,
        index: int,
        sample_ref: list[Optional[torch.Tensor]],
        ended: bool,
        output: torch.Tensor,
        output_offset: list[int],
        max_chunk_size: int,
    ) -> None:
        sample = sample_ref[0]
        sample_ref[0] = None
        assert sample is not None
        if index >= len(self.up_blocks):
            sample = self.conv_act(self.conv_norm_out(sample))
            if ended:
                _mark_conv3d_ended(self.conv_out)
            sample = self.conv_out(sample, causal=self.causal)
            if sample.shape[2] > 0:
                sample = _unpatchify(sample, self.patch_size)
                frame_count = sample.shape[2]
                output[:, :, output_offset[0] : output_offset[0] + frame_count].copy_(sample)
                output_offset[0] += frame_count
            return

        block = self.up_blocks[index]
        if ended:
            _mark_conv3d_ended(block)
        sample = block(sample, causal=self.causal)
        if sample.shape[2] == 0:
            return

        total_bytes = sample.numel() * sample.element_size()
        num_chunks = (total_bytes + max_chunk_size - 1) // max_chunk_size
        if num_chunks == 1:
            self._run_up(index + 1, [sample], ended, output, output_offset, max_chunk_size)
            return
        for chunk_index, chunk in enumerate(torch.chunk(sample, chunks=num_chunks, dim=2)):
            self._run_up(
                index + 1,
                [chunk],
                ended and chunk_index == num_chunks - 1,
                output,
                output_offset,
                max_chunk_size,
            )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        try:
            _mark_conv3d_ended(self.conv_in)
            sample = self.conv_in(sample, causal=self.causal)
            output = torch.empty(self._output_shape(sample.shape), dtype=sample.dtype, device=sample.device)
            self._run_up(0, [sample], True, output, [0], _get_max_chunk_size(sample.device))
            return output
        finally:
            _clear_temporal_cache(self)


class Ltx23VideoEncoder:
    """Encode the canonical I2V source frame at 256px or 512px."""

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        checkpoint = Ltx23Checkpoint(checkpoint_path)
        state = {
            name.removeprefix("vae.encoder."): checkpoint.tensor(name)
            for name in checkpoint.tensor_names
            if name.startswith("vae.encoder.")
        }
        with torch.device("meta"):
            self.model = _Encoder()
        incompatible = self.model.load_state_dict(state, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys or len(state) != 84:
            raise ValueError("unexpected pinned LTX 2.3 video encoder state")
        self.model.to(device=device, dtype=torch.bfloat16).eval()
        self._mean = checkpoint.tensor("vae.per_channel_statistics.mean-of-means").to(
            device=device, dtype=torch.bfloat16
        ).view(1, 128, 1, 1, 1)
        self._std = checkpoint.tensor("vae.per_channel_statistics.std-of-means").to(
            device=device, dtype=torch.bfloat16
        ).view(1, 128, 1, 1, 1)

    @torch.inference_mode()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if tuple(image.shape) not in ((1, 3, 256, 256), (1, 3, 512, 512)):
            raise ValueError("canonical I2V encode requires one 256px or 512px RGB frame")
        pixels = image.unsqueeze(2).mul(2.0).sub(1.0).to(
            device=self._mean.device, dtype=torch.bfloat16
        )
        means = self.model(pixels)
        return ((means - self._mean) / self._std).float().cpu()

    def close(self) -> None:
        self.model = None
        self._mean = None
        self._std = None
        torch.cuda.empty_cache()


class Ltx23VideoDecoder:
    """Decode a canonical 512px or 768px stage-two latent into 145 RGB frames."""

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        checkpoint = Ltx23Checkpoint(checkpoint_path)
        state = {
            name.removeprefix("vae.decoder."): checkpoint.tensor(name)
            for name in checkpoint.tensor_names
            if name.startswith("vae.decoder.")
        }
        with torch.device("meta"):
            self.model = _Decoder()
        incompatible = self.model.load_state_dict(state, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys or len(state) != 84:
            raise ValueError("unexpected pinned LTX 2.3 video decoder state")
        self.model.to(device=device, dtype=torch.bfloat16).eval()
        self._mean = checkpoint.tensor("vae.per_channel_statistics.mean-of-means").to(
            device=device, dtype=torch.bfloat16
        ).view(1, 128, 1, 1, 1)
        self._std = checkpoint.tensor("vae.per_channel_statistics.std-of-means").to(
            device=device, dtype=torch.bfloat16
        ).view(1, 128, 1, 1, 1)

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if tuple(latents.shape) not in (
            (1, 128, 19, 16, 16),
            (1, 128, 19, 24, 24),
        ):
            raise ValueError("the canonical T2V decoder expects a 512px or 768px latent")
        x = latents.to(device=self._mean.device, dtype=torch.bfloat16)
        # The pinned Comfy VAE wrapper's default post-processing remains active
        # for LTX 2.3: decoded RGB is mapped from [-1, 1] into display space.
        return self.model(x * self._std + self._mean).add_(1.0).div_(2.0).clamp_(0.0, 1.0)

    def close(self) -> None:
        self.model = None
        self._mean = None
        self._std = None
        torch.cuda.empty_cache()
