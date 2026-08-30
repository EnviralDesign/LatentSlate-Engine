"""Pinned LTX 2.3 waveform vocoder adapted without the Comfy runtime.

This is a narrow, source-conformant adaptation of the vocoder used by the
pinned Comfy reference.  The fixture uses its BigVGAN AMP1 path only; the
module retains that path's state layout and call order exactly.
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import Ltx23Checkpoint


def _cast_to(
    tensor: torch.Tensor, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """The pinned source's ``model_management.cast_to`` use in this module."""
    if tensor.dtype == dtype and tensor.device == device:
        return tensor
    return tensor.to(dtype=dtype, device=device)


def _sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x == 0,
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
        torch.sin(math.pi * x) / math.pi / x,
    )


def _kaiser_sinc_filter1d(
    cutoff: float, half_width: float, kernel_size: int
) -> torch.Tensor:
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


def _hann_sinc_filter1d(ratio: int) -> torch.Tensor:
    # Source defaults: rolloff=.99, lowpass_filter_width=6.
    rolloff = 0.99
    lowpass_filter_width = 6
    width = math.ceil(lowpass_filter_width / rolloff)
    kernel_size = 2 * width * ratio + 1
    time = (torch.arange(kernel_size) / ratio - width) * rolloff
    clamped = time.clamp(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(clamped * math.pi / lowpass_filter_width / 2) ** 2
    return (torch.sinc(time) * window * rolloff / ratio).view(1, 1, -1)


class _LowPassFilter1d(nn.Module):
    def __init__(
        self, cutoff: float, half_width: float, stride: int, kernel_size: int
    ) -> None:
        super().__init__()
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer(
            "filter", _kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, _ = x.shape
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        filter_ = _cast_to(
            self.filter.expand(channels, -1, -1), dtype=x.dtype, device=x.device
        )
        return F.conv1d(x, filter_, stride=self.stride, groups=channels)


class _UpSample1d(nn.Module):
    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int | None = None,
        *,
        persistent: bool = True,
        window_type: str = "kaiser",
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        if window_type == "hann":
            width = math.ceil(6 / 0.99)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio
            filter_ = _hann_sinc_filter1d(ratio)
        else:
            self.kernel_size = (
                int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            )
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = (
                self.pad * self.stride + (self.kernel_size - self.stride) // 2
            )
            self.pad_right = (
                self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            )
            filter_ = _kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self.kernel_size)
        self.register_buffer("filter", filter_, persistent=persistent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        filter_ = _cast_to(
            self.filter.expand(channels, -1, -1), dtype=x.dtype, device=x.device
        )
        x = self.ratio * F.conv_transpose1d(
            x, filter_, stride=self.stride, groups=channels
        )
        return x[..., self.pad_left : -self.pad_right]


class _DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()
        kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = _LowPassFilter1d(0.5 / ratio, 0.6 / ratio, ratio, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class _SnakeBeta(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(
            _cast_to(
                self.alpha.unsqueeze(0).unsqueeze(-1), dtype=x.dtype, device=x.device
            )
        )
        beta = torch.exp(
            _cast_to(
                self.beta.unsqueeze(0).unsqueeze(-1), dtype=x.dtype, device=x.device
            )
        )
        return x + (1.0 / (beta + self.eps)) * torch.sin(x * alpha).pow(2)


class _Activation1d(nn.Module):
    def __init__(self, activation: nn.Module) -> None:
        super().__init__()
        self.act = activation
        self.upsample = _UpSample1d(2, 12)
        self.downsample = _DownSample1d(2, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


def _padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class _AmpBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: list[int]) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=d,
                    padding=_padding(kernel_size, d),
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=_padding(kernel_size),
                )
                for _ in dilation
            ]
        )
        self.acts1 = nn.ModuleList(
            [_Activation1d(_SnakeBeta(channels)) for _ in self.convs1]
        )
        self.acts2 = nn.ModuleList(
            [_Activation1d(_SnakeBeta(channels)) for _ in self.convs2]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2, act1, act2 in zip(
            self.convs1, self.convs2, self.acts1, self.acts2
        ):
            residual = conv2(act2(conv1(act1(x))))
            x = x + residual
        return x


class _Vocoder(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        rates = config["upsample_rates"]
        kernels = config["upsample_kernel_sizes"]
        block_kernels = config["resblock_kernel_sizes"]
        dilations = config["resblock_dilation_sizes"]
        initial = int(config["upsample_initial_channel"])
        if config["resblock"] != "AMP1" or config["activation"] != "snakebeta":
            raise ValueError(
                "the canonical LTX 2.3 fixture requires the AMP1 snakebeta vocoder"
            )
        if not bool(config["stereo"]):
            raise ValueError("the canonical LTX 2.3 fixture requires stereo audio")
        self.use_tanh_at_final = bool(config["use_tanh_at_final"])
        self.apply_final_activation = bool(config.get("apply_final_activation", True))
        self.num_kernels = len(block_kernels)
        self.num_upsamples = len(rates)
        self.conv_pre = nn.Conv1d(128, initial, 7, 1, padding=3)
        self.ups = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    initial // (2**i),
                    initial // (2 ** (i + 1)),
                    kernel,
                    rate,
                    padding=(kernel - rate) // 2,
                )
                for i, (rate, kernel) in enumerate(zip(rates, kernels))
            ]
        )
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            channels = initial // (2 ** (i + 1))
            for kernel, dilation in zip(block_kernels, dilations):
                self.resblocks.append(_AmpBlock1(channels, int(kernel), dilation))
        self.act_post = _Activation1d(_SnakeBeta(channels))
        self.conv_post = nn.Conv1d(
            channels, 2, 7, 1, padding=3, bias=bool(config["use_bias_at_final"])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            if x.shape[1] != 2:
                raise ValueError("stereo mel input must have two channels")
            x = torch.cat((x[:, 0, :, :], x[:, 1, :, :]), dim=1)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            averaged = None
            for j in range(self.num_kernels):
                output = self.resblocks[i * self.num_kernels + j](x)
                if averaged is None:
                    averaged = output
                else:
                    averaged += output
            x = averaged / self.num_kernels
        x = self.conv_post(self.act_post(x))
        if self.apply_final_activation:
            return torch.tanh(x) if self.use_tanh_at_final else torch.clamp(x, -1, 1)
        return x


class _StftFn(nn.Module):
    def __init__(self, filter_length: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        frequencies = filter_length // 2 + 1
        self.register_buffer(
            "forward_basis", torch.zeros(frequencies * 2, 1, filter_length)
        )
        self.register_buffer(
            "inverse_basis", torch.zeros(frequencies * 2, 1, filter_length)
        )

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        waveform = F.pad(waveform, (max(0, self.win_length - self.hop_length), 0))
        basis = _cast_to(
            self.forward_basis, dtype=waveform.dtype, device=waveform.device
        )
        spectrum = F.conv1d(waveform, basis, stride=self.hop_length, padding=0)
        frequencies = spectrum.shape[1] // 2
        real, imaginary = spectrum[:, :frequencies], spectrum[:, frequencies:]
        return torch.sqrt(real**2 + imaginary**2), torch.atan2(
            imaginary.float(), real.float()
        ).to(real.dtype)


class _MelStft(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        frequencies = int(config["n_fft"]) // 2 + 1
        self.stft_fn = _StftFn(
            int(config["n_fft"]), int(config["hop_length"]), int(config["n_fft"])
        )
        self.register_buffer(
            "mel_basis", torch.zeros(int(config["num_mels"]), frequencies)
        )

    def mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        magnitude, _ = self.stft_fn(waveform)
        mel_basis = _cast_to(
            self.mel_basis, dtype=magnitude.dtype, device=waveform.device
        )
        return torch.log(torch.clamp(torch.matmul(mel_basis, magnitude), min=1e-5))


class _VocoderWithBwe(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        self.vocoder = _Vocoder(config["vocoder"])
        self.bwe_generator = _Vocoder(
            {**config["bwe"], "apply_final_activation": False}
        )
        bwe = config["bwe"]
        self.input_sample_rate = int(bwe["input_sampling_rate"])
        self.output_sample_rate = int(bwe["output_sampling_rate"])
        self.hop_length = int(bwe["hop_length"])
        self.mel_stft = _MelStft(bwe)
        self.resampler = _UpSample1d(
            self.output_sample_rate // self.input_sample_rate,
            persistent=False,
            window_type="hann",
        )

    def _compute_mel(self, audio: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = audio.shape
        mel = self.mel_stft.mel_spectrogram(audio.reshape(batch * channels, -1))
        return mel.reshape(batch, channels, mel.shape[1], mel.shape[2])

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        waveform = self.vocoder(mel_spec)
        _, _, low_length = waveform.shape
        output_length = low_length * self.output_sample_rate // self.input_sample_rate
        remainder = low_length % self.hop_length
        if remainder:
            waveform = F.pad(waveform, (0, self.hop_length - remainder))
        residual = self.bwe_generator(self._compute_mel(waveform))
        skip = self.resampler(waveform)
        if residual.shape != skip.shape:
            raise RuntimeError(
                f"pinned vocoder residual {residual.shape} != skip {skip.shape}"
            )
        return torch.clamp(residual + skip, -1, 1)[..., :output_length]


class Ltx23AudioVocoder:
    """Decode the canonical LTX 2.3 stereo mel tensor to 48 kHz waveform."""

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        checkpoint = Ltx23Checkpoint(checkpoint_path)
        config = json.loads(checkpoint.metadata["config"])["vocoder"]
        with torch.device("meta"):
            model = _VocoderWithBwe(config)
        state = {
            name.removeprefix("vocoder."): checkpoint.tensor(name)
            for name in checkpoint.tensor_names
            if name.startswith("vocoder.")
        }
        incompatible = model.load_state_dict(state, assign=True)
        if (
            incompatible.missing_keys
            or incompatible.unexpected_keys
            or len(state) != 1227
        ):
            raise ValueError("unexpected pinned LTX 2.3 vocoder state")
        # The reference deliberately keeps this Hann filter non-persistent; create
        # the identical source buffer after meta-state assignment.
        model.resampler.filter = _hann_sinc_filter1d(model.resampler.ratio)
        # The source mel is float32, so Comfy's manual-cast convolution stack
        # materializes this entire vocoder/BWE path in float32.
        self.model = model.to(device=device, dtype=torch.float32).eval()

    @torch.inference_mode()
    def decode(self, mel_spec: torch.Tensor) -> torch.Tensor:
        if tuple(mel_spec.shape[:3]) != (1, 2, 64):
            raise ValueError("the canonical T2V vocoder expects [1, 2, 64, frames]")
        return self.model(
            mel_spec.to(
                device=next(self.model.parameters()).device, dtype=torch.float32
            )
        )

    def close(self) -> None:
        self.model = None
        torch.cuda.empty_cache()
