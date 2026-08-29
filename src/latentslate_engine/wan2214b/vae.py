from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn

from .weights import TensorStore

CACHE_T = 2


def _cat(values: list[torch.Tensor | None], dim: int) -> torch.Tensor | None:
    values = [value for value in values if value is not None and value.shape[dim] > 0]
    if len(values) == 1:
        return values[0]
    if values:
        return torch.cat(values, dim)
    return None


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = 2 * self.padding[0]
        self.padding = (0, self.padding[1], self.padding[2])

    def forward(self, x, cache_x=None):
        if cache_x is None and x.shape[2] == 1:
            return F.conv3d(
                x,
                self.weight[:, :, -1:],
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        if self._padding > 0:
            needed = self._padding
            if cache_x is not None:
                cache_x = cache_x.to(x.device)
                needed = max(0, needed - cache_x.shape[2])
            shape = list(x.shape)
            shape[2] = needed
            padding = torch.zeros(shape, device=x.device, dtype=x.dtype)
            x = _cat([padding, cache_x, x], 2)
        return super().forward(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, images: bool = True):
        super().__init__()
        shape = (dim, 1, 1) if images else (dim, 1, 1, 1)
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma.to(x)


class Resample(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        self.mode = mode
        if mode in ("upsample2d", "upsample3d"):
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            if mode == "upsample3d":
                self.time_conv = CausalConv3d(
                    dim, dim * 2, (3, 1, 1), padding=(1, 0, 0)
                )
        else:
            raise ValueError(mode)

    def forward(self, x, cache, index):
        b, c, t, h, w = x.shape
        if self.mode == "upsample3d":
            slot = index[0]
            if cache[slot] is None:
                cache[slot] = "Rep"
                index[0] += 1
            else:
                previous = cache[slot]
                cache[slot] = x[:, :, -CACHE_T:]
                x = (
                    self.time_conv(x)
                    if previous == "Rep"
                    else self.time_conv(x, previous)
                )
                index[0] += 1
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), 3).reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        return x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3]).permute(
            0, 2, 1, 3, 4
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.residual = nn.ModuleList(
            [
                RMSNorm(in_dim, images=False),
                nn.SiLU(),
                CausalConv3d(in_dim, out_dim, 3, padding=1),
                RMSNorm(out_dim, images=False),
                nn.SiLU(),
                nn.Dropout(0.0),
                CausalConv3d(out_dim, out_dim, 3, padding=1),
            ]
        )
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, cache, index):
        old = x
        for layer in self.residual:
            if isinstance(layer, CausalConv3d):
                slot = index[0]
                current = x[:, :, -CACHE_T:]
                previous = cache[slot]
                cache[slot] = None
                x = layer(x, previous)
                cache[slot] = current
                index[0] += 1
            else:
                x = layer(x)
        return x + self.shortcut(old)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x, cache, index):
        identity = x
        b, c, t, _, _ = x.shape
        h, w = x.shape[-2:]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        shape = q.shape
        q = q.view(shape[0], 1, c, -1).transpose(2, 3)
        k = k.view(shape[0], 1, c, -1).transpose(2, 3)
        v = v.view(shape[0], 1, c, -1).transpose(2, 3)
        x = F.scaled_dot_product_attention(q, k, v).transpose(2, 3).reshape(shape)
        x = self.proj(x)
        x = x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + identity


class Decoder(nn.Module):
    def __init__(self, dim: int = 96, z_dim: int = 16):
        super().__init__()
        dims = [dim * value for value in [4, 4, 4, 2, 1]]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.ModuleList(
            [
                ResidualBlock(dims[0], dims[0]),
                AttentionBlock(dims[0]),
                ResidualBlock(dims[0], dims[0]),
            ]
        )
        layers: list[nn.Module] = []
        temporal = [True, True, False]
        for stage, (in_dim, out_dim) in enumerate(pairwise(dims)):
            if stage in (1, 2, 3):
                in_dim //= 2
            for _ in range(3):
                layers.append(ResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if stage != 3:
                layers.append(
                    Resample(out_dim, "upsample3d" if temporal[stage] else "upsample2d")
                )
        self.upsamples = nn.ModuleList(layers)
        self.head = nn.ModuleList(
            [RMSNorm(dim, images=False), nn.SiLU(), CausalConv3d(dim, 3, 3, padding=1)]
        )

    def _run_up(self, layer_index, x_ref, cache, index, chunks):
        x = x_ref[0]
        x_ref[0] = None
        if layer_index >= len(self.upsamples):
            for layer in self.head:
                if isinstance(layer, CausalConv3d):
                    slot = index[0]
                    current = x[:, :, -CACHE_T:]
                    previous = cache[slot]
                    cache[slot] = None
                    x = layer(x, previous)
                    cache[slot] = current
                    index[0] += 1
                else:
                    x = layer(x)
            chunks.append(x)
            return
        layer = self.upsamples[layer_index]
        x = layer(x, cache, index)
        if (
            isinstance(layer, Resample)
            and layer.mode == "upsample3d"
            and x.shape[2] > 2
        ):
            for frame in range(0, x.shape[2], 2):
                self._run_up(
                    layer_index + 1,
                    [x[:, :, frame : frame + 2]],
                    cache,
                    index.copy(),
                    chunks,
                )
            del x
            return
        next_ref = [x]
        del x
        self._run_up(layer_index + 1, next_ref, cache, index, chunks)

    def forward(self, x, cache, index):
        slot = index[0]
        current = x[:, :, -CACHE_T:]
        x = self.conv1(x, cache[slot])
        cache[slot] = current
        index[0] += 1
        for layer in self.middle:
            x = layer(x, cache, index)
        chunks = []
        self._run_up(0, [x], cache, index, chunks)
        return chunks


def _cache_layers(model: nn.Module) -> int:
    return sum(isinstance(module, CausalConv3d) for module in model.modules())


class WanVaeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = CausalConv3d(16, 16, 1)
        self.decoder = Decoder()

    def decode(self, z):
        cache = [None] * _cache_layers(self.decoder)
        x = self.conv2(z)
        output = []
        for index in range(1 + z.shape[2] // 2):
            feature_index = [0]
            if index == 0:
                output = self.decoder(x[:, :, :1], cache, feature_index)
            else:
                output += self.decoder(
                    x[:, :, 1 + 2 * (index - 1) : 1 + 2 * index], cache, feature_index
                )
        return torch.cat(output, 2)


def load_vae(path: str, device: torch.device) -> WanVaeDecoder:
    store = TensorStore(path)
    with torch.device("meta"):
        model = WanVaeDecoder()
    state = {
        key: store.tensor(key)
        for key in store.keys
        if key.startswith(("conv2.", "decoder."))
    }
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected or [key for key in missing if key.startswith(("conv2.", "decoder."))]:
        raise ValueError(
            f"Wan VAE state mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model.to(device=device, dtype=torch.bfloat16).eval()
