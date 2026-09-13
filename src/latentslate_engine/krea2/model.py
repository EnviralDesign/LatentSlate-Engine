"""Krea Turbo T2I transformer, narrowly adapted from pinned ComfyUI Krea 2.

Source: Comfy-Org/ComfyUI 12d5279438bfefc058a269eae805ceab6047777f,
comfy/ldm/krea2/model.py and flux positional embeddings (GPL-3.0).
Graph patches and reference-image conditioning are outside this native mode.
"""

from typing import Optional
import math
import torch
from torch import nn
from torch.nn import functional as F
from .attention import attention
import comfy_kitchen as ck


class EmbedND(nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.theta, self.axes_dim = theta, axes_dim

    def forward(self, ids):
        embeddings = []
        for index, dim in enumerate(self.axes_dim):
            device = ids.device if ids.device.type != "mps" else torch.device("cpu")
            scale = torch.linspace(
                0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device
            )
            omega = 1.0 / (self.theta**scale)
            out = torch.einsum(
                "...n,d->...nd",
                ids[..., index].to(device=device, dtype=torch.float32),
                omega,
            )
            out = torch.stack([out.cos(), -out.sin(), out.sin(), out.cos()], dim=-1)
            embeddings.append(
                out.reshape(*out.shape[:-1], 2, 2).to(
                    device=ids.device, dtype=torch.float32
                )
            )
        return torch.cat(embeddings, dim=-3).unsqueeze(1)


def timestep_embedding(t, dim):
    t = 1000.0 * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t)


class RMSNorm(nn.Module):
    """RMSNorm with the reference ``(1 + scale)`` weight convention (scale stored zero-centered)."""

    def __init__(
        self,
        features: int,
        eps: float = 1e-05,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.empty(features, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        weight = self.scale.to(dtype=torch.float32, device=x.device) + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), weight=weight, eps=self.eps).to(
            dtype
        )


class QKNorm(nn.Module):

    def __init__(self, dim: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.qnorm = RMSNorm(dim, device=device, dtype=dtype, operations=operations)
        self.knorm = RMSNorm(dim, device=device, dtype=dtype, operations=operations)

    def forward(self, q, k):
        return (self.qnorm(q), self.knorm(k))


class SwiGLU(nn.Module):

    def __init__(
        self,
        features: int,
        multiplier: int,
        bias: bool = False,
        multiple: int = 128,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = operations.Linear(
            features, mlpdim, bias=bias, device=device, dtype=dtype
        )
        self.up = operations.Linear(
            features, mlpdim, bias=bias, device=device, dtype=dtype
        )
        self.down = operations.Linear(
            mlpdim, features, bias=bias, device=device, dtype=dtype
        )

    def forward(self, x):
        return self.down(F.silu(self.gate(x)).mul_(self.up(x)))


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        heads: int,
        kvheads: Optional[int] = None,
        bias: bool = False,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = operations.Linear(
            dim, self.headdim * self.heads, bias=bias, device=device, dtype=dtype
        )
        self.wk = operations.Linear(
            dim, self.headdim * self.kvheads, bias=bias, device=device, dtype=dtype
        )
        self.wv = operations.Linear(
            dim, self.headdim * self.kvheads, bias=bias, device=device, dtype=dtype
        )
        self.gate = operations.Linear(dim, dim, bias=bias, device=device, dtype=dtype)
        self.qknorm = QKNorm(
            self.headdim, device=device, dtype=dtype, operations=operations
        )
        self.wo = operations.Linear(dim, dim, bias=bias, device=device, dtype=dtype)

    def forward(self, x, freqs=None, mask=None):
        q, k, v, gate = (self.wq(x), self.wk(x), self.wv(x), self.gate(x))
        q = q.reshape(x.shape[0], x.shape[1], self.heads, self.headdim).transpose(1, 2)
        k = k.reshape(x.shape[0], x.shape[1], self.kvheads, self.headdim).transpose(
            1, 2
        )
        v = v.reshape(x.shape[0], x.shape[1], self.kvheads, self.headdim).transpose(
            1, 2
        )
        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = ck.apply_rope(q, k, freqs)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = attention(q, k, v, mask)
        out = out.transpose(1, 2).reshape(x.shape[0], -1, self.heads * self.headdim)
        return self.wo(out * F.sigmoid(gate))


class SimpleModulation(nn.Module):

    def __init__(self, dim: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.lin = nn.Parameter(torch.empty(2, dim, device=device, dtype=dtype))

    def forward(self, vec):
        out = vec + self.lin.to(dtype=vec.dtype, device=vec.device).unsqueeze(0)
        scale, shift = out.chunk(2, dim=1)
        return (scale, shift)


class DoubleSharedModulation(nn.Module):

    def __init__(self, dim: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.lin = nn.Parameter(torch.empty(6 * dim, device=device, dtype=dtype))

    def forward(self, vec):
        out = vec + self.lin.to(dtype=vec.dtype, device=vec.device)
        return out.chunk(6, dim=-1)


class TextFusionBlock(nn.Module):

    def __init__(
        self,
        features,
        heads,
        multiplier,
        bias=False,
        kvheads=None,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.prenorm = RMSNorm(
            features, device=device, dtype=dtype, operations=operations
        )
        self.postnorm = RMSNorm(
            features, device=device, dtype=dtype, operations=operations
        )
        self.attn = Attention(
            features,
            heads,
            kvheads=kvheads,
            bias=bias,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.mlp = SwiGLU(
            features,
            multiplier,
            bias,
            device=device,
            dtype=dtype,
            operations=operations,
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.prenorm(x), mask=mask)
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(nn.Module):

    def __init__(
        self,
        num_txt_layers,
        txt_dim,
        heads,
        multiplier,
        bias=False,
        kvheads=None,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [
                TextFusionBlock(
                    txt_dim,
                    heads,
                    multiplier,
                    bias,
                    kvheads,
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
                for _ in range(2)
            ]
        )
        self.projector = operations.Linear(
            num_txt_layers, 1, bias=False, device=device, dtype=dtype
        )
        self.refiner_blocks = nn.ModuleList(
            [
                TextFusionBlock(
                    txt_dim,
                    heads,
                    multiplier,
                    bias,
                    kvheads,
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
                for _ in range(2)
            ]
        )

    def forward(self, x, mask=None):
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None)
        x = x.reshape(b, l, n, d).permute(0, 1, 3, 2)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask)
        return x


class SingleStreamBlock(nn.Module):

    def __init__(
        self,
        features,
        heads,
        multiplier,
        bias=False,
        kvheads=None,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.mod = DoubleSharedModulation(
            features, device=device, dtype=dtype, operations=operations
        )
        self.prenorm = RMSNorm(
            features, device=device, dtype=dtype, operations=operations
        )
        self.postnorm = RMSNorm(
            features, device=device, dtype=dtype, operations=operations
        )
        self.attn = Attention(
            features,
            heads,
            kvheads=kvheads,
            bias=bias,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.mlp = SwiGLU(
            features,
            multiplier,
            bias,
            device=device,
            dtype=dtype,
            operations=operations,
        )

    def forward(self, x, vec, freqs):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs)
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
        return x


class LastLayer(nn.Module):

    def __init__(
        self, features, patch, channels, device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.norm = RMSNorm(features, device=device, dtype=dtype, operations=operations)
        self.linear = operations.Linear(
            features, patch * patch * channels, bias=True, device=device, dtype=dtype
        )
        self.modulation = SimpleModulation(
            features, device=device, dtype=dtype, operations=operations
        )

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        return self.linear(x)


class SingleStreamDiT(nn.Module):

    def __init__(
        self,
        features=6144,
        tdim=256,
        txtdim=2560,
        heads=48,
        kvheads=12,
        multiplier=4,
        layers=28,
        patch=2,
        channels=16,
        bias=False,
        theta=1000.0,
        txtlayers=12,
        txtheads=20,
        txtkvheads=20,
        default_ref_method=None,
        image_model=None,
        device=None,
        dtype=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        self.patch = patch
        self.channels = channels
        self.tdim = tdim
        self.heads = heads
        self.txtdim = txtdim
        self.txtlayers = txtlayers
        headdim = features // heads
        axes = [
            headdim - 12 * (headdim // 16),
            6 * (headdim // 16),
            6 * (headdim // 16),
        ]
        assert sum(axes) == headdim, f"axes {axes} sum != headdim {headdim}"
        self.pe_embedder = EmbedND(dim=headdim, theta=int(theta), axes_dim=axes)
        self.first = operations.Linear(
            channels * patch**2, features, bias=True, device=device, dtype=dtype
        )
        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    features,
                    heads,
                    multiplier,
                    bias,
                    kvheads,
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
                for _ in range(layers)
            ]
        )
        self.tmlp = nn.Sequential(
            operations.Linear(tdim, features, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            operations.Linear(features, features, device=device, dtype=dtype),
        )
        self.txtfusion = TextFusionTransformer(
            txtlayers,
            txtdim,
            txtheads,
            multiplier,
            bias,
            txtkvheads,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(txtdim, device=device, dtype=dtype, operations=operations),
            operations.Linear(txtdim, features, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            operations.Linear(features, features, device=device, dtype=dtype),
        )
        self.last = LastLayer(
            features, patch, channels, device=device, dtype=dtype, operations=operations
        )
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            operations.Linear(features, features * 6, device=device, dtype=dtype),
        )

    def forward(self, x, timesteps, context):
        if x.ndim != 5 or x.shape[0] != 1 or x.shape[1] != 16 or (x.shape[2] != 1):
            raise ValueError("Krea Turbo T2I requires a single 16-channel image latent")
        x = x[:, :, 0]
        bs, _, h_orig, w_orig = x.shape
        context = self._unpack_context(context)
        img, imgpos, h_, w_ = self.process_img(x)
        img_tokens = img.shape[1]
        img = self.first(img)
        t = self.tmlp(
            timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype)
        )
        tvec = self.tproj(t)
        context = self.txtfusion(context)
        context = self.txtmlp(context)
        txtlen = context.shape[1]
        txtpos = torch.zeros(bs, txtlen, 3, device=context.device, dtype=torch.float32)
        combined = torch.cat((context, img), dim=1)
        del context, img
        pos = torch.cat((txtpos, imgpos), dim=1)
        freqs = self.pe_embedder(pos)
        del pos, txtpos, imgpos
        for block in self.blocks:
            combined = block(combined, tvec, freqs)
        final = self.last(combined, t)
        out = final[:, txtlen : txtlen + img_tokens, :]
        out = (
            out.reshape(bs, h_, w_, self.channels, self.patch, self.patch)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(bs, self.channels, h_ * self.patch, w_ * self.patch)
        )
        return out[:, :, :h_orig, :w_orig].unsqueeze(2)

    def process_img(self, x, index=0):
        patch = self.patch
        x = F.pad(
            x, (0, (-x.shape[-1]) % patch, 0, (-x.shape[-2]) % patch), mode="circular"
        )
        h, w = (x.shape[-2] // patch, x.shape[-1] // patch)
        img = (
            x.reshape(x.shape[0], self.channels, h, patch, w, patch)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(x.shape[0], h * w, self.channels * patch * patch)
        )
        img_ids = torch.zeros(h, w, 3, device=x.device, dtype=torch.float32)
        img_ids[..., 0] = index
        img_ids[..., 1] = torch.arange(h, device=x.device, dtype=torch.float32)[:, None]
        img_ids[..., 2] = torch.arange(w, device=x.device, dtype=torch.float32)[None, :]
        return (img, img_ids.reshape(1, h * w, 3).repeat(x.shape[0], 1, 1), h, w)

    def _unpack_context(self, context):
        b, seq, fused = context.shape
        if fused != self.txtlayers * self.txtdim:
            raise ValueError(
                f"Krea2 expects conditioning with {self.txtlayers}x{self.txtdim}={self.txtlayers * self.txtdim} features (a {self.txtlayers}-layer Qwen3-VL stack) but got {fused}. Use the Krea Qwen3-VL conditioning encoder."
            )
        return context.reshape(b, seq, self.txtlayers, self.txtdim)
