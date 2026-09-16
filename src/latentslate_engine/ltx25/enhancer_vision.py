"""Narrow E2B vision adaptation from ComfyUI 1a14b82e gemma4.py."""

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def rms_norm(x):
    return F.rms_norm(x, (x.shape[-1],), eps=1e-6)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x), self.eps)


def _apply_rotary_pos_emb(x, freqs_cis):
    cos, sin = freqs_cis[0], freqs_cis[1]
    half = x.shape[-1] // 2
    out = x * cos
    out[..., :half] -= x[..., half:] * sin[..., :half]
    out[..., half:] += x[..., :half] * sin[..., half:]
    return out


def _compute_vision_2d_rope(head_dim, pixel_position_ids, theta=100.0, device=None):
    """Compute 2D RoPE for vision: separate frequencies for x and y dimensions.

    Args:
        head_dim: dimension per head (e.g. 64)
        pixel_position_ids: [batch, num_patches, 2] with (x, y) coords
        theta: RoPE base frequency
    Returns:
        (cos, sin) each of shape [batch, num_patches, head_dim]
    """
    rotary_dim_per_axis = head_dim // 2
    freq_indices = torch.arange(0, rotary_dim_per_axis, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (freq_indices / rotary_dim_per_axis))

    all_cos, all_sin = [], []
    for i in range(2):  # x and y
        dim_positions = pixel_position_ids[:, :, i].float()  # [batch, num_patches]
        freqs = torch.einsum(
            "bi,j->bij", dim_positions, inv_freq.to(device)
        )  # [batch, num_patches, rotary_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [batch, num_patches, rotary_dim]
        all_cos.append(emb.cos())
        all_sin.append(emb.sin())

    cos = torch.cat(all_cos, dim=-1).to(
        pixel_position_ids.device
    )  # [batch, num_patches, head_dim]
    sin = torch.cat(all_sin, dim=-1).to(pixel_position_ids.device)
    return cos, sin


def _apply_vision_2d_rope(x, freqs):
    """Apply 2D RoPE (multidimensional) to vision query/key states.

    Splits x and cos/sin into ndim=2 parts, applies 1D RoPE to each independently.

    x: [batch, heads, seq, head_dim]
    freqs: (cos, sin) each [batch, seq, head_dim]
    """
    cos = freqs[0].unsqueeze(1)  # [batch, 1, seq, head_dim]
    sin = freqs[1].unsqueeze(1)
    half = x.shape[-1] // 2
    a = _apply_rotary_pos_emb(x[..., :half], (cos[..., :half], sin[..., :half]))
    b = _apply_rotary_pos_emb(x[..., half:], (cos[..., half:], sin[..., half:]))
    return torch.cat([a, b], dim=-1)


class ClippedLinear(nn.Module):
    """Linear layer with activation clipping (from quantization-aware training).

    Stores input_max/min and output_max/min as buffers loaded from checkpoint.
    """

    def __init__(
        self, in_features, out_features, bias=False, device=None, dtype=None, ops=None
    ):
        super().__init__()
        self.linear = ops.Linear(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.register_buffer(
            "input_max", torch.tensor(float("inf"), device=device, dtype=dtype)
        )
        self.register_buffer(
            "input_min", torch.tensor(float("-inf"), device=device, dtype=dtype)
        )
        self.register_buffer(
            "output_max", torch.tensor(float("inf"), device=device, dtype=dtype)
        )
        self.register_buffer(
            "output_min", torch.tensor(float("-inf"), device=device, dtype=dtype)
        )

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        x = x.clamp(min=self.input_min, max=self.input_max)
        x = self.linear(x)
        return x.clamp_(min=self.output_min, max=self.output_max)


class Gemma4VisionMLP(nn.Module):
    """SwiGLU MLP matching gate_proj/up_proj/down_proj structure."""

    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        self.gate_proj = ClippedLinear(
            hidden_size, intermediate_size, device=device, dtype=dtype, ops=ops
        )
        self.up_proj = ClippedLinear(
            hidden_size, intermediate_size, device=device, dtype=dtype, ops=ops
        )
        self.down_proj = ClippedLinear(
            intermediate_size, hidden_size, device=device, dtype=dtype, ops=ops
        )

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.gelu(self.gate_proj(x), approximate="tanh")
            * self.up_proj(x)
        )


class Gemma4VisionAttention(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_heads)

        self.q_proj = ClippedLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            device=device,
            dtype=dtype,
            ops=ops,
        )
        self.k_proj = ClippedLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            device=device,
            dtype=dtype,
            ops=ops,
        )
        self.v_proj = ClippedLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            device=device,
            dtype=dtype,
            ops=ops,
        )
        self.o_proj = ClippedLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            device=device,
            dtype=dtype,
            ops=ops,
        )

        self.q_norm = RMSNorm(
            self.head_dim, eps=config["rms_norm_eps"], device=device, dtype=dtype
        )
        self.k_norm = RMSNorm(
            self.head_dim, eps=config["rms_norm_eps"], device=device, dtype=dtype
        )

    def forward(self, x, freqs, attention_mask=None):
        batch_size, seq_length, _ = x.shape

        xq = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        xk = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        xv = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        xq = self.q_norm(xq).transpose(1, 2)
        xk = self.k_norm(xk).transpose(1, 2)
        xv = rms_norm(xv)

        xq = _apply_vision_2d_rope(xq, freqs)
        xk = _apply_vision_2d_rope(xk, freqs)

        xv = xv.to(xq.dtype).transpose(1, 2)

        backend = (
            sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ],
                set_priority=True,
            )
            if xq.nelement() >= 1024 * 128
            else nullcontext()
        )
        with backend:
            output = (
                F.scaled_dot_product_attention(
                    xq, xk, xv, attn_mask=attention_mask, scale=1.0
                )
                .transpose(1, 2)
                .reshape(batch_size, seq_length, self.hidden_size)
            )
        return self.o_proj(output)


class Gemma4VisionLayer(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.self_attn = Gemma4VisionAttention(
            config, device=device, dtype=dtype, ops=ops
        )
        self.mlp = Gemma4VisionMLP(config, device=device, dtype=dtype, ops=ops)
        norm_kwargs = {"eps": config["rms_norm_eps"], "device": device, "dtype": dtype}
        hidden = config["hidden_size"]
        self.input_layernorm = RMSNorm(hidden, **norm_kwargs)
        self.post_attention_layernorm = RMSNorm(hidden, **norm_kwargs)
        self.pre_feedforward_layernorm = RMSNorm(hidden, **norm_kwargs)
        self.post_feedforward_layernorm = RMSNorm(hidden, **norm_kwargs)

    def forward(self, x, freqs, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, freqs, attention_mask=attention_mask)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x
        return x


class Gemma4PatchEmbedder(nn.Module):
    """Patch embedding with learned 2D position embeddings via one-hot lookup."""

    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        hidden_size = config["hidden_size"]
        patch_size = config["patch_size"]
        self.patch_size = patch_size
        self.position_embedding_size = config.get("position_embedding_size", 10240)

        self.input_proj = ops.Linear(
            3 * patch_size * patch_size,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.position_embedding_table = nn.Parameter(
            torch.empty(
                2, self.position_embedding_size, hidden_size, device=device, dtype=dtype
            )
        )

    def forward(self, patches, pixel_position_ids):
        """
        patches: [B, num_patches, 3*patch_size²] in [0,1] range (normalized to [-1,1] inside, matching HF)
        pixel_position_ids: [B, num_patches, 2] with (x,y) positions, (-1,-1) for padding
        """
        hidden_states = self.input_proj(
            (2.0 * (patches - 0.5)).to(self.input_proj.weight.dtype)
        )

        clamped_positions = pixel_position_ids.clamp(min=0)
        pos_table = self.position_embedding_table.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        position_embeddings = (
            pos_table[0][clamped_positions[..., 0]]
            + pos_table[1][clamped_positions[..., 1]]
        )

        # Zero out position embeddings for padding patches (matching HF)
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        position_embeddings = torch.where(
            padding_positions.unsqueeze(-1), 0.0, position_embeddings
        )

        return hidden_states + position_embeddings


class Gemma4VisionEncoderLayers(nn.Module):
    """Wrapper to produce state dict keys as encoder.layers.X.*"""

    def __init__(self, config, dtype=None, device=None, ops=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Gemma4VisionLayer(config, device=device, dtype=dtype, ops=ops)
                for _ in range(config["num_hidden_layers"])
            ]
        )


class Gemma4VisionEncoder(nn.Module):
    def __init__(self, config, dtype=None, device=None, ops=None):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.head_dim = config.get(
            "head_dim", config["hidden_size"] // config["num_attention_heads"]
        )
        self.patch_size = config["patch_size"]
        self.pooling_kernel_size = config.get("pooling_kernel_size", 3)
        self.root_hidden_size = self.hidden_size**0.5

        self.patch_embedder = Gemma4PatchEmbedder(
            config, device=device, dtype=dtype, ops=ops
        )
        self.encoder = Gemma4VisionEncoderLayers(
            config, dtype=dtype, device=device, ops=ops
        )

    def forward(self, pixel_values, max_soft_tokens=None):
        """
        pixel_values: [B, C, H, W] in [0,1] range
        max_soft_tokens: if provided, pad to max_soft_tokens * k² total patches
        """
        batch_size, _, height, width = pixel_values.shape
        ps = self.patch_size
        k = self.pooling_kernel_size
        patches_h, patches_w = height // ps, width // ps
        num_patches = patches_h * patches_w
        output_length = (
            max_soft_tokens if max_soft_tokens is not None else num_patches // (k * k)
        )
        n_padding = output_length * k * k - num_patches

        # Patchify and build position grid
        patches = pixel_values.reshape(batch_size, -1, patches_h, ps, patches_w, ps)
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(batch_size, num_patches, -1)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(patches_h, device=pixel_values.device),
            torch.arange(patches_w, device=pixel_values.device),
            indexing="ij",
        )
        position_ids = (
            torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        # Append zero-pixel padding with (-1,-1) positions
        if n_padding > 0:
            patches = torch.cat(
                [patches, patches.new_zeros(batch_size, n_padding, patches.shape[-1])],
                dim=1,
            )
            position_ids = torch.cat(
                [position_ids, position_ids.new_full((batch_size, n_padding, 2), -1)],
                dim=1,
            )

        padding = (position_ids == -1).all(dim=-1)

        # Embed, encode, pool
        x = self.patch_embedder(patches, position_ids)
        freqs = _compute_vision_2d_rope(
            self.head_dim, position_ids, device=pixel_values.device
        )
        freqs = tuple(t.to(x.dtype) for t in freqs)
        if n_padding > 0:
            mask = (
                padding.unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, 1, position_ids.shape[1], -1)
            )
            mask = torch.zeros_like(mask, dtype=x.dtype).masked_fill_(
                mask, torch.finfo(x.dtype).min
            )
        else:
            mask = None

        for layer in self.encoder.layers:
            x = layer(x, freqs, attention_mask=mask)

        if n_padding > 0:
            x = x.masked_fill(padding.unsqueeze(-1), 0.0)

        # Average pool by spatial position
        clamped = position_ids.clamp(min=0)
        max_x = clamped[:, :, 0].max(dim=-1, keepdim=True)[0] + 1
        ki = torch.div(clamped, k, rounding_mode="floor")
        ki = ki[:, :, 0] + (max_x // k) * ki[:, :, 1]
        weights = torch.nn.functional.one_hot(ki.long(), output_length).float() / (
            k * k
        )
        x = (weights.transpose(1, 2) @ x.float()).to(x.dtype)

        # Strip empty output tokens
        valid_out = ~((weights == 0).all(dim=1))
        if valid_out.any() and not valid_out.all():
            x = x[:, valid_out[0]] if batch_size > 1 else x[valid_out].unsqueeze(0)

        return x * self.root_hidden_size


class Gemma4RMSNormProjector(nn.Module):
    """Shared projector: parameterless RMSNorm → linear. Used for both vision and audio."""

    def __init__(self, in_dim, out_dim, dtype=None, device=None, ops=None):
        super().__init__()
        self.embedding_projection = ops.Linear(
            in_dim, out_dim, bias=False, device=device, dtype=dtype
        )

    def forward(self, x):
        return self.embedding_projection(rms_norm(x))
