"""H3 Qwen3-VL visual features, adapted from ComfyUI 1a14b82e (GPL-3.0).

Only qwen35.py's exercised vision encoder and qwen3vl.py's DeepStack mergers;
no language-generation or Comfy runtime dependencies.
"""

import math

import comfy_kitchen as ck
import torch
from torch import nn
from torch.nn import functional as F

from latentslate_engine.torch_attention import attention

from .weights import Linear


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight.to(x.dtype),
            self.bias.to(x.dtype),
            self.eps,
        )


class Conv3d(nn.Conv3d):
    def forward(self, x):
        return F.conv3d(
            x,
            self.weight.to(x.dtype),
            self.bias.to(x.dtype),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class VisionOperations:
    Linear = Linear
    LayerNorm = LayerNorm
    Conv3d = Conv3d
    Embedding = nn.Embedding


def apply_rope(q, k, positions):
    cos, sin, neg_sin = positions
    half = sin.shape[-1]
    matrix = torch.stack((cos[..., :half], neg_sin, sin, cos[..., half:]), dim=-1)
    return ck.apply_rope_split_half(q, k, matrix.reshape(*matrix.shape[:-1], 2, 2))


def vision_attention(q, k, v, heads, skip_reshape=True):
    return attention(q, k, v).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


CONFIG = {
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "depth": 27,
    "deepstack_visual_indexes": [8, 16, 24],
    "num_heads": 16,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
    "out_hidden_size": 5120,
}


def image_patches(images):
    """Qwen visual preprocessing; this never determines the output canvas."""
    _, height, width, channels = images.shape
    if channels != 3:
        raise ValueError("H3 reference images must have three RGB channels")
    h_bar, w_bar = round(height / 32) * 32, round(width / 32) * 32
    if h_bar * w_bar > 12845056:
        beta = math.sqrt(height * width / 12845056)
        h_bar = max(32, math.floor(height / beta / 32) * 32)
        w_bar = max(32, math.floor(width / beta / 32) * 32)
    elif h_bar * w_bar < 3136:
        beta = math.sqrt(3136 / (height * width))
        h_bar = math.ceil(height * beta / 32) * 32
        w_bar = math.ceil(width * beta / 32) * 32
    resized = F.interpolate(
        images[:1].permute(0, 3, 1, 2),
        size=(h_bar, w_bar),
        mode="bilinear",
        align_corners=False,
    )[0]
    normalized = resized.clone()
    for channel in range(3):
        normalized[channel] = (resized[channel] - 0.5) / 0.5
    grid_h, grid_w = h_bar // 16, w_bar // 16
    grid = torch.tensor([[1, grid_h, grid_w]], device=images.device, dtype=torch.long)
    pixels = normalized.unsqueeze(0).repeat(2, 1, 1, 1)
    patches = pixels.reshape(1, 2, 3, grid_h // 2, 2, 16, grid_w // 2, 2, 16)
    return patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(
        grid_h * grid_w, 1536
    ), grid


def image_positions(spans, sequence_length, device):
    """Preserve Qwen's three-axis positions around expanded image spans."""
    positions = torch.zeros((3, sequence_length), device=device)
    offset = 0
    for index, (start, size, grid) in enumerate(spans):
        if index == 0:
            positions[:, :start] = torch.arange(start, device=device)
        end = start + size
        longest = int(grid.max()) // 2
        next_start = longest + start
        positions[:, end:] = torch.arange(
            next_start + offset,
            next_start + sequence_length - end + offset,
            device=device,
        )
        positions[0, start:end] = start + offset
        height, width = int(grid[0, 1]) // 2, int(grid[0, 2]) // 2
        positions[1, start:end] = (
            torch.arange(start + offset, start + height + offset, device=device)
            .unsqueeze(1)
            .repeat(1, math.ceil(size / height))
            .flatten()[:size]
        )
        positions[2, start:end] = (
            torch.arange(start + offset, start + width + offset, device=device)
            .unsqueeze(0)
            .repeat(math.ceil(size / width), 1)
            .flatten()[:size]
        )
        offset += longest - size
    return positions


def video_patches(frames):
    """Two real video frames occupy one Qwen temporal patch (minimax.py)."""
    count, height, width, channels = frames.shape
    if count != 2 or channels != 3:
        raise ValueError("H3 vision video blocks require two RGB frames")
    h_bar, w_bar = round(height / 32) * 32, round(width / 32) * 32
    if h_bar * w_bar > 12845056:
        beta = math.sqrt(height * width / 12845056)
        h_bar = max(32, math.floor(height / beta / 32) * 32)
        w_bar = max(32, math.floor(width / beta / 32) * 32)
    elif h_bar * w_bar < 3136:
        beta = math.sqrt(3136 / (height * width))
        h_bar = math.ceil(height * beta / 32) * 32
        w_bar = math.ceil(width * beta / 32) * 32
    pixels = F.interpolate(
        frames.permute(0, 3, 1, 2),
        size=(h_bar, w_bar),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor([0.5, 0.5, 0.5], device=pixels.device).view(1, 3, 1, 1)
    pixels = (pixels - mean) / mean
    grid_h, grid_w = h_bar // 16, w_bar // 16
    patches = pixels.reshape(1, 2, 3, grid_h // 2, 2, 16, grid_w // 2, 2, 16)
    grid = torch.tensor([[1, grid_h, grid_w]], device=frames.device, dtype=torch.long)
    return patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(
        grid_h * grid_w, 1536
    ), grid


class VisionPatchEmbed(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.patch_size = config["patch_size"]
        self.temporal_patch_size = config["temporal_patch_size"]
        self.in_channels = config["in_channels"]
        self.embed_dim = config["hidden_size"]
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = ops.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        x = x.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(x).view(-1, self.embed_dim)


class VisionMLP(nn.Module):
    def __init__(
        self, hidden_size, intermediate_size, device=None, dtype=None, ops=None
    ):
        super().__init__()

        self.linear_fc1 = ops.Linear(
            hidden_size, intermediate_size, bias=True, device=device, dtype=dtype
        )
        self.linear_fc2 = ops.Linear(
            intermediate_size, hidden_size, bias=True, device=device, dtype=dtype
        )

    def forward(self, hidden_state):
        return self.linear_fc2(
            F.gelu(self.linear_fc1(hidden_state), approximate="tanh")
        )


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen):
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class VisionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, device=None, dtype=None, ops=None):
        super().__init__()

        self.dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = ops.Linear(
            self.dim, self.dim * 3, bias=True, device=device, dtype=dtype
        )
        self.proj = ops.Linear(self.dim, self.dim, device=device, dtype=dtype)

    def forward(self, x, cu_seqlens, position_embeddings, optimized_attention=None):
        seq_length = x.shape[0]
        query_states, key_states, value_states = (
            self.qkv(x)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query_states, key_states = apply_rope(
            query_states, key_states, position_embeddings
        )

        # Process per-sequence attention
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits = torch.split(query_states, lengths, dim=0)
        k_splits = torch.split(key_states, lengths, dim=0)
        v_splits = torch.split(value_states, lengths, dim=0)

        attn_outputs = []
        for q, k, v in zip(q_splits, k_splits, v_splits):
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            attn_outputs.append(
                optimized_attention(q, k, v, self.num_heads, skip_reshape=True)
            )

        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.proj(attn_output)


class VisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        device=None,
        dtype=None,
        ops=None,
    ):
        super().__init__()

        self.norm1 = ops.LayerNorm(hidden_size, eps=1e-6, device=device, dtype=dtype)
        self.norm2 = ops.LayerNorm(hidden_size, eps=1e-6, device=device, dtype=dtype)
        self.attn = VisionAttention(
            hidden_size, num_heads, device=device, dtype=dtype, ops=ops
        )
        self.mlp = VisionMLP(
            hidden_size, intermediate_size, device=device, dtype=dtype, ops=ops
        )

    def forward(self, x, cu_seqlens, position_embeddings, optimized_attention=None):
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            optimized_attention=optimized_attention,
        )
        return x + self.mlp(self.norm2(x))


class VisionPatchMerger(nn.Module):
    def __init__(
        self,
        hidden_size,
        spatial_merge_size,
        out_hidden_size,
        device=None,
        dtype=None,
        ops=None,
    ):
        super().__init__()

        merge_dim = hidden_size * (spatial_merge_size**2)
        self.norm = ops.LayerNorm(hidden_size, eps=1e-6, device=device, dtype=dtype)
        self.linear_fc1 = ops.Linear(merge_dim, merge_dim, device=device, dtype=dtype)
        self.linear_fc2 = ops.Linear(
            merge_dim, out_hidden_size, device=device, dtype=dtype
        )
        self.merge_dim = merge_dim

    def forward(self, x):
        x = self.norm(x).view(-1, self.merge_dim)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class VisionModel(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.spatial_merge_size = config["spatial_merge_size"]
        self.patch_size = config["patch_size"]
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.num_position_embeddings = config["num_position_embeddings"]

        self.patch_embed = VisionPatchEmbed(config, device=device, dtype=dtype, ops=ops)
        self.pos_embed = ops.Embedding(
            self.num_position_embeddings, self.hidden_size, device=device, dtype=dtype
        )
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)
        self.rotary_pos_emb = VisionRotaryEmbedding(
            self.hidden_size // self.num_heads // 2
        )
        self.blocks = nn.ModuleList(
            [
                VisionBlock(
                    self.hidden_size,
                    self.num_heads,
                    config["intermediate_size"],
                    device=device,
                    dtype=dtype,
                    ops=ops,
                )
                for _ in range(config["depth"])
            ]
        )
        self.merger = VisionPatchMerger(
            self.hidden_size,
            self.spatial_merge_size,
            config["out_hidden_size"],
            device=device,
            dtype=dtype,
            ops=ops,
        )
        self.deepstack_visual_indexes = []  # DeepStack, per-layer visual features (Qwen3-VL)
        self.deepstack_merger_list = None

    def rot_pos_emb(self, grid_thw):
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()
        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        total_tokens = sum(int(t * h * w) for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_thw_list:
            num_frames, height, width = int(num_frames), int(height), int(width)
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )
            row_idx = row_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)
            col_idx = col_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens
        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_thw_list = grid_thw.tolist()
        grid_ts = [int(row[0]) for row in grid_thw_list]
        grid_hs = [int(row[1]) for row in grid_thw_list]
        grid_ws = [int(row[2]) for row in grid_thw_list]
        device = self.pos_embed.weight.device
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for t, h, w in grid_thw_list:
            h, w = int(h), int(w)
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device=device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device=device)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for j in range(4):
                idx_list[j].extend(indices[j].tolist())
                weight_list[j].extend(weights[j].tolist())
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=device
        )
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )
        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(
                    t, h // merge_size, merge_size, w // merge_size, merge_size, -1
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    def forward(self, x, grid_thw):
        x = self.patch_embed(x)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw).to(x.device)
        x = x + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw).to(x.device)
        seq_len = x.shape[0]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().unsqueeze(-2)
        sin = emb.sin().unsqueeze(-2)
        sin_half = sin.shape[-1] // 2
        position_embeddings = (cos, sin[..., :sin_half], -sin[..., sin_half:])
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        optimized_attention = vision_attention
        deepstack_features = []
        for layer_num, blk in enumerate(self.blocks):
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                optimized_attention=optimized_attention,
            )
            if (
                self.deepstack_merger_list is not None
                and layer_num in self.deepstack_visual_indexes
            ):
                deepstack_features.append(
                    self.deepstack_merger_list[
                        self.deepstack_visual_indexes.index(layer_num)
                    ](x)
                )
        merged = self.merger(x)
        if self.deepstack_merger_list is not None:
            return merged, deepstack_features
        return merged


class DeepstackMerger(nn.Module):
    # DeepStack merger: postshuffle LayerNorm (applied after spatial merge), unlike the main merger.
    def __init__(
        self,
        hidden_size,
        spatial_merge_size,
        out_hidden_size,
        device=None,
        dtype=None,
        ops=None,
    ):
        super().__init__()
        self.merge_dim = hidden_size * (spatial_merge_size**2)
        self.norm = ops.LayerNorm(self.merge_dim, eps=1e-6, device=device, dtype=dtype)
        self.linear_fc1 = ops.Linear(
            self.merge_dim, self.merge_dim, device=device, dtype=dtype
        )
        self.linear_fc2 = ops.Linear(
            self.merge_dim, out_hidden_size, device=device, dtype=dtype
        )

    def forward(self, x):
        x = self.norm(x.view(-1, self.merge_dim))
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class H3VisionModel(VisionModel):
    # Qwen3.5 vision + DeepStack
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__(config, device=device, dtype=dtype, ops=ops)
        self.deepstack_visual_indexes = config["deepstack_visual_indexes"]
        self.deepstack_merger_list = nn.ModuleList(
            [
                DeepstackMerger(
                    self.hidden_size,
                    self.spatial_merge_size,
                    config["out_hidden_size"],
                    device=device,
                    dtype=dtype,
                    ops=ops,
                )
                for _ in self.deepstack_visual_indexes
            ]
        )
