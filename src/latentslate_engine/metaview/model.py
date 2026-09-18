"""MetaView transformer math adapted from the official MetaView architecture (Apache-2.0)
and the exercised Comfy port. Uses Engine Qwen operations; no Comfy runtime dependency.
"""
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from latentslate_engine.qwen2511.model import QwenTimestepProjEmbeddings, LastLayer
from .prope import PropeDotProductAttention
from .rope import QwenEmbedRope, apply_rotary_emb_qwen, approximate_gelu

_DIM = 3072
_NUM_HEADS = 24
_HEAD_DIM = 128
_NUM_LAYERS = 60
_JOINT_DIM = 3584          # Qwen2.5-VL text hidden dim (txt_norm / txt_in input)
_3D_DIM = 6144             # DA3 feature channel dim (_3D_in input)
_ADD_IN_DIM = 3072         # projected 3D-feature width
_PATCH = 2
_IN_CHANNELS = 64          # 16 latent channels * patch(2) * patch(2)
_OUT_CHANNELS = 16
# PRoPE dimension arrangement. The MetaView reference demo (src/inference.py:86,
# src/inference_lowvram.py) uses the 4-term arrangement [64, 20, 20, 24] with the
# DEPTH channel active (len == 4 -> depth term), and feeds the DA3 depth map into
# PRoPE precompute. NOTE: MetaViewPipeline's signature default [16, 56, 56] is
# overridden by the demo and is NOT what the released weights were trained with.
_PROPE_DIM_ARRANGE = [64, 20, 20, 24]
_PROPE_FREQ_BASE = 10000.0


# ---------------------------------------------------------------------------
# attention helper (matches diffsynth qwen_image_flash_attention non-flash path)
# ---------------------------------------------------------------------------
def _attention(q, k, v, num_heads, attn_mask=None):
    # q,k,v: [b, heads, seq, head_dim]
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


# ---------------------------------------------------------------------------
# feed forward (diffsynth QwenFeedForward: ApproximateGELU = x*sigmoid(1.702x))
# ---------------------------------------------------------------------------
class _ProjGELU(nn.Module):
    """FeedForward.net[0] — holds the ``.proj`` Linear then the sigmoid-GELU activation."""

    def __init__(self, dim_in, dim_out, dtype, device, operations):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        return approximate_gelu(self.proj(x))


class MetaViewFeedForward(nn.Module):
    def __init__(self, dim, dim_out, dtype, device, operations):
        super().__init__()
        inner = dim * 4
        self.net = nn.ModuleList([
            _ProjGELU(dim, inner, dtype, device, operations),          # net.0 (.proj)
            nn.Dropout(0.0),                                           # net.1
            operations.Linear(inner, dim_out, bias=True, dtype=dtype, device=device),  # net.2
        ])

    def forward(self, x):
        for m in self.net:
            x = m(x)
        return x


# ---------------------------------------------------------------------------
# main double-stream attention (diffsynth QwenDoubleStreamAttention)
# In MetaView's shipped config this branch is invoked with prope=None, so image tokens receive
# the standard Qwen 2D RoPE (via image_rotary_emb) exactly like base Qwen-Image-Edit.
# ---------------------------------------------------------------------------
class QwenDoubleStreamAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, dtype, device, operations):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.to_k = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.to_v = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.norm_q = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)

        self.add_q_proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.add_k_proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.add_v_proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)

        self.to_out = nn.Sequential(operations.Linear(dim, dim, dtype=dtype, device=device))
        self.to_add_out = operations.Linear(dim, dim, dtype=dtype, device=device)

    def forward(self, image, text, image_rotary_emb=None, attention_mask=None, prope=None):
        img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
        txt_q, txt_k, txt_v = self.add_q_proj(text), self.add_k_proj(text), self.add_v_proj(text)
        seq_txt = txt_q.shape[1]

        img_q = rearrange(img_q, "b s (h d) -> b h s d", h=self.num_heads)
        img_k = rearrange(img_k, "b s (h d) -> b h s d", h=self.num_heads)
        img_v = rearrange(img_v, "b s (h d) -> b h s d", h=self.num_heads)
        txt_q = rearrange(txt_q, "b s (h d) -> b h s d", h=self.num_heads)
        txt_k = rearrange(txt_k, "b s (h d) -> b h s d", h=self.num_heads)
        txt_v = rearrange(txt_v, "b s (h d) -> b h s d", h=self.num_heads)

        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)
        txt_q, txt_k = self.norm_added_q(txt_q), self.norm_added_k(txt_k)

        if prope is not None and image_rotary_emb is not None:
            _, txt_freqs = image_rotary_emb
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs)
            img_q = prope._apply_to_q(img_q)
            img_k = prope._apply_to_kv(img_k)
        elif image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs)

        joint_q = torch.cat([txt_q, img_q], dim=2)
        joint_k = torch.cat([txt_k, img_k], dim=2)
        joint_v = torch.cat([txt_v, img_v], dim=2)

        out = _attention(joint_q, joint_k, joint_v, self.num_heads, attn_mask=attention_mask).to(joint_q.dtype)
        txt_out = out[:, :seq_txt, :]
        img_out = out[:, seq_txt:, :]

        img_out = self.to_out[0](img_out)
        txt_out = self.to_add_out(txt_out)
        return img_out, txt_out


# ---------------------------------------------------------------------------
# grafted parallel 3D attention (diffsynth MetaViewSelfAttention3D)
# ---------------------------------------------------------------------------
class MetaViewSelfAttention3D(nn.Module):
    def __init__(self, dim_a, dim_b, num_heads, head_dim, merge_3D, dtype, device, operations):
        super().__init__()
        self.merge_3D = merge_3D
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = operations.Linear(dim_a, dim_a, dtype=dtype, device=device)
        self.to_k = operations.Linear(dim_a, dim_a, dtype=dtype, device=device)
        self.to_v = operations.Linear(dim_a, dim_a, dtype=dtype, device=device)
        self.to_q_3D = operations.Linear(dim_b, dim_a, dtype=dtype, device=device)
        self.to_k_3D = operations.Linear(dim_b, dim_a, dtype=dtype, device=device)
        self.to_v_3D = operations.Linear(dim_b, dim_a, dtype=dtype, device=device)

        self.norm_q = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)

        self.to_out = nn.Sequential(operations.Linear(dim_a, dim_a, dtype=dtype, device=device))

    def forward(self, image, feat_3D, attention_mask=None, prope=None, add_prope=None):
        img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
        _3D_q, _3D_k, _3D_v = self.to_q_3D(feat_3D), self.to_k_3D(feat_3D), self.to_v_3D(feat_3D)
        seq_img = img_q.shape[1]

        img_q = rearrange(img_q, "b s (h d) -> b h s d", h=self.num_heads)
        img_k = rearrange(img_k, "b s (h d) -> b h s d", h=self.num_heads)
        img_v = rearrange(img_v, "b s (h d) -> b h s d", h=self.num_heads)
        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)

        _3D_q = rearrange(_3D_q, "b s (h d) -> b h s d", h=self.num_heads)
        _3D_k = rearrange(_3D_k, "b s (h d) -> b h s d", h=self.num_heads)
        _3D_v = rearrange(_3D_v, "b s (h d) -> b h s d", h=self.num_heads)
        _3D_q, _3D_k = self.norm_added_q(_3D_q), self.norm_added_k(_3D_k)

        if prope is not None:
            img_q = prope._apply_to_q(img_q)
            img_k = prope._apply_to_kv(img_k)
            img_v = prope._apply_to_kv(img_v)
            _3D_q = add_prope._apply_to_q(_3D_q)
            _3D_k = add_prope._apply_to_kv(_3D_k)
            _3D_v = add_prope._apply_to_kv(_3D_v)

        joint_q = torch.cat([img_q, _3D_q], dim=2)
        joint_k = torch.cat([img_k, _3D_k], dim=2)
        joint_v = torch.cat([img_v, _3D_v], dim=2)

        num_heads = img_q.shape[1]
        out = _attention(joint_q, joint_k, joint_v, num_heads, attn_mask=attention_mask).to(img_q.dtype)

        img_out = out[:, :seq_img, :]
        _3D_out = out[:, seq_img:, :]

        if prope is not None:
            img_out = rearrange(img_out, "b s (n d) -> b n s d", n=num_heads)
            img_out = prope._apply_to_o(img_out)
            img_out = rearrange(img_out, "b n s d -> b s (n d)", n=num_heads)
        img_out = self.to_out(img_out)

        if self.merge_3D:
            if add_prope is not None:
                _3D_out = rearrange(_3D_out, "b s (n d) -> b n s d", n=num_heads)
                _3D_out = add_prope._apply_to_o(_3D_out)
                _3D_out = rearrange(_3D_out, "b n s d -> b s (n d)", n=num_heads)
            _3D_out = self.to_out(_3D_out)
            return img_out, _3D_out
        return img_out, None


# ---------------------------------------------------------------------------
# grafted transformer block (diffsynth MetaViewTransformerBlock, merge_3D=True)
# ---------------------------------------------------------------------------
class MetaViewTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, add_in_dim, merge_3D, eps, dtype, device, operations):
        super().__init__()
        self.merge_3D = merge_3D

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device),
        )
        self.img_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.attn = QwenDoubleStreamAttention(dim, num_heads, head_dim, dtype, device, operations)
        self.prope_attn = MetaViewSelfAttention3D(dim, add_in_dim, num_heads, head_dim, merge_3D, dtype, device, operations)

        self.img_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_mlp = MetaViewFeedForward(dim, dim, dtype, device, operations)

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device),
        )
        self.txt_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_mlp = MetaViewFeedForward(dim, dim, dtype, device, operations)

    @staticmethod
    def _modulate(x, mod_params):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(self, image, text, temb, image_rotary_emb=None, attention_mask=None,
                prope=None, add_prope=None, add_attn=True, feat_3D=None):
        img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)
        txt_mod_attn, txt_mod_mlp = self.txt_mod(temb).chunk(2, dim=-1)

        img_normed = self.img_norm1(image)
        img_modulated, img_gate = self._modulate(img_normed, img_mod_attn)

        txt_normed = self.txt_norm1(text)
        txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)

        if self.merge_3D:
            feat_3D_modulated = self.img_norm1(feat_3D)
        else:
            feat_3D_modulated = feat_3D

        _3D_prope_out = None
        if add_attn and prope is not None:
            img_prope_out, _3D_prope_out = self.prope_attn(
                image=img_modulated, feat_3D=feat_3D_modulated,
                attention_mask=attention_mask, prope=prope, add_prope=add_prope,
            )
            img_attn_out, txt_attn_out = self.attn(
                image=img_modulated, text=txt_modulated,
                image_rotary_emb=image_rotary_emb, attention_mask=attention_mask, prope=None,
            )
            img_attn_out = img_attn_out + img_prope_out
        else:
            img_attn_out, txt_attn_out = self.attn(
                image=img_modulated, text=txt_modulated,
                image_rotary_emb=image_rotary_emb, attention_mask=attention_mask, prope=prope,
            )

        image = image + img_gate * img_attn_out
        text = text + txt_gate * txt_attn_out

        img_normed_2 = self.img_norm2(image)
        img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp)
        txt_normed_2 = self.txt_norm2(text)
        txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)

        image = image + img_gate_2 * self.img_mlp(img_modulated_2)
        text = text + txt_gate_2 * self.txt_mlp(txt_modulated_2)

        if self.merge_3D:
            feat_3D = feat_3D + _3D_prope_out
            feat_3D = feat_3D + self.img_mlp(self.img_norm2(feat_3D))
            return text, image, feat_3D
        return text, image, feat_3D


# ---------------------------------------------------------------------------
# full grafted DiT
# ---------------------------------------------------------------------------
class MetaViewDiT(nn.Module):
    """Qwen-Image-Edit transformer with camera-aware parallel geometry attention."""

    def __init__(self, image_model=None, dtype=None, device=None, operations=None, **kwargs):
        nn.Module.__init__(self)
        self.dtype = dtype
        self.patch_size = _PATCH
        self.in_channels = _IN_CHANNELS
        self.out_channels = _OUT_CHANNELS
        self.inner_dim = _DIM
        self.num_heads = _NUM_HEADS
        self.head_dim = _HEAD_DIM
        self.merge_3D = True

        # Text-stream RoPE (image stream is handled by PRoPE / Qwen rope via this too).
        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

        # Base Qwen-Image-Edit modules (names match the artifact keys exactly).
        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=768,
            dtype=dtype, device=device, operations=operations)
        self.txt_norm = operations.RMSNorm(_JOINT_DIM, eps=1e-6, dtype=dtype, device=device)
        self.img_in = operations.Linear(_IN_CHANNELS, self.inner_dim, dtype=dtype, device=device)
        self.txt_in = operations.Linear(_JOINT_DIM, self.inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList([
            MetaViewTransformerBlock(
                dim=self.inner_dim, num_heads=self.num_heads, head_dim=self.head_dim,
                add_in_dim=_ADD_IN_DIM, merge_3D=True, eps=1e-6,
                dtype=dtype, device=device, operations=operations)
            for _ in range(_NUM_LAYERS)
        ])

        self.norm_out = LastLayer(self.inner_dim, self.inner_dim, dtype=dtype, device=device, operations=operations)
        self.proj_out = operations.Linear(self.inner_dim, _PATCH * _PATCH * self.out_channels, bias=True, dtype=dtype, device=device)

        # Grafted 3D-feature input projection.
        self._3D_in = operations.Linear(_3D_DIM, _ADD_IN_DIM, dtype=dtype, device=device)

    # ---- helpers -----------------------------------------------------------
    def _patchify(self, latent):
        # latent [b, 16, H_lat, W_lat] -> tokens [b, (H_lat/2 * W_lat/2), 64]
        H = latent.shape[2] // _PATCH
        W = latent.shape[3] // _PATCH
        return rearrange(latent, "b c (H P) (W Q) -> b (H W) (c P Q)", H=H, W=W, P=_PATCH, Q=_PATCH), H, W

    def _build_prope(self, patches_x, patches_y, viewmats, ks, depth=None):
        p = PropeDotProductAttention(
            head_dim=self.head_dim, patches_x=patches_x, patches_y=patches_y,
            image_width=patches_x * 16, image_height=patches_y * 16,
            freq_base=_PROPE_FREQ_BASE, dim_arrange=_PROPE_DIM_ARRANGE,
        ).to(viewmats.device)
        p._precompute_and_cache_apply_fns(viewmats, ks, depth)
        return p

    # ---- main forward ------------------------------------------------------
    def forward(self, x, timesteps, context, attention_mask=None, ref_latents=None,
                 additional_t_cond=None, transformer_options={}, control=None,
                 metaview=None, **kwargs):
        orig_ndim = x.ndim
        if orig_ndim == 5:  # comfy passes [b, C, T=1, H, W] for image models
            x = x[:, :, 0]

        image, patches_y, patches_x = self._patchify(x)
        image_seq_len = image.shape[1]
        img_shapes = [(1, patches_y, patches_x)]

        if ref_latents is not None and len(ref_latents) > 0:
            edit = ref_latents[0]
            if edit.ndim == 5:
                edit = edit[:, :, 0]
            edit_tokens, e_H, e_W = self._patchify(edit.to(x.dtype))
            image = torch.cat([image, edit_tokens], dim=1)
            img_shapes.append((1, e_H, e_W))

        # text sequence lengths
        if attention_mask is not None:
            txt_seq_lens = attention_mask.sum(dim=1).tolist()
        else:
            txt_seq_lens = [context.shape[1]] * context.shape[0]

        image = self.img_in(image)
        temb = self.time_text_embed(timesteps, image, additional_t_cond)
        text = self.txt_in(self.txt_norm(context))

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=x.device)

        # ---- build PRoPE from cameras -------------------------------------
        prope = None
        add_prope = None
        feat_3D = None
        if metaview is not None and metaview.get("viewmats") is not None and metaview.get("ks") is not None:
            viewmats = metaview["viewmats"]
            ks = metaview["ks"]
            if viewmats.ndim == 3:
                viewmats = viewmats.unsqueeze(0)
            if ks.ndim == 3:
                ks = ks.unsqueeze(0)
            viewmats = viewmats.to(device=x.device, dtype=torch.float32)
            ks = ks.to(device=x.device, dtype=torch.float32)

            # Depth enters PRoPE (dim_arrange has 4 terms -> depth channel active),
            # bilinearly resized to the token grid, exactly like the reference
            # (MetaView_pipeline.py model_fn: F.interpolate to (h//16, w//16)).
            depth = metaview.get("depth")
            if depth is not None:
                if depth.ndim == 3:  # [n_cams, H, W] -> [1, n_cams, H, W]
                    depth = depth.unsqueeze(0)
                depth = depth.to(device=x.device, dtype=torch.float32)
                depth = F.interpolate(depth, size=(patches_y, patches_x),
                                      mode="bilinear", align_corners=False)

            prope = self._build_prope(patches_x, patches_y, viewmats, ks, depth)

            feat = metaview.get("feat3d")
            if feat is not None:
                if feat.ndim == 3:
                    feat = feat.unsqueeze(0)
                feat = feat.to(device=x.device, dtype=image.dtype)
                feat_3D = rearrange(feat, "b h w d -> b (h w) d")
                feat_3D = self._3D_in(feat_3D)
                # add_PRoPE uses the source camera only (index 1), incl. its depth
                add_prope = self._build_prope(
                    patches_x, patches_y, viewmats[:, 1:2], ks[:, 1:2],
                    depth[:, 1:2] if depth is not None else None)

        use_graft = prope is not None and feat_3D is not None

        for block in self.transformer_blocks:
            text, image, feat_3D = block(
                image=image, text=text, temb=temb,
                image_rotary_emb=image_rotary_emb, attention_mask=None,
                prope=prope, add_prope=add_prope, add_attn=use_graft, feat_3D=feat_3D,
            )

        image = self.norm_out(image, temb)
        image = self.proj_out(image)
        image = image[:, :image_seq_len]

        out = rearrange(image, "b (H W) (c P Q) -> b c (H P) (W Q)",
                        H=patches_y, W=patches_x, P=_PATCH, Q=_PATCH)
        if orig_ndim == 5:
            out = out.unsqueeze(2)
        return out
