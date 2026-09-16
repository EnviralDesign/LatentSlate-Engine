"""SDXL CLIP-L/G conditioning, adapted from ComfyUI 1a14b82e (GPL-3.0).

Sources: sd1_clip.py, sdxl_clip.py and clip_model.py. Textual inversion is
outside this ordinary text-only path.
"""

import torch
from safetensors import safe_open
from torch.nn import functional as F
from transformers import CLIPTokenizer

from latentslate_engine.torch_attention import attention


def weighted_segments(text, current=1.0):
    pieces, item, depth = [], "", 0
    for char in text:
        if char == "(":
            if depth == 0 and item:
                pieces.append(item)
                item = ""
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                pieces.append(item + char)
                item = ""
                continue
        item += char
    if item:
        pieces.append(item)
    result = []
    for item in pieces:
        if len(item) >= 2 and item.startswith("(") and item.endswith(")"):
            item = item[1:-1]
            weight = current * 1.1
            colon = item.rfind(":")
            if colon > 0:
                try:
                    weight = float(item[colon + 1 :])
                    item = item[:colon]
                except ValueError:
                    pass
            result.extend(weighted_segments(item, weight))
        else:
            result.append((item, current))
    return result


def tokenize(tokenizer, text, pad):
    escaped = text.replace("\\)", "\0\1").replace("\\(", "\0\2")
    batch = [(49406, 1.0)]
    batches = []
    for segment, weight in weighted_segments(escaped):
        segment = segment.replace("\0\1", ")").replace("\0\2", "(")
        group = [(int(t), weight) for t in tokenizer(segment)["input_ids"][1:-1]]
        large = len(group) >= 8
        while group:
            remaining = 76 - len(batch)
            if len(group) > remaining:
                if large:
                    batch.extend(group[:remaining])
                    group = group[remaining:]
                batch.append((49407, 1.0))
                batch.extend([(pad, 1.0)] * (77 - len(batch)))
                batches.append(batch)
                batch = [(49406, 1.0)]
            else:
                batch.extend(group)
                group = []
    batch.append((49407, 1.0))
    batch.extend([(pad, 1.0)] * (77 - len(batch)))
    batches.append(batch)
    return batches


class TextEncoder:
    """Retain checkpoint CLIP weights; cast to FP32 at each exercised operation."""

    def __init__(self, checkpoint, tokenizer, device):
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer, local_files_only=True)
        self.device = device
        self.weights = {}
        with safe_open(checkpoint, framework="pt") as source:
            tensor_names = source.keys()
            for prefix, name in (
                ("conditioner.embedders.0.transformer.", "l"),
                ("conditioner.embedders.1.model.", "g"),
            ):
                self.weights[name] = {
                    key[len(prefix) :]: source.get_tensor(key).to(
                        device=device, dtype=torch.float16
                    )
                    for key in tensor_names
                    if key.startswith(prefix)
                }

    def _encode(self, tokens, name):
        weights = self.weights[name]
        big = name == "g"
        dim, depth, heads = (1280, 32, 20) if big else (768, 12, 12)

        def linear(x, key):
            bias = weights.get(key + ".bias")
            return F.linear(
                x,
                weights[key + ".weight"].to(x.dtype),
                None if bias is None else bias.to(x.dtype),
            )

        def norm(x, key):
            return F.layer_norm(
                x,
                (dim,),
                weights[key + ".weight"].to(x.dtype),
                weights[key + ".bias"].to(x.dtype),
            )

        ids = torch.tensor(tokens, dtype=torch.long, device=self.device)
        token_key = (
            "token_embedding.weight"
            if big
            else "text_model.embeddings.token_embedding.weight"
        )
        position_key = (
            "positional_embedding"
            if big
            else "text_model.embeddings.position_embedding.weight"
        )
        x = F.embedding(ids, weights[token_key]).float() + weights[position_key].float()
        mask = torch.full((77, 77), -torch.finfo(x.dtype).max, device=x.device).triu_(1)
        for index in range(depth):
            prefix = (
                f"transformer.resblocks.{index}."
                if big
                else f"text_model.encoder.layers.{index}."
            )
            normalized = norm(x, prefix + ("ln_1" if big else "layer_norm1"))
            if big:
                w = weights[prefix + "attn.in_proj_weight"].chunk(3)
                b = weights[prefix + "attn.in_proj_bias"].chunk(3)
                projections = [
                    F.linear(normalized, wi.float(), bi.float()) for wi, bi in zip(w, b)
                ]
            else:
                projections = [
                    linear(normalized, prefix + "self_attn." + q + "_proj")
                    for q in ("q", "k", "v")
                ]
            q, k, v = [
                t.reshape(t.shape[0], 77, heads, dim // heads).transpose(1, 2)
                for t in projections
            ]
            out = attention(q, k, v, mask).transpose(1, 2).reshape(x.shape)
            x += linear(
                out, prefix + ("attn.out_proj" if big else "self_attn.out_proj")
            )
            h = linear(
                norm(x, prefix + ("ln_2" if big else "layer_norm2")),
                prefix + ("mlp.c_fc" if big else "mlp.fc1"),
            )
            h = F.gelu(h) if big else h * torch.sigmoid(1.702 * h)
            x += linear(h, prefix + ("mlp.c_proj" if big else "mlp.fc2"))
            if index == depth - 2:
                hidden = x.clone()
        x = norm(x, "ln_final" if big else "text_model.final_layer_norm")
        eos = (ids == 49407).int().argmax(dim=-1)
        pooled = x[torch.arange(x.shape[0], device=x.device), eos]
        if big:
            if "text_projection" in weights:
                pooled = F.linear(
                    pooled, weights["text_projection"].T.contiguous().float()
                )
            else:
                pooled = linear(pooled, "text_projection")
        return hidden, pooled

    @torch.inference_mode()
    def encode(self, text):
        outputs = {}
        for name, pad in (("g", 0), ("l", 49407)):
            batches = tokenize(self.tokenizer, text, pad)
            weighted = any(weight != 1 for batch in batches for _, weight in batch)
            tokens = [[t for t, _ in batch] for batch in batches]
            if weighted:
                tokens.append([49406, 49407] + [pad] * 75)
            hidden, pooled = self._encode(tokens, name)
            sections = []
            for index, batch in enumerate(batches):
                z = hidden[index : index + 1]
                if weighted:
                    for j, (_, weight) in enumerate(batch):
                        if weight != 1:
                            z[0, j] = (z[0, j] - hidden[-1, j]) * weight + hidden[-1, j]
                sections.append(z)
            outputs[name] = (torch.cat(sections, dim=-2).cpu(), pooled[:1].cpu())
        length = min(outputs["l"][0].shape[1], outputs["g"][0].shape[1])
        return torch.cat(
            (outputs["l"][0][:, :length], outputs["g"][0][:, :length]), dim=-1
        ), outputs["g"][1]
