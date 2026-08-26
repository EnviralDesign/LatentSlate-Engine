"""Standalone Gemma 3 text conditioning for the canonical LTX 2.3 T2V fixture."""

from __future__ import annotations

import math
import json

import sentencepiece as spm
import torch
from torch import nn
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout, TensorCoreNVFP4Layout
from transformers import Gemma3TextConfig, Gemma3TextModel

from .checkpoint import Ltx23Checkpoint


def _replace_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


class _MappedNvfp4Linear(nn.Module):
    """One Gemma NVFP4 linear backed directly by the mapped text checkpoint."""

    def __init__(self, checkpoint: Ltx23Checkpoint, prefix: str, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.shape = shape
        self.qdata = checkpoint.tensor(f"{prefix}.weight")
        self.scale = checkpoint.tensor(f"{prefix}.weight_scale_2")
        self.block_scale = checkpoint.tensor(f"{prefix}.weight_scale")
        if self.qdata.dtype is not torch.uint8 or self.scale.dtype is not torch.float32:
            raise ValueError(f"{prefix} is not an NVFP4 linear")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # AIMDO's direct file-reader rejects this mixed-FP4 checkpoint's slices.
        # Keep its mapped source tensor and use PyTorch's working CUDA copy path.
        qdata = self.qdata.to(device=input.device)
        scale = self.scale.to(device=input.device)
        block_scale = self.block_scale.to(device=input.device).view(torch.float8_e4m3fn)
        weight = QuantizedTensor(
            qdata,
            "TensorCoreNVFP4Layout",
            TensorCoreNVFP4Layout.Params(
                scale=scale,
                block_scale=block_scale,
                orig_dtype=torch.bfloat16,
                orig_shape=self.shape,
            ),
        )
        return torch.nn.functional.linear(input, weight)


class _MappedFp8Linear(nn.Module):
    """One mixed-checkpoint FP8 Gemma linear backed by the mapped checkpoint."""

    def __init__(self, checkpoint: Ltx23Checkpoint, prefix: str, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.shape = shape
        self.qdata = checkpoint.tensor(f"{prefix}.weight")
        self.scale = checkpoint.tensor(f"{prefix}.weight_scale")
        if self.qdata.dtype is not torch.float8_e4m3fn or self.scale.dtype is not torch.float32:
            raise ValueError(f"{prefix} is not an E4M3 FP8 linear")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qdata = self.qdata.to(device=input.device)
        scale = self.scale.to(device=input.device)
        weight = QuantizedTensor(
            qdata,
            "TensorCoreFP8Layout",
            TensorCoreFP8Layout.Params(
                scale=scale,
                orig_dtype=torch.bfloat16,
                orig_shape=self.shape,
            ),
        )
        return torch.nn.functional.linear(input, weight)


class Ltx23TextEncoder:
    """Own the canonical T2V text encoder for one LTX 2.3 model identity."""

    def __init__(
        self,
        text_checkpoint_path: str,
        model_checkpoint_path: str,
        device_index: int = 0,
    ) -> None:
        self.device = torch.device("cuda", device_index)
        self.text_checkpoint = Ltx23Checkpoint(text_checkpoint_path)
        self.model_checkpoint = Ltx23Checkpoint(model_checkpoint_path)
        self.tokenizer = spm.SentencePieceProcessor()
        if not self.tokenizer.LoadFromSerializedProto(
            self.text_checkpoint.tensor("spiece_model").numpy().tobytes()
        ):
            raise RuntimeError("unable to load the canonical Gemma sentencepiece model")

        config = Gemma3TextConfig(
            vocab_size=262208,
            hidden_size=3840,
            intermediate_size=15360,
            num_hidden_layers=48,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=256,
            rms_norm_eps=1e-6,
            sliding_window=1024,
            rope_parameters={
                "full_attention": {"rope_theta": 1_000_000.0, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"},
            },
        )
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            self.model = Gemma3TextModel(config)

        dynamic_modules = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        for name, module in dynamic_modules:
            prefix = f"model.{name}"
            quant_name = f"{prefix}.comfy_quant"
            if quant_name not in self.text_checkpoint.tensor_names:
                raise ValueError(f"missing NVFP4 metadata for {prefix}")
            quant_format = json.loads(self.text_checkpoint.tensor(quant_name).numpy().tobytes())["format"]
            if quant_format == "nvfp4":
                replacement = _MappedNvfp4Linear(
                    self.text_checkpoint, prefix, tuple(module.weight.shape)
                )
            elif quant_format == "float8_e4m3fn":
                replacement = _MappedFp8Linear(
                    self.text_checkpoint, prefix, tuple(module.weight.shape)
                )
            else:
                raise ValueError(f"unsupported canonical Gemma format {quant_format!r} for {prefix}")
            _replace_module(
                self.model,
                name,
                replacement,
            )

        for name, parameter in list(self.model.named_parameters()):
            source_name = f"model.{name}"
            if source_name not in self.text_checkpoint.tensor_names:
                raise ValueError(f"missing text parameter {source_name}")
            parent = self.model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            source = self.text_checkpoint.tensor(source_name)
            setattr(
                parent,
                parts[-1],
                nn.Parameter(source.to(device=self.device, dtype=torch.bfloat16), requires_grad=False),
            )
        self.model.embed_tokens.embed_scale = torch.tensor(
            math.sqrt(3840), device=self.device, dtype=torch.bfloat16
        )
        for attention_type, theta in (("sliding_attention", 10_000.0), ("full_attention", 1_000_000.0)):
            inv_freq = 1.0 / (
                theta
                ** (
                    torch.arange(0, 256, 2, device=self.device, dtype=torch.float32)
                    / 256
                )
            )
            setattr(self.model.rotary_emb, f"{attention_type}_inv_freq", inv_freq)
            setattr(self.model.rotary_emb, f"{attention_type}_original_inv_freq", inv_freq.clone())
        self.model.eval()

    def _tokens(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer.encode(text, out_type=int, add_bos=True, add_eos=False)
        if len(tokens) > 1024:
            raise ValueError("canonical LTX T2V text input exceeds 1024 Gemma tokens")
        padding = 1024 - len(tokens)
        input_ids = torch.tensor([[0] * padding + tokens], device=self.device, dtype=torch.long)
        attention_mask = torch.tensor([[0] * padding + [1] * len(tokens)], device=self.device, dtype=torch.long)
        return input_ids, attention_mask

    @torch.inference_mode()
    def encode(self, text: str) -> torch.Tensor:
        """Return the unprocessed LTX AV text embedding used by the transformer."""
        input_ids, attention_mask = self._tokens(text)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        active_tokens = int(attention_mask.sum().item())
        hidden_states = torch.stack(output.hidden_states, dim=1)[:, :, -active_tokens:]
        x = hidden_states.movedim(1, -1)
        x = (x * torch.rsqrt(torch.mean(x**2, dim=2, keepdim=True) + 1e-6)).flatten(start_dim=2)
        source_dim = hidden_states.shape[-1]
        video = self._project(
            "text_embedding_projection.video_aggregate_embed", x, 4096, source_dim
        )
        audio = self._project(
            "text_embedding_projection.audio_aggregate_embed", x, 2048, source_dim
        )
        return torch.cat((video, audio), dim=-1)

    def _project(self, prefix: str, x: torch.Tensor, out_features: int, source_dim: int) -> torch.Tensor:
        weight = self.model_checkpoint.tensor(f"{prefix}.weight").to(device=self.device, dtype=x.dtype)
        bias = self.model_checkpoint.tensor(f"{prefix}.bias").to(device=self.device, dtype=x.dtype)
        return torch.nn.functional.linear(x * math.sqrt(out_features / source_dim), weight, bias)

    def close(self) -> None:
        """Drop all text-side state belonging to this LTX 2.3 identity."""
        self.model = None
        self.tokenizer = None
        self.text_checkpoint = None
        self.model_checkpoint = None
        torch.cuda.empty_cache()
