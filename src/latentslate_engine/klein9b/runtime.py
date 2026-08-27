from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Self

import torch
from diffusers import AutoencoderKLFlux2
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from PIL import Image
from safetensors import safe_open
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from .model import KleinTransformer, Linear

RECIPE_ID = "flux2-klein-9b-distilled-t2i-768-v1"
KLEIN_PROMPT_TEMPLATE = (
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
TOKENIZER_FILES = (
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


@dataclass(frozen=True)
class ArtifactIdentity:
    path: Path
    size: int
    modified_ns: int

    @classmethod
    def from_path(cls, path: Path) -> ArtifactIdentity:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
        return cls(resolved, stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class Klein9BIdentity:
    diffusion: ArtifactIdentity
    text_encoder: ArtifactIdentity
    vae: ArtifactIdentity
    tokenizer: Path
    tokenizer_files: tuple[ArtifactIdentity, ...]
    text_encoder_config: ArtifactIdentity
    recipe: str = RECIPE_ID

    @classmethod
    def from_paths(
        cls, diffusion: Path, text_encoder: Path, vae: Path, tokenizer: Path
    ) -> Klein9BIdentity:
        tokenizer_path = tokenizer.resolve(strict=True)
        config_path = tokenizer_path.parent / "text_encoder" / "config.json"
        return cls(
            ArtifactIdentity.from_path(diffusion),
            ArtifactIdentity.from_path(text_encoder),
            ArtifactIdentity.from_path(vae),
            tokenizer_path,
            tuple(
                ArtifactIdentity.from_path(tokenizer_path / name)
                for name in TOKENIZER_FILES
            ),
            ArtifactIdentity.from_path(config_path),
        )


@dataclass(frozen=True)
class GenerationResult:
    output: Path
    elapsed_seconds: float
    conditioning_reused: bool
    models_reused: bool


class QuantizedLinear(nn.Module):
    def __init__(self, source: nn.Module, layout: str) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.layout = layout
        self.register_buffer("qdata", None)
        self.register_buffer("weight_scale", None)
        self.register_buffer("weight_scale_2", None)

    def forward(self, value: Tensor) -> Tensor:
        from comfy_kitchen.tensor import (
            QuantizedTensor,
            TensorCoreFP8Layout,
            TensorCoreNVFP4Layout,
        )

        if self.layout == "TensorCoreFP8Layout":
            parameters = TensorCoreFP8Layout.Params(
                scale=self.weight_scale,
                orig_dtype=value.dtype,
                orig_shape=(self.out_features, self.in_features),
            )
        else:
            parameters = TensorCoreNVFP4Layout.Params(
                scale=self.weight_scale_2,
                block_scale=self.weight_scale,
                orig_dtype=value.dtype,
                orig_shape=(self.out_features, self.in_features),
            )
        weight = QuantizedTensor(self.qdata, self.layout, parameters)
        return F.linear(value, weight.dequantize().to(value.dtype))


class PinnedQwenLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool, *, device: str = "meta"
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, value: Tensor) -> Tensor:
        bias = self.bias.to(value.dtype) if self.bias is not None else None
        return F.linear(value, self.weight.to(value.dtype), bias)


class PinnedQwenRMSNorm(nn.Module):
    def __init__(self, size: int, *, device: str = "meta") -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(size, device=device), requires_grad=False
        )

    def forward(self, value: Tensor) -> Tensor:
        return F.rms_norm(
            value, (value.shape[-1],), self.weight.to(value.dtype), eps=1e-6
        )


class PinnedQwenRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config, device: torch.device) -> None:
        super().__init__()
        head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
        positions = torch.arange(0, head_dim, 2, device=device).float()
        theta = config.rope_parameters["rope_theta"]
        self.register_buffer("inv_freq", 1.0 / (theta ** (positions / head_dim)), False)

    def forward(self, _: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inverse = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        positions = position_ids[:, None, :].float()
        frequencies = (inverse.float() @ positions).transpose(1, 2)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos(), embedding.sin()


def _apply_pinned_qwen_rope(
    query: Tensor, key: Tensor, cosine: Tensor, sine: Tensor, unsqueeze_dim: int = 1
) -> tuple[Tensor, Tensor]:
    cosine = cosine.unsqueeze(unsqueeze_dim)
    sine = sine.unsqueeze(unsqueeze_dim)

    query_output = query * cosine
    split = query_output.shape[-1] // 2
    query_output[..., :split].addcmul_(query[..., split:], -sine[..., split:])
    query_output[..., split:].addcmul_(query[..., :split], sine[..., :split])

    key_output = key * cosine
    split = key_output.shape[-1] // 2
    key_output[..., :split].addcmul_(key[..., split:], -sine[..., split:])
    key_output[..., split:].addcmul_(key[..., :split], sine[..., :split])
    return query_output.to(query.dtype), key_output.to(key.dtype)


def _resolve_module(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child_name = name.rpartition(".")
    return (root.get_submodule(parent_name) if parent_name else root), child_name


def _assign_parameter(root: nn.Module, name: str, value: Tensor) -> None:
    parent, child = _resolve_module(root, name)
    setattr(parent, child, nn.Parameter(value, requires_grad=False))


def _load_transformer(path: Path, device: torch.device) -> KleinTransformer:
    model = KleinTransformer()
    expected = set(model.state_dict().keys())
    loaded: set[str] = set()
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for key in list(checkpoint.keys()):
            if key.endswith((".weight", ".scale", ".weight_scale", ".input_scale")):
                tensor = checkpoint.get_tensor(key).to(device=device, non_blocking=True)
                if key.endswith((".weight", ".scale")):
                    if key not in expected:
                        raise ValueError(f"Unexpected diffusion tensor: {key}")
                    _assign_parameter(model, key, tensor)
                    loaded.add(key)
                else:
                    parent_name, _, child = key.rpartition(".")
                    module = model.get_submodule(parent_name)
                    if not isinstance(module, Linear):
                        raise ValueError(f"Quantization metadata on non-linear: {key}")
                    setattr(module, child, tensor)
    missing = expected - loaded
    if missing:
        raise ValueError(f"Missing diffusion tensors: {sorted(missing)[:5]}")
    return model.eval()


def _replace_text_quantized_linears(model: Qwen3Model, checkpoint_path: Path) -> None:
    replacements: list[tuple[str, str]] = []
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        for key in list(checkpoint.keys()):
            if not key.endswith(".comfy_quant"):
                continue
            module_name = key.removesuffix(".comfy_quant").removeprefix("model.")
            descriptor = bytes(checkpoint.get_tensor(key).tolist()).decode("utf-8")
            quant_format = json.loads(descriptor)["format"]
            layout = {
                "float8_e4m3fn": "TensorCoreFP8Layout",
                "nvfp4": "TensorCoreNVFP4Layout",
            }[quant_format]
            replacements.append((module_name, layout))
    for name, layout in replacements:
        parent, child = _resolve_module(model, name)
        source = getattr(parent, child)
        setattr(parent, child, QuantizedLinear(source, layout))


def _load_text_encoder(
    path: Path, tokenizer_path: Path, device: torch.device
) -> Qwen3Model:
    config_path = tokenizer_path.parent / "text_encoder"
    config = Qwen3Config.from_pretrained(config_path, local_files_only=True)
    config.use_cache = False
    with torch.device("meta"):
        model = Qwen3Model(config)
    for name, module in list(model.named_modules()):
        if isinstance(module, Qwen3RMSNorm):
            parent, child = _resolve_module(model, name)
            setattr(parent, child, PinnedQwenRMSNorm(module.weight.shape[0]))
        elif isinstance(module, nn.Linear):
            parent, child = _resolve_module(model, name)
            setattr(
                parent,
                child,
                PinnedQwenLinear(
                    module.in_features, module.out_features, module.bias is not None
                ),
            )
    model.rotary_emb = PinnedQwenRotaryEmbedding(config, device)
    modeling_qwen3.apply_rotary_pos_emb = _apply_pinned_qwen_rope
    _replace_text_quantized_linears(model, path)
    expected = {name for name, _ in model.named_parameters()}
    loaded: set[str] = set()
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        for name, _ in list(model.named_parameters()):
            checkpoint_name = f"model.{name}"
            if checkpoint_name not in keys:
                raise ValueError(f"Missing text tensor: {name}")
            _assign_parameter(
                model, name, checkpoint.get_tensor(checkpoint_name).to(device)
            )
            loaded.add(name)
        for module_name, module in model.named_modules():
            if not isinstance(module, QuantizedLinear):
                continue
            prefix = f"model.{module_name}"
            module.qdata = checkpoint.get_tensor(f"{prefix}.weight").to(device)
            module.weight_scale = checkpoint.get_tensor(f"{prefix}.weight_scale").to(
                device
            )
            if module.layout == "TensorCoreNVFP4Layout":
                module.weight_scale_2 = checkpoint.get_tensor(
                    f"{prefix}.weight_scale_2"
                ).to(device)
    missing = expected - loaded
    if missing:
        raise ValueError(f"Unloaded text parameters: {sorted(missing)[:5]}")
    return model.eval()


def _encode_prompt(
    prompt: str, checkpoint: Path, tokenizer_path: Path, device: torch.device
) -> Tensor:
    input_ids, attention_mask = _tokenize_prompt(prompt, tokenizer_path)
    encoder = _load_text_encoder(checkpoint, tokenizer_path, device)
    attention_backends = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
    with (
        torch.inference_mode(),
        sdpa_kernel(attention_backends, set_priority=True),
    ):
        embeddings = encoder.embed_tokens(input_ids.to(device)).float()
        output = encoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        selected = torch.stack(
            [output.hidden_states[index] for index in (9, 18, 27)], dim=2
        )
        context = selected.reshape(1, 512, 12288)
    del encoder, embeddings, output, selected
    gc.collect()
    torch.cuda.empty_cache()
    return context


def _tokenize_prompt(prompt: str, tokenizer_path: Path) -> tuple[Tensor, Tensor]:
    tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    text = KLEIN_PROMPT_TEMPLATE.format(prompt)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) > 512:
        token_ids = token_ids[:512]
    attention = [1] * len(token_ids)
    padding = 512 - len(token_ids)
    token_ids.extend([151643] * padding)
    attention.extend([0] * padding)
    return torch.tensor([token_ids]), torch.tensor([attention])


def _load_vae(path: Path, device: torch.device) -> AutoencoderKLFlux2:
    with torch.device("meta"):
        vae = AutoencoderKLFlux2(
            block_out_channels=(128, 256, 512, 512),
            decoder_block_out_channels=(96, 192, 384, 384),
            latent_channels=32,
            patch_size=(2, 2),
        )
    state: dict[str, Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for key in list(checkpoint.keys()):
            state[key] = checkpoint.get_tensor(key)
    converted = convert_ldm_vae_checkpoint(state, vae.config)
    for name in ("weight", "bias"):
        converted[f"quant_conv.{name}"] = state[f"encoder.quant_conv.{name}"]
        converted[f"post_quant_conv.{name}"] = state[f"decoder.post_quant_conv.{name}"]
    for name in ("running_mean", "running_var", "num_batches_tracked"):
        converted[f"bn.{name}"] = state[f"bn.{name}"]
    converted = {
        name: tensor.to(
            device=device,
            dtype=torch.bfloat16 if tensor.is_floating_point() else tensor.dtype,
        )
        for name, tensor in converted.items()
    }
    vae.load_state_dict(converted, assign=True)
    del state, converted
    return vae.eval()


def _sigmas(steps: int, device: torch.device) -> Tensor:
    sequence_length = 2304
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    m_200 = a2 * sequence_length + b2
    m_10 = a1 * sequence_length + b1
    slope = (m_200 - m_10) / 190
    intercept = m_200 - 200 * slope
    mu = slope * steps + intercept
    timesteps = torch.linspace(1, 0, steps + 1, device=device)
    return torch.exp(torch.tensor(mu, device=device)) / (
        torch.exp(torch.tensor(mu, device=device)) + (1 / timesteps - 1)
    )


def _unpack_latent(latent: Tensor) -> Tensor:
    batch, channels, height, width = latent.shape
    if channels != 128:
        raise ValueError(f"Expected 128 packed latent channels, got {channels}")
    latent = latent.reshape(batch, 32, 2, 2, height, width)
    return latent.permute(0, 1, 4, 2, 5, 3).reshape(batch, 32, height * 2, width * 2)


class Klein9BRuntime:
    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.identity: Klein9BIdentity | None = None
        self.transformer: KleinTransformer | None = None
        self.vae: AutoencoderKLFlux2 | None = None
        self.conditioning: tuple[str, Tensor] | None = None

    def close(self) -> None:
        self.transformer = None
        self.vae = None
        self.conditioning = None
        self.identity = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def ensure_identity(self, identity: Klein9BIdentity) -> bool:
        if self.identity == identity:
            return True
        self.close()
        self.identity = identity
        return False

    def generate(
        self, identity: Klein9BIdentity, prompt: str, seed: int, output: Path
    ) -> GenerationResult:
        started = time.perf_counter()
        models_reused = self.ensure_identity(identity) and self.transformer is not None
        conditioning_reused = (
            self.conditioning is not None and self.conditioning[0] == prompt
        )
        if not conditioning_reused:
            context = _encode_prompt(
                prompt, identity.text_encoder.path, identity.tokenizer, self.device
            )
            self.conditioning = (prompt, context)
        assert self.conditioning is not None
        _, context = self.conditioning
        if self.transformer is None:
            self.transformer = _load_transformer(identity.diffusion.path, self.device)
        if self.vae is None:
            self.vae = _load_vae(identity.vae.path, self.device)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        latent = torch.randn((1, 128, 48, 48), generator=generator).to(self.device)
        schedule = _sigmas(4, self.device)
        with torch.inference_mode():
            for current, following in pairwise(schedule):
                prediction = self.transformer(
                    latent.to(torch.bfloat16),
                    current.expand(1),
                    context.to(torch.bfloat16),
                    None,
                )
                denoised = latent - prediction.float() * current
                derivative = (latent - denoised) / current
                latent = latent + derivative * (following - current)
            running_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
                device=latent.device, dtype=latent.dtype
            )
            running_var = self.vae.bn.running_var.view(1, -1, 1, 1).to(
                device=latent.device, dtype=latent.dtype
            )
            latent = latent * torch.sqrt(running_var + 1e-4) + running_mean
            latent = _unpack_latent(latent)
            decoded = self.vae.decode(latent.to(torch.bfloat16), return_dict=False)[0]
        pixels = ((decoded.float().clamp(-1, 1) + 1) * 127.5).byte()
        pixels = pixels[0].permute(1, 2, 0).cpu().numpy()
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(output, format="PNG")
        return GenerationResult(
            output=output,
            elapsed_seconds=time.perf_counter() - started,
            conditioning_reused=conditioning_reused,
            models_reused=models_reused,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
