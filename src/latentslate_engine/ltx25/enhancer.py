"""Native owner of the curated E2B prompt enhancer and its image encoder."""

import torch
from tokenizers import Tokenizer

from latentslate_engine.ltx23.checkpoint import Ltx23Checkpoint
from latentslate_engine.ltx23.fp8_linear import Ltx23Int8Linear, _aimdo_modules

from .enhancement import decode_enhancement, tokenize_enhancement
from .enhancer_language import E2BTransformer
from .enhancer_vision import Gemma4RMSNormProjector, Gemma4VisionEncoder


class Ltx25Enhancer:
    """Own the enhancer weights; request KV state remains local to generation."""

    def __init__(self, path, device_index=0):
        self.device_index = device_index
        self.device = torch.device("cuda", device_index)
        self.checkpoint = Ltx23Checkpoint(path)
        self.tokenizer = Tokenizer.from_str(
            self.checkpoint.tensor("tokenizer_json").numpy().tobytes().decode("utf-8")
        )
        self.language = E2BTransformer(device="meta", dtype=torch.bfloat16)
        config = {
            "hidden_size": 768,
            "image_size": 896,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 16,
            "patch_size": 16,
            "head_dim": 64,
            "rms_norm_eps": 1e-6,
            "position_embedding_size": 10240,
            "pooling_kernel_size": 3,
        }
        self.vision = Gemma4VisionEncoder(
            config, device="meta", dtype=torch.bfloat16, ops=torch.nn
        )
        self.projector = Gemma4RMSNormProjector(
            768, 1536, device="meta", dtype=torch.bfloat16, ops=torch.nn
        )
        for prefix, module in (
            ("model.", self.language),
            ("vision_model.", self.vision),
            ("multi_modal_projector.", self.projector),
        ):
            self._load(prefix, module)

    def _load(self, prefix, module):
        state, bindings = {}, {}
        for name in module.state_dict():
            full = prefix + name
            value = self.checkpoint.tensor(full)
            if value.dtype == torch.int8:
                bindings[name] = Ltx23Int8Linear(
                    self.checkpoint, full.removesuffix(".weight")
                )
            else:
                state[name] = value.to(self.device)
        if bindings:
            model_vbar, _ = _aimdo_modules(self.device_index)
            vbar = model_vbar.ModelVBAR(
                2 * sum(binding.source_size for binding in bindings.values()),
                self.device_index,
            )
            for binding in bindings.values():
                binding.allocate(vbar)
            for name, binding in bindings.items():
                weight, _, _ = binding.materialize(self.device_index)
                try:
                    state[name] = weight.to(dtype=torch.bfloat16).clone()
                finally:
                    binding.unpin(self.device_index)
        module.load_state_dict(state, strict=True, assign=True)

    @torch.inference_mode()
    def enhance(self, prompt, image=None, *, progress=None):
        tokens = tokenize_enhancement(self.tokenizer, prompt, image)
        ids = torch.tensor(
            [[token for token, _ in tokens if isinstance(token, int)]],
            device=self.device,
        )
        embeds = self.language.embed(ids, out_dtype=torch.float32)
        if image is not None:
            index = next(
                i for i, (token, _) in enumerate(tokens) if isinstance(token, dict)
            )
            media = tokens[index][0]
            vision = self.vision(
                media["data"].movedim(-1, 1).to(self.device),
                max_soft_tokens=media["max_soft_tokens"],
            )
            projected = self.projector(vision).float()
            embeds = torch.cat((embeds[:, :index], projected, embeds[:, index:]), 1)
            ids = torch.cat(
                (
                    ids[:, :index],
                    torch.zeros(
                        1, projected.shape[1], dtype=torch.long, device=self.device
                    ),
                    ids[:, index:],
                ),
                1,
            )
            del vision, projected
        generated = self.language.generate(embeds, ids, progress=progress)
        return decode_enhancement(self.tokenizer, generated, prompt)

    def close(self):
        self.language = None
        self.vision = None
        self.projector = None
        self.checkpoint = None
        torch.cuda.empty_cache()
