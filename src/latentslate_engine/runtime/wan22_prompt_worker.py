from __future__ import annotations

import gc
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .framework.worker import (
    DisposableChildContext,
    parse_disposable_child_paths,
    run_disposable_child,
    sha256_fingerprint,
)


def _pad_embeddings(
    hidden_states: Any,
    attention_mask: Any,
    *,
    max_sequence_length: int,
) -> Any:
    import torch

    sequence_lengths = attention_mask.gt(0).sum(dim=1).long()
    rows = []
    for hidden_state, sequence_length in zip(hidden_states, sequence_lengths, strict=True):
        used = hidden_state[: int(sequence_length)]
        padding = used.new_zeros(max_sequence_length - used.shape[0], used.shape[1])
        rows.append(torch.cat((used, padding), dim=0))
    return torch.stack(rows, dim=0)


def _clean_prompt_pair(prompt: str, negative_prompt: str) -> tuple[str, str]:
    # Use the exact helper from the pinned WanPipeline implementation so staged
    # conditioning remains semantically equivalent to Pipeline.encode_prompt().
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    return prompt_clean(prompt), prompt_clean(negative_prompt)


def encode_prompt_pair(
    *,
    model_path: Path,
    prompt: str,
    negative_prompt: str,
    max_sequence_length: int,
) -> tuple[Any, Any]:
    import torch
    from transformers import T5TokenizerFast, UMT5EncoderModel

    tokenizer = T5TokenizerFast.from_pretrained(
        model_path,
        subfolder="tokenizer",
        local_files_only=True,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_path,
        subfolder="text_encoder",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    clean_prompt, clean_negative_prompt = _clean_prompt_pair(prompt, negative_prompt)
    text_inputs = tokenizer(
        [clean_prompt, clean_negative_prompt],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        hidden_states = text_encoder(
            text_inputs.input_ids,
            text_inputs.attention_mask,
        ).last_hidden_state
    hidden_states = _pad_embeddings(
        hidden_states,
        text_inputs.attention_mask,
        max_sequence_length=max_sequence_length,
    )
    prompt_embeds = hidden_states[0:1].detach().cpu().contiguous()
    negative_prompt_embeds = hidden_states[1:2].detach().cpu().contiguous()

    del hidden_states
    del text_inputs
    del text_encoder
    del tokenizer
    gc.collect()
    return prompt_embeds, negative_prompt_embeds


@dataclass(frozen=True, slots=True)
class _BoundPromptRequest:
    model_path: Path
    prompt: str
    negative_prompt: str
    max_sequence_length: int
    output_path: Path
    binding: str


class _WanPromptHandler:
    def bind_request(
        self, payload: Any, context: DisposableChildContext
    ) -> _BoundPromptRequest:
        context.stage = "validate_bound_request"
        expected = {
            "schema_version",
            "model_path",
            "prompt",
            "negative_prompt",
            "max_sequence_length",
            "output_path",
            "request_binding",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("Wan prompt worker request is not canonical")
        binding = payload["request_binding"]
        unsigned = {key: payload[key] for key in expected - {"request_binding"}}
        if (
            payload["schema_version"] != 1
            or not isinstance(binding, str)
            or binding != sha256_fingerprint(unsigned)
        ):
            raise ValueError("Wan prompt worker request binding is invalid")
        context.binding = binding
        model_path = Path(payload["model_path"]).resolve(strict=True)
        output_path = Path(payload["output_path"]).resolve(strict=False)
        if (
            not model_path.is_dir()
            or not isinstance(payload["prompt"], str)
            or not isinstance(payload["negative_prompt"], str)
            or isinstance(payload["max_sequence_length"], bool)
            or not isinstance(payload["max_sequence_length"], int)
            or payload["max_sequence_length"] <= 0
            or output_path.suffix.lower() != ".safetensors"
            or output_path.exists()
        ):
            raise ValueError("Wan prompt worker request fields are invalid")
        return _BoundPromptRequest(
            model_path,
            payload["prompt"],
            payload["negative_prompt"],
            payload["max_sequence_length"],
            output_path,
            binding,
        )

    def load(
        self, request: _BoundPromptRequest, context: DisposableChildContext
    ) -> None:
        context.stage = "ready"

    def run(
        self,
        runtime: None,
        request: _BoundPromptRequest,
        context: DisposableChildContext,
    ) -> Mapping[str, Any]:
        context.stage = "encode_prompt"
        context.publish_progress(0.01, "Encoding Wan prompt")
        prompt_embeds, negative_prompt_embeds = encode_prompt_pair(
            model_path=request.model_path,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            max_sequence_length=request.max_sequence_length,
        )
        context.stage = "save_conditioning"
        from safetensors.torch import save_file

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
            },
            str(request.output_path),
        )
        return {
            "schema_version": 1,
            "ok": True,
            "request_binding": request.binding,
            "output_path": str(request.output_path),
            "output_size_bytes": request.output_path.stat().st_size,
        }

    def unload(
        self,
        runtime: None,
        request: _BoundPromptRequest,
        context: DisposableChildContext,
    ) -> None:
        pass

    def failure_result(
        self, exc: BaseException, context: DisposableChildContext
    ) -> Mapping[str, Any]:
        safe_names = {
            "FileNotFoundError",
            "ImportError",
            "MemoryError",
            "OSError",
            "RuntimeError",
            "TimeoutError",
            "TypeError",
            "ValueError",
        }
        name = type(exc).__name__
        return {
            "schema_version": 1,
            "ok": False,
            "request_binding": context.binding,
            "error_type": name if name in safe_names else "Exception",
            "failure_stage": context.stage,
        }

    def stage_for_progress(self, message: str | None) -> str:
        return "encode_prompt"

    def protocol_error(self, reason: str) -> BaseException:
        return ValueError(f"Wan prompt worker protocol violation: {reason}")


def main(argv: list[str] | None = None) -> int:
    paths = parse_disposable_child_paths(
        argv, description="Isolated Wan 2.2 CPU prompt encoder"
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return run_disposable_child(paths, _WanPromptHandler())


if __name__ == "__main__":
    raise SystemExit(main())
