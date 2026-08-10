from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated Wan 2.2 CPU prompt encoder")
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    prompt_embeds, negative_prompt_embeds = encode_prompt_pair(
        model_path=Path(request["model_path"]),
        prompt=str(request["prompt"]),
        negative_prompt=str(request["negative_prompt"]),
        max_sequence_length=int(request["max_sequence_length"]),
    )

    from safetensors.torch import save_file

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        },
        str(args.output),
    )


if __name__ == "__main__":
    main()
