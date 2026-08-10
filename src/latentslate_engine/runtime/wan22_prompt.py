"""Comfy-first Wan prompt tokenization and stored UMT5 conditioning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

WAN_PROMPT_SEQUENCE_LENGTH = 512
WAN_UMT5_HIDDEN_SIZE = 4096
WAN_UMT5_VOCAB_SIZE = 256000
_MAX_SENTENCEPIECE_BYTES = 16 * 1024 * 1024


class SentencePieceLike(Protocol):
    def encode(
        self,
        text: str,
        *,
        out_type: type[int],
        add_bos: bool,
        add_eos: bool,
    ) -> list[int]: ...

    def pad_id(self) -> int: ...

    def eos_id(self) -> int: ...

    def bos_id(self) -> int: ...

    def unk_id(self) -> int: ...

    def vocab_size(self) -> int: ...


class UMT5SessionLike(Protocol):
    @property
    def tokenizer_sha256(self) -> str: ...

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        sequence_length: int,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class WanPromptTokens:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_counts: tuple[int, int]
    sequence_length: int
    tokenizer_sha256: str


@dataclass(frozen=True, slots=True)
class WanPromptConditioning:
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    sequence_length: int
    tokenizer_sha256: str


class ComfyWanTokenizer:
    """Raw SentencePiece tokenizer matching Comfy's Wan text convention."""

    def __init__(self, processor: SentencePieceLike, *, model_sha256: str):
        _validate_processor_contract(processor)
        if len(model_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in model_sha256
        ):
            raise ValueError("Wan tokenizer SHA-256 must be lowercase hexadecimal")
        self._processor = processor
        self.model_sha256 = model_sha256

    @classmethod
    def from_file(cls, model_path: Path) -> ComfyWanTokenizer:
        import sentencepiece

        path = Path(model_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("Wan tokenizer model must be a file")
        with path.open("rb") as handle:
            payload = handle.read(_MAX_SENTENCEPIECE_BYTES + 1)
        if len(payload) > _MAX_SENTENCEPIECE_BYTES:
            raise ValueError("Wan tokenizer model exceeds the 16 MiB safety limit")
        if not payload:
            raise ValueError("Wan tokenizer model is empty")
        processor = sentencepiece.SentencePieceProcessor(model_proto=payload)
        return cls(processor, model_sha256=hashlib.sha256(payload).hexdigest())

    def tokenize_pair(
        self,
        prompt: str,
        negative_prompt: str,
        *,
        sequence_length: int = WAN_PROMPT_SEQUENCE_LENGTH,
    ) -> WanPromptTokens:
        if sequence_length != WAN_PROMPT_SEQUENCE_LENGTH:
            raise ValueError("Comfy-first Wan prompt length is exactly 512 tokens")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Wan prompt must be a nonempty string")
        if not isinstance(negative_prompt, str):
            raise TypeError("Wan negative prompt must be a string")

        encoded = tuple(self._encode(text, sequence_length) for text in (prompt, negative_prompt))
        input_ids = torch.full((2, sequence_length), self._processor.pad_id(), dtype=torch.int64)
        attention_mask = torch.zeros((2, sequence_length), dtype=torch.int64)
        counts: list[int] = []
        for row, token_ids in enumerate(encoded):
            count = len(token_ids)
            input_ids[row, :count] = torch.tensor(token_ids, dtype=torch.int64)
            attention_mask[row, :count] = 1
            counts.append(count)
        return WanPromptTokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_counts=(counts[0], counts[1]),
            sequence_length=sequence_length,
            tokenizer_sha256=self.model_sha256,
        )

    def _encode(self, text: str, sequence_length: int) -> tuple[int, ...]:
        token_ids = tuple(self._processor.encode(text, out_type=int, add_bos=False, add_eos=True))
        if not token_ids or token_ids[-1] != self._processor.eos_id():
            raise RuntimeError("Wan SentencePiece tokenizer did not append EOS")
        if len(token_ids) > sequence_length:
            raise ValueError(
                f"Wan prompt uses {len(token_ids)} tokens; the Comfy-first limit is {sequence_length}"
            )
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            or token_id >= WAN_UMT5_VOCAB_SIZE
            for token_id in token_ids
        ):
            raise RuntimeError("Wan SentencePiece tokenizer returned an invalid token ID")
        return token_ids


def encode_wan_prompt_pair(
    session: UMT5SessionLike,
    tokens: WanPromptTokens,
) -> WanPromptConditioning:
    """Encode the positive/negative pair within an active stored-UMT5 session."""

    if (
        tokens.sequence_length != WAN_PROMPT_SEQUENCE_LENGTH
        or tokens.input_ids.shape != (2, WAN_PROMPT_SEQUENCE_LENGTH)
        or tokens.attention_mask.shape != tokens.input_ids.shape
    ):
        raise ValueError("Wan prompt token batch must be [2,512]")
    if session.tokenizer_sha256 != tokens.tokenizer_sha256:
        raise ValueError("Wan tokenizer does not match the embedded UMT5 tokenizer")
    hidden_states = session.encode(
        tokens.input_ids,
        tokens.attention_mask,
        sequence_length=tokens.sequence_length,
    )
    if (
        not isinstance(hidden_states, torch.Tensor)
        or hidden_states.shape != (2, WAN_PROMPT_SEQUENCE_LENGTH, WAN_UMT5_HIDDEN_SIZE)
        or hidden_states.dtype not in {torch.float16, torch.bfloat16}
        or not bool(torch.isfinite(hidden_states).all())
        or hidden_states.requires_grad
    ):
        raise RuntimeError("Wan UMT5 conditioning output is incompatible")
    mask = tokens.attention_mask.to(device=hidden_states.device, dtype=torch.bool).unsqueeze(-1)
    if bool(hidden_states.masked_select(~mask).ne(0).any()):
        raise RuntimeError("Wan UMT5 conditioning padded positions must be exactly zero")
    return WanPromptConditioning(
        prompt_embeds=hidden_states[0:1],
        negative_prompt_embeds=hidden_states[1:2],
        sequence_length=tokens.sequence_length,
        tokenizer_sha256=tokens.tokenizer_sha256,
    )


def _validate_processor_contract(processor: SentencePieceLike) -> None:
    actual = (
        processor.pad_id(),
        processor.eos_id(),
        processor.bos_id(),
        processor.unk_id(),
        processor.vocab_size(),
    )
    expected = (0, 1, 2, 3, WAN_UMT5_VOCAB_SIZE)
    if actual != expected:
        raise ValueError(f"Wan SentencePiece contract mismatch: {actual!r} != {expected!r}")
