from __future__ import annotations

from contextlib import AbstractContextManager

import pytest
import torch

from latentslate_engine.runtime.wan22_prompt import (
    WAN_PROMPT_SEQUENCE_LENGTH,
    ComfyWanTokenizer,
    WanPromptTokens,
    encode_wan_prompt_pair,
)


class FakeSentencePiece:
    def __init__(self):
        self.encoded = {
            "positive": [10, 11, 1],
            "": [1],
            "spaces": [273, 273, 20, 1],
            "too long": [10] * WAN_PROMPT_SEQUENCE_LENGTH + [1],
        }

    def encode(self, text, *, out_type, add_bos, add_eos):
        assert out_type is int and not add_bos and add_eos
        return list(self.encoded[text])

    def pad_id(self):
        return 0

    def eos_id(self):
        return 1

    def bos_id(self):
        return 2

    def unk_id(self):
        return 3

    def vocab_size(self):
        return 256000


def test_comfy_tokenizer_preserves_raw_piece_ids_and_pads_to_512():
    tokenizer = ComfyWanTokenizer(FakeSentencePiece(), model_sha256="a" * 64)
    tokens = tokenizer.tokenize_pair("spaces", "")

    assert tokens.input_ids.shape == (2, 512)
    assert tokens.input_ids[0, :4].tolist() == [273, 273, 20, 1]
    assert tokens.input_ids[1, :2].tolist() == [1, 0]
    assert tokens.token_counts == (4, 1)
    assert tokens.attention_mask[0].sum() == 4
    assert tokens.attention_mask[1].sum() == 1
    assert not tokens.attention_mask[:, 4:].any()


def test_comfy_tokenizer_fails_closed_instead_of_truncating():
    tokenizer = ComfyWanTokenizer(FakeSentencePiece(), model_sha256="b" * 64)
    with pytest.raises(ValueError, match="513 tokens"):
        tokenizer.tokenize_pair("too long", "")
    with pytest.raises(ValueError, match="nonempty"):
        tokenizer.tokenize_pair(" ", "")
    with pytest.raises(ValueError, match="exactly 512"):
        tokenizer.tokenize_pair("positive", "", sequence_length=226)


def test_tokenizer_file_read_is_bounded(tmp_path):
    path = tmp_path / "oversized.model"
    path.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="16 MiB"):
        ComfyWanTokenizer.from_file(path)


def test_tokenizer_rejects_wrong_special_token_contract():
    processor = FakeSentencePiece()
    processor.pad_id = lambda: 7
    with pytest.raises(ValueError, match="contract mismatch"):
        ComfyWanTokenizer(processor, model_sha256="c" * 64)


def test_prompt_pair_uses_stored_umt5_session_and_splits_conditioning():
    tokenizer = ComfyWanTokenizer(FakeSentencePiece(), model_sha256="d" * 64)
    tokens = tokenizer.tokenize_pair("positive", "")

    class Session:
        tokenizer_sha256 = "d" * 64

        def encode(self, input_ids, attention_mask, *, sequence_length):
            assert torch.equal(input_ids, tokens.input_ids)
            assert torch.equal(attention_mask, tokens.attention_mask)
            assert sequence_length == 512
            output = torch.ones((2, 512, 4096), dtype=torch.float16)
            output[1] *= 2
            output.masked_fill_(~attention_mask.to(dtype=torch.bool).unsqueeze(-1), 0)
            return output

    conditioning = encode_wan_prompt_pair(Session(), tokens)
    assert conditioning.prompt_embeds.shape == (1, 512, 4096)
    assert conditioning.negative_prompt_embeds.shape == (1, 512, 4096)
    assert conditioning.prompt_embeds[0, 0, 0] == 1
    assert conditioning.negative_prompt_embeds[0, 0, 0] == 2
    assert conditioning.tokenizer_sha256 == "d" * 64
    assert not conditioning.prompt_embeds.requires_grad


def test_prompt_conditioning_survives_session_exit():
    tokenizer = ComfyWanTokenizer(FakeSentencePiece(), model_sha256="f" * 64)
    tokens = tokenizer.tokenize_pair("positive", "")

    class Session(AbstractContextManager):
        tokenizer_sha256 = "f" * 64
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

        def encode(self, _input_ids, attention_mask, *, sequence_length):
            assert sequence_length == 512
            output = torch.ones((2, 512, 4096), dtype=torch.float16)
            return output.masked_fill(~attention_mask.to(dtype=torch.bool).unsqueeze(-1), 0)

    session = Session()
    with session:
        conditioning = encode_wan_prompt_pair(session, tokens)
    assert session.closed
    assert conditioning.prompt_embeds[0, 0, 0] == 1
    assert conditioning.negative_prompt_embeds[0, 0, 0] == 1


def test_prompt_pair_rejects_bad_batch_or_encoder_output():
    bad_tokens = WanPromptTokens(
        input_ids=torch.zeros((1, 512), dtype=torch.int64),
        attention_mask=torch.zeros((1, 512), dtype=torch.int64),
        token_counts=(1, 1),
        sequence_length=512,
        tokenizer_sha256="e" * 64,
    )
    with pytest.raises(ValueError, match=r"\[2,512\]"):
        encode_wan_prompt_pair(object(), bad_tokens)

    tokens = ComfyWanTokenizer(FakeSentencePiece(), model_sha256="e" * 64).tokenize_pair(
        "positive", ""
    )

    class BadSession:
        tokenizer_sha256 = "e" * 64

        def encode(self, *_args, **_kwargs):
            return torch.zeros((2, 512, 4), dtype=torch.float16)

    with pytest.raises(RuntimeError, match="incompatible"):
        encode_wan_prompt_pair(BadSession(), tokens)


def test_prompt_pair_rejects_tokenizer_mismatch_and_nonzero_padding():
    tokens = ComfyWanTokenizer(FakeSentencePiece(), model_sha256="1" * 64).tokenize_pair(
        "positive", ""
    )

    class Session:
        tokenizer_sha256 = "2" * 64

        def encode(self, *_args, **_kwargs):
            raise AssertionError("tokenizer mismatch reached the encoder")

    with pytest.raises(ValueError, match="embedded UMT5 tokenizer"):
        encode_wan_prompt_pair(Session(), tokens)

    Session.tokenizer_sha256 = "1" * 64
    Session.encode = lambda *_args, **_kwargs: torch.ones((2, 512, 4096), dtype=torch.float16)
    with pytest.raises(RuntimeError, match="padded positions"):
        encode_wan_prompt_pair(Session(), tokens)
