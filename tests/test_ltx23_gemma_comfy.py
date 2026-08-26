from types import SimpleNamespace

import torch

from latentslate_engine.runtime.ltx23_gemma_comfy import (
    LTX23ComfyGemmaForConditionalGeneration,
    LTX23ComfyRMSNorm,
    _scaled_dot_product_attention,
)
from latentslate_engine.runtime.ltx23_kitchen import _enhance_prompt


def _tiny_config(*, layers: int = 2):
    text = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["sliding_attention", "full_attention"][:layers],
        sliding_window=4,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000},
            "full_attention": {
                "rope_type": "linear",
                "factor": 8.0,
                "rope_theta": 1_000_000,
            },
        },
    )
    return SimpleNamespace(text_config=text)


def test_direct_gemma_preserves_namespace_and_49_state_shape_contract() -> None:
    model = LTX23ComfyGemmaForConditionalGeneration(_tiny_config())
    for parameter in model.parameters():
        parameter.data.zero_()
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[2, 3, 4, 5]]),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            output_hidden_states=True,
            use_cache=False,
        )

    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    assert len(model.model.language_model.layers) == 2
    assert output.hidden_states is not None
    assert len(output.hidden_states) == 3
    assert all(state.shape == (1, 4, 16) for state in output.hidden_states)
    assert torch.equal(output.last_hidden_state, output.hidden_states[-1])


def test_direct_gemma_generation_returns_source_plus_suffix_and_stops() -> None:
    model = LTX23ComfyGemmaForConditionalGeneration(_tiny_config(layers=1))
    for parameter in model.parameters():
        parameter.data.zero_()
    state = {"outer_active": False}
    events: list[str] = []

    def outer_pre(_module, _inputs):
        assert state["outer_active"] is False
        state["outer_active"] = True
        events.append("outer_pre")

    def outer_post(_module, _inputs, output):
        assert state["outer_active"] is True
        events.append("outer_post")
        state["outer_active"] = False
        return output

    def embedding_pre(_module, _inputs):
        assert state["outer_active"] is True
        events.append("embedding")

    original_logits = model._logits

    def checked_logits(hidden_states):
        assert state["outer_active"] is True
        events.append("logits")
        return original_logits(hidden_states)

    model.register_forward_pre_hook(outer_pre)
    model.register_forward_hook(outer_post)
    model.model.language_model.embed_tokens.register_forward_pre_hook(embedding_pre)
    model._logits = checked_logits

    generated = model.generate(
        input_ids=torch.tensor([[2, 3]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        max_new_tokens=4,
        eos_token_id=0,
        do_sample=False,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        seed=0,
        execution_dtype=torch.float32,
    )

    assert generated.tolist() == [[2, 3, 0]]
    assert state["outer_active"] is False
    assert events == ["outer_pre", "embedding", "logits", "embedding", "outer_post"]


def test_masked_gqa_expands_kv_before_sdpa(monkeypatch) -> None:
    observed = {}

    def fake_sdpa(query, key, value, **kwargs):
        observed.update(
            query_heads=query.shape[-3],
            key_heads=key.shape[-3],
            value_heads=value.shape[-3],
            **kwargs,
        )
        return torch.zeros_like(query)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    query = torch.zeros((1, 4, 2, 4))
    key = torch.zeros((1, 2, 2, 4))
    value = torch.zeros_like(key)
    mask = torch.zeros((1, 1, 2, 2))

    output = _scaled_dot_product_attention(query, key, value, mask)

    assert output.shape == query.shape
    assert observed["query_heads"] == observed["key_heads"] == observed["value_heads"] == 4
    assert observed["enable_gqa"] is False
    assert observed["attn_mask"] is mask
    assert observed["is_causal"] is False


def test_direct_gemma_rmsnorm_uses_comfy_weight_plus_one_contract() -> None:
    norm = LTX23ComfyRMSNorm(4, eps=1e-6)
    norm.weight.data.copy_(torch.tensor([0.0, 0.5, -0.25, 1.0]))
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    expected = torch.nn.functional.rms_norm(
        value,
        (4,),
        weight=norm.weight + 1.0,
        eps=1e-6,
    )
    assert torch.equal(norm(value), expected)


def test_prompt_enhancement_dispatches_exact_settings_to_direct_substrate() -> None:
    class Processor:
        class Tokenizer:
            @staticmethod
            def batch_decode(_values, **_kwargs):
                return ["direct enhanced prompt"]

        tokenizer = Tokenizer()

        @staticmethod
        def __call__(**_kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[2, 105, 11]]))

    class DirectModel:
        _latentslate_comfy_gemma_direct = True

        @staticmethod
        def generate(**kwargs):
            assert kwargs["do_sample"] is True
            assert kwargs["temperature"] == 0.7
            assert kwargs["top_k"] == 64
            assert kwargs["top_p"] == 0.95
            assert kwargs["min_p"] == 0.05
            assert kwargs["repetition_penalty"] == 1.05
            assert kwargs["seed"] == 0
            return torch.cat(
                (kwargs["input_ids"], torch.tensor([[106]], dtype=torch.long)), dim=1
            )

    enhanced, proof = _enhance_prompt(
        Processor(),
        DirectModel(),
        "source prompt",
        0,
        torch.device("cpu"),
        lambda: None,
    )

    assert enhanced == "direct enhanced prompt"
    assert proof["cache_present"] is False
