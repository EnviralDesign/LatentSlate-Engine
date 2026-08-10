from __future__ import annotations

import importlib.util

import pytest

from latentslate_engine.runtime.klein_support import (
    clear_klein_runtime_support_cache,
    klein_runtime_support,
)


def test_flux2_klein_runtime_imports_with_pinned_stack():
    if importlib.util.find_spec("diffusers") is None:
        pytest.skip("runtime dependency group is not installed")

    clear_klein_runtime_support_cache()
    support = klein_runtime_support()
    assert support.core_available, support.core_reason

    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
    from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

    assert Flux2KleinPipeline is not None
    assert Flux2Transformer2DModel is not None
    assert Qwen3ForCausalLM is not None
    assert Qwen2TokenizerFast is not None
