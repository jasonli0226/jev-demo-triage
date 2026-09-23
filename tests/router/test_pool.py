import pytest
from langchain_openai import ChatOpenAI

from jev_demo_triage.config import OPENROUTER_BASE_URL, MissingKeyError
from jev_router_bench.pool import (
    MAX_TOKENS,
    POOL,
    ROUTER_LLM,
    TIERS,
    content_text,
    make_chat_model,
    tier_rank,
)


def test_tiers_are_in_cost_order():
    assert TIERS == ("cheap", "mid", "strong")
    prices = [POOL[t].input_price for t in TIERS]
    assert prices == sorted(prices)
    assert [tier_rank(t) for t in TIERS] == [0, 1, 2]


def test_pool_model_ids_and_prices():
    assert POOL["cheap"].model_id == "qwen/qwen3.7-flash"
    assert POOL["mid"].model_id == "z-ai/glm-4.7"
    assert POOL["strong"].model_id == "anthropic/claude-sonnet-5"
    assert POOL["strong"].input_price == pytest.approx(2.0 / 1_000_000)
    assert POOL["strong"].output_price == pytest.approx(10.0 / 1_000_000)


def test_cost_uses_input_and_output_prices():
    assert POOL["mid"].cost(1_000_000, 1_000_000) == pytest.approx(0.40 + 1.75)
    assert POOL["cheap"].cost(0, 0) == 0.0


def test_router_llm_is_glm():
    assert ROUTER_LLM.model_id == "z-ai/glm-4.7"


def test_make_chat_model_targets_openrouter():
    model = make_chat_model("qwen/qwen3.7-flash", api_key="sk-or-test")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "qwen/qwen3.7-flash"
    assert model.openai_api_base == OPENROUTER_BASE_URL
    assert model.temperature == 0
    assert model.max_tokens == MAX_TOKENS == 2048


def test_make_chat_model_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingKeyError):
        make_chat_model("z-ai/glm-4.7")


def test_content_text_handles_strings_and_blocks():
    assert content_text("hi") == "hi"
    blocks = [
        {"type": "reasoning", "text": "hidden"},
        {"type": "text", "text": "ANSWER: "},
        "42",
    ]
    assert content_text(blocks) == "ANSWER: 42"
    assert content_text(7) == "7"
