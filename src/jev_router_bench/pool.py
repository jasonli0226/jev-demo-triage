"""Model pool the routers choose from, with estimated prices."""

from dataclasses import dataclass
from typing import Any, Literal

from langchain_openai import ChatOpenAI

from jev_demo_triage.config import OPENROUTER_BASE_URL, require_openrouter_key

Tier = Literal["cheap", "mid", "strong"]
TIERS: tuple[Tier, ...] = ("cheap", "mid", "strong")
MAX_TOKENS = 8192  # reasoning models spend part of this before answering


@dataclass(frozen=True)
class PoolModel:
    tier: Tier
    model_id: str
    input_price: float  # USD per input token
    output_price: float  # USD per output token

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * self.input_price + output_tokens * self.output_price


# Prices as listed by OpenRouter on 2026-09-24 (model list price; the provider that
# serves a request may charge less). Costs are estimates: prices change and are not
# read from the API. All three models have zero-data-retention endpoints.
POOL: dict[Tier, PoolModel] = {
    "cheap": PoolModel("cheap", "mistralai/mistral-small-3.2-24b-instruct", 0.075 / 1_000_000, 0.20 / 1_000_000),
    "mid": PoolModel("mid", "z-ai/glm-4.7", 0.40 / 1_000_000, 1.75 / 1_000_000),
    "strong": PoolModel("strong", "moonshotai/kimi-k3", 3.00 / 1_000_000, 15.00 / 1_000_000),
}
ROUTER_LLM = POOL["mid"]


def tier_rank(tier: Tier) -> int:
    return TIERS.index(tier)


def make_chat_model(model_id: str, api_key: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_id,
        api_key=api_key or require_openrouter_key(),
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        max_tokens=MAX_TOKENS,
    )


def content_text(content: Any) -> str:
    """Plain text of a message content; reasoning/thinking blocks are ignored."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)
