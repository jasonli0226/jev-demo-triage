"""Per-run counters. Jev cost is derived from tokens (output tokens are free)."""

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage

JEV_INPUT_PRICE_PER_TOKEN = 0.042 / 1_000_000


class UsageSink:
    """Shared, mutable Jev usage counter for one run."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * JEV_INPUT_PRICE_PER_TOKEN


@dataclass(frozen=True)
class GlmUsage:
    calls: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class RunMetrics:
    steps: int
    glm: GlmUsage
    jev_calls: int
    jev_tokens: int
    jev_cost_usd: float
    wall_seconds: float


def glm_usage(messages: Sequence[BaseMessage]) -> GlmUsage:
    ai = [m for m in messages if isinstance(m, AIMessage)]
    inp = sum((m.usage_metadata or {}).get("input_tokens", 0) for m in ai)
    out = sum((m.usage_metadata or {}).get("output_tokens", 0) for m in ai)
    return GlmUsage(calls=len(ai), input_tokens=inp, output_tokens=out)


def build_metrics(
    messages: Sequence[BaseMessage], sink: UsageSink | None, wall_seconds: float
) -> RunMetrics:
    glm = glm_usage(messages)
    return RunMetrics(
        steps=glm.calls,
        glm=glm,
        jev_calls=sink.calls if sink else 0,
        jev_tokens=(sink.input_tokens + sink.output_tokens) if sink else 0,
        jev_cost_usd=sink.cost_usd if sink else 0.0,
        wall_seconds=wall_seconds,
    )
