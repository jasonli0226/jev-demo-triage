"""Per-run counters. Jev cost is derived from tokens (output tokens are free)."""

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage

JEV_INPUT_PRICE_PER_TOKEN = 0.042 / 1_000_000
# z-ai/glm-4.7 prices as listed by OpenRouter on 2026-09-20. The cost is an estimate:
# prices change and the run does not read them from the API.
GLM_INPUT_PRICE_PER_TOKEN = 0.43 / 1_000_000
GLM_OUTPUT_PRICE_PER_TOKEN = 1.75 / 1_000_000


class UsageSink:
    """Shared, mutable classifier usage counter for one run (tokens, seconds, cost)."""

    def __init__(
        self,
        input_price: float = JEV_INPUT_PRICE_PER_TOKEN,
        output_price: float = 0.0,
    ) -> None:
        self.input_price = input_price
        self.output_price = output_price
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.seconds = 0.0

    def add(self, input_tokens: int, output_tokens: int, seconds: float = 0.0) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.seconds += seconds

    def add_seconds(self, seconds: float) -> None:
        self.seconds += seconds

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * self.input_price + self.output_tokens * self.output_price


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
    classifier_seconds: float = 0.0
    llm_gate_calls: int = 0
    llm_gate_tokens: int = 0
    llm_gate_cost_usd: float = 0.0


def glm_usage(messages: Sequence[BaseMessage]) -> GlmUsage:
    ai = [m for m in messages if isinstance(m, AIMessage)]
    inp = sum((m.usage_metadata or {}).get("input_tokens", 0) for m in ai)
    out = sum((m.usage_metadata or {}).get("output_tokens", 0) for m in ai)
    return GlmUsage(calls=len(ai), input_tokens=inp, output_tokens=out)


def build_metrics(
    messages: Sequence[BaseMessage],
    sink: UsageSink | None,
    wall_seconds: float,
    llm_sink: UsageSink | None = None,
) -> RunMetrics:
    glm = glm_usage(messages)
    return RunMetrics(
        steps=glm.calls,
        glm=glm,
        jev_calls=sink.calls if sink else 0,
        jev_tokens=(sink.input_tokens + sink.output_tokens) if sink else 0,
        jev_cost_usd=sink.cost_usd if sink else 0.0,
        wall_seconds=wall_seconds,
        classifier_seconds=(sink.seconds if sink else 0.0) + (llm_sink.seconds if llm_sink else 0.0),
        llm_gate_calls=llm_sink.calls if llm_sink else 0,
        llm_gate_tokens=(llm_sink.input_tokens + llm_sink.output_tokens) if llm_sink else 0,
        llm_gate_cost_usd=llm_sink.cost_usd if llm_sink else 0.0,
    )
