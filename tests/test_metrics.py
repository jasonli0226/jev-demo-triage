import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from jev_demo_triage.metrics import UsageSink, build_metrics, glm_usage


def _ai(inp, out, tool_calls=None):
    return AIMessage(
        content="x",
        tool_calls=tool_calls or [],
        usage_metadata={"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out},
    )


def test_usage_sink_accumulates_and_prices_input_only():
    sink = UsageSink()
    sink.add(285, 20)
    sink.add(115, 5)
    assert (sink.calls, sink.input_tokens, sink.output_tokens) == (2, 400, 25)
    assert sink.cost_usd == pytest.approx(400 * 0.042 / 1_000_000)


def test_usage_sink_matches_observed_spike_cost():
    sink = UsageSink()
    sink.add(285, 20)
    assert sink.cost_usd == pytest.approx(0.00001197)


def test_glm_usage_sums_ai_messages_only():
    msgs = [HumanMessage("hi"), _ai(100, 10), ToolMessage("r", tool_call_id="1"), _ai(200, 20)]
    usage = glm_usage(msgs)
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (2, 300, 30)


def test_glm_usage_handles_missing_metadata():
    msgs = [AIMessage(content="no usage")]
    usage = glm_usage(msgs)
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (1, 0, 0)


def test_build_metrics_without_sink_has_zero_jev():
    m = build_metrics([_ai(10, 1)], None, 1.5)
    assert (m.jev_calls, m.jev_tokens, m.jev_cost_usd) == (0, 0, 0.0)
    assert m.wall_seconds == 1.5
    assert m.steps == 1


def test_build_metrics_with_sink():
    sink = UsageSink()
    sink.add(300, 40)
    m = build_metrics([_ai(10, 1, [{"name": "t", "args": {}, "id": "1"}]), _ai(20, 2)], sink, 2.0)
    assert m.steps == 2
    assert m.jev_calls == 1
    assert m.jev_tokens == 340
    assert m.jev_cost_usd == pytest.approx(300 * 0.042 / 1_000_000)
