import json

import httpx2
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from jev_demo_triage.jev import make_classifier
from jev_demo_triage.metrics import JEV_INPUT_PRICE_PER_TOKEN
from jev_router_bench.llm_choice import LlmRouteError
from jev_router_bench.pool import ROUTER_LLM, TIERS
from jev_router_bench.routers import (
    ROUTE_QUESTION_ID,
    FixedRouter,
    JevRouter,
    LlmRouter,
    OracleRouter,
    RouteDecision,
)
from jev_router_bench.tasks import Task

TASK = Task("t1", "c", "cheap", "What is 2 + 2?", "numeric", 4.0)

CHOICE_BODY = {
    "model": "typesafe/jev-1.13",
    "answers": {
        "model_route": {
            "type": "choice",
            "choice": "mid",
            "probabilities": {"cheap": 0.2, "mid": 0.7, "strong": 0.1},
            "confidence": 0.6,
        }
    },
    "usage": {"input_tokens": 300, "output_tokens": 10},
}


def _jev_router(handler):
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return JevRouter(
        classifier_factory=lambda q, s: make_classifier(q, sink=s, api_key="k", client=client)
    )


def test_jev_router_sends_choice_question_with_prompt_only():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=CHOICE_BODY)

    decision = _jev_router(handler).route(TASK)
    question = seen["body"]["questions"][ROUTE_QUESTION_ID]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == set(TIERS)
    assert seen["body"]["state"] == "What is 2 + 2?"
    assert decision.tier == "mid"
    assert decision.probabilities == pytest.approx({"cheap": 0.2, "mid": 0.7, "strong": 0.1})
    assert (decision.input_tokens, decision.output_tokens) == (300, 10)
    assert decision.cost_usd == pytest.approx(300 * JEV_INPUT_PRICE_PER_TOKEN)
    assert decision.seconds >= 0.0


def test_jev_router_reports_per_call_usage():
    router = _jev_router(lambda r: httpx2.Response(200, json=CHOICE_BODY))
    router.route(TASK)
    second = router.route(TASK)
    assert second.input_tokens == 300


def test_jev_router_http_error_propagates():
    router = _jev_router(lambda r: httpx2.Response(500, json={"error": "boom"}))
    with pytest.raises(Exception):
        router.route(TASK)


def _fake_llm(text, usage=None):
    message = AIMessage(
        content=text,
        usage_metadata=usage or {"input_tokens": 200, "output_tokens": 20, "total_tokens": 220},
    )
    return FakeMessagesListChatModel(responses=[message])


def test_llm_router_parses_choice_and_costs_tokens():
    model = _fake_llm('{"choice": "strong", "probabilities": {"cheap": 0.1, "mid": 0.2, "strong": 0.7}}')
    decision = LlmRouter(model=model).route(TASK)
    assert decision.tier == "strong"
    assert decision.probabilities == {"cheap": 0.1, "mid": 0.2, "strong": 0.7}
    assert (decision.input_tokens, decision.output_tokens) == (200, 20)
    assert decision.cost_usd == pytest.approx(ROUTER_LLM.cost(200, 20))
    assert decision.reply == '{"choice": "strong", "probabilities": {"cheap": 0.1, "mid": 0.2, "strong": 0.7}}'


def test_llm_router_prompt_contains_criteria_and_task():
    seen = {}

    class Spy(FakeMessagesListChatModel):
        def invoke(self, messages, *args, **kwargs):
            seen["messages"] = messages
            return super().invoke(messages, *args, **kwargs)

    router = LlmRouter(model=Spy(responses=[AIMessage('{"choice": "cheap"}')]))
    router.route(TASK)
    human = seen["messages"][1].content
    for tier in TIERS:
        assert f"- {tier}:" in human
    assert json.dumps(TASK.prompt) in human


def test_llm_router_bad_answer_raises():
    with pytest.raises(LlmRouteError):
        LlmRouter(model=_fake_llm("I'd pick mid")).route(TASK)


def test_llm_router_bad_answer_error_keeps_the_reply():
    reply = "Thinking about it. " * 20 + "TAIL-MARKER"
    with pytest.raises(LlmRouteError, match="TAIL-MARKER"):
        LlmRouter(model=_fake_llm(reply)).route(TASK)


def test_fixed_router():
    router = FixedRouter("strong")
    assert router.name == "always-strong"
    assert router.route(TASK) == RouteDecision("strong")


def test_oracle_router_uses_gold_and_falls_back_to_strong():
    router = OracleRouter({"t1": "cheap", "t2": None})
    assert router.name == "oracle"
    assert router.route(TASK).tier == "cheap"
    assert router.route(Task("t2", "c", "mid", "p", "exact", "x")).tier == "strong"
    assert router.route(Task("t3", "c", "mid", "p", "exact", "x")).tier == "strong"
