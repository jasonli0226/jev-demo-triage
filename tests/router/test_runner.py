import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from jev_demo_triage.config import MissingKeyError
from jev_router_bench.pool import POOL
from jev_router_bench.routers import FixedRouter, RouteDecision
from jev_router_bench.runner import calibrate, e2e_eval, route_eval, run_task
from jev_router_bench.tasks import Task

TASK = Task("add", "arithmetic", "cheap", "What is 2 + 2?", "numeric", 4.0)
USAGE = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}


def _model(text):
    return FakeMessagesListChatModel(responses=[AIMessage(content=text, usage_metadata=USAGE)])


class Broken:
    def invoke(self, *args, **kwargs):
        raise TimeoutError("slow")


class FailingRouter:
    name = "bad"

    def route(self, task):
        raise ValueError("cannot route")


class KeylessRouter:
    name = "keyless"

    def route(self, task):
        raise MissingKeyError("no key")


def test_run_task_grades_and_costs():
    run = run_task(TASK, "mid", _model("ANSWER: 4"))
    assert (run.task_id, run.tier, run.passed, run.reason) == ("add", "mid", True, "ok")
    assert (run.input_tokens, run.output_tokens) == (100, 50)
    assert run.cost_usd == pytest.approx(POOL["mid"].cost(100, 50))
    assert run.error is None
    assert run.reply == "ANSWER: 4"


def test_run_task_wrong_answer():
    run = run_task(TASK, "cheap", _model("ANSWER: 5"))
    assert (run.passed, run.reason) == (False, "wrong")


def test_run_task_model_error_is_recorded():
    run = run_task(TASK, "strong", Broken())
    assert (run.passed, run.reason) == (False, "error")
    assert run.error == "TimeoutError: slow"
    assert run.cost_usd == 0.0
    assert run.reply is None


def test_run_task_grader_exception_is_recorded():
    def bad_grader(task, text):
        raise RuntimeError("boom")

    run = run_task(TASK, "mid", _model("ANSWER: 4"), grader=bad_grader)
    assert (run.passed, run.reason) == (False, "error")
    assert run.error == "RuntimeError: boom"
    assert run.reply == "ANSWER: 4"
    assert (run.input_tokens, run.output_tokens) == (100, 50)
    assert run.cost_usd == pytest.approx(POOL["mid"].cost(100, 50))


def test_calibrate_runs_every_tier_repeat_times():
    models = {"cheap": _model("ANSWER: 5"), "mid": _model("ANSWER: 4"), "strong": _model("ANSWER: 4")}
    seen = []
    runs = calibrate([TASK], models, repeat=2, on_result=seen.append)
    assert [(r.tier, r.passed) for r in runs] == [
        ("cheap", False), ("cheap", False), ("mid", True), ("mid", True), ("strong", True), ("strong", True),
    ]
    assert seen == runs


def test_route_eval_records_decisions_and_errors():
    records = route_eval([TASK], [FixedRouter("mid"), FailingRouter()], repeat=2)
    assert [(r.router, r.repeat) for r in records] == [
        ("always-mid", 0), ("always-mid", 1), ("bad", 0), ("bad", 1),
    ]
    assert records[0].decision == RouteDecision("mid")
    assert records[2].decision is None
    assert records[2].error == "ValueError: cannot route"


def test_missing_key_is_not_swallowed():
    with pytest.raises(MissingKeyError):
        route_eval([TASK], [KeylessRouter()], repeat=1)
    with pytest.raises(MissingKeyError):
        e2e_eval([TASK], [KeylessRouter()], {}, repeat=1)


def test_e2e_eval_runs_chosen_model():
    models = {"cheap": _model("ANSWER: 5"), "mid": _model("ANSWER: 4"), "strong": _model("ANSWER: 4")}
    records = e2e_eval([TASK], [FixedRouter("cheap"), FixedRouter("mid"), FailingRouter()], models, repeat=1)
    cheap, mid, bad = records
    assert cheap.run.tier == "cheap" and not cheap.run.passed
    assert mid.run.tier == "mid" and mid.run.passed
    assert bad.decision is None and bad.run is None and bad.error == "ValueError: cannot route"
