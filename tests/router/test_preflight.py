import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from jev_demo_triage.config import MissingKeyError
from jev_router_bench.pool import POOL
from jev_router_bench.preflight import check_models, check_routers, format_checks
from jev_router_bench.routers import FixedRouter
from jev_router_bench.tasks import Task

TASK = Task("add", "arithmetic", "cheap", "What is 2 + 2?", "numeric", 4.0)


def _ok_model():
    return FakeMessagesListChatModel(responses=[AIMessage("OK")])


class Broken:
    def invoke(self, *args, **kwargs):
        raise PermissionError("403 provider Terms Of Service")


class Keyless:
    def invoke(self, *args, **kwargs):
        raise MissingKeyError("OPENROUTER_API_KEY is not set.")


class FailingRouter:
    name = "llm"

    def route(self, task):
        raise ValueError("no JSON")


def test_check_models_reports_each_tier_with_model_id():
    checks = check_models({"cheap": _ok_model(), "strong": Broken()})
    assert [(c.name, c.target, c.ok) for c in checks] == [
        ("cheap", POOL["cheap"].model_id, True),
        ("strong", POOL["strong"].model_id, False),
    ]
    assert checks[0].error is None
    assert checks[1].error == "PermissionError: 403 provider Terms Of Service"


def test_check_routers_routes_one_task():
    seen = []

    class Spy(FixedRouter):
        def route(self, task):
            seen.append(task.id)
            return super().route(task)

    checks = check_routers([Spy("mid"), FailingRouter()], TASK)
    assert seen == ["add"]
    assert [(c.name, c.ok) for c in checks] == [("router always-mid", True), ("router llm", False)]
    assert checks[1].error == "ValueError: no JSON"


def test_missing_key_propagates():
    with pytest.raises(MissingKeyError):
        check_models({"cheap": Keyless()})


def test_format_checks_shows_status_and_error():
    text = format_checks(check_models({"cheap": _ok_model(), "strong": Broken()}))
    assert "cheap" in text and "ok" in text
    assert "FAIL" in text and "403 provider Terms Of Service" in text
