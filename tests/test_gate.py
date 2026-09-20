import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_typesafe.types import ClassificationResponse

from jev_demo_triage.gate import GATED_TOOLS, QUESTION_ID, OpenRouterAutoMode
from jev_demo_triage.metrics import UsageSink
from tests.conftest import ScriptedModel, ai_tool_call


class StubClassifier:
    def __init__(self, probability, sink=None):
        self.probability, self.sink, self.calls = probability, sink, 0

    def invoke(self, state, *args, **kwargs):
        self.calls += 1
        if self.sink:
            self.sink.add(200, 20)
        return ClassificationResponse.model_validate({
            "model": "m",
            "answers": {QUESTION_ID: {"type": "noul", "noul": self.probability}},
            "usage": {"input_tokens": 200, "output_tokens": 20},
        })


def _factory(probability):
    created = []

    def factory(questions, sink):
        clf = StubClassifier(probability, sink)
        clf.questions = questions
        created.append(clf)
        return clf

    return factory, created


def _run(probability, tool_name="run_shell", args=None):
    log = []

    @tool
    def run_shell(command: str) -> str:
        """Run a shell command."""
        log.append(command)
        return "ok"

    @tool
    def read_logs(service: str) -> str:
        """Read logs."""
        return "logs"

    factory, created = _factory(probability)
    sink = UsageSink()
    gate = OpenRouterAutoMode(sink=sink, classifier_factory=factory)
    model = ScriptedModel(responses=[
        ai_tool_call(tool_name, args or {"command": "rm -rf /data"}),
        AIMessage(content="done"),
    ])
    agent = create_agent(model, tools=[run_shell, read_logs], middleware=[gate])
    out = agent.invoke({"messages": [HumanMessage("go")]}, config={"recursion_limit": 15})
    return out, log, created, sink


def test_gated_tools_constant():
    assert GATED_TOOLS == ("run_shell", "restart_service")
    assert QUESTION_ID == "is_risky"


def test_high_risk_probability_blocks_call():
    out, log, created, _ = _run(0.9)
    assert log == []
    tool_msg = out["messages"][2]
    assert tool_msg.status == "error" and "blocked" in tool_msg.content
    assert created[0].calls == 1


def test_threshold_boundary_blocks_at_half():
    _, log, _, _ = _run(0.5)
    assert log == []


def test_low_risk_probability_allows_call():
    _, log, _, _ = _run(0.1)
    assert log == ["rm -rf /data"]


def test_ungated_tool_is_not_classified():
    _, _, created, _ = _run(0.99, tool_name="read_logs", args={"service": "api"})
    assert created[0].calls == 0


def test_classifier_receives_package_question_and_shared_sink():
    _, _, created, sink = _run(0.1)
    assert QUESTION_ID in created[0].questions
    assert sink.calls == 1 and sink.input_tokens == 200


def test_classifier_failure_fails_closed():
    log = []

    @tool
    def run_shell(command: str) -> str:
        """Run a shell command."""
        log.append(command)
        return "ok"

    class Boom:
        def invoke(self, *a, **k):
            raise RuntimeError("jev down")

    gate = OpenRouterAutoMode(classifier_factory=lambda q, s: Boom())
    model = ScriptedModel(responses=[ai_tool_call("run_shell", {"command": "ls"}), AIMessage(content="x")])
    agent = create_agent(model, tools=[run_shell], middleware=[gate])
    with pytest.raises(Exception) as info:
        agent.invoke({"messages": [HumanMessage("go")]})
    assert "jev down" in str(info.value)
    assert log == []


def test_placeholder_env_key_is_set_only_when_missing(monkeypatch):
    import os

    monkeypatch.setenv("TYPESAFE_API_KEY", "tmp")  # registers restore-on-teardown
    monkeypatch.delenv("TYPESAFE_API_KEY")
    OpenRouterAutoMode(classifier_factory=_factory(0.1)[0])
    assert os.environ["TYPESAFE_API_KEY"]  # placeholder present
    monkeypatch.setenv("TYPESAFE_API_KEY", "real")
    OpenRouterAutoMode(classifier_factory=_factory(0.1)[0])
    assert os.environ["TYPESAFE_API_KEY"] == "real"


def test_mirrored_constants_match_package():
    # Private names on purpose: this test exists to catch drift in langchain-typesafe.
    from langchain_typesafe.experimental.middleware import auto_mode

    from jev_demo_triage import gate, trace

    assert gate.QUESTION_ID == auto_mode._QUESTION_ID
    assert trace.GATE_BLOCK_THRESHOLD == auto_mode._PROBABILITY_THRESHOLD
