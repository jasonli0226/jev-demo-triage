import pytest
from langchain_core.messages import AIMessage
from langchain_typesafe.types import ClassificationResponse

from jev_demo_triage.agent import run_scenario
from jev_demo_triage.scenarios import SCENARIOS
from jev_demo_triage.trace import TraceHandler, Tracer, traced_factory
from tests.conftest import ScriptedModel, ai_tool_call
from tests.test_agent import _gate_factory


def _response(p: float) -> ClassificationResponse:
    return ClassificationResponse.model_validate({
        "model": "m",
        "answers": {"is_risky": {"type": "noul", "noul": p}},
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })


def _stub(p: float):
    class Clf:
        def __init__(self, questions, sink):
            self.questions = questions

        def invoke(self, state, *a, **k):
            return _response(p)

    return Clf


def _questions():
    from langchain_typesafe.types import Noul

    return {"is_risky": Noul(type="noul", instructions="Is this risky?")}


GATE_STATE = {"tool_call": {"name": "run_shell", "args": {"command": "rm -rf /x"}}}


def test_tracer_truncates_with_ellipsis():
    lines: list[str] = []
    Tracer(write=lines.append, max_len=10).line("GLM", "x" * 50)
    assert lines == ["GLM  " + "x" * 10 + "..."]


def test_tracer_flattens_newlines():
    lines: list[str] = []
    Tracer(write=lines.append).line("TOOL", "a\nb\nc")
    assert lines == ["TOOL a b c"]


def test_tracer_defaults_to_stderr(capsys):
    Tracer().line("DONE", "hi")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "DONE hi\n"


def test_tracer_header():
    lines: list[str] = []
    Tracer(write=lines.append).header("risky-bait", "both")
    assert lines == ["== risky-bait / both =="]


@pytest.mark.parametrize("p,verdict", [(0.95, "BLOCKED"), (0.10, "ALLOWED"), (0.5, "BLOCKED")])
def test_gate_call_prints_verdict(p, verdict):
    lines: list[str] = []
    clf = traced_factory(_stub(p), Tracer(write=lines.append))(_questions(), None)
    clf.invoke(GATE_STATE)
    gate = [ln for ln in lines if ln.startswith("GATE")]
    assert len(gate) == 1
    assert "run_shell" in gate[0] and "rm -rf /x" in gate[0]
    assert f"risk {p:.2f}" in gate[0] and gate[0].endswith(verdict)
    assert any(ln.startswith("JEV  -> is_risky: noul") for ln in lines)


def test_ask_jev_call_prints_summary():
    lines: list[str] = []
    clf = traced_factory(_stub(0.12), Tracer(write=lines.append))(_questions(), None)
    clf.invoke("disk is full")
    assert any(ln.startswith("JEV  -> is_risky") for ln in lines)
    assert any(ln.startswith("JEV  <- is_risky: no 0.12") for ln in lines)
    assert not any(ln.startswith("GATE") for ln in lines)


def test_classifier_error_is_traced_and_reraised():
    class Boom:
        def __init__(self, questions, sink):
            pass

        def invoke(self, state, *a, **k):
            raise RuntimeError("upstream 500")

    lines: list[str] = []
    clf = traced_factory(Boom, Tracer(write=lines.append))(_questions(), None)
    with pytest.raises(RuntimeError, match="upstream 500"):
        clf.invoke("x")
    assert any(ln.startswith("JEV  !! ") and "upstream 500" in ln for ln in lines)


def test_unknown_attributes_delegate_to_inner_classifier():
    questions = _questions()
    clf = traced_factory(_stub(0.1), Tracer(write=lambda s: None))(questions, None)
    assert clf.questions is questions


def _both_script():
    return ScriptedModel(responses=[
        ai_tool_call("ask_jev", {
            "state": "disk full, only customer archives left",
            "questions": {"enough_evidence": {"type": "noul", "instructions": "Is the evidence enough to act?"}},
        }),
        ai_tool_call("run_shell", {"command": "rm -rf /var/lib/reports/data"}, "2"),
        ai_tool_call("escalate_to_human", {"reason": "cannot safely delete"}, "3"),
        AIMessage(content="Escalated to a human."),
    ])


def _index(lines, prefix, start=0):
    for i in range(start, len(lines)):
        if lines[i].startswith(prefix):
            return i
    raise AssertionError(f"no line starting {prefix!r} after {start}: {lines}")


def test_both_mode_trace_order_and_same_result():
    lines: list[str] = []
    traced = run_scenario(
        SCENARIOS["risky-bait"], "both", model=_both_script(),
        classifier_factory=_gate_factory(0.95), tracer=Tracer(write=lines.append),
    )
    plain = run_scenario(
        SCENARIOS["risky-bait"], "both", model=_both_script(), classifier_factory=_gate_factory(0.95),
    )
    assert lines[0] == "== risky-bait / both (policy: tuned) =="
    i = _index(lines, "GLM  -> ask_jev")
    i = _index(lines, "JEV  <-", i)
    i = _index(lines, "GLM  -> run_shell", i)
    i = _index(lines, "GATE run_shell", i)
    assert lines[i].endswith("BLOCKED")
    i = _index(lines, "GLM  -> escalate_to_human", i)
    i = _index(lines, "DONE final answer: Escalated", i)
    assert (traced.passed, traced.actions, traced.metrics.jev_calls) == (
        plain.passed, plain.actions, plain.metrics.jev_calls,
    )


def test_baseline_trace_has_no_jev_lines_and_builds_no_classifier():
    built = []

    def factory(questions, sink):
        built.append(1)
        raise AssertionError("baseline must not build a classifier")

    lines: list[str] = []
    model = ScriptedModel(responses=[
        ai_tool_call("read_logs", {"service": "checkout"}),
        ai_tool_call("rollback_deploy", {"service": "checkout"}, "2"),
        AIMessage(content="Rolled back."),
    ])
    run_scenario(
        SCENARIOS["deploy-regression"], "baseline", model=model,
        classifier_factory=factory, tracer=Tracer(write=lines.append),
    )
    assert built == []
    assert any(ln.startswith("GLM  -> read_logs(") for ln in lines)
    assert any(ln.startswith("TOOL read_logs <-") for ln in lines)
    assert not any(ln.startswith(("JEV", "GATE")) for ln in lines)


def test_errored_run_traces_done_error():
    lines: list[str] = []
    looping = [ai_tool_call("read_logs", {"service": "billing"}, str(i)) for i in range(40)]
    run_scenario(SCENARIOS["simple-restart"], "baseline", model=ScriptedModel(responses=looping),
                 tracer=Tracer(write=lines.append))
    assert lines[-1].startswith("DONE ERROR: recursion limit")


def test_trace_handler_never_prints_secrets():
    # TraceHandler only ever formats tool-call names/args, message content and tool
    # output; kwargs, serialized/invocation params, tags and metadata are never read.
    from uuid import uuid4

    from langchain_core.messages import ToolMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    secret = "sk-or-SENTINEL"
    leaky = {
        "serialized": {"kwargs": {"openai_api_key": secret}},
        "invocation_params": {"api_key": secret, "headers": {"Authorization": f"Bearer {secret}"}},
        "tags": [secret],
        "metadata": {"key": secret},
    }
    lines: list[str] = []
    handler = TraceHandler(Tracer(write=lines.append))
    msg = ai_tool_call("read_logs", {"service": "billing"})
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id=uuid4(), **leaky)
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=AIMessage(content="all done"))]]), run_id=uuid4(), **leaky
    )
    handler.on_tool_end(ToolMessage(content="ok", tool_call_id="1"), run_id=uuid4(), **leaky)
    handler.on_tool_error(RuntimeError("boom"), run_id=uuid4(), **leaky)
    assert len(lines) == 4
    assert not any("SENTINEL" in ln for ln in lines)


@pytest.mark.parametrize("p,verdict", [(0.95, "BLOCKED"), (0.10, "ALLOWED")])
def test_long_gate_command_keeps_verdict(p, verdict):
    lines: list[str] = []
    clf = traced_factory(_stub(p), Tracer(write=lines.append))(_questions(), None)
    command = "find /var/lib -type f | xargs rm -rf && " + "x" * 200
    clf.invoke({"tool_call": {"name": "run_shell", "args": {"command": command}}})
    gate = next(ln for ln in lines if ln.startswith("GATE"))
    assert "..." in gate
    assert gate.endswith(f"risk {p:.2f} -> {verdict}")


def test_tool_result_line_names_the_tool():
    from uuid import uuid4

    lines: list[str] = []
    handler = TraceHandler(Tracer(write=lines.append))
    rid, other = uuid4(), uuid4()
    handler.on_tool_start({"name": "get_metrics"}, "{}", run_id=rid)
    handler.on_tool_start({}, "{}", run_id=other, name="read_logs")
    handler.on_tool_end("cpu 90%", run_id=rid)
    handler.on_tool_end("logs", run_id=other)
    handler.on_tool_end("mystery", run_id=uuid4())
    handler.on_tool_error(RuntimeError("boom"), run_id=rid)
    assert lines == [
        "TOOL get_metrics <- cpu 90%",
        "TOOL read_logs <- logs",
        "TOOL <- mystery",
        "TOOL <- error: boom",
    ]


def test_traced_classifier_private_attributes_do_not_recurse():
    import copy

    clf = traced_factory(_stub(0.1), Tracer(write=lambda s: None))(_questions(), None)
    bare = type(clf).__new__(type(clf))  # _inner unset, as during unpickle/deepcopy
    with pytest.raises(AttributeError):
        bare._inner
    assert copy.copy(bare) is not None


def test_tracer_header_shows_policy_when_given():
    lines: list[str] = []
    Tracer(write=lines.append).header("simple-restart", "gate", gate_policy="tuned")
    assert lines == ["== simple-restart / gate (policy: tuned) =="]


@pytest.mark.parametrize("mode,expected", [
    ("gate", "== risky-bait / gate (policy: default) =="),
    ("both", "== risky-bait / both (policy: default) =="),
    ("baseline", "== risky-bait / baseline =="),
    ("tool", "== risky-bait / tool =="),
])
def test_run_header_shows_policy_only_for_gate_modes(mode, expected):
    lines: list[str] = []
    run_scenario(
        SCENARIOS["risky-bait"], mode, model=ScriptedModel(responses=[AIMessage(content="ok")]),
        classifier_factory=_gate_factory(0.1), tracer=Tracer(write=lines.append), gate_policy="default",
    )
    assert lines[0] == expected
