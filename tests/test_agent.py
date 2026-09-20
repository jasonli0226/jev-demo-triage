import pytest
from langchain_core.messages import AIMessage

from jev_demo_triage.agent import MODES, RunResult, run_scenario
from jev_demo_triage.scenarios import SCENARIOS
from tests.conftest import ScriptedModel, ai_tool_call


def test_modes_constant():
    assert MODES == ("baseline", "tool", "gate", "both", "llm-gate")


def test_baseline_run_passes_when_model_takes_correct_action():
    model = ScriptedModel(responses=[
        ai_tool_call("read_logs", {"service": "checkout"}),
        ai_tool_call("rollback_deploy", {"service": "checkout"}, "2"),
        AIMessage(content="Rolled back checkout to v2.30.4."),
    ])
    result = run_scenario(SCENARIOS["deploy-regression"], "baseline", model=model)
    assert isinstance(result, RunResult)
    assert result.passed is True
    assert [a.name for a in result.actions] == ["rollback_deploy"]
    assert "Rolled back" in result.final_answer
    assert result.mode == "baseline"
    assert result.scenario == "deploy-regression"
    assert result.metrics.jev_calls == 0
    assert result.error is None


def test_baseline_run_fails_when_model_takes_wrong_action():
    model = ScriptedModel(responses=[
        ai_tool_call("restart_service", {"service": "checkout"}),
        AIMessage(content="Restarted."),
    ])
    result = run_scenario(SCENARIOS["deploy-regression"], "baseline", model=model)
    assert result.passed is False


def test_recursion_limit_is_reported_as_error_not_raised():
    looping = [ai_tool_call("read_logs", {"service": "billing"}, str(i)) for i in range(40)]
    model = ScriptedModel(responses=looping)
    result = run_scenario(SCENARIOS["simple-restart"], "baseline", model=model)
    assert result.passed is False
    assert result.error is not None and "recursion" in result.error.lower()


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        run_scenario(SCENARIOS["simple-restart"], "bogus", model=ScriptedModel(responses=[]))


def test_make_model_requires_key(monkeypatch):
    from jev_demo_triage.agent import make_model
    from jev_demo_triage.config import MissingKeyError

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingKeyError):
        make_model()


def test_make_model_uses_openrouter_and_temperature_zero(monkeypatch):
    from jev_demo_triage.agent import make_model

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("GLM_MODEL", raising=False)
    model = make_model()
    assert model.model_name == "z-ai/glm-4.7"
    assert model.temperature == 0
    assert str(model.openai_api_base).startswith("https://openrouter.ai/api/v1")


def test_tool_mode_uses_ask_jev_and_reports_jev_usage():
    from jev_demo_triage.metrics import UsageSink

    class FakeClf:
        def __init__(self, questions, sink):
            self.sink = sink

        def invoke(self, state):
            from langchain_typesafe.types import ClassificationResponse
            self.sink.add(300, 40)
            return ClassificationResponse.model_validate({
                "model": "m",
                "answers": {"enough": {"type": "noul", "noul": 0.1}},
                "usage": {"input_tokens": 300, "output_tokens": 40},
            })

    model = ScriptedModel(responses=[
        ai_tool_call("ask_jev", {
            "state": "search slow once, no errors",
            "questions": {"enough": {"type": "noul", "instructions": "Is the evidence enough to act?"}},
        }),
        ai_tool_call("escalate_to_human", {"reason": "insufficient evidence"}, "2"),
        AIMessage(content="Escalated: evidence insufficient."),
    ])
    result = run_scenario(SCENARIOS["ambiguous"], "tool", model=model, classifier_factory=FakeClf)
    assert result.passed is True
    assert result.metrics.jev_calls == 1
    assert result.metrics.jev_tokens == 340
    assert result.metrics.jev_cost_usd == pytest.approx(300 * 0.042 / 1_000_000)


def _gate_factory(probability):
    from langchain_typesafe.types import ClassificationResponse

    class Clf:
        def __init__(self, questions, sink):
            self.sink = sink

        def invoke(self, state, *a, **k):
            self.sink.add(250, 15)
            return ClassificationResponse.model_validate({
                "model": "m",
                "answers": {"is_risky": {"type": "noul", "noul": probability}},
                "usage": {"input_tokens": 250, "output_tokens": 15},
            })

    return Clf


def test_gate_mode_blocks_rm_rf_and_agent_recovers():
    model = ScriptedModel(responses=[
        ai_tool_call("run_shell", {"command": "rm -rf /var/lib/reports/data"}),
        ai_tool_call("escalate_to_human", {"reason": "disk full, cannot safely delete customer archives"}, "2"),
        AIMessage(content="Escalated."),
    ])
    result = run_scenario(SCENARIOS["risky-bait"], "gate", model=model, classifier_factory=_gate_factory(0.95))
    assert result.passed is True
    assert [a.name for a in result.actions] == ["escalate_to_human"]  # blocked call never recorded
    assert result.metrics.jev_calls == 1


def test_gate_that_allows_rm_rf_fails_the_pass_check():
    model = ScriptedModel(responses=[
        ai_tool_call("run_shell", {"command": "rm -rf /var/lib/reports/data"}),
        AIMessage(content="Freed space."),
    ])
    result = run_scenario(SCENARIOS["risky-bait"], "gate", model=model, classifier_factory=_gate_factory(0.1))
    assert result.passed is False  # gate allowed it, pass check catches the rm -rf


def test_both_mode_has_ask_jev_and_gate():
    model = ScriptedModel(responses=[
        ai_tool_call("ask_jev", {
            "state": "disk full, only customer archives left to delete",
            "questions": {"enough": {"type": "noul", "instructions": "Is the evidence enough to act?"}},
        }),
        ai_tool_call("run_shell", {"command": "rm -rf /var/lib/reports/data"}, "2"),
        ai_tool_call("escalate_to_human", {"reason": "cannot safely delete customer archives"}, "3"),
        AIMessage(content="Escalated."),
    ])
    result = run_scenario(SCENARIOS["risky-bait"], "both", model=model, classifier_factory=_gate_factory(0.95))
    assert result.metrics.jev_calls == 2  # one ask_jev call + one gate classification, one shared sink
    assert result.metrics.jev_tokens == 2 * (250 + 15)
    assert [a.name for a in result.actions] == ["escalate_to_human"]  # blocked run_shell never recorded
    assert result.passed is True


def test_recursion_limit_allows_fifteen_model_steps():
    calls = [ai_tool_call("read_logs", {"service": "billing"}, str(i)) for i in range(14)]
    model = ScriptedModel(responses=[*calls, AIMessage(content="done")])
    result = run_scenario(SCENARIOS["simple-restart"], "baseline", model=model)
    assert result.error is None
    assert result.metrics.glm.calls == 15


def test_classifier_failure_is_reported_as_error_not_raised():
    class Boom:
        def __init__(self, questions, sink):
            pass

        def invoke(self, state, *a, **k):
            raise RuntimeError("jev down")

    model = ScriptedModel(responses=[
        ai_tool_call("read_logs", {"service": "billing"}),
        ai_tool_call("run_shell", {"command": "ls"}, "2"),
        AIMessage(content="unreachable"),
    ])
    result = run_scenario(SCENARIOS["risky-bait"], "gate", model=model, classifier_factory=Boom)
    assert result.passed is False
    assert result.error is not None and "jev down" in result.error
    assert result.error.startswith("run failed: RuntimeError")
    assert [a.name for a in result.actions] == []  # blocked before execution


def test_missing_key_still_raises_from_run_scenario(monkeypatch):
    from jev_demo_triage.config import MissingKeyError

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingKeyError):
        run_scenario(SCENARIOS["simple-restart"], "baseline")


def test_keyboard_interrupt_propagates():
    class Interrupted(ScriptedModel):
        def _generate(self, *a, **k):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_scenario(SCENARIOS["simple-restart"], "baseline", model=Interrupted(responses=[]))


def test_no_callbacks_key_in_config_when_not_tracing(monkeypatch):
    import jev_demo_triage.agent as agent_mod
    from jev_demo_triage.trace import Tracer

    real_create = agent_mod.create_agent
    configs: list[dict] = []

    class Spy:
        def __init__(self, inner):
            self._inner = inner

        def invoke(self, payload, config=None, **kwargs):
            configs.append(config)
            return self._inner.invoke(payload, config=config, **kwargs)

    monkeypatch.setattr(agent_mod, "create_agent", lambda *a, **k: Spy(real_create(*a, **k)))

    def script():
        return ScriptedModel(responses=[AIMessage(content="nothing to do")])

    run_scenario(SCENARIOS["simple-restart"], "baseline", model=script())
    run_scenario(SCENARIOS["simple-restart"], "baseline", model=script(), tracer=Tracer(write=lambda s: None))
    assert "callbacks" not in configs[0]
    assert configs[0]["recursion_limit"] == agent_mod.RECURSION_LIMIT
    assert len(configs[1]["callbacks"]) == 1


def test_unknown_gate_policy_rejected_before_key_or_model(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError) as info:
        run_scenario(SCENARIOS["risky-bait"], "gate", model=None, gate_policy="lax")
    assert "tuned" in str(info.value) and "default" in str(info.value)


@pytest.mark.parametrize("policy", ["tuned", "default"])
@pytest.mark.parametrize("mode", ["gate", "both"])
def test_gate_and_both_record_gate_policy(mode, policy):
    model = ScriptedModel(responses=[AIMessage(content="nothing to do")])
    result = run_scenario(
        SCENARIOS["risky-bait"], mode, model=model,
        classifier_factory=_gate_factory(0.1), gate_policy=policy,
    )
    assert result.gate_policy == policy


def test_gate_policy_defaults_to_tuned():
    model = ScriptedModel(responses=[AIMessage(content="nothing to do")])
    result = run_scenario(
        SCENARIOS["risky-bait"], "gate", model=model, classifier_factory=_gate_factory(0.1),
    )
    assert result.gate_policy == "tuned"


@pytest.mark.parametrize("mode", ["baseline", "tool"])
def test_baseline_and_tool_have_no_gate_policy(mode):
    model = ScriptedModel(responses=[AIMessage(content="nothing to do")])
    result = run_scenario(
        SCENARIOS["risky-bait"], mode, model=model,
        classifier_factory=_gate_factory(0.1), gate_policy="default",
    )
    assert result.gate_policy is None


def _capturing_factory(seen):
    def factory(questions, sink):
        seen.append(questions)
        return _gate_factory(0.1)(questions, sink)

    return factory


def _gate_questions(seen):
    from jev_demo_triage.gate import QUESTION_ID

    return [q[QUESTION_ID] for q in seen if QUESTION_ID in q]


@pytest.mark.parametrize("mode", ["gate", "both"])
def test_tuned_policy_reaches_gate_classifier_through_run_scenario(mode):
    from jev_demo_triage.gate import TUNED_CRITERIA, TUNED_INSTRUCTIONS

    seen: list = []
    run_scenario(
        SCENARIOS["risky-bait"], mode, model=ScriptedModel(responses=[AIMessage(content="ok")]),
        classifier_factory=_capturing_factory(seen), gate_policy="tuned",
    )
    questions = _gate_questions(seen)
    assert len(questions) == 1
    assert questions[0].instructions == TUNED_INSTRUCTIONS
    assert questions[0].criteria == TUNED_CRITERIA


@pytest.mark.parametrize("mode", ["gate", "both"])
def test_default_policy_reaches_gate_classifier_through_run_scenario(mode):
    from jev_demo_triage.gate import TUNED_CRITERIA, TUNED_INSTRUCTIONS

    seen: list = []
    run_scenario(
        SCENARIOS["risky-bait"], mode, model=ScriptedModel(responses=[AIMessage(content="ok")]),
        classifier_factory=_capturing_factory(seen), gate_policy="default",
    )
    questions = _gate_questions(seen)
    assert len(questions) == 1
    assert "Only explicit user messages" in questions[0].instructions
    assert questions[0].instructions != TUNED_INSTRUCTIONS
    assert questions[0].criteria != TUNED_CRITERIA


class _GateModel:
    """Fake gate chat model: scripted replies, captures every prompt it is sent."""

    def __init__(self, replies, usage=(300, 12)):
        self.replies, self.usage, self.prompts = list(replies), usage, []

    def invoke(self, messages, *a, **k):
        self.prompts.append(messages)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=reply, usage_metadata={
            "input_tokens": self.usage[0], "output_tokens": self.usage[1],
            "total_tokens": sum(self.usage),
        })


def _rm_then_escalate():
    return ScriptedModel(responses=[
        ai_tool_call("run_shell", {"command": "rm -rf /var/lib/reports/data"}),
        ai_tool_call("escalate_to_human", {"reason": "cannot safely delete customer archives"}, "2"),
        AIMessage(content="Escalated."),
    ])


def test_llm_gate_blocks_risky_call_and_agent_recovers():
    gate_model = _GateModel(['{"probability": 0.9}'])
    result = run_scenario(SCENARIOS["risky-bait"], "llm-gate", model=_rm_then_escalate(), gate_model=gate_model)
    assert result.passed is True and result.error is None
    assert [a.name for a in result.actions] == ["escalate_to_human"]
    assert len(gate_model.prompts) == 1


def test_llm_gate_allows_low_risk_call():
    result = run_scenario(
        SCENARIOS["risky-bait"], "llm-gate", model=_rm_then_escalate(),
        gate_model=_GateModel(['{"probability": 0.1}']),
    )
    assert [a.name for a in result.actions] == ["run_shell", "escalate_to_human"]
    assert result.passed is False  # gate let the rm -rf through


def test_llm_gate_model_error_fails_closed_as_run_error():
    result = run_scenario(
        SCENARIOS["risky-bait"], "llm-gate", model=_rm_then_escalate(),
        gate_model=_GateModel([RuntimeError("glm gate down")]),
    )
    assert result.passed is False
    assert result.error is not None and "glm gate down" in result.error
    assert [a.name for a in result.actions] == []


def test_llm_gate_unparseable_answer_fails_closed():
    result = run_scenario(
        SCENARIOS["risky-bait"], "llm-gate", model=_rm_then_escalate(),
        gate_model=_GateModel(["no idea"]),
    )
    assert result.error is not None and "LlmGateError" in result.error
    assert [a.name for a in result.actions] == []


def test_llm_gate_metrics_use_glm_prices_and_leave_jev_fields_zero():
    from jev_demo_triage.metrics import GLM_INPUT_PRICE_PER_TOKEN, GLM_OUTPUT_PRICE_PER_TOKEN

    result = run_scenario(
        SCENARIOS["risky-bait"], "llm-gate", model=_rm_then_escalate(),
        gate_model=_GateModel(['{"probability": 0.9}'], usage=(300, 12)),
    )
    m = result.metrics
    assert m.llm_gate_calls == 1 and m.llm_gate_tokens == 312
    assert m.llm_gate_cost_usd == pytest.approx(
        300 * GLM_INPUT_PRICE_PER_TOKEN + 12 * GLM_OUTPUT_PRICE_PER_TOKEN
    )
    assert m.classifier_seconds > 0
    assert (m.jev_calls, m.jev_tokens, m.jev_cost_usd) == (0, 0, 0.0)
    assert result.gate_policy == "tuned"


@pytest.mark.parametrize("policy", ["tuned", "default"])
def test_llm_gate_policy_reaches_gate_model_prompt(policy):
    from jev_demo_triage.gate import TUNED_INSTRUCTIONS

    gate_model = _GateModel(['{"probability": 0.9}'])
    result = run_scenario(
        SCENARIOS["risky-bait"], "llm-gate", model=_rm_then_escalate(),
        gate_model=gate_model, gate_policy=policy,
    )
    human = gate_model.prompts[0][1].content
    assert (TUNED_INSTRUCTIONS in human) is (policy == "tuned")
    assert ("Only explicit user messages" in human) is (policy == "default")
    assert result.gate_policy == policy


def test_llm_gate_has_no_ask_jev_tool(monkeypatch):
    from jev_demo_triage import agent as agent_mod

    real = agent_mod.create_agent
    captured = {}

    def spy(model, **kwargs):
        captured.update(kwargs)
        return real(model, **kwargs)

    monkeypatch.setattr(agent_mod, "create_agent", spy)
    result = run_scenario(
        SCENARIOS["risky-bait"], "llm-gate",
        model=ScriptedModel(responses=[AIMessage(content="done")]),
        gate_model=_GateModel(["0.1"]),
    )
    assert "ask_jev" not in [t.name for t in captured["tools"]]
    assert "ask_jev" not in captured["system_prompt"]
    assert result.metrics.jev_calls == 0


@pytest.mark.parametrize("mode", ["gate", "both"])
def test_jev_gate_modes_have_no_llm_gate_usage(mode):
    model = ScriptedModel(responses=[AIMessage(content="ok")])
    result = run_scenario(
        SCENARIOS["risky-bait"], mode, model=model, classifier_factory=_gate_factory(0.1),
    )
    m = result.metrics
    assert (m.llm_gate_calls, m.llm_gate_tokens, m.llm_gate_cost_usd) == (0, 0, 0.0)


def test_llm_gate_classifier_factory_overrides_llm_factory():
    model = _rm_then_escalate()
    result = run_scenario(
        SCENARIOS["risky-bait"], "llm-gate", model=model,
        classifier_factory=_gate_factory(0.95), gate_model=_GateModel(["0.0"]),
    )
    assert [a.name for a in result.actions] == ["escalate_to_human"]


def test_llm_gate_missing_key_raises_when_gate_model_needed(monkeypatch):
    from jev_demo_triage.config import MissingKeyError

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingKeyError):
        run_scenario(SCENARIOS["risky-bait"], "llm-gate", model=None)


def test_llm_gate_gate_model_built_lazily_with_key_from_make_model(monkeypatch):
    from jev_demo_triage import agent as agent_mod

    made = []
    monkeypatch.setattr(agent_mod, "make_model", lambda: made.append(1) or _GateModel(["0.9"]))
    run_scenario(SCENARIOS["risky-bait"], "llm-gate", model=_rm_then_escalate())
    assert made == [1]  # only the gate model; the agent model was supplied


def test_baseline_does_not_build_gate_model(monkeypatch):
    from jev_demo_triage import agent as agent_mod

    monkeypatch.setattr(agent_mod, "make_model", lambda: pytest.fail("no model expected"))
    run_scenario(SCENARIOS["risky-bait"], "baseline", model=ScriptedModel(responses=[AIMessage(content="ok")]))


def test_unknown_mode_fails_before_key_work(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError):
        run_scenario(SCENARIOS["risky-bait"], "llm_gate", model=None)


def _usage_ai(msg: AIMessage, inp: int, out: int) -> AIMessage:
    return msg.model_copy(update={"usage_metadata": {
        "input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
    }})


def _llm_gate_separation_run():
    agent_model = ScriptedModel(responses=[
        _usage_ai(ai_tool_call("read_logs", {"service": "billing"}), 1000, 50),
        _usage_ai(ai_tool_call("run_shell", {"command": "rm -rf /var/lib/reports/data"}, "2"), 2000, 60),
        _usage_ai(ai_tool_call("run_shell", {"command": "df -h"}, "3"), 3000, 70),
        _usage_ai(AIMessage(content="Done."), 4000, 80),
    ])
    gate_model = _GateModel(['{"probability": 0.9}', '{"probability": 0.1}'], usage=(111, 22))
    result = run_scenario(SCENARIOS["risky-bait"], "llm-gate", model=agent_model, gate_model=gate_model)
    return result, gate_model


def test_llm_gate_calls_stay_out_of_agent_glm_metrics():
    result, gate_model = _llm_gate_separation_run()
    m = result.metrics
    assert len(gate_model.prompts) == 2  # two gated run_shell calls
    assert m.glm.calls == 4 and m.steps == 4  # agent AIMessages only
    assert m.glm.input_tokens == 1000 + 2000 + 3000 + 4000
    assert m.glm.output_tokens == 50 + 60 + 70 + 80
    assert m.llm_gate_calls == 2
    assert m.llm_gate_tokens == 2 * (111 + 22)
    assert (m.jev_calls, m.jev_tokens, m.jev_cost_usd) == (0, 0, 0.0)


def test_caller_supplied_llm_sink_receives_gate_usage_in_llm_gate_mode():
    from jev_demo_triage.metrics import UsageSink

    sink = UsageSink(input_price=1.0, output_price=2.0)
    result = run_scenario(
        SCENARIOS["risky-bait"], "llm-gate", model=_rm_then_escalate(), llm_sink=sink,
        gate_model=_GateModel(['{"probability": 0.9}'], usage=(111, 22)),
    )
    assert (sink.calls, sink.input_tokens, sink.output_tokens) == (1, 111, 22)
    assert result.metrics.llm_gate_calls == 1
    assert result.metrics.llm_gate_cost_usd == pytest.approx(111 + 44)
