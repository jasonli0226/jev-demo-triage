import json
import types

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_typesafe.types import ClassificationResponse, Noul, NoulCriteria

from jev_demo_triage import llm_gate
from jev_demo_triage.llm_gate import (
    LlmClassifier,
    LlmGateError,
    make_llm_factory,
    parse_probability,
    serialize_state,
)
from jev_demo_triage.metrics import UsageSink


@pytest.mark.parametrize("text,expected", [
    ("0.87", 0.87),
    ('{"probability": 0.87}', 0.87),
    ('```json\n{"probability": 0.3}\n```', 0.3),
    ('Sure. Here it is: {"probability": 0.42} hope that helps', 0.42),
    ("87%", 0.87),
    ("100%", 1.0),
    ('{"probability": 0.9} probability 0.1', 0.9),
    ("1", 1.0),
    ("0", 0.0),
    (".5", 0.5),
    ("1.0", 1.0),
    ('"0.5"', 0.5),
    ('{"probability": "0.9"}', 0.9),
    ("Step 1 of 3 done. " + '{"probability": 0.95}', 0.95),
    ('0 authorizations found. {"probability": 0.85}', 0.85),
    ('state was {"command": "rm -rf /"} so {"probability": 0.9}', 0.9),
    ('{"probability": 0.9} then again {"probability": 0.9}', 0.9),
    ('{"probability": 0.9, "echo": {"probability": "0.9"}}', 0.9),
    ('{"probability": 1}', 1.0),
])
def test_parse_probability_accepts(text, expected):
    assert parse_probability(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", [
    "2", "-0.1", "abc", "", "   ", '{"probability": "high"}', '{"probability": 1.5}',
    '{"other": 1}', "150%",
    "I think 0.6 is right", "0 authorizations found, so 0.85", "Step 1 of 3: risk is 0.95",
    "0.5.1", "probability: 0.5.1", "nan", "inf", "probability: 2", '{"probability": 2}',
    '{"probability": -0.5}', '{"command": "rm -rf /"}', '{"probability": "0.9.1"}',
    '{"probability": true}',
    "probability 1e-1", "probability: 5e-2", "probability = 1E-3", "probability of 0.85e0",
    '{"probability": 0.1} on reflection {"probability": 0.6}',
    '{"probability": 0.9} quoted {"probability": 0.0}',
    '{"probability":0.9,"quote":{"probability":0.0}}',
    '{"probability": 0.9} {"probability": 5}',
    "probability 1.e-1", "probability 1.E-1", "probability 0.5 probability 1e-1",
    "probability 0.5 then probability 0.7", "probability 0.2 then revised probability: 0.7", "probability 0.5 probability 2",
    '{"probability": 0.9, "probability": 0.1}', '{"probability": 0.9, "probability": 0.9}',
    '{"a": {"probability": 0.9, "probability": 0.1}}',
    '{"probability": 0.9} quoted {"command": "a", "command": "b"}',
    "probability: 0.85", "The probability of 0.85 seems right.", "Probability = .85.",
    "probability: 0.85.", "probability = .85", "Probability = 0.7.", "probability 0.85 exactly",
    "probability 0.5 and later probability 0.5", "probability of the call is high, probability 0.3",
    "probability 87%", "The probability is 0.85", "0.5 to 0.6",
    "probability 1/2", "probability 0/1", "probability 1 in 10", "probability 1 out of 2",
    "probability 1:2", "probability 0,5", "probability 1,000", "probability 0.5 to 0.6",
    "probability 0.9 or 0.1", "probability 0.9x", "probability 1_0", "probability 1 .5",
    "probability 0.5 nan",
])
def test_parse_probability_rejects(text):
    with pytest.raises(LlmGateError):
        parse_probability(text)


def test_serialize_state_messages_tool_call_and_description():
    state = {
        "messages": [
            HumanMessage("alert: disk full"),
            AIMessage(content="", tool_calls=[{"name": "run_shell", "args": {"command": "ls"}, "id": "c1"}]),
            ToolMessage(content="a b", tool_call_id="c1"),
            AIMessage(content=[{"type": "text", "text": "hello"}]),
        ],
        "tool_call": {"id": "c2", "name": "run_shell", "args": {"command": "rm -rf /x"}},
        "tool_description": "Run a shell command.",
    }
    data = json.loads(serialize_state(state))
    msgs = data["messages"]
    assert msgs[0] == {"role": "human", "content": "alert: disk full"}
    assert msgs[1]["role"] == "ai"
    assert msgs[1]["tool_calls"][0]["name"] == "run_shell"
    assert msgs[1]["tool_calls"][0]["args"] == {"command": "ls"}
    assert msgs[2] == {"role": "tool", "content": "a b", "tool_call_id": "c1"}
    assert msgs[3]["content"] == [{"type": "text", "text": "hello"}]
    assert data["tool_call"]["args"] == {"command": "rm -rf /x"}
    assert data["tool_description"] == "Run a shell command."


def test_serialize_state_falls_back_to_str_for_unknown_values():
    class Odd:
        def __str__(self):
            return "odd!"

    data = json.loads(serialize_state({"tool_call": {"args": {"x": Odd()}}, "extra": Odd()}))
    assert data["tool_call"]["args"]["x"] == "odd!"
    assert data["extra"] == "odd!"


def test_serialize_state_non_dict_state():
    assert json.loads(serialize_state("disk full")) == "disk full"


class RecordingModel(FakeMessagesListChatModel):
    seen: list = []

    def _generate(self, messages, *a, **k):
        self.seen.append(messages)
        return super()._generate(messages, *a, **k)


def _questions(criteria=NoulCriteria(true="bad", false="fine"), instructions="Is it risky?"):
    return {"is_risky": Noul(instructions=instructions, criteria=criteria)}


def _model(content, usage=(120, 8)):
    msg = AIMessage(content=content, usage_metadata={
        "input_tokens": usage[0], "output_tokens": usage[1], "total_tokens": sum(usage),
    })
    return RecordingModel(responses=[msg], seen=[])


STATE = {"messages": [HumanMessage("go")], "tool_call": {"id": "1", "name": "run_shell", "args": {"command": "ls"}}}


def test_classifier_returns_parsed_probability_and_records_usage(monkeypatch):
    ticks = iter([10.0, 10.75])
    # Patch only llm_gate's own `time` name so other users of the clock do not consume ticks.
    monkeypatch.setattr(llm_gate, "time", types.SimpleNamespace(perf_counter=lambda: next(ticks)))
    sink = UsageSink()
    clf = LlmClassifier(_questions(), _model('{"probability": 0.9}'), sink=sink)
    response = clf.invoke(STATE)
    assert isinstance(response, ClassificationResponse)
    assert response.nouls["is_risky"].noul == pytest.approx(0.9)
    assert response.usage.input_tokens == 120 and response.usage.output_tokens == 8
    assert (sink.calls, sink.input_tokens, sink.output_tokens) == (1, 120, 8)
    assert sink.seconds == pytest.approx(0.75)


def test_classifier_exposes_questions():
    questions = _questions()
    assert LlmClassifier(questions, _model("0.1")).questions is questions


def test_missing_usage_counts_as_zero():
    sink = UsageSink()
    model = RecordingModel(responses=[AIMessage(content="0.2")], seen=[])
    LlmClassifier(_questions(), model, sink=sink).invoke(STATE)
    assert (sink.calls, sink.input_tokens, sink.output_tokens) == (1, 0, 0)


def test_list_of_text_blocks_content_is_parsed():
    model = RecordingModel(
        responses=[AIMessage(content=[{"type": "text", "text": '{"probability": '}, {"type": "text", "text": "0.7}"}])],
        seen=[],
    )
    response = LlmClassifier(_questions(), model).invoke(STATE)
    assert response.nouls["is_risky"].noul == pytest.approx(0.7)


def test_reasoning_blocks_are_ignored():
    model = RecordingModel(responses=[AIMessage(content=[
        {"type": "reasoning", "text": 'maybe {"probability": 0.1} hmm'},
        {"type": "thinking", "thinking": "x", "text": "0.2"},
        {"type": "text", "text": '{"probability": 0.9}'},
    ])], seen=[])
    assert LlmClassifier(_questions(), model).invoke(STATE).nouls["is_risky"].noul == pytest.approx(0.9)


def test_only_reasoning_blocks_fails_closed():
    model = RecordingModel(responses=[AIMessage(content=[
        {"type": "reasoning", "text": '{"probability": 0.1}'},
    ])], seen=[])
    with pytest.raises(LlmGateError):
        LlmClassifier(_questions(), model).invoke(STATE)


def test_str_content_still_works():
    assert LlmClassifier(_questions(), _model("0.25")).invoke(STATE).nouls["is_risky"].noul == 0.25


def test_model_called_with_empty_callbacks_config():
    seen = {}

    class Spy(FakeMessagesListChatModel):
        def invoke(self, messages, config=None, **kw):
            seen["config"] = config
            return super().invoke(messages, config=config, **kw)

    LlmClassifier(_questions(), Spy(responses=[AIMessage(content="0.3")])).invoke(STATE)
    assert seen["config"] == {"callbacks": []}


def test_model_error_records_seconds_and_reraises(monkeypatch):
    ticks = iter([1.0, 3.5])
    # Patch only llm_gate's own `time` name so other users of the clock do not consume ticks.
    monkeypatch.setattr(llm_gate, "time", types.SimpleNamespace(perf_counter=lambda: next(ticks)))

    class Boom(FakeMessagesListChatModel):
        def _generate(self, *a, **k):
            raise RuntimeError("glm 500")

    sink = UsageSink()
    with pytest.raises(RuntimeError, match="glm 500"):
        LlmClassifier(_questions(), Boom(responses=[AIMessage(content="x")]), sink=sink).invoke(STATE)
    assert sink.calls == 0
    assert sink.seconds == pytest.approx(2.5)


def test_unparseable_answer_fails_closed():
    sink = UsageSink()
    with pytest.raises(LlmGateError):
        LlmClassifier(_questions(), _model("I cannot say"), sink=sink).invoke(STATE)


def test_prompt_contains_instructions_criteria_and_state():
    model = _model("0.1")
    LlmClassifier(_questions(), model).invoke(STATE)
    system, human = model.seen[0]
    assert isinstance(system, SystemMessage) and isinstance(human, HumanMessage)
    assert "probability" in system.content and "data" in system.content
    assert "Is it risky?" in human.content
    assert "if yes: bad" in human.content and "if no: fine" in human.content
    assert "run_shell" in human.content and '"ls"' in human.content


def test_criteria_omitted_when_none():
    model = _model("0.1")
    LlmClassifier(_questions(criteria=None), model).invoke(STATE)
    assert "if yes" not in model.seen[0][1].content


def test_structured_instructions_are_json_dumped():
    model = _model("0.1")
    LlmClassifier(_questions(instructions={"q": "risky?"}), model).invoke(STATE)
    assert '{"q": "risky?"}' in model.seen[0][1].content


@pytest.mark.parametrize("questions", [{}, {"is_risky": "not a noul"}, {"other": Noul(instructions="x")}])
def test_missing_or_non_noul_question_rejected(questions):
    with pytest.raises(ValueError):
        LlmClassifier(questions, _model("0.1"))


def test_custom_question_id():
    clf = LlmClassifier({"q": Noul(instructions="x")}, _model("0.4"), question_id="q")
    assert clf.invoke(STATE).nouls["q"].noul == pytest.approx(0.4)


def test_make_llm_factory_builds_classifier_lazily_per_classifier():
    built = []

    def model_factory():
        built.append(1)
        return _model("0.3")

    factory = make_llm_factory(model_factory)
    assert built == []
    sink = UsageSink()
    clf = factory(_questions(), sink)
    assert built == [1]
    assert isinstance(clf, LlmClassifier) and clf.questions is not None
    assert clf.invoke(STATE).nouls["is_risky"].noul == pytest.approx(0.3)
    assert sink.calls == 1


def _independent_json_values(text):
    decoder = json.JSONDecoder()
    values = []
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(obj, dict) and "probability" in obj:
            values.append(obj["probability"])
    return values


def _bare_value(text):
    import re

    m = re.fullmatch(r"(\d+\.?\d*|\.\d+)(%?)", text.strip().strip("`\"' \n\t").strip())
    if m is None:
        return None
    v = float(m.group(1))
    return v / 100 if m.group(2) else v


def test_parse_probability_property_never_returns_unjustified_value():
    import random

    rng = random.Random(1234)
    numbers = ["0", "1", "0.5", ".5", "0.9", "1.0", "2", "-0.1", "1e-1", "1.e-1", "0,5", "1/2",
               "0.5.1", "87%", "150%", "nan", "1_0", "0.85."]
    values = ['0.9', '0.1', '"0.9"', '"high"', '2', 'true', 'null', '0', '1', '-1', '1e-1']
    fragments = [
        lambda: '{"probability": %s}' % rng.choice(values),
        lambda: '{"probability": %s, "probability": %s}' % (rng.choice(values), rng.choice(values)),
        lambda: '{"probability": %s, "echo": {"probability": %s}}' % (rng.choice(values), rng.choice(values)),
        lambda: '{"command": "rm -rf /", "n": %s}' % rng.choice(numbers),
        lambda: "```json", lambda: "```", lambda: '"',
        lambda: "probability " + rng.choice(numbers), lambda: "probability: " + rng.choice(numbers),
        lambda: rng.choice(numbers), lambda: "so", lambda: "\n", lambda: " ",
    ]
    returned = 0
    for _ in range(4000):
        text = " ".join(rng.choice(fragments)() for _ in range(rng.randint(1, 4)))
        try:
            v = parse_probability(text)
        except LlmGateError:
            continue
        returned += 1
        assert 0.0 <= v <= 1.0
        json_values = _independent_json_values(text)
        if json_values:
            assert all(
                (float(x) if isinstance(x, str) else x) == v for x in json_values
            ), text
        else:
            assert _bare_value(text) == pytest.approx(v), text
    assert returned > 100  # the generator does produce valid answers


def test_ainvoke_is_explicitly_unsupported():
    import asyncio

    clf = LlmClassifier(_questions(), _model("0.1"))
    with pytest.raises(NotImplementedError, match="synchronous runs only"):
        asyncio.run(clf.ainvoke(STATE))
