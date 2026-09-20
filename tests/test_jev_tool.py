import pytest
from langchain_typesafe.types import ClassificationResponse

from jev_demo_triage.jev_tool import make_ask_jev, summarize_response
from jev_demo_triage.metrics import UsageSink

RESPONSE = ClassificationResponse.model_validate({
    "model": "m",
    "answers": {
        "urgent": {"type": "noul", "noul": 0.96},
        "risk": {"type": "choice", "choice": "destructive",
                 "probabilities": {"safe": 0, "risky": 0, "destructive": 1}, "confidence": 1},
        "sev": {"type": "score", "score": 1.99, "legend": {"0": "low", "1": "medium", "2": "high"},
                "probabilities": {"0": 0, "1": 0, "2": 1}, "confidence": 0.99},
    },
    "usage": {"input_tokens": 100, "output_tokens": 10},
})


class FakeClassifier:
    def __init__(self, questions, sink, response=RESPONSE, error=None):
        self.questions, self.sink, self.response, self.error = questions, sink, response, error
        self.states = []

    def invoke(self, state):
        self.states.append(state)
        if self.error:
            raise self.error
        self.sink.add(100, 10)
        return self.response


def _tool(error=None):
    sink = UsageSink()
    made = []

    def factory(questions, sink):
        clf = FakeClassifier(questions, sink, error=error)
        made.append(clf)
        return clf

    return make_ask_jev(sink, classifier_factory=factory), sink, made


def test_summarize_response_is_compact():
    text = summarize_response(RESPONSE)
    assert "urgent: yes 0.96" in text
    assert "risk: destructive (confidence 1.00)" in text
    assert "sev: 1.99 on [low, medium, high]" in text


def test_tool_validates_and_calls_classifier():
    tool, sink, made = _tool()
    out = tool.invoke({
        "state": "deploy failed",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Urgent?"},
            "risk": {"type": "choice", "instructions": "Risk?", "criteria": {"safe": "a", "risky": "b", "destructive": "c"}},
            "sev": {"type": "score", "instructions": "Severity", "criteria": ["low", "medium", "high"]},
        },
    })
    assert "urgent: yes 0.96" in out
    assert made[0].states == ["deploy failed"]
    assert set(made[0].questions) == {"urgent", "risk", "sev"}
    assert sink.calls == 1


def test_invalid_question_returns_message_and_does_not_call_jev():
    tool, sink, made = _tool()
    out = tool.invoke({"state": "x", "questions": {"q": {"type": "bogus"}}})
    assert "Invalid question 'q'" in out
    assert made == [] and sink.calls == 0


def test_empty_questions_rejected():
    tool, _, made = _tool()
    out = tool.invoke({"state": "x", "questions": {}})
    assert "at least one question" in out.lower()
    assert made == []


def test_classifier_failure_returns_error_string_not_exception():
    tool, _, _ = _tool(error=RuntimeError("boom"))
    out = tool.invoke({"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}})
    assert out.startswith("ask_jev failed")
    assert "boom" in out


def test_description_documents_question_shapes():
    tool, _, _ = _tool()
    for word in ("noul", "choice", "score", "criteria"):
        assert word in tool.description
