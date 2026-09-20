"""`ask_jev`: lets the agent hand classification sub-questions to Jev."""

from collections.abc import Callable

from langchain_core.tools import BaseTool, StructuredTool
from langchain_typesafe import Question
from langchain_typesafe.types import ClassificationResponse
from pydantic import TypeAdapter, ValidationError

from jev_demo_triage.jev import make_classifier
from jev_demo_triage.metrics import UsageSink

ASK_JEV_DESCRIPTION = (
    "Ask Jev, a fast classifier, typed questions about a state you describe. Use it for "
    "classification or yes/no judgments (severity, urgency, risk, is the evidence enough), "
    "NOT for open-ended reasoning. Arguments: `state` is plain text with the facts Jev needs; "
    "`questions` maps a name to one question object. Question shapes: "
    '{"type": "noul", "instructions": "<yes/no question>"} returns the probability of yes; '
    '{"type": "choice", "instructions": "<question>", "criteria": {"<option>": "<meaning>", ...}} '
    "picks one option; "
    '{"type": "score", "instructions": "<question>", "criteria": ["<lowest level>", ..., "<highest level>"]} '
    "returns a position on the ordered levels. You can send several questions in one call."
)

_QUESTION = TypeAdapter(Question)


def summarize_response(response: ClassificationResponse) -> str:
    lines: list[str] = []
    for name, ans in response.nouls.items():
        lines.append(f"{name}: {'yes' if ans.noul >= 0.5 else 'no'} {ans.noul:.2f}")
    for name, ans in response.choices.items():
        lines.append(f"{name}: {ans.choice} (confidence {ans.confidence:.2f})")
    for name, ans in response.scores.items():
        legend = ", ".join(ans.legend[k] for k in sorted(ans.legend, key=int))
        lines.append(f"{name}: {ans.score:.2f} on [{legend}] (confidence {ans.confidence:.2f})")
    return "\n".join(lines)


def make_ask_jev(
    sink: UsageSink,
    classifier_factory: Callable[..., object] = make_classifier,
) -> BaseTool:
    def ask_jev(state: str, questions: dict[str, dict]) -> str:
        if not questions:
            return "ask_jev needs at least one question."
        parsed: dict[str, Question] = {}
        for name, raw in questions.items():
            try:
                parsed[name] = _QUESTION.validate_python(raw)
            except ValidationError as exc:
                first = exc.errors()[0]
                return f"Invalid question '{name}': {first['msg']} at {'.'.join(map(str, first['loc']))}"
        try:
            response = classifier_factory(parsed, sink).invoke(state)
        except Exception as exc:  # noqa: BLE001 - tool must never crash the agent loop
            return f"ask_jev failed: {exc}"
        return summarize_response(response)

    return StructuredTool.from_function(
        func=ask_jev, name="ask_jev", description=ASK_JEV_DESCRIPTION
    )
