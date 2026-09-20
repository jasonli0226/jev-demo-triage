"""Human-readable step-by-step trace of one run (`--trace`)."""

import sys
from collections.abc import Callable
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from jev_demo_triage.gate import QUESTION_ID
from jev_demo_triage.jev_tool import summarize_response

# Mirrors AutoModeMiddleware's private _PROBABILITY_THRESHOLD: p >= this blocks the call.
GATE_BLOCK_THRESHOLD = 0.5


class Tracer:
    def __init__(self, write: Callable[[str], None] | None = None, max_len: int = 120) -> None:
        self._write = write or self._write_stderr
        self._max_len = max_len

    @staticmethod
    def _write_stderr(text: str) -> None:
        print(text, file=sys.stderr)

    def header(self, scenario: str, mode: str, gate_policy: str | None = None) -> None:
        suffix = f" (policy: {gate_policy})" if gate_policy else ""
        self._write(f"== {scenario} / {mode}{suffix} ==")

    def line(self, tag: str, text: str, suffix: str = "") -> None:
        flat = " ".join(str(text).split("\n"))
        if len(flat) > self._max_len:
            flat = flat[: self._max_len] + "..."
        self._write(f"{tag:<4} {flat}{suffix}")


def _format_call(name: str, args: dict) -> str:
    return f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"


def _text(output: Any) -> str:
    content = getattr(output, "content", output)
    return content if isinstance(content, str) else str(content)


class TraceHandler(BaseCallbackHandler):
    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer
        self._tool_names: dict[UUID, str] = {}

    def on_llm_end(self, response, **kwargs: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                message = getattr(gen, "message", None)
                calls = getattr(message, "tool_calls", None) or []
                for call in calls:
                    self._tracer.line("GLM", f"-> {_format_call(call['name'], call['args'])}")
                content = getattr(message, "content", "")
                if not calls and content:
                    self._tracer.line("GLM", f"-> final: {_text(message)}")

    def on_tool_start(
        self, serialized: dict[str, Any] | None, input_str: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name")
        if name:
            self._tool_names[run_id] = str(name)

    def _tool_label(self, run_id: UUID) -> str:
        name = self._tool_names.pop(run_id, None)
        return f"{name} " if name else ""

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._tracer.line("TOOL", f"{self._tool_label(run_id)}<- {_text(output)}")

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._tracer.line("TOOL", f"{self._tool_label(run_id)}<- error: {error}")


class _TracedClassifier:
    def __init__(self, inner: Any, questions: dict, tracer: Tracer) -> None:
        self._inner = inner
        self._questions = questions
        self._tracer = tracer

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def invoke(self, state, *args: Any, **kwargs: Any):
        for name, q in self._questions.items():
            self._tracer.line("JEV", f'-> {name}: {q.type} "{q.instructions}"')
        try:
            response = self._inner.invoke(state, *args, **kwargs)
        except Exception as exc:
            self._tracer.line("JEV", f"!! {exc}")
            raise
        answer = response.nouls.get(QUESTION_ID)
        if isinstance(state, dict) and "tool_call" in state and answer is not None:
            call = state["tool_call"]
            p = answer.noul
            verdict = "BLOCKED" if p >= GATE_BLOCK_THRESHOLD else "ALLOWED"
            self._tracer.line(
                "GATE", _format_call(call["name"], call["args"]), suffix=f" risk {p:.2f} -> {verdict}"
            )
        else:
            self._tracer.line("JEV", f"<- {' | '.join(summarize_response(response).splitlines())}")
        return response


def traced_factory(factory: Callable[..., Any], tracer: Tracer) -> Callable[..., Any]:
    def build(questions, sink):
        return _TracedClassifier(factory(questions, sink), questions, tracer)

    return build
