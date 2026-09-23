"""Answer extraction and graders. A missing or unparsable answer fails; nothing is guessed."""

import re
from collections.abc import Callable
from dataclasses import dataclass

from jev_router_bench.coderun import CodeRunResult, extract_code, run_code
from jev_router_bench.tasks import Task

_ANSWER_LINE = re.compile(r"^[\s>*_#`]*ANSWER[\s*_`]*:(.*)$", re.IGNORECASE | re.MULTILINE)
_WRAP = " \t*_`\"'"
_FRACTION = re.compile(r"^(-?\d+)\s*/\s*(\d+)$")

CodeRunner = Callable[[str, str], CodeRunResult]


@dataclass(frozen=True)
class GradeResult:
    passed: bool
    reason: str
    extracted: str | None


def _strip(value: str) -> str:
    return value.strip(_WRAP).rstrip(".").strip(_WRAP)


def extract_answer(text: str) -> str | None:
    found = _ANSWER_LINE.findall(text)
    if not found:
        return None
    return _strip(found[-1]) or None


def normalize(value: str) -> str:
    v = _strip(value)
    v = re.sub(r"\s*,\s*", ", ", v)
    v = re.sub(r"\s+", " ", v)
    return v.casefold()


def parse_number(value: str) -> float | None:
    v = _strip(value)
    for junk in (",", "$", "%", "°"):
        v = v.replace(junk, "")
    v = v.strip()
    fraction = _FRACTION.match(v)
    if fraction:
        denominator = int(fraction.group(2))
        return None if denominator == 0 else int(fraction.group(1)) / denominator
    try:
        return float(v)
    except ValueError:
        return None


def _exact(answer: str, task: Task) -> bool:
    return normalize(answer) == normalize(str(task.expected))


def _numeric(answer: str, task: Task) -> bool:
    number = parse_number(answer)
    return number is not None and abs(number - float(task.expected)) <= task.tolerance


def _set(answer: str, task: Task) -> bool:
    got = {normalize(item) for item in answer.split(",") if item.strip()}
    return got == {normalize(item) for item in task.expected}


_CHECKS: dict[str, Callable[[str, Task], bool]] = {
    "exact": _exact,
    "numeric": _numeric,
    "set": _set,
}


def grade(task: Task, text: str, code_runner: CodeRunner = run_code) -> GradeResult:
    if task.grader == "code":
        code = extract_code(text)
        if code is None:
            return GradeResult(False, "no_code", None)
        result = code_runner(code, task.tests)
        return GradeResult(result.passed, result.reason, code)
    answer = extract_answer(text)
    if answer is None:
        return GradeResult(False, "no_answer", None)
    passed = _CHECKS[task.grader](answer, task)
    return GradeResult(passed, "ok" if passed else "wrong", answer)
