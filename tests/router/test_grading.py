import pytest

from jev_router_bench.coderun import CodeRunResult
from jev_router_bench.grading import extract_answer, grade, normalize, parse_number
from jev_router_bench.tasks import Task


def _task(grader, expected="", tests="", tolerance=1e-6):
    return Task("t", "c", "cheap", "p", grader, expected, tolerance, tests)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("work\nANSWER: 42", "42"),
        ("answer: Canberra.", "Canberra"),
        ("**ANSWER:** `12:10`", "12:10"),
        ("ANSWER: 1\nmore\nANSWER: 2", "2"),
        ('ANSWER: "knave"', "knave"),
    ],
)
def test_extract_answer_takes_last_answer_line(text, expected):
    assert extract_answer(text) == expected


def test_extract_answer_none_without_line():
    assert extract_answer("The answer is 42") is None
    assert extract_answer("ANSWER:   ") is None


def test_normalize():
    assert normalize("  Apple,kiwi ,  Mango.  ") == "apple, kiwi, mango"
    assert normalize("HELLO   WORLD") == "hello world"


@pytest.mark.parametrize(
    ("value", "number"),
    [("1,234", 1234.0), ("$36", 36.0), ("15%", 15.0), ("3/8", 0.375), ("-2.5", -2.5), ("1°", 1.0)],
)
def test_parse_number(value, number):
    assert parse_number(value) == pytest.approx(number)


def test_parse_number_rejects_text_and_zero_denominator():
    assert parse_number("twelve") is None
    assert parse_number("1/0") is None


def test_exact_grader():
    task = _task("exact", "Canberra")
    assert grade(task, "ANSWER: canberra").passed
    result = grade(task, "ANSWER: Sydney")
    assert (result.passed, result.reason, result.extracted) == (False, "wrong", "Sydney")


def test_numeric_grader_uses_tolerance():
    task = _task("numeric", 2315.25, tolerance=0.01)
    assert grade(task, "ANSWER: $2,315.25").passed
    assert not grade(task, "ANSWER: 2315.3").passed
    assert not grade(task, "ANSWER: about 2315").passed


def test_set_grader_ignores_order_and_case():
    task = _task("set", ("11", "13", "17"))
    assert grade(task, "ANSWER: 17, 11,13").passed
    assert not grade(task, "ANSWER: 11, 13").passed


def test_missing_answer_is_no_answer():
    result = grade(_task("exact", "x"), "I think x")
    assert (result.passed, result.reason, result.extracted) == (False, "no_answer", None)


def test_code_grader_uses_runner():
    seen = {}

    def runner(source, tests):
        seen["args"] = (source, tests)
        return CodeRunResult(True, "ok")

    task = _task("code", tests="assert f() == 1")
    result = grade(task, "```python\ndef f():\n    return 1\n```", code_runner=runner)
    assert result.passed and result.reason == "ok"
    assert seen["args"] == ("def f():\n    return 1\n", "assert f() == 1")


def test_code_grader_without_block_is_no_code():
    result = grade(_task("code", tests="assert True"), "def f(): return 1")
    assert (result.passed, result.reason) == (False, "no_code")


def test_code_grader_real_runner_failure():
    task = _task("code", tests="assert f() == 1")
    result = grade(task, "```python\ndef f():\n    return 2\n```")
    assert (result.passed, result.reason) == (False, "tests_failed")
