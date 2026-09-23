from collections import Counter

import pytest

from jev_router_bench.grading import grade
from jev_router_bench.tasks import TASKS, Task, select_tasks

REFERENCE_CODE = {
    "code-rle": '''
def run_length_encode(s):
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{s[i]}{j - i}")
        i = j
    return "".join(out)
''',
    "code-palindrome": '''
def is_palindrome(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]
''',
    "code-merge-intervals": '''
def merge_intervals(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out
''',
    "code-eval-expr": '''
def evaluate(expr):
    s = expr.replace(" ", "")
    pos = 0

    def peek():
        return s[pos] if pos < len(s) else ""

    def parse_expr():
        nonlocal pos
        value = parse_term()
        while peek() in ("+", "-") and peek():
            op = s[pos]
            pos += 1
            rhs = parse_term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def parse_term():
        nonlocal pos
        value = parse_factor()
        while peek() in ("*", "/") and peek():
            op = s[pos]
            pos += 1
            rhs = parse_factor()
            if op == "*":
                value *= rhs
            else:
                q = abs(value) // abs(rhs)
                value = q if (value >= 0) == (rhs > 0) else -q
        return value

    def parse_factor():
        nonlocal pos
        if peek() == "-":
            pos += 1
            return -parse_factor()
        if peek() == "(":
            pos += 1
            value = parse_expr()
            pos += 1
            return value
        start = pos
        while peek().isdigit():
            pos += 1
        return int(s[start:pos])

    return parse_expr()
''',
    "code-version-compare": '''
def compare_versions(a, b):
    xs = [int(p) for p in a.split(".")]
    ys = [int(p) for p in b.split(".")]
    n = max(len(xs), len(ys))
    xs += [0] * (n - len(xs))
    ys += [0] * (n - len(ys))
    return (xs > ys) - (xs < ys)
''',
    "code-spiral": '''
def spiral_order(matrix):
    out = []
    rows = [list(r) for r in matrix]
    while rows:
        out += rows.pop(0)
        if rows and rows[0]:
            for r in rows:
                out.append(r.pop())
        if rows:
            out += rows.pop()[::-1]
        if rows and rows[0]:
            for r in rows[::-1]:
                out.append(r.pop(0))
        rows = [r for r in rows if r]
    return out
''',
}

WRONG_CODE = {
    "code-rle": "def run_length_encode(s):\n    return s\n",
    "code-palindrome": "def is_palindrome(s):\n    return s == s[::-1]\n",
    "code-merge-intervals": "def merge_intervals(intervals):\n    return intervals\n",
    "code-eval-expr": "def evaluate(expr):\n    return eval(expr)\n",
    "code-version-compare": "def compare_versions(a, b):\n    return (a > b) - (a < b)\n",
    "code-spiral": "def spiral_order(matrix):\n    return [x for r in matrix for x in r]\n",
}


def _reference_answer(task: Task) -> str:
    if task.grader == "code":
        return f"```python\n{REFERENCE_CODE[task.id]}```"
    if task.grader == "numeric":
        return f"ANSWER: {task.expected:g}"
    if task.grader == "set":
        return f"ANSWER: {', '.join(task.expected)}"
    return f"ANSWER: {task.expected}"


def test_thirty_tasks_ten_per_intended_tier():
    assert len(TASKS) == 30
    assert Counter(t.intended_tier for t in TASKS) == {"cheap": 10, "mid": 10, "strong": 10}


def test_ids_are_unique():
    assert len({t.id for t in TASKS}) == len(TASKS)


def test_code_tasks_have_tests_and_references():
    code = {t.id for t in TASKS if t.grader == "code"}
    assert code == set(REFERENCE_CODE) == set(WRONG_CODE)
    assert all(t.tests for t in TASKS if t.grader == "code")


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_reference_answer_passes(task):
    assert grade(task, _reference_answer(task)).passed


@pytest.mark.parametrize("task_id", sorted(WRONG_CODE))
def test_wrong_code_fails(task_id):
    task = select_tasks([task_id])[0]
    assert not grade(task, f"```python\n{WRONG_CODE[task_id]}```").passed


def test_select_tasks():
    assert select_tasks(None) == TASKS
    assert [t.id for t in select_tasks(["add-simple", "add-simple"])] == ["add-simple"]
    with pytest.raises(KeyError, match="Unknown task"):
        select_tasks(["nope"])
