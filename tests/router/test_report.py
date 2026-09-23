from jev_router_bench.calibration import build_calibration
from jev_router_bench.metrics import E2ESummary, RouteSummary
from jev_router_bench.report import format_calibration, format_e2e, format_route, format_rows
from jev_router_bench.runner import TaskRun
from jev_router_bench.tasks import Task


def test_format_rows_aligns_columns():
    text = format_rows(("a", "bb"), [("xxx", "y")])
    assert text.splitlines() == ["a    bb", "xxx  y"]


def test_format_calibration_shows_pass_counts_and_gold():
    task = Task("add", "arithmetic", "cheap", "p", "numeric", 4.0)
    runs = [
        TaskRun("add", "cheap", False, "wrong", 1, 1, 0.0, 0.1),
        TaskRun("add", "mid", True, "ok", 1, 1, 0.0, 0.1),
        TaskRun("add", "strong", True, "ok", 1, 1, 0.0, 0.1),
    ]
    text = format_calibration(build_calibration(runs, 1), [task])
    header, row = text.splitlines()
    assert header.split() == ["task", "intended", "cheap", "mid", "strong", "gold"]
    assert row.split() == ["add", "cheap", "0/1", "1/1", "1/1", "mid"]


def test_format_calibration_shows_error_count_in_cell():
    task = Task("add", "arithmetic", "cheap", "p", "numeric", 4.0)
    runs = [
        TaskRun("add", "cheap", False, "error", 0, 0, 0.0, 0.1, "TimeoutError: x"),
        TaskRun("add", "cheap", True, "ok", 1, 1, 0.0, 0.1),
        TaskRun("add", "mid", True, "ok", 1, 1, 0.0, 0.1),
        TaskRun("add", "strong", True, "ok", 1, 1, 0.0, 0.1),
    ]
    text = format_calibration(build_calibration(runs, 2), [task])
    row = text.splitlines()[1]
    assert "1/2 E1" in row


def test_format_calibration_marks_unsolved_as_none():
    task = Task("add", "arithmetic", "cheap", "p", "numeric", 4.0)
    runs = [TaskRun("add", t, False, "wrong", 1, 1, 0.0, 0.1) for t in ("cheap", "mid", "strong")]
    assert format_calibration(build_calibration(runs, 1), [task]).splitlines()[1].split()[-1] == "none"


def test_format_route_includes_references_and_confusion():
    confusion = {g: {c: 0 for c in ("cheap", "mid", "strong")} for g in ("cheap", "mid", "strong")}
    confusion["mid"]["cheap"] = 2
    summary = RouteSummary("jev", 3, 1, 0.5, 0.25, 0.0, confusion, 0.0123, 0.00001, 0.4)
    text = format_route([summary], {"always-strong": 0.1, "oracle": 0.01})
    assert "jev" in text and "50.0%" in text and "$0.012300" in text
    assert "always-strong $0.100000" in text and "oracle $0.010000" in text
    assert "confusion jev" in text


def test_format_route_includes_unpriced_column():
    confusion = {g: {c: 0 for c in ("cheap", "mid", "strong")} for g in ("cheap", "mid", "strong")}
    summary = RouteSummary("jev", 3, 1, 0.5, 0.25, 0.0, confusion, 0.0123, 0.00001, 0.4, unpriced=2)
    text = format_route([summary], {})
    header, row = text.splitlines()[0], text.splitlines()[1]
    assert "unpriced" in header
    assert "2" in row.split()


def test_format_e2e_handles_missing_ratios():
    summary = E2ESummary("jev", 4, 3, 1, 0.75, 0.05, 2.0, {"cheap": 2, "mid": 1, "strong": 1})
    text = format_e2e([summary])
    assert "75.0%" in text and "$0.050000" in text and "c2/m1/s1" in text and "-" in text
