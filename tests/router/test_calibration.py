import json
from datetime import datetime

import pytest

from jev_router_bench.calibration import (
    build_calibration,
    calibration_warnings,
    gold_tier,
    latest_calibration,
    load_calibration,
    save_calibration,
)
from jev_router_bench.runner import TaskRun


def _run(task_id, tier, passed, cost=0.01, error=None):
    return TaskRun(task_id, tier, passed, "ok" if passed else "wrong", 10, 5, cost, 0.5, error)


def _runs(task_id, cheap, mid, strong, repeat=2):
    out = []
    for tier, passes in (("cheap", cheap), ("mid", mid), ("strong", strong)):
        out += [_run(task_id, tier, i < passes) for i in range(repeat)]
    return out


def test_gold_is_cheapest_tier_passing_every_repeat():
    assert gold_tier(_runs("a", 2, 2, 2), 2) == "cheap"
    assert gold_tier(_runs("a", 1, 2, 2), 2) == "mid"
    assert gold_tier(_runs("a", 0, 1, 2), 2) == "strong"
    assert gold_tier(_runs("a", 1, 1, 1), 2) is None


def test_gold_needs_full_repeat_count():
    assert gold_tier([_run("a", "cheap", True)], 2) is None


def test_gold_is_none_when_a_tier_errors_before_a_passing_tier():
    runs = [
        _run("a", "cheap", False, error="TimeoutError: x"),
        _run("a", "cheap", True),
        _run("a", "mid", True),
        _run("a", "mid", True),
        _run("a", "strong", True),
        _run("a", "strong", True),
    ]
    assert gold_tier(runs, 2) is None


def test_gold_ignores_errors_on_tiers_after_gold_is_already_found():
    runs = _runs("a", 2, 2, 2)  # cheap passes cleanly -> gold is cheap
    runs.append(TaskRun("a", "mid", False, "error", 0, 0, 0.0, 1.0, "TimeoutError: x"))
    assert gold_tier(runs, 2) == "cheap"


def test_build_calibration_gold_and_mean_cost():
    runs = _runs("a", 2, 2, 2) + _runs("b", 0, 2, 2)
    runs.append(TaskRun("b", "strong", False, "error", 0, 0, 0.0, 1.0, "TimeoutError: x"))
    cal = build_calibration(runs, 2)
    assert cal.gold == {"a": "cheap", "b": "mid"}
    assert cal.mean_cost["a"] == pytest.approx({"cheap": 0.01, "mid": 0.01, "strong": 0.01})
    assert cal.mean_cost["b"]["strong"] == pytest.approx(0.01)  # error run excluded from the mean


def test_mean_cost_is_none_when_every_run_of_a_tier_errored():
    runs = [
        TaskRun("a", "cheap", False, "error", 0, 0, 0.0, 1.0, "TimeoutError: x"),
        TaskRun("a", "cheap", False, "error", 0, 0, 0.0, 1.0, "TimeoutError: x"),
    ]
    cal = build_calibration(runs, 2)
    assert cal.mean_cost["a"]["cheap"] is None


def test_warnings():
    easy = build_calibration([r for t in "abcde" for r in _runs(t, 2, 2, 2)], 2)
    assert any("too easy" in w for w in calibration_warnings(easy))
    hard = build_calibration(_runs("a", 0, 0, 0) + _runs("b", 0, 2, 2), 2)
    assert any("no tier" in w for w in calibration_warnings(hard))
    fine = build_calibration(_runs("a", 2, 2, 2) + _runs("b", 0, 2, 2), 2)
    assert calibration_warnings(fine) == []


def test_warnings_name_tasks_with_unknown_gold_from_errors():
    runs = [
        _run("a", "cheap", False, error="TimeoutError: x"),
        _run("a", "cheap", True),
        _run("a", "mid", True),
        _run("a", "mid", True),
    ]
    cal = build_calibration(runs, 2)
    assert cal.gold["a"] is None
    warnings = calibration_warnings(cal)
    assert any("1 calibration run(s) errored" in w and "a" in w for w in warnings)


def test_save_and_load_round_trip(tmp_path):
    cal = build_calibration(_runs("a", 0, 2, 2), 2)
    path = save_calibration(cal, tmp_path, now=datetime(2026, 9, 24, 12, 0, 0))
    data = json.loads(path.read_text())
    assert data["pool"]["strong"]["model_id"] == "anthropic/claude-sonnet-5"
    assert data["pool"]["strong"]["input_usd_per_million"] == pytest.approx(2.0)
    assert load_calibration(path) == cal
    assert latest_calibration(tmp_path) == path


def test_load_rejects_unknown_gold_tier(tmp_path):
    path = tmp_path / "calib-x.json"
    path.write_text(json.dumps({"repeat": 1, "gold": {"a": "huge"}, "mean_cost": {}, "runs": []}))
    with pytest.raises(ValueError, match="gold"):
        load_calibration(path)


def test_load_rejects_incomplete_mean_cost(tmp_path):
    path = tmp_path / "calib-y.json"
    path.write_text(json.dumps({
        "repeat": 1,
        "gold": {"a": "cheap"},
        "mean_cost": {"a": {"cheap": 0.01, "mid": 0.02}},  # missing "strong"
        "runs": [],
    }))
    with pytest.raises(ValueError, match="mean_cost"):
        load_calibration(path)


def test_load_rejects_non_numeric_mean_cost(tmp_path):
    path = tmp_path / "calib-z.json"
    path.write_text(json.dumps({
        "repeat": 1,
        "gold": {"a": "cheap"},
        "mean_cost": {"a": {"cheap": "oops", "mid": 0.02, "strong": None}},
        "runs": [],
    }))
    with pytest.raises(ValueError, match="mean_cost"):
        load_calibration(path)


def test_load_accepts_none_mean_cost(tmp_path):
    path = tmp_path / "calib-w.json"
    path.write_text(json.dumps({
        "repeat": 1,
        "gold": {"a": None},
        "mean_cost": {"a": {"cheap": None, "mid": 0.02, "strong": 0.1}},
        "runs": [],
    }))
    assert load_calibration(path).mean_cost["a"]["cheap"] is None
