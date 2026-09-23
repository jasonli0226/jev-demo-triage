import pytest

from jev_router_bench.calibration import Calibration
from jev_router_bench.metrics import reference_costs, summarize_e2e, summarize_route
from jev_router_bench.routers import RouteDecision
from jev_router_bench.runner import E2ERecord, RouteRecord, TaskRun

COSTS = {"cheap": 0.001, "mid": 0.01, "strong": 0.1}
CAL = Calibration(
    repeat=1,
    gold={"a": "cheap", "b": "mid", "c": "strong", "x": None},
    mean_cost={t: dict(COSTS) for t in "abcx"},
    runs=(),
)


def _rr(router, task, tier, cost=0.0, seconds=0.0):
    return RouteRecord(router, task, 0, RouteDecision(tier, cost_usd=cost, seconds=seconds))


def test_route_summary_accuracy_under_over_and_errors():
    records = [
        _rr("jev", "a", "cheap", cost=0.0001, seconds=0.2),  # correct
        _rr("jev", "b", "cheap", cost=0.0001, seconds=0.2),  # under
        _rr("jev", "c", "strong", cost=0.0001, seconds=0.2),  # correct
        RouteRecord("jev", "b", 1, None, "ValueError: x"),  # error
        _rr("jev", "x", "strong"),  # gold None: ignored
    ]
    (s,) = summarize_route(records, CAL)
    assert (s.router, s.n, s.errors) == ("jev", 4, 1)
    assert s.accuracy == pytest.approx(2 / 4)
    assert s.under_rate == pytest.approx(1 / 3)
    assert s.over_rate == 0.0
    assert s.confusion["mid"]["cheap"] == 1
    assert s.confusion["strong"]["strong"] == 1
    assert s.expected_cost == pytest.approx((0.001 + 0.001 + 0.1) / 3 + 0.0001)
    assert s.router_cost_mean == pytest.approx(0.0001)
    assert s.router_seconds_mean == pytest.approx(0.2)


def test_route_summary_keeps_router_order_and_over_route():
    records = [_rr("llm", "a", "strong"), _rr("jev", "a", "cheap")]
    llm, jev = summarize_route(records, CAL)
    assert (llm.router, llm.over_rate, jev.router, jev.accuracy) == ("llm", 1.0, "jev", 1.0)


def test_reference_costs():
    refs = reference_costs(CAL, ["a", "b", "c", "x"])
    assert refs["always-strong"] == pytest.approx(0.1)
    assert refs["oracle"] == pytest.approx((0.001 + 0.01 + 0.1) / 3)


UNPRICED_CAL = Calibration(
    repeat=1,
    gold={"a": "cheap", "b": "cheap"},
    mean_cost={
        "a": {"cheap": None, "mid": 0.01, "strong": None},
        "b": {"cheap": 0.001, "mid": 0.01, "strong": 0.1},
    },
    runs=(),
)


def test_route_summary_skips_unpriced_decisions_for_expected_cost():
    records = [
        _rr("jev", "a", "cheap", cost=0.0001),
        _rr("jev", "b", "cheap", cost=0.0001),
    ]
    (s,) = summarize_route(records, UNPRICED_CAL)
    assert s.unpriced == 1
    assert s.expected_cost == pytest.approx(0.001 + 0.0001)


def test_reference_costs_skips_unpriced_tasks():
    refs = reference_costs(UNPRICED_CAL, ["a", "b"])
    assert refs["always-strong"] == pytest.approx(0.1)  # a's strong cost is unknown
    assert refs["oracle"] == pytest.approx(0.001)  # a's cheap (gold) cost is unknown


def _run(tier, passed, cost, seconds=1.0, error=None):
    return TaskRun("a", tier, passed, "ok", 1, 1, cost, seconds, error)


def _er(router, tier, passed, cost, router_cost=0.0):
    return E2ERecord(router, "a", 0, RouteDecision(tier, cost_usd=router_cost, seconds=0.5), _run(tier, passed, cost))


def test_e2e_summary_savings_and_quality():
    records = [
        _er("always-strong", "strong", True, 0.1),
        _er("always-strong", "strong", True, 0.1),
        _er("jev", "cheap", True, 0.001, router_cost=0.0001),
        _er("jev", "strong", False, 0.1, router_cost=0.0001),
        E2ERecord("jev", "a", 2, None, None, "ValueError: x"),
        E2ERecord("jev", "a", 3, RouteDecision("mid"), _run("mid", False, 0.0, error="TimeoutError: t")),
    ]
    strong, jev = summarize_e2e(records)
    assert (strong.passed, strong.pass_rate, strong.total_cost) == (2, 1.0, pytest.approx(0.2))
    assert strong.savings_vs_strong == pytest.approx(0.0)
    assert (jev.n, jev.passed, jev.errors) == (4, 1, 2)
    assert jev.pass_rate == pytest.approx(0.25)
    assert jev.total_cost == pytest.approx(0.101 + 0.0002)
    assert jev.savings_vs_strong == pytest.approx(1 - 0.1012 / 0.2)
    assert jev.quality_vs_strong == pytest.approx(0.25)
    assert jev.tier_mix == {"cheap": 1, "mid": 1, "strong": 1}
    assert jev.mean_seconds == pytest.approx((1.5 + 1.5 + 0.0 + 1.0) / 4)


def test_e2e_without_strong_baseline_has_no_ratios():
    (jev,) = summarize_e2e([_er("jev", "cheap", True, 0.001)])
    assert jev.savings_vs_strong is None and jev.quality_vs_strong is None
