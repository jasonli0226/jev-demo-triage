"""Route and end-to-end summaries per router."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace

from jev_router_bench.calibration import Calibration
from jev_router_bench.pool import TIERS, tier_rank
from jev_router_bench.runner import E2ERecord, RouteRecord

STRONG_BASELINE = "always-strong"


@dataclass(frozen=True)
class RouteSummary:
    router: str
    n: int
    errors: int
    accuracy: float
    under_rate: float
    over_rate: float
    confusion: dict[str, dict[str, int]]  # gold -> chosen -> count
    expected_cost: float
    router_cost_mean: float
    router_seconds_mean: float
    unpriced: int = 0  # decided rows skipped from expected_cost: calibrated cost unknown


@dataclass(frozen=True)
class E2ESummary:
    router: str
    n: int
    passed: int
    errors: int
    pass_rate: float
    total_cost: float
    mean_seconds: float
    tier_mix: dict[str, int]
    savings_vs_strong: float | None = None
    quality_vs_strong: float | None = None


def _ratio(part: float, whole: float) -> float:
    return part / whole if whole else 0.0


def _mean(values: Sequence[float]) -> float:
    return _ratio(sum(values), len(values))


def _names(records: Sequence[RouteRecord | E2ERecord]) -> list[str]:
    return list(dict.fromkeys(r.router for r in records))


def _priced_costs(decided: Sequence[RouteRecord], cal: Calibration) -> tuple[list[float], int]:
    """Expected cost per decided row (task cost at the chosen tier + router cost).

    A row whose calibrated tier cost is unknown (None, because every calibration run of
    that task/tier errored) is skipped and counted as unpriced rather than treated as free.
    """
    priced: list[float] = []
    unpriced = 0
    for r in decided:
        tier_cost = cal.mean_cost[r.task_id][r.decision.tier]
        if tier_cost is None:
            unpriced += 1
        else:
            priced.append(tier_cost + r.decision.cost_usd)
    return priced, unpriced


def _route_summary(name: str, records: Sequence[RouteRecord], cal: Calibration) -> RouteSummary:
    decided = [r for r in records if r.decision is not None]
    pairs = Counter((cal.gold[r.task_id], r.decision.tier) for r in decided)
    diffs = [tier_rank(r.decision.tier) - tier_rank(cal.gold[r.task_id]) for r in decided]
    priced_costs, unpriced = _priced_costs(decided, cal)
    return RouteSummary(
        router=name,
        n=len(records),
        errors=len(records) - len(decided),
        accuracy=_ratio(sum(d == 0 for d in diffs), len(records)),
        under_rate=_ratio(sum(d < 0 for d in diffs), len(decided)),
        over_rate=_ratio(sum(d > 0 for d in diffs), len(decided)),
        confusion={g: {c: pairs[(g, c)] for c in TIERS} for g in TIERS},
        expected_cost=_mean(priced_costs),
        router_cost_mean=_mean([r.decision.cost_usd for r in decided]),
        router_seconds_mean=_mean([r.decision.seconds for r in decided]),
        unpriced=unpriced,
    )


def summarize_route(records: Sequence[RouteRecord], cal: Calibration) -> list[RouteSummary]:
    graded = [r for r in records if cal.gold.get(r.task_id) is not None]
    return [_route_summary(n, [r for r in graded if r.router == n], cal) for n in _names(records)]


def reference_costs(cal: Calibration, task_ids: Sequence[str]) -> dict[str, float]:
    graded = [t for t in task_ids if cal.gold.get(t) is not None]
    strong_costs = [cal.mean_cost[t]["strong"] for t in graded if cal.mean_cost[t]["strong"] is not None]
    oracle_costs = [
        cal.mean_cost[t][cal.gold[t]] for t in graded if cal.mean_cost[t][cal.gold[t]] is not None
    ]
    return {
        STRONG_BASELINE: _mean(strong_costs),
        "oracle": _mean(oracle_costs),
    }


def _record_cost(r: E2ERecord) -> float:
    return (r.decision.cost_usd if r.decision else 0.0) + (r.run.cost_usd if r.run else 0.0)


def _record_seconds(r: E2ERecord) -> float:
    return (r.decision.seconds if r.decision else 0.0) + (r.run.seconds if r.run else 0.0)


def _e2e_summary(name: str, records: Sequence[E2ERecord]) -> E2ESummary:
    passed = sum(1 for r in records if r.run is not None and r.run.passed)
    mix = Counter(r.decision.tier for r in records if r.decision is not None)
    return E2ESummary(
        router=name,
        n=len(records),
        passed=passed,
        errors=sum(1 for r in records if r.error or (r.run is not None and r.run.error)),
        pass_rate=_ratio(passed, len(records)),
        total_cost=sum(_record_cost(r) for r in records),
        mean_seconds=_mean([_record_seconds(r) for r in records]),
        tier_mix={t: mix[t] for t in TIERS},
    )


def summarize_e2e(records: Sequence[E2ERecord]) -> list[E2ESummary]:
    base = [_e2e_summary(n, [r for r in records if r.router == n]) for n in _names(records)]
    strong = next((s for s in base if s.router == STRONG_BASELINE), None)
    if strong is None:
        return base
    return [
        replace(
            s,
            savings_vs_strong=1 - s.total_cost / strong.total_cost if strong.total_cost else None,
            quality_vs_strong=s.pass_rate / strong.pass_rate if strong.pass_rate else None,
        )
        for s in base
    ]
