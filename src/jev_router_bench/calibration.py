"""Gold tiers: the cheapest tier that passes a task on every calibration repeat."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from jev_router_bench.pool import POOL, TIERS, Tier
from jev_router_bench.runner import TaskRun
from jev_router_bench.store import latest_file, write_json

CALIB_PREFIX = "calib"
TOO_EASY_SHARE = 0.8
TOO_HARD_SHARE = 0.3


@dataclass(frozen=True)
class Calibration:
    repeat: int
    gold: dict[str, Tier | None]
    # task id -> tier -> mean cost of non-error runs; None when the tier has no non-error run
    mean_cost: dict[str, dict[str, float | None]]
    runs: tuple[TaskRun, ...]


def gold_tier(runs: Sequence[TaskRun], repeat: int) -> Tier | None:
    """The cheapest tier that passes `repeat`/`repeat` with no errors.

    Tiers are walked cheapest-first. A tier with any errored run makes the task's gold
    unknown (None) before a passing tier is ever reached: an error means we don't actually
    know whether the model can solve the task at that tier, so we can't silently move it to
    a pricier tier as if it had simply failed. A tier with only non-error failures (not an
    error, just wrong) still moves on to the next tier, as before.
    """
    for tier in TIERS:
        tier_runs = [r for r in runs if r.tier == tier]
        if any(r.error is not None for r in tier_runs):
            return None
        if len(tier_runs) == repeat and all(r.passed for r in tier_runs):
            return tier
    return None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_calibration(runs: Sequence[TaskRun], repeat: int) -> Calibration:
    task_ids = list(dict.fromkeys(r.task_id for r in runs))
    by_task = {t: [r for r in runs if r.task_id == t] for t in task_ids}
    gold = {t: gold_tier(by_task[t], repeat) for t in task_ids}
    mean_cost = {
        t: {
            tier: _mean_or_none(
                [r.cost_usd for r in by_task[t] if r.tier == tier and r.error is None]
            )
            for tier in TIERS
        }
        for t in task_ids
    }
    return Calibration(repeat, gold, mean_cost, tuple(runs))


def calibration_warnings(cal: Calibration) -> list[str]:
    total = len(cal.gold)
    if total == 0:
        return []
    cheap = sum(1 for g in cal.gold.values() if g == "cheap") / total
    unsolved = sum(1 for g in cal.gold.values() if g is None) / total
    warnings = []
    if cheap > TOO_EASY_SHARE:
        warnings.append(
            f"{cheap:.0%} of tasks have gold 'cheap': the task set may be too easy to show routing value"
        )
    if unsolved > TOO_HARD_SHARE:
        warnings.append(
            f"{unsolved:.0%} of tasks have no tier passing {cal.repeat}/{cal.repeat}: "
            "they are left out of routing accuracy"
        )
    error_runs = [r for r in cal.runs if r.error is not None]
    if error_runs:
        error_task_ids = sorted(
            {r.task_id for r in error_runs if cal.gold.get(r.task_id) is None}
        )
        if error_task_ids:
            warnings.append(
                f"{len(error_runs)} calibration run(s) errored; gold is unknown for "
                f"task(s): {', '.join(error_task_ids)}"
            )
    return warnings


def save_calibration(cal: Calibration, directory: Path, now: datetime | None = None) -> Path:
    payload = {
        "repeat": cal.repeat,
        "pool": {
            tier: {
                "model_id": m.model_id,
                "input_usd_per_million": m.input_price * 1_000_000,
                "output_usd_per_million": m.output_price * 1_000_000,
            }
            for tier, m in POOL.items()
        },
        "gold": cal.gold,
        "mean_cost": cal.mean_cost,
        "runs": [asdict(r) for r in cal.runs],
    }
    return write_json(CALIB_PREFIX, payload, directory, now)


def _validate_mean_cost(mean_cost: Mapping[str, Mapping[str, object]], path: Path) -> None:
    for task_id, costs in mean_cost.items():
        missing = [tier for tier in TIERS if tier not in costs]
        bad_values = [
            v for k, v in costs.items() if k in TIERS and v is not None and not isinstance(v, (int, float))
        ]
        if missing or bad_values:
            raise ValueError(f"invalid mean_cost for task {task_id!r} in {path}: {dict(costs)}")


def load_calibration(path: Path) -> Calibration:
    data = json.loads(Path(path).read_text())
    gold = dict(data["gold"])
    bad = {t: g for t, g in gold.items() if g is not None and g not in TIERS}
    if bad:
        raise ValueError(f"unknown gold tier(s) in {path}: {bad}")
    mean_cost = {t: dict(costs) for t, costs in data["mean_cost"].items()}
    _validate_mean_cost(mean_cost, path)
    return Calibration(
        repeat=int(data["repeat"]),
        gold=gold,
        mean_cost=mean_cost,
        runs=tuple(TaskRun(**r) for r in data["runs"]),
    )


def latest_calibration(directory: Path) -> Path | None:
    return latest_file(CALIB_PREFIX, directory)
