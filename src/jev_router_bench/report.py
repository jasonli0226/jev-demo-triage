"""Plain-text tables for calibration, routing and end-to-end results."""

from collections.abc import Mapping, Sequence

from jev_router_bench.calibration import Calibration
from jev_router_bench.metrics import E2ESummary, RouteSummary
from jev_router_bench.pool import TIERS
from jev_router_bench.tasks import Task


def format_rows(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    table = [tuple(headers), *(tuple(r) for r in rows)]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in table
    )


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def _usd(value: float) -> str:
    return f"${value:.6f}"


def format_calibration(cal: Calibration, tasks: Sequence[Task]) -> str:
    rows = []
    for task in tasks:
        cells = []
        for tier in TIERS:
            runs = [r for r in cal.runs if r.task_id == task.id and r.tier == tier]
            errors = sum(1 for r in runs if r.error is not None)
            cell = f"{sum(r.passed for r in runs)}/{len(runs)}"
            cells.append(f"{cell} E{errors}" if errors else cell)
        rows.append((task.id, task.intended_tier, *cells, cal.gold.get(task.id) or "none"))
    return format_rows(("task", "intended", *TIERS, "gold"), rows)


def _confusion(summary: RouteSummary) -> str:
    rows = [(f"gold {g}", *(str(summary.confusion[g][c]) for c in TIERS)) for g in TIERS]
    return f"confusion {summary.router} (rows gold, columns chosen)\n" + format_rows(("", *TIERS), rows)


def format_route(summaries: Sequence[RouteSummary], refs: Mapping[str, float]) -> str:
    headers = (
        "router", "n", "errors", "accuracy", "under", "over", "exp_cost", "unpriced",
        "router_cost", "router_s",
    )
    rows = [
        (
            s.router, str(s.n), str(s.errors), _pct(s.accuracy), _pct(s.under_rate), _pct(s.over_rate),
            _usd(s.expected_cost), str(s.unpriced), _usd(s.router_cost_mean), f"{s.router_seconds_mean:.2f}s",
        )
        for s in summaries
    ]
    reference = "reference expected cost per task: " + ", ".join(f"{k} {_usd(v)}" for k, v in refs.items())
    return "\n\n".join([format_rows(headers, rows), reference, *(_confusion(s) for s in summaries)])


def format_e2e(summaries: Sequence[E2ESummary]) -> str:
    headers = ("router", "n", "passed", "errors", "pass_rate", "total_cost", "savings", "quality", "mean_s", "mix")
    rows = [
        (
            s.router, str(s.n), str(s.passed), str(s.errors), _pct(s.pass_rate), _usd(s.total_cost),
            _pct(s.savings_vs_strong), _pct(s.quality_vs_strong), f"{s.mean_seconds:.1f}s",
            "/".join(f"{t[0]}{s.tier_mix.get(t, 0)}" for t in TIERS),
        )
        for s in summaries
    ]
    return format_rows(headers, rows)
