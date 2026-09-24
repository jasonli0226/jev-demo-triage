"""CLI: jev-playground router {preflight,calibrate,route,e2e}"""

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jev_demo_triage.config import MissingKeyError
from jev_router_bench.calibration import (
    Calibration,
    build_calibration,
    calibration_warnings,
    latest_calibration,
    load_calibration,
    save_calibration,
)
from jev_router_bench.metrics import reference_costs, summarize_e2e, summarize_route
from jev_router_bench.pool import POOL, TIERS, Tier, make_chat_model
from jev_router_bench.preflight import Check, check_models, check_routers, format_checks
from jev_router_bench.report import format_calibration, format_e2e, format_route
from jev_router_bench.routers import FixedRouter, JevRouter, LlmRouter, OracleRouter, Router
from jev_router_bench.runner import calibrate, e2e_eval, route_eval
from jev_router_bench.store import write_json
from jev_router_bench.tasks import Task, select_tasks

ROUTER_NAMES = ("jev", "llm")

ModelFactory = Callable[[Tier], Any]
RouterFactory = Callable[[str], Router]


def default_model_factory(tier: Tier) -> Any:
    return make_chat_model(POOL[tier].model_id)


def default_router_factory(name: str) -> Router:
    return JevRouter() if name == "jev" else LlmRouter()


def _positive_int(value: str) -> int:
    i = int(value)
    if i <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {i}")
    return i


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jev-playground router")
    sub = p.add_subparsers(dest="command", required=True)
    pf = sub.add_parser("preflight", help="one call per tier and per router; exit 1 if any fails")
    pf.add_argument("--router", choices=(*ROUTER_NAMES, "all"), default="all")
    pf.add_argument("--task", action="append", help="task the routers route; default the first task")
    c = sub.add_parser("calibrate", help="run every task on every tier to find gold tiers")
    c.add_argument("--repeat", type=_positive_int, default=3)
    c.add_argument("--task", action="append", help="task id (repeatable); default all")
    c.add_argument("--no-preflight", action="store_true", help="skip the preflight check")
    for name, repeat, text in (
        ("route", 3, "score routers against calibrated gold tiers (no task execution)"),
        ("e2e", 1, "route, run the chosen model and grade its answer"),
    ):
        s = sub.add_parser(name, help=text)
        s.add_argument("--router", choices=(*ROUTER_NAMES, "all"), default="all")
        s.add_argument("--repeat", type=_positive_int, default=repeat)
        s.add_argument("--task", action="append", help="task id (repeatable); default all")
        s.add_argument("--calib", type=Path, help="calibration JSON; default newest runs/calib-*.json")
        s.add_argument("--no-preflight", action="store_true", help="skip the preflight check")
    return p


def _progress(_record: object) -> None:
    print(".", end="", file=sys.stderr, flush=True)


def _end_progress() -> None:
    print(file=sys.stderr)


def _router_names(choice: str) -> tuple[str, ...]:
    return ROUTER_NAMES if choice == "all" else (choice,)


def _preflight_passes(checks: Sequence[Check]) -> bool:
    """Prints the checks to stderr; False means the command should stop before any real run."""
    print(format_checks(checks), file=sys.stderr)
    if all(c.ok for c in checks):
        return True
    print("Preflight failed: fix the errors above, or pass --no-preflight to run anyway.", file=sys.stderr)
    return False


def _preflight(args, tasks: Sequence[Task], model_factory, router_factory, runs_dir: Path) -> int:
    models = {tier: model_factory(tier) for tier in TIERS}
    routers = [router_factory(n) for n in _router_names(args.router)]
    checks = check_models(models) + check_routers(routers, tasks[0])
    print(format_checks(checks))
    return 0 if all(c.ok for c in checks) else 1


class CalibrationReadError(RuntimeError):
    """The calibration file could not be read or parsed."""


def _load(args: argparse.Namespace, runs_dir: Path) -> tuple[Path | None, Calibration | None]:
    path = args.calib or latest_calibration(runs_dir)
    if path is None:
        return None, None
    try:
        return path, load_calibration(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:  # JSONDecodeError is a ValueError
        raise CalibrationReadError(f"Could not read calibration {path}: {exc}") from exc


def _calibrate(args, tasks: Sequence[Task], model_factory, router_factory, runs_dir: Path) -> int:
    models = {tier: model_factory(tier) for tier in TIERS}
    if not args.no_preflight and not _preflight_passes(check_models(models)):
        return 1
    runs = calibrate(tasks, models, args.repeat, on_result=_progress)
    _end_progress()
    cal = build_calibration(runs, args.repeat)
    print(format_calibration(cal, tasks))
    for warning in calibration_warnings(cal):
        print(f"WARNING: {warning}")
    print(f"\nSaved {save_calibration(cal, runs_dir)}")
    return 0


def _split_by_calibration(
    tasks: Sequence[Task], cal: Calibration
) -> tuple[list[Task], list[Task], list[Task]]:
    """Selected tasks split into: gold known, gold explicitly None, and absent from cal."""
    graded, gold_none, not_in_cal = [], [], []
    for t in tasks:
        if t.id not in cal.gold:
            not_in_cal.append(t)
        elif cal.gold[t.id] is None:
            gold_none.append(t)
        else:
            graded.append(t)
    return graded, gold_none, not_in_cal


def _route(args, tasks: Sequence[Task], model_factory, router_factory, runs_dir: Path) -> int:
    path, cal = _load(args, runs_dir)
    if cal is None:
        print(f"No calibration file in {runs_dir}/. Run `jev-playground router calibrate` first.", file=sys.stderr)
        return 2
    graded, gold_none, not_in_cal = _split_by_calibration(tasks, cal)
    print(
        f"Routing {len(graded)} graded task(s); skipped: {len(gold_none)} with gold none, "
        f"{len(not_in_cal)} not in calibration {path}",
        file=sys.stderr,
    )
    if not graded:
        return 2
    routers = [router_factory(n) for n in _router_names(args.router)]
    if not args.no_preflight and not _preflight_passes(check_routers(routers, graded[0])):
        return 1
    records = route_eval(graded, routers, args.repeat, on_result=_progress)
    _end_progress()
    summaries = summarize_route(records, cal)
    refs = reference_costs(cal, [t.id for t in graded])
    print(format_route(summaries, refs))
    payload = {
        "calibration": str(path),
        "records": [asdict(r) for r in records],
        "summaries": [asdict(s) for s in summaries],
        "reference_costs": refs,
    }
    print(f"\nSaved {write_json('router-route', payload, runs_dir)}")
    return 0


def _e2e(args, tasks: Sequence[Task], model_factory, router_factory, runs_dir: Path) -> int:
    path, cal = _load(args, runs_dir)
    named: list[Router] = [router_factory(n) for n in _router_names(args.router)]
    routers: list[Router] = [*named, *(FixedRouter(tier) for tier in TIERS)]
    if cal is None:
        print("No calibration file: the oracle baseline is skipped.", file=sys.stderr)
    else:
        routers.append(OracleRouter(cal.gold))
        missing = [t for t in tasks if t.id not in cal.gold]
        if missing:
            print(
                f"{len(missing)} selected task(s) not in calibration {path}; "
                "the oracle routes them to strong.",
                file=sys.stderr,
            )
    models = {tier: model_factory(tier) for tier in TIERS}
    if not args.no_preflight:
        checks = check_models(models) + check_routers(named, tasks[0])
        if not _preflight_passes(checks):
            return 1
    records = e2e_eval(tasks, routers, models, args.repeat, on_result=_progress)
    _end_progress()
    summaries = summarize_e2e(records)
    print(format_e2e(summaries))
    payload = {
        "calibration": str(path) if path else None,
        "records": [asdict(r) for r in records],
        "summaries": [asdict(s) for s in summaries],
    }
    print(f"\nSaved {write_json('router-e2e', payload, runs_dir)}")
    return 0


_COMMANDS = {"preflight": _preflight, "calibrate": _calibrate, "route": _route, "e2e": _e2e}


def main(
    argv: Sequence[str] | None = None,
    *,
    model_factory: ModelFactory = default_model_factory,
    router_factory: RouterFactory = default_router_factory,
    runs_dir: Path = Path("runs"),
) -> int:
    args = _parser().parse_args(argv)
    try:
        tasks = select_tasks(args.task)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2
    try:
        return _COMMANDS[args.command](args, tasks, model_factory, router_factory, runs_dir)
    except (MissingKeyError, CalibrationReadError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
