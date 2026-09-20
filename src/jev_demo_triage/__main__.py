"""CLI: python -m jev_demo_triage --mode baseline --scenario deploy-regression"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from jev_demo_triage.agent import MODES, run_scenario
from jev_demo_triage.config import MissingKeyError
from jev_demo_triage.gate import GATE_POLICIES
from jev_demo_triage.report import format_table, summarize_repeats, write_json
from jev_demo_triage.scenarios import SCENARIOS, get_scenario
from jev_demo_triage.trace import Tracer

AVAILABLE_MODES = MODES


def _positive_int(value: str) -> int:
    i = int(value)
    if i <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {i}")
    return i


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jev_demo_triage")
    p.add_argument("--mode", choices=MODES, default="baseline")
    p.add_argument("--scenario", default=next(iter(SCENARIOS)))
    p.add_argument("--compare", action="store_true", help="run every scenario under every available mode")
    p.add_argument("--repeat", type=_positive_int, default=1)
    p.add_argument("--trace", action="store_true", help="print a step-by-step trace of each run to stderr")
    p.add_argument(
        "--gate-policy",
        choices=GATE_POLICIES,
        default="tuned",
        help=(
            "which instructions the gate classifier uses: 'tuned' = operator policy in gate.py, "
            "'default' = the package's own; only affects gate/both modes"
        ),
    )
    return p


def main(argv: Sequence[str] | None = None, runner=run_scenario) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.compare:
            pairs = [(get_scenario(n), m) for n in SCENARIOS for m in AVAILABLE_MODES]
        else:
            pairs = [(get_scenario(args.scenario), args.mode)]
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2

    kwargs: dict = {"gate_policy": args.gate_policy}
    if args.trace:
        kwargs["tracer"] = Tracer()
    try:
        results = [runner(s, m, **kwargs) for s, m in pairs for _ in range(args.repeat)]
    except MissingKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(format_table(results))
    if args.repeat > 1:
        print()
        print(summarize_repeats(results))
    if args.compare:
        print(f"\nSaved {write_json(results, Path('runs'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
