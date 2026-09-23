"""CLI: jev-playground {triage,router} ..."""

import sys
from collections.abc import Callable, Mapping, Sequence

from jev_demo_triage.__main__ import main as triage_main
from jev_router_bench import cli

USAGE = (
    "usage: jev-playground {triage,router} ...\n"
    "  triage  incident-triage agent: baseline vs Jev tool/gate (see docs/triage.md)\n"
    "  router  smart-router benchmark: Jev vs LLM router (see docs/router.md)\n"
    "Run `jev-playground <command> --help` for command options."
)

COMMANDS: dict[str, Callable[[Sequence[str]], int]] = {"triage": triage_main, "router": cli.main}


def main(
    argv: Sequence[str] | None = None,
    commands: Mapping[str, Callable[[Sequence[str]], int]] | None = None,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    table = COMMANDS if commands is None else commands
    if args and args[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if not args or args[0] not in table:
        print(USAGE, file=sys.stderr)
        return 2
    return table[args[0]](args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
