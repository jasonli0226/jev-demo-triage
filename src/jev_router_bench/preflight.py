"""One cheap call per model tier and per router, so a broken setup fails in seconds, not hours."""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage

from jev_demo_triage.config import MissingKeyError
from jev_router_bench.pool import POOL, Tier
from jev_router_bench.report import format_rows
from jev_router_bench.routers import Router
from jev_router_bench.runner import describe_error
from jev_router_bench.tasks import Task

PREFLIGHT_PROMPT = "Reply with the single word OK."


@dataclass(frozen=True)
class Check:
    name: str
    target: str
    ok: bool
    seconds: float
    error: str | None = None


def _timed(name: str, target: str, call: Any) -> Check:
    started = time.perf_counter()
    try:
        call()
    except MissingKeyError:
        raise
    except Exception as exc:  # any failure here would fail every run of the benchmark
        return Check(name, target, False, time.perf_counter() - started, describe_error(exc))
    return Check(name, target, True, time.perf_counter() - started)


def check_models(models: Mapping[Tier, Any]) -> list[Check]:
    """Only reachability is checked; the answer text is not graded."""
    messages = [HumanMessage(PREFLIGHT_PROMPT)]
    return [
        _timed(tier, POOL[tier].model_id, lambda m=model: m.invoke(messages, config={"callbacks": []}))
        for tier, model in models.items()
    ]


def check_routers(routers: Sequence[Router], task: Task) -> list[Check]:
    return [_timed(f"router {r.name}", r.name, lambda r=r: r.route(task)) for r in routers]


def format_checks(checks: Sequence[Check]) -> str:
    rows = [
        (c.name, c.target, "ok" if c.ok else "FAIL", f"{c.seconds:.1f}s", c.error or "")
        for c in checks
    ]
    return format_rows(("check", "target", "status", "time", "error"), rows)
