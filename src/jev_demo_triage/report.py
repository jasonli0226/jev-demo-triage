"""Table, repeat summary and JSON output."""

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from jev_demo_triage.agent import RunResult

_HEADERS = ("scenario", "mode", "outcome", "steps", "glm_calls", "jev_calls", "tokens", "jev_cost", "time",
            "llm_gate_calls", "llm_gate_cost", "clf_s")


def _outcome(r: RunResult) -> str:
    if r.error:
        return "ERROR"
    return "PASS" if r.passed else "FAIL"


def _row(r: RunResult) -> tuple[str, ...]:
    m = r.metrics
    tokens = m.glm.input_tokens + m.glm.output_tokens + m.jev_tokens + m.llm_gate_tokens
    return (
        r.scenario, r.mode, _outcome(r), str(m.steps), str(m.glm.calls),
        str(m.jev_calls), str(tokens), f"${m.jev_cost_usd:.6f}", f"{m.wall_seconds:.1f}s",
        str(m.llm_gate_calls), f"${m.llm_gate_cost_usd:.6f}", f"{m.classifier_seconds:.1f}s",
    )


def format_table(results: Sequence[RunResult]) -> str:
    rows = [_HEADERS, *(_row(r) for r in results)]
    widths = [max(len(row[i]) for row in rows) for i in range(len(_HEADERS))]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)


def summarize_repeats(results: Sequence[RunResult]) -> str:
    by_mode: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_mode[r.mode].append(r.passed)
    return "\n".join(
        f"{mode}: {sum(p)}/{len(p)} passed" for mode, p in by_mode.items()
    )


def write_json(results: Sequence[RunResult], directory: Path, now: datetime | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    path = directory / f"run-{stamp}.json"
    path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    return path
