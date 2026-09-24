"""Runs tasks on pool models and routers over tasks; returns frozen records."""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from jev_demo_triage.config import MissingKeyError
from jev_router_bench.grading import grade
from jev_router_bench.pool import POOL, Tier, content_text
from jev_router_bench.routers import RouteDecision, Router
from jev_router_bench.tasks import ANSWER_SYSTEM_PROMPT, Task

Grader = Callable[[Task, str], Any]
OnResult = Callable[[object], None] | None


@dataclass(frozen=True)
class TaskRun:
    task_id: str
    tier: Tier
    passed: bool
    reason: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    seconds: float
    error: str | None = None
    reply: str | None = None  # the model's answer text; None when the call failed


@dataclass(frozen=True)
class RouteRecord:
    router: str
    task_id: str
    repeat: int
    decision: RouteDecision | None
    error: str | None = None


@dataclass(frozen=True)
class E2ERecord:
    router: str
    task_id: str
    repeat: int
    decision: RouteDecision | None
    run: TaskRun | None
    error: str | None = None  # router error; a model error is in run.error


def describe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def run_task(task: Task, tier: Tier, model: Any, grader: Grader = grade) -> TaskRun:
    messages = [SystemMessage(ANSWER_SYSTEM_PROMPT), HumanMessage(task.prompt)]
    started = time.perf_counter()
    try:
        reply = model.invoke(messages, config={"callbacks": []})
    except MissingKeyError:
        raise
    except Exception as exc:  # recorded as an ERROR row; the run continues
        seconds = time.perf_counter() - started
        return TaskRun(task.id, tier, False, "error", 0, 0, 0.0, seconds, describe_error(exc))
    seconds = time.perf_counter() - started
    usage = getattr(reply, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    cost = POOL[tier].cost(input_tokens, output_tokens)
    text = content_text(reply.content)
    try:
        result = grader(task, text)
    except MissingKeyError:
        raise
    except Exception as exc:  # recorded as an ERROR row; measured tokens/cost are kept
        return TaskRun(
            task.id, tier, False, "error", input_tokens, output_tokens, cost, seconds, describe_error(exc), text,
        )
    return TaskRun(
        task.id, tier, result.passed, result.reason, input_tokens, output_tokens, cost, seconds, reply=text,
    )


def _emit(on_result: OnResult, record: object) -> None:
    if on_result is not None:
        on_result(record)


def calibrate(
    tasks: Sequence[Task],
    models: Mapping[Tier, Any],
    repeat: int,
    on_result: OnResult = None,
    grader: Grader = grade,
) -> list[TaskRun]:
    runs: list[TaskRun] = []
    for task in tasks:
        for tier, model in models.items():
            for _ in range(repeat):
                run = run_task(task, tier, model, grader)
                runs.append(run)
                _emit(on_result, run)
    return runs


def _decide(router: Router, task: Task) -> tuple[RouteDecision | None, str | None]:
    try:
        return router.route(task), None
    except MissingKeyError:
        raise
    except Exception as exc:  # recorded as an ERROR row, never a default tier
        return None, describe_error(exc)


def route_eval(
    tasks: Sequence[Task], routers: Sequence[Router], repeat: int, on_result: OnResult = None
) -> list[RouteRecord]:
    records: list[RouteRecord] = []
    for router in routers:
        for task in tasks:
            for i in range(repeat):
                decision, error = _decide(router, task)
                record = RouteRecord(router.name, task.id, i, decision, error)
                records.append(record)
                _emit(on_result, record)
    return records


def e2e_eval(
    tasks: Sequence[Task],
    routers: Sequence[Router],
    models: Mapping[Tier, Any],
    repeat: int,
    on_result: OnResult = None,
    grader: Grader = grade,
) -> list[E2ERecord]:
    records: list[E2ERecord] = []
    for router in routers:
        for task in tasks:
            for i in range(repeat):
                decision, error = _decide(router, task)
                run = None if decision is None else run_task(task, decision.tier, models[decision.tier], grader)
                record = E2ERecord(router.name, task.id, i, decision, run, error)
                records.append(record)
                _emit(on_result, record)
    return records
