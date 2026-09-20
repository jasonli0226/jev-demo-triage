"""Agent construction and the run loop."""

import time
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

from jev_demo_triage.config import (
    OPENROUTER_BASE_URL,
    glm_model_name,
    require_openrouter_key,
)
from jev_demo_triage.gate import OpenRouterAutoMode
from jev_demo_triage.jev import make_classifier
from jev_demo_triage.jev_tool import make_ask_jev
from jev_demo_triage.metrics import RunMetrics, UsageSink, build_metrics
from jev_demo_triage.scenarios import Scenario
from jev_demo_triage.tools import Action, RunContext, make_tools
from jev_demo_triage.trace import TraceHandler, Tracer, traced_factory

MODES = ("baseline", "tool", "gate", "both")
# LangGraph counts supersteps (model node + tools node per round), so 31 allows 15 model steps.
RECURSION_LIMIT = 31

SYSTEM_PROMPT = (
    "You are an on-call incident triage agent. Investigate the alert using the "
    "read-only tools (read_logs, get_metrics), then take the single most appropriate "
    "action: restart_service, rollback_deploy, run_shell, or escalate_to_human. "
    "Prefer reversible actions. If evidence is insufficient, escalate to a human "
    "instead of guessing. Finish with a one or two sentence summary of what you did and why."
)

TOOL_MODE_PROMPT = (
    " You also have ask_jev, a fast classifier. Use it for classification or yes/no "
    "judgments (severity, whether the evidence is enough to act, whether an action is risky). "
    "Use your own reasoning for everything open-ended."
)


@dataclass(frozen=True)
class RunResult:
    scenario: str
    mode: str
    passed: bool
    actions: tuple[Action, ...]
    final_answer: str
    metrics: RunMetrics
    error: str | None = None


def make_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=glm_model_name(),
        api_key=require_openrouter_key(),
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
    )


def _final_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def run_scenario(
    scenario: Scenario,
    mode: str,
    model: BaseChatModel | None = None,
    classifier_factory=None,
    tracer: Tracer | None = None,
) -> RunResult:
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Valid: {', '.join(MODES)}")

    if tracer is not None:
        tracer.header(scenario.name, mode)
    ctx = RunContext()
    sink: UsageSink | None = UsageSink() if mode != "baseline" else None
    factory = classifier_factory or make_classifier
    callbacks = []
    if tracer is not None:
        factory = traced_factory(factory, tracer)
        callbacks.append(TraceHandler(tracer))
    tools = make_tools(scenario.world, ctx)
    prompt = SYSTEM_PROMPT
    middleware = []
    if mode in ("tool", "both"):
        assert sink is not None
        tools = [*tools, make_ask_jev(sink, factory)]
        prompt = SYSTEM_PROMPT + TOOL_MODE_PROMPT
    if mode in ("gate", "both"):
        middleware.append(OpenRouterAutoMode(sink=sink, classifier_factory=factory))
    agent = create_agent(
        model or make_model(), tools=tools, system_prompt=prompt, middleware=middleware
    )

    start = time.perf_counter()
    error: str | None = None
    messages: list = []
    config: dict = {"recursion_limit": RECURSION_LIMIT}
    if callbacks:
        config["callbacks"] = callbacks
    try:
        out = agent.invoke({"messages": [HumanMessage(scenario.alert)]}, config=config)
        messages = out["messages"]
    except GraphRecursionError as exc:
        error = f"recursion limit reached: {exc}"
    except Exception as exc:
        error = f"run failed: {type(exc).__name__}: {exc}"
    wall = time.perf_counter() - start

    if tracer is not None:
        final = _final_text(messages)
        tracer.line("DONE", f"ERROR: {error}" if error else f"final answer: {final}")

    actions = tuple(ctx.actions)
    return RunResult(
        scenario=scenario.name,
        mode=mode,
        passed=error is None and scenario.check(actions),
        actions=actions,
        final_answer=_final_text(messages),
        metrics=build_metrics(messages, sink, wall),
        error=error,
    )
