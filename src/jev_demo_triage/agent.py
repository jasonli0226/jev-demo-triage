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
from jev_demo_triage.gate import GATE_POLICIES, OpenRouterAutoMode
from jev_demo_triage.jev import make_classifier
from jev_demo_triage.jev_tool import make_ask_jev
from jev_demo_triage.llm_gate import make_llm_factory
from jev_demo_triage.metrics import (
    GLM_INPUT_PRICE_PER_TOKEN,
    GLM_OUTPUT_PRICE_PER_TOKEN,
    RunMetrics,
    UsageSink,
    build_metrics,
)
from jev_demo_triage.scenarios import Scenario
from jev_demo_triage.tools import Action, RunContext, make_tools
from jev_demo_triage.trace import TraceHandler, Tracer, traced_factory

MODES = ("baseline", "tool", "gate", "both", "llm-gate")
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
    gate_policy: str | None = None


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
    gate_policy: str = "tuned",
    llm_sink: UsageSink | None = None,
    gate_model: BaseChatModel | None = None,
) -> RunResult:
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Valid: {', '.join(MODES)}")
    if gate_policy not in GATE_POLICIES:
        raise ValueError(f"Unknown gate policy '{gate_policy}'. Valid: {', '.join(GATE_POLICIES)}")

    gated = mode in ("gate", "both", "llm-gate")
    if tracer is not None:
        tracer.header(scenario.name, mode, gate_policy if gated else None)
    ctx = RunContext()
    llm_gate = mode == "llm-gate"
    sink: UsageSink | None = UsageSink() if mode not in ("baseline", "llm-gate") else None
    if llm_gate:
        llm_sink = llm_sink or UsageSink(
            input_price=GLM_INPUT_PRICE_PER_TOKEN, output_price=GLM_OUTPUT_PRICE_PER_TOKEN
        )
        # A supplied classifier_factory takes precedence over gate_model.
        # Deferred: the gate model (and its API key check) is built only when the gate is.
        factory = classifier_factory or make_llm_factory(lambda: gate_model or make_model())
    else:
        factory = classifier_factory or make_classifier
    callbacks = []
    if tracer is not None:
        factory = traced_factory(factory, tracer, label="LLM" if llm_gate else "JEV")
        callbacks.append(TraceHandler(tracer))
    tools = make_tools(scenario.world, ctx)
    prompt = SYSTEM_PROMPT
    middleware = []
    if mode in ("tool", "both"):
        assert sink is not None
        tools = [*tools, make_ask_jev(sink, factory)]
        prompt = SYSTEM_PROMPT + TOOL_MODE_PROMPT
    if gated:
        middleware.append(
            OpenRouterAutoMode(
                sink=llm_sink if llm_gate else sink, classifier_factory=factory, policy=gate_policy
            )
        )
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
        metrics=build_metrics(messages, sink, wall, llm_sink=llm_sink),
        error=error,
        gate_policy=gate_policy if gated else None,
    )
