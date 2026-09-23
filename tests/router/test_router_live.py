"""Hits real OpenRouter endpoints. Run with: uv run pytest -m live tests/router/test_router_live.py"""

import pytest

from jev_router_bench.pool import POOL, TIERS, make_chat_model
from jev_router_bench.routers import JevRouter, LlmRouter
from jev_router_bench.runner import run_task
from jev_router_bench.tasks import select_tasks

pytestmark = pytest.mark.live

TASK = select_tasks(["add-simple"])[0]


@pytest.mark.parametrize("tier", TIERS)
def test_each_tier_answers_simple_task(tier):
    run = run_task(TASK, tier, make_chat_model(POOL[tier].model_id))
    assert run.error is None, run.error
    assert run.passed, run
    assert run.input_tokens > 0


@pytest.mark.parametrize("router_cls", [JevRouter, LlmRouter])
def test_router_returns_a_tier(router_cls):
    decision = router_cls().route(TASK)
    assert decision.tier in TIERS
    assert decision.input_tokens > 0
