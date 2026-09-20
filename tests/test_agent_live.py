import os

import pytest

from jev_demo_triage.agent import run_scenario
from jev_demo_triage.scenarios import SCENARIOS

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="needs OPENROUTER_API_KEY")
def test_glm_calls_tools_on_simple_scenario():
    result = run_scenario(SCENARIOS["simple-restart"], "baseline")
    assert result.error is None, result.error
    assert result.metrics.glm.calls >= 2  # at least one tool round trip plus the final answer
    assert result.metrics.glm.input_tokens > 0
