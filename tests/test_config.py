import pytest

from jev_demo_triage.config import (
    DEFAULT_GLM_MODEL,
    MissingKeyError,
    glm_model_name,
    require_openrouter_key,
)


def test_require_key_returns_value():
    assert require_openrouter_key({"OPENROUTER_API_KEY": "sk-or-abc"}) == "sk-or-abc"


@pytest.mark.parametrize("env", [{}, {"OPENROUTER_API_KEY": ""}, {"OPENROUTER_API_KEY": "   "}])
def test_require_key_missing_or_blank(env):
    with pytest.raises(MissingKeyError) as info:
        require_openrouter_key(env)
    assert "OPENROUTER_API_KEY" in str(info.value)


def test_error_message_does_not_contain_key_value():
    with pytest.raises(MissingKeyError) as info:
        require_openrouter_key({"OPENROUTER_API_KEY": " "})
    assert "sk-" not in str(info.value)


def test_glm_model_default_and_override():
    assert glm_model_name({}) == DEFAULT_GLM_MODEL
    assert glm_model_name({"GLM_MODEL": "z-ai/glm-5"}) == "z-ai/glm-5"
    assert glm_model_name({"GLM_MODEL": " "}) == DEFAULT_GLM_MODEL
