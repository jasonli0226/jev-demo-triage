"""Environment and key handling."""

import os
from collections.abc import Mapping

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_GLM_MODEL = "z-ai/glm-4.7"


class MissingKeyError(RuntimeError):
    """Raised when a required API key is not configured."""


def _env(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def require_openrouter_key(environ: Mapping[str, str] | None = None) -> str:
    value = _env(environ).get("OPENROUTER_API_KEY", "").strip()
    if not value:
        raise MissingKeyError(
            "OPENROUTER_API_KEY is not set. Export it before running "
            "(agent A and Jev both go through OpenRouter)."
        )
    return value


def glm_model_name(environ: Mapping[str, str] | None = None) -> str:
    return _env(environ).get("GLM_MODEL", "").strip() or DEFAULT_GLM_MODEL
