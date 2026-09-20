"""Jev classifier that talks to OpenRouter's (alpha) decisions endpoint."""

import time
from typing import Any

import httpx2
from langchain_typesafe import Question, TypeSafeClassifier
from langchain_typesafe.client import TypeSafeNotFoundError
from langchain_typesafe.types import ClassificationResponse
from pydantic import Field

from jev_demo_triage.config import require_openrouter_key
from jev_demo_triage.metrics import UsageSink

# Single place to change if the alpha path moves.
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "~typesafe/jev-latest"


class EndpointChangedError(RuntimeError):
    """The OpenRouter decisions endpoint returned 404; the alpha path likely moved."""


class OpenRouterClassifier(TypeSafeClassifier):
    sink: Any = Field(default=None, exclude=True, repr=False)

    @property
    def _endpoint(self) -> str:
        return DECISIONS_URL

    def _record_usage(self, response: ClassificationResponse) -> ClassificationResponse:
        if self.sink is not None:
            self.sink.add(response.usage.input_tokens or 0, response.usage.output_tokens or 0)
        return super()._record_usage(response)

    def _record_seconds(self, started: float) -> None:
        if self.sink is not None:
            self.sink.add_seconds(time.perf_counter() - started)

    def _classify(self, state):  # noqa: ANN001
        started = time.perf_counter()
        try:
            return super()._classify(state)
        except TypeSafeNotFoundError as exc:
            raise EndpointChangedError(
                f"{DECISIONS_URL} returned 404; the OpenRouter alpha endpoint may have moved."
            ) from exc
        finally:
            self._record_seconds(started)

    async def _aclassify(self, state):  # noqa: ANN001
        started = time.perf_counter()
        try:
            return await super()._aclassify(state)
        except TypeSafeNotFoundError as exc:
            raise EndpointChangedError(
                f"{DECISIONS_URL} returned 404; the OpenRouter alpha endpoint may have moved."
            ) from exc
        finally:
            self._record_seconds(started)


def make_classifier(
    questions: dict[str, Question],
    sink: UsageSink | None = None,
    api_key: str | None = None,
    client: httpx2.Client | None = None,
) -> OpenRouterClassifier:
    return OpenRouterClassifier(
        api_key=api_key or require_openrouter_key(),
        model=JEV_MODEL,
        questions=questions,
        sink=sink,
        client=client,
    )
