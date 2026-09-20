"""Tool-risk gate: package AutoModeMiddleware, but classifying through OpenRouter."""

import os

from langchain_typesafe.experimental.middleware import AutoModeMiddleware

from jev_demo_triage.jev import make_classifier
from jev_demo_triage.metrics import UsageSink

GATED_TOOLS = ("run_shell", "restart_service")
QUESTION_ID = "is_risky"

# AutoModeMiddleware.__init__ builds its own TypeSafeClassifier and requires this
# variable to exist. The placeholder is never sent anywhere: we replace the
# classifier immediately afterwards.
_PLACEHOLDER_KEY = "unused-placeholder"


class OpenRouterAutoMode(AutoModeMiddleware):
    def __init__(self, sink: UsageSink | None = None, classifier_factory=make_classifier) -> None:
        os.environ.setdefault("TYPESAFE_API_KEY", _PLACEHOLDER_KEY)
        super().__init__(tools=list(GATED_TOOLS))
        self.classifier = classifier_factory(self.classifier.questions, sink)
