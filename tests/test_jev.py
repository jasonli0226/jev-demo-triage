import json
import types

import httpx2
import pytest
from langchain_typesafe import Choice, Noul, Score

from jev_demo_triage.jev import DECISIONS_URL, JEV_MODEL, EndpointChangedError, make_classifier
from jev_demo_triage.metrics import UsageSink

NOUL_BODY = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {"urgent": {"type": "noul", "noul": 0.96}},
    "usage": {"input_tokens": 285, "output_tokens": 20, "cost": 1.197e-05},
    "id": "gen-dec-1",
    "provider": "TypeSafe",
}


def _client(handler):
    return httpx2.Client(transport=httpx2.MockTransport(handler))


def test_posts_native_shaped_body_to_openrouter_decisions_endpoint():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=NOUL_BODY)

    clf = make_classifier(
        {"urgent": Noul(instructions="Does this need attention now?")},
        api_key="sk-or-test",
        client=_client(handler),
    )
    response = clf.invoke("deploy failed twice")
    assert seen["url"] == DECISIONS_URL == "https://openrouter.ai/api/alpha/decisions"
    assert seen["auth"] == "Bearer sk-or-test"
    assert seen["body"]["model"] == JEV_MODEL == "~typesafe/jev-latest"
    assert seen["body"]["state"] == "deploy failed twice"
    assert seen["body"]["questions"]["urgent"] == {
        "type": "noul", "instructions": "Does this need attention now?"
    }
    assert response.nouls["urgent"].noul == pytest.approx(0.96)


def test_usage_is_recorded_into_sink():
    sink = UsageSink()
    clf = make_classifier(
        {"urgent": Noul(instructions="q")},
        sink=sink,
        api_key="k",
        client=_client(lambda r: httpx2.Response(200, json=NOUL_BODY)),
    )
    clf.invoke("x")
    clf.invoke("y")
    assert (sink.calls, sink.input_tokens, sink.output_tokens) == (2, 570, 40)


def test_supports_choice_and_score_questions():
    body = {
        "model": "m",
        "answers": {
            "risk": {"type": "choice", "choice": "destructive",
                     "probabilities": {"safe": 0, "risky": 0, "destructive": 1}, "confidence": 1},
            "sev": {"type": "score", "score": 1.99, "legend": {"0": "low", "1": "medium", "2": "high"},
                    "probabilities": {"0": 0, "1": 0, "2": 1}, "confidence": 0.99},
        },
        "usage": {"input_tokens": 377, "output_tokens": 57},
    }
    clf = make_classifier(
        {
            "risk": Choice(instructions="Risk?", criteria={"safe": "a", "risky": "b", "destructive": "c"}),
            "sev": Score(instructions="Severity", criteria=["low", "medium", "high"]),
        },
        api_key="k",
        client=_client(lambda r: httpx2.Response(200, json=body)),
    )
    out = clf.invoke("rm -rf /var/lib/postgres")
    assert out.choices["risk"].choice == "destructive"
    assert out.scores["sev"].score == pytest.approx(1.99)


def test_404_raises_endpoint_changed_error():
    clf = make_classifier(
        {"urgent": Noul(instructions="q")},
        api_key="k",
        client=_client(lambda r: httpx2.Response(404, json={"error": {"message": "Not Found", "code": 404}})),
    )
    with pytest.raises(EndpointChangedError) as info:
        clf.invoke("x")
    assert DECISIONS_URL in str(info.value)


def test_missing_key_raises_before_any_request(monkeypatch):
    from jev_demo_triage.config import MissingKeyError

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingKeyError):
        make_classifier({"urgent": Noul(instructions="q")})


@pytest.mark.live
def test_live_decisions_endpoint():
    import os

    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("needs OPENROUTER_API_KEY")
    sink = UsageSink()
    clf = make_classifier({"urgent": Noul(instructions="Does this need attention now?")}, sink=sink)
    out = clf.invoke("The deploy failed twice and customers see 500s.")
    assert out.nouls["urgent"].noul > 0.5
    assert sink.calls == 1


def _ticking_clock(monkeypatch, step=0.5):
    import jev_demo_triage.jev as jev_mod

    state = {"t": 0.0}

    def fake():
        state["t"] += step
        return state["t"]

    # Patch only jev's own `time` name so the HTTP library's clock use does not consume ticks.
    monkeypatch.setattr(jev_mod, "time", types.SimpleNamespace(perf_counter=fake))


def test_classifier_time_added_to_sink_per_call_without_double_counting(monkeypatch):
    _ticking_clock(monkeypatch)
    sink = UsageSink()
    clf = make_classifier(
        {"urgent": Noul(instructions="q")},
        sink=sink,
        api_key="k",
        client=_client(lambda r: httpx2.Response(200, json=NOUL_BODY)),
    )
    clf.invoke("x")
    assert sink.seconds == pytest.approx(0.5)
    clf.invoke("y")
    assert sink.seconds == pytest.approx(1.0)
    assert (sink.calls, sink.input_tokens, sink.output_tokens) == (2, 570, 40)


def test_classifier_time_recorded_when_call_raises(monkeypatch):
    _ticking_clock(monkeypatch)
    sink = UsageSink()
    clf = make_classifier(
        {"urgent": Noul(instructions="q")},
        sink=sink,
        api_key="k",
        client=_client(lambda r: httpx2.Response(404, json={"error": {"message": "nf", "code": 404}})),
    )
    with pytest.raises(EndpointChangedError):
        clf.invoke("x")
    assert sink.seconds == pytest.approx(0.5)
    assert sink.calls == 0


def test_async_classifier_time_recorded(monkeypatch):
    import asyncio

    from langchain_typesafe import TypeSafeClassifier

    async def fake_aclassify(self, state):  # noqa: ANN001
        return "ok"

    monkeypatch.setattr(TypeSafeClassifier, "_aclassify", fake_aclassify)
    _ticking_clock(monkeypatch)
    sink = UsageSink()
    clf = make_classifier({"urgent": Noul(instructions="q")}, sink=sink, api_key="k")
    assert asyncio.run(clf._aclassify("x")) == "ok"
    assert sink.seconds == pytest.approx(0.5)
