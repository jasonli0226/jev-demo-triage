import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from jev_demo_triage.config import MissingKeyError
from jev_router_bench.cli import main
from jev_router_bench.routers import RouteDecision

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
ANSWERS = {"cheap": "ANSWER: 900", "mid": "ANSWER: 936", "strong": "ANSWER: 936"}


def model_factory(tier):
    return FakeMessagesListChatModel(responses=[AIMessage(content=ANSWERS[tier], usage_metadata=USAGE)])


class FakeRouter:
    def __init__(self, name, tier="mid"):
        self.name = name
        self.tier = tier

    def route(self, task):
        return RouteDecision(self.tier, cost_usd=0.00001, seconds=0.01)


def router_factory(name):
    return FakeRouter(name, "mid" if name == "jev" else "strong")


def _main(argv, tmp_path, **kw):
    kw.setdefault("model_factory", model_factory)
    kw.setdefault("router_factory", router_factory)
    return main(argv, runs_dir=tmp_path, **kw)


def test_calibrate_writes_file_and_prints_gold(tmp_path, capsys):
    assert _main(["calibrate", "--task", "add-simple", "--repeat", "2"], tmp_path) == 0
    out = capsys.readouterr().out
    assert "add-simple" in out and "0/2" in out and "mid" in out
    assert list(tmp_path.glob("calib-*.json"))


def test_route_needs_calibration(tmp_path, capsys):
    assert _main(["route", "--task", "add-simple"], tmp_path) == 2
    assert "calibrate" in capsys.readouterr().err


def test_route_after_calibration(tmp_path, capsys):
    _main(["calibrate", "--task", "add-simple", "--repeat", "1"], tmp_path)
    capsys.readouterr()
    assert _main(["route", "--task", "add-simple", "--repeat", "1"], tmp_path) == 0
    out = capsys.readouterr().out
    assert "jev" in out and "llm" in out and "100.0%" in out
    assert list(tmp_path.glob("router-route-*.json"))


def test_route_skips_ungraded_and_not_in_calibration(tmp_path, capsys):
    _main(["calibrate", "--task", "add-simple", "--repeat", "1"], tmp_path)
    capsys.readouterr()
    calls = []

    class RecordingRouter(FakeRouter):
        def route(self, task):
            calls.append(task.id)
            return super().route(task)

    def factory(name):
        return RecordingRouter(name, "mid" if name == "jev" else "strong")

    code = _main(
        ["route", "--task", "add-simple", "--task", "percent", "--repeat", "1", "--no-preflight"],
        tmp_path,
        router_factory=factory,
    )
    assert code == 0
    assert calls == ["add-simple", "add-simple"]
    err = capsys.readouterr().err
    assert "1 not in calibration" in err
    assert "Routing 1 graded" in err


def test_route_all_tasks_not_in_calibration_exits_2(tmp_path, capsys):
    _main(["calibrate", "--task", "add-simple", "--repeat", "1"], tmp_path)
    capsys.readouterr()
    calls = []

    class RecordingRouter(FakeRouter):
        def route(self, task):
            calls.append(task.id)
            return super().route(task)

    def factory(name):
        return RecordingRouter(name, "mid" if name == "jev" else "strong")

    code = _main(
        ["route", "--task", "percent", "--repeat", "1", "--no-preflight"], tmp_path, router_factory=factory
    )
    assert code == 2
    assert calls == []
    assert "not in calibration" in capsys.readouterr().err


def test_route_single_router(tmp_path, capsys):
    _main(["calibrate", "--task", "add-simple", "--repeat", "1"], tmp_path)
    capsys.readouterr()
    _main(["route", "--task", "add-simple", "--repeat", "1", "--router", "jev"], tmp_path)
    table = capsys.readouterr().out.split("\n\n")[0]
    assert "jev" in table and "llm" not in table


def test_route_with_unreadable_calibration(tmp_path, capsys):
    bad = tmp_path / "broken.json"
    bad.write_text("{")
    assert _main(["route", "--calib", str(bad)], tmp_path) == 2
    assert "Could not read calibration" in capsys.readouterr().err


def test_e2e_without_calibration_skips_oracle(tmp_path, capsys):
    assert _main(["e2e", "--task", "add-simple"], tmp_path) == 0
    captured = capsys.readouterr()
    table = captured.out.split("\nSaved")[0]
    for name in ("jev", "llm", "always-cheap", "always-mid", "always-strong"):
        assert name in table
    assert "oracle" not in table
    assert "oracle" in captured.err
    assert list(tmp_path.glob("router-e2e-*.json"))


def test_e2e_with_calibration_includes_oracle(tmp_path, capsys):
    _main(["calibrate", "--task", "add-simple", "--repeat", "1"], tmp_path)
    capsys.readouterr()
    assert _main(["e2e", "--task", "add-simple"], tmp_path) == 0
    assert "oracle" in capsys.readouterr().out


def test_e2e_warns_about_tasks_not_in_calibration(tmp_path, capsys):
    _main(["calibrate", "--task", "add-simple", "--repeat", "1"], tmp_path)
    capsys.readouterr()
    assert _main(["e2e", "--task", "add-simple", "--task", "percent"], tmp_path) == 0
    err = capsys.readouterr().err
    assert "1" in err and "not in calibration" in err and "strong" in err


def test_unknown_task_exits_2(tmp_path, capsys):
    assert _main(["calibrate", "--task", "nope"], tmp_path) == 2
    assert "Unknown task" in capsys.readouterr().err


def test_missing_key_exits_2(tmp_path, capsys):
    def keyless(name):
        raise MissingKeyError("OPENROUTER_API_KEY is not set.")

    _main(["calibrate", "--task", "add-simple", "--repeat", "1"], tmp_path)
    assert _main(["route", "--task", "add-simple"], tmp_path, router_factory=keyless) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_repeat_must_be_positive(tmp_path):
    with pytest.raises(SystemExit):
        _main(["calibrate", "--repeat", "0"], tmp_path)


class BrokenModel:
    def invoke(self, *args, **kwargs):
        raise PermissionError("403 provider Terms Of Service")


def broken_strong_factory(tier):
    return BrokenModel() if tier == "strong" else model_factory(tier)


class BrokenRouter(FakeRouter):
    def route(self, task):
        raise ValueError("no JSON")


def test_preflight_command_passes(tmp_path, capsys):
    assert _main(["preflight"], tmp_path) == 0
    out = capsys.readouterr().out
    for name in ("cheap", "mid", "strong", "router jev", "router llm"):
        assert name in out
    assert "FAIL" not in out


def test_preflight_command_fails_on_broken_tier(tmp_path, capsys):
    assert _main(["preflight"], tmp_path, model_factory=broken_strong_factory) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "Terms Of Service" in out


def test_calibrate_aborts_when_preflight_fails(tmp_path, capsys):
    code = _main(["calibrate", "--task", "add-simple", "--repeat", "1"], tmp_path, model_factory=broken_strong_factory)
    assert code == 1
    assert "Preflight failed" in capsys.readouterr().err
    assert not list(tmp_path.glob("calib-*.json"))


def test_calibrate_no_preflight_runs_anyway(tmp_path):
    code = _main(
        ["calibrate", "--task", "add-simple", "--repeat", "1", "--no-preflight"],
        tmp_path,
        model_factory=broken_strong_factory,
    )
    assert code == 0
    assert list(tmp_path.glob("calib-*.json"))


def test_route_aborts_when_router_preflight_fails(tmp_path, capsys):
    _main(["calibrate", "--task", "add-simple", "--repeat", "1"], tmp_path)
    capsys.readouterr()
    code = _main(["route", "--task", "add-simple"], tmp_path, router_factory=lambda n: BrokenRouter(n))
    assert code == 1
    assert "Preflight failed" in capsys.readouterr().err
    assert not list(tmp_path.glob("router-route-*.json"))


def test_e2e_aborts_when_model_preflight_fails(tmp_path, capsys):
    assert _main(["e2e", "--task", "add-simple"], tmp_path, model_factory=broken_strong_factory) == 1
    assert "Preflight failed" in capsys.readouterr().err
    assert not list(tmp_path.glob("router-e2e-*.json"))
