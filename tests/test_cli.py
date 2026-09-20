import pytest

from jev_demo_triage.__main__ import main
from jev_demo_triage.config import MissingKeyError
from tests.test_report import _result


def test_single_run_prints_table_and_returns_zero(capsys):
    calls = []

    def runner(scenario, mode, **kw):
        calls.append((scenario.name, mode))
        return _result(scenario.name, mode)

    code = main(["--scenario", "deploy-regression"], runner=runner)
    assert code == 0
    assert calls == [("deploy-regression", "baseline")]
    assert "deploy-regression" in capsys.readouterr().out


def test_compare_runs_every_scenario(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def runner(scenario, mode, **kw):
        calls.append((scenario.name, mode))
        return _result(scenario.name, mode)

    code = main(["--compare"], runner=runner)
    assert code == 0
    assert len(calls) == 40  # 8 scenarios x 5 modes
    assert list((tmp_path / "runs").glob("run-*.json"))


def test_repeat_prints_pass_rate(capsys):
    def runner(scenario, mode, **kw):
        return _result(scenario.name, mode)

    main(["--scenario", "simple-restart", "--repeat", "3"], runner=runner)
    assert "3/3" in capsys.readouterr().out


def test_missing_key_exit_code_2(capsys):
    def runner(scenario, mode, **kw):
        raise MissingKeyError("OPENROUTER_API_KEY is not set")

    assert main([], runner=runner) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_unknown_scenario_exit_code_2(capsys):
    assert main(["--scenario", "nope"], runner=lambda *a, **k: None) == 2
    err = capsys.readouterr().err
    assert "Unknown scenario" in err
    assert "simple-restart" in err
    assert "deploy-regression" in err


def test_repeat_zero_exits_code_2():
    with pytest.raises(SystemExit) as exc_info:
        main(["--repeat", "0"], runner=lambda *a, **k: _result())
    assert exc_info.value.code == 2


def test_repeat_negative_exits_code_2():
    with pytest.raises(SystemExit) as exc_info:
        main(["--repeat", "-1"], runner=lambda *a, **k: _result())
    assert exc_info.value.code == 2


def test_runner_keyerror_propagates():
    def runner(scenario, mode, **kw):
        raise KeyError("some_internal_error")

    with pytest.raises(KeyError) as exc_info:
        main(["--scenario", "simple-restart"], runner=runner)
    assert "some_internal_error" in str(exc_info.value)


def test_trace_flag_passes_tracer_to_runner():
    from jev_demo_triage.trace import Tracer

    seen = []

    def runner(scenario, mode, **kw):
        seen.append(kw)
        return _result(scenario.name, mode)

    assert main(["--scenario", "simple-restart", "--trace"], runner=runner) == 0
    assert isinstance(seen[0]["tracer"], Tracer)


def test_no_trace_flag_calls_runner_without_tracer():
    seen = []

    def runner(scenario, mode, **kw):
        seen.append(kw)
        return _result(scenario.name, mode)

    main(["--scenario", "simple-restart"], runner=runner)
    assert seen == [{"gate_policy": "tuned"}]


def _capture():
    seen = []

    def runner(scenario, mode, **kw):
        seen.append(kw)
        return _result(scenario.name, mode)

    return seen, runner


def test_gate_policy_flag_reaches_runner():
    seen, runner = _capture()
    assert main(["--mode", "gate", "--gate-policy", "default"], runner=runner) == 0
    assert seen[0]["gate_policy"] == "default"


def test_gate_policy_defaults_to_tuned():
    seen, runner = _capture()
    main(["--mode", "gate"], runner=runner)
    assert seen[0]["gate_policy"] == "tuned"


def test_compare_passes_chosen_policy_to_every_cell(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen, runner = _capture()
    main(["--compare", "--gate-policy", "default"], runner=runner)
    assert seen and all(kw["gate_policy"] == "default" for kw in seen)


def test_invalid_gate_policy_exits_code_2():
    with pytest.raises(SystemExit) as info:
        main(["--gate-policy", "lax"], runner=lambda *a, **k: _result())
    assert info.value.code == 2


def test_llm_gate_mode_accepted_and_passed_to_runner():
    calls = []

    def runner(scenario, mode, **kw):
        calls.append((mode, kw["gate_policy"]))
        return _result(scenario.name, mode)

    assert main(["--mode", "llm-gate", "--gate-policy", "default"], runner=runner) == 0
    assert calls == [("llm-gate", "default")]


def test_compare_includes_llm_gate_for_every_scenario(tmp_path, monkeypatch):
    from jev_demo_triage.scenarios import SCENARIOS

    monkeypatch.chdir(tmp_path)
    calls = []

    def runner(scenario, mode, **kw):
        calls.append((scenario.name, mode))
        return _result(scenario.name, mode)

    main(["--compare"], runner=runner)
    assert sorted(n for n, m in calls if m == "llm-gate") == sorted(SCENARIOS)


def test_invalid_mode_exits_code_2():
    with pytest.raises(SystemExit) as info:
        main(["--mode", "llm_gate"], runner=lambda *a, **k: _result())
    assert info.value.code == 2
