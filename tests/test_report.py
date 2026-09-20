import json
from datetime import datetime

from jev_demo_triage.agent import RunResult
from jev_demo_triage.metrics import GlmUsage, RunMetrics
from jev_demo_triage.report import format_table, summarize_repeats, write_json
from jev_demo_triage.tools import Action


def _result(scenario="s1", mode="baseline", passed=True, jev_calls=0, cost=0.0):
    m = RunMetrics(steps=3, glm=GlmUsage(3, 500, 50), jev_calls=jev_calls,
                   jev_tokens=jev_calls * 300, jev_cost_usd=cost, wall_seconds=2.345)
    return RunResult(scenario, mode, passed, (Action("restart_service", {"service": "a"}),), "done", m, None)


def test_format_table_has_header_and_rows():
    text = format_table([_result(), _result("s2", "tool", False, 2, 0.0000252)])
    lines = text.splitlines()
    assert "scenario" in lines[0] and "jev_cost" in lines[0]
    assert any("s1" in ln and "baseline" in ln and "PASS" in ln for ln in lines)
    assert any("s2" in ln and "tool" in ln and "FAIL" in ln for ln in lines)
    assert "2.3s" in text


def test_summarize_repeats_reports_pass_rate_per_mode():
    results = [_result(mode="baseline", passed=True), _result(mode="baseline", passed=False),
               _result(mode="tool", passed=True), _result(mode="tool", passed=True)]
    text = summarize_repeats(results)
    assert "baseline" in text and "1/2" in text
    assert "tool" in text and "2/2" in text


def test_write_json_roundtrips(tmp_path):
    path = write_json([_result()], tmp_path, now=datetime(2026, 9, 20, 12, 30, 5))
    assert path.name == "run-20260920-123005.json"
    data = json.loads(path.read_text())
    assert data[0]["scenario"] == "s1"
    assert data[0]["actions"] == [{"name": "restart_service", "args": {"service": "a"}}]
    assert data[0]["metrics"]["glm"]["input_tokens"] == 500


def test_errored_run_shows_error_and_plain_failure_shows_fail():
    errored = _result("s3", "gate", False)
    errored = RunResult(errored.scenario, errored.mode, False, (), "", errored.metrics, "recursion limit reached")
    lines = format_table([errored, _result("s4", "tool", False)]).splitlines()
    assert any("s3" in ln and "ERROR" in ln and "FAIL" not in ln for ln in lines)
    assert any("s4" in ln and "FAIL" in ln and "ERROR" not in ln for ln in lines)


def test_write_json_records_gate_policy_per_run(tmp_path):
    from dataclasses import replace

    runs = [_result(mode="baseline"), replace(_result(mode="gate"), gate_policy="tuned")]
    data = json.loads(write_json(runs, tmp_path, now=datetime(2026, 9, 20, 12, 30, 5)).read_text())
    assert [d["gate_policy"] for d in data] == [None, "tuned"]


def _llm_result():
    m = RunMetrics(steps=3, glm=GlmUsage(3, 500, 50), jev_calls=0, jev_tokens=0, jev_cost_usd=0.0,
                   wall_seconds=2.0, classifier_seconds=1.34, llm_gate_calls=4,
                   llm_gate_tokens=700, llm_gate_cost_usd=0.000512)
    return RunResult("s9", "gate", True, (), "done", m, None)


def test_table_has_new_columns_and_llm_gate_values():
    lines = format_table([_llm_result()]).splitlines()
    for col in ("llm_gate_calls", "llm_gate_cost", "clf_s"):
        assert col in lines[0]
    assert lines[0].split()[-3:] == ["llm_gate_calls", "llm_gate_cost", "clf_s"]
    assert lines[1].split()[-3:] == ["4", "$0.000512", "1.3s"]


def test_tokens_column_includes_glm_jev_and_llm_gate_tokens():
    m = _llm_result().metrics
    r = RunResult("s9", "gate", True, (), "done", RunMetrics(
        steps=3, glm=m.glm, jev_calls=1, jev_tokens=300, jev_cost_usd=0.0, wall_seconds=2.0,
        llm_gate_tokens=700), None)
    row = format_table([r]).splitlines()[1].split()
    header = format_table([r]).splitlines()[0].split()
    assert row[header.index("tokens")] == str(500 + 50 + 300 + 700)


def test_write_json_includes_new_metric_fields(tmp_path):
    path = write_json([_llm_result()], tmp_path, now=datetime(2026, 9, 20, 12, 30, 5))
    m = json.loads(path.read_text())[0]["metrics"]
    assert m["classifier_seconds"] == 1.34
    assert (m["llm_gate_calls"], m["llm_gate_tokens"], m["llm_gate_cost_usd"]) == (4, 700, 0.000512)
