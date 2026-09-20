import pytest

from jev_demo_triage.tools import Action, RunContext, World, make_tools


def _tools(ctx=None):
    ctx = ctx or RunContext()
    world = World(
        logs={"api": "OOM at 12:01", "db": "ok"},
        metrics={"api": "mem=98%"},
    )
    return ctx, {t.name: t for t in make_tools(world, ctx)}


def test_tool_names():
    _, tools = _tools()
    assert set(tools) == {
        "read_logs", "get_metrics", "run_shell",
        "restart_service", "rollback_deploy", "escalate_to_human",
    }


def test_read_tools_return_scripted_text_and_record_nothing():
    ctx, tools = _tools()
    assert tools["read_logs"].invoke({"service": "api"}) == "OOM at 12:01"
    assert tools["get_metrics"].invoke({"service": "api"}) == "mem=98%"
    assert ctx.actions == []


def test_unknown_service_is_reported_not_raised():
    _, tools = _tools()
    assert "no logs" in tools["read_logs"].invoke({"service": "zzz"}).lower()


def test_state_changing_tools_record_actions():
    ctx, tools = _tools()
    tools["run_shell"].invoke({"command": "ls /tmp"})
    tools["restart_service"].invoke({"service": "api"})
    tools["rollback_deploy"].invoke({"service": "api"})
    tools["escalate_to_human"].invoke({"reason": "unclear"})
    assert ctx.actions == [
        Action("run_shell", {"command": "ls /tmp"}),
        Action("restart_service", {"service": "api"}),
        Action("rollback_deploy", {"service": "api"}),
        Action("escalate_to_human", {"reason": "unclear"}),
    ]


def test_executed_filters_by_name():
    ctx, tools = _tools()
    tools["restart_service"].invoke({"service": "api"})
    tools["run_shell"].invoke({"command": "df -h"})
    assert [a.name for a in ctx.executed("run_shell")] == ["run_shell"]


def test_contexts_are_isolated():
    ctx1, tools1 = _tools()
    ctx2, _ = _tools()
    tools1["restart_service"].invoke({"service": "api"})
    assert ctx2.actions == []
    assert len(ctx1.actions) == 1


def _shell_tool(shell, ctx=None):
    ctx = ctx or RunContext()
    world = World(logs={}, metrics={}, shell=shell)
    return ctx, {t.name: t for t in make_tools(world, ctx)}["run_shell"]


def test_world_without_shell_falls_back_to_mock_text():
    ctx, tools = _tools()
    assert World(logs={}, metrics={}).shell == {}
    assert tools["run_shell"].invoke({"command": "df -h"}) == "[mock] executed: df -h"
    assert ctx.actions == [Action("run_shell", {"command": "df -h"})]


def test_matching_key_returns_scripted_text_and_records_action():
    ctx, run = _shell_tool({"df": "Filesystem 99%"})
    assert run.invoke({"command": "df -h"}) == "Filesystem 99%"
    assert ctx.actions == [Action("run_shell", {"command": "df -h"})]


def test_first_matching_key_in_insertion_order_wins():
    _, run = _shell_tool({"du": "first", "df": "second"})
    assert run.invoke({"command": "df -h; du -sh /"}) == "first"
    _, run = _shell_tool({"df": "second", "du": "first"})
    assert run.invoke({"command": "df -h; du -sh /"}) == "second"


@pytest.mark.parametrize("command", ["df -h", "sudo df -h /var", "ls; df", "df", "echo x && df", "ls | df", "x && df"])
def test_key_matches_whole_command_word(command):
    _, run = _shell_tool({"df": "SCRIPTED"})
    assert run.invoke({"command": command}) == "SCRIPTED"


@pytest.mark.parametrize("command", ["cdf -h", "echo undf", "dfx", "echo dfx", "ls /tmp", "ls /var/df/file", "ls --df", 'echo "df"'])
def test_key_does_not_match_inside_other_words(command):
    _, run = _shell_tool({"df": "SCRIPTED"})
    assert run.invoke({"command": command}) == f"[mock] executed: {command}"


def test_worlds_do_not_share_shell_state():
    a = World(logs={}, metrics={})
    b = World(logs={}, metrics={})
    assert a.shell is not b.shell


def test_key_du_does_not_match_dust():
    _, run = _shell_tool({"du": "SCRIPTED"})
    assert run.invoke({"command": "dust /var"}) != "SCRIPTED"
    assert run.invoke({"command": "du -sh /var"}) == "SCRIPTED"
