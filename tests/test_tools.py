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
