from jev_demo_triage.__main__ import main as triage_main
from jev_router_bench import cli
from jev_router_bench.playground import COMMANDS, main


def test_dispatches_to_subcommand_with_remaining_args():
    seen = []
    commands = {"triage": lambda argv: seen.append(("triage", argv)) or 0,
                "router": lambda argv: seen.append(("router", argv)) or 5}
    assert main(["triage", "--mode", "gate"], commands) == 0
    assert main(["router", "route"], commands) == 5
    assert seen == [("triage", ["--mode", "gate"]), ("router", ["route"])]


def test_unknown_or_missing_command(capsys):
    assert main(["nope"], {}) == 2
    assert main([], {}) == 2
    assert "usage" in capsys.readouterr().err


def test_help(capsys):
    assert main(["--help"], {}) == 0
    assert "triage" in capsys.readouterr().out


def test_default_commands():
    assert COMMANDS == {"triage": triage_main, "router": cli.main}
