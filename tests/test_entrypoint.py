from importlib.metadata import entry_points

from jev_demo_triage.__main__ import main


def test_console_script_points_at_main():
    scripts = {ep.name: ep for ep in entry_points(group="console_scripts")}
    assert "jev-demo-triage" in scripts
    assert scripts["jev-demo-triage"].value == "jev_demo_triage.__main__:main"
    assert scripts["jev-demo-triage"].load() is main
