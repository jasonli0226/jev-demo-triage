import json
from datetime import datetime

from jev_router_bench.store import latest_file, write_json


def test_write_json_uses_prefix_and_timestamp(tmp_path):
    path = write_json("calib", {"a": 1}, tmp_path / "runs", now=datetime(2026, 9, 24, 10, 5, 3))
    assert path.name == "calib-20260924-100503.json"
    assert json.loads(path.read_text()) == {"a": 1}


def test_latest_file_picks_newest_name(tmp_path):
    assert latest_file("calib", tmp_path) is None
    write_json("calib", {}, tmp_path, now=datetime(2026, 9, 24, 10, 0, 0))
    newest = write_json("calib", {}, tmp_path, now=datetime(2026, 9, 25, 9, 0, 0))
    write_json("router-route", {}, tmp_path, now=datetime(2026, 9, 26, 9, 0, 0))
    assert latest_file("calib", tmp_path) == newest
