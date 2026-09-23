"""Timestamped JSON files in the (gitignored) runs directory."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def write_json(prefix: str, payload: Any, directory: Path, now: datetime | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    path = directory / f"{prefix}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def latest_file(prefix: str, directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob(f"{prefix}-[0-9]*.json"))
    return candidates[-1] if candidates else None
