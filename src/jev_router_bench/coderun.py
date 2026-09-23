"""Runs model-written code against hidden asserts in a subprocess.

This is NOT a sandbox: the code runs on the local machine as the current user. It is
limited only by `python -I`, a temporary working directory, an environment with no
secrets and a timeout.
"""

import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

CODE_TIMEOUT_SECONDS = 5.0
SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT")

_PYTHON_BLOCK = re.compile(r"```python[^\n]*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class CodeRunResult:
    passed: bool
    reason: str


def extract_code(text: str) -> str | None:
    """The single ```python block in `text`, or None when there are zero or several."""
    blocks = _PYTHON_BLOCK.findall(text)
    return blocks[0] if len(blocks) == 1 else None


def child_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return {k: source[k] for k in SAFE_ENV_KEYS if k in source}


def run_code(source: str, tests: str, timeout: float = CODE_TIMEOUT_SECONDS) -> CodeRunResult:
    # A nonce printed AFTER the hidden tests is the only proof they actually ran: model
    # code that calls sys.exit(0) or unittest.main() can end the process with rc 0 before
    # the hidden asserts execute, which would otherwise look like a pass.
    nonce = secrets.token_hex(16)
    with tempfile.TemporaryDirectory() as workdir:
        script = Path(workdir) / "solution_test.py"
        script.write_text(f"{source}\n\n{tests}\nprint({nonce!r})\n")
        proc = subprocess.Popen(
            [sys.executable, "-I", str(script)],
            cwd=workdir,
            env=child_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            stdout, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
            return CodeRunResult(False, "timeout")
    if proc.returncode == 0 and nonce in stdout.decode("utf-8", "replace"):
        return CodeRunResult(True, "ok")
    return CodeRunResult(False, "tests_failed")
