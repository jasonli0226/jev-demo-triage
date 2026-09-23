import time

from jev_router_bench.coderun import child_env, extract_code, run_code


def test_extract_single_python_block():
    text = "Here:\n```python\ndef f():\n    return 1\n```\nDone."
    assert extract_code(text) == "def f():\n    return 1\n"


def test_extract_rejects_zero_or_several_blocks():
    assert extract_code("no code") is None
    two = "```python\na = 1\n```\n```python\nb = 2\n```"
    assert extract_code(two) is None


def test_extract_ignores_non_python_fences():
    assert extract_code("```\nx = 1\n```") is None


def test_passing_code():
    result = run_code("def f():\n    return 2\n", "assert f() == 2")
    assert (result.passed, result.reason) == (True, "ok")


def test_failing_assert():
    result = run_code("def f():\n    return 3\n", "assert f() == 2")
    assert (result.passed, result.reason) == (False, "tests_failed")


def test_syntax_error_fails():
    result = run_code("def f(:\n", "assert f() == 2")
    assert (result.passed, result.reason) == (False, "tests_failed")


def test_timeout():
    result = run_code("while True:\n    pass\n", "", timeout=0.5)
    assert (result.passed, result.reason) == (False, "timeout")


def test_child_env_drops_secrets():
    env = child_env({"PATH": "/bin", "OPENROUTER_API_KEY": "sk", "HOME": "/h"})
    assert env == {"PATH": "/bin"}


def test_child_process_cannot_see_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    result = run_code("import os\n", "assert 'OPENROUTER_API_KEY' not in os.environ")
    assert result.passed


def test_sys_exit_before_asserts_does_not_pass():
    source = "import sys\ndef f():\n    return 1\nsys.exit(0)\n"
    result = run_code(source, "assert f() == 2")
    assert (result.passed, result.reason) == (False, "tests_failed")


def test_unittest_main_exit_before_hidden_asserts_does_not_pass():
    source = (
        "import unittest\n"
        "def f():\n"
        "    return 1\n"
        "class T(unittest.TestCase):\n"
        "    def test_f(self):\n"
        "        self.assertEqual(f(), 1)\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )
    result = run_code(source, "assert f() == 2")
    assert (result.passed, result.reason) == (False, "tests_failed")


def test_input_call_fails_fast_without_hanging():
    started = time.perf_counter()
    result = run_code("def f():\n    return input()\n", "assert f() == '2'")
    elapsed = time.perf_counter() - started
    assert (result.passed, result.reason) == (False, "tests_failed")
    assert elapsed < 1.0
