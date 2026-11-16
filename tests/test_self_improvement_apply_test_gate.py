"""`run_test_gate` is the extra bar `new_tool`-kind applies are held to
(LLM-authored logic, not a single string replacement) -- it must run the
real test suite in a real subprocess and never silently succeed. These
tests point it at a throwaway fixture directory via a monkeypatched
`REPO_ROOT`, real `pytest` subprocess calls, never the actual Noesis
`tests/` tree."""
from __future__ import annotations

import knowledge_bases.tools.self_improvement_apply as sia
from knowledge_bases.tools.self_improvement_apply import GateFailure, run_test_gate


def _write_test_file(tests_dir, name, content):
    (tests_dir / name).write_text(content)


def test_run_test_gate_passes_for_a_passing_suite(tmp_path, monkeypatch):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write_test_file(tests_dir, "test_ok.py", "def test_ok():\n    assert True\n")
    monkeypatch.setattr(sia, "REPO_ROOT", tmp_path)

    run_test_gate()  # must not raise


def test_run_test_gate_raises_on_a_failing_test(tmp_path, monkeypatch):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write_test_file(tests_dir, "test_broken.py", "def test_broken():\n    assert False\n")
    monkeypatch.setattr(sia, "REPO_ROOT", tmp_path)

    try:
        run_test_gate()
    except GateFailure as exc:
        assert "Test suite failed" in str(exc)
        assert "test_broken" in exc.output
    else:
        raise AssertionError("Expected GateFailure")


def test_run_test_gate_raises_on_a_syntax_error_in_generated_code(tmp_path, monkeypatch):
    """A `new_tool` draft that produced code with a collection-time error
    (e.g. broken syntax that slipped past `_validate_handler_source`
    somehow, or an import error) must fail the gate too, not just an
    assertion failure."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write_test_file(tests_dir, "test_collection_error.py", "def test_x(:\n    pass")
    monkeypatch.setattr(sia, "REPO_ROOT", tmp_path)

    try:
        run_test_gate()
    except GateFailure as exc:
        assert "Test suite failed" in str(exc)
    else:
        raise AssertionError("Expected GateFailure")


def test_run_test_gate_raises_on_timeout(tmp_path, monkeypatch):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write_test_file(
        tests_dir, "test_slow.py",
        "import time\ndef test_slow():\n    time.sleep(5)\n",
    )
    monkeypatch.setattr(sia, "REPO_ROOT", tmp_path)

    try:
        run_test_gate(timeout=1)
    except GateFailure as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("Expected GateFailure")
