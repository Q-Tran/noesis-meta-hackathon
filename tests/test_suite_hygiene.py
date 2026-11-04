"""Every file pytest collects must actually contain tests.

This suite accumulated module-level scripts named ``test_*.py`` -- they
connected to databases, downloaded embedding models, or shelled out at import
time, and collected zero test items (one even shelled into a hardcoded path
from a previous machine). They were pruned; this test is the fence that keeps
the pattern from growing back. Debugging entry points belong in ``scripts/``.
"""
from __future__ import annotations

import ast
import pathlib


def test_every_collected_file_defines_tests():
    tests_dir = pathlib.Path(__file__).resolve().parent
    empty = []
    for path in sorted(tests_dir.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        defines = any(
            (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            )
            or (isinstance(node, ast.ClassDef) and node.name.startswith("Test"))
            for node in tree.body
        )
        if not defines:
            empty.append(path.name)

    assert not empty, (
        "test files that define no tests -- module-level scripts belong in "
        f"scripts/, not the suite: {empty}"
    )
