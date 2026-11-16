"""The SPA's boot path is executed headlessly by scripts/spa_smoke.mjs.

`node --check` catches syntax errors only; the harness runs the page's script
against a DOM stub and drives its views, asserting the graph explorer
constructs and merges data -- the regression guard for boot-time runtime
errors in web/index.html, which have shipped to users twice.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_the_spa_boots_and_exercises_its_views():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; run scripts/spa_smoke.mjs manually")

    script = Path(__file__).resolve().parents[1] / "scripts" / "spa_smoke.mjs"
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, (
        f"SPA smoke test failed:\n{result.stdout}\n{result.stderr}"
    )
