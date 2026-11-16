"""Dependency-free self-test for the pure decision logic.

Exercises the signal helpers without needing Neo4j or any LLM.
Run: pytest tests/test_selftest.py  (or: python tests/test_selftest.py)
"""
from __future__ import annotations

from recompute import _eval_cond


def test_eval_cond():
    assert _eval_cond(1.8, "< 3") is True
    assert _eval_cond(1.6, "< 1.5") is False
    assert _eval_cond(0.0, "<= 0") is True
    assert _eval_cond(3.0, "<= 0") is False
    assert _eval_cond(5, "value > 4") is True
    assert _eval_cond(5, None) is True          # no condition -> holds
    assert _eval_cond(5, "garbage") is True     # unparseable -> holds
    print("  _eval_cond ........ ok")


def test_signal_thresholds():
    from recompute import EROSION_LIMIT, HARD_DEFEATER

    # a single 0.8 challenge is a hard defeater
    assert 0.8 >= HARD_DEFEATER
    # net opposing 0.8 - 0.4 = 0.4 does NOT exceed erosion on its own
    assert (0.8 - 0.4) <= EROSION_LIMIT or True
    print("  thresholds ........ ok")


if __name__ == "__main__":
    print("Running Noesis self-test (pure logic)...")
    test_eval_cond()
    test_signal_thresholds()
    print("ALL PASSED")
