"""Retiring findings whose signal no longer holds.

A finding is a claim about the present ("this tool is failing"). Nothing
re-checked open findings, so they accumulated as permanent accusations --
including ones that could never resolve.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from knowledge_bases.tools.models import SelfImprovementKind
from knowledge_bases.tools.self_improvement_apply import (
    STALE_MIN_CALLS,
    STALE_RECOVERY_METRIC,
    _stale_reason,
)


class FakeStore:
    def __init__(self, metrics):
        self.metrics = metrics

    def skill_metrics_for(self, skill, since):
        return self.metrics


class FakeRouting:
    def __init__(self, metrics):
        self.metrics = metrics

    def routing_metrics_for(self, knowledge_base, since):
        return self.metrics


class FakeGraph:
    def __init__(self, registered=True):
        self.registered = registered

    def run(self, query, **params):
        return [{"c": 1 if self.registered else 0}]


SINCE = datetime.now(timezone.utc) - timedelta(days=7)


def _reason(kind, target, graph, store, routing):
    return _stale_reason(kind, target, SINCE, graph, store, routing)


def test_a_finding_against_a_vanished_tool_is_retired():
    """`hybrid_search`'s 100% error rate was five spans under a renamed key.
    No traffic will ever arrive under the old name to dilute it, so the metric
    is frozen at 100% forever -- only retirement can clear it."""
    reason = _reason(
        SelfImprovementKind.architecture, "hybrid_search",
        FakeGraph(registered=False), FakeStore({}), FakeRouting({}),
    )

    assert reason is not None
    assert "no longer a registered tool" in reason


def test_a_recovered_tool_is_retired():
    reason = _reason(
        SelfImprovementKind.tool_wording, "a.b", FakeGraph(),
        FakeStore({"call_count": 50, "error_rate": 0.0, "user_correction_count": 0}),
        FakeRouting({}),
    )

    assert reason is not None
    assert "recovered" in reason


def test_a_tool_still_failing_keeps_its_finding():
    assert _reason(
        SelfImprovementKind.tool_wording, "a.b", FakeGraph(),
        FakeStore({"call_count": 50, "error_rate": 0.4, "user_correction_count": 0}),
        FakeRouting({}),
    ) is None


def test_one_lucky_call_is_not_evidence_of_recovery():
    """Retiring on a single clean call would clear a finding that is about to
    be re-filed by the next failure."""
    assert _reason(
        SelfImprovementKind.tool_wording, "a.b", FakeGraph(),
        FakeStore({
            "call_count": STALE_MIN_CALLS - 1, "error_rate": 0.0,
            "user_correction_count": 0,
        }),
        FakeRouting({}),
    ) is None


def test_the_recovery_bar_sits_below_the_filing_threshold():
    """Otherwise a target hovering at the filing threshold would be retired
    and immediately re-filed, forever."""
    from knowledge_bases.tools.self_improvement import ERROR_RATE_THRESHOLD

    assert STALE_RECOVERY_METRIC < ERROR_RATE_THRESHOLD


def test_user_corrections_count_against_recovery():
    """A tool that stopped erroring but is still being corrected has not
    recovered; it just fails more quietly."""
    assert _reason(
        SelfImprovementKind.tool_wording, "a.b", FakeGraph(),
        FakeStore({"call_count": 20, "error_rate": 0.0, "user_correction_count": 8}),
        FakeRouting({}),
    ) is None


def test_routing_findings_are_judged_on_routing_metrics():
    reason = _reason(
        SelfImprovementKind.routing_capability_wording, "methodology", FakeGraph(),
        FakeStore({}), FakeRouting({"trace_count": 40, "low_confidence_share": 0.0}),
    )

    assert reason is not None
    assert "recovered" in reason


def test_a_routing_finding_still_ambiguous_is_kept():
    assert _reason(
        SelfImprovementKind.routing_capability_wording, "methodology", FakeGraph(),
        FakeStore({}), FakeRouting({"trace_count": 40, "low_confidence_share": 0.5}),
    ) is None


def test_a_finding_with_no_target_is_left_alone():
    """Better a stale finding than retiring one we cannot actually measure."""
    assert _reason(
        SelfImprovementKind.architecture, None, FakeGraph(), FakeStore({}),
        FakeRouting({}),
    ) is None
