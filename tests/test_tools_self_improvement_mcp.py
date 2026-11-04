"""`propose_self_improvements_tool` is the MCP-facing entry point for the
self-improvement skill; it must never touch the graph catalog itself and
must respect dry_run."""
from knowledge_bases.tools import mcp_tools


class FakeGraph:
    def run(self, query, **params):
        return []


def test_propose_self_improvements_tool_respects_dry_run(monkeypatch):
    from knowledge_bases.tools.models import (
        RoutingGapSignal,
        SelfImprovementKind,
        SelfImprovementSuggestion,
    )

    monkeypatch.setattr(mcp_tools, "collect_skill_health_signals", lambda *a, **k: [])
    monkeypatch.setattr(
        mcp_tools, "collect_routing_gap_signals",
        lambda *a, **k: [RoutingGapSignal(
            knowledge_base="methodology", low_confidence_count=10,
            clarification_needed_count=6, sample_questions=["q"],
        )],
    )
    suggestion = SelfImprovementSuggestion(
        kind=SelfImprovementKind.routing_capability_wording, target="methodology",
        diagnosis="d", suggested_change="s", confidence=0.6,
    )
    from knowledge_bases.tools.models import SelfImprovementAnalysis
    monkeypatch.setattr(
        mcp_tools, "draft_suggestions", lambda *a, **k: SelfImprovementAnalysis(suggestions=[suggestion]),
    )

    called = {"file_suggestions": False}

    def fake_file_suggestions(*args, **kwargs):
        called["file_suggestions"] = True
        return []

    monkeypatch.setattr(mcp_tools, "file_suggestions", fake_file_suggestions)

    result = mcp_tools.propose_self_improvements_tool(
        {"dry_run": True, "min_traces": 1, "min_calls": 1}, FakeGraph(),
    )

    assert result["dry_run"] is True
    assert len(result["suggestions"]) == 1
    assert result["filed"] == []
    assert called["file_suggestions"] is False  # never files when dry_run


def test_propose_self_improvements_tool_includes_finding_key_in_filed_entries(monkeypatch):
    """`apply_self_improvement` needs `finding_key` to look the finding back
    up -- returning it here means the caller (an LLM-driven agent) never has
    to reconstruct `self_improvement::{kind}::{target}` itself."""
    from knowledge_bases.tools.models import SelfImprovementKind, SelfImprovementSuggestion

    monkeypatch.setattr(mcp_tools, "collect_skill_health_signals", lambda *a, **k: [])
    monkeypatch.setattr(mcp_tools, "collect_routing_gap_signals", lambda *a, **k: [])
    suggestion = SelfImprovementSuggestion(
        kind=SelfImprovementKind.tool_wording,
        target="business_dictionary.lookup_business_term",
        diagnosis="d", suggested_change="s", confidence=0.6,
    )
    from knowledge_bases.tools.models import SelfImprovementAnalysis
    monkeypatch.setattr(
        mcp_tools, "draft_suggestions", lambda *a, **k: SelfImprovementAnalysis(suggestions=[suggestion]),
    )
    monkeypatch.setattr(mcp_tools, "file_suggestions", lambda *a, **k: [suggestion])

    result = mcp_tools.propose_self_improvements_tool({}, FakeGraph())

    assert result["filed"][0]["finding_key"] == (
        "self_improvement::tool_wording::business_dictionary.lookup_business_term"
    )


def test_propose_self_improvements_tool_files_when_not_dry_run(monkeypatch):
    from knowledge_bases.tools.models import SelfImprovementAnalysis

    monkeypatch.setattr(mcp_tools, "collect_skill_health_signals", lambda *a, **k: [])
    monkeypatch.setattr(mcp_tools, "collect_routing_gap_signals", lambda *a, **k: [])
    monkeypatch.setattr(
        mcp_tools, "draft_suggestions", lambda *a, **k: SelfImprovementAnalysis(suggestions=[]),
    )

    called = {"file_suggestions": False}

    def fake_file_suggestions(*args, **kwargs):
        called["file_suggestions"] = True
        return []

    monkeypatch.setattr(mcp_tools, "file_suggestions", fake_file_suggestions)

    result = mcp_tools.propose_self_improvements_tool({}, FakeGraph())

    assert result["dry_run"] is False
    assert called["file_suggestions"] is True
