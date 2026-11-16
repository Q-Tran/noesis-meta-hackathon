"""MCP-callable handlers for the Skills & Tools skill catalog itself.

See `knowledge_bases.business_dictionary.mcp_tools` for the pattern. These
read execution telemetry (`Span`/`TraceOutcome` nodes, see `trace_store.py`)
from the same graph every other skill handler uses.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from graph import Graph
from knowledge_bases.tools import tool_admin
from knowledge_bases.tools.queries import get_span, get_trace, get_trace_metrics
from knowledge_bases.tools.self_improvement import (
    DEFAULT_LOOKBACK,
    MIN_CALLS,
    MIN_TRACES,
    collect_expert_correction_signals,
    collect_routing_gap_signals,
    collect_skill_health_signals,
    draft_suggestions,
    file_suggestions,
    finding_key,
)
from knowledge_bases.tools.self_improvement_apply import (
    ApplyError,
    apply_finding_by_key,
    reevaluate_applied_findings,
    retire_stale_findings,
)
from knowledge_bases.tools.trace_store import TraceStore



def _versioning_repository(graph: Graph):
    """The installed versioning engine, or a refusal.

    Self-improvement does not just compute a suggestion; it FILES one as a versioned
    `ReviewFinding` for a human to accept or reject. With no versioning engine
    installed there is nowhere to file it, and returning a suggestion the caller
    believes was filed is worse than refusing -- the whole point of the
    propose-confirm posture is that a suggestion survives until somebody rules on it.
    """
    from knowledge_bases.registry import capability

    repository = capability("versioning.repository")
    if repository is None:
        raise RuntimeError(
            "self-improvement files suggestions as versioned ReviewFinding "
            "artifacts, which needs an installed package providing the "
            "'versioning.repository' capability (the methodology ontology); "
            "this product ships without one"
        )
    return repository(graph)

def override_skill_input_schema_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return tool_admin.override_skill_input_schema(arguments, graph)


def retire_skill_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return tool_admin.retire_skill(arguments, graph)


def reactivate_skill_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return tool_admin.reactivate_skill(arguments, graph)


def get_span_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return get_span(arguments["span_id"], graph)


def get_trace_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return get_trace(arguments["trace_id"], graph)


def get_trace_metrics_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return get_trace_metrics(arguments["trace_id"], graph)


def propose_self_improvements_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Analyze recent skill/routing traces and file reviewable suggestions
    as `ReviewFinding`s in the `noesis-self` methodology project, `status: "open"`.

    Filing itself changes nothing -- see
    `knowledge_bases.tools.self_improvement_apply.apply_finding_by_key`
    (the `apply_self_improvement` tool) for the step that does, and
    `knowledge_bases.tools.self_improvement` for the drafting pipeline.
    Each entry in `filed` includes its own `finding_key`, ready to pass
    straight to `apply_self_improvement` with no client-side reconstruction.
    """
    from knowledge_bases.orchestration.repository import RoutingRepository

    lookback_days = arguments.get("lookback_days")
    since = datetime.now(timezone.utc) - (
        timedelta(days=lookback_days) if lookback_days else DEFAULT_LOOKBACK
    )
    min_calls = arguments.get("min_calls", MIN_CALLS)
    min_traces = arguments.get("min_traces", MIN_TRACES)
    dry_run = arguments.get("dry_run", False)

    store = TraceStore(graph)
    skill_signals = collect_skill_health_signals(store, since, min_calls=min_calls)
    routing_signals = collect_routing_gap_signals(RoutingRepository(graph), min_traces=min_traces)
    expert_signals = collect_expert_correction_signals(store, since, min_traces=min_traces)
    analysis = draft_suggestions(skill_signals, routing_signals, expert_signals=expert_signals)

    filed = [] if dry_run else file_suggestions(analysis.suggestions, _versioning_repository(graph))
    return {
        "skill_health_signals": [s.model_dump(mode="json") for s in skill_signals],
        "routing_gap_signals": [s.model_dump(mode="json") for s in routing_signals],
        "expert_correction_signals": [s.model_dump(mode="json") for s in expert_signals],
        "suggestions": [s.model_dump(mode="json") for s in analysis.suggestions],
        "filed": [
            {**s.model_dump(mode="json"), "finding_key": finding_key(s)} for s in filed
        ],
        "dry_run": dry_run,
    }


def apply_self_improvement_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Apply one already-filed `ReviewFinding`'s suggested change -- fully
    automatic by explicit design (see
    `knowledge_bases.tools.self_improvement_apply` and
    `docs/architecture/self_improvement.md`). Wording kinds
    (`tool_wording`, `routing_capability_wording`, `expert_prompt_wording`)
    are versioned graph edits, gated against vocabulary pollution and
    routing-eval regressions; `new_tool` is a test-gated, git-committed
    source edit. Snapshots the metric the finding was drafted from before
    changing anything, so `reevaluate_self_improvements` can later measure
    whether it helped and revert if not.

    Returns an `{"error": ...}` payload (never raises through MCP) for an
    `architecture`-kind finding -- that kind is never auto-applied, always
    routed to `resolve_review_finding` (`methodology.resolve_review_finding`)
    instead -- or if the finding doesn't exist, isn't `"open"`, or the
    edit/commit fails. Never partially applies.
    """

    try:
        return apply_finding_by_key(
            arguments["finding_key"], repo=_versioning_repository(graph), store=TraceStore(graph),
        )
    except ApplyError as exc:
        return {"error": "ApplyError", "message": str(exc)}


def retire_stale_self_improvements_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Close open self-improvement findings whose signal no longer holds.

    A finding is a claim about the present ("this tool is failing"). Nothing
    re-checked open findings, so they accumulated as permanent accusations --
    including ones that can never resolve, because the failures sit under a
    tool key that no longer exists and will never receive traffic again.

    Retires only `open` findings; `applied` ones stay owned by
    `reevaluate_self_improvements`, which can revert their source edit.
    """

    return {"results": retire_stale_findings(repo=_versioning_repository(graph), store=TraceStore(graph))}


def reevaluate_self_improvements_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Re-measure every applied self-improvement finding whose cooldown
    has elapsed, and automatically keep or `git revert` each one based on
    whether its metric actually improved. This is the "did it help?" step;
    call it on a schedule or on demand, never speculatively mid-conversation
    (see `.opencode/agents/noesis-orchestrator.md`).
    """

    return {"results": reevaluate_applied_findings(repo=_versioning_repository(graph), store=TraceStore(graph))}
