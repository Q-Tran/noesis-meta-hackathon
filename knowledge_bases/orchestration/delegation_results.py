"""What each expert produced for one delegation, as structured data.

Until now a delegated expert's findings came back only as prose, through the
harness's own subagent mechanism, and the orchestrator re-read that prose to
synthesize. Nothing structured survived the hand-off: a second expert could not
build on the first's resolved entities without them being re-typed into a brief,
and a resumed session had nothing to recover.

This is the "pass variables between agents" primitive, and it is deliberately
two-tiered, because the two tiers have very different guarantees:

**Tier 1 -- derived, and therefore guaranteed.** Every dispatch already writes a
`Span` carrying `trace_id`, `expert_key`, the skill called, its status, and its
serialized output (`knowledge_bases.tools.tracing.record_tool_span`). Since
all experts dispatch through one server, `derive_delegation_results` can rebuild
"what did this expert actually do and find" from spans alone, server-side. No
expert has to cooperate; nothing can be skipped, forgotten, or fabricated; and
it works retroactively on every trace already in the graph.

**Tier 2 -- declared, and therefore optional.** Spans cannot contain an expert's
*judgement*: how confident it is, what it decided it could not settle, why it
stopped. `record_delegation_result` lets an expert add exactly that. When it
does not, the derived record still stands and is marked `declared: false` --
which is itself a per-expert compliance signal that feeds the existing
self-improvement loop, so non-compliance corrects itself rather than needing a
human to notice.

The honest limit: an expert can be *made* to leave a factual record, but not to
volunteer judgement. Tier 1 is what makes that acceptable.
"""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from graph import Graph

# Entity-ish keys a skill result commonly uses for the thing it resolved.
# Checked in order, so a payload with both `key` and `id` reports `key`.
_ENTITY_KEY_FIELDS = ("key", "id", "name", "canonical_name", "term", "metric", "project_id")
_ENTITY_LABEL_FIELDS = ("label", "type", "entity_type", "kind", "knowledge_base")

MAX_SUMMARY_CHARS = 2000
MAX_ENTITIES = 50


class DeclaredDelegationResult(BaseModel):
    """The judgement half of a delegation result -- what only the expert knows."""

    trace_id: str
    expert_key: str
    summary: str = ""
    confidence: float = Field(default=0.5, ge=0, le=1)
    unresolved: list[str] = Field(default_factory=list)
    stopped_because: str = ""


def _entities_from(payload: Any) -> list[tuple[str, str]]:
    """Best-effort `(label, key)` pairs from a skill's serialized output.

    Structural, not semantic: it walks dicts/lists looking for the identifier
    fields Noesis payloads actually use. Missing an entity only makes the
    derived record thinner, never wrong, so this stays deliberately simple
    rather than trying to understand every skill's shape.
    """
    found: list[tuple[str, str]] = []

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 4 or len(found) >= MAX_ENTITIES:
            return
        if isinstance(node, dict):
            key = next(
                (str(node[f]) for f in _ENTITY_KEY_FIELDS
                 if isinstance(node.get(f), (str, int))),
                None,
            )
            if key:
                label = next(
                    (str(node[f]) for f in _ENTITY_LABEL_FIELDS
                     if isinstance(node.get(f), str)),
                    "unknown",
                )
                found.append((label, key))
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            for item in node:
                walk(item, depth + 1)

    walk(payload)
    # Preserve order, drop duplicates.
    seen, unique = set(), []
    for pair in found:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    return unique[:MAX_ENTITIES]


def derive_delegation_results(trace_id: str, graph: Graph) -> list[dict[str, Any]]:
    """Rebuild per-expert delegation results from this trace's spans.

    Tier 1: no expert participation, unforgeable, retroactive.
    """
    from knowledge_bases.tools.trace_store import TraceStore

    spans = TraceStore(graph).get_trace_spans(trace_id)
    by_expert: dict[str, dict[str, Any]] = {}
    for span in spans:
        expert_key = span.get("expert_key") or "unattributed"
        record = by_expert.setdefault(expert_key, {
            "trace_id": trace_id, "expert_key": expert_key,
            "tools_called": [], "errors": [], "entities": [],
            "span_count": 0, "declared": False,
        })
        record["span_count"] += 1
        skill = span.get("skill")
        if skill and skill not in record["tools_called"]:
            record["tools_called"].append(skill)
        if span.get("status") == "error" and span.get("error"):
            record["errors"].append({"skill": skill, "error": span["error"]})
        for label, key in _entities_from(span.get("raw_output")):
            entity = {"label": label, "key": key}
            if entity not in record["entities"]:
                record["entities"].append(entity)

    declared = {
        row["expert_key"]: row for row in _load_declared(trace_id, graph)
    }
    for expert_key, record in by_expert.items():
        record["entities"] = record["entities"][:MAX_ENTITIES]
        if expert_key in declared:
            record.update(declared.pop(expert_key))
            record["declared"] = True

    # A declared result with no spans is unusual but real: tracing is
    # best-effort and swallows write failures
    # (`knowledge_bases.tools.tracing`), and an expert can conclude without
    # calling a tool. Dropping it here would lose the only record of that
    # delegation, so surface it with empty derived facts rather than silently.
    for expert_key, row in declared.items():
        by_expert[expert_key] = {
            "trace_id": trace_id, "expert_key": expert_key,
            "tools_called": [], "errors": [], "entities": [],
            "span_count": 0, "declared": True, **row,
        }
    return sorted(by_expert.values(), key=lambda r: r["expert_key"])


def _load_declared(trace_id: str, graph: Graph) -> list[dict[str, Any]]:
    rows = graph.run(
        "MATCH (r:DelegationResult {trace_id: $trace_id}) "
        "RETURN r.expert_key AS expert_key, r.summary AS summary, "
        "r.confidence AS confidence, r.unresolved AS unresolved, "
        "r.stopped_because AS stopped_because, r.recorded_at AS recorded_at "
        "ORDER BY r.expert_key",
        trace_id=trace_id,
    )
    return [dict(row) for row in rows]


def record_delegation_result(arguments: dict[str, Any], graph: Graph) -> dict[str, Any]:
    """Tier 2: an expert adds its own judgement for one delegation.

    Stores the entity/skill facts alongside it too, derived from the same spans,
    so a single node carries the whole picture -- which is what the session
    digest and the orchestrator's synthesis read.
    """
    declared = DeclaredDelegationResult.model_validate(arguments)
    derived = next(
        (r for r in derive_delegation_results(declared.trace_id, graph)
         if r["expert_key"] == declared.expert_key),
        {"tools_called": [], "entities": []},
    )
    entities = derived.get("entities", [])
    result_id = f"{declared.trace_id}:{declared.expert_key}"
    graph.run(
        "MERGE (r:DelegationResult {id: $id}) "
        "ON CREATE SET r.created_at = datetime() "
        "SET r.trace_id = $trace_id, r.expert_key = $expert_key, "
        "r.summary = $summary, r.confidence = $confidence, "
        "r.unresolved = $unresolved, r.stopped_because = $stopped_because, "
        "r.tools_called = $tools_called, "
        "r.entity_labels = $entity_labels, r.entity_keys = $entity_keys, "
        "r.recorded_at = datetime() "
        "WITH r "
        "OPTIONAL MATCH (t:RoutingTrace {id: $trace_id}) "
        "FOREACH (_ IN CASE WHEN t IS NULL THEN [] ELSE [1] END | "
        "  MERGE (t)-[ri:RESULTED_IN]->(r) ON CREATE SET ri.created_at = datetime())",
        id=result_id,
        trace_id=declared.trace_id,
        expert_key=declared.expert_key,
        summary=declared.summary[:MAX_SUMMARY_CHARS],
        confidence=declared.confidence,
        unresolved=declared.unresolved,
        stopped_because=declared.stopped_because,
        tools_called=derived.get("tools_called", []),
        entity_labels=[e["label"] for e in entities],
        entity_keys=[e["key"] for e in entities],
    )
    return {
        "recorded": result_id,
        "expert_key": declared.expert_key,
        "entities_captured": len(entities),
        "tools_called": derived.get("tools_called", []),
    }


def get_delegation_results(arguments: dict[str, Any], graph: Graph) -> dict[str, Any]:
    """Every expert's result for one trace, derived plus declared.

    What the orchestrator reads before synthesizing, and what a dependent
    delegation reads to build on an earlier expert's findings instead of having
    them re-typed into its brief.
    """
    trace_id = arguments["trace_id"]
    results = derive_delegation_results(trace_id, graph)
    return {
        "trace_id": trace_id,
        "results": results,
        "experts_without_declared_results": [
            r["expert_key"] for r in results if not r["declared"]
        ],
    }
