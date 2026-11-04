"""Neo4j persistence for expert profiles and routing traces."""
from __future__ import annotations

import hashlib
from typing import Any

from graph import Graph
from knowledge_bases.orchestration.models import (
    ExpertProfile,
    ExpertResult,
    RoutingPlan,
)


_EXPERT_FIELDS = (
    "name", "description", "priority", "system_prompt", "entry_tools",
    "allowed_tools", "model_hint", "routable", "capabilities",
    "capability_overlays", "prompt_overlays", "prompt_prefix_overlays",
)

# Graph-side additions that a seed re-sync must never discard (see
# `_write_expert`'s `preserve_overlays`). Kept as one tuple so adding an
# overlay kind can't be half-implemented.
_OVERLAY_FIELDS = ("capability_overlays", "prompt_overlays", "prompt_prefix_overlays")

# Cypher fragment shared by every expert read. Base `capabilities` and the
# graph-side overlays are node properties; the `ExpertCapability` nodes are
# a derived projection of the effective list for graph browsing.
_EXPERT_RETURN = (
    "RETURN e.key AS key, e.name AS name, e.description AS description, "
    "e.priority AS priority, e.system_prompt AS system_prompt, "
    "e.entry_tools AS entry_tools, e.allowed_tools AS allowed_tools, "
    "e.model_hint AS model_hint, e.version AS version, e.active AS active, "
    "e.routable AS routable, e.capabilities AS capabilities, "
    "e.capability_overlays AS capability_overlays, e.prompt_overlays AS prompt_overlays, "
    "e.prompt_prefix_overlays AS prompt_prefix_overlays "
)


def _skill_allowed(skill_key: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern == skill_key or (pattern.endswith(".*") and skill_key.startswith(pattern[:-1])):
            return True
    return False


class RoutingRepository:
    def __init__(self, graph: Graph):
        self.graph = graph

    # --- experts --------------------------------------------------------
    def _expert_props(self, expert: ExpertProfile) -> dict[str, Any]:
        return {
            field: getattr(expert, field) for field in _EXPERT_FIELDS
            if field != "capabilities" and getattr(expert, field) is not None
        }

    def _write_expert(
        self, tx: Any, expert: ExpertProfile, reason: str, preserve_overlays: bool = False,
    ) -> int:
        """MERGE the live `KnowledgeBaseExpert` node, snapshot the new state
        as an `ExpertRevision` (chained via `SUPERSEDES`), and rebuild its
        `HAS_CAPABILITY` (from the *effective* capabilities) and `CAN_USE`
        edges. Returns the new version number. A definition identical to
        the current one is a no-op.

        `preserve_overlays` (catalog sync) keeps whatever graph-side
        overlays the node already carries when the incoming definition has
        none, so re-seeding never discards a self-improvement."""
        current = tx.run(
            "MATCH (e:KnowledgeBaseExpert {key: $key}) "
            "RETURN properties(e) AS props",
            key=expert.key,
        ).single()
        existing = current["props"] if current is not None else None
        if preserve_overlays and existing is not None:
            expert = expert.model_copy(update={
                field: getattr(expert, field) or list(existing.get(field) or [])
                for field in _OVERLAY_FIELDS
            })
        props = self._expert_props(expert)
        props["capabilities"] = list(expert.capabilities)
        if existing is not None:
            unchanged = all(
                (existing.get(field) or ([] if isinstance(props[field], list) else None)) == props[field]
                for field in props
            )
            if unchanged:
                return int(existing.get("version") or 1)
            version = int(existing.get("version") or 0) + 1
        else:
            version = 1
        # Separate statements on purpose: deleting and re-MERGEing the same
        # relationship type inside one statement lets the MERGE match edges
        # the DELETE has not yet removed, which then vanish with it.
        tx.run(
            "MERGE (e:KnowledgeBaseExpert {key: $key}) "
            "ON CREATE SET e.created_at = datetime() "
            "SET e += $props, e.version = $version, e.active = true, e.updated_at = datetime()",
            key=expert.key, props=props, version=version,
        ).consume()
        tx.run(
            "MATCH (e:KnowledgeBaseExpert {key: $key}) "
            "OPTIONAL MATCH (e)-[old:HAS_CAPABILITY]->(:ExpertCapability) DELETE old "
            "WITH DISTINCT e "
            "OPTIONAL MATCH (e)-[use:CAN_USE]->(:Tool) DELETE use",
            key=expert.key,
        ).consume()
        tx.run(
            "MATCH (e:KnowledgeBaseExpert {key: $key}) "
            "OPTIONAL MATCH (e)-[cur:CURRENT]->(previous:ExpertRevision) "
            "DELETE cur "
            "WITH e, previous "
            "CREATE (r:ExpertRevision {id: $key + ':' + toString($version), key: $key, "
            "version: $version, reason: $reason, created_at: datetime()}) "
            "SET r += $props "
            "CREATE (e)-[:CURRENT {created_at: datetime()}]->(r) "
            "FOREACH (_ IN CASE WHEN previous IS NULL THEN [] ELSE [1] END | "
            "  CREATE (r)-[:SUPERSEDES {created_at: datetime()}]->(previous))",
            key=expert.key, props=props, version=version, reason=reason,
        ).consume()
        tx.run(
            "MATCH (e:KnowledgeBaseExpert {key: $key}) "
            "UNWIND range(0, size($capabilities) - 1) AS position "
            "WITH e, position, $capabilities[position] AS statement "
            "MERGE (c:ExpertCapability {id: $key + ':' + toString(position)}) "
            "ON CREATE SET c.created_at = datetime() "
            "SET c.statement = statement, c.position = position "
            "MERGE (e)-[hc:HAS_CAPABILITY]->(c) "
            "ON CREATE SET hc.created_at = datetime()",
            key=expert.key, capabilities=expert.effective_capabilities,
        ).consume()
        tx.run(
            "MATCH (e:KnowledgeBaseExpert {key: $key}) "
            "MATCH (s:Tool {active: true}) "
            "WHERE any(p IN $patterns WHERE p = s.key "
            "  OR (p ENDS WITH '.*' AND s.key STARTS WITH left(p, size(p) - 1))) "
            "MERGE (e)-[cu:CAN_USE]->(s) "
            "ON CREATE SET cu.created_at = datetime()",
            key=expert.key, patterns=list(expert.allowed_tools),
        ).consume()
        return version

    def sync_experts(self, profiles: list[ExpertProfile], reason: str = "catalog_sync") -> None:
        def work(tx: Any) -> None:
            for profile in profiles:
                self._write_expert(tx, profile, reason, preserve_overlays=True)

        self.graph.execute_write(work)

    def revise_expert(self, expert: ExpertProfile, reason: str) -> int:
        """Write one changed expert definition as a new revision (what
        self-improvement calls instead of editing source)."""
        return self.graph.execute_write(lambda tx: self._write_expert(tx, expert, reason))

    def load_experts(self, include_unroutable: bool = False) -> list[ExpertProfile]:
        rows = self.graph.run(
            "MATCH (e:KnowledgeBaseExpert {active: true}) "
            "WHERE $all OR coalesce(e.routable, true) = true "
            + _EXPERT_RETURN + "ORDER BY e.priority",
            all=include_unroutable,
        )
        return [self._to_expert(row) for row in rows]

    def get_expert(self, key: str) -> ExpertProfile | None:
        rows = self.graph.run(
            "MATCH (e:KnowledgeBaseExpert {key: $key}) " + _EXPERT_RETURN, key=key,
        )
        return self._to_expert(rows[0]) if rows else None

    @staticmethod
    def _to_expert(row: dict[str, Any]) -> ExpertProfile:
        row = {key: value for key, value in dict(row).items() if value is not None}
        return ExpertProfile.model_validate(row)

    # `kind` as callers name it -> the node property it lives on.
    OVERLAY_FIELDS = {
        "capability": "capability_overlays",
        "prompt": "prompt_overlays",
        "prompt_prefix": "prompt_prefix_overlays",
    }

    def revert_expert_overlay(self, key: str, kind: str, value: str, reason: str) -> int:
        """Remove one overlay item as a new revision -- the graph-native
        "revert" for a wording change. `kind` is `capability`, `prompt`, or
        `prompt_prefix`."""
        expert = self.get_expert(key)
        if expert is None:
            raise ValueError(f"Unknown expert {key!r}")
        try:
            field = self.OVERLAY_FIELDS[kind]
        except KeyError as exc:
            raise ValueError(
                f"Unknown overlay kind {kind!r}; expected one of "
                f"{sorted(self.OVERLAY_FIELDS)}"
            ) from exc
        overlays = [o for o in getattr(expert, field) if o != value]
        return self.revise_expert(expert.model_copy(update={field: overlays}), reason)

    def expert_tools(self, expert: ExpertProfile) -> list[dict[str, Any]]:
        """Active `Tool`s this expert may call, resolved live from its
        `allowed_tools` patterns (not only the `CAN_USE` edges written at
        sync time), each with the schema a harness needs to call it."""
        from knowledge_bases.tools.repository import ToolRepository

        skills = ToolRepository(self.graph).load_tools()
        return [
            {
                "key": skill.key,
                "knowledge_base": skill.knowledge_base,
                "operation": skill.key.split(".", 1)[1] if "." in skill.key else skill.key,
                "description": skill.description,
                "input_schema": skill.input_schema,
            }
            for skill in skills if _skill_allowed(skill.key, expert.allowed_tools)
        ]

    def expert_revisions(self, key: str) -> list[dict[str, Any]]:
        return self.graph.run(
            "MATCH (r:ExpertRevision {key: $key}) RETURN properties(r) AS revision "
            "ORDER BY r.version DESC",
            key=key,
        )

    def record_plan(self, plan: RoutingPlan, question: str) -> None:
        request_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
        # Correlate this trace with its conversation. Resolved here, from
        # deterministic per-connection state, so no harness or model ever
        # bookkeeps a session id (see `knowledge_bases.tools.active_trace`).
        from knowledge_bases.tools.active_trace import current_session_id, resolve_identity

        session_id = current_session_id()
        identity = resolve_identity()

        def work(tx: Any) -> None:
            tx.run(
                "CREATE (t:RoutingTrace {id: $id, request_hash: $request_hash, "
                "question: $question, question_summary: $summary, "
                "started_at: datetime(), status: 'planned', "
                "strategy: 'graph_capability_scoring', primary_expert: $primary, "
                "confidence: $confidence, clarification_needed: $clarification, "
                "session_id: $session_id})",
                id=plan.trace_id,
                request_hash=request_hash,
                question=question,
                summary=plan.question_summary,
                primary=plan.primary_expert,
                confidence=plan.confidence,
                clarification=plan.clarification_needed,
                session_id=session_id,
            ).consume()
            tx.run(
                "MERGE (s:Session {id: $session_id}) "
                "ON CREATE SET s.started_at = datetime(), s.created_at = datetime() "
                "SET s.last_active_at = datetime(), "
                "s.user_id = coalesce($user_id, s.user_id), "
                "s.client = coalesce($client, s.client) "
                "WITH s "
                "MATCH (t:RoutingTrace {id: $trace_id}) "
                "MERGE (s)-[ht:HAS_TRACE]->(t) "
                "ON CREATE SET ht.created_at = datetime()",
                session_id=session_id, trace_id=plan.trace_id,
                user_id=identity.get("user_id"), client=identity.get("client"),
            ).consume()
            for rank, candidate in enumerate(plan.candidates):
                tx.run(
                    "MATCH (t:RoutingTrace {id: $trace_id}), "
                    "(e:KnowledgeBaseExpert {key: $expert}) "
                    "CREATE (t)-[:CONSIDERED {rank: $rank, score: $score, "
                    "reasons: $reasons, created_at: datetime()}]->(e)",
                    trace_id=plan.trace_id,
                    expert=candidate.knowledge_base,
                    rank=rank,
                    score=candidate.score,
                    reasons=candidate.reasons,
                ).consume()
            for delegation in plan.delegations:
                tx.run(
                    "MATCH (t:RoutingTrace {id: $trace_id}), "
                    "(e:KnowledgeBaseExpert {key: $expert}) "
                    "CREATE (t)-[:DELEGATED_TO {sequence: $sequence, "
                    "operation: $operation, status: 'planned', error: '', "
                    "created_at: datetime()}]->(e)",
                    trace_id=plan.trace_id,
                    expert=delegation.knowledge_base,
                    sequence=delegation.sequence,
                    operation=delegation.operation,
                ).consume()

        self.graph.execute_write(work)

    def complete_trace(self, trace_id: str, results: list[ExpertResult], status: str) -> None:
        def work(tx: Any) -> None:
            tx.run(
                "MATCH (t:RoutingTrace {id: $trace_id}) "
                "SET t.status = $status, t.completed_at = datetime()",
                trace_id=trace_id,
                status=status,
            ).consume()
            for sequence, result in enumerate(results):
                tx.run(
                    "MATCH (t:RoutingTrace {id: $trace_id})"
                    "-[d:DELEGATED_TO {sequence: $sequence}]->"
                    "(:KnowledgeBaseExpert {key: $expert}) "
                    "SET d.status = $status, d.completed_at = datetime(), d.error = $error",
                    trace_id=trace_id,
                    sequence=sequence,
                    expert=result.knowledge_base,
                    status=result.status,
                    error=result.error,
                ).consume()

        self.graph.execute_write(work)

    def routing_gaps(self, min_traces: int = 5) -> list[dict]:
        """Per-expert low-confidence / clarification-needed `RoutingTrace`
        stats -- a signal that an `ExpertCapability` statement's wording
        doesn't match how questions are actually phrased, distinct from a
        skill *failing*: routing can be low-confidence even when every
        delegated skill call succeeds. `question_summary` is the 240-char
        list-view truncation -- sampling it is what keeps this signal's prompt
        bounded; `record_plan` writes the full verbatim prompt beside it as
        `question`, and the `RoutingTrace` is the graph's single store of what
        the user said (see `docs/architecture/provenance.md`).
        """
        rows = self.graph.run(
            "MATCH (t:RoutingTrace) "
            "WHERE t.confidence < 0.5 OR t.clarification_needed = true "
            "WITH t.primary_expert AS knowledge_base, t, "
            "CASE WHEN t.confidence < 0.5 THEN 1 ELSE 0 END AS low_confidence, "
            "CASE WHEN t.clarification_needed THEN 1 ELSE 0 END AS needs_clarification "
            "WITH knowledge_base, sum(low_confidence) AS low_confidence_count, "
            "sum(needs_clarification) AS clarification_needed_count, "
            "collect(t.question_summary)[..5] AS sample_questions, count(t) AS traces "
            "WHERE traces >= $min_traces "
            "RETURN knowledge_base, low_confidence_count, clarification_needed_count, "
            "sample_questions "
            "ORDER BY low_confidence_count DESC",
            min_traces=min_traces,
        )
        return rows

    def routing_metrics_for(self, knowledge_base: str, since: str) -> dict:
        """Raw routing-confidence stats for exactly one knowledge base's
        `RoutingTrace`s where it was `primary_expert`, started at or after
        `since` (an ISO-8601 string), with no minimum-traffic threshold --
        unlike `routing_gaps()` (which is for *discovering* which knowledge
        bases qualify across the whole graph), this is for *re-measuring*
        one specific knowledge base self-improvement already applied a
        capability-wording change to (see
        `knowledge_bases.tools.self_improvement_apply`).
        """
        rows = self.graph.run(
            "MATCH (t:RoutingTrace {primary_expert: $knowledge_base}) "
            "WHERE t.started_at >= datetime($since) "
            "WITH t, "
            "CASE WHEN t.confidence < 0.5 THEN 1 ELSE 0 END AS low_confidence, "
            "CASE WHEN t.clarification_needed THEN 1 ELSE 0 END AS needs_clarification "
            "RETURN count(t) AS trace_count, "
            "sum(low_confidence) AS low_confidence_count, "
            "sum(needs_clarification) AS clarification_needed_count, "
            "avg(t.confidence) AS avg_confidence",
            knowledge_base=knowledge_base,
            since=since,
        )
        row = rows[0] if rows else {}
        trace_count = row.get("trace_count") or 0
        low_confidence_count = row.get("low_confidence_count") or 0
        clarification_needed_count = row.get("clarification_needed_count") or 0
        return {
            "trace_count": trace_count,
            "low_confidence_count": low_confidence_count,
            "clarification_needed_count": clarification_needed_count,
            "low_confidence_share": (
                (low_confidence_count + clarification_needed_count) / trace_count
                if trace_count else 0.0
            ),
            "avg_confidence": row.get("avg_confidence"),
        }

    def get_trace(self, trace_id: str) -> dict | None:
        rows = self.graph.run(
            "MATCH (t:RoutingTrace {id: $trace_id}) "
            "OPTIONAL MATCH (t)-[c:CONSIDERED]->(candidate:KnowledgeBaseExpert) "
            "OPTIONAL MATCH (t)-[d:DELEGATED_TO]->(expert:KnowledgeBaseExpert) "
            "RETURN properties(t) AS trace, "
            "collect(DISTINCT {expert: candidate.key, rank: c.rank, score: c.score, "
            "reasons: c.reasons}) AS candidates, "
            "collect(DISTINCT {expert: expert.key, sequence: d.sequence, "
            "operation: d.operation, status: d.status, error: d.error}) AS delegations",
            trace_id=trace_id,
        )
        return rows[0] if rows else None
