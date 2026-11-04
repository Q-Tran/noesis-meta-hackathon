"""Graph-backed reads/writes for `Span` and `TraceOutcome` nodes.

Execution telemetry lives in the same Neo4j graph as the `Tool` catalog it
describes: every `Span` is `MERGE`d on `span_id` (so a retried write is a
no-op, never a duplicate) and linked `(:Span)-[:CALLED]->(:Tool)` when the
skill it recorded is registered, and `(:Span)-[:PART_OF]->(:RoutingTrace)`
when the trace was minted by the orchestrator. `TraceOutcome` nodes are an
append-only log per `trace_id`; the *current* verdict for a trace is the
outcome with the latest `recorded_at`.

Keeping this in the graph -- rather than a per-machine DuckDB file behind a
Kafka worker -- means every user's harness, every MCP server process, and
the self-improvement loop see the same telemetry concurrently, with no
single-writer lock and no worker to deploy.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from graph import Graph
from knowledge_bases.tools.models import SpanEvent, TraceOutcome


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _utc(value: Any) -> Optional[datetime]:
    """Normalize a datetime/ISO string to a tz-aware UTC datetime.

    Neo4j can't compare a zoned `DateTime` with a naive `LocalDateTime`, so
    every timestamp written or used as a cutoff goes through here first;
    naive inputs are taken as UTC.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _json(value: Any) -> Optional[str]:
    return None if value is None else json.dumps(value, default=str)


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


class TraceStore:
    def __init__(self, graph: Graph | None = None):
        self.graph = graph or Graph()

    # --- writes ---------------------------------------------------------
    def insert_span(self, event: SpanEvent) -> None:
        props = event.model_dump(mode="python")
        props["kind"] = _enum_value(event.kind)
        props["status"] = _enum_value(event.status)
        props["started_at"] = _utc(event.started_at)
        props["ended_at"] = _utc(event.ended_at)
        props["raw_input"] = _json(event.raw_input)
        props["raw_output"] = _json(event.raw_output)
        props = {key: value for key, value in props.items() if value is not None}
        self.graph.run(
            "MERGE (s:Span {span_id: $span_id}) "
            "ON CREATE SET s += $props, s.created_at = datetime() "
            "WITH s "
            "OPTIONAL MATCH (k:Tool {key: $skill}) "
            "FOREACH (_ IN CASE WHEN k IS NULL THEN [] ELSE [1] END | "
            "  MERGE (s)-[c:CALLED]->(k) ON CREATE SET c.created_at = datetime())",
            span_id=event.span_id, props=props, skill=event.skill,
        )
        if event.trace_id:
            self.graph.run(
                "MATCH (s:Span {span_id: $span_id}) "
                "MERGE (t:RoutingTrace {id: $trace_id}) "
                "ON CREATE SET t.started_at = s.started_at, t.status = 'active', t.session_id = $session_id, t.created_at = datetime() "
                "SET t.last_active_at = coalesce(s.ended_at, s.started_at), "
                "    t.status = CASE WHEN s.status = 'error' THEN 'error' ELSE t.status END, "
                "    t.error = CASE WHEN s.status = 'error' THEN coalesce(s.error, 'Tool execution error') ELSE t.error END "
                "MERGE (s)-[p:PART_OF]->(t) "
                "ON CREATE SET p.created_at = datetime() "
                "WITH t "
                "FOREACH (_ IN CASE WHEN $session_id IS NULL THEN [] ELSE [1] END | "
                "    MERGE (sess:Session {id: $session_id}) "
                "    ON CREATE SET sess.started_at = t.started_at, sess.created_at = datetime() "
                "    SET sess.last_active_at = t.last_active_at "
                "    MERGE (sess)-[ht:HAS_TRACE]->(t) "
                "    ON CREATE SET ht.created_at = datetime() "
                ")",
                span_id=event.span_id, trace_id=event.trace_id, session_id=event.session_id,
            )

    def insert_outcome(self, outcome: TraceOutcome) -> None:
        self.graph.run(
            "MERGE (o:TraceOutcome {outcome_id: $outcome_id}) "
            "ON CREATE SET o.trace_id = $trace_id, o.success = $success, o.reason = $reason, "
            "o.source = $source, o.recorded_at = $recorded_at, o.created_at = datetime() "
            "WITH o "
            "MERGE (t:RoutingTrace {id: $trace_id}) "
            "ON CREATE SET t.started_at = o.recorded_at, t.created_at = datetime(), t.status = CASE WHEN $success THEN 'completed' ELSE 'error' END "
            "SET t.status = CASE WHEN NOT $success THEN 'error' ELSE CASE WHEN t.status IN ['planned', 'active'] THEN 'completed' ELSE t.status END END, "
            "    t.completed_at = coalesce(t.completed_at, o.recorded_at), "
            "    t.error = CASE WHEN NOT $success THEN coalesce(o.reason, t.error) ELSE t.error END "
            "MERGE (o)-[j:JUDGES]->(t) "
            "ON CREATE SET j.created_at = datetime()",
            outcome_id=outcome.outcome_id, trace_id=outcome.trace_id,
            success=outcome.success, reason=outcome.reason,
            source=_enum_value(outcome.source), recorded_at=_utc(outcome.recorded_at),
        )

    def update_trace(
        self,
        trace_id: str,
        status: str,
        error: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.graph.run(
            "MERGE (t:RoutingTrace {id: $trace_id}) "
            "ON CREATE SET t.started_at = datetime(), t.status = $status, t.session_id = $session_id, t.created_at = datetime() "
            "SET t.status = $status, "
            "    t.last_active_at = datetime(), "
            "    t.completed_at = CASE WHEN $status IN ['completed', 'error'] THEN datetime() ELSE t.completed_at END, "
            "    t.error = coalesce($error, t.error) "
            "WITH t "
            "FOREACH (_ IN CASE WHEN $session_id IS NULL THEN [] ELSE [1] END | "
            "    MERGE (sess:Session {id: $session_id}) "
            "    ON CREATE SET sess.started_at = datetime(), sess.created_at = datetime() "
            "    SET sess.last_active_at = datetime() "
            "    MERGE (sess)-[ht:HAS_TRACE]->(t) "
            "    ON CREATE SET ht.created_at = datetime() "
            ")",
            trace_id=trace_id,
            status=status,
            error=error,
            session_id=session_id,
        )

    def record_turn_start(
        self,
        trace_id: str,
        session_id: str,
        question: str,
        client: str | None = None,
        user_id: str | None = None,
    ) -> None:
        import hashlib
        request_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
        summary = question[:240].strip()
        # `question` (verbatim) and the summary are set unconditionally, not
        # under ON CREATE: a tool-call span's `MERGE (t:RoutingTrace {id})` in
        # `insert_span` frequently creates the node first (the detached tracer
        # spawn races the MCP self-trace), so ON CREATE writes were skipped and
        # 46 of 57 traces carried no prompt text at all. The RoutingTrace is
        # the graph's single store of what the user said: a claim filed from
        # the conversation is substantiated by
        # `(Claim)-[:FOUND_IN {role: "origin"}]->(RoutingTrace)` (see
        # `knowledge_bases/provenance/conversation.py`), so nothing copies the
        # text into a `Chunk` or a `SourceNode`.
        self.graph.run(
            "MERGE (t:RoutingTrace {id: $trace_id}) "
            "ON CREATE SET t.started_at = datetime(), t.status = 'active', t.created_at = datetime() "
            "SET t.question = $question, t.question_summary = $summary, "
            "    t.request_hash = $request_hash, t.session_id = $session_id, "
            "    t.last_active_at = datetime() "
            "WITH t "
            "MERGE (s:Session {id: $session_id}) "
            "ON CREATE SET s.started_at = datetime(), s.created_at = datetime() "
            "SET s.last_active_at = datetime(), "
            "    s.user_id = coalesce($user_id, s.user_id), "
            "    s.client = coalesce($client, s.client) "
            "MERGE (s)-[ht:HAS_TRACE]->(t) "
            "ON CREATE SET ht.created_at = datetime()",
            trace_id=trace_id, session_id=session_id, question=question,
            request_hash=request_hash, summary=summary, client=client, user_id=user_id,
        )

    # --- reads ----------------------------------------------------------
    @staticmethod
    def _span_row(props: dict) -> dict:
        props["raw_input"] = _loads(props.get("raw_input"))
        props["raw_output"] = _loads(props.get("raw_output"))
        return props

    def get_span(self, span_id: str) -> Optional[dict]:
        rows = self.graph.run(
            "MATCH (s:Span {span_id: $span_id}) RETURN properties(s) AS span", span_id=span_id,
        )
        return self._span_row(rows[0]["span"]) if rows else None

    def get_trace_spans(self, trace_id: str) -> list[dict]:
        rows = self.graph.run(
            "MATCH (s:Span {trace_id: $trace_id}) RETURN properties(s) AS span "
            "ORDER BY s.started_at",
            trace_id=trace_id,
        )
        return [self._span_row(row["span"]) for row in rows]

    def current_outcome(self, trace_id: str) -> Optional[dict]:
        rows = self.graph.run(
            "MATCH (o:TraceOutcome {trace_id: $trace_id}) "
            "RETURN o.trace_id AS trace_id, o.success AS success, o.reason AS reason, "
            "o.source AS source, o.recorded_at AS recorded_at "
            "ORDER BY o.recorded_at DESC LIMIT 1",
            trace_id=trace_id,
        )
        return rows[0] if rows else None

    def trace_metrics(self, trace_id: str) -> Optional[dict]:
        rows = self.graph.run(
            "MATCH (s:Span {trace_id: $trace_id}) "
            "RETURN s.trace_id AS trace_id, sum(s.total_tokens) AS total_tokens, "
            "sum(s.cost_usd) AS total_cost_usd, "
            "sum(CASE WHEN s.kind = 'tool_call' THEN 1 ELSE 0 END) AS tool_call_count",
            trace_id=trace_id,
        )
        if not rows:
            return None
        outcome = self.current_outcome(trace_id)
        row = rows[0]
        row["success"] = outcome["success"] if outcome else None
        return row

    def _spans_with_outcome(self, since: Any, skill: str | None = None) -> list[dict]:
        """Every skill span since `since`, each joined with its trace's
        *current* (latest) outcome, for aggregation in Python -- the
        volumes here are small, and one plain query is easier to keep
        correct than a nested Cypher aggregation."""
        skill_filter = "s.skill = $skill" if skill else "s.skill IS NOT NULL"
        return self.graph.run(
            f"MATCH (s:Span) WHERE {skill_filter} AND s.started_at >= $since "
            "OPTIONAL MATCH (o:TraceOutcome {trace_id: s.trace_id}) "
            "WITH s, o ORDER BY o.recorded_at DESC "
            "WITH s, collect(o)[0] AS current "
            "RETURN s.skill AS skill, s.knowledge_base AS knowledge_base, "
            "s.status AS status, s.duration_ms AS duration_ms, s.error AS error, "
            "s.trace_id AS trace_id, s.user_id AS user_id, s.client AS client, "
            "s.expert_key AS expert_key, "
            "current.success AS outcome_success, current.source AS outcome_source, "
            "current.reason AS outcome_reason",
            since=_utc(since), skill=skill,
        )

    @staticmethod
    def _corrected(row: dict) -> bool:
        return row.get("outcome_success") is False and row.get("outcome_source") == "user_correction"

    def skill_health(self, since: Any, min_calls: int = 5) -> list[dict]:
        """Per-skill health over `Span`/`TraceOutcome` since `since`.

        `error_count`/`error_rate` come straight from each span's own
        `status`. `user_correction_count` is a stronger, cross-span signal:
        the number of *distinct traces* containing a call to this skill
        whose current (latest) outcome was a `user_correction` marked
        `success = false` -- i.e. the user said the end result was wrong,
        not just that this one call errored. Both matter: a skill can have
        a 0% error rate and still be the thing users keep correcting (wrong
        answer, not a crash), or vice versa (transient errors that still
        produced an accepted final answer).

        `by_client`/`by_user` break `call_count`/`error_count` down per
        harness and per user so a skill failing for one client only is
        distinguishable from one failing everywhere.
        """
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in self._spans_with_outcome(since):
            grouped[row["skill"]].append(row)
        results = []
        for skill, rows in grouped.items():
            if len(rows) < min_calls:
                continue
            durations = [r["duration_ms"] for r in rows if r["duration_ms"] is not None]
            errors = [r["error"] for r in rows if r["status"] == "error"]
            sample_errors: list[str] = []
            for error in errors:
                if error and error not in sample_errors:
                    sample_errors.append(error)
            by_client: dict[str, dict] = defaultdict(lambda: {"call_count": 0, "error_count": 0})
            by_user: dict[str, dict] = defaultdict(lambda: {"call_count": 0, "error_count": 0})
            for r in rows:
                for bucket, key in ((by_client, r.get("client")), (by_user, r.get("user_id"))):
                    stats = bucket[key or "unknown"]
                    stats["call_count"] += 1
                    stats["error_count"] += int(r["status"] == "error")
            results.append({
                "skill": skill,
                "knowledge_base": rows[0]["knowledge_base"],
                "call_count": len(rows),
                "error_count": len(errors),
                "error_rate": len(errors) / len(rows),
                "avg_duration_ms": sum(durations) / len(durations) if durations else None,
                "user_correction_count": len({r["trace_id"] for r in rows if self._corrected(r)}),
                "sample_errors": sample_errors[:5],
                "by_client": dict(by_client),
                "by_user": dict(by_user),
            })
        return sorted(
            results, key=lambda r: (-r["error_count"], -r["user_correction_count"]),
        )

    def search_retry_loops(
        self, since: Any, min_calls: int = 3, min_traces: int = 2,
    ) -> list[dict]:
        """Search tools called repeatedly within a single trace.

        A retry loop is a failure of *expression*, not of execution: every call
        can return `status = ok` while the caller still never got what it
        wanted. That makes it invisible to `skill_health`, which sees only
        errors and user corrections.

        The level of failure is separated, because the two have different
        fixes:

        - `empty_calls` -- the search ran and returned nothing. The caller's
          vocabulary does not match the data, so the *tool's description* (what
          it is for, what it covers) is what misled them.
        - the remainder -- the search returned rows and was called again
          anyway. The caller got the wrong *kind* of answer, which points at
          scope/entity-type wording rather than at the index.

        A trace only counts once per tool, so one pathological session cannot
        by itself make a tool look broken.
        """
        rows = self.graph.run(
            # Retrieval tools only. A dispatcher like `call_noesis_tool` is
            # *expected* to be called many times in one turn -- that is it
            # doing its job, not a caller struggling -- so counting it here
            # would swamp the real signal.
            "MATCH (s:Span) WHERE s.started_at >= $since AND s.skill IS NOT NULL "
            "AND (s.name CONTAINS 'search' OR s.name CONTAINS 'lookup' "
            "     OR s.name CONTAINS 'find' OR s.name CONTAINS 'resolve' "
            "     OR s.name CONTAINS 'trace_' OR s.name CONTAINS 'get_') "
            "WITH s.trace_id AS trace_id, s.skill AS skill, "
            "     s.knowledge_base AS knowledge_base, count(s) AS calls, "
            "     sum(CASE WHEN s.result_count = 0 THEN 1 ELSE 0 END) AS empty_calls, "
            "     collect(s.raw_input)[..3] AS sample_inputs "
            "WHERE calls >= $min_calls "
            "WITH skill, knowledge_base, count(trace_id) AS retry_traces, "
            "     sum(calls) AS total_calls, sum(empty_calls) AS empty_calls, "
            "     collect(sample_inputs)[0] AS sample_inputs "
            "WHERE retry_traces >= $min_traces "
            "RETURN skill, knowledge_base, retry_traces, total_calls, empty_calls, "
            "       sample_inputs "
            "ORDER BY retry_traces DESC, total_calls DESC",
            since=since, min_calls=min_calls, min_traces=min_traces,
        )
        return [dict(row) for row in rows]

    def expert_corrections(self, since: Any) -> list[dict]:
        """Per graph-defined expert (`Span.expert_key`), how many distinct
        traces it acted in since `since` and how many of those the user
        later corrected -- the signal that an expert's *prompt* (not one
        skill's wording) is what needs improving."""
        by_expert: dict[str, dict[str, Any]] = {}
        for row in self._spans_with_outcome(since):
            key = row.get("expert_key")
            if not key:
                continue
            stats = by_expert.setdefault(key, {
                "expert_key": key, "traces": set(), "corrected": set(),
                "reasons": [], "skills": set(),
            })
            stats["traces"].add(row["trace_id"])
            stats["skills"].add(row["skill"])
            if self._corrected(row):
                stats["corrected"].add(row["trace_id"])
                reason = row.get("outcome_reason")
                if reason and reason not in stats["reasons"]:
                    stats["reasons"].append(reason)
        results = []
        for stats in by_expert.values():
            results.append({
                "expert_key": stats["expert_key"],
                "trace_count": len(stats["traces"]),
                "corrected_trace_count": len(stats["corrected"]),
                "correction_share": len(stats["corrected"]) / len(stats["traces"]),
                "sample_reasons": stats["reasons"][:5],
                "skills": sorted(s for s in stats["skills"] if s),
            })
        return sorted(results, key=lambda r: -r["corrected_trace_count"])

    def expert_metrics_for(self, expert_key: str, since: Any) -> dict:
        rows = [r for r in self.expert_corrections(since) if r["expert_key"] == expert_key]
        if not rows:
            return {"trace_count": 0, "corrected_trace_count": 0, "correction_share": 0.0}
        return {k: rows[0][k] for k in ("trace_count", "corrected_trace_count", "correction_share")}

    def skill_metrics_for(self, skill: str, since: Any) -> dict:
        """Raw `call_count`/`error_rate`/`user_correction_count` for exactly
        one skill since `since`, with no minimum-call threshold -- unlike
        `skill_health()` (which is for *discovering* qualifying skills
        across the whole catalog), this is for *re-measuring* one specific
        skill self-improvement already applied a change to, where the
        window since that change may be short and call_count could be low
        or zero (see `knowledge_bases.tools.self_improvement_apply`).
        """
        rows = self._spans_with_outcome(since, skill=skill)
        call_count = len(rows)
        error_count = sum(1 for r in rows if r["status"] == "error")
        return {
            "call_count": call_count,
            "error_count": error_count,
            "error_rate": error_count / call_count if call_count else 0.0,
            "user_correction_count": len({r["trace_id"] for r in rows if self._corrected(r)}),
        }
