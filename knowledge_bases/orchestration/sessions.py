"""Durable conversations: `:Session` nodes and the resume digest.

Before this, a "session" was a `uuid4()` minted per server process and written
as a bare property on `Span` nodes -- never a node, never indexed, never
queried. Closing the terminal lost the thread entirely, and the only way to
carry context across turns was the model's own window.

A session is now a node, so a conversation can be picked up later. The thing
that is *not* stored is a serialized blob of state: `resume_noesis_session`
recomputes a bounded digest from the graph on demand
(`RoutingTrace`/`Span`/`TraceOutcome`/`DelegationResult` the session already
produced). That is the deliberate difference from a REPL harness snapshotting
its kernel namespace:

- a digest cannot be a code-execution vector, where unpickling one is;
- it improves as the graph improves, instead of freezing what one process
  happened to hold in memory;
- it is queryable and attributable, because it *is* the trace record.

The cost is honest: a digest is a summary of what was established, not a
perfect continuation of the model's reasoning. For a knowledge system that is
the right trade -- the durable part of a conversation is the knowledge it
touched, not the token stream.

Nothing here is written by the model: `record_session` is called by the same
deterministic code that mints the trace, so a harness never bookkeeps ids
(see `knowledge_bases.tools.active_trace`).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from graph import Graph

# Bounds on everything the digest returns. A resume digest is injected into a
# fresh context window, so it has to be small enough to be worth reading;
# these are the limits that make it a summary instead of a transcript.
MAX_DIGEST_TRACES = 10
MAX_DIGEST_ENTITIES = 25
MAX_DIGEST_OPEN_ITEMS = 10


class SessionSummary(BaseModel):
    id: str
    started_at: Any = None
    last_active_at: Any = None
    user_id: str | None = None
    client: str | None = None
    trace_count: int = 0
    first_question: str | None = None


class SessionRepository:
    def __init__(self, graph: Graph):
        self.graph = graph

    # --- writes ---------------------------------------------------------
    def record_session(
        self, session_id: str, user_id: str | None = None, client: str | None = None,
    ) -> None:
        """MERGE the session node and bump its activity clock.

        Idempotent per turn: called on every routed question, so `started_at`
        is set once and `last_active_at` moves.
        """
        self.graph.run(
            "MERGE (s:Session {id: $id}) "
            "ON CREATE SET s.started_at = datetime(), s.created_at = datetime() "
            "SET s.last_active_at = datetime(), "
            "s.user_id = coalesce($user_id, s.user_id), "
            "s.client = coalesce($client, s.client)",
            id=session_id, user_id=user_id, client=client,
        )

    def link_trace(self, session_id: str, trace_id: str) -> None:
        """`(:Session)-[:HAS_TRACE]->(:RoutingTrace)`, so a session's history
        is a traversal rather than a property scan."""
        self.graph.run(
            "MATCH (s:Session {id: $session_id}) "
            "MATCH (t:RoutingTrace {id: $trace_id}) "
            "MERGE (s)-[ht:HAS_TRACE]->(t) "
            "ON CREATE SET ht.created_at = datetime()",
            session_id=session_id, trace_id=trace_id,
        )

    # --- reads ----------------------------------------------------------
    def list_sessions(
        self, user_id: str | None = None, limit: int = 10,
    ) -> list[SessionSummary]:
        rows = self.graph.run(
            "MATCH (s:Session) "
            "WHERE $user_id IS NULL OR s.user_id = $user_id "
            "OPTIONAL MATCH (s)-[:HAS_TRACE]->(t:RoutingTrace) "
            "WITH s, count(t) AS trace_count, "
            "  head(collect(t.question_summary)) AS first_question "
            "RETURN s.id AS id, s.started_at AS started_at, "
            "s.last_active_at AS last_active_at, s.user_id AS user_id, "
            "s.client AS client, trace_count, first_question "
            "ORDER BY s.last_active_at DESC LIMIT $limit",
            user_id=user_id, limit=max(1, min(limit, 50)),
        )
        return [SessionSummary.model_validate(dict(row)) for row in rows]

    def most_recent(self, user_id: str | None = None) -> str | None:
        sessions = self.list_sessions(user_id=user_id, limit=1)
        return sessions[0].id if sessions else None

    def digest(self, session_id: str) -> dict[str, Any]:
        """Everything worth knowing to continue this session, derived live.

        Four things, each bounded: what was asked and how it resolved, which
        entities were established, what is still unresolved, and any open
        questions the session recorded as feedback.
        """
        session = self.graph.run(
            "MATCH (s:Session {id: $id}) "
            "RETURN s.id AS id, s.started_at AS started_at, "
            "s.last_active_at AS last_active_at, s.user_id AS user_id, s.client AS client",
            id=session_id,
        )
        if not session:
            return {"session_id": session_id, "found": False}

        return {
            "session_id": session_id,
            "found": True,
            "session": dict(session[0]),
            "turns": self._turns(session_id),
            "established_entities": self._entities(session_id),
            "unresolved": self._unresolved(session_id),
            "open_questions": self._open_questions(session_id),
            "note": (
                "Derived from this session's recorded traces, not a saved "
                "transcript: it summarizes what was established, not the "
                "earlier conversation verbatim."
            ),
        }

    def _turns(self, session_id: str) -> list[dict[str, Any]]:
        """Each routed question with its expert and final verdict."""
        rows = self.graph.run(
            "MATCH (s:Session {id: $id})-[:HAS_TRACE]->(t:RoutingTrace) "
            "OPTIONAL MATCH (o:TraceOutcome {trace_id: t.id}) "
            "WITH t, o ORDER BY o.recorded_at DESC "
            "WITH t, collect(o)[0] AS outcome "
            "RETURN t.id AS trace_id, t.question_summary AS question, "
            "t.primary_expert AS primary_expert, t.confidence AS confidence, "
            "t.status AS status, outcome.success AS success, "
            "outcome.reason AS outcome_reason "
            "ORDER BY t.started_at DESC LIMIT $limit",
            id=session_id, limit=MAX_DIGEST_TRACES,
        )
        return [dict(row) for row in rows]

    def _entities(self, session_id: str) -> list[dict[str, Any]]:
        """Entities this session's delegations actually resolved.

        Read from `DelegationResult` (see
        `knowledge_bases.orchestration.delegation_results`) rather than
        re-parsing span payloads, so the digest reuses the one structured
        record of what each expert found.
        """
        rows = self.graph.run(
            "MATCH (s:Session {id: $id})-[:HAS_TRACE]->(t:RoutingTrace) "
            "MATCH (t)-[:RESULTED_IN]->(r:DelegationResult) "
            "WHERE r.entity_labels IS NOT NULL "
            "UNWIND range(0, size(r.entity_keys) - 1) AS i "
            "RETURN DISTINCT r.entity_labels[i] AS label, r.entity_keys[i] AS key, "
            "r.expert_key AS expert_key LIMIT $limit",
            id=session_id, limit=MAX_DIGEST_ENTITIES,
        )
        return [dict(row) for row in rows]

    def _unresolved(self, session_id: str) -> list[dict[str, Any]]:
        """What an expert explicitly said it could not settle -- the most
        useful thing to hand a resumed conversation."""
        rows = self.graph.run(
            "MATCH (s:Session {id: $id})-[:HAS_TRACE]->(t:RoutingTrace) "
            "MATCH (t)-[:RESULTED_IN]->(r:DelegationResult) "
            "WHERE r.unresolved IS NOT NULL AND size(r.unresolved) > 0 "
            "RETURN r.expert_key AS expert_key, r.unresolved AS unresolved, "
            "r.trace_id AS trace_id LIMIT $limit",
            id=session_id, limit=MAX_DIGEST_OPEN_ITEMS,
        )
        return [dict(row) for row in rows]

    def _open_questions(self, session_id: str) -> list[str]:
        """Open questions catalogued as feedback during this session."""
        rows = self.graph.run(
            "MATCH (s:Session {id: $id})-[:HAS_TRACE]->(t:RoutingTrace) "
            "MATCH (c:Claim) WHERE c.trace_id = t.id AND c.kind = 'open_question' "
            "RETURN c.text AS text LIMIT $limit",
            id=session_id, limit=MAX_DIGEST_OPEN_ITEMS,
        )
        return [row["text"] for row in rows if row.get("text")]
