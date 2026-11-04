"""Tests for automatic harness-to-graph tracing and lifecycle telemetry."""
from __future__ import annotations

from typing import Any
from datetime import datetime, timezone
import pytest

from knowledge_bases.tools import harness_tracer
from knowledge_bases.tools import tracing
from knowledge_bases.tools.active_trace import active_trace_id, reset_for_testing
from knowledge_bases.tools.models import OutcomeSource, SpanEvent, SpanKind, SpanStatus, TraceOutcome
from knowledge_bases.tools.trace_store import TraceStore


class FakeGraph:
    def __init__(self):
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append((query, params))
        return []


class FakeSink:
    def __init__(self):
        self.spans: list[SpanEvent] = []
        self.outcomes: list[TraceOutcome] = []
        self.turns: list[dict[str, Any]] = []
        self.trace_updates: list[dict[str, Any]] = []

        class Store:
            def __init__(store_self, outer_sink):
                store_self.outer = outer_sink

            def record_turn_start(store_self, **kwargs):
                store_self.outer.turns.append(kwargs)

            def update_trace(store_self, **kwargs):
                store_self.outer.trace_updates.append(kwargs)

        self.store = Store(self)

    def publish_span(self, event: SpanEvent) -> None:
        self.spans.append(event)

    def publish_outcome(self, outcome: TraceOutcome) -> None:
        self.outcomes.append(outcome)


@pytest.fixture(autouse=True)
def clean_scopes():
    reset_for_testing()
    yield
    reset_for_testing()


def test_record_turn_start_initiates_active_trace(monkeypatch):
    fake = FakeSink()
    monkeypatch.setattr(harness_tracer, "_get_sink", lambda: fake)

    trace_id = harness_tracer.record_turn_start(
        prompt="Show me quarterly revenue",
        session_id="session-42",
        client="opencode",
        user_id="user-bob",
    )

    assert trace_id
    assert active_trace_id() == trace_id
    assert len(fake.turns) == 1
    turn = fake.turns[0]
    assert turn["trace_id"] == trace_id
    assert turn["session_id"] == "session-42"
    assert turn["question"] == "Show me quarterly revenue"
    assert turn["client"] == "opencode"
    assert turn["user_id"] == "user-bob"


def test_record_tool_execution_logs_span_to_sink(monkeypatch):
    fake = FakeSink()
    monkeypatch.setattr(harness_tracer, "_get_sink", lambda: fake)
    monkeypatch.setattr(tracing, "_get_sink", lambda: fake)

    span_id = harness_tracer.record_tool_execution(
        tool_name="bash",
        raw_input={"command": "pytest"},
        raw_output={"exit_code": 0},
        duration_ms=120,
        status="ok",
        trace_id="tr-test-1",
        session_id="sess-test-1",
    )

    assert span_id
    assert len(fake.spans) == 1
    span = fake.spans[0]
    assert span.span_id == span_id
    assert span.name == "bash"
    assert span.status == SpanStatus.ok
    assert span.trace_id == "tr-test-1"
    assert span.duration_ms == 120
    assert len(fake.outcomes) == 0


def test_record_tool_execution_error_infers_failure_outcome(monkeypatch):
    fake = FakeSink()
    monkeypatch.setattr(harness_tracer, "_get_sink", lambda: fake)
    monkeypatch.setattr(tracing, "_get_sink", lambda: fake)

    span_id = harness_tracer.record_tool_execution(
        tool_name="edit_file",
        raw_input={"path": "main.py"},
        status="error",
        error="File not found",
        trace_id="tr-error-1",
        session_id="sess-error-1",
    )

    assert span_id
    assert len(fake.spans) == 1
    assert fake.spans[0].status == SpanStatus.error
    assert len(fake.outcomes) == 1
    outcome = fake.outcomes[0]
    assert outcome.trace_id == "tr-error-1"
    assert outcome.success is False
    assert "File not found" in outcome.reason
    assert outcome.source == OutcomeSource.harness_inferred


def test_record_turn_completion_updates_status_and_outcome(monkeypatch):
    fake = FakeSink()
    monkeypatch.setattr(harness_tracer, "_get_sink", lambda: fake)
    monkeypatch.setattr(tracing, "_get_sink", lambda: fake)

    ok = harness_tracer.record_turn_completion(
        trace_id="tr-finish-1",
        session_id="sess-finish-1",
        success=True,
    )

    assert ok is True
    assert len(fake.trace_updates) == 1
    assert fake.trace_updates[0]["status"] == "completed"
    assert fake.trace_updates[0]["trace_id"] == "tr-finish-1"

    assert len(fake.outcomes) == 1
    outcome = fake.outcomes[0]
    assert outcome.trace_id == "tr-finish-1"
    assert outcome.success is True
    assert outcome.source == OutcomeSource.harness_inferred


def test_trace_store_insert_span_automatically_merges_routing_trace():
    graph = FakeGraph()
    store = TraceStore(graph)

    event = SpanEvent(
        trace_id="tr-auto-1",
        session_id="sess-auto-1",
        name="hybrid_search",
        kind=SpanKind.tool_call,
        knowledge_base="business_dictionary",
        skill="business_dictionary.hybrid_search",
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        duration_ms=50,
        status=SpanStatus.ok,
    )

    store.insert_span(event)

    # Verifies both span merge and routing trace auto-creation queries ran
    assert len(graph.queries) == 2
    span_query, span_params = graph.queries[0]
    assert "MERGE (s:Span {span_id: $span_id})" in span_query

    trace_query, trace_params = graph.queries[1]
    assert "MERGE (t:RoutingTrace {id: $trace_id})" in trace_query
    assert "MERGE (s)-[p:PART_OF]->(t)" in trace_query
    assert "ON CREATE SET p.created_at = datetime()" in trace_query
    assert "MERGE (sess:Session {id: $session_id})" in trace_query
    assert trace_params["trace_id"] == "tr-auto-1"
    assert trace_params["session_id"] == "sess-auto-1"


def test_trace_store_insert_outcome_automatically_merges_routing_trace():
    graph = FakeGraph()
    store = TraceStore(graph)

    outcome = TraceOutcome(
        trace_id="tr-outcome-1",
        success=True,
        source=OutcomeSource.harness_inferred,
    )

    store.insert_outcome(outcome)

    assert len(graph.queries) == 1
    query, params = graph.queries[0]
    assert "MERGE (o:TraceOutcome {outcome_id: $outcome_id})" in query
    assert "MERGE (t:RoutingTrace {id: $trace_id})" in query
    assert "MERGE (o)-[j:JUDGES]->(t)" in query
    assert "ON CREATE SET j.created_at = datetime()" in query
    assert params["trace_id"] == "tr-outcome-1"
    assert params["success"] is True
