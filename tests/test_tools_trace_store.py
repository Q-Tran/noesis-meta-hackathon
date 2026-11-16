import pytest

from knowledge_bases.tools.trace_store import TraceStore  # noqa: E402
from tests.trace_store_fakes import InMemoryTraceStore  # noqa: E402
from knowledge_bases.tools.models import (  # noqa: E402
    OutcomeSource,
    SpanEvent,
    SpanKind,
    TraceOutcome,
)


def make_store() -> TraceStore:
    return InMemoryTraceStore()


def test_insert_span_is_idempotent_on_span_id():
    store = make_store()
    event = SpanEvent(
        span_id="span-1", trace_id="trace-1", name="skill.handle",
        kind=SpanKind.skill_call, knowledge_base="kb", started_at="2026-08-17T00:00:00",
    )

    store.insert_span(event)
    store.insert_span(event)  # simulate Kafka redelivery

    rows = store.get_trace_spans("trace-1")
    assert len(rows) == 1
    assert rows[0]["span_id"] == "span-1"


def test_get_trace_spans_reconstructs_tree_via_parent_span_id():
    store = make_store()
    root = SpanEvent(
        span_id="root", trace_id="trace-1", parent_span_id=None, name="skill.handle",
        kind=SpanKind.skill_call, knowledge_base="kb", started_at="2026-08-17T00:00:00",
    )
    child = SpanEvent(
        span_id="child", trace_id="trace-1", parent_span_id="root", name="tool.query",
        kind=SpanKind.tool_call, knowledge_base="kb", started_at="2026-08-17T00:00:01",
    )
    store.insert_span(root)
    store.insert_span(child)

    rows = store.get_trace_spans("trace-1")
    assert [r["span_id"] for r in rows] == ["root", "child"]
    assert rows[1]["parent_span_id"] == "root"


def test_current_outcome_is_latest_recorded_row_not_the_first():
    store = make_store()
    store.insert_outcome(TraceOutcome(
        trace_id="trace-1", success=True, source=OutcomeSource.self_assessed,
        recorded_at="2026-08-17T00:00:00",
    ))
    store.insert_outcome(TraceOutcome(
        trace_id="trace-1", success=False, reason="user asked for changes",
        source=OutcomeSource.user_correction, recorded_at="2026-08-17T00:05:00",
    ))

    current = store.current_outcome("trace-1")
    assert current["success"] is False
    assert current["source"] == "user_correction"


def test_trace_metrics_aggregates_tokens_cost_and_tool_calls():
    store = make_store()
    store.insert_span(SpanEvent(
        span_id="root", trace_id="trace-1", name="skill.handle", kind=SpanKind.skill_call,
        knowledge_base="kb", started_at="2026-08-17T00:00:00", total_tokens=100, cost_usd=0.01,
    ))
    store.insert_span(SpanEvent(
        span_id="tool-1", trace_id="trace-1", parent_span_id="root", name="tool.query",
        kind=SpanKind.tool_call, knowledge_base="kb", started_at="2026-08-17T00:00:01",
    ))
    store.insert_span(SpanEvent(
        span_id="tool-2", trace_id="trace-1", parent_span_id="root", name="tool.query2",
        kind=SpanKind.tool_call, knowledge_base="kb", started_at="2026-08-17T00:00:02",
        total_tokens=50, cost_usd=0.02,
    ))
    store.insert_outcome(TraceOutcome(
        trace_id="trace-1", success=True, source=OutcomeSource.self_assessed,
    ))

    metrics = store.trace_metrics("trace-1")
    assert metrics["success"] is True
    assert metrics["total_tokens"] == 150
    assert metrics["total_cost_usd"] == pytest.approx(0.03)
    assert metrics["tool_call_count"] == 2


def test_get_span_returns_none_for_unknown_span():
    store = make_store()
    assert store.get_span("does-not-exist") is None


def test_skill_health_aggregates_errors_corrections_and_duration():
    store = make_store()
    for i in range(6):
        store.insert_span(SpanEvent(
            span_id=f"span-{i}", trace_id=f"trace-{i}", name="lookup",
            kind=SpanKind.skill_call, knowledge_base="business_dictionary",
            skill="business_dictionary.lookup_business_term",
            started_at="2026-08-17T00:00:00", duration_ms=100 + i,
            status="error" if i < 2 else "ok",
            error="KeyError: term" if i < 2 else None,
        ))
        store.insert_outcome(TraceOutcome(
            trace_id=f"trace-{i}", success=(i >= 3), source=OutcomeSource.user_correction,
        ))

    rows = store.skill_health(since="2026-08-16T00:00:00", min_calls=5)

    assert len(rows) == 1
    row = rows[0]
    assert row["skill"] == "business_dictionary.lookup_business_term"
    assert row["knowledge_base"] == "business_dictionary"
    assert row["call_count"] == 6
    assert row["error_count"] == 2
    assert row["error_rate"] == pytest.approx(2 / 6)
    assert row["user_correction_count"] == 3
    assert row["sample_errors"] == ["KeyError: term"]
    assert row["avg_duration_ms"] == pytest.approx(102.5)


def test_skill_health_excludes_skills_below_min_calls():
    store = make_store()
    store.insert_span(SpanEvent(
        span_id="span-1", trace_id="trace-1", name="lookup", kind=SpanKind.skill_call,
        knowledge_base="epistemology", skill="epistemology.get_claim_provenance",
        started_at="2026-08-17T00:00:00", status="error", error="boom",
    ))

    rows = store.skill_health(since="2026-08-16T00:00:00", min_calls=5)

    assert rows == []


def test_skill_health_excludes_spans_before_the_since_cutoff():
    store = make_store()
    for i in range(5):
        store.insert_span(SpanEvent(
            span_id=f"old-{i}", trace_id=f"old-trace-{i}", name="lookup",
            kind=SpanKind.skill_call, knowledge_base="epistemology",
            skill="epistemology.get_claim_provenance", started_at="2026-01-01T00:00:00",
            status="error", error="boom",
        ))

    rows = store.skill_health(since="2026-08-01T00:00:00", min_calls=1)

    assert rows == []


def test_skill_metrics_for_aggregates_one_skill_with_no_minimum_call_threshold():
    """Unlike `skill_health()`, `skill_metrics_for()` is for re-evaluating
    one specific skill self-improvement already applied a change to, where
    the post-apply window may have very few calls -- it must never require
    `min_calls` the way `skill_health()` does."""
    store = make_store()
    store.insert_span(SpanEvent(
        span_id="span-1", trace_id="trace-1", name="lookup", kind=SpanKind.skill_call,
        knowledge_base="business_dictionary", skill="business_dictionary.lookup_business_term",
        started_at="2026-08-17T00:00:00", status="error", error="boom",
    ))
    store.insert_span(SpanEvent(
        span_id="span-2", trace_id="trace-2", name="lookup", kind=SpanKind.skill_call,
        knowledge_base="business_dictionary", skill="business_dictionary.lookup_business_term",
        started_at="2026-08-17T00:00:01", status="ok",
    ))
    store.insert_outcome(TraceOutcome(
        trace_id="trace-1", success=False, source=OutcomeSource.user_correction,
    ))

    metrics = store.skill_metrics_for(
        "business_dictionary.lookup_business_term", since="2026-08-16T00:00:00",
    )

    assert metrics["call_count"] == 2
    assert metrics["error_count"] == 1
    assert metrics["error_rate"] == pytest.approx(0.5)
    assert metrics["user_correction_count"] == 1


def test_skill_metrics_for_returns_zeroed_metrics_when_there_are_no_calls():
    store = make_store()

    metrics = store.skill_metrics_for("nothing.called_yet", since="2026-08-16T00:00:00")

    assert metrics == {
        "call_count": 0, "error_count": 0, "error_rate": 0.0, "user_correction_count": 0,
    }


def test_skill_metrics_for_excludes_calls_before_the_since_cutoff():
    store = make_store()
    store.insert_span(SpanEvent(
        span_id="old-span", trace_id="old-trace", name="lookup", kind=SpanKind.skill_call,
        knowledge_base="epistemology", skill="epistemology.get_claim_provenance",
        started_at="2026-01-01T00:00:00", status="error", error="boom",
    ))

    metrics = store.skill_metrics_for(
        "epistemology.get_claim_provenance", since="2026-08-01T00:00:00",
    )

    assert metrics["call_count"] == 0


def test_skill_metrics_for_excludes_calls_to_other_skills():
    store = make_store()
    store.insert_span(SpanEvent(
        span_id="span-1", trace_id="trace-1", name="lookup", kind=SpanKind.skill_call,
        knowledge_base="epistemology", skill="epistemology.get_claim_provenance",
        started_at="2026-08-17T00:00:00", status="error", error="boom",
    ))

    metrics = store.skill_metrics_for(
        "business_dictionary.lookup_business_term", since="2026-08-16T00:00:00",
    )

    assert metrics["call_count"] == 0


# -- live graph round-trip -------------------------------------------------------

def _live_graph():
    """A real graph connection, or skip: the Cypher in `TraceStore` is what
    the in-memory fake above deliberately does not cover."""
    from graph import Graph

    try:
        graph = Graph()
        graph.run("RETURN 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"graph unavailable: {exc}")
    return graph


def test_graph_trace_store_round_trips_spans_outcomes_and_links_to_skill():
    graph = _live_graph()
    store = TraceStore(graph)
    trace_id = f"test-trace-{pytest.__name__}-{id(store)}"
    try:
        graph.run(
            "MERGE (k:Tool {key: 'test_kb.test_skill'}) "
            "SET k.name = 'test', k.version = '0', k.knowledge_base = 'test_kb', k.active = false"
        )
        event = SpanEvent(
            trace_id=trace_id, name="test_skill", kind=SpanKind.skill_call,
            knowledge_base="test_kb", skill="test_kb.test_skill",
            started_at="2026-08-17T00:00:00", duration_ms=5, status="error", error="boom",
            user_id="alice", client="pytest", raw_input={"a": 1}, raw_output={"b": [1, 2]},
        )
        store.insert_span(event)
        store.insert_span(event)
        store.insert_outcome(TraceOutcome(
            trace_id=trace_id, success=False, source=OutcomeSource.user_correction,
        ))

        spans = store.get_trace_spans(trace_id)
        assert len(spans) == 1
        assert spans[0]["raw_input"] == {"a": 1}
        assert spans[0]["raw_output"] == {"b": [1, 2]}
        assert spans[0]["user_id"] == "alice"
        assert store.get_span(event.span_id)["client"] == "pytest"
        assert store.current_outcome(trace_id)["success"] is False
        assert store.trace_metrics(trace_id)["success"] is False

        linked = graph.run(
            "MATCH (s:Span {span_id: $id})-[:CALLED]->(k:Tool) RETURN k.key AS key",
            id=event.span_id,
        )
        assert linked == [{"key": "test_kb.test_skill"}]

        metrics = store.skill_metrics_for("test_kb.test_skill", since="2026-08-16T00:00:00")
        assert metrics["call_count"] == 1
        assert metrics["error_count"] == 1
        assert metrics["user_correction_count"] == 1
        health = [
            r for r in store.skill_health(since="2026-08-16T00:00:00", min_calls=1)
            if r["skill"] == "test_kb.test_skill"
        ]
        assert health and health[0]["by_client"] == {"pytest": {"call_count": 1, "error_count": 1}}
    finally:
        graph.run("MATCH (s:Span {trace_id: $t}) DETACH DELETE s", t=trace_id)
        graph.run("MATCH (o:TraceOutcome {trace_id: $t}) DETACH DELETE o", t=trace_id)
        graph.run("MATCH (k:Tool {key: 'test_kb.test_skill'}) DETACH DELETE k")
