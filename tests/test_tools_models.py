from knowledge_bases.tools.models import (
    OutcomeSource,
    SpanEvent,
    SpanKind,
    SpanStatus,
    TraceOutcome,
)


def test_span_event_generates_ids_and_defaults():
    event = SpanEvent(name="skill.handle", kind=SpanKind.skill_call,
                       knowledge_base="business_dictionary", started_at="2026-08-17T00:00:00Z")

    assert event.span_id
    assert event.trace_id
    assert event.span_id != event.trace_id
    assert event.status == SpanStatus.ok
    assert event.environment == "dev"
    assert event.parent_span_id is None


def test_trace_outcome_defaults_recorded_at_and_generates_outcome_id():
    outcome = TraceOutcome(trace_id="trace-1", success=True, source=OutcomeSource.self_assessed)

    assert outcome.outcome_id
    assert outcome.recorded_at is not None


def test_two_span_events_get_distinct_ids():
    a = SpanEvent(name="a", kind=SpanKind.tool_call, knowledge_base="kb", started_at="2026-08-17T00:00:00Z")
    b = SpanEvent(name="b", kind=SpanKind.tool_call, knowledge_base="kb", started_at="2026-08-17T00:00:00Z")

    assert a.span_id != b.span_id
    assert a.trace_id != b.trace_id
