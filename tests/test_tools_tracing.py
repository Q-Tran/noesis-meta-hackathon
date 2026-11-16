"""`record_tool_span` is what actually turns a skill/tool dispatch into a
recorded `SpanEvent`; these tests never touch a real Kafka broker."""
import knowledge_bases.tools.tracing as tracing
from knowledge_bases.tools.models import SpanKind, SpanStatus


class FakePublisher:
    def __init__(self):
        self.spans = []

    def publish_span(self, event):
        self.spans.append(event)


def test_record_skill_span_publishes_on_success_with_timing_and_status_ok(monkeypatch):
    fake = FakePublisher()
    monkeypatch.setattr(tracing, "_get_sink", lambda: fake)

    with tracing.record_tool_span(
        "business_dictionary", "lookup_business_term", {"term": "ARR"},
        skill="business_dictionary.lookup_business_term", skill_version="1.0.0",
        trace_id="trace-1",
    ) as span:
        span.raw_output = {"ok": True}

    assert len(fake.spans) == 1
    published = fake.spans[0]
    assert published.trace_id == "trace-1"
    assert published.knowledge_base == "business_dictionary"
    assert published.skill == "business_dictionary.lookup_business_term"
    assert published.skill_version == "1.0.0"
    assert published.kind == SpanKind.skill_call
    assert published.status == SpanStatus.ok
    assert published.raw_input == {"term": "ARR"}
    assert published.raw_output == {"ok": True}
    assert published.started_at is not None
    assert published.ended_at is not None
    assert published.duration_ms is not None and published.duration_ms >= 0


def test_record_skill_span_marks_error_status_and_reraises_on_exception(monkeypatch):
    fake = FakePublisher()
    monkeypatch.setattr(tracing, "_get_sink", lambda: fake)

    try:
        with tracing.record_tool_span("epistemology", "broken_tool", {}):
            raise ValueError("boom")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError to propagate")

    assert len(fake.spans) == 1
    assert fake.spans[0].status == SpanStatus.error
    assert fake.spans[0].error == "boom"


def test_record_skill_span_generates_trace_id_when_none_given(monkeypatch):
    fake = FakePublisher()
    monkeypatch.setattr(tracing, "_get_sink", lambda: fake)

    with tracing.record_tool_span("epistemology", "standalone_tool", {}):
        pass

    assert fake.spans[0].trace_id


def test_record_skill_span_swallows_publish_failures(monkeypatch):
    class BrokenPublisher:
        def publish_span(self, event):
            raise RuntimeError("kafka is down")

    monkeypatch.setattr(tracing, "_get_sink", lambda: BrokenPublisher())

    with tracing.record_tool_span("epistemology", "resilient_tool", {}):
        pass  # must not raise despite publish failure


def test_record_skill_span_is_a_no_op_when_tracing_disabled(monkeypatch):
    monkeypatch.setattr(tracing, "_get_sink", lambda: None)

    with tracing.record_tool_span("epistemology", "untraced_tool", {}) as span:
        assert span.trace_id
