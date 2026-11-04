"""Automatic self-improvement: triggered by failure, but never mid-task.

Self-improvement makes git-committed source edits without the usual human
gate, so the tests that matter here are about *restraint* -- when it declines
to run, and what it declines to do -- rather than about it running.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from knowledge_bases.tools import auto_improve
from knowledge_bases.tools.auto_improve import (
    DEFAULT_MIN_INTERVAL,
    maybe_run_self_improvement,
    should_run,
)


class FakeGraph:
    def __init__(self, last=None):
        self.last = last
        self.writes = []

    def run(self, query, **params):
        if "RETURN p.last_auto_self_improvement" in query:
            return [{"last": self.last}]
        if "SET p.last_auto_self_improvement" in query:
            self.writes.append(params["when"])
            return []
        return []

    def close(self):
        pass


@pytest.fixture
def enabled(monkeypatch):
    """`is_enabled` is off under pytest by design; turn it on to test the rest."""
    monkeypatch.setattr(auto_improve, "is_enabled", lambda: True)


# -- when it runs ------------------------------------------------------------------

def test_it_does_not_run_under_pytest_by_default():
    """A test recording a failed outcome is asserting something about tracing,
    not asking to spend an LLM budget rewriting descriptions against the
    developer's live graph. This tripled the suite runtime when it was missed."""
    assert auto_improve.is_enabled() is False


@pytest.mark.parametrize("value,expected", [
    ("0", False), ("false", False), ("off", False), ("no", False),
    ("1", True), ("true", True),
])
def test_it_can_be_disabled_by_environment(monkeypatch, value, expected):
    """An automatic source edit is exactly the thing an operator may want to
    turn off without editing code."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv(auto_improve.ENV_FLAG, value)

    assert auto_improve.is_enabled() is expected


def test_a_first_ever_run_is_due(enabled):
    assert should_run(FakeGraph(last=None))


def test_a_recent_run_is_not_due(enabled):
    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

    assert not should_run(FakeGraph(last=recent))


def test_an_old_run_is_due_again(enabled):
    old = (datetime.now(timezone.utc) - DEFAULT_MIN_INTERVAL - timedelta(hours=1))

    assert should_run(FakeGraph(last=old.isoformat()))


def test_an_unparseable_timestamp_is_treated_as_never_run(enabled):
    """Failing open is right here: the cost of an extra pass is bounded, the
    cost of never running again is silent."""
    assert should_run(FakeGraph(last="not-a-date"))


# -- what it does ------------------------------------------------------------------

def test_it_files_findings_without_applying_them(monkeypatch, enabled):
    """Applying edits code and commits to git. Doing that as a side effect of a
    user's task failing is a surprise no threshold tuning makes acceptable."""
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement.run_self_improvement",
        lambda graph=None: [],
    )
    result = maybe_run_self_improvement(FakeGraph(last=None))

    assert result["ran"] is True
    assert result["applied"] is False


def test_the_slot_is_consumed_before_the_pass_runs(monkeypatch, enabled):
    """A pass that crashes halfway must not retry on every later failure."""
    def explode(graph=None):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement.run_self_improvement", explode,
    )
    graph = FakeGraph(last=None)
    result = maybe_run_self_improvement(graph)

    assert graph.writes, "the run should have been marked before it failed"
    assert result["ran"] is False


def test_a_failing_pass_never_raises(monkeypatch, enabled):
    """This runs off the back of someone else's failure; turning that into a
    second, louder failure is strictly worse than not improving."""
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement.run_self_improvement",
        lambda graph=None: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert maybe_run_self_improvement(FakeGraph(last=None))["ran"] is False


def test_a_pass_that_is_not_due_does_nothing(monkeypatch, enabled):
    called = []
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement.run_self_improvement",
        lambda graph=None: called.append(1) or [],
    )
    recent = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()

    result = maybe_run_self_improvement(FakeGraph(last=recent))

    assert result == {"ran": False, "reason": "not due"}
    assert not called


# -- the trigger itself ------------------------------------------------------------

def test_a_failing_tool_call_triggers_a_pass(monkeypatch):
    """The common case, and the one that was missed: the trigger originally
    hung off `record_trace_outcome`, which only fires when a caller explicitly
    reports a bad result. The graph had 32 error spans across 169 traces and
    one recorded outcome, so the loop almost never ran."""
    from datetime import datetime, timezone

    from knowledge_bases.tools import tracing
    from knowledge_bases.tools.models import SpanEvent, SpanKind, SpanStatus

    triggered = []
    monkeypatch.setattr(tracing, "_trigger_self_improvement", lambda: triggered.append(1))
    monkeypatch.setattr(tracing, "_get_sink", lambda: type("S", (), {
        "publish_span": lambda self, event: None,
    })())

    def span(status):
        return SpanEvent(
            name="search", kind=SpanKind.skill_call, knowledge_base="x",
            started_at=datetime.now(timezone.utc), status=status,
        )

    tracing._publish(span(SpanStatus.ok))
    assert triggered == [], "a successful call is not evidence of a problem"

    tracing._publish(span(SpanStatus.error))
    assert len(triggered) == 1


def test_a_burst_of_failures_causes_one_pass_not_many(monkeypatch, enabled):
    """A tool failing twenty times in an afternoon is one signal, not twenty
    proposals."""
    passes = []
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement.run_self_improvement",
        lambda graph=None: passes.append(1) or [],
    )

    graph = FakeGraph(last=None)
    for _ in range(10):
        maybe_run_self_improvement(graph)
        # After the first call the marker is set, so the rest are not due.
        graph.last = graph.writes[-1]

    assert len(passes) == 1


def test_the_trigger_returns_immediately(monkeypatch, enabled):
    """A pass is several LLM round-trips, and the trigger fires inside the
    caller's failing call. Blocking would turn a 30ms failure into a hang --
    a worse bug than the one being diagnosed."""
    import time

    started = threading.Event()
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement.run_self_improvement",
        lambda graph=None: (started.set(), time.sleep(0.5), [])[-1],
    )
    monkeypatch.setattr(auto_improve, "maybe_run_self_improvement", lambda **kw: (
        __import__("knowledge_bases.tools.self_improvement", fromlist=["x"])
        .run_self_improvement()
    ))

    began = time.monotonic()
    auto_improve.trigger_in_background()
    elapsed = time.monotonic() - began

    assert elapsed < 0.1, "the trigger must not block the failing call"
    assert started.wait(timeout=2.0), "the pass should still run, just not inline"


def test_a_disabled_trigger_starts_no_thread(monkeypatch):
    """Off under pytest, so nothing spawns threads during the suite."""
    monkeypatch.setattr(auto_improve, "is_enabled", lambda: False)
    before = threading.active_count()

    auto_improve.trigger_in_background()

    assert threading.active_count() == before


def test_only_a_failed_outcome_triggers_a_pass(monkeypatch):
    """A successful turn is not evidence of anything needing improvement."""
    from knowledge_bases.tools import tracing
    from knowledge_bases.tools.models import OutcomeSource, TraceOutcome

    triggered = []
    monkeypatch.setattr(tracing, "_trigger_self_improvement", lambda: triggered.append(1))

    class Sink:
        def publish_outcome(self, outcome):
            pass

    monkeypatch.setattr(tracing, "_get_sink", lambda: Sink())

    tracing.record_trace_outcome(TraceOutcome(
        trace_id="t1", success=True, source=OutcomeSource.user_correction,
    ))
    assert triggered == []

    tracing.record_trace_outcome(TraceOutcome(
        trace_id="t2", success=False, source=OutcomeSource.user_correction,
    ))
    assert len(triggered) == 1
