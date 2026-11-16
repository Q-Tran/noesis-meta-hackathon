"""Wrap a graph-registered skill dispatch in a `SpanEvent`.

This is what actually closes the loop between "the orchestrator dispatches
a skill" and "the skill/tool call shows up in a trace": every call through
`knowledge_bases.mcp.dispatch_tool` is wrapped by `record_tool_span`,
which writes a `Span` node (kind=`skill_call`) to the graph on exit, tagged
with the *orchestration* `RoutingTrace.trace_id` when one is available so
`get_trace(trace_id)` can return routing and skill/tool spans together.

Tracing must never be the reason a real dispatch fails: write errors
(e.g. graph unavailable) are logged and swallowed, never raised.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator
from uuid import uuid4

from config import settings
from knowledge_bases.tools.models import SpanEvent, SpanKind, SpanStatus, TraceOutcome

logger = logging.getLogger(__name__)


class GraphSpanSink:
    """Writes spans/outcomes straight into the graph via `TraceStore`.

    Owns its own `Graph` connection (the neo4j driver is thread-safe and
    pooled) so tracing never borrows -- or blocks on -- the connection a
    handler is using for its real work.
    """

    def __init__(self, graph: Any | None = None):
        from graph import Graph
        from knowledge_bases.tools.trace_store import TraceStore

        self.store = TraceStore(graph or Graph())

    def publish_span(self, event: SpanEvent) -> None:
        self.store.insert_span(event)

    def publish_outcome(self, outcome: TraceOutcome) -> None:
        self.store.insert_outcome(outcome)


_sink: GraphSpanSink | None = None
_sink_unavailable = False


def _get_sink() -> GraphSpanSink | None:
    """Lazily construct the shared graph sink.

    Cached as unavailable after the first construction failure (e.g. no
    `neo4j` driver installed) so every subsequent skill dispatch doesn't
    retry a doomed connection; write failures on an already-open sink are
    handled per-call in `record_tool_span` instead. Disabled entirely when
    `ENABLE_SKILL_TRACING=false`.
    """
    global _sink, _sink_unavailable
    if _sink_unavailable or not settings.enable_skill_tracing:
        return None
    if _sink is None:
        try:
            _sink = GraphSpanSink()
        except Exception:
            logger.warning("Tool span tracing disabled: cannot open graph sink", exc_info=True)
            _sink_unavailable = True
            return None
    return _sink


@contextmanager
def record_tool_span(
    knowledge_base: str,
    name: str,
    arguments: dict[str, Any],
    *,
    kind: SpanKind = SpanKind.skill_call,
    skill: str | None = None,
    skill_version: str | None = None,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    client: str | None = None,
    expert_key: str | None = None,
) -> Iterator[SpanEvent]:
    """Time one skill/tool dispatch and record it as a `SpanEvent`.

    `trace_id` defaults to a fresh id when the caller has no enclosing
    orchestration trace (e.g. a domain MCP server invoked directly by a
    harness, outside `noesis-orchestrator`), so every dispatch is still
    traceable on its own even when it isn't part of a routed question.
    """
    event = SpanEvent(
        trace_id=trace_id or str(uuid4()),
        parent_span_id=parent_span_id,
        name=name,
        kind=kind,
        knowledge_base=knowledge_base,
        skill=skill or name,
        skill_version=skill_version,
        started_at=datetime.now(timezone.utc),
        session_id=session_id,
        user_id=user_id,
        client=client,
        expert_key=expert_key,
        raw_input=arguments,
    )
    try:
        yield event
    except Exception as exc:
        event.status = SpanStatus.error
        event.error = str(exc)
        raise
    finally:
        event.ended_at = datetime.now(timezone.utc)
        event.duration_ms = int(
            (event.ended_at - event.started_at).total_seconds() * 1000
        )
        _publish(event)


def _publish(event: SpanEvent) -> None:
    sink = _get_sink()
    if sink is None:
        return
    try:
        sink.publish_span(event)
    except Exception:
        logger.warning("Failed to record skill span %s", event.span_id, exc_info=True)
        return

    # A failing tool call is the most common failure there is, and for a long
    # time nothing acted on it: the trigger hung off `record_trace_outcome`,
    # which only fires when a caller explicitly reports a bad result. The graph
    # had 32 error spans across 169 traces and exactly one recorded outcome, so
    # in practice the "automatic" loop almost never ran. Triggering from the
    # span itself is what makes a task failing actually mean something.
    if event.status == SpanStatus.error:
        _trigger_self_improvement()


def record_trace_outcome(outcome: TraceOutcome) -> bool:
    """Record one `TraceOutcome` for an existing `trace_id`.

    A client-side agent (or a human confirming/correcting an answer)
    reports whether a trace's result held up, feeding `skill_health()`'s
    `user_correction_count` signal. Shares `record_tool_span`'s cached
    sink/availability handling and never raises -- an unavailable graph
    degrades to "not recorded" rather than failing the caller. Returns
    whether the outcome was actually written.
    """
    sink = _get_sink()
    if sink is None:
        return False
    try:
        sink.publish_outcome(outcome)
    except Exception:
        logger.warning("Failed to record trace outcome %s", outcome.outcome_id, exc_info=True)
        return False

    # A recorded failure is the trigger for self-improvement, and this is the
    # right moment for it: the turn is over (its outcome is what we just
    # wrote), so nothing fires mid-task. The pass is debounced to at most once
    # a day and files findings without applying them -- see
    # `knowledge_bases/tools/auto_improve.py` for why both of those matter.
    if not outcome.success:
        _trigger_self_improvement()
    return True


def _trigger_self_improvement() -> None:
    """Best-effort automatic pass. Never raises: this runs off the back of
    someone else's failure, and turning that into a second failure is worse
    than not improving."""
    try:
        from knowledge_bases.tools.auto_improve import trigger_in_background

        # Backgrounded: a pass is several LLM round-trips, and this runs inside
        # the failing call. Blocking here would make a 30ms failure take tens
        # of seconds to surface.
        trigger_in_background()
    except Exception:
        logger.warning("Could not trigger self-improvement", exc_info=True)
