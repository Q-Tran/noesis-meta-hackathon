"""In-memory `TraceStore` for unit tests that need span/outcome telemetry
without a live graph. Only the graph round-trip (`_spans_with_outcome` and
the simple point reads) is faked; `skill_health`/`skill_metrics_for` run the
real aggregation code. `tests/test_skills_trace_store.py` covers the Cypher
against a live Neo4j when one is reachable."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from knowledge_bases.tools.models import SpanEvent, TraceOutcome
from knowledge_bases.tools.trace_store import TraceStore, _enum_value, _utc


class InMemoryTraceStore(TraceStore):
    def __init__(self):
        self.spans: dict[str, SpanEvent] = {}
        self.outcomes: dict[str, TraceOutcome] = {}

    def insert_span(self, event: SpanEvent) -> None:
        self.spans.setdefault(event.span_id, event)

    def insert_outcome(self, outcome: TraceOutcome) -> None:
        self.outcomes.setdefault(outcome.outcome_id, outcome)

    def _props(self, event: SpanEvent) -> dict:
        props = event.model_dump(mode="python")
        props["kind"] = _enum_value(event.kind)
        props["status"] = _enum_value(event.status)
        return props

    def get_span(self, span_id: str) -> Optional[dict]:
        event = self.spans.get(span_id)
        return self._props(event) if event else None

    def get_trace_spans(self, trace_id: str) -> list[dict]:
        events = sorted(
            (e for e in self.spans.values() if e.trace_id == trace_id),
            key=lambda e: _utc(e.started_at),
        )
        return [self._props(e) for e in events]

    def current_outcome(self, trace_id: str) -> Optional[dict]:
        outcomes = [o for o in self.outcomes.values() if o.trace_id == trace_id]
        if not outcomes:
            return None
        latest = max(outcomes, key=lambda o: _utc(o.recorded_at))
        return {
            "trace_id": latest.trace_id, "success": latest.success, "reason": latest.reason,
            "source": _enum_value(latest.source), "recorded_at": latest.recorded_at,
        }

    def trace_metrics(self, trace_id: str) -> Optional[dict]:
        spans = [e for e in self.spans.values() if e.trace_id == trace_id]
        if not spans:
            return None
        outcome = self.current_outcome(trace_id)
        return {
            "trace_id": trace_id,
            "total_tokens": sum(e.total_tokens or 0 for e in spans),
            "total_cost_usd": sum(e.cost_usd or 0.0 for e in spans),
            "tool_call_count": sum(1 for e in spans if _enum_value(e.kind) == "tool_call"),
            "success": outcome["success"] if outcome else None,
        }

    def _spans_with_outcome(self, since: Any, skill: str | None = None) -> list[dict]:
        cutoff = _utc(since)
        rows = []
        for e in self.spans.values():
            if e.skill is None or (skill and e.skill != skill) or _utc(e.started_at) < cutoff:
                continue
            outcome = self.current_outcome(e.trace_id)
            rows.append({
                "skill": e.skill, "knowledge_base": e.knowledge_base,
                "status": _enum_value(e.status), "duration_ms": e.duration_ms,
                "error": e.error, "trace_id": e.trace_id, "user_id": e.user_id,
                "client": e.client, "expert_key": e.expert_key,
                "outcome_success": outcome["success"] if outcome else None,
                "outcome_source": outcome["source"] if outcome else None,
                "outcome_reason": outcome["reason"] if outcome else None,
            })
        return rows
