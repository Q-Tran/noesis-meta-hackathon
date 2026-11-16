"""Automated outcome scoring for traces.

No human supervision is required: every finished trace gets a composite
score from deterministic verifiers, behavioral signals already recorded in
`Span`s, recorded user corrections, and an optional LLM judge. Scores are
append-only `TraceOutcome`s; the judge, rubric, and weights are versioned
(`SCORER_VERSION`) so a drift is visible rather than silent.

The scorer is deliberately conservative about calibration: `run_calibration`
re-scores a frozen, known-good trace set and fails loudly if the current
scorer no longer reproduces it.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from graph import Graph
from knowledge_bases.tools.trace_store import TraceStore
from knowledge_bases.tools.models import OutcomeSource, TraceOutcome

SCORER_VERSION = "outcome-scorer/1.0"
JUDGE_RUBRIC_VERSION = "judge-rubric/1.0"

# Gate thresholds used by the reflection eval gate. `correctness` and
# `groundedness` are hard signals: a change may not trade them for speed.
REGRESSION_TOLERANCE = 2.0
HARD_SIGNAL_TOLERANCE = 0.0


def _loads(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


@dataclass(frozen=True)
class JudgeScore:
    correctness: float = 100.0
    groundedness: float = 100.0
    completeness: float = 100.0
    reason: str = ""


@dataclass(frozen=True)
class ScoreBreakdown:
    correctness: float
    groundedness: float
    behavior: float
    cost: float
    judge: JudgeScore | None = None
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def composite(self) -> float:
        return _clamp(
            0.45 * self.correctness
            + 0.30 * self.groundedness
            + 0.15 * self.behavior
            + 0.10 * self.cost
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "correctness": self.correctness,
            "groundedness": self.groundedness,
            "behavior": self.behavior,
            "cost": self.cost,
            "composite": self.composite,
            "judge": (
                {"correctness": self.judge.correctness, "groundedness": self.judge.groundedness,
                 "completeness": self.judge.completeness, "reason": self.judge.reason}
                if self.judge is not None else None
            ),
            "signals": self.signals,
        }


def _behavior_score(spans: list[dict]) -> tuple[float, dict[str, Any]]:
    if not spans:
        return 100.0, {"tool_calls": 0, "error_calls": 0, "empty_calls": 0, "loops": 0}
    names = [str(s.get("name") or "") for s in spans]
    counts = Counter(names)
    loops = sum(1 for name, count in counts.items() if count >= 3 and any(
        marker in name for marker in ("search", "lookup", "find", "resolve", "get_")
    ))
    errors = sum(1 for s in spans if s.get("status") == "error")
    empty = sum(1 for s in spans if s.get("result_count") == 0)
    score = 100.0 - (30.0 * errors / len(spans)) - (10.0 * min(loops, 4)) - (5.0 * empty)
    return _clamp(score), {
        "tool_calls": len(spans),
        "error_calls": errors,
        "empty_calls": empty,
        "loops": loops,
    }


def _cost_score(spans: list[dict]) -> tuple[float, dict[str, Any]]:
    tokens = sum(int(s.get("total_tokens") or 0) for s in spans)
    cost = sum(float(s.get("cost_usd") or 0.0) for s in spans)
    duration = sum(int(s.get("duration_ms") or 0) for s in spans)
    # Cost is intentionally a soft signal: more work is not failure.
    score = 100.0 - min(20.0, tokens / 2500.0) - min(10.0, cost * 20.0) - min(10.0, duration / 60000.0)
    return _clamp(score), {"total_tokens": tokens, "cost_usd": cost, "duration_ms": duration}


def score_trace(
    trace_id: str,
    store: TraceStore | None = None,
    *,
    judge: Callable[[dict, list[dict]], JudgeScore] | None = None,
    outcomes: list[dict] | None = None,
) -> ScoreBreakdown:
    """Score one trace from its spans and latest outcome.

    Correctness starts from the recorded outcome (no outcome is treated as
    neutral, not as success). Groundedness penalizes empty retrieval and
    unsupported claims inferred from spans with zero results. The optional
    judge can only adjust correctness/groundedness; it never hides the
    deterministic signals beneath it.
    """
    store = store or TraceStore()
    spans = store.get_trace_spans(trace_id)
    outcome = outcomes[-1] if outcomes else store.current_outcome(trace_id)
    outcome_success = (outcome or {}).get("success")
    outcome_source = (outcome or {}).get("source")
    corrected = outcome_success is False and outcome_source == "user_correction"

    error_rate = sum(1 for s in spans if s.get("status") == "error") / len(spans) if spans else 0.0
    correctness = 100.0
    if outcome_success is True:
        correctness -= 0.0
    elif outcome_success is False:
        correctness -= 35.0 if corrected else 20.0
    correctness -= 20.0 * error_rate

    empty = sum(1 for s in spans if s.get("result_count") == 0)
    groundedness = _clamp(100.0 - (15.0 * empty) - (10.0 if corrected else 0.0))
    behavior, behavior_signals = _behavior_score(spans)
    cost, cost_signals = _cost_score(spans)

    judge_score: JudgeScore | None = None
    if judge is not None:
        judge_score = judge(outcome or {}, spans)
        correctness = _clamp(0.65 * correctness + 0.35 * judge_score.correctness)
        groundedness = _clamp(0.65 * groundedness + 0.35 * judge_score.groundedness)

    signals = {
        "has_spans": bool(spans),
        "trace_status": "completed" if outcome_success is not False else "failed",
        "current_outcome_source": outcome_source,
        "current_outcome_success": outcome_success,
        "corrected": corrected,
        "error_rate": error_rate,
        **behavior_signals,
        **cost_signals,
    }
    return ScoreBreakdown(
        correctness=_clamp(correctness),
        groundedness=_clamp(groundedness),
        behavior=behavior,
        cost=cost,
        judge=judge_score,
        signals=signals,
    )


def record_trace_score(
    trace_id: str,
    store: TraceStore | None = None,
    *,
    judge: Callable[[dict, list[dict]], JudgeScore] | None = None,
) -> TraceOutcome:
    """Score a trace and append its `TraceOutcome` (source `automated`)."""
    store = store or TraceStore()
    breakdown = score_trace(trace_id, store=store, judge=judge)
    outcome = TraceOutcome(
        trace_id=trace_id,
        success=breakdown.composite >= 60.0,
        source=OutcomeSource.automated,
        score=breakdown.composite,
        score_breakdown_json=json.dumps(breakdown.as_dict(), sort_keys=True),
        scorer_version=SCORER_VERSION,
        reason=f"automated score {breakdown.composite:.1f} ({SCORER_VERSION})",
    )
    store.insert_outcome(outcome)
    return outcome


def score_worker_traces(
    worker_key: str,
    graph: Graph | None = None,
    store: TraceStore | None = None,
    *,
    since: str | None = None,
) -> list[dict]:
    """Score every trace that ran under a worker key (or, during migration,
    under an expert key that maps to the worker)."""
    store = store or TraceStore(graph)
    query = (
        "MATCH (s:Span) "
        "WHERE s.expert_key = $worker_key "
        "RETURN DISTINCT s.trace_id AS trace_id"
    )
    if since:
        query = (
            "MATCH (s:Span) "
            "WHERE s.expert_key = $worker_key AND s.started_at >= datetime($since) "
            "RETURN DISTINCT s.trace_id AS trace_id"
        )
    rows = store.graph.run(query, worker_key=worker_key, since=since)
    results = []
    for row in rows:
        trace_id = row["trace_id"]
        breakdown = score_trace(trace_id, store=store)
        results.append({"trace_id": trace_id, "score": breakdown.composite, "breakdown": breakdown.as_dict()})
    return results


def compare_scores(baseline: ScoreBreakdown, candidate: ScoreBreakdown) -> dict:
    """The eval-gate comparison. Composite must not regress beyond
    tolerance; correctness and groundedness may not regress at all."""
    composite_delta = candidate.composite - baseline.composite
    correctness_delta = candidate.correctness - baseline.correctness
    groundedness_delta = candidate.groundedness - baseline.groundedness
    passed = (
        composite_delta >= -REGRESSION_TOLERANCE
        and correctness_delta >= -HARD_SIGNAL_TOLERANCE
        and groundedness_delta >= -HARD_SIGNAL_TOLERANCE
    )
    return {
        "passed": passed,
        "composite_delta": composite_delta,
        "correctness_delta": correctness_delta,
        "groundedness_delta": groundedness_delta,
        "reason": (
            "candidate passes" if passed
            else "candidate regresses composite/correctness/groundedness beyond tolerance"
        ),
    }


def run_calibration(
    graph: Graph | None = None,
    *,
    trace_ids: list[str],
    expected: dict[str, float],
    tolerance: float = 10.0,
    store: TraceStore | None = None,
) -> dict:
    """Re-score a frozen calibration set and confirm the current scorer
    still reproduces it. This is the drift alarm for scorer changes."""
    store = store or TraceStore(graph)
    results = []
    for trace_id in trace_ids:
        actual = score_trace(trace_id, store=store).composite
        wanted = expected.get(trace_id)
        if wanted is None:
            continue
        results.append({
            "trace_id": trace_id,
            "expected": wanted,
            "actual": actual,
            "within_tolerance": abs(actual - wanted) <= tolerance,
        })
    return {
        "passed": bool(results) and all(r["within_tolerance"] for r in results),
        "scorer_version": SCORER_VERSION,
        "results": results,
    }
