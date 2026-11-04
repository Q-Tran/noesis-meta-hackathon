"""Automatic tracing integration for AI harnesses (e.g. OpenCode, Claude Desktop).

Bridges harness execution events directly into the Noesis knowledge graph
telemetry (`:RoutingTrace`, `:Span`, `:Session`, and `:TraceOutcome` nodes)
without requiring model-facing manual tool calls.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from knowledge_bases.tools.active_trace import (
    current_scope,
    current_session_id,
    resolve_identity,
    resolve_trace_context,
    set_active_identity,
    set_active_trace,
)
from knowledge_bases.tools.models import OutcomeSource, SpanEvent, SpanKind, SpanStatus, TraceOutcome
from knowledge_bases.tools.tracing import _get_sink, record_trace_outcome

logger = logging.getLogger(__name__)


def record_turn_start(
    prompt: str,
    *,
    session_id: str | None = None,
    trace_id: str | None = None,
    client: str = "opencode",
    user_id: str | None = None,
    graph: Any | None = None,
) -> str:
    """Record the start of a turn in the harness, establishing a correlated trace."""
    resolved_trace_id = trace_id or str(uuid4())
    resolved_session_id = session_id or current_session_id()

    set_active_trace(resolved_trace_id)
    set_active_identity(user_id=user_id, client=client)

    sink = _get_sink()
    if sink is not None:
        try:
            sink.store.record_turn_start(
                trace_id=resolved_trace_id,
                session_id=resolved_session_id,
                question=prompt,
                client=client,
                user_id=user_id,
            )
        except Exception:
            logger.warning("Failed to record harness turn start into graph", exc_info=True)

    return resolved_trace_id


def record_tool_execution(
    tool_name: str,
    *,
    raw_input: Any = None,
    raw_output: Any = None,
    duration_ms: int = 0,
    status: str = "ok",
    error: str | None = None,
    trace_id: str | None = None,
    session_id: str | None = None,
    knowledge_base: str = "harness",
    client: str = "opencode",
    user_id: str | None = None,
    expert_key: str | None = None,
) -> str:
    """Record an executed tool (built-in harness tool or MCP tool) as a :Span node."""
    resolved_trace_id, resolved_session_id = resolve_trace_context(trace_id, session_id)
    resolved_trace_id = resolved_trace_id or str(uuid4())

    span_status = SpanStatus.error if status == "error" or error else SpanStatus.ok
    now = datetime.now(timezone.utc)

    event = SpanEvent(
        trace_id=resolved_trace_id,
        session_id=resolved_session_id,
        name=tool_name,
        kind=SpanKind.tool_call,
        knowledge_base=knowledge_base,
        skill=tool_name,
        started_at=now,
        ended_at=now,
        duration_ms=duration_ms,
        status=span_status,
        error=error,
        raw_input=raw_input if isinstance(raw_input, (dict, list)) else {"input": raw_input},
        raw_output=raw_output,
        client=client or resolve_identity().get("client"),
        user_id=user_id or resolve_identity().get("user_id"),
        expert_key=expert_key or resolve_identity().get("expert_key"),
    )

    sink = _get_sink()
    if sink is not None:
        try:
            sink.publish_span(event)
        except Exception:
            logger.warning("Failed to record harness tool span %s", event.span_id, exc_info=True)

    # Automatically record outcome on failure
    if span_status == SpanStatus.error:
        record_trace_outcome(
            TraceOutcome(
                trace_id=resolved_trace_id,
                success=False,
                reason=f"Tool error in {tool_name}: {error}" if error else f"Tool {tool_name} failed",
                source=OutcomeSource.harness_inferred,
            )
        )

    return event.span_id


def record_turn_completion(
    trace_id: str | None = None,
    *,
    success: bool = True,
    error: str | None = None,
    session_id: str | None = None,
    source: str = "harness_inferred",
) -> bool:
    """Record the completion of a turn from the harness, updating status and outcome."""
    resolved_trace_id, resolved_session_id = resolve_trace_context(trace_id, session_id)
    if not resolved_trace_id:
        return False

    status = "completed" if success else "error"
    sink = _get_sink()
    if sink is not None:
        try:
            sink.store.update_trace(
                trace_id=resolved_trace_id,
                status=status,
                error=error,
                session_id=resolved_session_id,
            )
        except Exception:
            logger.warning("Failed to update trace %s completion status", resolved_trace_id, exc_info=True)

    outcome_source = OutcomeSource.harness_inferred
    try:
        outcome_source = OutcomeSource(source)
    except ValueError:
        pass

    outcome = TraceOutcome(
        trace_id=resolved_trace_id,
        success=success,
        reason=error,
        source=outcome_source,
    )
    return record_trace_outcome(outcome)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Noesis harness tracing CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # turn
    turn_parser = subparsers.add_parser("turn", help="Record turn start")
    turn_parser.add_argument("--prompt", required=True, help="User prompt/question text")
    turn_parser.add_argument("--session-id", help="Session ID")
    turn_parser.add_argument("--trace-id", help="Trace ID")
    turn_parser.add_argument("--client", default="opencode", help="Harness client name")
    turn_parser.add_argument("--user-id", help="User ID")

    # span
    span_parser = subparsers.add_parser("span", help="Record tool execution span")
    span_parser.add_argument("--tool", required=True, help="Tool name")
    span_parser.add_argument("--input", default="{}", help="Input JSON")
    span_parser.add_argument("--output", default="{}", help="Output JSON or string")
    span_parser.add_argument("--duration-ms", type=int, default=0, help="Duration in ms")
    span_parser.add_argument("--status", choices=["ok", "error"], default="ok", help="Status")
    span_parser.add_argument("--error", help="Error message")
    span_parser.add_argument("--trace-id", help="Trace ID")
    span_parser.add_argument("--session-id", help="Session ID")
    span_parser.add_argument("--client", default="opencode", help="Client name")

    # complete
    comp_parser = subparsers.add_parser("complete", help="Record turn completion")
    comp_parser.add_argument("--trace-id", help="Trace ID")
    comp_parser.add_argument("--session-id", help="Session ID")
    comp_parser.add_argument("--success", type=lambda s: s.lower() in ("true", "1", "yes"), default=True)
    comp_parser.add_argument("--error", help="Error message if failed")
    comp_parser.add_argument("--source", default="harness_inferred", help="Outcome source")

    args = parser.parse_args()

    if args.subcommand == "turn":
        trace_id = record_turn_start(
            prompt=args.prompt,
            session_id=args.session_id,
            trace_id=args.trace_id,
            client=args.client,
            user_id=args.user_id,
        )
        print(json.dumps({"trace_id": trace_id, "session_id": args.session_id or current_session_id()}))
    elif args.subcommand == "span":
        raw_in = json.loads(args.input) if args.input.startswith(("{", "[")) else args.input
        raw_out = json.loads(args.output) if args.output.startswith(("{", "[")) else args.output
        span_id = record_tool_execution(
            tool_name=args.tool,
            raw_input=raw_in,
            raw_output=raw_out,
            duration_ms=args.duration_ms,
            status=args.status,
            error=args.error,
            trace_id=args.trace_id,
            session_id=args.session_id,
            client=args.client,
        )
        print(json.dumps({"span_id": span_id}))
    elif args.subcommand == "complete":
        ok = record_turn_completion(
            trace_id=args.trace_id,
            success=args.success,
            error=args.error,
            session_id=args.session_id,
            source=args.source,
        )
        print(json.dumps({"recorded": ok}))


if __name__ == "__main__":
    _cli()
