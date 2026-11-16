"""Queries owned by Skills/Tools/Traces."""
from __future__ import annotations

from typing import Optional

from graph import Graph
from knowledge_bases.tools.trace_store import TraceStore


def _store(graph: Optional[Graph], store: Optional[TraceStore]) -> TraceStore:
    return store or TraceStore(graph)


def get_span(
    span_id: str, graph: Optional[Graph] = None, store: Optional[TraceStore] = None,
) -> Optional[dict]:
    return _store(graph, store).get_span(span_id)


def get_trace(
    trace_id: str, graph: Optional[Graph] = None, store: Optional[TraceStore] = None,
) -> list[dict]:
    return _store(graph, store).get_trace_spans(trace_id)


def get_trace_metrics(
    trace_id: str, graph: Optional[Graph] = None, store: Optional[TraceStore] = None,
) -> Optional[dict]:
    return _store(graph, store).trace_metrics(trace_id)


__all__ = ["get_span", "get_trace", "get_trace_metrics"]
