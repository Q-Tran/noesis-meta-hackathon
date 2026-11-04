"""`get_noesis_routing_trace_tool` merges the graph's routing decision with
the DuckDB skill/tool spans it produced, correlated by trace_id."""
from knowledge_bases.orchestration import mcp_tools


def test_get_noesis_routing_trace_tool_merges_routing_and_spans(monkeypatch):
    monkeypatch.setattr(
        mcp_tools.RoutingRepository, "get_trace",
        lambda self, trace_id: {"trace": {"id": trace_id, "status": "completed"}},
    )
    monkeypatch.setattr(
        mcp_tools, "get_trace",
        lambda trace_id, graph=None: [{"span_id": "span-1", "trace_id": trace_id}],
    )
    monkeypatch.setattr(
        mcp_tools, "get_trace_metrics",
        lambda trace_id, graph=None: {"trace_id": trace_id, "tool_call_count": 1},
    )

    result = mcp_tools.get_noesis_routing_trace_tool({"trace_id": "trace-1"}, graph=object())

    assert result["routing"]["trace"]["id"] == "trace-1"
    assert result["spans"] == [{"span_id": "span-1", "trace_id": "trace-1"}]
    assert result["metrics"]["tool_call_count"] == 1
