"""Dispatch-level argument validation."""
from __future__ import annotations


def test_a_missing_required_argument_names_the_tool_and_the_field():
    """A missing argument used to reach the handler and surface as a bare
    `KeyError: 'trace_id'`, which names the key but not the tool, the caller,
    or the fact that it was missing rather than wrong. Callers then retried
    the identical call."""
    import pytest

    from knowledge_bases.mcp import _require_schema_fields

    class Tool:
        key = "tools.get_trace_metrics"
        input_schema = {
            "type": "object",
            "properties": {"trace_id": {"type": "string"}},
            "required": ["trace_id"],
        }

    with pytest.raises(ValueError) as exc:
        _require_schema_fields(Tool(), {})

    message = str(exc.value)
    assert "tools.get_trace_metrics" in message
    assert "trace_id" in message
    assert "not supplied" in message


def test_a_tool_with_no_required_fields_accepts_anything():
    from knowledge_bases.mcp import _require_schema_fields

    class Tool:
        key = "x.y"
        input_schema = {"type": "object", "properties": {}}

    _require_schema_fields(Tool(), {})
