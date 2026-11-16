"""Runtime skill lifecycle: schema override, retire, reactivate.

The point of the schema override is that a wrong `input_schema` is a data
problem, so it should not need a deploy. The point of the validation is that
an override which drops an argument the handler indexes is *worse* than the
wrong schema it replaces -- it turns every call into a KeyError.
"""
from __future__ import annotations

import json

import pytest

from knowledge_bases.tools import tool_admin
from knowledge_bases.tools.tool_admin import (
    SkillAdminError,
    validate_input_schema_override,
)


# -- schema validation -----------------------------------------------------------

BASE = {
    "type": "object",
    "properties": {"question": {"type": "string"}, "limit": {"type": "integer"}},
    "required": ["question"],
}


def test_a_well_formed_override_that_keeps_required_arguments_is_accepted():
    override = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Now documented."},
            "limit": {"type": "integer", "maximum": 50},
        },
        "required": ["question"],
    }
    assert validate_input_schema_override(override, BASE, handler="") == override


def test_an_override_may_relax_required_but_not_hide_the_argument():
    relaxed = {"type": "object", "properties": dict(BASE["properties"])}
    validate_input_schema_override(relaxed, BASE, handler="")

    hidden = {"type": "object", "properties": {"limit": {"type": "integer"}}}
    with pytest.raises(SkillAdminError, match="drops required argument"):
        validate_input_schema_override(hidden, BASE, handler="")


@pytest.mark.parametrize("bad", [
    {"type": "array", "properties": {}},
    {"type": "object"},
    {"type": "object", "properties": []},
    "not a schema",
])
def test_malformed_overrides_are_refused(bad):
    with pytest.raises(SkillAdminError):
        validate_input_schema_override(bad, BASE, handler="")


def test_trace_correlation_arguments_never_count_as_dropped():
    """Every seeded schema carries trace_id/session_id; an override that omits
    them is normal, not a regression."""
    base = {**BASE, "required": ["question", "trace_id", "session_id"]}
    override = {"type": "object", "properties": {"question": {"type": "string"}},
                "required": ["question"]}
    validate_input_schema_override(override, base, handler="")


def test_override_is_validated_against_what_the_handler_actually_indexes():
    """The check that makes this safe: the real handler reads
    arguments["span_id"], so an override omitting it is refused even though
    the override is perfectly valid JSON Schema."""
    handler = "knowledge_bases.tools.mcp_tools:get_span_tool"
    empty_base = {"type": "object", "properties": {}}

    with pytest.raises(SkillAdminError, match="span_id"):
        validate_input_schema_override(
            {"type": "object", "properties": {"unrelated": {"type": "string"}}},
            empty_base, handler,
        )

    ok = {"type": "object", "properties": {"span_id": {"type": "string"}},
          "required": ["span_id"]}
    assert validate_input_schema_override(ok, empty_base, handler) == ok


def test_an_unreadable_handler_falls_back_to_schema_only_validation():
    """A dotted path that no longer imports must not disable the other checks."""
    with pytest.raises(SkillAdminError, match="drops required argument"):
        validate_input_schema_override(
            {"type": "object", "properties": {}}, BASE, "does.not:exist",
        )


# -- graph plumbing --------------------------------------------------------------

class FakeGraph:
    def __init__(self, skills):
        self.skills = skills
        self.writes = []

    def run(self, query, **params):
        self.writes.append((query, params))
        if "MATCH (s:Tool {active: true})" in query:
            return self.skills
        row = next((s for s in self.skills if s["key"] == params.get("key")), None)
        if row is None:
            return []
        if "RETURN s.input_schema AS base" in query:
            return [{"base": row["input_schema"], "handler": row["handler"]}]
        if "s.input_schema_override" in query:
            return [{"previous_override": row.get("input_schema_override")}]
        if "s.active = $active" in query:
            return [{"previous": row["active"]}]
        return [row]


def _skill_row(**overrides):
    row = {"key": "project_management.list_project_blockers", "name": "n", "version": "1",
           "knowledge_base": "project_management", "active": True, "description": "d",
           "input_schema": json.dumps(BASE), "handler": "m:f"}
    row.update(overrides)
    return row


@pytest.fixture(autouse=True)
def no_reindex(monkeypatch):
    monkeypatch.setattr(tool_admin, "_reindex", lambda graph: None)


def test_override_writes_the_graph_override_not_the_seeded_base():
    graph = FakeGraph([_skill_row()])
    override = {"type": "object", "properties": {"question": {"type": "string"}},
                "required": ["question"]}
    result = tool_admin.override_skill_input_schema(
        {"key": "project_management.list_project_blockers", "input_schema": override}, graph,
    )

    assert result["cleared"] is False
    write = next(p for q, p in graph.writes if "SET s.input_schema_override" in q)
    assert json.loads(write["input_schema"]) == override
    # The seeded base is never a target of this write.
    assert all("SET s.input_schema =" not in q for q, _ in graph.writes)


def test_an_empty_schema_clears_the_override():
    graph = FakeGraph([_skill_row(input_schema_override=json.dumps(BASE))])
    result = tool_admin.override_skill_input_schema(
        {"key": "project_management.list_project_blockers", "input_schema": {}}, graph,
    )

    assert result["cleared"] is True
    write = next(p for q, p in graph.writes if "SET s.input_schema_override" in q)
    assert write["input_schema"] is None


def test_overriding_an_unknown_skill_is_refused():
    with pytest.raises(SkillAdminError, match="Unknown skill"):
        tool_admin.override_skill_input_schema(
            {"key": "nope.nothing", "input_schema": {"type": "object", "properties": {}}},
            FakeGraph([]),
        )


def test_retire_and_reactivate_flip_active_without_deleting():
    graph = FakeGraph([_skill_row()])
    retired = tool_admin.retire_skill(
        {"key": "project_management.list_project_blockers", "reason": "misfires"}, graph,
    )
    assert retired["active"] is False
    assert retired["previously_active"] is True
    assert "intact" in retired["note"]
    assert all("DELETE" not in q.upper() for q, _ in graph.writes)

    graph = FakeGraph([_skill_row(active=False)])
    back = tool_admin.reactivate_skill({"key": "project_management.list_project_blockers"}, graph)
    assert back["active"] is True


def test_retiring_an_unknown_skill_is_refused():
    with pytest.raises(SkillAdminError, match="Unknown skill"):
        tool_admin.retire_skill({"key": "nope.nothing"}, FakeGraph([]))


def test_load_skills_prefers_the_override_for_both_description_and_schema():
    """Discovery and dispatch must see overrides, or the write does nothing."""
    from knowledge_bases.tools.repository import ToolRepository

    captured = {}

    class CaptureGraph:
        def run(self, query, **params):
            captured["query"] = query
            return []

    ToolRepository(CaptureGraph()).load_tools()
    assert "coalesce(s.description_override, s.description)" in captured["query"]
    assert "coalesce(s.input_schema_override, s.input_schema)" in captured["query"]
