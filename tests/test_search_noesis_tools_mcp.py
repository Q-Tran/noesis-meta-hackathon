"""`search_noesis_tools_tool` is the MCP-callable entry point for
`knowledge_bases.tools.search.search_noesis_tools` -- pins the handler's
own argument wiring (need/limit/knowledge_base) separately from the search
scoring logic itself (see `tests/test_skills_search.py`)."""
from __future__ import annotations

import json

from knowledge_bases.orchestration.mcp_tools import search_noesis_tools_tool


class NoIndex:
    """Stands in for `CatalogIndex` when no vector store is available."""

    def search_skills(self, query, top_k=20):
        return {}


class FakeGraph:
    def __init__(self, rows):
        self.rows = rows

    def run(self, query, **params):
        return self.rows


def _skill_row(key, knowledge_base, name, description):
    return {
        "key": key, "name": name, "version": "1.0.0",
        "knowledge_base": knowledge_base, "active": True,
        "description": description,
        "input_schema": json.dumps({"type": "object", "properties": {}}),
        "handler": "module:function",
    }


def test_search_noesis_tools_tool_finds_a_matching_tool():
    graph = FakeGraph([
        _skill_row(
            "project_management.list_project_blockers", "project_management", "List Project Blockers",
            "List blocked work, assignees, projects, and delayed decisions.",
        ),
    ])

    results = search_noesis_tools_tool({"need": "check project blockers"}, graph)

    assert results[0]["tool_name"] == "list_project_blockers"


def test_search_noesis_tools_tool_passes_through_the_knowledge_base_filter():
    graph = FakeGraph([_skill_row("project_management.list_project_blockers", "project_management", "X", "Y")])

    # No assertion on graph internals beyond "it ran without error" --
    # `ToolRepository.load_tools` receiving the filter is covered by
    # `tests/test_skills_search.py`; this only pins the tool wiring it.
    search_noesis_tools_tool(
        {"need": "project", "knowledge_base": "project_management"}, graph,
    )


def test_search_noesis_tools_tool_passes_through_the_limit():
    graph = FakeGraph([
        _skill_row(f"project_management.tool_{i}", "project_management", f"Project Tool {i}", "project project")
        for i in range(5)
    ])

    results = search_noesis_tools_tool({"need": "project", "limit": 2}, graph)

    assert len(results) == 2


def test_search_noesis_tools_tool_defaults_limit_to_ten():
    graph = FakeGraph([
        _skill_row(f"project_management.tool_{i}", "project_management", f"Project Tool {i}", "project project")
        for i in range(15)
    ])

    results = search_noesis_tools_tool({"need": "project"}, graph)

    assert len(results) == 10
