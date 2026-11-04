"""`search_noesis_tools` makes the entire graph-registered tool catalog
discoverable by plain-language need, so a caller doesn't need to already
know a tool's exact name (or go looking for a third-party MCP server) --
every Noesis tool is already deployed everywhere a `noesis-orchestrator`
process runs, so a match is immediately callable with no separate install
step (see `knowledge_bases.tools.search`)."""
from __future__ import annotations

import json

from knowledge_bases.tools.search import search_noesis_tools, search_tokens


class NoIndex:
    """Stands in for `CatalogIndex` when no vector store is available."""

    def search_skills(self, query, top_k=20):
        return {}


class FakeGraph:
    def __init__(self, rows):
        self.rows = rows

    def run(self, query, **params):
        return self.rows


def _skill_row(key, knowledge_base, name, description, input_schema=None):
    return {
        "key": key, "name": name, "version": "1.0.0",
        "knowledge_base": knowledge_base, "active": True,
        "description": description,
        "input_schema": json.dumps(input_schema or {"type": "object", "properties": {}}),
        "handler": "module:function",
    }


def test_search_tokens_strips_stopwords_and_short_tokens():
    tokens = search_tokens("I need to check project blockers")

    assert tokens == {"check", "project", "blockers"}
    assert "need" not in tokens  # filler word, deliberately in STOPWORDS
    assert "to" not in tokens


def test_search_ranks_the_best_name_match_first():
    graph = FakeGraph([
        _skill_row(
            "project_management.list_project_blockers", "project_management", "List Project Blockers",
            "List blocked work, assignees, projects, and delayed decisions.",
        ),
        _skill_row(
            "project_management.get_project_hierarchy", "project_management", "Get Project Hierarchy",
            "Get a project with its epics, stories, and tasks.",
        ),
    ])

    results = search_noesis_tools("I need to check project blockers", graph=graph, index=NoIndex(), llm=None)

    assert results[0]["tool_name"] == "list_project_blockers"
    assert results[0]["score"] > results[1]["score"]


def test_search_returns_only_tools_that_actually_match():
    graph = FakeGraph([
        _skill_row(
            "project_management.list_project_blockers", "project_management", "List Project Blockers",
            "List blocked work, assignees, projects, and delayed decisions.",
        ),
        _skill_row(
            "methodology.consult_workflow", "methodology", "Consult Workflow",
            "Continue the product development workflow.",
        ),
    ])

    results = search_noesis_tools("project blockers", graph=graph, index=NoIndex(), llm=None)

    assert len(results) == 1
    assert results[0]["tool_name"] == "list_project_blockers"


def test_search_returns_the_real_mcp_tool_name_not_the_prefixed_skill_key():
    graph = FakeGraph([
        _skill_row(
            "business_dictionary.resolve_business_language", "business_dictionary",
            "Resolve Business Language", "Resolve terms embedded in a question.",
        ),
    ])

    results = search_noesis_tools("resolve business language", graph=graph, index=NoIndex(), llm=None)

    assert results[0]["tool_name"] == "resolve_business_language"
    assert results[0]["skill_key"] == "business_dictionary.resolve_business_language"


def test_search_respects_the_knowledge_base_filter():
    graph = FakeGraph([_skill_row("project_management.list_project_blockers", "project_management", "X", "Y")])

    search_noesis_tools("project", graph=graph, knowledge_base="project_management")

    assert graph.rows  # sanity: fixture wired correctly


def test_search_respects_the_limit():
    graph = FakeGraph([
        _skill_row(f"project_management.tool_{i}", "project_management", f"Project Tool {i}", "project project")
        for i in range(5)
    ])

    results = search_noesis_tools("project", limit=2, graph=graph, index=NoIndex(), llm=None)

    assert len(results) == 2


def test_search_includes_the_tools_own_input_schema_so_a_caller_can_use_it_immediately():
    graph = FakeGraph([
        _skill_row(
            "project_management.list_project_blockers", "project_management", "List Project Blockers",
            "List blocked work.", input_schema={"type": "object", "properties": {"x": {}}},
        ),
    ])

    results = search_noesis_tools("project blockers", graph=graph, index=NoIndex(), llm=None)

    assert results[0]["input_schema"] == {"type": "object", "properties": {"x": {}}}


def test_search_returns_empty_list_for_a_need_with_no_matches():
    graph = FakeGraph([
        _skill_row("methodology.consult_workflow", "methodology", "Consult Workflow", "Continue the product development workflow."),
    ])

    results = search_noesis_tools("xyzzy plugh", graph=graph, index=NoIndex(), llm=None)

    assert results == []


def test_a_result_carries_the_tool_key_and_still_the_old_skill_key():
    """`skill_key` was this field's name when a tool was called a skill. New
    callers read `tool_key`; the old name is kept so anything already reading
    it does not break on the rename."""
    graph = FakeGraph([
        _skill_row(
            "project_management.list_project_blockers", "project_management", "List Project Blockers",
            "List blocked work, assignees, projects, and delayed decisions.",
        ),
    ])

    results = search_noesis_tools(
        "project blockers", graph=graph, index=NoIndex(), llm=None,
    )

    assert results[0]["tool_key"] == "project_management.list_project_blockers"
    assert results[0]["skill_key"] == results[0]["tool_key"]
