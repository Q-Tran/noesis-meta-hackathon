"""Harness Agent Skills are graph data, looked up dynamically.

Two different things used to share the word "skill": a callable operation with
a handler (now `Tool`) and an instruction bundle a harness discloses to a model
(now `AgentSkill`). These tests pin the distinction and the dynamic lookup.
"""
from __future__ import annotations

import pytest

from knowledge_bases.orchestration.agent_skills import AgentSkillRepository
from knowledge_bases.orchestration.agent_skills_catalog import (
    bootstrap_agent_skills,
    render_body,
    render_description,
    skill_name,
)
from knowledge_bases.orchestration.models import (
    MAX_SKILL_DESCRIPTION_CHARS,
    AgentSkill,
    AgentSkillError,
)


def _tools():
    return [
        {"key": "project_management.list_project_blockers", "knowledge_base": "project_management",
         "operation": "list_project_blockers", "description": "Blockers for a project."},
        {"key": "project_management.list_epics", "knowledge_base": "project_management",
         "operation": "list_epics", "description": "Epics of a project."},
    ]


class FakeGraph:
    """Records writes and replays canned rows, like the other repository fakes."""

    def __init__(self, rows: list[dict] | None = None):
        self.rows = rows or []
        self.queries: list[tuple[str, dict]] = []
        self.rels: list[tuple] = []

    def run(self, query: str, **params):
        self.queries.append((query, params))
        return self.rows

    def upsert_rel(self, start_label, start_key, rel_type, end_label, end_key,
                   props=None):
        self.rels.append((start_label, start_key, rel_type, end_label, end_key))
        return []


# -- the distinction itself --------------------------------------------------------

def test_an_agent_skill_is_not_a_tool():
    """An AgentSkill carries instructions and no handler; that is the whole
    point of splitting the label. If this ever gains a handler, the collision
    the rename removed has come back."""
    skill = AgentSkill(key="noesis-project-management", name="noesis-project-management", description="d")

    assert not hasattr(skill, "handler")
    assert not hasattr(skill, "input_schema")


def test_the_two_labels_are_owned_deliberately():
    from knowledge_bases.registry import KNOWLEDGE_BASES

    assert "Tool" in KNOWLEDGE_BASES["tools"].labels
    assert "AgentSkill" not in KNOWLEDGE_BASES["tools"].labels
    # Instructions an agent is given, not an operation it can call.
    assert "AgentSkill" in KNOWLEDGE_BASES["orchestration"].labels


# -- harness constraints are enforced, not hoped for -------------------------------

def test_a_skill_without_a_description_is_refused():
    """A harness drops a description-less skill silently and entirely, so the
    graph must refuse to store one."""
    with pytest.raises(AgentSkillError, match="no description"):
        AgentSkill(key="k", name="noesis-project-management", description="  ").validate_for_harness()


def test_an_over_long_description_is_refused():
    with pytest.raises(AgentSkillError, match="harness limit"):
        AgentSkill(
            key="k", name="noesis-project-management", description="x" * 2000,
        ).validate_for_harness()


@pytest.mark.parametrize("name", [
    "noesis_execution",   # underscores are illegal
    "Noesis-Execution",   # uppercase is illegal
    "-delivery", "delivery-", "noesis--execution",
])
def test_names_a_harness_would_reject_are_refused(name):
    with pytest.raises(AgentSkillError):
        AgentSkill(key="k", name=name, description="d").validate_for_harness()


def test_generated_names_are_always_harness_legal():
    """Knowledge base keys contain underscores; skill names may not."""
    assert skill_name("semantic_layer") == "noesis-semantic-layer"
    for key in ("semantic_layer", "business_dictionary", "strategy"):
        AgentSkill(
            key=skill_name(key), name=skill_name(key), description="d",
        ).validate_for_harness()


def test_a_long_description_is_clipped_rather_than_dropped():
    """Losing the tail of a sentence beats losing the whole skill."""
    text = render_description("Execution", "word " * 500)

    assert len(text) <= MAX_SKILL_DESCRIPTION_CHARS
    AgentSkill(key="k", name="noesis-project-management", description=text).validate_for_harness()


# -- the body is derived from the live catalog -------------------------------------

def test_the_body_lists_the_tools_it_was_given():
    body = render_body("project_management", "Execution", _tools())

    assert "`list_project_blockers`" in body
    assert "`list_epics`" in body
    assert "call_noesis_tool" in body


def test_the_body_says_it_was_read_from_the_graph_not_exported():
    body = render_body("project_management", "Execution", _tools())

    assert "read from the graph" in body
    assert "search_noesis_tools" in body


def test_a_retired_tool_disappears_from_a_regenerated_body():
    """The behaviour the static export could not deliver."""
    before = render_body("project_management", "Execution", _tools())
    after = render_body("project_management", "Execution", _tools()[:1])

    assert "`list_epics`" in before
    assert "`list_epics`" not in after


def test_one_skill_is_generated_per_knowledge_base_that_owns_tools():
    skills = bootstrap_agent_skills(_tools())

    assert [s.key for s in skills] == ["noesis-project-management"]
    assert skills[0].knowledge_base == "project_management"
    for skill in skills:
        skill.validate_for_harness()


def test_a_knowledge_base_that_is_not_registered_is_skipped():
    skills = bootstrap_agent_skills(
        [{"key": "ghost.x", "knowledge_base": "ghost", "operation": "x", "description": "d"}]
    )

    assert skills == []


# -- persistence ------------------------------------------------------------------

def test_sync_validates_before_writing():
    graph = FakeGraph()

    with pytest.raises(AgentSkillError):
        AgentSkillRepository(graph).sync_skill(
            AgentSkill(key="k", name="noesis-project-management", description="")
        )
    assert graph.queries == []


def test_sync_merges_on_key_and_versions_real_changes():
    graph = FakeGraph()
    AgentSkillRepository(graph).sync_skill(
        AgentSkill(key="noesis-project-management", name="noesis-project-management", description="d", body="b")
    )

    query, params = graph.queries[0]
    assert "MERGE (s:AgentSkill {key: $key})" in query
    # A reworded body is a new version, an identical re-sync is not.
    assert "s.version = CASE WHEN" in query
    assert params["key"] == "noesis-project-management"
    assert params["body"] == "b"


def test_sync_links_each_skill_to_the_tools_it_documents():
    """`AgentSkill -[:DOCUMENTS]-> Tool` makes "the skill contains its tools"
    graph-true at sync time, rather than a convention of the generated body."""
    from knowledge_bases.orchestration.agent_skills_catalog import sync_agent_skills

    rows = [
        {"key": t["key"], "name": t["operation"], "version": "1",
         "knowledge_base": t["knowledge_base"], "active": True,
         "description": t["description"], "input_schema": None,
         "handler": "mod:fn"}
        for t in _tools()
    ]
    graph = FakeGraph(rows)

    skills = sync_agent_skills(graph)

    assert [s.key for s in skills] == ["noesis-project-management"]
    expected = {
        ("AgentSkill", "noesis-project-management", "DOCUMENTS", "Tool", t["key"])
        for t in _tools()
    }
    actual = {
        (start, start_key["key"], rel, end, end_key["key"])
        for start, start_key, rel, end, end_key in graph.rels
    }
    assert actual == expected


def test_loading_filters_retired_skills_by_default():
    graph = FakeGraph()
    AgentSkillRepository(graph).load_skills()
    assert "WHERE s.active = true" in graph.queries[0][0]

    graph = FakeGraph()
    AgentSkillRepository(graph).load_skills(include_inactive=True)
    assert "WHERE s.active = true" not in graph.queries[0][0]


def test_a_row_with_a_null_body_still_loads():
    """Neo4j returns null for a property that was never written."""
    graph = FakeGraph([{
        "key": "noesis-project-management", "name": "noesis-project-management", "description": "d",
        "body": None, "knowledge_base": "project_management", "version": 2, "active": True,
    }])

    skill = AgentSkillRepository(graph).get_skill("noesis-project-management")

    assert skill is not None and skill.body == ""


def test_getting_an_unknown_skill_returns_none():
    assert AgentSkillRepository(FakeGraph([])).get_skill("nope") is None


def test_retiring_an_unknown_skill_is_refused():
    with pytest.raises(ValueError, match="Unknown agent skill"):
        AgentSkillRepository(FakeGraph([])).set_active("nope", False)


def test_every_knowledge_base_with_tools_gets_an_agent_skill():
    """A SKILL.md stub is written for every knowledge base that owns tools, and
    the stub tells the harness to fetch its body from an `AgentSkill` node. If
    the node is missing the harness's first fetch fails, so the two sets have
    to match.

    This regressed when `self_catalog` gained its first tools: the export wrote
    a tenth stub while the graph still held nine skills, because only an MCP
    tool synced them and `scripts/sync_tools_catalog.py` did not.
    """
    from knowledge_bases.orchestration.agent_skills_catalog import (
        bootstrap_agent_skills,
    )
    from knowledge_bases.tools.catalog import bootstrap_catalog

    tools = [
        {
            "key": tool.key,
            "knowledge_base": tool.knowledge_base,
            # Same shape `_tool_rows` builds from the graph.
            "operation": tool.key.split(".", 1)[1] if "." in tool.key else tool.key,
            "description": tool.description,
        }
        for tool in bootstrap_catalog()
    ]
    knowledge_bases_with_tools = {tool["knowledge_base"] for tool in tools}
    skills = bootstrap_agent_skills(tools)

    assert {skill.knowledge_base for skill in skills} == knowledge_bases_with_tools
