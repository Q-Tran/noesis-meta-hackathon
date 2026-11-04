"""Graph-defined experts: the seed catalog, versioned graph sync, and the MCP
surface that serves them."""
from __future__ import annotations

import json

import pytest

from knowledge_bases.orchestration import mcp_tools
from knowledge_bases.orchestration.experts_catalog import (
    EXPERT_LOOP_CONTRACT,
    ORCHESTRATOR_KEY,
    bootstrap_experts,
    bootstrap_profiles,
)
from knowledge_bases.orchestration.models import ExpertDefinition
from knowledge_bases.orchestration.repository import RoutingRepository, _skill_allowed
from knowledge_bases.tools.catalog import bootstrap_catalog


# -- catalog ------------------------------------------------------------------

def test_every_routable_expert_has_a_prompt_entry_skills_and_the_loop_contract():
    for expert in bootstrap_profiles():
        assert expert.routable
        assert expert.system_prompt.startswith(f"You are the Noesis {expert.name} expert.")
        assert EXPERT_LOOP_CONTRACT in expert.system_prompt
        assert expert.entry_tools, expert.key
        assert expert.capabilities, expert.key


def test_orchestrator_is_defined_but_not_routable():
    experts = {e.key: e for e in bootstrap_experts()}
    orchestrator = experts[ORCHESTRATOR_KEY]
    assert not orchestrator.routable
    assert ORCHESTRATOR_KEY not in {e.key for e in bootstrap_profiles()}
    assert "route_noesis_question" in orchestrator.system_prompt
    assert "automatically logged and updated" in orchestrator.system_prompt


def test_entry_and_allowed_skills_reference_real_catalog_skills():
    catalog_keys = {skill.key for skill in bootstrap_catalog()}
    for expert in bootstrap_experts():
        for key in expert.entry_tools:
            assert key in catalog_keys, f"{expert.key} entry skill {key} is not in the catalog"
        for pattern in expert.allowed_tools:
            if pattern.endswith(".*"):
                assert any(k.startswith(pattern[:-1]) for k in catalog_keys), pattern
            else:
                assert pattern in catalog_keys, pattern
        assert any(_skill_allowed(k, expert.allowed_tools) for k in expert.entry_tools)


def test_business_dictionary_capabilities_do_not_carry_methodology_vocabulary():
    """Regression for the self-improvement commit that appended methodology/startup
    wording to business_dictionary and made methodology questions co-route there."""
    bd = next(e for e in bootstrap_profiles() if e.key == "business_dictionary")
    joined = " ".join(bd.capabilities).lower()
    assert "methodology" not in joined and "startup" not in joined


# -- repository sync -----------------------------------------------------------

class FakeRecord(dict):
    pass


class FakeTx:
    """Enough of a neo4j transaction for `_write_expert`: remembers the
    current expert state so a second identical sync is detected as a no-op."""

    def __init__(self, existing=None):
        self.existing = existing
        self.writes = []

    def run(self, query, **params):
        tx = self

        class Result:
            def single(self_inner):
                if "RETURN properties(e) AS props" in query and tx.existing is not None:
                    return {"props": tx.existing["props"]}
                return None

            def consume(self_inner):
                tx.writes.append((query, params))

        return Result()


class FakeGraph:
    def __init__(self, tx):
        self.tx = tx

    def execute_write(self, work):
        return work(self.tx)


def _expert(**overrides):
    base = dict(
        key="project_management", name="Execution", description="d", capabilities=["a", "b"],
        priority=60, system_prompt="p", entry_tools=["project_management.list_project_blockers"],
        allowed_tools=["project_management.*"],
    )
    base.update(overrides)
    return ExpertDefinition(**base)


def test_first_sync_writes_version_1_with_revision_and_can_use_edges():
    tx = FakeTx()
    RoutingRepository(FakeGraph(tx)).sync_experts([_expert()])

    queries = [q for q, _ in tx.writes]
    assert any("MERGE (e:KnowledgeBaseExpert {key: $key})" in q for q in queries)
    revision = next(q for q in queries if "CREATE (r:ExpertRevision" in q)
    # Every edge this catalog writes is created with a `created_at`, so the
    # registry is datable: "when did this expert get this tool" is a question
    # the graph can answer. Pinned here because these statements are raw Cypher.
    assert "CREATE (e)-[:CURRENT {created_at: datetime()}]->(r)" in revision
    assert any("MERGE (e)-[cu:CAN_USE]->(s)" in q for q in queries)
    assert any("ON CREATE SET cu.created_at = datetime()" in q for q in queries)
    params = next(p for q, p in tx.writes if "MERGE (e:KnowledgeBaseExpert" in q)
    assert params["version"] == 1
    assert params["props"]["system_prompt"] == "p"
    caps = next(p for q, p in tx.writes if "MERGE (e)-[hc:HAS_CAPABILITY]->(c)" in q)
    assert caps["capabilities"] == ["a", "b"]


def test_unchanged_definition_does_not_create_a_new_revision():
    expert = _expert()
    existing = {
        "props": {**{f: getattr(expert, f) for f in (
            "name", "description", "priority", "system_prompt", "entry_tools",
            "allowed_tools", "routable", "capabilities",
        )}, "version": 3},
    }
    tx = FakeTx(existing)
    version = RoutingRepository(FakeGraph(tx)).revise_expert(expert, reason="noop")

    assert version == 3
    assert tx.writes == []


def test_changed_prompt_bumps_the_version_and_supersedes_the_previous_revision():
    expert = _expert()
    existing = {
        "props": {"name": "Execution", "description": "d", "priority": 60,
                  "system_prompt": "old", "entry_tools": expert.entry_tools,
                  "allowed_tools": expert.allowed_tools, "routable": True, "version": 3,
                  "capabilities": ["a", "b"]},
    }
    tx = FakeTx(existing)
    version = RoutingRepository(FakeGraph(tx)).revise_expert(expert, reason="self_improvement")

    assert version == 4
    params = next(p for q, p in tx.writes if "CREATE (r:ExpertRevision" in q)
    assert params["reason"] == "self_improvement"
    supersession = next(q for q, _ in tx.writes if "CREATE (r:ExpertRevision" in q)
    assert "CREATE (r)-[:SUPERSEDES {created_at: datetime()}]->(previous)" in supersession


def test_catalog_sync_preserves_graph_side_overlays_and_projects_effective_capabilities():
    expert = _expert()
    existing = {"props": {"name": "Execution", "description": "d", "priority": 60,
                          "system_prompt": "p", "entry_tools": expert.entry_tools,
                          "allowed_tools": expert.allowed_tools, "routable": True,
                          "version": 2, "capabilities": ["a", "b"],
                          "capability_overlays": ["learned phrase"],
                          "prompt_overlays": ["Also do X."]}}
    tx = FakeTx(existing)
    RoutingRepository(FakeGraph(tx)).sync_experts([expert])

    assert tx.writes == []  # seed unchanged + overlays preserved -> no new revision

    tx = FakeTx(existing)
    RoutingRepository(FakeGraph(tx)).sync_experts([_expert(system_prompt="p2")])
    params = next(p for q, p in tx.writes if "MERGE (e:KnowledgeBaseExpert" in q)
    assert params["props"]["capability_overlays"] == ["learned phrase"]
    assert params["props"]["prompt_overlays"] == ["Also do X."]
    caps = next(p for q, p in tx.writes if "MERGE (e)-[hc:HAS_CAPABILITY]->(c)" in q)
    assert caps["capabilities"] == ["a", "b", "learned phrase"]


def test_effective_prompt_and_capabilities_include_overlays():
    expert = _expert(capability_overlays=["x"], prompt_overlays=["More guidance."])
    assert expert.effective_capabilities == ["a", "b", "x"]
    assert expert.effective_prompt == "p\n\nMore guidance."


def test_skill_allowed_matches_exact_keys_and_knowledge_base_globs():
    assert _skill_allowed("project_management.list_project_blockers", ["project_management.*"])
    assert _skill_allowed(
        "semantic_layer.run_snowflake_query",
        ["semantic_layer.*", "semantic_layer.run_snowflake_query"],
    )
    assert not _skill_allowed("methodology.consult_workflow", ["project_management.*"])


# -- MCP tools -----------------------------------------------------------------

class ExpertGraph:
    """Returns expert rows for the expert queries and skill rows for the
    skill query, so `get_noesis_expert` can resolve `skills` live."""

    def __init__(self, experts, skills):
        self.experts, self.skills = experts, skills

    def run(self, query, **params):
        if "MATCH (s:Tool {active: true})" in query:
            return self.skills
        rows = [e.model_dump() for e in self.experts]
        if "key: $key" in query:
            rows = [r for r in rows if r["key"] == params["key"]]
        return rows


def _skill_row(key, kb):
    return {
        "key": key, "name": key, "version": "1", "knowledge_base": kb, "active": True,
        "description": "", "input_schema": json.dumps({"type": "object", "properties": {"question": {"type": "string"}}}),
        "handler": "m:f",
    }


def test_get_noesis_expert_returns_prompt_and_live_allowed_tools(monkeypatch):
    from knowledge_bases.tools import active_trace

    active_trace.reset_for_testing()
    graph = ExpertGraph(
        [_expert(allowed_tools=["project_management.*", "semantic_layer.run_snowflake_query"])],
        [_skill_row("project_management.list_project_blockers", "project_management"),
         _skill_row("semantic_layer.run_snowflake_query", "semantic_layer"),
         _skill_row("methodology.consult_workflow", "methodology")],
    )
    payload = mcp_tools.get_noesis_expert_tool({"key": "project_management"}, graph)

    assert payload["system_prompt"] == "p"
    assert payload["entry_tools"] == ["project_management.list_project_blockers"]
    assert [s["key"] for s in payload["tools"]] == [
        "project_management.list_project_blockers", "semantic_layer.run_snowflake_query",
    ]
    assert payload["tools"][0]["operation"] == "list_project_blockers"
    assert active_trace.resolve_identity()["expert_key"] == "project_management"


def test_get_noesis_expert_rejects_unknown_keys():
    with pytest.raises(ValueError):
        mcp_tools.get_noesis_expert_tool({"key": "nope"}, ExpertGraph([], []))


def test_list_noesis_experts_includes_unroutable_orchestrator():
    graph = ExpertGraph([_expert(), _expert(key=ORCHESTRATOR_KEY, routable=False)], [])
    payload = mcp_tools.list_noesis_experts_tool({}, graph)

    assert {e["key"] for e in payload["experts"]} == {"project_management", ORCHESTRATOR_KEY}
    assert "skills" not in payload["experts"][0]


def test_the_orchestrator_is_told_not_to_guess_when_nothing_routed():
    """`plan_route` returns `primary_expert=None` with no delegations when no
    expert matched. The prompt has to say what to do with that, or the model
    fills the gap by picking a candidate off the ranked list -- which is the
    arbitrary choice routing just refused to make."""
    orchestrator = next(
        e for e in bootstrap_experts() if e.key == "orchestrator"
    )

    prompt = orchestrator.system_prompt
    assert "`null`" in prompt and "delegations` is empty" in prompt
    assert "do not spawn a subagent on" in prompt
    # Questions about Noesis itself are the common case of this.
    assert "own code or behaviour" in prompt
