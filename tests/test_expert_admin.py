"""Runtime expert administration: the prompt prefix, and expert CRUD.

Covers the two things that used to require editing `experts_catalog.py` and
re-exporting: framing an expert's prompt, and changing the roster at all.
"""
from __future__ import annotations

import json

import pytest

from knowledge_bases.orchestration import expert_admin
from knowledge_bases.orchestration.expert_admin import ExpertAdminError
from knowledge_bases.orchestration.experts_catalog import (
    EXPERT_LOOP_CONTRACT,
    ORCHESTRATOR_KEY,
)
from knowledge_bases.orchestration.models import (
    MAX_PROMPT_PREFIX_CHARS,
    MAX_PROMPT_PREFIX_ENTRIES,
    PROMPT_PREFIX_FOOTER,
    PROMPT_PREFIX_HEADER,
    ExpertDefinition,
    PromptPrefixError,
    validate_prompt_prefix_overlays,
)


# -- prompt prefix rendering and budget -----------------------------------------

def _expert(**overrides) -> ExpertDefinition:
    base = dict(key="project_management", name="Execution", description="d",
                capabilities=["blocked epics and owners"], priority=60,
                system_prompt="BASE", entry_tools=["project_management.list_project_blockers"],
                allowed_tools=["project_management.*"])
    base.update(overrides)
    return ExpertDefinition(**base)


def test_prefix_renders_ahead_of_the_base_prompt_inside_a_labelled_block():
    expert = _expert(prompt_prefix_overlays=["Prefer EMEA data."],
                     prompt_overlays=["State the timeframe."])
    prompt = expert.effective_prompt

    assert prompt.index(PROMPT_PREFIX_HEADER) < prompt.index("Prefer EMEA data.")
    assert prompt.index("Prefer EMEA data.") < prompt.index(PROMPT_PREFIX_FOOTER)
    assert prompt.index(PROMPT_PREFIX_FOOTER) < prompt.index("BASE")
    assert prompt.index("BASE") < prompt.index("State the timeframe.")
    assert "advisory only" in prompt


def test_no_prefix_leaves_the_prompt_completely_unchanged():
    assert _expert().effective_prompt == "BASE"
    assert _expert(prompt_prefix_overlays=["  ", ""]).effective_prompt == "BASE"


def test_budget_is_enforced_by_refusing_never_by_truncating():
    with pytest.raises(PromptPrefixError, match="At most"):
        validate_prompt_prefix_overlays(["a"] * (MAX_PROMPT_PREFIX_ENTRIES + 1))
    with pytest.raises(PromptPrefixError, match="characters"):
        validate_prompt_prefix_overlays(["x" * (MAX_PROMPT_PREFIX_CHARS + 1)])

    kept = validate_prompt_prefix_overlays([" one ", "", "two"])
    assert kept == ["one", "two"]


def test_a_prefix_entry_cannot_forge_the_block_delimiters():
    """Otherwise a prefix could close the advisory block and continue as if
    it were part of the base prompt."""
    with pytest.raises(PromptPrefixError, match="delimiters"):
        validate_prompt_prefix_overlays([f"nice try {PROMPT_PREFIX_FOOTER}"])


# -- graph plumbing -------------------------------------------------------------

class FakeGraph:
    """Minimal graph: expert rows, skill rows, and recorded writes."""

    def __init__(self, experts=None, skills=None):
        self.experts = {e.key: e for e in (experts or [])}
        self.skills = skills or []
        self.writes = []

    def run(self, query, **params):
        if "MATCH (s:Tool {active: true})" in query:
            return self.skills
        if "KnowledgeBaseExpert" in query and "SET e.active = false" in query:
            self.writes.append(("retire", params))
            return []
        if "MATCH (e:KnowledgeBaseExpert" in query:
            rows = [e.model_dump() for e in self.experts.values()]
            if "key: $key" in query:
                rows = [r for r in rows if r["key"] == params.get("key")]
            return rows
        return []

    def execute_write(self, work):
        class Tx:
            def __init__(self, outer):
                self.outer = outer

            def run(self_inner, query, **params):
                outer = self_inner.outer

                class Result:
                    def single(self_r):
                        if "RETURN properties(e) AS props" in query:
                            existing = outer.experts.get(params.get("key"))
                            return {"props": {**existing.model_dump(), "version": existing.version}} if existing else None
                        return None

                    def consume(self_r):
                        outer.writes.append((query, params))

                return Result()

        return work(Tx(self))


def _skill_row(key, kb):
    return {"key": key, "name": key, "version": "1", "knowledge_base": kb,
            "active": True, "description": "", "handler": "m:f",
            "input_schema": json.dumps({"type": "object", "properties": {}})}


@pytest.fixture
def no_routing_gates(monkeypatch):
    """Neutralize the routing gates for tests about everything else."""
    monkeypatch.setattr(expert_admin, "_check_routing_impact", lambda *a, **k: {"gated": False})
    monkeypatch.setattr(expert_admin, "_reindex_experts", lambda graph: None)


# -- create ---------------------------------------------------------------------

def test_create_expert_composes_existing_skills_and_attaches_the_loop_contract(no_routing_gates):
    graph = FakeGraph(skills=[_skill_row("project_management.list_project_blockers", "project_management")])
    result = expert_admin.create_expert({
        "key": "programs", "name": "Programs", "description": "Program view",
        "system_prompt": "You own program rollups.",
        "capabilities": ["program rollup status across projects"],
        "entry_tools": ["project_management.list_project_blockers"],
        "allowed_tools": ["project_management.*"],
    }, graph)

    assert result["created"] == "programs"
    props = next(p for q, p in graph.writes if "MERGE (e:KnowledgeBaseExpert" in q)["props"]
    assert props["system_prompt"].startswith("You are the Noesis Programs expert.")
    assert EXPERT_LOOP_CONTRACT in props["system_prompt"]


def test_create_expert_refuses_skills_that_do_not_exist(no_routing_gates):
    """The boundary in practice: an expert may only be composed of
    capabilities that already exist, because adding one is a code change."""
    graph = FakeGraph(skills=[_skill_row("project_management.list_project_blockers", "project_management")])
    with pytest.raises(ExpertAdminError, match="No active skill matches"):
        expert_admin.create_expert({
            "key": "forecasting", "name": "Forecasting", "description": "d",
            "system_prompt": "p", "capabilities": ["forecast"],
            "allowed_tools": ["forecasting.*"],
        }, graph)


def test_create_expert_refuses_entry_tools_outside_allowed_tools(no_routing_gates):
    graph = FakeGraph(skills=[_skill_row("project_management.list_project_blockers", "project_management"),
                              _skill_row("methodology.consult_workflow", "methodology")])
    with pytest.raises(ExpertAdminError, match="not covered by allowed_tools"):
        expert_admin.create_expert({
            "key": "programs", "name": "Programs", "description": "d",
            "system_prompt": "p", "capabilities": ["c"],
            "entry_tools": ["methodology.consult_workflow"], "allowed_tools": ["project_management.*"],
        }, graph)


def test_routable_expert_without_capabilities_is_refused(no_routing_gates):
    graph = FakeGraph(skills=[_skill_row("project_management.list_project_blockers", "project_management")])
    with pytest.raises(ExpertAdminError, match="capability phrase"):
        expert_admin.create_expert({
            "key": "programs", "name": "Programs", "description": "d",
            "system_prompt": "p", "allowed_tools": ["project_management.*"],
        }, graph)


def test_create_expert_refuses_an_empty_skill_set(no_routing_gates):
    with pytest.raises(ExpertAdminError, match="could not do anything"):
        expert_admin.create_expert({
            "key": "empty", "name": "E", "description": "d", "system_prompt": "p",
            "capabilities": ["c"], "allowed_tools": [],
        }, FakeGraph())


def test_create_expert_refuses_a_duplicate_key(no_routing_gates):
    graph = FakeGraph([_expert()], [_skill_row("project_management.list_project_blockers", "project_management")])
    with pytest.raises(ExpertAdminError, match="already exists"):
        expert_admin.create_expert({
            "key": "project_management", "name": "D", "description": "d", "system_prompt": "p",
            "capabilities": ["c"], "allowed_tools": ["project_management.*"],
        }, graph)


# -- the orchestrator is protected ---------------------------------------------

@pytest.mark.parametrize("operation,payload", [
    ("create_expert", {"key": ORCHESTRATOR_KEY, "name": "O", "description": "d",
                       "system_prompt": "p", "capabilities": ["c"],
                       "allowed_tools": ["orchestration.*"]}),
    ("revise_expert", {"key": ORCHESTRATOR_KEY, "system_prompt": "hijacked"}),
    ("retire_expert", {"key": ORCHESTRATOR_KEY}),
])
def test_the_control_loop_cannot_be_reshaped_at_runtime(operation, payload, no_routing_gates):
    graph = FakeGraph([_expert(key=ORCHESTRATOR_KEY, routable=False)], [])
    with pytest.raises(ExpertAdminError, match=ORCHESTRATOR_KEY):
        getattr(expert_admin, operation)(payload, graph)


def test_but_the_orchestrator_can_still_be_framed_with_a_prefix(no_routing_gates):
    """The prefix is the sanctioned way to shape the primary agent."""
    graph = FakeGraph([_expert(key=ORCHESTRATOR_KEY, routable=False)], [])
    result = expert_admin.set_expert_prompt_prefix(
        {"key": ORCHESTRATOR_KEY, "entries": ["This deployment is EMEA-only."]}, graph,
    )

    assert result["entries"] == ["This deployment is EMEA-only."]
    props = next(p for q, p in graph.writes if "MERGE (e:KnowledgeBaseExpert" in q)["props"]
    assert props["prompt_prefix_overlays"] == ["This deployment is EMEA-only."]


def test_setting_a_prefix_replaces_rather_than_appends(no_routing_gates):
    graph = FakeGraph([_expert(prompt_prefix_overlays=["old"])], [])
    expert_admin.set_expert_prompt_prefix({"key": "project_management", "entries": ["new"]}, graph)

    props = next(p for q, p in graph.writes if "MERGE (e:KnowledgeBaseExpert" in q)["props"]
    assert props["prompt_prefix_overlays"] == ["new"]


def test_over_budget_prefix_is_refused_at_the_write(no_routing_gates):
    graph = FakeGraph([_expert()], [])
    with pytest.raises(ExpertAdminError, match="characters"):
        expert_admin.set_expert_prompt_prefix(
            {"key": "project_management", "entries": ["x" * (MAX_PROMPT_PREFIX_CHARS + 1)]}, graph,
        )


# -- revise / retire ------------------------------------------------------------

def test_revise_expert_changes_only_supplied_fields(no_routing_gates):
    graph = FakeGraph([_expert()], [_skill_row("project_management.list_project_blockers", "project_management")])
    result = expert_admin.revise_expert({"key": "project_management", "priority": 5}, graph)

    assert result["changed"] == ["priority"]
    props = next(p for q, p in graph.writes if "MERGE (e:KnowledgeBaseExpert" in q)["props"]
    assert props["priority"] == 5
    assert props["system_prompt"] == "BASE"


def test_revise_expert_with_nothing_to_change_is_refused(no_routing_gates):
    graph = FakeGraph([_expert()], [])
    with pytest.raises(ExpertAdminError, match="Nothing to revise"):
        expert_admin.revise_expert({"key": "project_management"}, graph)


def test_revise_expert_keeps_the_loop_contract_attached(no_routing_gates):
    graph = FakeGraph([_expert()], [_skill_row("project_management.list_project_blockers", "project_management")])
    expert_admin.revise_expert({"key": "project_management", "system_prompt": "New body."}, graph)

    props = next(p for q, p in graph.writes if "MERGE (e:KnowledgeBaseExpert" in q)["props"]
    assert EXPERT_LOOP_CONTRACT in props["system_prompt"]


def test_retire_deactivates_without_deleting(no_routing_gates):
    graph = FakeGraph([_expert()], [])
    result = expert_admin.retire_expert({"key": "project_management", "reason": "merged"}, graph)

    assert result["retired"] == "project_management"
    assert any(kind == "retire" for kind, _ in graph.writes)
    assert not any("DELETE" in str(q).upper() for q, _ in graph.writes if isinstance(q, str))


def test_retiring_an_unknown_expert_is_refused(no_routing_gates):
    with pytest.raises(ExpertAdminError, match="Unknown expert"):
        expert_admin.retire_expert({"key": "ghost"}, FakeGraph())


# -- routing gates are real -----------------------------------------------------

def test_a_new_expert_that_would_lower_routing_accuracy_is_refused(monkeypatch):
    """Adding an expert perturbs routing for every question, so the golden
    eval -- not just the pollution guard -- has to gate it."""
    from knowledge_bases.tools import graph_edits

    monkeypatch.setattr(expert_admin, "_reindex_experts", lambda graph: None)
    # Signature mirrors the real one: the gate now passes the filtered case list
    # explicitly so `before` and `after` are scored on identical cases.
    monkeypatch.setattr(
        graph_edits, "evaluate_routing",
        lambda experts, cases=None: {
            "accuracy": 0.5 if len(experts) > 1 else 0.9, "misses": [],
        },
    )
    graph = FakeGraph(
        [_expert()], [_skill_row("project_management.list_project_blockers", "project_management")],
    )
    with pytest.raises(ExpertAdminError, match="accuracy would drop"):
        expert_admin.create_expert({
            "key": "programs", "name": "Programs", "description": "d",
            "system_prompt": "p", "capabilities": ["program rollups"],
            "allowed_tools": ["project_management.*"],
        }, graph)
