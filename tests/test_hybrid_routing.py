"""Retrieval-based routing (`knowledge_bases.orchestration.routing`) and the
fused skill search (`knowledge_bases.tools.search.rank_skills`): keyword
and semantic evidence combined, schema-driven starting delegations, and the
optional external selection hook."""
from __future__ import annotations

from knowledge_bases.orchestration.experts_catalog import bootstrap_profiles
from knowledge_bases.orchestration.models import ExpertProfile
from knowledge_bases.orchestration.routing import (
    SELECT_THRESHOLD,
    plan_route,
    score_experts,
    starting_delegation,
    tokens,
)
from knowledge_bases.tools.models import Tool
from knowledge_bases.tools.search import rank_skills


def _profiles():
    return bootstrap_profiles()


def _skill(key, props=None, required=None, description=""):
    return Tool(
        key=key, name=key.split(".")[1].replace("_", " ").title(), version="1",
        knowledge_base=key.split(".")[0], description=description,
        input_schema={"type": "object", "properties": props or {}, "required": required or []},
        handler="m:f",
    )


# -- scoring --------------------------------------------------------------------

def test_tokens_fold_plurals_and_drop_function_words():
    assert "target" in tokens("Q3 targets") and "targets" in tokens("Q3 targets")
    assert not tokens("who is this for") & {"who", "this", "for", "is"}


def test_semantic_evidence_surfaces_an_expert_with_no_keyword_overlap():
    question = "where do these numbers come from"
    keyword_only = {c.knowledge_base: c.score for c in score_experts(question, _profiles())}
    assert keyword_only["semantic_layer"] == 0.0

    fused = score_experts(
        question, _profiles(),
        skill_hits={"semantic_layer.trace_metric_lineage": 0.55},
    )
    top = fused[0]
    assert top.knowledge_base == "semantic_layer"
    assert top.score >= SELECT_THRESHOLD
    assert top.suggested_tools == ["semantic_layer.trace_metric_lineage"]
    assert any(r.startswith("skill:semantic_layer.trace_metric_lineage") for r in top.reasons)


def test_generic_control_plane_skills_carry_no_routing_evidence():
    scores = {
        c.knowledge_base: c.score for c in score_experts(
            "anything at all", _profiles(),
            skill_hits={"orchestration.hybrid_search": 0.9, "tools.get_span": 0.9},
        )
    }
    assert all(score == 0.0 for score in scores.values())


def test_agreeing_keyword_and_semantic_evidence_scores_higher_than_either_alone():
    question = "which decisions need review"
    keyword = score_experts(question, _profiles())[0]
    both = score_experts(
        question, _profiles(),
        expert_hits={"strategy": {"similarity": 0.6, "matched": "x"}},
    )[0]
    assert both.knowledge_base == keyword.knowledge_base == "strategy"
    assert both.score > keyword.score


# -- selection / plan -------------------------------------------------------------

def test_plan_route_uses_semantic_hits_to_select_and_reports_high_confidence():
    plan = plan_route(
        "who is on the hook for the blocked epic", _profiles(),
        expert_hits={"project_management": {"similarity": 0.5, "matched": "delivery:0"}},
        skill_hits={"project_management.list_project_blockers": 0.45},
    )
    assert [d.knowledge_base for d in plan.delegations] == ["project_management"]
    assert plan.delegations[0].operation == "list_project_blockers"
    assert plan.confidence > 0.5 and not plan.clarification_needed


def test_external_select_overrides_threshold_selection():
    plan = plan_route(
        "What evidence challenges our architecture decision?", _profiles(),
        select=lambda candidates: ["strategy"],
    )
    assert [d.knowledge_base for d in plan.delegations] == ["strategy"]


def test_external_select_ignores_unknown_experts_and_falls_back_to_top_candidate():
    plan = plan_route("Which decisions need review?", _profiles(), select=lambda c: ["nope"])
    assert plan.delegations[0].knowledge_base == plan.candidates[0].knowledge_base


def test_metric_rule_still_adds_dictionary_and_semantic_layer_with_external_select():
    plan = plan_route(
        "trailing 12 months net adds by month", _profiles(), select=lambda c: ["project_management"],
    )
    experts = [d.knowledge_base for d in plan.delegations]
    assert experts[:2] == ["business_dictionary", "semantic_layer"] and "project_management" in experts


# -- starting delegations ------------------------------------------------------------

def test_starting_delegation_fills_arguments_from_the_entry_skill_schema():
    profile = ExpertProfile(
        key="business_dictionary", name="BD", description="",
        entry_tools=["business_dictionary.resolve_business_language"],
    )
    skills = {
        "business_dictionary.resolve_business_language": _skill(
            "business_dictionary.resolve_business_language",
            props={"question": {"type": "string"}, "limit": {"type": "integer"}},
            required=["question"],
        ),
    }
    delegation = starting_delegation(profile, "what is ARR", 0, skills)
    assert delegation.operation == "resolve_business_language"
    assert delegation.arguments == {"question": "what is ARR"}


def test_starting_delegation_skips_entry_skills_needing_an_identifier_and_falls_back_to_hybrid_search():
    profile = ExpertProfile(
        key="epistemology", name="Epistemology", description="",
        entry_tools=["epistemology.get_claim_provenance"],
    )
    skills = {
        "epistemology.get_claim_provenance": _skill(
            "epistemology.get_claim_provenance",
            props={"claim_id": {"type": "string"}}, required=["claim_id"],
        ),
    }
    delegation = starting_delegation(profile, "why do we believe churn rose", 0, skills)
    assert delegation.operation == "hybrid_search"
    assert delegation.arguments["query"] == "why do we believe churn rose"
    assert set(delegation.arguments["entity_types"]) == {
        "epistemology.Claim", "epistemology.DocumentNode",
    }


def test_starting_delegation_without_schemas_uses_name_heuristics():
    delivery = ExpertProfile(
        key="project_management", name="D", description="", entry_tools=["project_management.list_project_blockers"],
    )
    assert starting_delegation(delivery, "q", 0).operation == "list_project_blockers"
    evidence = ExpertProfile(
        key="epistemology", name="E", description="", entry_tools=["epistemology.get_claim_provenance"],
    )
    assert starting_delegation(evidence, "q", 0).operation == "hybrid_search"


# -- skill search fusion -------------------------------------------------------------

def test_rank_skills_fuses_keyword_and_semantic_rankings():
    skills = [
        _skill("semantic_layer.trace_metric_lineage", description="Trace a metric to its source tables."),
        _skill("project_management.list_project_blockers", description="List blocked work."),
        _skill("methodology.consult_workflow", description="Continue the product development workflow."),
    ]
    ranked = rank_skills(
        "where do the numbers behind this metric come from", skills,
        semantic={"semantic_layer.trace_metric_lineage": 0.6, "methodology.consult_workflow": 0.2},
    )
    assert ranked[0]["skill_key"] == "semantic_layer.trace_metric_lineage"
    assert ranked[0]["similarity"] == 0.6
    assert any(r.startswith("semantic:") for r in ranked[0]["reasons"])
    assert "text" not in ranked[0]
    assert {r["skill_key"] for r in ranked} == {
        "semantic_layer.trace_metric_lineage", "methodology.consult_workflow",
    }


def test_rank_skills_applies_the_reranker_when_one_is_given():
    class ReverseReranker:
        def rerank(self, query, candidates, text_key="text"):
            return list(reversed(candidates))

    skills = [
        _skill("a.first", description="project blockers"),
        _skill("b.second", description="project"),
    ]
    ranked = rank_skills("project blockers", skills, llm=ReverseReranker())
    assert [r["skill_key"] for r in ranked] == ["b.second", "a.first"]
