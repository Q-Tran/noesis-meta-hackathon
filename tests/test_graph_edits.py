"""Wording-kind self-improvements are graph writes
(`knowledge_bases.tools.graph_edits`): skill description overrides and
expert overlays, guarded against vocabulary pollution and routing-eval
regressions, and revertible item-by-item."""
from __future__ import annotations

import pytest

from knowledge_bases.orchestration.experts_catalog import bootstrap_experts
from knowledge_bases.orchestration.models import ExpertDefinition
from knowledge_bases.tools.graph_edits import (
    GraphEditError,
    apply_graph_edit,
    check_capability_pollution,
    check_routing_eval,
    revert_graph_edit,
)
from knowledge_bases.tools.models import SelfImprovementKind, SelfImprovementSuggestion
from knowledge_bases.tools.source_editor import apply_suggestion


def _experts():
    return bootstrap_experts()


# -- pollution guard -------------------------------------------------------------

def test_pollution_guard_accepts_a_phrase_in_the_targets_own_vocabulary():
    report = check_capability_pollution(
        "glossary definition of a business term or acronym", "business_dictionary", _experts(),
    )
    assert report["own"] > report["closest_other_score"]


def test_pollution_guard_rejects_the_phrase_that_originally_polluted_business_dictionary():
    phrase = (
        "Business metrics, KPI/target definitions, governed metric context, semantic layer, "
        "SQL analytics business context, startup concept evaluation, methodology, beachhead "
        "buyer/use case, assumptions vs evidence"
    )
    with pytest.raises(GraphEditError, match="reads more like"):
        check_capability_pollution(phrase, "business_dictionary", _experts())


def test_pollution_guard_uses_embeddings_when_an_llm_is_given():
    class FakeEmbedder:
        def embed(self, text):
            # "methodology" texts and the phrase point one way, everything else another
            return [1.0, 0.0] if "methodology" in text.lower() or "workflow" in text.lower() else [0.0, 1.0]

    with pytest.raises(GraphEditError):
        check_capability_pollution(
            "continue the guided workflow", "project_management", _experts(), llm=FakeEmbedder(),
        )


# -- eval gate ---------------------------------------------------------------------

def test_eval_gate_rejects_a_change_that_misroutes_golden_questions():
    before = _experts()
    after = [
        e.model_copy(update={"capability_overlays": [
            "decision goal alternative review compromised stale assumption evidence health",
        ]}) if e.key == "business_dictionary" else e
        for e in before
    ]
    with pytest.raises(GraphEditError, match="accuracy would drop"):
        check_routing_eval(before, after)


def test_eval_gate_accepts_a_neutral_change():
    before = _experts()
    after = [
        e.model_copy(update={"capability_overlays": ["blocked epics and their owners"]})
        if e.key == "project_management" else e for e in before
    ]
    report = check_routing_eval(before, after)
    assert report["eval_accuracy_after"] >= report["eval_accuracy_before"]


# -- apply / revert against fakes -------------------------------------------------------

class FakeSkillRepository:
    def __init__(self):
        self.overrides = {}

    def override_description(self, key, description):
        previous = self.overrides.get(key)
        self.overrides[key] = description
        return {"previous_override": previous, "previous": previous or "base"}

    def load_tools(self):
        return []


class FakeRoutingRepository:
    def __init__(self, experts):
        self.experts = {e.key: e for e in experts}
        self.revisions = []

    def load_experts(self, include_unroutable=False):
        return list(self.experts.values())

    def get_expert(self, key):
        return self.experts.get(key)

    def revise_expert(self, expert, reason):
        self.experts[expert.key] = expert.model_copy(update={"version": expert.version + 1})
        self.revisions.append((expert.key, reason))
        return self.experts[expert.key].version

    def revert_expert_overlay(self, key, kind, value, reason):
        expert = self.experts[key]
        field = "capability_overlays" if kind == "capability" else "prompt_overlays"
        return self.revise_expert(
            expert.model_copy(update={field: [o for o in getattr(expert, field) if o != value]}), reason,
        )


@pytest.fixture
def fakes(monkeypatch):
    skills = FakeSkillRepository()
    routing = FakeRoutingRepository(_experts())
    monkeypatch.setattr("knowledge_bases.tools.graph_edits.ToolRepository", lambda graph: skills)
    monkeypatch.setattr("knowledge_bases.tools.graph_edits.RoutingRepository", lambda graph: routing)
    monkeypatch.setattr("knowledge_bases.tools.graph_edits._reindex", lambda *a, **k: None)
    return skills, routing


def test_skill_wording_sets_an_override_and_revert_restores_the_previous_one(fakes):
    skills, _ = fakes
    suggestion = SelfImprovementSuggestion(
        kind=SelfImprovementKind.tool_wording, target="project_management.list_project_blockers",
        diagnosis="d", suggested_change="Clearer description.",
    )
    record = apply_graph_edit(suggestion, graph=object())

    assert skills.overrides["project_management.list_project_blockers"] == "Clearer description."
    assert record["previous_override"] is None

    revert_graph_edit(record, graph=object())
    assert skills.overrides["project_management.list_project_blockers"] is None


def test_routing_wording_appends_an_overlay_as_a_new_revision_and_revert_removes_it(fakes):
    _, routing = fakes
    suggestion = SelfImprovementSuggestion(
        kind=SelfImprovementKind.routing_capability_wording, target="project_management",
        diagnosis="d", suggested_change="blocked epics and their owners",
    )
    record = apply_graph_edit(suggestion, graph=object())

    assert routing.experts["project_management"].capability_overlays == ["blocked epics and their owners"]
    assert routing.experts["project_management"].capabilities == next(e for e in bootstrap_experts() if e.key == "project_management").capabilities  # base untouched
    assert record["version"] == 2 and "pollution_check" in record and "eval_accuracy_after" in record

    revert_graph_edit(record, graph=object())
    assert routing.experts["project_management"].capability_overlays == []
    assert routing.revisions[-1] == ("project_management", "self_improvement_revert:routing_capability_wording")


def test_routing_wording_is_rejected_when_it_pollutes_another_experts_vocabulary(fakes):
    suggestion = SelfImprovementSuggestion(
        kind=SelfImprovementKind.routing_capability_wording, target="business_dictionary",
        diagnosis="d", suggested_change="guided product development workflow phase gate",
    )
    with pytest.raises(GraphEditError, match="reads more like"):
        apply_graph_edit(suggestion, graph=object())


def test_expert_prompt_wording_appends_guidance_and_revert_removes_it(fakes):
    _, routing = fakes
    suggestion = SelfImprovementSuggestion(
        kind=SelfImprovementKind.expert_prompt_wording, target="semantic_layer",
        diagnosis="d", suggested_change="Always state the timeframe you assumed.",
    )
    record = apply_graph_edit(suggestion, graph=object())
    expert = routing.experts["semantic_layer"]
    assert expert.prompt_overlays == ["Always state the timeframe you assumed."]
    assert expert.effective_prompt.endswith("Always state the timeframe you assumed.")

    revert_graph_edit(record, graph=object())
    assert routing.experts["semantic_layer"].prompt_overlays == []


def test_apply_graph_edit_rejects_empty_changes_and_unknown_experts(fakes):
    with pytest.raises(GraphEditError, match="no suggested_change"):
        apply_graph_edit(SelfImprovementSuggestion(
            kind=SelfImprovementKind.tool_wording, target="x", diagnosis="d", suggested_change="  ",
        ), graph=object())
    with pytest.raises(GraphEditError, match="Unknown expert"):
        apply_graph_edit(SelfImprovementSuggestion(
            kind=SelfImprovementKind.expert_prompt_wording, target="nope", diagnosis="d",
            suggested_change="c",
        ), graph=object())


# -- source_editor only handles new_tool now ---------------------------------------------

def test_source_editor_rejects_wording_kinds():
    for kind in (SelfImprovementKind.tool_wording, SelfImprovementKind.routing_capability_wording,
                 SelfImprovementKind.architecture):
        with pytest.raises(ValueError, match="No source-edit strategy"):
            apply_suggestion(SelfImprovementSuggestion(
                kind=kind, target="t", diagnosis="d", suggested_change="c",
            ))
