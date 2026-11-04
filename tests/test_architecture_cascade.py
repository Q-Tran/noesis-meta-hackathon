"""The forward hop: structure that a decision requires, not structure that exists.

Noesis can already read a codebase and report its components. This hop does something
that looks similar and is not: it records the structure a decision COMMITS TO, before
any code exists. These tests pin the difference, because collapsing the two is the
mistake that would make the architecture ontology a duplicate of the code index.
"""
from __future__ import annotations

import pytest

from knowledge_bases.cascade import StrategyGuidance, plan_cascade, run_cascade
from knowledge_bases.registry import (
    cascade_steps,
    entity_types,
    relationship_types,
    versioned_kinds,
)

STEP_KEY = "architecture.structure_from_decisions"


def test_the_step_is_installed_and_runs_forward_from_a_decision():
    step = cascade_steps()[STEP_KEY]

    assert step.trigger_labels == ("Decision",)
    assert step.produces_labels == ("Architecture", "Component", "Interface")
    assert step.relationship == "AFFECTS"
    assert step.steerable is True
    assert step.requires_confirmation is True


def test_the_step_supplies_exactly_the_edge_a_decision_is_incomplete_without():
    """This is what makes the hop structural rather than decorative.

    `decisions.strategy_from_goals` writes a Decision that is reported `incomplete`
    because its kind requires an outbound AFFECTS to an Architecture. This step's
    declared relationship IS that edge. The two facts are asserted together so that
    changing either one without the other fails: a Decision left permanently incomplete
    is a chain that reports work it never does.
    """
    decision_kind = versioned_kinds()["Decision"]
    required = [
        (requirement.direction, requirement.relationship, requirement.endpoint_kinds)
        for requirement in decision_kind.requirements
    ]

    assert ("outbound", "AFFECTS", ("Architecture",)) in required
    assert cascade_steps()[STEP_KEY].relationship == "AFFECTS"
    assert "Architecture" in cascade_steps()[STEP_KEY].produces_labels


def test_the_hop_does_not_claim_accountability_the_way_the_strategy_hop_does():
    """The asymmetry, asserted so it reads as a decision rather than an omission.

    A Decision is a commitment and the graph must record who made it. An Architecture
    is a description of structure, judged by whether it is right; `AFFECTS` ties it to
    the Decision, which IS attributed, so ownership flows through that edge.

    Requiring a second name here would be worse than redundant: it would teach
    reviewers that every confirmation needs a name typed into it, and a name given as a
    reflex looks like accountability in the graph without being it.
    """
    assert cascade_steps()[STEP_KEY].human_attribution is None
    assert cascade_steps()["strategy.strategy_from_goals"].human_attribution == "MADE_BY"


def test_the_step_cannot_write_the_edge_that_joins_intent_to_real_code():
    """Where reasoning stops and measurement begins.

    `CATALOGS` (Architecture -> CodeRepository) asserts that an architecture describes
    a particular checkout. That is a claim about a directory that exists, so it is
    written by the indexer that actually found it -- never by a reasoner, which would
    be guessing about the filesystem.

    Enforced by the ontology rather than by this step's good behaviour: CATALOGS is
    `extractable=False`, so `validate_triplet` refuses it from any cascade step.
    """
    catalogs = relationship_types()["CATALOGS"]

    assert catalogs.extractable is False
    assert catalogs.subject == ("Architecture",)
    assert catalogs.object == ("CodeRepository",)


def test_no_step_produces_a_code_entity():
    """The rule the whole `development` hop turns on.

    Every Code* label is machine-derived: its identity is built from a real path and a
    real qualified name, and a full sync re-emits everything it finds. A reasoned
    CodeFunction would be a checkable assertion about a file, usually wrong, and it
    would be deprecated by the next sync anyway.

    So "development ontology is built automatically" is satisfied by INDEXING, not by
    reasoning -- a different mechanism for a different kind of fact.
    """
    code_labels = {
        label for label, entity in entity_types().items()
        if label.startswith("Code")
    }
    assert code_labels, "no code labels found; this test has lost its subject"
    assert all(not entity_types()[label].extractable for label in code_labels)

    for step in cascade_steps().values():
        assert not code_labels.intersection(step.produces_labels), (
            f"{step.key} reasons about code entities, which are measured from source"
        )


def test_a_decision_plans_both_the_architecture_and_the_work_hops():
    """Both follow from a Decision and neither feeds the other, so they belong in the
    same wave -- structure and delivery are parallel consequences of one choice, not
    sequential ones."""
    run = plan_cascade({"Decision"})

    assert len(run.waves) == 1
    assert set(run.waves[0]) == {
        STEP_KEY, "project_management.work_from_decisions",
    }


def test_a_goal_now_plans_the_whole_chain_through_to_structure():
    """End to end: the user's "start anywhere" requirement, from the new entry point."""
    run = plan_cascade({"Goal"})

    planned = [key for wave in run.waves for key in wave]
    assert planned.index("strategy.strategy_from_goals") < planned.index(STEP_KEY)
    assert STEP_KEY in planned


class FakeGraph:
    def __init__(self):
        self.rows = [{"id": "d1", "labels": ["Decision"], "props": {}}]
        self.nodes, self.rels = [], []

    def run(self, *args, **kwargs):
        return self.rows

    def upsert_node(self, label, key, props=None):
        self.nodes.append((label, key))
        return [{}]

    def upsert_rel(self, sl, sk, rel, ol, ok, props=None):
        self.rels.append((sl, rel, ol))
        return [{}]

    def attach_source(self, *args, **kwargs):
        return [{}]


def _structure(step, triggers):
    from knowledge_bases.triplets import Triplet

    return [
        Triplet(
            subject_label="Decision", subject_identity="d1",
            relationship="AFFECTS",
            object_label="Architecture", object_identity="arch-1",
            object_props={"name": "Billing", "description": "One provider"},
        ),
        Triplet(
            subject_label="Component", subject_identity="c-gateway",
            relationship="EXPOSES",
            object_label="Interface", object_identity="i-charge",
        ),
    ]


def test_proposed_structure_is_written_once_confirmed():
    graph = FakeGraph()

    run = run_cascade(graph, {"Decision"}, _structure,
                      steps={STEP_KEY: cascade_steps()[STEP_KEY]},
                      confirm=lambda step, proposed: True)

    assert run.executed == (STEP_KEY,)
    assert ("Decision", "AFFECTS", "Architecture") in graph.rels
    assert ("Component", "EXPOSES", "Interface") in graph.rels


def test_an_architecture_with_no_capability_is_reported_incomplete():
    """Honest rather than tidy.

    `Architecture` requires an outbound IMPLEMENTS to a Capability, and a reasoner
    should NOT invent one -- Capability has its own requirement (SATISFIES a
    Requirement), so fabricating one to satisfy this check would cascade a second
    invented entity out of the first.

    Note that `Capability` belongs to `project_management`, so on a product that omits
    it this requirement can never be satisfied at all. That is recorded as a known
    conformance finding rather than papered over: an Architecture that is permanently
    `incomplete` on a personal product is the truth, and inventing a local Capability
    to silence it would be a lie in the graph.
    """
    graph = FakeGraph()

    run = run_cascade(graph, {"Decision"}, _structure,
                      steps={STEP_KEY: cascade_steps()[STEP_KEY]},
                      confirm=lambda step, proposed: True)

    incomplete = run.produced[0]["result"]["incomplete"]
    assert any(
        item["label"] == "Architecture" and "IMPLEMENTS" in item["requirement"]
        for item in incomplete
    ), f"expected an IMPLEMENTS/Capability gap, got {incomplete}"


def test_an_architect_can_steer_the_proposal():
    """Architecture proposals are almost never wholesale right, so accept-or-discard is
    the wrong only-option: discarding loses a design that was mostly correct."""
    graph = FakeGraph()
    asked = []

    def recording(step, triggers):
        asked.append(step.question)
        return []

    run_cascade(graph, {"Decision"}, recording,
                steps={STEP_KEY: cascade_steps()[STEP_KEY]},
                guidance={STEP_KEY: StrategyGuidance(
                    excluded=("a new message bus",),
                    emphasis=("keep the existing deploy boundary",),
                )})

    assert "a new message bus" in asked[0]
    assert "keep the existing deploy boundary" in asked[0]
