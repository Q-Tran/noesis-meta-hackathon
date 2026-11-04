"""Every route offers a way to record what the question asserts.

The audit found 0 of 9 ingestion-phrased questions producing a write, despite 17
bounded write commands existing -- "Create a project for the mobile app rewrite"
routed to `list_project_blockers`. The fix is deliberately NOT a write-intent
detector. Intent detection was considered and rejected: whether information can be
recorded is decided by whether it fits the ontology, not by whether the phrasing
sounded like a request to store something. Assertions and questions arrive
together ("churn is 5%, why is it up?"), so a router that waits to be asked never
writes.

So the offer is unconditional, and these tests pin that it cannot quietly become
conditional again.
"""
from __future__ import annotations

import pytest

from knowledge_bases.orchestration import routing
from knowledge_bases.orchestration.experts_catalog import bootstrap_experts
from knowledge_bases.orchestration.models import ExpertProfile


@pytest.fixture
def profiles():
    return [
        ExpertProfile(
            key=expert.key, name=expert.name, description=expert.description,
            capabilities=tuple(expert.capabilities),
            entry_tools=tuple(expert.entry_tools),
            priority=expert.priority, system_prompt=expert.system_prompt,
        )
        for expert in bootstrap_experts()
    ]


QUESTIONS = [
    # Plainly an instruction to record something.
    "Create a project for the mobile app rewrite",
    # An assertion and a question in one breath -- the case intent detection
    # handles worst, because the sentence ends in a question mark.
    "Churn is 5% this month, why is it up?",
    # A correction to existing knowledge.
    "Our definition of active user changed to a 28-day window",
    # A pure read. Still offered: a read often carries a correction in passing.
    "What is the revenue metric?",
]


@pytest.mark.parametrize("question", QUESTIONS)
def test_every_plan_offers_ingestion(profiles, question):
    plan = routing.plan_route(question, profiles)

    assert plan.ingestion is not None, (
        f"{question!r} produced a plan with no way to record what it asserts"
    )
    assert plan.ingestion.describe_tool == "epistemology.describe_ontology"
    assert plan.ingestion.ingest_tool == "epistemology.ingest_information"


def test_a_question_matching_no_expert_still_offers_ingestion(profiles):
    """The most important case. No expert matching does not mean the question
    carries nothing worth recording -- a question about a domain no expert owns is
    precisely where the graph is missing knowledge."""
    plan = routing.plan_route("asdfgh qwerty zxcvb", profiles)

    assert plan.primary_expert is None
    assert plan.delegations == []
    assert plan.ingestion is not None


def test_the_offer_names_the_two_calls_in_order(profiles):
    """Ingesting without fetching the vocabulary first means guessing type names,
    and guessed types are rejected -- so the order is part of the offer."""
    plan = routing.plan_route("Churn is 5%", profiles)

    guidance = plan.ingestion.guidance
    assert guidance.index("describe_tool") < guidance.index("ingest_tool")
    assert "append-only" in guidance


def test_the_offer_does_not_displace_the_routed_delegations(profiles):
    """Ingestion is an addition, not a replacement: a question that reads and
    asserts must still reach its expert."""
    plan = routing.plan_route("Create a project for the mobile app rewrite", profiles)

    assert plan.delegations, "routing lost its delegations"
    assert plan.primary_expert is not None
    assert plan.ingestion is not None


def test_ingestion_is_not_gated_on_phrasing(profiles):
    """The regression guard for the rejected design. If someone reintroduces
    token-based write detection, a plainly declarative sentence with no imperative
    verb is what will stop being offered."""
    declarative = routing.plan_route(
        "The EMEA region closed 40 deals last quarter", profiles
    )
    interrogative = routing.plan_route("How many deals closed in EMEA?", profiles)

    assert declarative.ingestion is not None
    assert interrogative.ingestion is not None
