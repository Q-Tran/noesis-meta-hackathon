from knowledge_bases.orchestration.models import ExpertProfile
from knowledge_bases.orchestration.experts_catalog import bootstrap_profiles
from knowledge_bases.orchestration.routing import plan_route, score_experts


def test_every_graph_expert_is_routable():
    """Each routable expert owns one knowledge base. A question about Noesis
    itself has an owner -- `development`, which indexes Noesis's own source as
    one repository among others -- rather than falling through to an arbitrary
    domain expert."""
    profiles = bootstrap_profiles()

    assert {profile.key for profile in profiles} == {
        "semantic_layer",
        "business_dictionary",
        "epistemology",
        "strategy",
        "architecture",
        "project_management",
        "methodology",
        "development",
    }


def test_metric_question_routes_dictionary_before_semantic_layer():
    plan = plan_route(
        "Are dedicated accounts on track against Q3 target metrics?",
        bootstrap_profiles(),
    )

    assert [delegation.knowledge_base for delegation in plan.delegations[:2]] == [
        "business_dictionary",
        "semantic_layer",
    ]
    assert plan.delegations[0].operation == "resolve_business_language"
    assert plan.delegations[1].operation == "resolve_metric_question"


def test_plain_volume_question_still_routes_semantic_layer():
    # No jargon like "metric"/"target"/"attainment" -- just the kind of
    # plain business-volume phrasing a real user types. This used to score
    # 0 against every expert's capability keywords, so `plan_route` fell
    # back to a single arbitrary expert (business_dictionary) and never
    # delegated to semantic_layer at all, even though the question is
    # squarely a metric question.
    plan = plan_route(
        "Go ahead and give me net adds dedicated across attb for every "
        "month for the last 12 months",
        bootstrap_profiles(),
    )

    experts = [delegation.knowledge_base for delegation in plan.delegations]
    assert "business_dictionary" in experts
    assert "semantic_layer" in experts


def test_architecture_question_routes_to_methodology():
    plan = plan_route(
        "Validate the architecture components against the specification baseline",
        bootstrap_profiles(),
    )

    assert "methodology" in [delegation.knowledge_base for delegation in plan.delegations]
    methodology = next(d for d in plan.delegations if d.knowledge_base == "methodology")
    assert methodology.operation == "consult_workflow"
    assert methodology.arguments == {
        "question": "Validate the architecture components against the specification baseline"
    }


def test_explicit_multi_domain_question_routes_every_matching_expert():
    plan = plan_route(
        "What evidence challenges our architecture decision and which delivery tasks are blocked?",
        bootstrap_profiles(),
    )

    experts = [delegation.knowledge_base for delegation in plan.delegations]
    assert "epistemology" in experts
    assert "strategy" in experts
    assert "architecture" in experts
    assert "project_management" in experts


def test_unknown_question_requests_clarification():
    profiles = [
        ExpertProfile(
            key="semantic_layer",
            name="Semantic Layer",
            description="metrics",
            capabilities=["metric"],
        )
    ]

    plan = plan_route("hello there", profiles)

    assert plan.clarification_needed


# -- the zero-evidence fallback ----------------------------------------------------

def _zero_score_question():
    """A question no expert's capability vocabulary matches at all.

    This used to be "Rename this module and update the imports" -- a question
    about Noesis itself, which no expert owned. The `development` expert now
    owns exactly that, so it is no longer zero-evidence, and reusing it here
    would test the opposite of what it says. The behaviour under test (refuse
    to route rather than hand a coin flip to an arbitrary expert) still
    matters, so the example is replaced with one genuinely outside every
    expert's vocabulary. (Note "book a table for four" does *not* work: it scores
    on "table" from the semantic layer -- keyword scoring matches the word, not
    the sense.)"""
    return "Arrange flowers for the lobby"


def test_a_question_nothing_matches_scores_zero_for_every_expert():
    """Establishes the premise of the tests below: there is genuinely no
    evidence here, rather than weak evidence for one expert."""
    candidates = score_experts(_zero_score_question(), bootstrap_profiles())

    assert all(candidate.score == 0.0 for candidate in candidates)


def test_zero_evidence_routing_says_so_instead_of_inventing_a_primary_expert():
    """With no evidence at all, `candidates[0]` is whatever the sort happened
    to put first -- an arbitrary tie-break. Reporting it as `primary_expert`
    made a coin flip look like a decision, and the orchestrator spawned a
    subagent for a domain the question has nothing to do with."""
    plan = plan_route(_zero_score_question(), bootstrap_profiles())

    assert plan.clarification_needed
    assert plan.primary_expert is None
    assert plan.delegations == []


def test_zero_evidence_routing_still_ranks_candidates_for_a_human_to_choose():
    """Refusing to guess is not refusing to help: the caller still gets the
    roster so it can ask, or pick, with the scores visible."""
    plan = plan_route(_zero_score_question(), bootstrap_profiles())

    assert len(plan.candidates) == len(bootstrap_profiles())
    assert plan.confidence == 0.0


def test_weak_but_real_evidence_still_routes():
    """The fallback is only suppressed when there is *no* evidence. A weak
    single match is still a signal and must still produce a delegation."""
    profiles = bootstrap_profiles()
    plan = plan_route("What does ARR mean?", profiles)

    assert plan.primary_expert is not None
    assert plan.delegations


def test_route_bundles_expert_prompt_and_suggested_tool_schemas():
    """Routing directly supplies the expert's system prompt and search-derived
    suggested tools with schemas, so an agent harness like OpenCode can spawn its
    subagent in one shot without redundant round trips."""
    from knowledge_bases.tools.models import Tool
    dict_tool = Tool(
        key="business_dictionary.lookup_business_term",
        knowledge_base="business_dictionary",
        name="Lookup Business Term",
        version="1",
        description="Lookup official business term definitions and terms",
        input_schema={"type": "object", "properties": {"term": {"type": "string"}}},
        handler="fake:handler",
    )
    metric_tool = Tool(
        key="semantic_layer.get_metric",
        knowledge_base="semantic_layer",
        name="Get Metric",
        version="1",
        description="Fetch a metric calculation and lineage",
        input_schema={"type": "object", "properties": {"metric_name": {"type": "string"}}},
        handler="fake:handler",
    )
    skills_by_key = {
        "business_dictionary.lookup_business_term": dict_tool,
        "semantic_layer.get_metric": metric_tool,
    }
    profiles = bootstrap_profiles()
    plan = plan_route("calculate churn rate metric and lookup term", profiles, skills_by_key=skills_by_key)

    assert plan.primary_expert_prompt is not None
    assert plan.suggested_tools
    assert plan.suggested_tools[0]["input_schema"]

    # Every delegation also has the prompt and suggested tools
    semantic_delegation = next(d for d in plan.delegations if d.knowledge_base == "semantic_layer")
    assert "Semantic Layer" in semantic_delegation.system_prompt
    assert semantic_delegation.suggested_tools
    assert semantic_delegation.suggested_tools[0]["tool_key"] == "semantic_layer.get_metric"
    assert semantic_delegation.suggested_tools[0]["input_schema"] == {
        "type": "object", "properties": {"metric_name": {"type": "string"}},
    }

