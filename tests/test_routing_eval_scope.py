"""The eval set is scored on what a product installs.

Written when composing `noesis-individual` -- six ontologies, not eleven -- scored
0.53 on the golden set while routing perfectly. Every one of its 25 misses expected
`business_dictionary`, `semantic_layer` or `project_management`: experts it does not
install and therefore cannot rank first. The number measured the profile, not the
router.
"""
from __future__ import annotations

from knowledge_bases.orchestration.experts_catalog import bootstrap_profiles
from knowledge_bases.orchestration.routing_eval import (
    answerable_cases,
    evaluate_routing,
    load_eval_set,
)


def test_the_filter_drops_nothing_on_the_product_that_installs_every_expert():
    """The enterprise product must be scored on the WHOLE set.

    This is the direction that would go unnoticed. A filter slightly too eager
    silently shrinks the set the self-improvement gate runs on, and a smaller set is
    a weaker gate -- nothing fails, the accuracy may even rise, and coverage has been
    lost. Asserting equality here is what makes the filter safe to apply
    unconditionally at both call sites.
    """
    profiles = bootstrap_profiles()
    cases = load_eval_set()

    assert answerable_cases(profiles, cases) == cases


def test_every_case_the_filter_keeps_names_an_expert_this_product_routes_to():
    """The property the filter exists to guarantee, stated independently of any
    particular profile: a kept case must be winnable."""
    profiles = bootstrap_profiles()
    # `bootstrap_profiles()` returns `ExpertDefinition`, whose knowledge base is
    # `key`. That the two shapes disagree is exactly why `answerable_cases` reads
    # both.
    routable = {p.key for p in profiles if p.routable}

    kept = answerable_cases(profiles, load_eval_set())

    assert kept, "filtering must not empty the set"
    assert all(case["expert"] in routable for case in kept)


def test_the_filter_is_scored_against_the_experts_it_was_given():
    """Not against the installed product. `check_routing_eval` passes candidate
    expert lists -- including one holding an expert that does not exist yet -- so
    the filter must read the argument, never the profile.

    A hardcoded product lookup would pass every test above and then score a
    proposed expert's own eval cases as unwinnable, rejecting an addition for
    failing questions it was created to answer.
    """
    profiles = bootstrap_profiles()
    one = [p for p in profiles if p.routable][:1]

    narrow = answerable_cases(one, load_eval_set())
    wide = answerable_cases(profiles, load_eval_set())

    assert len(narrow) < len(wide), (
        "a single-expert list must keep fewer cases than every expert; if these are "
        "equal the filter is ignoring its argument"
    )
    assert {c['expert'] for c in narrow} == {one[0].key}


def test_the_enterprise_product_still_routes_every_answerable_case_correctly():
    """The accuracy this work must not disturb: 53/53. Recorded here beside the
    filter so a change to either is scored in the same run."""
    profiles = bootstrap_profiles()
    cases = answerable_cases(profiles, load_eval_set())

    result = evaluate_routing(profiles, cases)

    assert result["n"] == 53
    assert result["accuracy"] == 1.0, f"misses: {result['misses']}"
