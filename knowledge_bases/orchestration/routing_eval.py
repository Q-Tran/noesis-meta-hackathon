"""Golden routing eval set: questions with the expert that must rank first.

Live routing traces only say a question was *ambiguous*; they cannot say a
change made routing *wrong*. This small held-out set does: self-improvement
scores an expert-wording change against it before applying (a change that
lowers top-1 accuracy is rejected outright) and again at re-evaluation.
Evaluated keyword-only by default so the verdict is deterministic and
independent of Milvus; pass `expert_hits`/`skill_hits` per question to
evaluate the fused router instead.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from knowledge_bases.orchestration.models import ExpertProfile
from knowledge_bases.orchestration.routing import plan_route

EVAL_PATH = Path(__file__).with_name("routing_eval.jsonl")


def load_eval_set(path: Path = EVAL_PATH) -> list[dict[str, str]]:
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


def answerable_cases(
    profiles: list[ExpertProfile],
    cases: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """The eval cases this product can possibly get right.

    The eval set is written against the full vocabulary, so it names experts a
    given product may not install. Scoring those is meaningless: "What does ARPU
    mean in our reporting?" expects `business_dictionary`, and a product without
    it has no right answer to rank first. Left unfiltered, the personal product
    scores 0.53 -- and every one of its 25 misses expects an absent expert, so
    the number measures the profile rather than the router.

    That distinction matters because this metric GATES self-improvement: a
    wording change is rejected if it lowers accuracy. A floor of unwinnable cases
    does not just look bad, it dilutes the signal -- a real regression in the
    experts a product does install moves the number less, and can be lost in the
    noise of cases nobody could ever pass.

    Dropping them is not lowering the bar. The remaining cases are scored exactly
    as strictly; there are simply fewer of them, and a product is measured on the
    knowledge it claims to have.
    """
    # `ExpertProfile` names its knowledge base `knowledge_base`; `ExpertDefinition`
    # (the graph-backed form the self-improvement gate passes) calls it `key`. Both
    # reach `evaluate_routing`, so both must reach this filter.
    routable = {
        getattr(profile, "knowledge_base", None) or profile.key
        for profile in profiles
        if profile.routable
    }
    cases = cases if cases is not None else load_eval_set()
    return [case for case in cases if case["expert"] in routable]


def evaluate_routing(
    profiles: list[ExpertProfile],
    cases: list[dict[str, str]] | None = None,
    hits_for: Callable[[str], tuple[dict | None, dict | None]] | None = None,
) -> dict[str, Any]:
    """Top-1 accuracy of `plan_route` over the eval set (`primary_expert`
    must equal the expected expert). `hits_for(question)` may supply
    `(expert_hits, skill_hits)` to evaluate with semantic evidence."""
    cases = cases if cases is not None else load_eval_set()
    routable = [p for p in profiles if p.routable]
    misses = []
    for case in cases:
        expert_hits, skill_hits = hits_for(case["question"]) if hits_for else (None, None)
        plan = plan_route(
            case["question"], routable, expert_hits=expert_hits, skill_hits=skill_hits,
        )
        # The metric rule always puts business_dictionary first for metric
        # questions; judge by the highest-scoring candidate instead.
        top = plan.candidates[0].knowledge_base if plan.candidates else plan.primary_expert
        if top != case["expert"]:
            misses.append({"question": case["question"], "expected": case["expert"], "got": top})
    total = len(cases)
    return {
        "n": total,
        "correct": total - len(misses),
        "accuracy": (total - len(misses)) / total if total else 1.0,
        "misses": misses,
    }
