"""Retrieval-based routing: score graph-defined experts against a question.

Two signals per expert, fused:

- **keyword**: token overlap between the question and the expert's
  capability phrases / description (always available, deterministic).
- **semantic**: cosine similarity from the `noesis_experts` / `noesis_skills`
  vector index (`knowledge_bases.tools.catalog_index`) -- the best-matching
  capability phrase, or the best-matching skill the expert is allowed to
  call. Absent without Milvus; keyword alone then decides.

The result is a ranked candidate list with reasons and, per selected
expert, a starting delegation built from the expert's `entry_tools` and
those skills' input schemas -- no per-expert Python mapping. The harness's
orchestrator LLM makes the final call among close candidates; the optional
server-side `decide()` does the same with Noesis's own LLM for headless
callers (see `service.py`).
"""
from __future__ import annotations

import re
from typing import Any, Callable
from uuid import uuid4

from knowledge_bases.orchestration.models import (
    Delegation,
    ExpertCandidate,
    ExpertProfile,
    IngestionOffer,
    RoutingPlan,
)


TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOPWORDS = {
    "a", "about", "after", "against", "all", "an", "and", "any", "are", "as", "at",
    "back", "be", "before", "by", "can", "come", "could", "did", "do", "does",
    "for", "from", "get", "give", "go", "has", "have", "help", "here", "how", "i",
    "if", "in", "into", "is", "it", "its", "just", "like", "me", "my", "need",
    "no", "not", "now", "of", "on", "one", "or", "our", "out", "please", "show",
    "so", "some", "tell", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "to", "up", "us", "use", "want", "was",
    "we", "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your",
}

# A question that means "route me" in this many keyword hits or this much
# cosine similarity counts as a full-strength match on that channel.
KEYWORD_SATURATION = 4
SIMILARITY_FLOOR, SIMILARITY_CEILING = 0.20, 0.60
# Experts at/above this fused score are selected; below it only the top
# candidate is, and only when nothing else qualified.
SELECT_THRESHOLD = 0.40
CLARIFICATION_THRESHOLD = 0.30
# Control-plane skills (search/route/hybrid_search/writes) are allowed to
# most experts and read as relevant to almost any question; they carry no
# routing evidence for a particular expert.
GENERIC_SKILL_PREFIXES = ("orchestration.", "tools.")

# Arguments a starting delegation can fill from the question alone.
QUESTION_ARGUMENTS = ("question", "query", "need", "text", "term", "phrase")


def tokens(text: str) -> set[str]:
    """Lowercased content tokens, with a naive plural fold so "targets"
    matches a capability phrase that says "target"."""
    found = set()
    for token in TOKEN_RE.findall(text.lower()):
        if len(token) <= 1 or token in STOPWORDS:
            continue
        found.add(token)
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            found.add(token[:-1])
    return found


METRIC_INTENT_TOKENS = {
    "metric", "metrics", "kpi", "target", "quota", "budget", "attainment",
    "variance",
    # Quantitative business-measurement words a metric question routinely
    # uses without ever saying "metric": plain volume/count language
    # ("net adds", "gross adds", "how many", "totals") and time-series/trend
    # phrasing ("monthly", "trailing 12 months", "YoY", "MoM"). Without
    # these, a request like "net adds dedicated across attb for every month
    # for the last 12 months" scores 0 against every expert's capability
    # keywords and never reaches `semantic_layer` at all (see
    # tests/test_orchestration_routing.py::test_plain_volume_question_still_routes_semantic_layer).
    "net", "gross", "adds", "add", "disconnects", "disconnect", "churn",
    "count", "counts", "total", "totals", "volume", "volumes", "actuals",
    "actual", "value", "values", "number", "numbers",
    "monthly", "quarterly", "weekly", "yearly", "annual", "trailing",
    "yoy", "mom", "qoq", "trend",
}


def skill_allowed(skill_key: str, patterns: list[str]) -> bool:
    return any(
        pattern == skill_key
        or (pattern.endswith(".*") and skill_key.startswith(pattern[:-1]))
        for pattern in patterns
    )


def _scale_similarity(similarity: float | None) -> float:
    if similarity is None:
        return 0.0
    return max(0.0, min(1.0, (similarity - SIMILARITY_FLOOR) / (SIMILARITY_CEILING - SIMILARITY_FLOOR)))


def score_experts(
    question: str,
    profiles: list[ExpertProfile],
    expert_hits: dict[str, dict[str, Any]] | None = None,
    skill_hits: dict[str, float] | None = None,
) -> list[ExpertCandidate]:
    """Rank experts by fused keyword + semantic evidence.

    `expert_hits` is `{expert_key: {"similarity", "matched"}}` from
    `CatalogIndex.search_experts`; `skill_hits` is `{skill_key: similarity}`
    from `CatalogIndex.search_skills`. Both optional.
    """
    query_tokens = tokens(question)
    expert_hits = expert_hits or {}
    skill_hits = skill_hits or {}
    candidates = []
    for profile in profiles:
        description_tokens = tokens(profile.description)
        capability_tokens = tokens(" ".join(profile.effective_capabilities))
        capability_matches = sorted(query_tokens & capability_tokens)
        description_matches = sorted(query_tokens & description_tokens)
        keyword = len(capability_matches) * 2 + len(description_matches)
        keyword_component = min(1.0, keyword / KEYWORD_SATURATION)

        reasons = []
        if capability_matches:
            reasons.append("capabilities:" + ",".join(capability_matches))
        if description_matches:
            reasons.append("description:" + ",".join(description_matches))

        hit = expert_hits.get(profile.key)
        capability_similarity = hit["similarity"] if hit else None
        allowed_skill_hits = {
            key: sim for key, sim in skill_hits.items()
            if skill_allowed(key, profile.allowed_tools)
            and not key.startswith(GENERIC_SKILL_PREFIXES)
        }
        best_skill = max(allowed_skill_hits, key=allowed_skill_hits.get) if allowed_skill_hits else None
        skill_similarity = allowed_skill_hits[best_skill] if best_skill else None
        semantic = max(
            (s for s in (capability_similarity, skill_similarity) if s is not None),
            default=None,
        )
        semantic_component = _scale_similarity(semantic)
        if capability_similarity is not None:
            reasons.append(f"semantic:{capability_similarity:.2f}")
        if best_skill is not None:
            reasons.append(f"skill:{best_skill}:{skill_similarity:.2f}")

        high, low = max(keyword_component, semantic_component), min(keyword_component, semantic_component)
        score = round(0.7 * high + 0.3 * low, 4)
        suggested = sorted(allowed_skill_hits, key=allowed_skill_hits.get, reverse=True)[:3]
        candidates.append(
            ExpertCandidate(
                knowledge_base=profile.key,
                name=profile.name,
                score=score,
                reasons=reasons,
                suggested_tools=suggested,
                system_prompt=profile.effective_prompt,
            )
        )
    priorities = {p.key: p.priority for p in profiles}
    return sorted(
        candidates, key=lambda c: (-c.score, priorities[c.knowledge_base]),
    )


def _fill_arguments(schema: dict[str, Any] | None, question: str) -> dict[str, Any] | None:
    """Build a starting argument payload from a skill's input schema, or
    return None when a required argument cannot be derived from the
    question (e.g. an entity id)."""
    if not schema:
        return {"question": question}
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    arguments: dict[str, Any] = {}
    for name in QUESTION_ARGUMENTS:
        if name in properties:
            arguments[name] = question
            break
    for name in required:
        if name not in arguments:
            return None
    return arguments


def _family_entity_types(expert_key: str) -> list[str]:
    from knowledge_bases.orchestration.hybrid_search import SEARCH_TYPES

    return sorted(
        qualified for qualified, config in SEARCH_TYPES.items()
        if config.family == expert_key and qualified.split(".", 1)[1] != "Value"
    )


def starting_delegation(
    profile: ExpertProfile,
    question: str,
    sequence: int,
    skills_by_key: dict[str, Any] | None = None,
    suggested_tools: list[dict[str, Any]] | None = None,
) -> Delegation:
    """The first bounded call a subagent for `profile` should make: its
    first `entry_skill` whose arguments can be filled from the question,
    else a `hybrid_search` over the expert's own entity family."""
    skills_by_key = skills_by_key or {}
    tools_payload = suggested_tools or []
    for skill_key in profile.entry_tools:
        skill = skills_by_key.get(skill_key)
        schema = skill.input_schema if skill is not None else None
        operation = skill_key.split(".", 1)[1] if "." in skill_key else skill_key
        if schema is None and operation.startswith(("get_", "describe_")):
            continue  # needs an identifier we don't have
        arguments = _fill_arguments(schema, question)
        if arguments is not None:
            return Delegation(
                knowledge_base=profile.key, operation=operation,
                arguments=arguments, sequence=sequence,
                system_prompt=profile.effective_prompt,
                suggested_tools=tools_payload,
            )
    entity_types = _family_entity_types(profile.key)
    if entity_types:
        return Delegation(
            knowledge_base=profile.key, operation="hybrid_search",
            arguments={"query": question, "entity_types": entity_types, "limit": 5},
            sequence=sequence,
            system_prompt=profile.effective_prompt,
            suggested_tools=tools_payload,
        )
    fallback = profile.entry_tools[0] if profile.entry_tools else "hybrid_search"
    return Delegation(
        knowledge_base=profile.key,
        operation=fallback.split(".", 1)[1] if "." in fallback else fallback,
        arguments={}, sequence=sequence,
        system_prompt=profile.effective_prompt,
        suggested_tools=tools_payload,
    )


def plan_route(
    question: str,
    profiles: list[ExpertProfile],
    *,
    expert_hits: dict[str, dict[str, Any]] | None = None,
    skill_hits: dict[str, float] | None = None,
    skills_by_key: dict[str, Any] | None = None,
    select: Callable[[list[ExpertCandidate]], list[str]] | None = None,
) -> RoutingPlan:
    """Score, select, and build starting delegations.

    `select`, when given, overrides the threshold-based selection with an
    external decision over the ranked candidates (e.g. an LLM) -- see
    `service.ExpertOrchestrator.route(decide=True)`.
    """
    if not profiles:
        raise ValueError("At least one graph expert profile is required")
    by_key = {profile.key: profile for profile in profiles}
    candidates = score_experts(question, profiles, expert_hits, skill_hits)

    if select is not None:
        selected_keys = [key for key in select(candidates) if key in by_key]
    else:
        selected_keys = [c.knowledge_base for c in candidates if c.score >= SELECT_THRESHOLD]
    metric_intent = bool(tokens(question) & METRIC_INTENT_TOKENS)
    if metric_intent:
        selected_keys = [k for k in ("business_dictionary", "semantic_layer") if k in by_key] + selected_keys
    if not selected_keys and candidates[0].score > 0:
        # Weak evidence is still evidence: fall back to the single best
        # candidate. Zero evidence is not, so no fallback -- see below.
        selected_keys = [candidates[0].knowledge_base]
    selected_keys = sorted(set(selected_keys), key=lambda key: by_key[key].priority)

    delegations = []
    for sequence, key in enumerate(selected_keys):
        prof = by_key[key]
        expert_suggested: list[dict[str, Any]] = []
        if skills_by_key:
            allowed_skills = [
                s for s in skills_by_key.values()
                if skill_allowed(s.key, prof.allowed_tools)
                and not s.key.startswith(GENERIC_SKILL_PREFIXES)
            ]
            if allowed_skills:
                from knowledge_bases.tools.search import rank_skills
                expert_suggested = rank_skills(
                    question, allowed_skills, semantic=skill_hits, limit=3
                )
        delegations.append(
            starting_delegation(
                prof, question, sequence, skills_by_key, suggested_tools=expert_suggested
            )
        )

    if not selected_keys:
        # Nothing matched at all. `candidates[0]` is then whatever the sort
        # happened to put first -- an arbitrary tie-break among equal zeros --
        # and naming it `primary_expert` presented a coin flip as a decision:
        # the orchestrator would spawn a subagent for a domain the question has
        # nothing to do with, and that subagent's confident-sounding "I found
        # nothing" is much worse than being told the question did not route.
        # Questions about Noesis itself land here, since no domain expert owns
        # them. The ranked candidates are still returned so a caller can choose.
        return RoutingPlan(
            trace_id=str(uuid4()),
            question_summary=question.strip()[:240],
            primary_expert=None,
            confidence=0.0,
            candidates=candidates,
            delegations=[],
            clarification_needed=True,
            # Offered even here. Nothing matching an expert does not mean the
            # question carries nothing worth recording -- and a question about a
            # domain no expert owns is exactly the case where the graph is
            # missing the knowledge.
            ingestion=IngestionOffer(),
            primary_expert_prompt=None,
            suggested_tools=[],
        )

    top = candidates[0].score
    runner_up = candidates[1].score if len(candidates) > 1 else 0.0
    # Confidence is the top score, damped when the runner-up is close:
    # a clear winner at 0.6 is more trustworthy than a 0.6 tied with 0.58.
    confidence = round(min(1.0, top * (0.75 + 0.25 * min(1.0, (top - runner_up) / max(top, 1e-9)))), 4)
    if metric_intent and confidence < SELECT_THRESHOLD:
        confidence = SELECT_THRESHOLD  # the metric rule is itself evidence

    primary_profile = by_key.get(delegations[0].knowledge_base) if delegations else None
    primary_prompt = primary_profile.effective_prompt if primary_profile else None
    primary_tools = delegations[0].suggested_tools if delegations else []

    return RoutingPlan(
        trace_id=str(uuid4()),
        question_summary=question.strip()[:240],
        primary_expert=delegations[0].knowledge_base,
        confidence=confidence,
        candidates=candidates,
        delegations=delegations,
        clarification_needed=confidence < CLARIFICATION_THRESHOLD,
        ingestion=IngestionOffer(),
        primary_expert_prompt=primary_prompt,
        suggested_tools=primary_tools,
    )
