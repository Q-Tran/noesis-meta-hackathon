"""Ingestion helpers.

- `define_metric`: register a Metric node + its provenance edges (USES columns,
  COMPUTED_FROM tables).
- `add_decision` / `add_goal` / `add_alternative`: create Layer-C nodes.
- stance edges: `support` / `challenge` connect a Claim to a Decision.
- `ingest_text`: pass unstructured text to the open-world reviewer (see review.py).

Imported external evidence is registered directly as a `SourceNode` (see
`knowledge_bases.provenance.queries.register_source`) rather than through a
helper here; there's no separate `ExternalInfo` node anymore (see
`docs/architecture/provenance.md`).
"""
from __future__ import annotations

from graph import Graph
from knowledge_bases.semantic_layer.metric_registry import register_metric
from knowledge_bases.strategy.models import Alternative, Decision, Goal
from knowledge_bases.semantic_layer.models import Metric


def define_metric(metric: Metric, graph: Graph | None = None) -> str:
    graph = graph or Graph()
    register_metric(metric, graph=graph)
    return metric.id


def add_goal(goal: Goal, graph: Graph | None = None) -> str:
    graph = graph or Graph()
    graph.upsert_node("Goal", {"id": goal.id}, {
        "target": goal.target,
        "metric_id": goal.metric_id,
        "target_value": goal.target_value,
        "comparator": goal.comparator,
        "created_at": goal.created_at,
    })
    return goal.id


def add_alternative(alt: Alternative, graph: Graph | None = None) -> str:
    graph = graph or Graph()
    graph.upsert_node("Alternative", {"id": alt.id}, {"name": alt.name, "score": alt.score})
    return alt.id


def add_decision(decision: Decision, goal_id: str | None = None,
                  measured_by_claim: str | None = None,
                  made_by_person_id: str | None = None,
                  chosen_over: list[tuple[str, str]] | None = None,
                  graph: Graph | None = None) -> str:
    """Create a Decision and wire INTENDED_FOR / MEASURED_BY / MADE_BY / CHOSEN_OVER edges.

    `chosen_over` is a list of (alternative_id, better) where better is
    'higher' or 'lower' (direction in which a better score beats the decision).
    """
    graph = graph or Graph()
    graph.upsert_node("Decision", {"id": decision.id}, {
        "description": decision.description,
        "status": decision.status.value,
        "score": decision.score,
        "review_reasons": decision.review_reasons,
        "made_at": decision.made_at,
    })
    if goal_id:
        graph.upsert_rel("Decision", {"id": decision.id}, "INTENDED_FOR", "Goal", {"id": goal_id})
    if measured_by_claim:
        graph.upsert_rel("Decision", {"id": decision.id}, "MEASURED_BY",
                         "Claim", {"id": measured_by_claim})
    if made_by_person_id:
        graph.upsert_rel("Decision", {"id": decision.id}, "MADE_BY", "Person", {"id": made_by_person_id})
    for alt_id, better in (chosen_over or []):
        graph.upsert_rel("Decision", {"id": decision.id}, "CHOSEN_OVER",
                         "Alternative", {"id": alt_id}, {"better": better})
    return decision.id


def support(claim_id: str, decision_id: str, weight: float = 0.5,
            valid_if: str | None = None, status: str = "active",
            graph: Graph | None = None) -> None:
    """Claim -> SUPPORTS -> Decision. `valid_if` is a premise condition,
    e.g. '< 3' or 'value <= 1.5', evaluated against the claim value."""
    graph = graph or Graph()
    graph.upsert_rel("Claim", {"id": claim_id}, "SUPPORTS",
                     "Decision", {"id": decision_id},
                     {"weight": weight, "valid_if": valid_if, "status": status})


def challenge(claim_id: str, decision_id: str, weight: float = 0.5,
              kind: str = "rebutting", status: str = "active",
              graph: Graph | None = None) -> None:
    """Claim -> CHALLENGES -> Decision (opposing evidence)."""
    graph = graph or Graph()
    graph.upsert_rel("Claim", {"id": claim_id}, "CHALLENGES",
                     "Decision", {"id": decision_id},
                     {"weight": weight, "kind": kind, "status": status})


def ingest_text(texts: list[str], graph: Graph | None = None) -> bool:
    """Run unstructured text through the open-world reviewer."""
    from review import open_world_review

    return open_world_review(texts, graph=graph)
