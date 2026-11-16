"""User conversations with the agent as a provenance source -- with zero
duplication.

The graph already keeps the conversation: every turn is recorded as
`(:Session)-[:HAS_TRACE]->(:RoutingTrace)`, and `RoutingTrace.question`
carries the verbatim user prompt (the trace store and the routing repository
each write it on their traces). This module adds exactly one thing: the
grounding edge that makes a user-supplied claim walkable to those words.

`(Claim)-[:FOUND_IN {role: "origin"}]->(RoutingTrace)` -- the same edge, same
role, that `extract.py` gives an ingested claim pointing at its verbatim
`Chunk`. The trace is the conversation's chunk: the verbatim record of one
unit of the source, under its source anchor (`Session`, via `HAS_TRACE`).
No `Chunk` copies, no conversation `SourceNode`, no embedding of the
transcript: the words live once, on the turn, and "why is this data the way
it is?" resolves `(Claim)-[:FOUND_IN {origin}]->(:RoutingTrace {question})`
in one hop.
"""
from __future__ import annotations

from graph import Graph


def ground_claim_in_trace(graph: Graph, trace_id: str, claim_id: str) -> dict:
    """Substantiate a filed `Claim` by the conversation turn that produced it.

    Writes `(Claim)-[:FOUND_IN {role: "origin"}]->(RoutingTrace)` through
    `graph.upsert_rel`, not hand-rolled Cypher: that is the boundary which
    validates the type and both endpoint labels against the registry and stamps
    `created_at`, so the grounding edge is datable like every other edge.
    `FOUND_IN` declares only `role`, so a bespoke `recorded_at` written here
    would be undeclared property drift. The turn's recorded `question` is the
    verbatim evidence; nothing here copies it. Raises when either endpoint is
    missing -- `upsert_rel` matches both ends, so an empty result means exactly
    that, and a grounding edge to a trace that does not exist would be a
    provenance lie.
    """
    if not trace_id or not trace_id.strip():
        raise ValueError("A trace_id is required to ground a claim in the conversation")
    rows = graph.upsert_rel(
        "Claim", {"id": claim_id}, "FOUND_IN",
        "RoutingTrace", {"id": trace_id.strip()}, {"role": "origin"},
    )
    if not rows:
        raise ValueError(
            f"Cannot ground claim {claim_id}: no Claim or RoutingTrace {trace_id!r} "
            "exists to substantiate it"
        )
    return {"claim_id": claim_id, "trace_id": trace_id.strip(), "grounded": True}
