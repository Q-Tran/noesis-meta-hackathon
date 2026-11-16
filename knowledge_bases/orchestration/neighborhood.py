"""Walk out from an entity to what the ontology says is related to it.

`hybrid_search` finds entities by *meaning* -- lexical and semantic channels
fused -- but stops there. The reason to keep knowledge in a graph rather than a
vector index is the edges, and reading them took hand-written Cypher per domain
(`semantic_layer/lineage.py` is one). This is the general form: given something
search found, return its neighbourhood.

The traversal is ontology-bounded in both directions:

- relationship types are validated against the registry, so a caller cannot
  probe for edges that are not declared ontology; and
- results are grouped by relationship type, so the answer says *how* things are
  related rather than returning an undifferentiated blob.

Deprecated nodes are excluded by default, matching `hybrid_search` and the
lifecycle convention (`docs/architecture/lifecycle.md`).
"""
from __future__ import annotations

from typing import Any

from graph import Graph
from knowledge_bases.ontology import is_known_relationship

DEFAULT_DEPTH = 1
MAX_DEPTH = 3


class NeighborhoodError(ValueError):
    """A traversal that would leave the declared ontology."""


def _validate_relationships(relationship_types: list[str] | None) -> list[str]:
    if not relationship_types:
        return []
    unknown = [r for r in relationship_types if not is_known_relationship(r.upper())]
    if unknown:
        raise NeighborhoodError(
            f"Undeclared relationship type(s): {', '.join(sorted(unknown))}. "
            "Traversal is bounded by the registry; see knowledge_bases/registry.py."
        )
    return [r.upper() for r in relationship_types]


def entity_neighborhood(
    graph: Graph,
    label: str,
    identity: str,
    *,
    identity_property: str = "id",
    relationship_types: list[str] | None = None,
    depth: int = DEFAULT_DEPTH,
    limit: int = 50,
    include_deprecated: bool = False,
) -> dict[str, Any]:
    """Return what `identity` is connected to, grouped by relationship type.

    `depth` is capped: an unbounded walk on a graph with 158k `FOUND_IN` edges
    returns the whole component, which is never what a caller means by
    "related".
    """
    if depth < 1 or depth > MAX_DEPTH:
        raise NeighborhoodError(
            f"depth must be between 1 and {MAX_DEPTH}; got {depth}. An unbounded "
            "walk returns the whole connected component, not a neighbourhood."
        )
    types = _validate_relationships(relationship_types)
    type_filter = f":{'|'.join(types)}" if types else ""
    deprecated_guard = "" if include_deprecated else "AND NOT related:Deprecated "

    rows = graph.run(
        f"MATCH (n:`{label}` {{`{identity_property}`: $identity}}) "
        f"MATCH path = (n)-[r{type_filter}*1..{depth}]-(related) "
        f"WHERE related <> n {deprecated_guard}"
        "WITH related, relationships(path) AS rels, length(path) AS distance "
        "RETURN DISTINCT head([x IN labels(related) WHERE x <> 'Deprecated']) AS label, "
        "coalesce(related.id, related.name, related.key) AS identity, "
        "coalesce(related.name, related.title, related.statement, "
        "         related.description, '') AS display, "
        "type(head(rels)) AS relationship, distance "
        "ORDER BY distance, relationship, identity "
        "LIMIT $limit",
        identity=identity, limit=limit,
    )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["relationship"], []).append({
            "label": row["label"],
            "identity": row["identity"],
            "display": row["display"],
            "distance": row["distance"],
        })

    return {
        "entity": {"label": label, "identity": identity},
        "depth": depth,
        "total": sum(len(v) for v in grouped.values()),
        "related_by": grouped,
    }
