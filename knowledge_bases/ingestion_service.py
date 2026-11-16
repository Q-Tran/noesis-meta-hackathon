"""The single ingestion entry point: information in, typed triplets written.

`ingest_information` is what makes "write anything that fits the ontology" real.
It takes already-extracted triplets (the LLM call itself belongs to the caller,
which is what keeps this module testable without a model), validates each against
the *described* ontology, writes what fits append-only, and reports what it
declined and why.

Three decisions worth stating, because each is a policy and not an accident:

- **Fitting the ontology is the authorization.** There is no propose/confirm step
  and no write-intent detection. A triplet whose labels, relationship, and
  endpoint pair are all declared and described is a fact this product models, so
  recording it needs no further ceremony. The bounded surface is the ontology
  itself, which is immutable at runtime (`knowledge_bases/boundary.py`) -- so
  ingestion can add unlimited *data* and no *schema*.
- **Partial success is the normal outcome.** One unrecognised sentence must not
  discard a whole document, so rejects are collected rather than raised.
- **A rejection is a signal, not just an error.** Repeated rejects for the same
  shape mean the ontology is missing a type, which is exactly what
  `catalog_noesis_feedback` exists to capture. The reasons are returned so a
  caller can do that instead of silently dropping knowledge.
"""
from __future__ import annotations

from typing import Any

from graph import Graph
from knowledge_bases import ingestion
from knowledge_bases.registry import entity_types
from knowledge_bases.triplets import (
    Triplet,
    filter_properties,
    partition,
    unmet_kind_requirements,
)


def _identity_key(
    label: str,
    identity_value: str,
    props: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The key dict that decides whether two mentions are the same entity.

    Each entity type names its own identity property (`name` for a BusinessTerm,
    `id` for most others), so re-ingesting the same fact converges on one node
    instead of creating one per phrasing.

    Identity can be composite -- `ReviewFinding` is matched on
    `(project_id, key)` -- in which case the extra properties must come from the
    triplet's own props, since a triplet carries only one identity string. A
    composite key with a missing part would silently converge on the wrong node,
    so the missing property is reported rather than defaulted.
    """
    declared = entity_types().get(label)
    if declared is None:
        return {"id": identity_value}

    properties = declared.identity_properties
    key: dict[str, Any] = {properties[0]: identity_value}
    for extra in properties[1:]:
        if not props or extra not in props:
            raise ValueError(
                f"{label} identity is composite {properties}; "
                f"{extra!r} must be supplied in the triplet's properties"
            )
        key[extra] = props[extra]
    return key


def ingest_triplets(
    graph: Graph,
    triplets: list[Triplet],
    *,
    source_id: str | None = None,
    source_props: dict[str, Any] | None = None,
    actor: str = "ingestion",
    extractable_only: bool = True,
) -> dict[str, Any]:
    """Write every triplet that fits the ontology; report the rest.

    Both endpoints are created (or converged onto, by identity) before the edge,
    because `graph.upsert_rel` matches existing nodes and would silently write
    nothing for an endpoint that does not exist yet.
    """
    accepted, rejected = partition(triplets, extractable_only=extractable_only)

    written_entities: list[dict[str, Any]] = []
    written_relationships: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for triplet in accepted:
        try:
            for label, identity, props in (
                (triplet.subject_label, triplet.subject_identity, triplet.subject_props),
                (triplet.object_label, triplet.object_identity, triplet.object_props),
            ):
                if (label, identity) in seen:
                    continue
                seen.add((label, identity))
                written_entities.append(ingestion.create_entity(
                    graph, label, _identity_key(label, identity, props),
                    filter_properties(label, props),
                    source_id=source_id, source_props=source_props, actor=actor,
                ))
            written_relationships.append(ingestion.relate(
                graph,
                triplet.subject_label,
                _identity_key(triplet.subject_label, triplet.subject_identity,
                              triplet.subject_props),
                triplet.relationship,
                triplet.object_label,
                _identity_key(triplet.object_label, triplet.object_identity,
                              triplet.object_props),
                props={"confidence": triplet.confidence} if triplet.confidence != 1.0 else None,
            ))
        except Exception as exc:  # a graph-level refusal, not a typing error
            failures.append({"triplet": triplet, "reason": str(exc)})

    return {
        "entities_written": len(written_entities),
        "relationships_written": len(written_relationships),
        "entities": written_entities,
        "relationships": written_relationships,
        # Everything the ontology declined, with the reason -- repeated rejects
        # of the same shape are the signal that a type is missing.
        "rejected": [
            {"reason": r["reason"], "triplet": _describe(r["triplet"])}
            for r in rejected
        ],
        "failed": [
            {"reason": f["reason"], "triplet": _describe(f["triplet"])}
            for f in failures
        ],
        # Kinds this batch created without the companion edge their own rules
        # require -- a Goal with nothing measuring it, a Decision affecting no
        # architecture. `submit_changeset` refuses these outright; ingestion
        # records them and says so, because incomplete knowledge is still
        # knowledge and the alternative is discarding the document. Surfacing it
        # is what keeps the two paths honest about holding the same standard.
        "incomplete": unmet_kind_requirements(accepted),
    }


def _describe(triplet: Triplet) -> str:
    return (
        f"({triplet.subject_label} {triplet.subject_identity})"
        f"-[:{triplet.relationship}]->"
        f"({triplet.object_label} {triplet.object_identity})"
    )
