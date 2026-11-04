"""What a runtime writer may add to the graph, checked against the registry.

The graph ontology -- which labels and relationship types exist -- is an
IMMUTABLE surface (`knowledge_bases/boundary.py`): changing it takes a human
writing code and a reviewed commit. But two runtime paths could add to it
anyway, because both interpolate an LLM-supplied type straight into Cypher:

- `extract.py` asks an LLM for a free-text `relationship_type` (its own prompt
  suggests `WORKS_FOR`, which exists in no knowledge base) and passes it to
  `graph.upsert_rel()`, which does ``MERGE (a)-[r:`{rel_type}`]->(b)``.
- session-memory triplet extraction, which would have exactly the same shape.

This module is the check both go through. It is deliberately *data*-driven:
the allowlist is derived from `KNOWLEDGE_BASES`, so declaring ontology in the
registry is what widens it -- there is no second list to keep in sync, and no
runtime path that can add to it.

See `docs/architecture/ontology_audit.md` for the audit this came out of.
"""
from __future__ import annotations

from dataclasses import dataclass

from knowledge_bases.registry import KNOWLEDGE_BASES, relationship_types

# Labels that are real but deliberately undeclared, with the reason:
#
# - `<Kind>Revision` is derived per managed entity kind and always co-labelled
#   `:Revision`, which IS declared. Handled by `is_known_label` rather than
#   enumerated, so adding an entity kind doesn't require touching this file.
# - `SchemaMigration` is migration bookkeeping, not domain ontology.
# - `Deprecated` is the lifecycle marker (see `docs/architecture/lifecycle.md`).
INFRASTRUCTURE_LABELS = frozenset({"SchemaMigration", "Deprecated"})

DERIVED_LABEL_SUFFIX = "Revision"


def allowed_relationships() -> frozenset[str]:
    """Every relationship type any knowledge base declares."""
    return frozenset(
        relationship
        for kb in KNOWLEDGE_BASES.values()
        for relationship in kb.relationships
    )


def endpoint_reason(predicate: str, subject_label: str, object_label: str) -> str | None:
    """Why this relationship does not accept these endpoints, or `None` if it does.

    A relationship declares which labels may sit on each end. An empty tuple means
    any label. `relationship_types()` is the single source of that typing, so this
    reads the same declarations the changeset validator reads -- there is no second
    list to drift.
    """
    rel = relationship_types().get(predicate)
    if rel is None:
        return f"undeclared relationship type {predicate!r}"
    if rel.subject and subject_label not in rel.subject:
        return (
            f"{predicate} does not accept subject label {subject_label!r}; "
            f"legal: {', '.join(rel.subject)}"
        )
    if rel.object and object_label not in rel.object:
        return (
            f"{predicate} does not accept object label {object_label!r}; "
            f"legal: {', '.join(rel.object)}"
        )
    return None


def allowed_labels() -> frozenset[str]:
    """Every label any knowledge base declares."""
    return frozenset(label for kb in KNOWLEDGE_BASES.values() for label in kb.labels)


def is_known_label(label: str) -> bool:
    """Whether `label` is declared, derived from a declared one, or plumbing.

    `<Kind>Revision` counts as known because every such node also carries the
    declared `:Revision` label; enumerating them would duplicate the composed
    versioned-kind registry (`methodology/kinds.py`) without adding a
    guarantee.
    """
    if label in allowed_labels() or label in INFRASTRUCTURE_LABELS:
        return True
    return label.endswith(DERIVED_LABEL_SUFFIX) and label != DERIVED_LABEL_SUFFIX


def is_known_relationship(relationship: str) -> bool:
    return relationship in allowed_relationships()


class OntologyError(ValueError):
    """A write that would add a label or relationship type at runtime."""


def require_known_relationship(relationship: str, *, actor: str) -> str:
    """Return `relationship`, or raise if it is not declared ontology.

    Raising rather than silently skipping is the point: a dropped edge looks
    like a quiet extraction miss, while a refused one is a visible signal that
    either the extractor is wrong or the ontology needs a reviewed change.
    """
    if not is_known_relationship(relationship):
        raise OntologyError(
            f"{actor} may not create relationship type {relationship!r}: it is "
            "not declared by any knowledge base. Graph ontology is an "
            "architecture surface -- declare it in knowledge_bases/registry.py "
            "in a reviewed commit, or map the claim onto an existing type."
        )
    return relationship


def require_known_label(label: str, *, actor: str) -> str:
    if not is_known_label(label):
        raise OntologyError(
            f"{actor} may not create label {label!r}: it is not declared by any "
            "knowledge base. Declare it in knowledge_bases/registry.py in a "
            "reviewed commit."
        )
    return label


@dataclass(frozen=True)
class Triplet:
    """One extracted `(subject)-[predicate]->(object)` claim.

    `subject_id`/`object_id` are graph ids. Whether a missing object may be
    *created* is the caller's policy (session memory allows it, guarded by
    name-matching); what is never negotiable is the label and predicate, which
    must already be declared ontology. `source_text` is kept so a rejected
    triplet can be reviewed against what was actually said.
    """

    subject_id: str
    subject_label: str
    predicate: str
    object_id: str
    object_label: str
    # Human-readable name of the object, used to match an existing entity
    # before creating a new one. Without it a created entity has nothing to
    # deduplicate against later.
    object_name: str = ""
    source_text: str = ""

    def rejection_reason(self) -> str | None:
        """Why this triplet may not be written, or `None` if it may.

        The endpoint check is the one that was missing. `is_known_relationship`
        only asks that the NAME is declared, so a model that names a legal
        relationship with illegal endpoints was written: `SUPPORTS` declares
        `object=('Decision', 'Claim', ...)`, yet `(:Claim)-[:SUPPORTS]->(:Project)`
        passed. The changeset path already checked endpoints via
        `knowledge_bases.triplets.validate_triplet`; the extraction path did not,
        so the graph accumulated edges its own ontology forbids.
        """
        if not is_known_label(self.subject_label):
            return f"undeclared subject label {self.subject_label!r}"
        if not is_known_label(self.object_label):
            return f"undeclared object label {self.object_label!r}"
        if not is_known_relationship(self.predicate):
            return f"undeclared relationship type {self.predicate!r}"
        if self.subject_id == self.object_id:
            return "self-loop"
        reason = endpoint_reason(self.predicate, self.subject_label, self.object_label)
        if reason is not None:
            return reason
        return None


def partition_triplets(
    triplets: list[Triplet],
) -> tuple[list[Triplet], list[dict[str, str]]]:
    """Split into (writable, rejected-with-reason).

    Rejected triplets are returned rather than dropped so the caller can record
    them as reviewable evidence -- an extractor repeatedly reaching for a
    relationship the ontology lacks is exactly the signal that should reach a
    human, not something to swallow.
    """
    writable, rejected = [], []
    for triplet in triplets:
        reason = triplet.rejection_reason()
        if reason is None:
            writable.append(triplet)
        else:
            rejected.append({
                "subject_id": triplet.subject_id,
                "predicate": triplet.predicate,
                "object_id": triplet.object_id,
                "source_text": triplet.source_text,
                "reason": reason,
            })
    return writable, rejected
