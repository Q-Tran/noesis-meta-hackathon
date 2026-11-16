"""Ontology-driven triplet extraction: the single ingestion path.

Any incoming information -- a sentence a user typed, a meeting transcript, a
document -- is turned into `(subject, relationship, object)` triplets typed
against the *described* ontology, and whatever fits is written. There is no
write-intent detection and no changeset ceremony: fitting the ontology is what
makes a fact ingestable.

This module differs from `extract.py` in the way that matters for products.
`extract.py` hardcodes ten Pydantic classes (`ExPerson`, `ExTable`, ...), so it
can only ever extract the entities someone wrote a class for, and an installed
ontology package's types are invisible to it. Here the extractable vocabulary is
*derived* from `registry.entity_types()`/`relationship_types()`, so installing a
package extends ingestion automatically and uninstalling one narrows it -- the
same composition rule every other surface follows.

Descriptions are load-bearing, not documentation. A model asked to choose among
bare names like `AFFECTS`, `CHANGES`, `CONSTRAINS`, `OVERRIDES` has nothing to
choose on; language is context-dependent, so each type must say what it means
and what it is not. `render_ontology_prompt` is what the model actually sees, and
it is built from those descriptions plus each relationship's legal endpoints.

Endpoint typing is the second half of disambiguation and the part a prompt cannot
guarantee. `validate_triplet` rejects a well-named edge between the wrong kinds
of thing -- nothing else in Noesis stops `(Metric)-[:REPORTS_TO]->(Table)`,
because `graph.upsert_rel` validates that the label and the relationship each
exist, never that the combination is meaningful.

Writes go through `knowledge_bases.ingestion`, so they are append-only: a fact
that restates or contradicts an existing one is related to it rather than
overwriting it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from knowledge_bases.base import (
    EntityType,
    PropertyType,
    RelationshipType,
    described_properties,
    property_names,
)
from knowledge_bases.registry import entity_types, relationship_types


class TripletError(ValueError):
    """A triplet that does not fit the described ontology."""


@dataclass(frozen=True)
class Triplet:
    """One extracted fact, typed against the ontology."""

    subject_label: str
    subject_identity: str
    relationship: str
    object_label: str
    object_identity: str
    # Extractor's own properties for each endpoint, filtered to the declared
    # `properties` of its entity type before anything is written.
    subject_props: dict[str, Any] | None = None
    object_props: dict[str, Any] | None = None
    # Properties of the edge itself -- `CALLS {confidence, reason, line}`,
    # `FOUND_IN {role}` and `INCLUDES {position}` are all real.
    properties: dict[str, Any] | None = None
    confidence: float = 1.0


def _present_endpoints(endpoints: tuple[str, ...], entities: dict) -> str:
    """Render an endpoint list, keeping only labels this prompt defines.

    Empty declared endpoints mean "any label", which stays `any`. A declared list that
    survives filtering renders normally. A declared list left EMPTY by filtering also
    renders as `any` -- and that case cannot arise while `drop_unsatisfiable` is on,
    because a relationship with no offered endpoint is removed outright; it matters only
    for the unpruned diagnostic view, where showing the edge with a truthful "any" beats
    printing an empty bracket.
    """
    present = [name for name in endpoints if name in entities]
    return ", ".join(present) if present else "any"


def _present_contrasts(
    contrasts: tuple[str, ...],
    entities: dict,
    rels: dict,
) -> list[str]:
    """Keep only the contrasts naming a type this prompt actually offers.

    `not_to_be_confused_with` is written against the FULL vocabulary, and that is
    correct: `Goal` really is confusable with `Metric`, and saying so is the most
    load-bearing line in its description. But a contrast is only useful if the reader
    can act on it. Rendering "Goal, NOT Metric" to a model whose prompt never defines
    `Metric` does the opposite of the intent -- it introduces a type, asserts it
    exists, and gives no way to choose it. The model has been taught a label it will
    then never see offered, which is how a scoped extraction starts inventing
    out-of-scope types.

    This became reachable the moment products could install different ontology sets. A
    personal product that omits `project_management` and `semantic_layer` was rendering
    21 distinct absent names -- `Requirement`, `Metric`, `Epic`, `AcceptanceCriterion`
    -- from 53 stale contrasts. Nothing was mis-declared; the declarations are right,
    the prompt just cannot honour them all.

    Filtered rather than fixed at the declaration, deliberately. Dropping `Metric` from
    `Goal`'s contrasts to satisfy the personal product would DEGRADE the enterprise
    prompt, where the confusion is real and the guidance earns its tokens. The contrast
    set is a property of the vocabulary; what a given prompt can say about it is a
    property of the installed product.

    Also prunes contrasts pointing at a type withheld by `include_internal=False` or
    dropped by scoping -- same reasoning, since both are absent from this prompt.
    """
    offered = set(entities) | set(rels)
    return [name for name in contrasts if name in offered]


def render_ontology_prompt(
    labels: tuple[str, ...] | None = None,
    relationships: tuple[str, ...] | None = None,
    include_internal: bool = False,
    drop_unsatisfiable: bool = True,
) -> str:
    """The extractable vocabulary, with meanings, as the model sees it.

    Built from the composed registry rather than a hand-maintained list, so a
    newly installed ontology package is extractable without touching this file.
    `labels`/`relationships` narrow the vocabulary for a scoped extraction (one
    knowledge base, say); `None` offers everything installed.

    `drop_unsatisfiable` additionally removes relationships that no offered label
    could legally participate in. Pass `False` to see the unpruned vocabulary for a
    label set -- useful when diagnosing whether a scope was chosen too narrowly.
    """
    entities = entity_types()
    rels = relationship_types()
    # Types only repository code writes are withheld by default. Offering a
    # versioning spine or a telemetry row to a model reading prose invites a
    # confident bad guess, and the extractor has no way to know the difference.
    if not include_internal:
        entities = {k: v for k, v in entities.items() if v.extractable}
        rels = {k: v for k, v in rels.items() if v.extractable}
    if labels is not None:
        entities = {k: v for k, v in entities.items() if k in labels}
    if relationships is not None:
        rels = {k: v for k, v in rels.items() if k in relationships}
    if labels is not None and drop_unsatisfiable:
        # Scoping labels alone is barely worth doing: relationship descriptions are
        # ~42% of the prompt's bytes (measured), so narrowing the labels while
        # keeping all 61 relationships still leaves ~68% of the full cost. An edge
        # whose legal subject or object has no member in the offered label set can
        # never be validly produced -- `validate_triplet` would reject it -- so
        # describing it spends tokens teaching a move the model is not allowed to
        # make.
        #
        # Filtered against the FINAL label set, never per knowledge base: 19 of the
        # 61 extractable relationships cross package boundaries, so filtering one
        # KB at a time would drop `MADE_BY` from a decisions-scoped prompt and lose
        # the actor in "Alice decided X".
        offered = set(entities)
        rels = {
            name: rel for name, rel in rels.items()
            if (not rel.subject or offered & set(rel.subject))
            and (not rel.object or offered & set(rel.object))
        }

    lines = ["ENTITY TYPES -- choose the one whose description fits:"]
    for label in sorted(entities):
        entity = entities[label]
        lines.append(f"- {label}: {entity.description}")
        described = described_properties(entity.properties)
        if entity.properties:
            names = property_names(entity.properties)
            plain = [n for n in names if n not in described]
            if plain:
                lines.append(f"    properties: {', '.join(plain)}")
            # Described properties earn their own line: a name alone cannot tell
            # an extractor that `status` means todo/in_progress/done here and
            # ok/error elsewhere.
            for prop in described.values():
                detail = f"    - {prop.name}: {prop.description}"
                if prop.enum:
                    detail += f" One of: {', '.join(prop.enum)}."
                if prop.required:
                    detail += " Required."
                if prop.example:
                    detail += f" Example: {prop.example}"
                lines.append(detail)
        contrasts = _present_contrasts(entity.not_to_be_confused_with, entities, rels)
        if contrasts:
            lines.append(f"    NOT: {', '.join(contrasts)}")

    lines.append("")
    lines.append("RELATIONSHIP TYPES -- legal endpoints are enforced:")
    for name in sorted(rels):
        rel = rels[name]
        # Endpoints are narrowed to the labels this prompt offers, for the same
        # reason the contrasts are. A declaration lists every label that is a legal
        # endpoint across the WHOLE vocabulary -- `IMPLEMENTS` legitimately accepts
        # `Epic`, `UserStory` and `Requirement` -- but naming them to a model whose
        # prompt defines none of them advertises a triplet `validate_triplet` will
        # reject. Worse than wasted tokens: the model is told the shape it should aim
        # for, and the closest thing it can build is refused.
        #
        # A product without `project_management` was offering five uncreatable
        # endpoints on `IMPLEMENTS` alone, across seven relationships.
        subject = _present_endpoints(rel.subject, entities)
        obj = _present_endpoints(rel.object, entities)
        lines.append(f"- ({subject}) -[:{name}]-> ({obj}): {rel.description}")
        contrasts = _present_contrasts(rel.not_to_be_confused_with, entities, rels)
        if contrasts:
            lines.append(f"    NOT: {', '.join(contrasts)}")
        for prop in described_properties(rel.properties).values():
            detail = f"    - {prop.name}: {prop.description}"
            if prop.enum:
                detail += f" One of: {', '.join(prop.enum)}."
            lines.append(detail)

    lines.append("")
    lines.append(
        "Emit only triplets whose subject and object labels are legal for the "
        "relationship. If no relationship fits, omit the fact rather than "
        "forcing the closest name."
    )
    return "\n".join(lines)


def validate_triplet(triplet: Triplet, *, extractable_only: bool = True) -> Triplet:
    """Return `triplet` if it fits the described ontology, else raise.

    Checks the three things a prompt cannot guarantee: that both labels and the
    relationship are described at all, and that this relationship permits these
    endpoint labels. The endpoint check is the one with no equivalent elsewhere
    in Noesis.
    """
    entities = entity_types()
    rels = relationship_types()

    for role, label in (("subject", triplet.subject_label),
                        ("object", triplet.object_label)):
        if label not in entities:
            raise TripletError(
                f"{role} label {label!r} is not a described entity type; "
                f"an undescribed label cannot be extracted against"
            )
    rel = rels.get(triplet.relationship)
    if rel is None:
        raise TripletError(
            f"relationship {triplet.relationship!r} is not a described "
            "relationship type"
        )
    # The prompt withholds these, but a model can still name one from memory, and
    # writing to a versioning spine or telemetry row from prose would corrupt
    # structure that repository code maintains. A source that IS repository code --
    # a code scan, a schema import -- is allowed past this, and `extractable_only`
    # is how its caller says so. The label, relationship, and endpoint checks
    # still run: relaxing who may write is not relaxing what may be written.
    if extractable_only and not rel.extractable:
        raise TripletError(
            f"{triplet.relationship} is written by repository code only and "
            "cannot be ingested from extracted information"
        )
    if extractable_only:
        for role, label in (("subject", triplet.subject_label),
                            ("object", triplet.object_label)):
            if not entities[label].extractable:
                raise TripletError(
                    f"{role} label {label!r} is created by repository code only "
                    "and cannot be ingested from extracted information"
                )
    if rel.subject and triplet.subject_label not in rel.subject:
        raise TripletError(
            f"{triplet.relationship} does not accept subject "
            f"{triplet.subject_label!r} (legal: {', '.join(rel.subject)})"
        )
    if rel.object and triplet.object_label not in rel.object:
        raise TripletError(
            f"{triplet.relationship} does not accept object "
            f"{triplet.object_label!r} (legal: {', '.join(rel.object)})"
        )

    # Property values are checked against their descriptors, so a value from the
    # wrong enum is refused here rather than stored as plausible-looking noise.
    for role, label, props in (
        ("subject", triplet.subject_label, triplet.subject_props),
        ("object", triplet.object_label, triplet.object_props),
    ):
        problems = check_properties(entities[label].properties, props)
        if problems:
            raise TripletError(f"{role} {label}: {'; '.join(problems)}")
    problems = check_properties(rel.properties, triplet.properties)
    if problems:
        raise TripletError(f"{triplet.relationship}: {'; '.join(problems)}")

    return triplet


def filter_properties(label: str, props: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only properties the entity type declares, dropping the rest.

    An extractor inventing a property would put un-queryable data on a node that
    every reader then has to guess about, so unknown keys are dropped rather
    than written.
    """
    if not props:
        return {}
    declared = entity_types().get(label)
    if declared is None or not declared.properties:
        return {}
    allowed = set(property_names(declared.properties))
    return {k: v for k, v in props.items() if k in allowed}


def check_properties(
    properties: tuple[str | PropertyType, ...],
    props: dict[str, Any] | None,
) -> list[str]:
    """Why these property values do not fit their descriptors, if they do not.

    Only described properties can be checked; a bare name declares no constraint,
    which is exactly why describing them matters. The enum check is the valuable
    one: `status` is backed by five different enums across the labels that
    declare it, so a value from the wrong set looks plausible and silently makes
    the property unqueryable.
    """
    described = described_properties(properties)
    supplied = props or {}
    problems: list[str] = []

    for name, prop in described.items():
        if prop.required and name not in supplied:
            problems.append(f"{name!r} is required but was not extracted")
        value = supplied.get(name)
        if value is None:
            continue
        if prop.enum:
            values = value if isinstance(value, list) else [value]
            for item in values:
                if item not in prop.enum:
                    problems.append(
                        f"{name}={item!r} is not one of "
                        f"{', '.join(prop.enum)}"
                    )
    return problems


def partition(
    triplets: list[Triplet], *, extractable_only: bool = True
) -> tuple[list[Triplet], list[dict[str, Any]]]:
    """Split triplets into those that fit the ontology and those that do not.

    Ingestion is best-effort by design: one bad triplet in a document must not
    discard the rest. Rejections are returned with their reason so a tool can
    report what it declined to write, which is also the signal for a genuinely
    missing ontology type (`catalog_noesis_feedback`).
    """
    accepted: list[Triplet] = []
    rejected: list[dict[str, Any]] = []
    for triplet in triplets:
        try:
            accepted.append(validate_triplet(triplet, extractable_only=extractable_only))
        except TripletError as exc:
            rejected.append({
                "triplet": triplet,
                "reason": str(exc),
            })
    return accepted, rejected

def unmet_kind_requirements(triplets: list[Triplet]) -> list[dict[str, Any]]:
    """Kinds created by this batch whose traceability rules it does not satisfy.

    Some entity kinds are only meaningful with a companion edge, and
    `methodology/kinds.py` already states which: a Goal needs an inbound
    `MEASURES` from a SuccessMeasure, a Decision needs an outgoing `AFFECTS` to an
    Architecture, a Capability needs `SATISFIES` a Requirement. `submit_changeset`
    enforces those, so the same fact arriving through a changeset and through
    extraction would otherwise be held to different standards -- extraction being
    the lenient one, which is how an unmeasurable Goal gets into the graph.

    Checked across the whole batch rather than per triplet, because the companion
    edge normally comes from a different sentence of the same document.

    Returned rather than raised: the requirement is about knowledge being
    incomplete, not malformed, and refusing the batch would discard the part that
    is fine. The caller decides whether an incomplete kind is worth recording.
    """
    from knowledge_bases.registry import versioned_kinds

    kinds = versioned_kinds()
    present: set[tuple[str, str]] = set()
    edges: list[tuple[str, str, str, str, str]] = []
    for triplet in triplets:
        present.add((triplet.subject_label, triplet.subject_identity))
        present.add((triplet.object_label, triplet.object_identity))
        edges.append((
            triplet.subject_label, triplet.subject_identity,
            triplet.relationship,
            triplet.object_label, triplet.object_identity,
        ))

    gaps: list[dict[str, Any]] = []
    for label, identity in sorted(present):
        kind = kinds.get(label)
        if kind is None or not kind.requirements:
            continue
        for requirement in kind.requirements:
            if requirement.direction == "outbound":
                satisfied = any(
                    s_label == label and s_id == identity
                    and rel == requirement.relationship
                    and o_label in requirement.endpoint_kinds
                    for s_label, s_id, rel, o_label, _ in edges
                )
            else:
                satisfied = any(
                    o_label == label and o_id == identity
                    and rel == requirement.relationship
                    and s_label in requirement.endpoint_kinds
                    for s_label, _, rel, o_label, o_id in edges
                )
            if not satisfied:
                gaps.append({
                    "label": label,
                    "identity": identity,
                    "requirement": (
                        f"{label} needs an {requirement.direction} "
                        f"{requirement.relationship} to "
                        f"{' or '.join(requirement.endpoint_kinds)}"
                    ),
                    "message": kind.message,
                })
    return gaps
