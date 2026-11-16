"""Runtime writers cannot add ontology; the registry is the only source.

`boundary.py` declares graph ontology an architecture surface -- a human
writing code and a reviewed commit. Two runtime paths interpolate an
LLM-supplied type into Cypher and so could have bypassed that. These tests
pin the check that closes it. See `docs/architecture/ontology_audit.md`.
"""
from __future__ import annotations

import pytest

from knowledge_bases.ontology import (
    OntologyError,
    Triplet,
    allowed_labels,
    allowed_relationships,
    is_known_label,
    is_known_relationship,
    partition_triplets,
    require_known_label,
    require_known_relationship,
)
from knowledge_bases.registry import KNOWLEDGE_BASES


# -- the allowlist is the registry, not a second list ------------------------------

def test_the_allowlist_is_derived_from_the_registry():
    """No parallel list to keep in sync: declaring ontology in the registry is
    what widens what may be written."""
    for kb in KNOWLEDGE_BASES.values():
        for relationship in kb.relationships:
            assert is_known_relationship(relationship)
        for label in kb.labels:
            assert is_known_label(label)


def test_a_relationship_no_knowledge_base_declares_is_unknown():
    # The exact type extract.py used to suggest to the LLM, which no knowledge
    # base declares -- see the ontology audit.
    assert not is_known_relationship("WORKS_FOR")
    assert "WORKS_FOR" not in allowed_relationships()


def test_methodology_structural_relationships_are_declared():
    """These are written by methodology/repository.py and enforced by its validation,
    so they are real ontology; the registry simply never mirrored them."""
    for relationship in ("HAS_KNOWLEDGE", "HAS_RUN", "HAS_BASELINE", "RECORDED"):
        assert is_known_relationship(relationship)


def test_derived_revision_labels_are_known_without_being_enumerated():
    """`<Kind>Revision` nodes always also carry the declared `:Revision`."""
    assert is_known_label("ReviewFindingRevision")
    assert is_known_label("RequirementRevision")
    # The suffix alone is not a licence: an unrelated label is still refused.
    assert not is_known_label("TotallyMadeUpThing")


def test_infrastructure_labels_are_known():
    assert is_known_label("SchemaMigration")
    assert is_known_label("Deprecated")


# -- refusing, not silently skipping -----------------------------------------------

def test_an_undeclared_relationship_is_refused_with_a_reviewable_message():
    with pytest.raises(OntologyError) as exc:
        require_known_relationship("WORKS_FOR", actor="extract")

    message = str(exc.value)
    assert "WORKS_FOR" in message
    assert "registry.py" in message  # tells the human where to declare it
    assert "reviewed commit" in message


def test_an_undeclared_label_is_refused():
    with pytest.raises(OntologyError, match="MadeUpLabel"):
        require_known_label("MadeUpLabel", actor="extract")


def test_a_declared_relationship_passes_through():
    assert require_known_relationship("SUPPORTS", actor="extract") == "SUPPORTS"


# -- triplet partitioning ----------------------------------------------------------

def _triplet(**overrides):
    base = dict(
        subject_id="a", subject_label="Claim", predicate="SUPPORTS",
        object_id="b", object_label="Decision", source_text="t",
    )
    base.update(overrides)
    return Triplet(**base)


def test_a_valid_triplet_is_writable():
    writable, rejected = partition_triplets([_triplet()])

    assert len(writable) == 1 and rejected == []


@pytest.mark.parametrize("overrides,reason", [
    ({"predicate": "WORKS_FOR"}, "undeclared relationship type"),
    ({"subject_label": "Nonsense"}, "undeclared subject label"),
    ({"object_label": "Nonsense"}, "undeclared object label"),
    ({"object_id": "a"}, "self-loop"),
])
def test_an_invalid_triplet_is_rejected_with_its_reason(overrides, reason):
    writable, rejected = partition_triplets([_triplet(**overrides)])

    assert writable == []
    assert len(rejected) == 1
    assert reason in rejected[0]["reason"]


def test_rejected_triplets_keep_their_source_text_for_review():
    """A rejection is evidence the ontology may need a reviewed change, so it
    has to carry enough context to judge -- dropping it silently would hide
    exactly the signal worth acting on."""
    _, rejected = partition_triplets([
        _triplet(predicate="WORKS_FOR", source_text="Ana works for Platform"),
    ])

    assert rejected[0]["source_text"] == "Ana works for Platform"
    assert rejected[0]["predicate"] == "WORKS_FOR"


def test_one_bad_triplet_does_not_discard_the_good_ones():
    writable, rejected = partition_triplets([
        _triplet(subject_id="a", object_id="b"),
        _triplet(subject_id="c", object_id="d", predicate="WORKS_FOR"),
        _triplet(subject_id="e", object_id="f", object_label="Decision",
                  predicate="SUPPORTS"),
    ])

    assert len(writable) == 2 and len(rejected) == 1


# -- the write boundary itself -----------------------------------------------------

class RecordingGraph:
    """Captures the Cypher `upsert_rel` would run, without a database."""

    def __init__(self):
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))
        return []


def test_upsert_rel_refuses_an_undeclared_relationship_type():
    """The hole this closes: `upsert_rel` interpolates `rel_type` into Cypher,
    so an LLM-supplied type became graph ontology at runtime."""
    from graph import Graph

    graph = Graph.__new__(Graph)
    graph.run = RecordingGraph().run

    with pytest.raises(OntologyError, match="WORKS_FOR"):
        graph.upsert_rel("Person", {"id": "p"}, "WORKS_FOR", "Project", {"id": "x"})


def test_upsert_rel_refuses_an_undeclared_label():
    from graph import Graph

    graph = Graph.__new__(Graph)
    graph.run = RecordingGraph().run

    with pytest.raises(OntologyError, match="MadeUpLabel"):
        graph.upsert_rel("MadeUpLabel", {"id": "p"}, "SUPPORTS", "Decision", {"id": "x"})


def test_upsert_rel_still_writes_a_declared_relationship():
    from graph import Graph

    recorder = RecordingGraph()
    graph = Graph.__new__(Graph)
    graph.run = recorder.run

    graph.upsert_rel("Claim", {"id": "i"}, "SUPPORTS", "Decision", {"id": "p"})

    assert recorder.queries
    assert "`SUPPORTS`" in recorder.queries[0][0]
