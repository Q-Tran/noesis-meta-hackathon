"""`extract.py`'s vocabulary comes from the ontology, not from a constant.

The relationship list offered to the extracting LLM used to be
`allowed_relationships()` minus a hardcoded set of 24 names kept in `extract.py`.
That constant was a drift hazard in the direction that fails quietly: a package
adding a versioning or telemetry edge would have it offered to the model, and
`graph.upsert_rel` would happily write it, because the edge really is declared.
"""
from __future__ import annotations

import extract
from knowledge_bases.ontology import allowed_relationships
from knowledge_bases.registry import relationship_types

# The exact set `extract.py` used to exclude by hand, kept here as the thing the
# ontology flags must keep reproducing.
HISTORICAL_INTERNAL = {
    "CURRENT", "SUPERSEDED", "SUPERSEDED_BY", "SUPERSEDES", "INCLUDES",
    "INCLUDES_RELATIONSHIP", "CATALOGS", "HAS_TRACE", "HAS_CAPABILITY",
    "CAN_USE", "CALLED", "JUDGES", "PART_OF", "DELEGATED_TO", "RESULTED_IN",
    "CONSIDERED", "PRODUCED", "RECORDED", "HAS_RUN",
    "HAS_BASELINE", "HAS_KNOWLEDGE", "APPROVED", "CREATED",
    "EXTRACTED_FROM", "MENTIONS",
}


def test_the_derived_vocabulary_matches_the_list_it_replaced():
    """Proves the refactor changed where the answer comes from, not the answer."""
    expected = sorted(allowed_relationships() - HISTORICAL_INTERNAL)

    assert extract._extractable_relationships() == expected


def test_every_historically_internal_edge_is_flagged_in_its_own_package():
    """The reason an edge is withheld now lives next to its declaration, so the
    package that adds such an edge is the package that must mark it."""
    described = relationship_types()

    for name in sorted(HISTORICAL_INTERNAL):
        rel = described.get(name)
        assert rel is not None, f"{name} is no longer described"
        assert not rel.extractable, (
            f"{name} is written by repository code but is offered to extractors"
        )


def test_the_extractor_is_never_offered_a_type_no_package_declares():
    """The original bug this vocabulary fixed: a free-text hint suggested
    `WORKS_FOR`, which nothing declares, and `upsert_rel` would have written it --
    adding ontology at runtime."""
    declared = allowed_relationships()

    for name in extract._extractable_relationships():
        assert name in declared, f"{name} is offered but not declared"


def test_ingestion_refuses_an_internal_edge_even_when_asked_directly():
    """The prompt withholds these, but a model can still name one from memory, so
    the refusal has to be enforced at the write boundary too."""
    import pytest

    from knowledge_bases.triplets import Triplet, TripletError, validate_triplet

    with pytest.raises(TripletError, match="repository code only"):
        validate_triplet(Triplet(
            subject_label="Metric", subject_identity="a",
            relationship="SUPERSEDES",
            object_label="Metric", object_identity="b",
        ))
