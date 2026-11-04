"""Traversal is bounded by the declared ontology.

`hybrid_search` finds entities by meaning; this is the step that says what they
connect to. The tests that matter are the bounds -- an unbounded walk on a graph
with 158k `FOUND_IN` edges returns the whole component, and an unvalidated
relationship type would let a caller probe for edges outside the ontology.
"""
from __future__ import annotations

import pytest

from knowledge_bases.orchestration.neighborhood import (
    MAX_DEPTH,
    NeighborhoodError,
    entity_neighborhood,
)


class FakeGraph:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))
        return self.rows


def _row(relationship="SUPPORTS", identity="x", distance=1):
    return {
        "label": "Project", "identity": identity, "display": "X",
        "relationship": relationship, "distance": distance,
    }


def test_results_are_grouped_by_relationship_type():
    """An undifferentiated blob does not say *how* things are related."""
    graph = FakeGraph([_row("SUPPORTS", "a"), _row("DEPENDS_ON", "b"), _row("SUPPORTS", "c")])

    result = entity_neighborhood(graph, "Project", "p")

    assert set(result["related_by"]) == {"SUPPORTS", "DEPENDS_ON"}
    assert len(result["related_by"]["SUPPORTS"]) == 2
    assert result["total"] == 3


def test_an_undeclared_relationship_type_is_refused():
    with pytest.raises(NeighborhoodError, match="WORKS_FOR"):
        entity_neighborhood(
            FakeGraph(), "Project", "p", relationship_types=["WORKS_FOR"],
        )


def test_a_declared_relationship_type_is_accepted_and_filters_the_walk():
    graph = FakeGraph([])
    entity_neighborhood(graph, "Project", "p", relationship_types=["depends_on"])

    assert ":DEPENDS_ON" in graph.queries[0][0]


@pytest.mark.parametrize("depth", [0, -1, MAX_DEPTH + 1])
def test_depth_is_capped(depth):
    """An unbounded walk returns the whole connected component."""
    with pytest.raises(NeighborhoodError, match="depth"):
        entity_neighborhood(FakeGraph(), "Project", "p", depth=depth)


def test_deprecated_neighbours_are_excluded_by_default():
    graph = FakeGraph([])
    entity_neighborhood(graph, "Project", "p")

    assert "NOT related:Deprecated" in graph.queries[0][0]


def test_deprecated_neighbours_can_be_included_for_history():
    graph = FakeGraph([])
    entity_neighborhood(graph, "Project", "p", include_deprecated=True)

    assert "NOT related:Deprecated" not in graph.queries[0][0]


def test_the_entity_itself_is_not_returned_as_its_own_neighbour():
    graph = FakeGraph([])
    entity_neighborhood(graph, "Project", "p")

    assert "related <> n" in graph.queries[0][0]
