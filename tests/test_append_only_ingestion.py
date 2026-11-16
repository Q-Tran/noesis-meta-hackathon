"""Append-only ingestion: a newer fact relates to the older one, never replaces it.

This is what removes the need for a changeset on ordinary ingestion. If the graph
keeps every version and links them, it is already its own history, so there is
nothing for a changeset to version. Two things then have to be true, and both are
tested here: superseding must leave the old node untouched, and reads must be
able to ask for only the current version.
"""
from __future__ import annotations

import pytest

from knowledge_bases.ingestion import (
    CONFLICTS_WITH,
    SUPERSEDES,
    create_entity,
    current_filter,
    new_id,
    relate,
)


class FakeGraph:
    def __init__(self):
        self.nodes = []
        self.rels = []
        self.sources = []

    def upsert_node(self, label, key, props=None):
        self.nodes.append((label, dict(key), dict(props or {})))
        return [{}]

    def upsert_rel(self, sl, sk, rel, ol, ok, props=None):
        self.rels.append((sl, dict(sk), rel, ol, dict(ok), props))
        return [{}]

    def attach_source(self, label, key, source_id, source_props=None, role=None):
        self.sources.append((label, dict(key), source_id, role))
        return [{}]


def test_superseding_writes_a_new_node_and_leaves_the_old_one_alone():
    """The core promise: the old fact keeps its properties and its edges. Only a
    new incoming SUPERSEDES appears."""
    graph = FakeGraph()

    create_entity(graph, "Metric", {"name": "Revenue v2"},
                  {"definition": "net of refunds"},
                  supersedes={"name": "Revenue v1"})

    written = [n[1] for n in graph.nodes]
    assert {"name": "Revenue v2"} in written
    assert {"name": "Revenue v1"} not in written, "the superseded node was rewritten"

    assert graph.rels == [
        ("Metric", {"name": "Revenue v2"}, SUPERSEDES, "Metric", {"name": "Revenue v1"}, None)
    ]


def test_supersedes_points_from_new_to_old():
    """Direction is load-bearing: `current_filter` defines 'current' as
    'nothing supersedes me', so a reversed edge would hide the newest fact."""
    graph = FakeGraph()

    create_entity(graph, "Metric", {"name": "new"}, supersedes={"name": "old"})

    _, start_key, rel, _, end_key, _ = graph.rels[0]
    assert rel == SUPERSEDES
    assert start_key == {"name": "new"}
    assert end_key == {"name": "old"}


def test_conflicting_facts_coexist_and_are_marked_active():
    """When two facts disagree and neither wins, both stay and the disagreement
    itself is recorded -- that is the alternative to overwriting."""
    graph = FakeGraph()

    create_entity(graph, "Metric", {"name": "churn (finance)"},
                  conflicts_with={"name": "churn (product)"})

    _, _, rel, _, _, props = graph.rels[0]
    assert rel == CONFLICTS_WITH
    assert props == {"status": "active"}


def test_current_filter_excludes_superseded_and_deprecated_nodes():
    """Append-only storage means a read that forgets this returns every
    historical version alongside the current one."""
    clause = current_filter("m")

    assert f"NOT (m)<-[:{SUPERSEDES}]-()" in clause
    assert "NOT m:Deprecated" in clause


def test_a_plain_create_asserts_nothing_about_authority():
    """Passing neither supersedes nor conflicts_with means 'this fact is new',
    not 'this fact wins' -- so no relationship is invented."""
    graph = FakeGraph()

    create_entity(graph, "Metric", {"name": "Revenue"})

    assert graph.rels == []


def test_relate_refuses_nothing_the_ontology_allows_but_records_direction():
    graph = FakeGraph()

    result = relate(graph, "Metric", {"name": "Revenue"}, "USES",
                    "Table", {"name": "fct_orders"})

    assert result["relationship"] == "USES"
    assert result["from"]["label"] == "Metric"
    assert result["to"]["label"] == "Table"


def test_ids_are_readable_and_unique():
    first, second = new_id("metric"), new_id("metric")

    assert first.startswith("metric-") and second.startswith("metric-")
    assert first != second
