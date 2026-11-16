"""Ingestion writes what fits the ontology and explains what it refused.

These tests use a fake graph: the point under test is the ingestion policy
(convergence on identity, endpoint creation before edges, partial success), not
Neo4j's behaviour.
"""
from __future__ import annotations

import pytest

from knowledge_bases.base import EntityType, RelationshipType
from knowledge_bases.ingestion_service import ingest_triplets
from knowledge_bases.triplets import Triplet


class FakeGraph:
    def __init__(self):
        self.nodes = []
        self.rels = []
        self.sources = []

    def upsert_node(self, label, key, props=None):
        self.nodes.append((label, key, props or {}))
        return [{}]

    def upsert_rel(self, sl, sk, rel, ol, ok, props=None):
        self.rels.append((sl, sk, rel, ol, ok, props))
        return [{}]

    def attach_source(self, label, key, source_id, source_props=None, role=None):
        self.sources.append((label, key, source_id, role))
        return [{}]


@pytest.fixture(autouse=True)
def described(monkeypatch):
    entities = {
        "Metric": EntityType(label="Metric", description="A computed business quantity.",
                             identity="name", properties=("name", "definition")),
        "Table": EntityType(label="Table", description="A physical warehouse table.",
                            identity="name", properties=("name",)),
        "BusinessTerm": EntityType(label="BusinessTerm",
                                   description="A canonical business word or phrase.",
                                   identity="name"),
    }
    rels = {
        "USES": RelationshipType(relationship="USES",
                                 description="A metric reads from a table.",
                                 subject=("Metric",), object=("Table",)),
    }
    for module in ("knowledge_bases.triplets", "knowledge_bases.ingestion_service"):
        try:
            monkeypatch.setattr(f"{module}.entity_types", lambda: entities)
        except AttributeError:
            pass
    monkeypatch.setattr("knowledge_bases.triplets.relationship_types", lambda: rels)
    return entities


def _t(**kw):
    d = dict(subject_label="Metric", subject_identity="Revenue",
             relationship="USES", object_label="Table", object_identity="fct_orders")
    d.update(kw)
    return Triplet(**d)


def test_a_fitting_triplet_writes_both_endpoints_then_the_edge():
    """`upsert_rel` MATCHes existing nodes, so an edge written before its
    endpoints would silently write nothing."""
    graph = FakeGraph()

    result = ingest_triplets(graph, [_t()])

    assert result["entities_written"] == 2
    assert result["relationships_written"] == 1
    assert [n[0] for n in graph.nodes] == ["Metric", "Table"]
    assert graph.rels[0][2] == "USES"


def test_entities_converge_on_their_declared_identity_property():
    """Identity is what stops re-ingesting the same fact from creating a node
    per phrasing. Metric declares `name`, not `id`."""
    graph = FakeGraph()

    ingest_triplets(graph, [_t()])

    label, key, _ = graph.nodes[0]
    assert key == {"name": "Revenue"}


def test_a_repeated_endpoint_is_written_once_per_call():
    graph = FakeGraph()

    ingest_triplets(graph, [_t(), _t(object_identity="fct_returns")])

    metrics = [n for n in graph.nodes if n[0] == "Metric"]
    assert len(metrics) == 1


def test_one_bad_triplet_does_not_discard_the_good_ones():
    """Partial success is the normal outcome: a document with one unrecognised
    sentence must still contribute everything else."""
    graph = FakeGraph()

    result = ingest_triplets(graph, [_t(), _t(relationship="VIBES_WITH")])

    assert result["relationships_written"] == 1
    assert len(result["rejected"]) == 1
    assert "VIBES_WITH" in result["rejected"][0]["reason"]


def test_a_rejection_names_the_triplet_it_refused():
    """The reason is the signal that an ontology type is missing, so it has to
    identify the shape, not just fail."""
    graph = FakeGraph()

    result = ingest_triplets(graph, [_t(subject_label="Table", object_label="Table")])

    assert result["relationships_written"] == 0
    assert "does not accept subject" in result["rejected"][0]["reason"]
    assert "Table" in result["rejected"][0]["triplet"]


def test_undeclared_properties_never_reach_the_graph():
    graph = FakeGraph()

    ingest_triplets(graph, [_t(subject_props={"name": "Revenue", "vibe": "great"})])

    _, _, props = graph.nodes[0]
    assert "vibe" not in props
    assert props["name"] == "Revenue"


def test_ingestion_stamps_provenance_when_a_source_is_given():
    """A fact that cannot be traced to where it came from is not auditable."""
    graph = FakeGraph()

    ingest_triplets(graph, [_t()], source_id="doc-1")

    assert graph.sources
    assert graph.sources[0][2] == "doc-1"


def test_every_written_node_records_who_ingested_it_and_when():
    graph = FakeGraph()

    ingest_triplets(graph, [_t()], actor="orchestrator")

    _, _, props = graph.nodes[0]
    assert props["ingested_by"] == "orchestrator"
    assert "ingested_at" in props

def test_a_composite_identity_uses_every_declared_property(monkeypatch):
    """`ReviewFinding` is matched on (project_id, key) everywhere in the code, so
    converging on `project_id` alone would merge every finding in a project into
    one node."""
    entities = {
        "ReviewFinding": EntityType(
            label="ReviewFinding", description="A finding raised during review.",
            identity=("project_id", "key"), properties=("project_id", "key"),
        ),
        "Table": EntityType(label="Table", description="A physical warehouse table.",
                            identity="name", properties=("name",)),
    }
    rels = {"SUPPORTS": RelationshipType(relationship="SUPPORTS",
                                      description="A finding supports a thing.")}
    monkeypatch.setattr("knowledge_bases.triplets.entity_types", lambda: entities)
    monkeypatch.setattr("knowledge_bases.ingestion_service.entity_types", lambda: entities)
    monkeypatch.setattr("knowledge_bases.triplets.relationship_types", lambda: rels)
    graph = FakeGraph()

    result = ingest_triplets(graph, [Triplet(
        subject_label="ReviewFinding", subject_identity="proj-1",
        subject_props={"project_id": "proj-1", "key": "finding-7"},
        relationship="SUPPORTS",
        object_label="Table", object_identity="fct_orders",
    )])

    assert result["relationships_written"] == 1
    finding_key = next(n[1] for n in graph.nodes if n[0] == "ReviewFinding")
    assert finding_key == {"project_id": "proj-1", "key": "finding-7"}


def test_a_composite_identity_missing_a_part_is_reported_not_guessed(monkeypatch):
    """Defaulting the missing half would silently attach the fact to the wrong
    node, which is worse than refusing it."""
    entities = {
        "ReviewFinding": EntityType(
            label="ReviewFinding", description="A finding raised during review.",
            identity=("project_id", "key"), properties=("project_id", "key"),
        ),
        "Table": EntityType(label="Table", description="A physical warehouse table.",
                            identity="name"),
    }
    rels = {"SUPPORTS": RelationshipType(relationship="SUPPORTS",
                                      description="A finding supports a thing.")}
    monkeypatch.setattr("knowledge_bases.triplets.entity_types", lambda: entities)
    monkeypatch.setattr("knowledge_bases.ingestion_service.entity_types", lambda: entities)
    monkeypatch.setattr("knowledge_bases.triplets.relationship_types", lambda: rels)
    graph = FakeGraph()

    result = ingest_triplets(graph, [Triplet(
        subject_label="ReviewFinding", subject_identity="proj-1",
        relationship="SUPPORTS",
        object_label="Table", object_identity="fct_orders",
    )])

    assert result["relationships_written"] == 0
    assert "composite" in result["failed"][0]["reason"]
