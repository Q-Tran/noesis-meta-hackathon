"""Unit tests for the shared lifecycle helpers (`Graph.mark_deprecated` /
`Graph.reactivate`), exercised without a live Memgraph connection by calling
the unbound methods against a small recorder (mirrors the RecordingGraph
pattern used in `tests/test_provenance.py`)."""
from graph import Graph


class RecordingGraph:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))
        return [{"n": {"id": "d-1"}}]


def test_mark_deprecated_appends_set_label_to_caller_supplied_match():
    graph = RecordingGraph()

    Graph.mark_deprecated(
        graph, "MATCH (d:Decision {id: $id})", {"id": "d-1"},
    )

    assert len(graph.calls) == 1
    query, params = graph.calls[0]
    assert query == "MATCH (d:Decision {id: $id}) SET n:Deprecated RETURN n"
    assert params == {"id": "d-1"}


def test_mark_deprecated_supports_multi_hop_bulk_match_clauses():
    graph = RecordingGraph()

    Graph.mark_deprecated(
        graph,
        "MATCH (p:Project {id: $id})-[:HAS_EPIC|HAS_STORY|HAS_TASK*0..]->(n)",
        {"id": "project-1"},
    )

    query, params = graph.calls[0]
    assert query.startswith(
        "MATCH (p:Project {id: $id})-[:HAS_EPIC|HAS_STORY|HAS_TASK*0..]->(n) "
        "SET n:Deprecated"
    )
    assert params == {"id": "project-1"}


def test_mark_deprecated_links_superseded_by_when_replacement_given():
    graph = RecordingGraph()

    Graph.mark_deprecated(
        graph, "MATCH (m:Metric {id: $id})", {"id": "metric-old"},
        superseded_by_key={"id": "metric-new"}, superseded_by_label="Metric",
    )

    query, params = graph.calls[0]
    assert "SET n:Deprecated" in query
    assert "MATCH (s:`Metric` {`id`: $sup_id})" in query
    assert "MERGE (n)-[sb:SUPERSEDED_BY]->(s)" in query
    assert "MERGE (s)-[sv:SUPERSEDES]->(n)" in query
    # These two edges are built in raw Cypher rather than through
    # `graph.upsert_rel`, so the boundary's `created_at` stamp has to be
    # spelled out here -- and pinned here, or it silently leaks.
    assert "ON CREATE SET sb.created_at = datetime()" in query
    assert "ON CREATE SET sv.created_at = datetime()" in query
    assert params == {"id": "metric-old", "sup_id": "metric-new"}


def test_mark_deprecated_without_replacement_omits_superseded_by():
    graph = RecordingGraph()

    Graph.mark_deprecated(graph, "MATCH (d:Decision {id: $id})", {"id": "d-1"})

    query, _ = graph.calls[0]
    assert "SUPERSEDED_BY" not in query
    assert "SUPERSEDES" not in query


def test_reactivate_removes_deprecated_label():
    graph = RecordingGraph()

    Graph.reactivate(graph, "MATCH (d:Decision {id: $id})", {"id": "d-1"})

    query, params = graph.calls[0]
    assert query == "MATCH (d:Decision {id: $id}) REMOVE n:Deprecated RETURN n"
    assert params == {"id": "d-1"}


def test_mark_deprecated_defaults_params_to_empty_dict():
    graph = RecordingGraph()

    Graph.mark_deprecated(graph, "MATCH (n:Table {id: 'fixed'})")

    query, params = graph.calls[0]
    assert params == {}
