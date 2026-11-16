"""Live smoke tests for corpus.py's Cypher.

The unit tests (test_corpus.py) fake the graph, so they cannot catch a query
that does not parse, a property the rows no longer carry, or a LIMIT parameter
Neo4j rejects. These run the real thing, read-only, and skip without a graph
so a laptop without Docker still runs the suite.
"""
from __future__ import annotations

import pytest

import corpus


@pytest.fixture(scope="module")
def graph():
    from graph import Graph

    try:
        g = Graph()
        g.run("RETURN 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"graph unavailable: {exc}")
    yield g
    g.close()


@pytest.fixture(autouse=True)
def _clear_overview_cache():
    corpus._overview_cache.update(at=0.0, data=None)
    yield
    corpus._overview_cache.update(at=0.0, data=None)


def test_overview_counts_the_real_corpus(graph):
    data = corpus.overview(graph)

    assert data["total_nodes"] > 0
    assert data["total_relationships"] > 0
    assert data["labels"], "an empty label list means the UNWIND shape changed"
    assert all(row["count"] > 0 for row in data["labels"])


def test_search_round_trips_a_node_we_can_open(graph):
    (hit,) = corpus.search_nodes("hybrid_search", label="Tool", limit=1, graph=graph)

    assert hit["labels"] == ["Tool"]
    assert hit["title"]

    detail = corpus.node_detail(hit["element_id"], graph)
    assert detail is not None
    assert detail["element_id"] == hit["element_id"]


def test_search_with_a_label_that_exists_but_is_unmatched_returns_empty(graph):
    assert corpus.search_nodes("zzz-no-such-thing-zzz", label="Tool", graph=graph) == []


def test_neighbors_of_a_tool_have_the_expected_shape(graph):
    (hit,) = corpus.search_nodes("hybrid_search", label="Tool", limit=1, graph=graph)

    hood = corpus.neighbors(hit["element_id"], limit=10, graph=graph)

    assert hood is not None
    assert hood["nodes"][0]["element_id"] == hit["element_id"]
    for edge in hood["edges"]:
        assert edge["type"]
        # Direction is consistent: every edge touches the center node.
        assert hit["element_id"] in (edge["source"], edge["target"])


def test_growth_and_activity_parse_and_return_rows(graph):
    days = corpus.growth(7, graph)
    assert all(set(row) == {"day", "count"} for row in days)

    activity = corpus.recent_activity(3, graph)
    assert all(row["kind"] in {"Session", "RoutingTrace"} for row in activity)


def test_graph_seed_against_the_real_corpus(graph):
    seed = corpus.graph_seed(10, graph)

    assert seed is not None
    hub = seed["nodes"][0]
    assert hub["degree"] > 0
    assert seed["edges"], "the most-connected node having no edges means the degree query drifted"
    for edge in seed["edges"]:
        assert hub["element_id"] in (edge["source"], edge["target"])


def test_subset_against_the_real_corpus(graph):
    data = corpus.subset(100, graph)

    assert len(data["nodes"]) >= 100
    # The walk starts at the top hub and grows by adjacency; the tool catalog
    # is appended explicitly, so the map may exceed `limit` by the tools and
    # their expert/skill bridges.
    assert len(data["nodes"]) <= 100 + corpus.TOOL_SUBSET_LIMIT + 20
    degrees = [row[2] for row in data["nodes"]]
    assert degrees[0] == max(degrees)
    assert data["edges"], "the subset is one related map"
    n = len(data["ids"])
    assert all(0 <= src < n and 0 <= tgt < n for _, src, tgt in data["edges"])
    # Canonical only, and no codebase nodes in the selected map.
    leaked = set(data["labels_dict"]) & (
        {"Claim", "Argument"} | set(corpus.LAYERS["provenance"]) | set(corpus.CODEBASE_LABELS)
    )
    assert not leaked
    # The tool catalog is represented by the most-connected active tools.
    tool_count = sum(1 for row in data["nodes"] if data["labels_dict"][row[0]] == "Tool")
    assert 0 < tool_count <= corpus.TOOL_SUBSET_LIMIT
    # No silos: every node is related to something (the tool island is a
    # second connected component by nature -- the product's callable surface).
    degree = {}
    for _, src, tgt in data["edges"]:
        degree[src] = degree.get(src, 0) + 1
        degree[tgt] = degree.get(tgt, 0) + 1
    isolated = [data["nodes"][i][1] for i in range(n) if degree.get(i, 0) == 0]
    assert not isolated, f"isolated nodes in the selected map: {isolated[:5]}"


def test_layer_neighborhood_against_the_real_corpus(graph):
    subset_ids = corpus.subset(100, graph)["ids"]

    data = corpus.layer_neighborhood(subset_ids, "epistemology", 5, graph)

    assert data["nodes"], "the top hubs are justified"
    # The bound holds per canonical node, not just in total.
    per_canonical: dict[str, int] = {}
    for row in data["edges"]:
        # The canonical end of the edge is whichever end is a subset id.
        canonical_end = row["source"] if row["source"] in subset_ids else row["target"]
        per_canonical[canonical_end] = per_canonical.get(canonical_end, 0) + 1
    assert all(count <= 5 for count in per_canonical.values())
    # Only epistemology labels came back.
    leaked = {l for n in data["nodes"] for l in n["labels"]} - {"Claim", "Argument"}
    assert not leaked


def test_full_canonical_layer_against_the_real_corpus(graph):
    data = corpus.full_layer(["canonical"], graph)

    assert len(data["nodes"]) > 5000, "the canonical layer should be most of the corpus"
    assert data["edges"], "canonical entities are connected"
    assert len(set(data["ids"])) == len(data["nodes"]), "one id per node"
    # Ground and justification never leak into canonical.
    leaked = set(data["labels_dict"]) & (
        {"Claim", "Argument"} | set(corpus.LAYERS["provenance"])
    )
    assert not leaked
    # Every edge points at real nodes.
    n = len(data["ids"])
    assert all(0 <= src < n and 0 <= tgt < n for _, src, tgt in data["edges"])
