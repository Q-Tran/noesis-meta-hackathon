"""Unit tests for corpus.py, the read-only query layer behind the web UI.

The graph is faked: canned rows are routed to the query that asks for them by
a unique marker substring, so each test sees exactly the transformation under
test. The Cypher itself is validated against a real database in
test_corpus_live.py.
"""
from __future__ import annotations

import pytest

import corpus


class FakeGraph:
    """Answer `run()` with the canned rows whose marker appears in the query.

    The LONGEST matching marker wins, so a specific query is never shadowed by
    a generic one.
    """

    def __init__(self, rows_by_marker: dict[str, list[dict]]):
        self.rows_by_marker = rows_by_marker
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **params) -> list[dict]:
        self.calls.append((query, params))
        matches = [(len(m), rows) for m, rows in self.rows_by_marker.items() if m in query]
        if not matches:
            return []
        return max(matches, key=lambda t: t[0])[1]


@pytest.fixture(autouse=True)
def _clear_overview_cache():
    corpus._overview_cache.update(at=0.0, data=None)
    corpus._full_cache.clear()
    yield
    corpus._overview_cache.update(at=0.0, data=None)
    corpus._full_cache.clear()


# -- overview -----------------------------------------------------------------


def test_overview_aggregates_totals_per_label_and_relationship():
    fake = FakeGraph({
        "UNWIND labels(n) AS label": [
            {"label": "Claim", "count": 10},
            {"label": "Tool", "count": 3},
        ],
        "type(r) AS type": [{"type": "FOUND_IN", "count": 40}],
    })

    data = corpus.overview(fake)

    assert data["total_nodes"] == 13
    assert data["total_relationships"] == 40
    assert data["labels"] == [
        {"label": "Claim", "count": 10},
        {"label": "Tool", "count": 3},
    ]
    assert data["relationships"] == [{"type": "FOUND_IN", "count": 40}]


def test_overview_is_cached_within_the_ttl():
    fake = FakeGraph({"UNWIND labels(n) AS label": [{"label": "Claim", "count": 1}]})

    corpus.overview(fake)
    corpus.overview(fake)

    label_queries = [q for q, _ in fake.calls if "UNWIND labels(n)" in q]
    assert len(label_queries) == 1


def test_overview_refresh_bypasses_the_cache():
    fake = FakeGraph({"UNWIND labels(n) AS label": [{"label": "Claim", "count": 1}]})

    corpus.overview(fake)
    corpus.overview(fake, refresh=True)

    label_queries = [q for q, _ in fake.calls if "UNWIND labels(n)" in q]
    assert len(label_queries) == 2


# -- growth -------------------------------------------------------------------


def test_growth_returns_days_in_chronological_order():
    fake = FakeGraph({
        "substring(toString(n.created_at), 0, 10)": [
            {"day": "2026-11-03", "count": 5},
            {"day": "2026-11-01", "count": 2},
        ],
    })

    rows = corpus.growth(30, fake)

    assert rows == [
        {"day": "2026-11-01", "count": 2},
        {"day": "2026-11-03", "count": 5},
    ]
    _, params = fake.calls[0]
    assert params["days"] == 30


# -- search -------------------------------------------------------------------


def test_search_short_circuits_on_a_blank_query_with_no_label():
    fake = FakeGraph({})

    assert corpus.search_nodes("   ", graph=fake) == []
    assert fake.calls == []


def test_a_blank_query_with_a_label_browses_that_kind():
    fake = FakeGraph({
        "props, 0 AS score": [
            {"element_id": "e1", "labels": ["Claim"], "props": {"statement": "newest"}, "score": 0},
        ],
    })

    rows = corpus.search_nodes("", label="Claim", graph=fake)

    assert [r["title"] for r in rows] == ["newest"]
    query, params = fake.calls[0]
    assert "$title_keys" not in query  # browse skips the property scan
    assert params == {"label": "Claim", "limit": 25}


def test_search_lowercases_the_query_and_passes_the_label_filter():
    fake = FakeGraph({"$title_keys": []})

    corpus.search_nodes("  Late ORDERS ", label="Claim", limit=10, graph=fake)

    _, params = fake.calls[0]
    assert params["q"] == "late orders"
    assert params["label"] == "Claim"
    assert params["limit"] == 10


def test_search_without_a_label_searches_every_kind():
    fake = FakeGraph({"$title_keys": []})

    corpus.search_nodes("orders", graph=fake)

    _, params = fake.calls[0]
    assert params["label"] is None


def test_search_picks_the_title_from_the_first_available_key():
    fake = FakeGraph({
        "$title_keys": [
            {"element_id": "e1", "labels": ["Claim"], "props": {"statement": "Orders are late", "id": "c1"}, "score": 2},
            {"element_id": "e2", "labels": ["Tool"], "props": {"name": "hybrid_search", "id": "t1"}, "score": 1},
            {"element_id": "e3", "labels": ["Chunk"], "props": {"text": "…orders…", "id": "ch1"}, "score": 0},
        ],
    })

    rows = corpus.search_nodes("orders", graph=fake)

    assert [r["title"] for r in rows] == ["Orders are late", "hybrid_search", "…orders…"]
    assert rows[0]["labels"] == ["Claim"]
    assert rows[0]["score"] == 2


def test_search_falls_back_to_id_then_unnamed_for_a_title():
    fake = FakeGraph({
        "$title_keys": [
            {"element_id": "e1", "labels": ["Span"], "props": {"id": "span-1"}, "score": 0},
            {"element_id": "e2", "labels": ["Span"], "props": {}, "score": 0},
        ],
    })

    rows = corpus.search_nodes("span", graph=fake)

    assert [r["title"] for r in rows] == ["span-1", "(unnamed)"]


def test_search_trims_long_strings_and_records_the_original_length():
    long_text = "x" * 5000
    fake = FakeGraph({
        "$title_keys": [
            {"element_id": "e1", "labels": ["Chunk"], "props": {"text": long_text, "index": 4}, "score": 0},
        ],
    })

    (row,) = corpus.search_nodes("x", graph=fake)

    assert len(row["properties"]["text"]) < len(long_text)
    assert row["properties"]["text"].endswith("…")
    assert row["properties"]["_text_length"] == 5000
    assert row["properties"]["index"] == 4  # non-strings pass through untouched


# -- node detail ----------------------------------------------------------------


def test_node_detail_returns_none_when_the_node_is_gone():
    fake = FakeGraph({})

    assert corpus.node_detail("missing", graph=fake) is None


def test_node_detail_keeps_properties_untrimmed():
    long_text = "y" * 5000
    fake = FakeGraph({
        "MATCH (n) WHERE elementId(n) = $element_id": [
            {"element_id": "e1", "labels": ["Chunk"], "props": {"text": long_text}},
        ],
    })

    node = corpus.node_detail("e1", graph=fake)

    assert node["properties"]["text"] == long_text
    assert "_text_length" not in node["properties"]


# -- neighbors ------------------------------------------------------------------


def test_neighbors_assembles_center_first_then_neighbors_with_direction():
    fake = FakeGraph({
        "MATCH (n) WHERE elementId(n) = $element_id": [
            {"element_id": "c", "labels": ["Claim"], "props": {"statement": "center"}},
        ],
        "MATCH (n)-[r]-(m)": [
            {"element_id": "n1", "labels": ["Chunk"], "props": {"text": "out"}, "rel_type": "FOUND_IN", "rel_id": "r1", "outgoing": True},
            {"element_id": "n2", "labels": ["Argument"], "props": {"surface_text": "in"}, "rel_type": "CONCLUDES", "rel_id": "r2", "outgoing": False},
        ],
    })

    hood = corpus.neighbors("c", limit=25, graph=fake)

    assert hood["nodes"][0]["element_id"] == "c"
    assert hood["nodes"][0]["title"] == "center"
    assert hood["edges"] == [
        {"element_id": "r1", "type": "FOUND_IN", "source": "c", "target": "n1", "outgoing": True},
        {"element_id": "r2", "type": "CONCLUDES", "source": "n2", "target": "c", "outgoing": False},
    ]
    _, params = fake.calls[-1]
    assert params["limit"] == 25


def test_neighbors_returns_none_when_the_center_is_gone():
    fake = FakeGraph({})

    assert corpus.neighbors("missing", graph=fake) is None


# -- subset ---------------------------------------------------------------------


def test_subset_walks_by_connectivity_and_excludes_silos():
    """The walk is the point of the selection: a silo outranks mid and low on
    raw degree, but nothing connects it to the map, so it must not be chosen.
    """
    fake = FakeGraph({
        "UNWIND labels(n) AS label": [{"label": "Component"}, {"label": "Claim"}],
        "COUNT { (n)-[]-() }": [
            {"element_id": "hub", "labels": ["Component"], "title": "The Hub", "degree": 90},
            {"element_id": "silo", "labels": ["Component"], "title": "Silo", "degree": 85},
            {"element_id": "mid", "labels": ["Component"], "title": "Middle", "degree": 40},
            {"element_id": "low", "labels": ["Component"], "title": "Low", "degree": 30},
        ],
        "UNWIND $ids AS id": [
            {"rel_id": "rel1", "source": "hub", "target": "mid", "type": "REALIZES"},
            {"rel_id": "rel1", "source": "mid", "target": "hub", "type": "REALIZES"},
            {"rel_id": "rel2", "source": "mid", "target": "low", "type": "USES"},
            {"rel_id": "rel2", "source": "low", "target": "mid", "type": "USES"},
        ],
    })

    data = corpus.subset(3, fake)

    assert data["ids"] == ["hub", "mid", "low"]
    assert "silo" not in data["ids"]
    assert data["nodes"] == [
        [0, "The Hub", 90],
        [0, "Middle", 40],
        [0, "Low", 30],
    ]
    assert data["labels_dict"] == ["Component"]
    assert data["rel_types"] == ["REALIZES", "USES"]
    assert data["edges"] == [[0, 0, 1], [1, 1, 2]]  # deduped on elementId(r)
    _, params = fake.calls[1]
    assert params["labels"] == ["Component"]  # canonical complement of the fake db
    assert params["limit"] == 15  # the candidate pool is limit×5


def test_subset_excludes_codebase_labels():
    fake = FakeGraph({
        "UNWIND labels(n) AS label": [{"label": "Component"}, {"label": "CodeFile"}, {"label": "Claim"}],
        "COUNT { (n)-[]-() }": [],
        "UNWIND $ids AS id": [],
    })

    corpus.subset(100, fake)

    _, params = fake.calls[1]
    assert params["labels"] == ["Component"]  # CodeFile is out, Claim was never in


def test_subset_is_cached_and_refresh_bypasses():
    fake = FakeGraph({
        "UNWIND labels(n) AS label": [{"label": "Component"}],
        "COUNT { (n)-[]-() }": [],
        "UNWIND $ids AS id": [],
    })

    corpus.subset(100, fake)
    corpus.subset(100, fake)
    corpus.subset(100, fake, refresh=True)

    ranking_queries = [q for q, _ in fake.calls if "COUNT { (n)-[]-() }" in q]
    assert len(ranking_queries) == 2


def test_subset_returns_the_empty_structure_without_canonical_labels():
    fake = FakeGraph({"UNWIND labels(n) AS label": [{"label": "Claim"}]})

    data = corpus.subset(100, fake)

    assert data["nodes"] == [] and data["edges"] == []
    assert len(fake.calls) == 1  # stops after the labels read


# -- full layer -----------------------------------------------------------------


def test_layer_labels_rejects_an_unknown_layer():
    with pytest.raises(ValueError):
        corpus.layer_labels("truth")


def test_canonical_layer_is_the_complement_of_the_other_two():
    fake = FakeGraph({
        "UNWIND labels(n) AS label": [
            {"label": "Claim"}, {"label": "Argument"}, {"label": "Chunk"},
            {"label": "Tool"}, {"label": "Component"}, {"label": "Metric"},
        ],
    })

    assert corpus.layer_labels("canonical", fake) == ["Component", "Metric", "Tool"]
    assert corpus.layer_labels("epistemology") == ["Argument", "Claim"]
    assert "Chunk" in corpus.layer_labels("provenance")


def test_full_layer_returns_the_indexed_wire_form():
    fake = FakeGraph({
        "UNWIND labels(n) AS label": [{"label": "Component"}],
        "MATCH (n) WHERE any(label IN labels(n) WHERE label IN $labels)": [
            # titles are projected in Cypher (200 chars), never full properties
            {"element_id": "c1", "labels": ["Component"], "title": "Orders API"},
            {"element_id": "c2", "labels": ["Component"], "title": None},
        ],
        "MATCH ()-[r]->()": [
            {"type": "REALIZES", "source": "c1", "target": "c2"},
            {"type": "REALIZES", "source": "c1", "target": "stranger"},  # filtered out
        ],
    })

    data = corpus.full_layer(["canonical"], fake)

    # With one label in the db, the "all labels" fast path scanned edges plain;
    # rows are still filtered to known nodes on this side.
    assert data["ids"] == ["c1", "c2"]
    assert data["labels_dict"] == ["Component"]
    assert data["rel_types"] == ["REALIZES"]
    assert data["nodes"] == [[0, "Orders API"], [0, "(unnamed)"]]
    assert data["edges"] == [[0, 0, 1]]
    assert data["layers"] == ["canonical"]


def test_full_layer_is_cached_per_layer_set_and_refresh_bypasses():
    fake = FakeGraph({
        "UNWIND labels(n) AS label": [{"label": "Component"}],
        "MATCH (n) WHERE any(label IN labels(n) WHERE label IN $labels)": [],
    })

    corpus.full_layer(["canonical"], fake)
    corpus.full_layer(["canonical"], fake)
    corpus.full_layer(["canonical"], fake, refresh=True)
    corpus.full_layer(["epistemology"], fake)

    node_queries = [q for q, _ in fake.calls if "title_key" in q]
    assert len(node_queries) == 3  # cold, refresh, different layer set


def test_full_layer_passes_the_union_of_requested_layer_labels():
    fake = FakeGraph({"UNWIND labels(n) AS label": [{"label": "Component"}]})

    corpus.full_layer(["epistemology", "provenance"], fake)

    _, params = fake.calls[-1]
    assert params["labels"] == [
        "Argument", "Chunk", "Claim", "RoutingTrace", "Session", "SourceNode", "Span", "TraceOutcome",
    ]


# -- layer neighborhood -----------------------------------------------------------


def test_layer_neighborhood_bounds_each_canonical_node_and_shapes_edges():
    fake = FakeGraph({
        "collect({rel: r, node: n})": [
            {"cid": "hub", "element_id": "arg-1", "labels": ["Argument"], "props": {"surface_text": "argues"}, "rel_type": "CONCLUDES", "rel_id": "r1", "outgoing": False},
            {"cid": "hub", "element_id": "claim-1", "labels": ["Claim"], "props": {"statement": "premise"}, "rel_type": "USES_PREMISE", "rel_id": "r2", "outgoing": True},
        ],
    })

    data = corpus.layer_neighborhood(["hub", "mid"], "epistemology", 5, fake)

    _, params = fake.calls[0]
    assert params["ids"] == ["hub", "mid"]
    assert params["labels"] == ["Argument", "Claim"]
    assert params["per_node"] == 5
    # centers are not included; the caller already has them
    assert [n["element_id"] for n in data["nodes"]] == ["arg-1", "claim-1"]
    assert data["nodes"][0]["title"] == "argues"
    assert data["edges"] == [
        {"element_id": "r1", "type": "CONCLUDES", "source": "arg-1", "target": "hub", "outgoing": False},
        {"element_id": "r2", "type": "USES_PREMISE", "source": "hub", "target": "claim-1", "outgoing": True},
    ]
    assert data["layer"] == "epistemology"


def test_layer_neighborhood_rejects_the_canonical_layer():
    with pytest.raises(ValueError):
        corpus.layer_neighborhood(["hub"], "canonical", graph=FakeGraph({}))


# -- graph seed -----------------------------------------------------------------


def test_graph_seed_returns_the_hub_neighborhood_with_degree():
    fake = FakeGraph({
        "count(r) AS degree": [
            {"element_id": "hub", "labels": ["Component"], "props": {"name": "hub"}, "degree": 153},
        ],
        "MATCH (n) WHERE elementId(n) = $element_id": [
            {"element_id": "hub", "labels": ["Component"], "props": {"name": "hub"}},
        ],
        "MATCH (n)-[r]-(m)": [
            {"element_id": "n1", "labels": ["Tool"], "props": {"name": "t"}, "rel_type": "USES", "rel_id": "r1", "outgoing": False},
        ],
    })

    seed = corpus.graph_seed(10, fake)

    assert seed["nodes"][0]["element_id"] == "hub"
    assert seed["nodes"][0]["degree"] == 153
    assert seed["edges"][0]["type"] == "USES"


def test_graph_seed_returns_none_when_the_corpus_is_empty():
    fake = FakeGraph({})

    assert corpus.graph_seed(graph=fake) is None


# -- recent activity ------------------------------------------------------------


def test_recent_activity_maps_sessions_and_traces():
    fake = FakeGraph({
        "UNION ALL": [
            {"kind": "Session", "id": "s-1", "started": "2026-11-03T10:00:00", "active": "2026-11-03T10:05:00", "status": None, "error": None},
            {"kind": "RoutingTrace", "id": "t-1", "started": "2026-11-03T09:00:00", "active": "2026-11-03T09:01:00", "status": "done", "error": None},
        ],
    })

    rows = corpus.recent_activity(5, graph=fake)

    assert rows == [
        {"kind": "Session", "id": "s-1", "started_at": "2026-11-03T10:00:00", "last_active_at": "2026-11-03T10:05:00", "status": None, "error": None},
        {"kind": "RoutingTrace", "id": "t-1", "started_at": "2026-11-03T09:00:00", "last_active_at": "2026-11-03T09:01:00", "status": "done", "error": None},
    ]
    _, params = fake.calls[0]
    assert params["limit"] == 5


def test_subset_appends_the_most_connected_tools_with_their_bridges():
    """The tool catalog is a separate corpus component; it joins the map as an
    explicit appendix (top tools + the experts/skills that connect them), so
    the tools read as a related island instead of isolated dots."""
    fake = FakeGraph({
        "UNWIND labels(n) AS label": [{"label": "Component"}, {"label": "Tool"},
                                      {"label": "KnowledgeBaseExpert"}],
        "COUNT { (n)-[]-() }": [
            {"element_id": "hub", "labels": ["Component"], "title": "Hub", "degree": 90},
            {"element_id": "mid", "labels": ["Component"], "title": "Mid", "degree": 40},
        ],
        "MATCH (t:Tool) WHERE t.active = true": [
            {"element_id": "tool-1", "labels": ["Tool"], "title": "Register Tool", "degree": 63},
        ],
        "MATCH (t:Tool)-[r]-(n)": [
            {"element_id": "expert-1", "labels": ["KnowledgeBaseExpert"],
             "title": "Orchestrator", "degree": 20},
        ],
        "UNWIND $ids AS id": [
            {"rel_id": "rel1", "source": "hub", "target": "mid", "type": "REALIZES"},
            {"rel_id": "rel1", "source": "mid", "target": "hub", "type": "REALIZES"},
            {"rel_id": "rel2", "source": "tool-1", "target": "expert-1", "type": "CAN_USE"},
            {"rel_id": "rel2", "source": "expert-1", "target": "tool-1", "type": "CAN_USE"},
        ],
    })

    data = corpus.subset(2, fake)

    assert data["ids"] == ["hub", "mid", "tool-1", "expert-1"]
    assert data["labels_dict"] == ["Component", "KnowledgeBaseExpert", "Tool"]
    assert data["nodes"] == [
        [0, "Hub", 90],
        [0, "Mid", 40],
        [2, "Register Tool", 63],
        [1, "Orchestrator", 20],
    ]
    # rel_types are sorted: CAN_USE=0, REALIZES=1. CAN_USE keeps the first
    # orientation seen (tool→expert); the duplicate reverse row is dropped.
    assert data["edges"] == [[1, 0, 1], [0, 2, 3]]


def test_subset_does_not_rank_the_tool_cluster_into_the_docs_walk():
    """Tools and their experts are canonical in the taxonomy but excluded from
    the subset's pool, so their high degree cannot displace the domain map."""
    fake = FakeGraph({
        "UNWIND labels(n) AS label": [{"label": "Component"}, {"label": "Tool"},
                                      {"label": "KnowledgeBaseExpert"}, {"label": "AgentSkill"}],
        "COUNT { (n)-[]-() }": [],
        "MATCH (t:Tool) WHERE t.active = true": [],
        "UNWIND $ids AS id": [],
    })

    corpus.subset(2, fake)

    _, params = fake.calls[1]
    assert params["labels"] == ["Component"]  # no Tool/Expert/Skill in the walk pool


def test_trace_detail_round_trips_the_turns_spans():
    fake = FakeGraph({
        "MATCH (s:Span {trace_id: $trace_id})": [
            {"span": {"span_id": "s-2", "name": "get_entity", "kind": "tool",
                      "status": "ok", "duration_ms": 41, "trace_id": "chat-1"}},
            {"span": {"span_id": "s-1", "name": "search_corpus", "kind": "tool",
                      "status": "ok", "duration_ms": 90, "trace_id": "chat-1",
                      "internal": "not-on-the-wire"}},
        ],
    })

    data = corpus.trace_detail("chat-1", graph=fake)

    assert data["trace_id"] == "chat-1"
    assert [s["span_id"] for s in data["spans"]] == ["s-2", "s-1"]  # store order kept
    # only the wire keys survive
    assert "internal" not in data["spans"][1]
    assert data["spans"][0]["name"] == "get_entity"


def test_entity_neighborhood_returns_centers_and_a_bounded_depth1_ring():
    fake = FakeGraph({
        "MATCH (n) WHERE n.id IN $ids": [
            {"element_id": "E1", "labels": ["Capability"], "props": {"id": "cap-1", "name": "Tool search"}},
        ],
        "collect({rel: r, node: n})": [
            {"cid": "E1", "element_id": "A1", "labels": ["Argument"], "props": {"surface_text": "argues"},
             "rel_type": "CONCLUDES", "rel_id": "r1", "outgoing": False},
            {"cid": "E1", "element_id": "C1", "labels": ["Claim"], "props": {"statement": "premise"},
             "rel_type": "USES_PREMISE", "rel_id": "r2", "outgoing": False},
        ],
    })

    data = corpus.entity_neighborhood(["cap-1"], 3, fake)

    assert data["centers"] == ["E1"]
    assert [n["element_id"] for n in data["nodes"]] == ["E1", "A1", "C1"]
    assert data["nodes"][0]["title"] == "Tool search"
    assert data["edges"][0] == {"element_id": "r1", "type": "CONCLUDES",
                                "source": "A1", "target": "E1", "outgoing": False}
    _, params = fake.calls[1]
    assert params["eids"] == ["E1"]
    assert params["per_node"] == 3


def test_entity_neighborhood_rejects_empty_ids():
    try:
        corpus.entity_neighborhood([], 3, FakeGraph({}))
        raise AssertionError("empty ids must raise")
    except ValueError:
        pass
