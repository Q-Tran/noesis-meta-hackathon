"""Tests for the web UI surface (app.py) and its single-page app artifact.

The graph dependency is overridden with a fake, so these run without Neo4j;
the SPA itself is checked for the properties the design promises (self-
contained, no external resources) rather than by driving a browser.
"""
from __future__ import annotations

import pathlib
import re

import pytest
from fastapi.testclient import TestClient

import app as web_app
import corpus


@pytest.fixture(autouse=True)
def _clear_overview_cache():
    corpus._overview_cache.update(at=0.0, data=None)
    corpus._full_cache.clear()
    yield
    corpus._overview_cache.update(at=0.0, data=None)
    corpus._full_cache.clear()


class FakeGraph:
    """Answer `run()` with the canned rows whose marker appears in the query.

    The LONGEST matching marker wins, so a specific query (e.g. the full-layer
    edge query) is never shadowed by a generic marker (e.g. "type(r) AS type").
    """

    def __init__(self, rows_by_marker: dict[str, list[dict]] | None = None, error: Exception | None = None):
        self.rows_by_marker = rows_by_marker or {}
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **params) -> list[dict]:
        if self.error is not None:
            raise self.error
        self.calls.append((query, params))
        matches = [(len(m), rows) for m, rows in self.rows_by_marker.items() if m in query]
        if not matches:
            return []
        return max(matches, key=lambda t: t[0])[1]


@pytest.fixture
def client():
    # One fake per test, shared across requests so assertions can inspect the
    # recorded Cypher calls.
    fake = FakeGraph({
        "UNWIND labels(n) AS label": [{"label": "Claim", "count": 7}],
        "type(r) AS type": [{"type": "FOUND_IN", "count": 9}],
        "UNION ALL": [{"kind": "Session", "id": "s-1", "started": None, "active": "2026-11-03T10:00:00", "status": None, "error": None}],
        "count(r) AS degree": [{"element_id": "hub", "labels": ["Component"], "props": {"name": "hub"}, "degree": 5}],
        "MATCH (n) WHERE elementId(n) = $element_id": [{"element_id": "hub", "labels": ["Component"], "props": {"name": "hub"}}],
    })
    web_app.app.dependency_overrides[web_app.get_graph] = lambda: fake
    with TestClient(web_app.app) as test_client:
        test_client.fake = fake
        yield test_client
    web_app.app.dependency_overrides.clear()


@pytest.fixture
def empty_client():
    web_app.app.dependency_overrides[web_app.get_graph] = lambda: FakeGraph()
    with TestClient(web_app.app) as test_client:
        yield test_client
    web_app.app.dependency_overrides.clear()


# -- the page -------------------------------------------------------------------


def test_the_landing_page_is_self_contained_and_judge_ready(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    html = response.text
    # The pitch, the two-part argument, the labeled demo, the impact stats.
    assert "The truth isn't enough." in html
    assert "canonical truth" in html
    assert "argumentative framework" in html
    assert "dev.meta.ai" in html
    assert 'id="anim-canvas"' in html
    assert 'id="anim-prev"' in html and 'id="anim-next"' in html and 'id="anim-play"' in html
    assert "$3.1" in html and "$12.9" in html and "$600" in html
    # Step six hides the layers beneath the canonical truth.
    assert "layersOff: true" in html
    # Self-contained: no external resources slowing first paint.
    external = re.findall(r'(?:src|href)\s*=\s*["\']https?://', html)
    assert external == [], f"external resources would slow first paint: {external}"
    # The navbar is exactly brand + the three tabs.
    assert '<a href="/app">Graph</a>' in html
    assert '<a href="/app#view=dashboard">Dashboard</a>' in html
    assert '<a href="/app#view=ask">Ask</a>' in html


def test_the_page_serves_the_single_page_app(client):
    response = client.get("/app")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert "<title>Noesis · Corpus Explorer</title>" in response.text
    assert 'id="search"' in response.text
    assert 'id="main"' in response.text


def test_health_reports_process_and_graph_state(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "graph": "up"}


def test_the_single_page_app_is_self_contained():
    """Snappy means no CDN round-trips: every resource must be inline."""
    html = pathlib.Path(web_app.WEB_DIR, "index.html").read_text(encoding="utf-8")

    external = re.findall(r'(?:src|href)\s*=\s*["\']https?://', html)
    assert external == [], f"external resources would slow first paint: {external}"
    assert "<script>" in html and "<style>" in html


# -- the JSON API -----------------------------------------------------------------


def test_overview_returns_the_corpus_totals(client):
    response = client.get("/api/overview")

    assert response.status_code == 200
    data = response.json()
    assert data["total_nodes"] == 7
    assert data["total_relationships"] == 9
    assert data["labels"] == [{"label": "Claim", "count": 7}]


def test_search_passes_the_query_and_label_through(client):
    response = client.get("/api/search", params={"q": "orders", "label": "Claim", "limit": 5})

    assert response.status_code == 200
    assert response.json() == []
    _, params = client.fake.calls[-1]
    assert params["q"] == "orders"
    assert params["label"] == "Claim"
    assert params["limit"] == 5


def test_a_missing_node_is_a_404(empty_client):
    assert empty_client.get("/api/node/4:abc:0").status_code == 404
    assert empty_client.get("/api/node/4:abc:0/neighbors").status_code == 404


def test_limit_bounds_are_validated(client):
    assert client.get("/api/search", params={"limit": 0}).status_code == 422
    assert client.get("/api/node/x/neighbors", params={"limit": 1000}).status_code == 422


def test_activity_returns_the_recent_items(client):
    response = client.get("/api/activity", params={"limit": 3})

    assert response.status_code == 200
    assert response.json() == [
        {"kind": "Session", "id": "s-1", "started_at": None,
         "last_active_at": "2026-11-03T10:00:00", "status": None, "error": None}
    ]


def test_graph_seed_serves_the_explorer_starting_point(client):
    response = client.get("/api/graph/seed")

    assert response.status_code == 200


def test_graph_seed_is_a_404_when_the_corpus_is_empty(empty_client):
    assert empty_client.get("/api/graph/seed").status_code == 404


def test_full_layer_defaults_to_canonical(client):
    # Give the fake one epistemology label and one canonical label; only the
    # canonical one may come back from a layerless request.
    client.fake.rows_by_marker["UNWIND labels(n) AS label"] = [
        {"label": "Claim", "count": 7},
        {"label": "Component", "count": 3},
    ]
    client.fake.rows_by_marker["MATCH (n) WHERE any(label IN labels(n) WHERE label IN $labels)"] = [
        {"element_id": "c1", "labels": ["Component"], "title": "Orders API"},
    ]
    # This fake takes the cypher-filtered edge path (a Claim is in the db but
    # outside the canonical layer), which contains the overview's rel-type
    # marker as a substring; override it so the edge query gets no rows.
    client.fake.rows_by_marker["type(r) AS type"] = []

    response = client.get("/api/graph/full")

    assert response.status_code == 200
    body = response.json()
    assert body["ids"] == ["c1"]
    assert body["labels_dict"] == ["Component"]
    assert body["nodes"] == [[0, "Orders API"]]
    assert body["edges"] == []
    assert body["layers"] == ["canonical"]


def test_full_layer_validates_layer_names(client):
    assert client.get("/api/graph/full", params={"layers": "canonical,truth"}).status_code == 422
    assert client.get("/api/graph/full", params={"layers": ""}).status_code == 422


def test_subset_serves_the_home_map(client):
    client.fake.rows_by_marker["UNWIND labels(n) AS label"] = [
        {"label": "Claim", "count": 7},
        {"label": "Component", "count": 3},
    ]
    client.fake.rows_by_marker["COUNT { (n)-[]-() }"] = [
        {"element_id": "hub", "labels": ["Component"], "title": "The Hub", "degree": 90},
    ]
    client.fake.rows_by_marker["UNWIND $ids AS id"] = []

    response = client.get("/api/graph/subset")

    assert response.status_code == 200
    body = response.json()
    assert body["ids"] == ["hub"]
    assert body["nodes"] == [[0, "The Hub", 90]]
    assert body["edges"] == []


def test_subset_validates_the_limit(client):
    assert client.get("/api/graph/subset", params={"limit": 0}).status_code == 422
    assert client.get("/api/graph/subset", params={"limit": 1000}).status_code == 422


def test_the_showcase_subgraph_route_assembles_the_story_core(client):
    row = {k: None for k in (
        "cap_labels", "cap_props", "arg_labels", "arg_props", "ce_eid", "ce_type",
        "up_eid", "up_type", "c_labels", "c_props", "fi_eid", "fi_type",
        "ev_labels", "ev_props", "fs_eid", "fs_type", "src_labels", "src_props",
        "att_eid", "att_labels", "att_props", "at_eid", "at_type", "up2_eid", "up2_type",
        "c2_eid", "c2_labels", "c2_props", "fi2_eid", "fi2_type",
        "t2_eid", "t2_labels", "t2_props",
        "new_eid", "new_labels", "new_props", "ce2_eid", "ce2_type",
        "up3_eid", "up3_type", "c3_eid", "c3_labels", "c3_props",
        "fi3_eid", "fi3_type", "ev3_eid", "ev3_labels", "ev3_props",
    )}
    row.update({
        "cap_eid": "cap-1", "cap_labels": ["Capability"],
        "cap_props": {"name": "Tool search", "id": "extent-591dfe996e"},
        "arg_eid": "arg-1", "arg_labels": ["Argument"],
        "arg_props": {"id": "extarg-8dd04b0c8e-fc846d6c", "status": "in"},
        "ce_eid": "r-ce", "ce_type": "CONCLUDES",
        "up_eid": "r-up", "up_type": "USES_PREMISE",
        "c_eid": "claim-1", "c_labels": ["Claim"],
        "c_props": {"statement": "Tool search is not available on Chat Completions", "status": "active"},
        "fi_eid": "r-fi", "fi_type": "FOUND_IN",
        "ev_eid": "chunk-1", "ev_labels": ["Chunk"], "ev_props": {"text": "# Tool search"},
        "fs_eid": "r-fs", "fs_type": "FOUND_IN",
        "src_eid": "src-1", "src_labels": ["SourceNode"], "src_props": {"name": "Tool search"},
    })
    client.fake.rows_by_marker["MATCH (cap:Capability {id: $cap_id})"] = [row]

    response = client.get("/api/showcase/subgraph")

    assert response.status_code == 200
    body = response.json()
    assert body["capability_id"] == "extent-591dfe996e"
    assert body["stale_argument_id"] == "extarg-8dd04b0c8e-fc846d6c"
    titles = {n["title"] for n in body["nodes"]}
    assert "Tool search" in titles
    assert "Tool search is not available on Chat Completions" in titles
    assert {n["labels"][0] for n in body["nodes"]} == {
        "Capability", "Argument", "Claim", "Chunk", "SourceNode"
    }
    types = {(e["type"], e["source"], e["target"]) for e in body["edges"]}
    assert ("CONCLUDES", "arg-1", "cap-1") in types
    assert ("FOUND_IN", "claim-1", "chunk-1") in types


def test_layer_neighbors_round_trips_the_bounded_expansion(client):
    client.fake.rows_by_marker["collect({rel: r, node: n})"] = [
        {"cid": "hub", "element_id": "claim-1", "labels": ["Claim"], "props": {"statement": "premise"}, "rel_type": "USES_PREMISE", "rel_id": "r2", "outgoing": True},
    ]

    response = client.post("/api/graph/layer_neighbors", json={
        "ids": ["hub"], "layer": "epistemology", "per_node": 5,
    })

    assert response.status_code == 200
    body = response.json()
    assert [n["element_id"] for n in body["nodes"]] == ["claim-1"]
    assert body["edges"][0]["source"] == "hub"


def test_layer_neighbors_validates_the_request(client):
    assert client.post("/api/graph/layer_neighbors", json={"ids": ["hub"], "layer": "canonical"}).status_code == 422
    assert client.post("/api/graph/layer_neighbors", json={"ids": [], "layer": "epistemology"}).status_code == 422
    assert client.post("/api/graph/layer_neighbors", json={"ids": ["hub"] * 501, "layer": "epistemology"}).status_code == 422
    assert client.post("/api/graph/layer_neighbors", json={"ids": ["hub"], "layer": "epistemology", "per_node": 0}).status_code == 422


def test_a_dead_graph_is_a_503_the_ui_can_render():
    web_app.app.dependency_overrides[web_app.get_graph] = lambda: FakeGraph(
        error=RuntimeError("connection refused")
    )
    with TestClient(web_app.app) as test_client:
        response = test_client.get("/api/overview")
    web_app.app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "connection refused" in response.json()["detail"]


# -- ask --------------------------------------------------------------------------


def test_ask_delegates_to_the_demo_agent_loop(client, monkeypatch):
    """The explorer's Ask tab is the proven demo loop (bounded tools, Meta tool
    search, turn spans, per-turn delta) served at /api/ask -- not a second agent."""
    seen = {}

    def fake_chat(payload):
        seen.update(payload)
        return {
            "answer": "answer to: what orders are late?",
            "tools": [{"name": "search_corpus"}],
            "writes": [],
            "trace_id": "chat-abc",
            "session_id": payload["session_id"],
            "tool_search": [{"searched": {"query": "orders"}, "loaded": ["search_corpus"]}],
            "delta": {"nodes": [], "edges": [], "explanation": []},
        }

    monkeypatch.setattr(web_app.demo_server, "chat", fake_chat)

    response = client.post("/api/ask", json={"question": "what orders are late?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "answer to: what orders are late?"
    assert body["trace_id"] == "chat-abc"
    assert body["tool_search"][0]["loaded"] == ["search_corpus"]
    assert seen["messages"] == [{"role": "user", "content": "what orders are late?"}]
    assert seen["session_id"].startswith("web-")


def test_ask_rejects_a_blank_question(client):
    assert client.post("/api/ask", json={"question": "   "}).status_code == 422


def test_ask_maps_an_agent_failure_to_a_502(client, monkeypatch):
    def boom(payload):
        raise RuntimeError("no LLM configured")

    monkeypatch.setattr(web_app.demo_server, "chat", boom)

    response = client.post("/api/ask", json={"question": "anything"})

    assert response.status_code == 502


def test_ask_maps_a_bad_request_to_a_422(client, monkeypatch):
    def bad(payload):
        raise ValueError("messages must be a non-empty list")

    monkeypatch.setattr(web_app.demo_server, "chat", bad)

    response = client.post("/api/ask", json={"question": "anything"})

    assert response.status_code == 422


def test_trace_round_trips_the_recorded_spans(client, monkeypatch):
    monkeypatch.setattr(
        web_app.corpus, "trace_detail",
        lambda trace_id, graph=None: {
            "trace_id": trace_id,
            "spans": [{"span_id": "s-1", "name": "search_corpus", "status": "ok"}],
        },
    )

    response = client.get("/api/trace/chat-abc")

    assert response.status_code == 200
    assert response.json() == {
        "trace_id": "chat-abc",
        "spans": [{"span_id": "s-1", "name": "search_corpus", "status": "ok"}],
    }


# -- related (chat mini map) ---------------------------------------------------


def test_related_round_trips_the_depth1_neighborhood(client):
    response = client.post("/api/graph/related", json={"ids": ["cap-1"], "per_node": 3})

    assert response.status_code == 200
    assert response.json() == {"nodes": [], "edges": [], "centers": []}


def test_related_validates_the_request(client):
    assert client.post("/api/graph/related", json={"ids": []}).status_code == 422
    assert client.post("/api/graph/related", json={"ids": [f"e-{i}" for i in range(13)]}).status_code == 422
    assert client.post("/api/graph/related", json={"ids": ["x"], "per_node": 0}).status_code == 422
    assert client.post("/api/graph/related", json={"ids": ["x"], "per_node": 9}).status_code == 422


def test_ask_stream_emits_progress_then_the_final_payload(client, monkeypatch):
    """The stream is progress events as the turn unfolds, then one final event
    carrying the /api/ask payload unchanged."""
    import json as _json

    def fake_chat(payload, on_event=None):
        on_event({"type": "start", "trace_id": "chat-abc"})
        on_event({"type": "tool_done", "name": "search_corpus", "duration_ms": 12,
                  "summary": {"name": "search_corpus", "hits": 2}, "write": False})
        return {"answer": "streamed", "tools": [], "writes": [], "trace_id": "chat-abc",
                "session_id": payload["session_id"], "tool_search": [], "delta": None}

    monkeypatch.setattr(web_app.demo_server, "chat", fake_chat)

    response = client.post("/api/ask/stream", json={"question": "hi?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [_json.loads(line[5:])
              for line in response.text.splitlines() if line.startswith("data:")]
    assert [e["type"] for e in events] == ["start", "tool_done", "final"]
    assert events[-1]["result"]["answer"] == "streamed"
    assert events[-1]["result"]["trace_id"] == "chat-abc"


def test_ask_stream_rejects_a_blank_question(client):
    assert client.post("/api/ask/stream", json={"question": "   "}).status_code == 422
