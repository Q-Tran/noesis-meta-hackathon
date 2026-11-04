"""The chat's write boundary is graph code, so it can be checked without a model.

The interesting failures are identity and provenance, not prose: a qualified
type must not become a label, `BusinessTerm` resolves by name, two argument
bundles must not collapse, and a claim's chunk must exist before anything is
written. The HTTP test pins the endpoint the browser actually calls.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "demo" / "graph_delta"
sys.path.insert(0, str(DEMO))

from chat_tools import TOOL_SPECS, ChatTools, _bare_label, _qualified_type  # noqa: E402


class FakeGraph:
    """Only the reads/writes `ChatTools` uses, with merge-by-key semantics."""

    def __init__(self, entities=None, claims=None, chunks=None, arguments=None,
                 claim_chunks=None, premise_args=None):
        self.entities = {(label, identity): dict(props)
                         for (label, identity), props in (entities or {}).items()}
        self.claims = set(claims or ())
        self.chunks = set(chunks or ())
        self.arguments = dict(arguments or {})
        self.claim_chunks = dict(claim_chunks or {})
        self.premise_args = dict(premise_args or {})
        self.nodes = []
        self.rels = []
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))
        if "RETURN properties(n) AS props" in query:
            label = query.split("(n:`", 1)[1].split("`", 1)[0]
            identity = params.get("ident", params.get("id"))
            props = self.entities.get((label, identity))
            return [{"props": dict(props)}] if props is not None else []
        if "MATCH (c:Claim) WHERE c.id IN $ids" in query:
            return [{"id": cid} for cid in params["ids"] if cid in self.claims]
        if "RETURN DISTINCT ch.id AS chunk_id" in query:
            return [{"chunk_id": self.claim_chunks[cid]}
                    for cid in params["ids"] if cid in self.claim_chunks]
        if "MATCH (c:Chunk {id: $id})" in query:
            return [{"id": params["id"]}] if params["id"] in self.chunks else []
        if "MATCH (arg:Argument {id: $id}) RETURN properties(arg) AS props" in query:
            props = self.arguments.get(params["id"])
            return [{"props": dict(props)}] if props is not None else []
        if "MATCH (arg:Argument)-[:USES_PREMISE]->(c:Claim {id: $id})" in query:
            props = self.premise_args.get(params["id"])
            return [{"id": props["id"], "props": dict(props)}] if props is not None else []
        if "RETURN labels(n) AS labels" in query:
            for (label, identity), props in self.entities.items():
                if identity == params["id"]:
                    return [{"labels": [label]}]
            return []
        return []

    def upsert_node(self, label, key, props=None):
        self.nodes.append((label, dict(key), dict(props or {})))

    def upsert_rel(self, start_label, start_key, rel_type, end_label, end_key,
                   props=None):
        self.rels.append((start_label, dict(start_key), rel_type, end_label,
                          dict(end_key), dict(props or {})))
        # ground_claim_in_trace raises on an empty write; the fake merges.
        return [{}]


def _titled(label, identity, name=None, **props):
    return (label, identity), {"id": identity, "name": name or identity, **props}


# -- type aliases ---------------------------------------------------------------

def test_entity_type_aliases_resolve_both_ways():
    assert _qualified_type("Component") == "architecture.Component"
    assert _qualified_type("architecture.Component") == "architecture.Component"
    assert _bare_label("architecture.Component") == "Component"
    assert _bare_label("Component") == "Component"
    assert _qualified_type("NotAType") is None
    assert _bare_label("architecture.NotAType") is None


# -- claims ---------------------------------------------------------------------

def test_a_claim_grounds_in_the_chunk_it_cites():
    graph = FakeGraph(chunks={"chunk-1"})
    tools = ChatTools(graph, "test")

    result = tools.add_claim("Muse Wonder accepts image input.", chunk_id="chunk-1")

    assert result["grounded_in"] == "chunk:chunk-1"
    assert graph.nodes[0][0] == "Claim"
    assert graph.rels == [(
        "Claim", {"id": result["claim_id"]}, "FOUND_IN", "Chunk",
        {"id": "chunk-1"}, {"role": "extracted"},
    )]


def test_a_bad_chunk_does_not_leave_an_orphan_claim():
    graph = FakeGraph()
    tools = ChatTools(graph, "test")

    result = tools.add_claim("Muse Wonder accepts image input.", chunk_id="missing")

    assert "not found" in result["error"]
    assert graph.nodes == []
    assert graph.rels == []


def test_a_claim_without_a_trace_or_chunk_is_rejected_before_writing():
    graph = FakeGraph()
    result = ChatTools(graph, "test").add_claim("Muse Wonder accepts image input.")

    assert "no active chat turn" in result["error"]
    assert graph.nodes == []


def test_the_claim_tool_advertises_chunk_grounding():
    spec = next(t for t in TOOL_SPECS if t["function"]["name"] == "add_claim")
    assert "chunk_id" in spec["function"]["parameters"]["properties"]
    assert "Chunk" in spec["function"]["description"]


# -- arguments and identities ---------------------------------------------------

def test_an_argument_to_a_name_identified_entity_uses_its_name_key():
    graph = FakeGraph(
        entities=dict([_titled("BusinessTerm", "ARR", name="ARR")]),
        claims={"claim-1"},
    )
    tools = ChatTools(graph, "test")

    result = tools.add_argument("BusinessTerm", ["claim-1"], entity_name="ARR")

    assert result["concludes"] == "BusinessTerm:ARR"
    assert ("Argument", {"id": result["argument_id"]}, "CONCLUDES",
            "BusinessTerm", {"name": "ARR"}) in [
        (a, b, c, d, e) for a, b, c, d, e, _ in graph.rels
    ]


def test_a_single_claim_id_string_is_accepted():
    graph = FakeGraph(
        entities=dict([_titled("Component", "demo-comp", name="Muse Code")]),
        claims={"claim-1"},
    )
    result = ChatTools(graph, "test").add_argument(
        "Component", "claim-1", entity_id="demo-comp",
    )

    assert result["premises"] == ["claim-1"]


def test_arguments_for_different_claims_do_not_collapse():
    graph = FakeGraph(
        entities=dict([_titled("Component", "demo-comp", name="Muse Code")]),
        claims={"claim-1", "claim-2"},
    )
    tools = ChatTools(graph, "test")

    first = tools.add_argument("Component", ["claim-1"], entity_id="demo-comp")
    second = tools.add_argument("Component", ["claim-2"], entity_id="demo-comp")

    assert first["argument_id"] != second["argument_id"]


def test_relating_a_term_resolves_its_name_not_a_fake_id():
    graph = FakeGraph(entities=dict([
        _titled("BusinessTerm", "ARR", name="ARR"),
        _titled("Component", "demo-comp", name="Muse Code"),
    ]))
    tools = ChatTools(graph, "test")

    result = tools.relate_entities(
        "business_dictionary.BusinessTerm", "ARR", "DEPENDS_ON",
        "architecture.Component", "demo-comp",
    )

    assert result["from"] == "BusinessTerm:ARR"
    assert graph.rels[-1][1] == {"name": "ARR"}
    assert graph.rels[-1][3] == "Component"
    assert graph.rels[-1][4] == {"id": "demo-comp"}


def test_a_qualified_write_type_stays_a_bare_label():
    graph = FakeGraph(entities=dict([
        _titled("Component", "demo-comp", name="Muse Code"),
    ]))
    tools = ChatTools(graph, "test")

    result = tools.update_entity(
        "architecture.Component", "demo-comp", {"review_status": "ok"},
    )

    assert result["updated"] == "Component:demo-comp"
    assert graph.nodes[-1][0] == "Component"
    assert graph.nodes[-1][2] == {"review_status": "ok"}


def test_update_rejects_properties_outside_the_allowlist():
    graph = FakeGraph(entities=dict([
        _titled("Component", "demo-comp", name="Muse Code"),
    ]))
    result = ChatTools(graph, "test").update_entity(
        "Component", "demo-comp", {"secret": "x"},
    )

    assert "no allowed properties" in result["error"]
    assert graph.nodes == []


# -- HTTP -----------------------------------------------------------------------

def _server_module():
    spec = importlib.util.spec_from_file_location(
        "graph_delta_server_for_test", DEMO / "server.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_browser_endpoint_serves_the_page_and_a_chat_reply(monkeypatch):
    server = _server_module()
    seen = {}

    def fake_chat(payload):
        seen.update(payload)
        return {
            "answer": "grounded", "tools": [], "writes": [],
            "trace_id": "chat-test", "session_id": payload.get("session_id"),
        }

    monkeypatch.setattr(server, "chat", fake_chat)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as response:
            assert response.status == 200
            assert json.loads(response.read())["ok"] is True

        with urllib.request.urlopen(base + "/", timeout=5) as response:
            assert response.status == 200
            assert "Ask the graph" in response.read().decode()

        request = urllib.request.Request(
            base + "/api/chat",
            data=json.dumps({
                "session_id": "http-test",
                "messages": [{"role": "user", "content": "hi"}],
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = json.loads(response.read())
        assert body["answer"] == "grounded"
        assert seen["session_id"] == "http-test"

        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/api/nope", timeout=5)
        assert error.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


# -- corrections --------------------------------------------------------------

def test_challenging_an_argument_attacks_its_warrant():
    """The showcase dispute: the reason becomes a claim grounded in the turn,
    the new argument attacks the old one's warrant, and the old argument's
    status is untouched (resolution belongs to repoint_reference)."""
    graph = FakeGraph(
        chunks={"chunk-1"},
        arguments={"arg-1": {"id": "arg-1", "chunk_id": "chunk-1", "status": "in"}},
    )
    tools = ChatTools(graph, "test")
    tools.begin_turn("this argument does not make sense from this chunk")

    result = tools.challenge_argument(
        "arg-1", "The chunk describes streaming, not native tool calling."
    )

    assert result["attack_target"] == "warrant"
    assert result["grounded_in"] == f"trace:{tools.trace_id}"
    # No write touched the disputed argument.
    assert not any(key.get("id") == "arg-1" for _, key, _ in graph.nodes)
    attack_node = next(n for n in graph.nodes if n[1]["id"] == result["attack_argument_id"])
    assert attack_node[2]["status"] == "in"
    assert attack_node[2]["chunk_id"] == "chunk-1"
    assert (
        "Argument", {"id": result["attack_argument_id"]}, "USES_PREMISE",
        "Claim", {"id": result["premise_claim_id"]}, {}
    ) in graph.rels
    assert (
        "Argument", {"id": result["attack_argument_id"]}, "ATTACKS",
        "Argument", {"id": "arg-1"},
        {"target": "warrant", "reason": "The chunk describes streaming, not native tool calling."},
    ) in graph.rels
    # The premise claim is substantiated by the conversation turn.
    assert (
        "Claim", {"id": result["premise_claim_id"]}, "FOUND_IN",
        "RoutingTrace", {"id": tools.trace_id}, {"role": "origin"},
    ) in graph.rels


def test_challenging_a_missing_argument_errors():
    result = ChatTools(FakeGraph(), "test").challenge_argument(
        "arg-nope", "This reason is long enough to pass validation."
    )

    assert result == {"error": "argument 'arg-nope' not found"}


def test_repointing_a_reference_supersedes_and_records_the_turn():
    """The resolution: the engine's repoint does the supersede/rewire work, and
    the correction claim records that this conversation originated the change --
    no re-ingestion of the chunk."""
    _, entity = _titled("Component", "comp-muse-wonder", name="Muse Wonder")
    graph = FakeGraph(entities={("Component", "comp-muse-wonder"): entity},
                      chunks={"chunk-1"})
    calls = {}

    def fake_repoint(chunk_id, old_id, new_id, new_entity_label=None, reason=None):
        calls["args"] = (chunk_id, old_id, new_id, new_entity_label, reason)
        return {
            "chunk_id": chunk_id, "old_entity_id": old_id,
            "new_entity_id": new_id, "old_arg_id": "arg-old",
            "new_arg_id": "arg-new", "transferred_properties": ["description"],
            "repointed_relationships": 2, "deprecated_old_entity": False,
            "status": "repointed",
        }

    graph.repoint_entity_reference = fake_repoint
    tools = ChatTools(graph, "test")
    tools.begin_turn("repoint that reference to Muse Code")

    result = tools.repoint_reference(
        "chunk-1", "comp-muse-wonder", "comp-muse-code", reason="misread extraction",
    )

    assert calls["args"] == ("chunk-1", "comp-muse-wonder", "comp-muse-code", None, "misread extraction")
    assert result["new_arg_id"] == "arg-new"
    assert result["reingested"] is False
    assert result["grounded_in"] == f"trace:{tools.trace_id}"
    # The correction claim is grounded in the turn, not in the chunk.
    correction = next(r for r in graph.rels if r[2] == "FOUND_IN")
    assert correction == (
        "Claim", {"id": result["correction_claim_id"]}, "FOUND_IN",
        "RoutingTrace", {"id": tools.trace_id}, {"role": "origin"},
    )


def test_repointing_with_a_missing_chunk_never_reaches_the_engine():
    graph = FakeGraph()
    graph.repoint_entity_reference = lambda *a, **kw: pytest.fail("repoint must not run")

    result = ChatTools(graph, "test").repoint_reference(
        "chunk-nope", "comp-a", "comp-b",
    )

    assert result == {"error": "chunk 'chunk-nope' not found"}


def test_an_argument_from_chunk_grounded_claims_is_a_resolution_candidate():
    """Without the candidate stamp, repoint_reference cannot find the argument
    it must supersede (chunk_id + candidate_id is how the engine identifies the
    resolution a reference came from)."""
    _, entity = _titled("Component", "comp-muse-wonder", name="Muse Wonder")
    graph = FakeGraph(
        entities={("Component", "comp-muse-wonder"): entity},
        claims={"claim-1"},
        claim_chunks={"claim-1": "chunk-1"},
    )
    tools = ChatTools(graph, "test")

    result = tools.add_argument("architecture.Component", ["claim-1"],
                                entity_id="comp-muse-wonder")

    arg_node = next(n for n in graph.nodes if n[0] == "Argument")
    assert arg_node[1]["id"] == result["argument_id"]
    assert arg_node[2]["chunk_id"] == "chunk-1"
    assert arg_node[2]["candidate_id"] == "comp-muse-wonder"
    assert arg_node[2]["candidate_label"] == "Component"


def test_an_argument_from_turn_grounded_claims_has_no_candidate_stamp():
    """A claim grounded in the conversation is not a chunk reference, so the
    argument must not pretend to be a resolution candidate for one."""
    _, entity = _titled("Component", "comp-muse-wonder", name="Muse Wonder")
    graph = FakeGraph(
        entities={("Component", "comp-muse-wonder"): entity},
        claims={"claim-1"},
    )
    tools = ChatTools(graph, "test")

    tools.add_argument("architecture.Component", ["claim-1"],
                       entity_id="comp-muse-wonder")

    arg_node = next(n for n in graph.nodes if n[0] == "Argument")
    assert "chunk_id" not in arg_node[2]
    assert "candidate_id" not in arg_node[2]


# -- showcase page and its endpoints -------------------------------------------

class FakeSubgraphGraph:
    """The one big read (and the reset deletes) `server.showcase_*` runs."""

    def __init__(self):
        self.deletes = []
        self.closed = False

    def run(self, query, **params):
        if "DETACH DELETE" in query:
            self.deletes.append((query, params))
            return [{"c": 2}]
        assert "MATCH (cap:Capability {id: $cap_id})" in query
        assert params["stale_arg"] == "extarg-8dd04b0c8e-fc846d6c"
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
        return [row]

    def close(self):
        self.closed = True


def _boot_showcase_server(monkeypatch):
    server = importlib.import_module("server")
    monkeypatch.setattr(server, "Graph", FakeSubgraphGraph)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


def test_the_showcase_page_and_subgraph_endpoint(monkeypatch):
    httpd, thread = _boot_showcase_server(monkeypatch)
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        with urllib.request.urlopen(f"{base}/showcase") as resp:
            html = resp.read().decode()
        assert "changing the canonical truth" in html
        assert 'src="http' not in html and 'href="http' not in html  # self-contained

        with urllib.request.urlopen(f"{base}/api/showcase/subgraph") as resp:
            data = json.load(resp)
        assert data["capability_id"] == "extent-591dfe996e"
        titles = {n["title"] for n in data["nodes"]}
        assert "Tool search" in titles
        assert "Tool search is not available on Chat Completions" in titles
        labels = {n["labels"][0] for n in data["nodes"]}
        assert labels == {"Capability", "Argument", "Claim", "Chunk", "SourceNode"}
        types = {(e["type"], e["source"], e["target"]) for e in data["edges"]}
        assert ("CONCLUDES", "arg-1", "cap-1") in types
        assert ("FOUND_IN", "claim-1", "chunk-1") in types
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_the_showcase_reset_only_deletes_scenario_writes(monkeypatch):
    server = importlib.import_module("server")
    created = []

    def factory():
        fake = FakeSubgraphGraph()
        created.append(fake)
        return fake

    monkeypatch.setattr(server, "Graph", factory)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        req = urllib.request.Request(f"{base}/api/showcase/reset", data=b"{}", method="POST")
        with urllib.request.urlopen(req) as resp:
            body = json.load(resp)
        assert body["deleted_nodes"] == 8  # four deletes x the fake's 2 each
        fake = created[0]
        assert fake.closed
        queries = " ".join(q for q, _ in fake.deletes)
        # Scoped to the scenario: chat-prefixed writes, the stale argument, the capability.
        assert "chatarg-" in queries
        assert "chatclaim-" in queries
        params_blob = json.dumps([p for _, p in fake.deletes])
        assert "extarg-8dd04b0c8e-fc846d6c" in params_blob
        assert "extent-591dfe996e" in params_blob
        # And never a blanket delete.
        assert "MATCH (n) DETACH DELETE" not in queries
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_the_embedding_model_path_resolves_against_the_repo_root(monkeypatch):
    """The server may start from any CWD (starting it inside demo/ is exactly
    how the weights went "missing" at demo time); the path must still resolve."""
    import types

    import chat_tools

    captured = {}

    class FakeSentenceTransformer:
        def __init__(self, path, local_files_only=False):
            captured["path"] = path
            captured["local_files_only"] = local_files_only

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(chat_tools, "_model", None)
    monkeypatch.setattr(chat_tools, "_store", object())
    monkeypatch.setattr(chat_tools, "_llm", object())

    chat_tools._models()

    expected = Path(chat_tools.__file__).resolve().parents[2] / "models/all-MiniLM-L6-v2"
    assert captured["path"] == str(expected)
    assert captured["local_files_only"] is True
    assert expected.is_dir(), "the weights must actually be on disk"
    monkeypatch.setattr(chat_tools, "_model", None)


def test_get_entity_returns_the_concluding_argument_ids_verbatim():
    """challenge_argument needs an argument id; the only honest way to get one
    is to read it, and this is the field the demo agent reads it from."""
    _, entity = _titled("Component", "comp-muse-wonder", name="Muse Wonder")
    graph = FakeGraph(entities={("Component", "comp-muse-wonder"): entity})
    tools = ChatTools(graph, "test")

    result = tools.get_entity("architecture.Component", "comp-muse-wonder")

    assert result["arguments"] == []  # the fake holds none; shape is the contract
    queries = [q for q, _ in graph.queries if "MATCH (arg:Argument)-[:CONCLUDES]->(n)" in q]
    assert queries, "get_entity must look the arguments up, not guess them"
    # The lookup lands on the turn's mini-map list for the ask view.
    assert tools.turn_entities == {
        "comp-muse-wonder": {"id": "comp-muse-wonder", "title": "Muse Wonder", "label": "Component"},
    }


def test_note_entity_caps_the_mini_map_at_a_readable_size():
    tools = ChatTools(FakeGraph(), "test")
    for i in range(12):
        tools._note_entity(f"e-{i}", f"entity {i}", "Component")
    assert len(tools.turn_entities) == 8
    assert tools.turn_entities["e-0"]["title"] == "entity 0"


def test_challenging_by_claim_attacks_the_argument_behind_it():
    """The reliable demo path: name the statement, and the tool resolves which
    argument uses it as a premise -- no guessing argument ids."""
    graph = FakeGraph(
        premise_args={"claim-1": {"id": "arg-behind-it", "chunk_id": "chunk-1", "status": "in"}},
    )
    tools = ChatTools(graph, "test")
    tools.begin_turn("challenge this claim")

    result = tools.challenge_argument(
        claim_id="claim-1",
        reason="The chunk predates the launch, so it cannot support this claim.",
    )

    assert result["attacks"] == "arg-behind-it"
    assert result["attack_target"] == "warrant"


def test_challenging_by_an_unknown_claim_errors():
    result = ChatTools(FakeGraph(), "test").challenge_argument(
        claim_id="claim-nope",
        reason="The chunk predates the launch, so it cannot support this claim.",
    )

    assert result == {"error": "no argument uses claim 'claim-nope' as a premise"}


def test_challenging_without_any_target_errors():
    result = ChatTools(FakeGraph(), "test").challenge_argument(
        reason="The chunk predates the launch, so it cannot support this claim.",
    )

    assert result == {"error": "argument_id (or claim_id) is required"}


# -- turn delta --------------------------------------------------------------

class FakeDeltaGraph:
    """The three queries `turn_delta` runs, answered from canned rows."""

    def __init__(self, nodes=None, trace_claims=(), edges=None):
        self.nodes = list(nodes or [])
        self.trace_claims = set(trace_claims)
        self.edges = list(edges or [])
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))
        if "MATCH (n) WHERE n.id IN $ids" in query:
            wanted = set(params["ids"])
            return [row for row in self.nodes if row["props"]["id"] in wanted]
        if "MATCH (c:Claim)-[:FOUND_IN]->(t:RoutingTrace {id: $tid})" in query:
            return [{"id": cid} for cid in params["ids"] if cid in self.trace_claims]
        if "MATCH (a)-[r]-(b) WHERE a.id IN $ids OR b.id IN $ids" in query:
            return self.edges
        return []

    def close(self):
        pass


def _delta_row(eid, label, **props):
    return {"element_id": eid, "labels": [label], "props": props}


def test_turn_delta_reports_new_nodes_edges_and_the_source_of_each_claim():
    server = importlib.import_module("server")
    graph = FakeDeltaGraph(
        nodes=[
            _delta_row("e1", "Claim", id="claim-new", statement="Tool search is everywhere now.", status="active"),
            _delta_row("e2", "Claim", id="claim-premise", statement="The docs predate the launch.", status="active"),
            _delta_row("e3", "Argument", id="arg-attack", status="in"),
        ],
        trace_claims={"claim-new", "claim-premise"},
        edges=[
            {"element_id": "r1", "type": "ATTACKS", "source_id": "arg-attack", "source_label": "Argument",
             "target_id": "extarg-old", "target_label": "Argument"},
            {"element_id": "r2", "type": "FOUND_IN", "source_id": "claim-new", "source_label": "Claim",
             "target_id": "trace-1", "target_label": "RoutingTrace"},
        ],
    )
    writes = [
        {"name": "challenge_argument", "attack_argument_id": "arg-attack",
         "attacks": "extarg-old", "attack_target": "warrant", "premise_claim_id": "claim-premise"},
        {"name": "add_claim", "claim_id": "claim-new", "grounded_in": "trace:trace-1"},
    ]

    delta = server.turn_delta(graph, "trace-1", writes)

    assert [n["id"] for n in delta["nodes"]] == ["claim-new", "claim-premise", "arg-attack"]
    assert delta["edges"][0]["type"] == "ATTACKS"
    texts = [x["text"] for x in delta["explanation"]]
    assert any("Tool search is everywhere now" in t for t in texts)
    # Trace-sourced claims are attributed to the user, chunk-sourced to the docs.
    new_claim_line = next(x for x in delta["explanation"] if "Tool search is everywhere now" in x["text"])
    assert "you are the source" in new_claim_line["why"]
    attack_line = next(x for x in delta["explanation"] if "attacks extarg-old" in x["text"])
    assert attack_line["mark"] == "⚔"
    assert "challenged, not deleted" in attack_line["why"]


def test_turn_delta_reports_the_corrections_flips():
    server = importlib.import_module("server")
    graph = FakeDeltaGraph(
        nodes=[_delta_row("e9", "Argument", id="arg-new", status="in")],
    )
    writes = [{
        "name": "repoint_reference", "new_arg_id": "arg-new", "old_arg_id": "arg-old",
        "old_entity_id": "comp-old", "new_entity_id": "comp-new",
        "deprecated_old_entity": True, "status": "repointed",
    }]

    delta = server.turn_delta(graph, "trace-1", writes)

    assert delta["flips"] == [
        {"kind": "Argument", "title": "arg-old", "from": "in", "to": "out"},
        {"kind": "Entity", "title": "comp-old", "from": "active", "to": "Deprecated"},
    ]
    assert any(x["text"] == "the chunk was not re-ingested" for x in delta["explanation"])


def test_the_chat_response_includes_the_turn_delta(monkeypatch):
    server = importlib.import_module("server")
    monkeypatch.setattr(server, "api_key", lambda: "key")

    calls = iter([
        {"output": [
            {"type": "function_call", "call_id": "c1", "name": "add_claim",
             "arguments": json.dumps({"statement": "Tool search is available everywhere now."})},
        ]},
        {"output": [
            {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
        ]},
    ])
    monkeypatch.setattr(server, "model_call", lambda key, conversation: next(calls))

    class TurnTools:
        def __init__(self, graph, session_id):
            self.trace_id = None
            self.turn_entities = {}

        def begin_turn(self, question):
            self.trace_id = "trace-1"
            return self.trace_id

        def add_claim(self, statement, **kwargs):
            self._note_entity("extent-ts", "Tool search", "Capability")
            return {"claim_id": "claim-new", "grounded_in": "trace:trace-1", "statement": statement}

        def _note_entity(self, entity_id, title, label):
            self.turn_entities[entity_id] = {"id": entity_id, "title": title, "label": label}

    monkeypatch.setattr(server, "ChatTools", TurnTools)
    monkeypatch.setattr(server, "Graph", lambda: FakeDeltaGraph(
        nodes=[_delta_row("e1", "Claim", id="claim-new",
                          statement="Tool search is available everywhere now.", status="active")],
        trace_claims={"claim-new"},
    ))

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        req = urllib.request.Request(
            f"{base}/api/chat",
            data=json.dumps({"session_id": "delta-test",
                             "messages": [{"role": "user", "content": "update it"}]}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.load(resp)
        assert body["answer"] == "done"
        assert body["delta"]["nodes"][0]["id"] == "claim-new"
        assert body["delta"]["nodes"][0]["mark"] == "+"
        assert body["delta"]["explanation"][0]["mark"] == "+"
        assert "you are the source" in body["delta"]["explanation"][0]["why"]
        # The mini map + model attribution ride the same payload.
        assert body["entities"] == [{"id": "extent-ts", "title": "Tool search", "label": "Capability"}]
        assert body["model"] == server.MODEL
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


# -- tool search dogfooding + turn spans --------------------------------------

def test_responses_tools_are_all_deferred_with_one_tool_search_marker():
    server = importlib.import_module("server")
    tools = server.responses_tools()

    fns = [t for t in tools if t["type"] == "function"]
    markers = [t for t in tools if t["type"] == "tool_search"]
    assert len(fns) == len(TOOL_SPECS) == 9
    assert all(t["defer_loading"] is True for t in fns), (
        "every declared tool is deferred -- that is the token saving the demo is about"
    )
    assert markers == [{"type": "tool_search"}]


class _FakeResponsesTurn:
    def __init__(self, graph, session_id):
        self.trace_id = None
        self.turn_entities = {}

    def begin_turn(self, question):
        self.trace_id = "trace-ts"
        return self.trace_id

    def search_corpus(self, query, top_k=5):
        return {"results": [{"chunk_id": "chunk-1", "page": "Tool search", "url": "u"}]}


class _CapturingTraceStore:
    events = []

    def __init__(self, graph):
        pass

    def insert_span(self, event):
        _CapturingTraceStore.events.append(event)


def test_the_chat_loop_uses_metas_tool_search_and_records_turn_spans(monkeypatch):
    server = importlib.import_module("server")
    monkeypatch.setattr(server, "api_key", lambda: "key")
    monkeypatch.setattr(server, "ChatTools", _FakeResponsesTurn)
    monkeypatch.setattr(server, "TraceStore", _CapturingTraceStore)
    monkeypatch.setattr(server, "Graph", lambda: FakeDeltaGraph())
    _CapturingTraceStore.events = []

    responses = iter([
        {"output": [
            {"type": "tool_search_call", "arguments": {"paths": ["tool search"]}},
            {"type": "tool_search_output", "tools": [{"name": "search_corpus"}]},
            {"type": "function_call", "call_id": "call-1", "name": "search_corpus",
             "arguments": json.dumps({"query": "tool search"})},
        ]},
        {"output": [
            {"type": "message", "content": [{"type": "output_text",
                                             "text": "Tool search defers loading."}]},
        ]},
    ])
    monkeypatch.setattr(server, "model_call", lambda key, items: next(responses))

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        req = urllib.request.Request(
            f"{base}/api/chat",
            data=json.dumps({"session_id": "ts-test",
                             "messages": [{"role": "user", "content": "what is tool search?"}]}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.load(resp)

        assert body["answer"] == "Tool search defers loading."
        assert body["tool_search"] == [
            {"searched": {"paths": ["tool search"]}, "loaded": ["search_corpus"]}
        ]
        assert len(_CapturingTraceStore.events) == 1
        span = _CapturingTraceStore.events[0]
        assert span.name == "search_corpus"
        assert span.trace_id == "trace-ts"
        assert span.raw_input == {"arguments": {"query": "tool search"}}
        # function_call_output fed the next call, keeping the protocol loop alive
        assert body["session_id"] == "ts-test"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_the_trace_endpoint_returns_the_turns_recorded_spans(monkeypatch):
    server = importlib.import_module("server")

    class _SpanStore:
        def __init__(self, graph):
            pass

        def get_trace_spans(self, trace_id):
            assert trace_id == "t-1"
            return [{"span_id": "s-1", "name": "search_corpus", "kind": "tool_call",
                     "status": "ok", "duration_ms": 12, "session_id": "s",
                     "client": "web-chat", "raw_input": {"arguments": {"query": "x"}},
                     "raw_output": {"results": 1}}]

    monkeypatch.setattr(server, "TraceStore", _SpanStore)
    monkeypatch.setattr(server, "Graph", lambda: FakeDeltaGraph())

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        with urllib.request.urlopen(f"{base}/api/trace/t-1") as resp:
            body = json.load(resp)
        assert body["trace_id"] == "t-1"
        assert body["spans"][0]["name"] == "search_corpus"
        assert body["spans"][0]["duration_ms"] == 12
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_repoint_reference_stamps_extraction_era_arguments_first():
    """An unstamped old argument ('in', no chunk/candidate) would survive the
    repoint and keep vouching for the rejected entity. The stamp makes the
    engine's supersede chain find and retire it."""
    _, entity = _titled("Capability", "extent-dd194c867a", name="ToolSearch")
    graph = FakeGraph(entities={("Capability", "extent-dd194c867a"): entity},
                      chunks={"chunk-1"})
    graph.queries = []
    # The stale-args read: one extraction-era argument concludes the entity.
    original_run = graph.run

    def run(query, **params):
        if "MATCH (arg:Argument)-[:CONCLUDES]->(n {id: $old_id})" in query:
            return [{"id": "extarg-753185a92d-421def60"}]
        return original_run(query, **params)

    graph.run = run
    graph.repoint_entity_reference = lambda *a, **kw: {
        "status": "repointed", "old_arg_id": "extarg-753185a92d-421def60",
        "new_arg_id": "arg-new", "chunk_id": "chunk-1",
        "old_entity_id": "extent-dd194c867a", "new_entity_id": "extent-a80d8e3c54",
        "deprecated_old_entity": True,
    }

    tools = ChatTools(graph, "test")
    tools.begin_turn("dedup")
    result = tools.repoint_reference("chunk-1", "extent-dd194c867a", "extent-a80d8e3c54")

    assert result["stamped_old_arguments"] == ["extarg-753185a92d-421def60"]
    stamp_queries = [p for q, p in graph.queries
                     if "SET arg.chunk_id" in q]
    assert stamp_queries == [{
        "id": "extarg-753185a92d-421def60", "chunk_id": "chunk-1",
        "old_id": "extent-dd194c867a", "label": "Capability",
    }]


def test_turn_delta_marks_created_and_defeated_nodes_and_edges_for_the_visual():
    """The visual diff colors by mark: created nodes/edges green (+), the
    attacked argument red (⚔), the superseded argument/entity amber (Δ)."""
    server = importlib.import_module("server")
    graph = FakeDeltaGraph(
        nodes=[
            _delta_row("e1", "Argument", id="arg-new", status="in"),
            _delta_row("e2", "Argument", id="extarg-old", status="out"),
            _delta_row("e3", "Capability", id="extent-old", name="ToolSearch"),
        ],
        edges=[
            {"element_id": "r1", "type": "ATTACKS", "source_id": "arg-new", "source_label": "Argument",
             "target_id": "extarg-old", "target_label": "Argument"},
            {"element_id": "r2", "type": "SUPERSEDES", "source_id": "arg-new", "source_label": "Argument",
             "target_id": "extarg-old", "target_label": "Argument"},
            {"element_id": "r3", "type": "CONCLUDES", "source_id": "arg-new", "source_label": "Argument",
             "target_id": "extent-real", "target_label": "Capability"},
        ],
    )
    writes = [{
        "name": "repoint_reference", "new_arg_id": "arg-new", "old_arg_id": "extarg-old",
        "old_entity_id": "extent-old", "deprecated_old_entity": True,
    }]

    delta = server.turn_delta(graph, "trace-1", writes)

    marks = {n["id"]: n["mark"] for n in delta["nodes"]}
    assert marks == {"arg-new": "+", "extarg-old": "Δ", "extent-old": "Δ"}
    edge_marks = {e["element_id"]: e["mark"] for e in delta["edges"]}
    assert edge_marks == {"r1": "⚔", "r2": "Δ", "r3": "+"}


def test_turn_delta_orients_the_correction_edges_from_the_writes():
    """The undirected edge match can return an ATTACKS edge pointing from the
    attacked argument to the attacker; the writes name the real direction, and
    the visual's arrows and node roles depend on it."""
    server = importlib.import_module("server")
    graph = FakeDeltaGraph(
        nodes=[
            _delta_row("e1", "Argument", id="arg-attack", status="in"),
            _delta_row("e2", "Argument", id="extarg-old", status="in"),
        ],
        edges=[
            # Wrong-way binding, as the undirected match is free to return.
            {"element_id": "r1", "type": "ATTACKS", "source_id": "extarg-old", "source_label": "Argument",
             "target_id": "arg-attack", "target_label": "Argument"},
        ],
    )
    writes = [{"name": "challenge_argument", "attack_argument_id": "arg-attack",
               "attacks": "extarg-old", "attack_target": "warrant"}]

    delta = server.turn_delta(graph, "trace-1", writes)

    edge = delta["edges"][0]
    assert (edge["source"], edge["target"]) == ("arg-attack", "extarg-old")
    marks = {n["id"]: n["mark"] for n in delta["nodes"]}
    assert marks == {"arg-attack": "+", "extarg-old": "⚔"}


def test_chat_streams_progress_events_and_returns_the_same_payload(monkeypatch):
    """on_event observes the turn as it unfolds (thinking, tool search, tool
    start/done) while the returned payload is the unchanged final result."""
    server = importlib.import_module("server")
    monkeypatch.setattr(server, "api_key", lambda: "key")
    monkeypatch.setattr(server, "ChatTools", _FakeResponsesTurn)
    monkeypatch.setattr(server, "Graph", lambda: FakeDeltaGraph())

    responses = iter([
        {"output": [
            {"type": "tool_search_call", "arguments": {"paths": ["tool search"]}},
            {"type": "tool_search_output", "tools": [{"name": "search_corpus"}]},
            {"type": "function_call", "call_id": "call-1", "name": "search_corpus",
             "arguments": json.dumps({"query": "tool search"})},
        ]},
        {"output": [
            {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
        ]},
    ])
    monkeypatch.setattr(server, "model_call", lambda key, items: next(responses))

    events = []
    result = server.chat(
        {"messages": [{"role": "user", "content": "what is tool search?"}]},
        on_event=events.append,
    )

    assert result["answer"] == "done"
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start" and events[0]["trace_id"] == "trace-ts"
    assert "tool_search" in kinds
    search_event = next(e for e in events if e["type"] == "tool_search")
    assert search_event["loaded"] == ["search_corpus"]
    done = next(e for e in events if e["type"] == "tool_done")
    assert done["name"] == "search_corpus"
    assert done["write"] is False
    assert done["summary"]["hits"] == 1
    assert kinds.index("tool_start") < kinds.index("tool_done")
