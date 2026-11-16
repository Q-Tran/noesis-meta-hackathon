"""Unit tests for the shared provenance helpers (`Graph.attach_source` /
`Graph.trace_sources`), exercised without a live Memgraph connection by
calling the unbound methods against a small recorder (mirrors the FakeGraph
pattern used elsewhere, e.g. `tests/test_migrations.py`)."""
import json
from graph import Graph


class RecordingGraph:
    def __init__(self):
        self.upserted_nodes = []
        self.upserted_rels = []
        self.calls = []

    def upsert_node(self, label, key, props=None):
        self.upserted_nodes.append((label, key, props))
        return []

    def upsert_rel(self, start_label, start_key, rel_type, end_label, end_key, props=None):
        self.upserted_rels.append((start_label, start_key, rel_type, end_label, end_key, props))
        return []

    def run(self, query, **params):
        self.calls.append((query, params))
        return []


def test_attach_source_upserts_source_node_and_found_in_edge():
    graph = RecordingGraph()

    Graph.attach_source(
        graph, "Table", {"name": "orders"}, "connector::local",
        {"name": "local", "type": "database_connector"}, role="hosted_by",
    )

    assert graph.upserted_nodes == [
        ("SourceNode", {"id": "connector::local"}, {"name": "local", "type": "database_connector"}),
    ]
    assert graph.upserted_rels == [
        ("Table", {"name": "orders"}, "FOUND_IN", "SourceNode",
         {"id": "connector::local"}, {"role": "hosted_by"}),
    ]


def test_attach_source_without_role_omits_role_property():
    graph = RecordingGraph()

    Graph.attach_source(graph, "Person", {"id": "p1"}, "source::doc", {"name": "doc"})

    assert graph.upserted_rels[0][-1] is None


def test_attach_source_allows_multiple_roles_on_the_same_node():
    graph = RecordingGraph()

    Graph.attach_source(graph, "Table", {"name": "orders"}, "connector::local",
                        {"type": "database_connector"}, role="hosted_by")
    Graph.attach_source(graph, "Table", {"name": "orders"}, "llm-describer::LLMDescriber",
                        {"type": "llm_description"}, role="description")

    roles = [rel[-1]["role"] for rel in graph.upserted_rels]
    assert roles == ["hosted_by", "description"]


def test_trace_sources_queries_found_in_edges():
    graph = RecordingGraph()

    Graph.trace_sources(graph, "Table", {"name": "orders"})

    assert len(graph.calls) == 1
    query, params = graph.calls[0]
    assert "FOUND_IN" in query
    assert "SourceNode" in query
    assert params == {"key_name": "orders"}


def test_attach_chunk_upserts_chunk_and_extracted_from_and_mentions_edges():
    graph = RecordingGraph()

    Graph.attach_chunk(
        graph, "Claim", {"id": "c1"}, "chunk::1",
        {"text": "Sample text", "index": 0}, role="extracted",
    )

    assert graph.upserted_nodes == [
        ("Chunk", {"id": "chunk::1"}, {"text": "Sample text", "index": 0}),
    ]
    assert graph.upserted_rels == [
        ("Claim", {"id": "c1"}, "EXTRACTED_FROM", "Chunk",
         {"id": "chunk::1"}, {"role": "extracted"}),
        ("Chunk", {"id": "chunk::1"}, "MENTIONS", "Claim",
         {"id": "c1"}, None),
    ]


def test_trace_chunks_queries_extracted_from_edges():
    graph = RecordingGraph()

    Graph.trace_chunks(graph, "Claim", {"id": "c1"})

    assert len(graph.calls) == 1
    query, params = graph.calls[0]
    assert "EXTRACTED_FROM" in query
    assert "Chunk" in query
    assert params == {"key_id": "c1"}


def test_trace_chunk_entities_queries_extracted_from():
    graph = RecordingGraph()

    Graph.trace_chunk_entities(graph, "chunk::1")

    assert len(graph.calls) == 1
    query, params = graph.calls[0]
    assert "EXTRACTED_FROM" in query
    assert params == {"chunk_id": "chunk::1"}


def test_trace_chunk_relationships_queries_chunk_id():
    graph = RecordingGraph()

    Graph.trace_chunk_relationships(graph, "chunk::1")

    assert len(graph.calls) == 1
    query, params = graph.calls[0]
    assert "r.chunk_id = $chunk_id" in query
    assert params == {"chunk_id": "chunk::1"}


def test_trace_relationship_chunk_queries_backing_chunk():
    graph = RecordingGraph()

    Graph.trace_relationship_chunk(
        graph, "Claim", {"id": "c1"}, "SUPPORTS", "Claim", {"id": "c2"}
    )

    assert len(graph.calls) == 1
    query, params = graph.calls[0]
    assert "r.chunk_id IS NOT NULL" in query
    assert "Chunk" in query
    assert params == {"s_id": "c1", "e_id": "c2"}


def test_search_provenance_coordinates_diskann_and_graph():
    from knowledge_bases.provenance.queries import search_provenance

    class FakeVectorStore:
        def search_chunks(self, embedding, top_k=5):
            return [{"id": "chunk::101", "similarity": 0.92, "text": "latency root cause"}]

    class FakeGraphForTrace:
        def trace_chunk_provenance(self, chunk_id):
            return {
                "chunk_id": chunk_id,
                "chunk": {"id": chunk_id, "text": "latency root cause"},
                "source": {"id": "source::doc1", "name": "incident doc"},
                "entities": [{"id": "claim::1", "label": "Claim"}],
                "relationships": [{"predicate": "CAUSED_BY", "chunk_id": chunk_id}],
            }

    class FakeLLM:
        def embed(self, query):
            return [0.1, 0.2, 0.3]

    results = search_provenance(
        "why is it slow?",
        top_k=3,
        graph=FakeGraphForTrace(),
        vector_store=FakeVectorStore(),
        llm=FakeLLM(),
    )

    assert len(results) == 1
    hit = results[0]
    assert hit["chunk_id"] == "chunk::101"
    assert hit["similarity"] == 0.92
    assert hit["text"] == "latency root cause"
    assert hit["source"]["id"] == "source::doc1"
    assert len(hit["entities"]) == 1
    assert len(hit["relationships"]) == 1


class FakeVectorStore:
    def __init__(self):
        self.upserted_chunks = []

    def upsert_chunk(self, id, embedding, metadata=None):
        self.upserted_chunks.append({"id": id, "embedding": embedding, "metadata": metadata})

    def upsert_vector(self, collection, id, embedding, metadata=None):
        pass

    def search_vectors(self, collection, embedding, top_k=5):
        return []


def test_extract_claims_vectorizes_chunks_and_tracks_provenance(monkeypatch):
    from extract import extract_claims

    vector_store = FakeVectorStore()
    upserted_chunks = vector_store.upserted_chunks

    class RecordingGraphWithReturn(RecordingGraph):
        def upsert_node(self, label, key, props=None):
            super().upsert_node(label, key, props)
            return [{"n": {**key, **(props or {})}}]

        def upsert_rel(self, start_label, start_key, rel_type, end_label, end_key, props=None):
            super().upsert_rel(start_label, start_key, rel_type, end_label, end_key, props)
            return [{"r": props or {}}]

    graph = RecordingGraphWithReturn()

    class FakeExtractionLLM:
        def embed(self, text):
            return [0.5, 0.5, 0.5]

        def extract(self, prompt, schema):
            return schema()

    monkeypatch.setattr("extract.get_llm", lambda: FakeExtractionLLM())

    chunk_id = extract_claims(
        text="Deploy completed without errors.",
        doc_title="Release Notes",
        source_name="CI/CD",
        url="https://ci.example.com/123",
        graph=graph,
        vector_store=vector_store,
    )

    # 1. Chunk was vectorized and upserted to vector store
    assert len(upserted_chunks) == 1
    assert upserted_chunks[0]["id"] == chunk_id
    assert upserted_chunks[0]["embedding"] == [0.5, 0.5, 0.5]
    assert upserted_chunks[0]["metadata"]["text"] == "Deploy completed without errors."

    # 2. Chunk node was upserted to graph
    chunk_nodes = [node for node in graph.upserted_nodes if node[0] == "Chunk"]
    assert len(chunk_nodes) == 1
    assert chunk_nodes[0][1]["id"] == chunk_id

    # 3. Chunk is connected to SourceNode via FOUND_IN
    chunk_to_source = [
        rel for rel in graph.upserted_rels
        if rel[0] == "Chunk" and rel[2] == "FOUND_IN" and rel[3] == "SourceNode"
    ]
    assert len(chunk_to_source) == 1
    assert chunk_to_source[0][1]["id"] == chunk_id


def test_extract_claims_records_property_origins_and_relationship_argument_ids(monkeypatch):
    from extract import extract_claims
    from knowledge_bases.base import ExtractionShape
    from pydantic import BaseModel

    class ExPerson(BaseModel):
        ref_id: str
        name: str

    class ExRel(BaseModel):
        source_ref_id: str
        target_ref_id: str
        relationship_type: str

    person_shape = ExtractionShape(
        label="Person",
        field="persons",
        model="fake:model",
        props="fake:props",
        identity="fake:identity",
    )

    seed_map = {
        "fake:model": ExPerson,
        "fake:props": lambda p, url: {"id": f"person::{p.name.lower()}", "name": p.name},
        "fake:identity": lambda props: props["id"],
    }

    monkeypatch.setattr("knowledge_bases.registry.extraction_shapes", lambda: [person_shape])
    monkeypatch.setattr("knowledge_bases.registry.resolve_seed", lambda s: seed_map.get(s, s))
    monkeypatch.setattr("knowledge_bases.ontology.is_known_label", lambda l: True)
    monkeypatch.setattr("extract.is_known_label", lambda l: True)
    monkeypatch.setattr("knowledge_bases.ontology.is_known_relationship", lambda r: True)
    monkeypatch.setattr("knowledge_bases.ontology.partition_triplets", lambda candidates: (candidates, []))
    monkeypatch.setattr("knowledge_bases.ontology.require_known_relationship", lambda r, **kw: r)
    monkeypatch.setattr("knowledge_bases.ontology.require_known_label", lambda l, **kw: l)

    graph = InMemoryProvenanceGraph()

    class FakeExtractionLLM:
        def embed(self, text):
            return [0.1, 0.2, 0.3]

        def extract(self, prompt, schema):
            res = schema()
            res.persons = [
                ExPerson(ref_id="p1", name="Alice"),
                ExPerson(ref_id="p2", name="Bob"),
            ]
            res.relationships = [
                ExRel(source_ref_id="p1", target_ref_id="p2", relationship_type="COLLABORATES_WITH"),
            ]
            return res

    monkeypatch.setattr("extract.get_llm", lambda: FakeExtractionLLM())

    chunk_id = extract_claims(
        text="Alice collaborates with Bob.",
        doc_title="Collab Notes",
        source_name="Notes",
        url="file://notes.txt",
        graph=graph,
        vector_store=FakeVectorStore(),
    )

    # Check that Alice and Bob have _property_origins recorded
    alice_node = graph.nodes["person::alice"]
    origins = json.loads(alice_node["props"]["_property_origins"])
    assert "name" in origins
    assert origins["name"]["chunk_id"] == chunk_id
    assert origins["name"]["status"] == "active"
    assert origins["name"]["argument_id"] is not None

    # Check relationship has chunk_id and argument_id
    collab_rel = next(r for r in graph.rels if r["type"] == "COLLABORATES_WITH")
    assert collab_rel["props"]["chunk_id"] == chunk_id
    assert collab_rel["props"]["status"] == "active"
    assert collab_rel["props"]["argument_id"] is not None
    assert collab_rel["props"]["subject_arg_id"] == origins["name"]["argument_id"]



class InMemoryProvenanceGraph(Graph):
    def __init__(self):
        self.nodes = {}  # id -> {"label": str, "labels": set, "props": dict}
        self.rels = []   # list of dicts: {"start_id", "start_label", "type", "end_id", "end_label", "props"}
        self.calls = []

    def upsert_node(self, label, key, props=None):
        node_id = key.get("id")
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "label": label,
                "labels": {label},
                "props": {**key, **(props or {})},
            }
        else:
            self.nodes[node_id]["props"].update(props or {})
        return [{"n": self.nodes[node_id]["props"]}]

    def upsert_rel(self, start_label, start_key, rel_type, end_label, end_key, props=None):
        s_id = start_key.get("id")
        e_id = end_key.get("id")
        rel_entry = next((r for r in self.rels if r["start_id"] == s_id and r["end_id"] == e_id and r["type"] == rel_type), None)
        if rel_entry:
            rel_entry["props"].update(props or {})
        else:
            rel_entry = {
                "start_id": s_id,
                "start_label": start_label,
                "type": rel_type,
                "end_id": e_id,
                "end_label": end_label,
                "props": dict(props or {}),
            }
            self.rels.append(rel_entry)
        return [{"r": rel_entry["props"]}]

    def mark_deprecated(self, match_clause: str, params: dict | None = None,
                        superseded_by_key: dict | None = None,
                        superseded_by_label: str | None = None):
        node_id = (params or {}).get("id")
        if node_id in self.nodes:
            self.nodes[node_id]["labels"].add("Deprecated")
        return []

    def run(self, query, **params):
        self.calls.append((query, params))
        q = query.replace("`", "")
        # Handle MATCH (n {id: $id}) RETURN n, labels(n) AS labels
        if "RETURN n, labels(n) AS labels" in q:
            node_id = params.get("id")
            if node_id in self.nodes:
                n = self.nodes[node_id]
                return [{"n": dict(n["props"]), "labels": list(n["labels"])}]
            return []
        # Handle MATCH (n:`{label}` {{id: $id}}) RETURN n._property_origins AS origins
        if "RETURN n._property_origins AS origins" in q:
            node_id = params.get("id")
            if node_id in self.nodes:
                return [{"origins": self.nodes[node_id]["props"].get("_property_origins")}]
            return []
        # Handle Argument queries: MATCH (arg:Argument {chunk_id: $chunk_id, candidate_id: $old_id}) RETURN arg
        if "MATCH (arg:Argument {chunk_id: $chunk_id" in q:
            chunk_id = params.get("chunk_id")
            cand_id = params.get("old_id") or params.get("new_id")
            for n in self.nodes.values():
                if "Argument" in n["labels"] and n["props"].get("chunk_id") == chunk_id and n["props"].get("candidate_id") == cand_id:
                    return [{"arg": dict(n["props"])}]
            return []
        # Handle SET arg.status = ...
        if "MATCH (arg:Argument {id: $id}) SET" in q:
            arg_id = params.get("id")
            if arg_id in self.nodes:
                if "status = 'out'" in q:
                    self.nodes[arg_id]["props"]["status"] = "out"
                elif "status = 'in'" in q:
                    self.nodes[arg_id]["props"]["status"] = "in"
                if "ts" in params:
                    self.nodes[arg_id]["props"]["updated_at"] = params["ts"]
            return []
        # Handle MATCH (arg:Argument {id: $id})-[:CONCLUDES]->(c:Claim) SET c.status
        if "MATCH (arg:Argument {id: $id})-[:CONCLUDES]->(c:Claim) SET c.status" in q:
            arg_id = params.get("id")
            new_status = "superseded" if "superseded" in q else "active"
            for r in self.rels:
                if r["start_id"] == arg_id and r["type"] == "CONCLUDES":
                    claim_id = r["end_id"]
                    if claim_id in self.nodes:
                        self.nodes[claim_id]["props"]["status"] = new_status
            return []
        # Handle outgoing rels: MATCH (a {id: $old_id})-[r]->(b) WHERE r.chunk_id = $chunk_id
        if "MATCH (a {id: $old_id})-[r]->(b) WHERE r.chunk_id = $chunk_id" in q:
            old_id = params.get("old_id")
            chunk_id = params.get("chunk_id")
            res = []
            for r in self.rels:
                if r["start_id"] == old_id and r["props"].get("chunk_id") == chunk_id and r["props"].get("status", "active") == "active":
                    s_node = self.nodes.get(r["start_id"], {"labels": [r["start_label"]], "props": {"id": r["start_id"]}})
                    o_node = self.nodes.get(r["end_id"], {"labels": [r["end_label"]], "props": {"id": r["end_id"]}})
                    res.append({
                        "s_labels": list(s_node["labels"]),
                        "s_props": dict(s_node["props"]),
                        "rel_type": r["type"],
                        "r_props": dict(r["props"]),
                        "o_labels": list(o_node["labels"]),
                        "o_props": dict(o_node["props"]),
                    })
            return res
        # Handle incoming rels: MATCH (a)-[r]->(b {id: $old_id}) WHERE r.chunk_id = $chunk_id
        if "MATCH (a)-[r]->(b {id: $old_id}) WHERE r.chunk_id = $chunk_id" in query:
            old_id = params.get("old_id")
            chunk_id = params.get("chunk_id")
            res = []
            for r in self.rels:
                if r["end_id"] == old_id and r["props"].get("chunk_id") == chunk_id and r["props"].get("status", "active") == "active":
                    s_node = self.nodes.get(r["start_id"], {"labels": [r["start_label"]], "props": {"id": r["start_id"]}})
                    o_node = self.nodes.get(r["end_id"], {"labels": [r["end_label"]], "props": {"id": r["end_id"]}})
                    res.append({
                        "s_labels": list(s_node["labels"]),
                        "s_props": dict(s_node["props"]),
                        "rel_type": r["type"],
                        "r_props": dict(r["props"]),
                        "o_labels": list(o_node["labels"]),
                        "o_props": dict(o_node["props"]),
                    })
            return res
        # Handle SET r.status = 'superseded', r.superseded_by = ...
        if "SET r.status = 'superseded'" in query:
            old_id = params.get("old_id")
            new_rel_id = params.get("new_rel_id")
            ts = params.get("ts")
            chunk_id = params.get("chunk_id")
            for r in self.rels:
                if (r["start_id"] == old_id or r["end_id"] == old_id) and r["props"].get("chunk_id") == chunk_id:
                    r["props"]["status"] = "superseded"
                    r["props"]["superseded_by"] = new_rel_id
                    r["props"]["superseded_at"] = ts
            return []
        # Handle SET n._property_origins = $origins
        if "SET n._property_origins = $origins" in query:
            node_id = params.get("id")
            if node_id in self.nodes:
                self.nodes[node_id]["props"]["_property_origins"] = params.get("origins")
            return []
        # Handle rollback: SET n.`{k}` = null ..., n._property_origins = $rem_origins
        if "n._property_origins = $rem_origins" in query:
            node_id = params.get("id")
            if node_id in self.nodes:
                self.nodes[node_id]["props"]["_property_origins"] = params.get("rem_origins")
                for k in list(self.nodes[node_id]["props"].keys()):
                    if f"n.`{k}` = null" in query:
                        del self.nodes[node_id]["props"][k]
            return []
        # Handle DELETE MENTIONS rel
        if "DELETE r" in query and "MENTIONS" in query:
            old_id = params.get("old_id")
            chunk_id = params.get("chunk_id")
            self.rels = [r for r in self.rels if not (r["start_id"] == chunk_id and r["end_id"] == old_id and r["type"] == "MENTIONS")]
            return []
        # Handle DELETE EXTRACTED_FROM rel
        if "DELETE r" in query and "EXTRACTED_FROM" in query:
            old_id = params.get("old_id")
            chunk_id = params.get("chunk_id")
            self.rels = [r for r in self.rels if not (r["start_id"] == old_id and r["end_id"] == chunk_id and r["type"] == "EXTRACTED_FROM")]
            return []
        # Handle active edges count
        if "RETURN count(r) AS c" in query:
            node_id = params.get("id")
            c = sum(1 for r in self.rels if (r["start_id"] == node_id or r["end_id"] == node_id) and r["props"].get("status", "active") == "active" and r["type"] not in ["SUPERSEDES", "SUPERSEDED_BY", "MENTIONS", "EXTRACTED_FROM"])
            return [{"c": c}]
        # Handle SET n:Deprecated
        if "SET n:Deprecated" in query:
            node_id = params.get("id")
            if node_id in self.nodes:
                self.nodes[node_id]["labels"].add("Deprecated")
            return []
        # Handle trace_relationship_history: MATCH (a)-[r {id: $id}]->(b) RETURN a, r, type(r) AS predicate, b
        if "MATCH (a)-[r {id: $id}]->(b)" in query:
            rel_id = params.get("id")
            for r in self.rels:
                if r["props"].get("id") == rel_id:
                    s_node = self.nodes.get(r["start_id"], {"props": {"id": r["start_id"]}})
                    o_node = self.nodes.get(r["end_id"], {"props": {"id": r["end_id"]}})
                    return [{"a": dict(s_node["props"]), "r": dict(r["props"]), "predicate": r["type"], "b": dict(o_node["props"])}]
            return []
        # Handle MATCH ()-[r {id: $id}]->() RETURN
        if "MATCH ()-[r {id: $id}]->() RETURN" in q:
            rel_id = params.get("id")
            for r in self.rels:
                if r["props"].get("id") == rel_id:
                    return [{"r": dict(r["props"])}]
            return []
        # Handle trace_argument_lineage: MATCH (arg:Argument {id: $id}) RETURN arg
        if "MATCH (arg:Argument {id: $id}) RETURN arg" in q:
            arg_id = params.get("id")
            if arg_id in self.nodes:
                return [{"arg": dict(self.nodes[arg_id]["props"])}]
            return []
        # Handle attacks and claims for argument lineage
        if "MATCH (arg:Argument {id: $id})-[:CONCLUDES]->(c:Claim) RETURN c" in q:
            arg_id = params.get("id")
            res = []
            for r in self.rels:
                if r["start_id"] == arg_id and r["type"] == "CONCLUDES":
                    if r["end_id"] in self.nodes:
                        res.append({"c": dict(self.nodes[r["end_id"]]["props"])})
            return res
        if "MATCH (arg:Argument {id: $id})-[r:ATTACKS]->(target:Argument) RETURN r, target" in q:
            arg_id = params.get("id")
            res = []
            for r in self.rels:
                if r["start_id"] == arg_id and r["type"] == "ATTACKS":
                    if r["end_id"] in self.nodes:
                        res.append({"target": dict(self.nodes[r["end_id"]]["props"]), "r": dict(r["props"])})
            return res
        if "MATCH (attacker:Argument)-[r:ATTACKS]->(arg:Argument {id: $id}) RETURN attacker, r" in q:
            arg_id = params.get("id")
            res = []
            for r in self.rels:
                if r["end_id"] == arg_id and r["type"] == "ATTACKS":
                    if r["start_id"] in self.nodes:
                        res.append({"attacker": dict(self.nodes[r["start_id"]]["props"]), "r": dict(r["props"])})
            return res

        return []


def test_record_resolution_arguments_persists_top_k_candidates_and_attacks(monkeypatch):
    monkeypatch.setattr("knowledge_bases.ontology.is_known_label", lambda l: True)
    graph = InMemoryProvenanceGraph()

    candidates = [
        {"id": "p1", "label": "Person", "text": "Tom", "score": 0.98, "is_new": False, "status": "in"},
        {"id": "p2", "label": "Person", "text": "Tom Cat", "score": 0.88, "is_new": False, "status": "out"},
        {"id": "p_new", "label": "Person", "text": "Tom", "score": 0.5, "is_new": True, "status": "out"},
    ]

    win_arg = graph.record_resolution_arguments(
        chunk_id="chunk::1",
        entity_label="Person",
        surface_text="Tom",
        candidates=candidates,
        ref_id="p1_ref",
    )

    assert win_arg is not None
    # Verify claims created
    claims = [n for n in graph.nodes.values() if n["label"] == "Claim"]
    assert len(claims) == 3
    active_claim = next(c for c in claims if c["props"]["status"] == "active")
    assert "Tom" in active_claim["props"]["statement"]
    assert active_claim["props"]["claim_type"] == "derivation"

    # Verify arguments created
    arguments = [n for n in graph.nodes.values() if n["label"] == "Argument"]
    assert len(arguments) == 3
    winner_arg = next(a for a in arguments if a["props"]["status"] == "in")
    assert winner_arg["props"]["candidate_id"] == "p1"

    # Verify attacks edges from winner to runner-ups
    attacks = [r for r in graph.rels if r["type"] == "ATTACKS"]
    assert len(attacks) == 2
    for att in attacks:
        assert att["start_id"] == win_arg
        assert att["props"]["target"] == "conclusion"
        assert att["props"]["reason"] == "higher_confidence"


def test_repoint_entity_reference_rewires_edges_rolls_back_properties_and_deprecates(monkeypatch):
    monkeypatch.setattr("knowledge_bases.ontology.is_known_label", lambda l: True)
    graph = InMemoryProvenanceGraph()

    chunk_id = "chunk::10"
    old_id = "person::tom_old"
    new_id = "person::tom_new"
    proj_id = "proj::phoenix"

    # Set up initial state
    origins = {
        "name": {"argument_id": "arg::old", "chunk_id": chunk_id, "status": "active"},
        "status": {"argument_id": "arg::old", "chunk_id": chunk_id, "status": "active"},
    }
    graph.upsert_node("Person", {"id": old_id}, {
        "name": "Tom",
        "status": "blocked",
        "_property_origins": json.dumps(origins),
    })
    graph.upsert_node("Person", {"id": new_id}, {"id": new_id})
    graph.upsert_node("Project", {"id": proj_id}, {"name": "Phoenix"})
    graph.upsert_node("Chunk", {"id": chunk_id}, {"text": "Tom is blocked on Phoenix."})

    # Arguments: old was winner, new was runner-up
    graph.upsert_node("Argument", {"id": "arg::old"}, {"chunk_id": chunk_id, "candidate_id": old_id, "status": "in"})
    graph.upsert_node("Argument", {"id": "arg::runner_up"}, {"chunk_id": chunk_id, "candidate_id": new_id, "status": "out"})
    graph.upsert_node("Claim", {"id": "claim::old"}, {"status": "active"})
    graph.upsert_node("Claim", {"id": "claim::runner_up"}, {"status": "superseded"})
    graph.upsert_rel("Argument", {"id": "arg::old"}, "CONCLUDES", "Claim", {"id": "claim::old"})
    graph.upsert_rel("Argument", {"id": "arg::runner_up"}, "CONCLUDES", "Claim", {"id": "claim::runner_up"})

    # Active relationship from old entity
    graph.upsert_rel(
        "Person", {"id": old_id},
        "WORKS_ON",
        "Project", {"id": proj_id},
        props={"id": "rel::old", "status": "active", "chunk_id": chunk_id},
    )
    graph.upsert_rel("Person", {"id": old_id}, "EXTRACTED_FROM", "Chunk", {"id": chunk_id}, {"role": "extracted"})
    graph.upsert_rel("Chunk", {"id": chunk_id}, "MENTIONS", "Person", {"id": old_id}, {"id": "mention::old"})

    # Execute repoint
    result = graph.repoint_entity_reference(
        chunk_id=chunk_id,
        old_entity_id=old_id,
        new_entity_id=new_id,
        reason="human_feedback",
    )

    assert result["status"] == "repointed"
    assert "name" in result["transferred_properties"]
    assert "status" in result["transferred_properties"]

    # 1. Runner-up promoted to "in", old set to "out"
    assert graph.nodes["arg::runner_up"]["props"]["status"] == "in"
    assert graph.nodes["arg::old"]["props"]["status"] == "out"
    assert graph.nodes["claim::runner_up"]["props"]["status"] == "active"
    assert graph.nodes["claim::old"]["props"]["status"] == "superseded"

    # 2. Attack relations created
    attacks_to_old = [r for r in graph.rels if r["type"] == "ATTACKS" and r["end_id"] == "arg::old"]
    assert len(attacks_to_old) >= 1

    # 3. Old relationship marked superseded with link to new
    old_rel = next(r for r in graph.rels if r["props"].get("id") == "rel::old")
    assert old_rel["props"]["status"] == "superseded"
    assert "superseded_by" in old_rel["props"]
    assert old_rel["props"]["superseded_at"] is not None
    new_rel_id = old_rel["props"]["superseded_by"]

    # 4. New active relationship exists from new_id to proj_id
    # The chunk must not keep pointing at the rejected entity.
    old_mentions = [r for r in graph.rels if r["type"] == "MENTIONS" and r["end_id"] == old_id]
    assert old_mentions == []
    new_mentions = [r for r in graph.rels if r["type"] == "MENTIONS" and r["end_id"] == new_id]
    assert new_mentions, "the corrected mention was not rewired to the new entity"

    new_rel = next(r for r in graph.rels if r["props"].get("id") == new_rel_id)
    assert new_rel["start_id"] == new_id
    assert new_rel["end_id"] == proj_id
    assert new_rel["props"]["status"] == "active"
    assert new_rel["props"]["supersedes"] == "rel::old"

    # 5. Properties transferred from old to new
    assert graph.nodes[new_id]["props"]["name"] == "Tom"
    assert graph.nodes[new_id]["props"]["status"] == "blocked"
    new_origins = json.loads(graph.nodes[new_id]["props"]["_property_origins"])
    assert "name" in new_origins
    assert "status" in new_origins

    assert "name" not in graph.nodes[old_id]["props"]
    assert "status" not in graph.nodes[old_id]["props"]

    # 6. EXTRACTED_FROM rewired
    assert any(r for r in graph.rels if r["start_id"] == new_id and r["end_id"] == chunk_id and r["type"] == "EXTRACTED_FROM")
    assert not any(r for r in graph.rels if r["start_id"] == old_id and r["end_id"] == chunk_id and r["type"] == "EXTRACTED_FROM")

    # 7. Old entity has no remaining edges/origins -> Deprecated
    assert "Deprecated" in graph.nodes[old_id]["labels"]


def test_trace_relationship_history_traverses_supersedes_chain(monkeypatch):
    monkeypatch.setattr("knowledge_bases.ontology.is_known_label", lambda l: True)
    graph = InMemoryProvenanceGraph()

    # Create v1 and v2 edges
    graph.upsert_node("Person", {"id": "p1"}, {"name": "P1"})
    graph.upsert_node("Person", {"id": "p2"}, {"name": "P2"})
    graph.upsert_node("Project", {"id": "proj"}, {"name": "Phoenix"})

    graph.upsert_rel(
        "Person", {"id": "p1"}, "WORKS_ON", "Project", {"id": "proj"},
        props={"id": "rel::v1", "status": "superseded", "superseded_by": "rel::v2"},
    )
    graph.upsert_rel(
        "Person", {"id": "p2"}, "WORKS_ON", "Project", {"id": "proj"},
        props={"id": "rel::v2", "status": "active", "supersedes": "rel::v1"},
    )

    history = graph.trace_relationship_history("rel::v2")
    assert history["found"] is not False
    assert len(history["lineage"]) == 2
    assert history["lineage"][0]["id"] == "rel::v1"
    assert history["lineage"][1]["id"] == "rel::v2"

