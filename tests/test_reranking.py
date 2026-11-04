from query import search_claims
from knowledge_bases.semantic_layer.models import Metric
from knowledge_bases.semantic_layer.metric_registry import (
    resolve_or_create_business_term,
)


class FakeLLM:
    def __init__(self, preferred_text: str):
        self.preferred_text = preferred_text

    def embed(self, text: str) -> list[float]:
        return [1.0]

    def complete(self, prompt: str, system: str | None = None) -> str:
        return "yes"

    def rerank_candidate_count(self, top_k: int, *, dedupe: bool = False) -> int:
        return 25 if dedupe else 50

    def rerank(self, query: str, candidates: list[dict], text_key: str = "text") -> list[dict]:
        scored = []
        for candidate in candidates:
            updated = dict(candidate)
            updated["rerank_score"] = 10.0 if self.preferred_text in str(candidate.get(text_key) or "") else 0.0
            scored.append(updated)
        return sorted(scored, key=lambda item: item["rerank_score"], reverse=True)

    def extract(self, prompt, model, system=None):
        return model()


class FakeVectorStore:
    def __init__(self, hits: list[dict]):
        self.hits = hits
        self.requested_top_k = None

    def ensure_collections(self):
        return None

    def search_claims(self, embedding: list[float], top_k: int = 5):
        self.requested_top_k = top_k
        return self.hits[:top_k]

    def search_business_terms(self, embedding: list[float], top_k: int = 10):
        self.requested_top_k = top_k
        return self.hits[:top_k]

    def search_vectors(self, collection_name: str, embedding: list[float], top_k: int = 5):
        self.requested_top_k = top_k
        return self.hits[:top_k]


class FakeGraph:
    def __init__(self, rows_by_id: dict[str, dict] | None = None):
        self.rows_by_id = rows_by_id or {}

    def run(self, query: str, **params):
        if "claim_ids" in params:
            return [self.rows_by_id[item] for item in params["claim_ids"] if item in self.rows_by_id]
        if "MATCH (t:BusinessTerm)" in query:
            return []
        if "id" in params and params["id"] in self.rows_by_id:
            return [self.rows_by_id[params["id"]]]
        return []

    def upsert_node(self, *args, **kwargs):
        return []


def test_search_claims_reranks_expanded_vector_candidates(monkeypatch):
    llm = FakeLLM("best")
    monkeypatch.setattr("llm.get_llm", lambda: llm)
    hits = [
        {"id": "i1", "similarity": 0.99},
        {"id": "i2", "similarity": 0.80},
    ]
    graph = FakeGraph({
        "i1": {"id": "i1", "statement": "okay match", "weight": 1, "source_doc": "doc"},
        "i2": {"id": "i2", "statement": "best match", "weight": 1, "source_doc": "doc"},
    })
    vector_store = FakeVectorStore(hits)

    results = search_claims("query", top_k=1, graph=graph, vector_store=vector_store)

    assert vector_store.requested_top_k == 50
    assert results[0]["id"] == "i2"
    assert results[0]["rerank_score"] == 10.0


def test_business_term_resolution_uses_reranked_candidate_first(monkeypatch):
    monkeypatch.setattr(
        "knowledge_bases.semantic_layer.metric_registry._BUSINESS_TERMS_INDEXED",
        True,
    )
    llm = FakeLLM("best")
    graph = FakeGraph()
    vector_store = FakeVectorStore([
        {"id": "bt1", "name": "Wrong Term", "embedding_text": "okay", "similarity": 0.99},
        {"id": "bt2", "name": "Best Term", "embedding_text": "best", "similarity": 0.94},
    ])
    metric = Metric(id="m1", name="Best", definition="best metric")

    result = resolve_or_create_business_term(metric, graph, vector_store, llm=llm, semantic={})

    assert vector_store.requested_top_k == 25
    assert result["name"] == "Best Term"
    assert result["method"] == "vector"


def test_fuzzy_identity_uses_reranked_candidate_first():
    import sys
    import types

    pymilvus = types.ModuleType("pymilvus")
    pymilvus.MilvusClient = object
    pymilvus.DataType = object
    sys.modules.setdefault("pymilvus", pymilvus)

    from extract import _resolve_fuzzy_identity

    llm = FakeLLM("best")
    graph = FakeGraph({
        "n1": {"text": "okay entity"},
        "n2": {"text": "best entity"},
    })
    vector_store = FakeVectorStore([
        {"id": "n1", "similarity": 0.99},
        {"id": "n2", "similarity": 0.94},
    ])

    resolved_id, is_existing, _ = _resolve_fuzzy_identity(
        graph, vector_store, llm, "Claim", "claims", "new entity", "claim", "statement"
    )

    assert vector_store.requested_top_k == 25
    assert resolved_id == "n2"
    assert is_existing is True
