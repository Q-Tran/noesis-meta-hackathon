"""Entity resolution: id, then name, then meaning.

The layering is the design. Milvus is an accelerator, never a dependency, so
every test that involves it has a counterpart proving the same call still works
when it is gone -- that is the property that lets callers omit their own
fallback.
"""
from __future__ import annotations

import pytest

from knowledge_bases.orchestration.resolution import (
    ARBITER_SIMILARITY,
    AUTO_MERGE_SIMILARITY,
    SEMANTIC_COLLECTIONS,
    Resolution,
    resolve_by_embedding,
    resolve_by_name,
    resolve_entity,
)


class FakeGraph:
    """Answers id lookups, name lookups, and text lookups from dicts."""

    def __init__(self, by_id=(), by_name=None, texts=None):
        self.by_id = set(by_id)
        self.by_name = by_name or {}
        self.texts = texts or {}
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))
        if "RETURN n.id AS id LIMIT 1" in query:
            return [{"id": params["id"]}] if params.get("id") in self.by_id else []
        if "WHERE n.name = $name" in query:
            for name, node_id in self.by_name.items():
                if name.lower() == str(params.get("name", "")).lower():
                    return [{"id": node_id, "name": name}]
            return []
        if "AS text" in query:
            text = self.texts.get(params.get("id"))
            return [{"text": text}] if text else []
        return []


class FakeLLM:
    def __init__(self, arbiter_answer="No", fail_on=None):
        self.arbiter_answer = arbiter_answer
        self.fail_on = fail_on or set()
        self.completions = []

    def embed(self, text):
        if "embed" in self.fail_on:
            raise RuntimeError("no local embedding model")
        return [0.1, 0.2, 0.3]

    def rerank_candidate_count(self, top_k, dedupe=False):
        return 25

    def rerank(self, query, candidates, text_key="text"):
        if "rerank" in self.fail_on:
            raise RuntimeError("reranker unavailable")
        return candidates

    def complete(self, prompt):
        self.completions.append(prompt)
        if "complete" in self.fail_on:
            raise RuntimeError("llm down")
        return self.arbiter_answer


class FakeVectorStore:
    def __init__(self, hits=(), fail=False):
        self.hits = list(hits)
        self.fail = fail

    def search_vectors(self, collection, embedding, top_k=5):
        if self.fail:
            raise RuntimeError("Fail connecting to server on localhost:19530")
        return self.hits


# -- layering ----------------------------------------------------------------------

def test_an_exact_id_wins_without_touching_the_vector_store():
    """Cheapest and most certain first; a store that would raise proves it is
    never consulted."""
    resolution = resolve_entity(
        FakeGraph(by_id={"core"}), "Project", "core", "Core",
        vector_store=FakeVectorStore(fail=True), llm=FakeLLM(),
    )

    assert resolution == Resolution("core", "id")


def test_a_name_match_wins_before_embeddings():
    resolution = resolve_entity(
        FakeGraph(by_name={"Routing": "routing"}), "Project", "guessed-id", "routing",
        vector_store=FakeVectorStore(fail=True), llm=FakeLLM(),
    )

    assert resolution.entity_id == "routing"
    assert resolution.method == "name"


def test_an_exact_name_is_preferred_over_a_case_insensitive_one():
    """Otherwise the winner depends on storage order."""
    graph = FakeGraph(by_name={"Routing": "routing"})
    resolve_by_name(graph, "Project", "Routing")

    assert "ORDER BY CASE WHEN n.name = $name THEN 0 ELSE 1 END" in graph.queries[0][0]


# -- semantic ----------------------------------------------------------------------

def test_a_paraphrase_resolves_by_embedding_when_name_matching_misses():
    """The case the deterministic path cannot see: different words, same thing."""
    graph = FakeGraph(texts={"vsf": "Vector Store Fallback"})
    resolution = resolve_by_embedding(
        graph, "Project", "the fallback path for embeddings",
        vector_store=FakeVectorStore([{"id": "vsf", "similarity": 0.97}]),
        llm=FakeLLM(),
    )

    assert resolution.entity_id == "vsf"
    assert resolution.method == "embedding"
    assert resolution.similarity == 0.97


def test_a_near_miss_asks_the_arbiter_rather_than_guessing():
    """Similar wording is not the same referent: 'Q3 revenue target' and 'Q4
    revenue target' score alike."""
    llm = FakeLLM(arbiter_answer="Yes")
    resolution = resolve_by_embedding(
        FakeGraph(texts={"q3": "Q3 revenue target"}), "Project", "Q4 revenue target",
        vector_store=FakeVectorStore([{"id": "q3", "similarity": 0.88}]), llm=llm,
    )

    assert resolution.method == "arbiter"
    assert llm.completions, "the arbiter should have been consulted"


def test_the_arbiter_saying_no_leaves_it_unresolved():
    resolution = resolve_by_embedding(
        FakeGraph(texts={"q3": "Q3 revenue target"}), "Project", "Q4 revenue target",
        vector_store=FakeVectorStore([{"id": "q3", "similarity": 0.88}]),
        llm=FakeLLM(arbiter_answer="No"),
    )

    assert not resolution.found


def test_a_weak_similarity_never_matches():
    resolution = resolve_by_embedding(
        FakeGraph(texts={"x": "something else"}), "Project", "unrelated thing",
        vector_store=FakeVectorStore([{"id": "x", "similarity": 0.4}]), llm=FakeLLM(),
    )

    assert not resolution.found


def test_thresholds_match_the_ingestion_path():
    """Two thresholds for one question in two places would be two things to
    keep in sync; `extract.py` uses these numbers for document ingestion."""
    import extract

    source = extract.__file__ and open(extract.__file__).read()
    assert f"{AUTO_MERGE_SIMILARITY}" in source
    assert f"{ARBITER_SIMILARITY}" in source


# -- degradation -------------------------------------------------------------------

def test_an_unreachable_milvus_resolves_to_nothing_rather_than_raising():
    """The property that lets callers skip their own fallback."""
    resolution = resolve_by_embedding(
        FakeGraph(), "Project", "anything",
        vector_store=FakeVectorStore(fail=True), llm=FakeLLM(),
    )

    assert resolution == Resolution(None, "none")


def test_a_missing_embedding_model_resolves_to_nothing_rather_than_raising():
    resolution = resolve_by_embedding(
        FakeGraph(), "Project", "anything",
        vector_store=FakeVectorStore(), llm=FakeLLM(fail_on={"embed"}),
    )

    assert not resolution.found


def test_a_broken_reranker_still_uses_the_raw_similarities():
    """Reranking improves ordering; losing it should not lose the match."""
    resolution = resolve_by_embedding(
        FakeGraph(texts={"vsf": "Vector Store Fallback"}), "Project", "fallback path",
        vector_store=FakeVectorStore([{"id": "vsf", "similarity": 0.99}]),
        llm=FakeLLM(fail_on={"rerank"}),
    )

    assert resolution.entity_id == "vsf"


def test_a_failed_arbiter_call_does_not_assert_a_match():
    """An unanswered question is not a yes."""
    resolution = resolve_by_embedding(
        FakeGraph(texts={"q3": "Q3 revenue target"}), "Project", "Q4 revenue target",
        vector_store=FakeVectorStore([{"id": "q3", "similarity": 0.88}]),
        llm=FakeLLM(fail_on={"complete"}),
    )

    assert not resolution.found


def test_semantic_resolution_can_be_turned_off():
    resolution = resolve_entity(
        FakeGraph(), "Project", "nope", "nothing like it", semantic=False,
        vector_store=FakeVectorStore(fail=True), llm=FakeLLM(),
    )

    assert not resolution.found


def test_a_label_without_a_collection_stops_at_name_matching():
    """`CodeRepository` has no vector collection, so there is nothing to
    search -- searching one repo *name* semantically is meaningless."""
    assert "CodeRepository" not in SEMANTIC_COLLECTIONS

    resolution = resolve_by_embedding(
        FakeGraph(), "CodeRepository", "anything",
        vector_store=FakeVectorStore(fail=True), llm=FakeLLM(),
    )

    assert not resolution.found


def test_a_hit_whose_node_has_no_text_is_skipped():
    """A stale vector pointing at a deleted node must not resolve."""
    resolution = resolve_by_embedding(
        FakeGraph(texts={}), "Project", "anything",
        vector_store=FakeVectorStore([{"id": "ghost", "similarity": 0.99}]),
        llm=FakeLLM(),
    )

    assert not resolution.found
