import pytest

from knowledge_bases.orchestration.hybrid_search import hybrid_search


class FakeLlm:
    def embed(self, text):
        return [1.0]

    def rerank_candidate_count(self, limit):
        return 10

    def rerank(self, query, rows, text_key):
        return sorted(rows, key=lambda row: row["score"], reverse=True)


class FakeVectorStore:
    def search_vectors(self, collection, embedding, top_k):
        assert collection == "bmad_entities"
        return [{
            "id": "vector-id", "entity_id": "req-1", "project_id": "project-1",
            "entity_type": "Requirement", "similarity": 0.9,
        }]


class FakeGraph:
    def __init__(self):
        self.fulltext_queries = []

    def run(self, query, **params):
        if "queryNodes" in query:
            self.fulltext_queries.append(params["query_text"])
            return [{
                "labels": ["BmadEntity", "Requirement"],
                "properties": {
                    "id": "req-1", "project_id": "project-1",
                    "key": "REQ-001", "search_text": "Guest checkout must be supported",
                },
                "score": 2.0,
            }, {
                "labels": ["BmadEntity", "TestCase"],
                "properties": {"id": "test-1", "project_id": "project-1"},
                "score": 1.0,
            }]
        assert params["scoped_identities"] == [
            {"identity": "req-1", "project_id": "project-1"},
        ]
        return [{
            "identity": "req-1", "project_id": "project-1", "key": "REQ-001",
            "properties": {"entity_type": "Requirement", "status": "approved"},
            "text": "Guest checkout must be supported",
        }]


def test_hybrid_search_fuses_channels_within_explicit_scope():
    rows = hybrid_search(
        "guest checkout", ["project_management.Requirement"], {"status": "approved"},
        graph=FakeGraph(), vector_store=FakeVectorStore(), llm=FakeLlm(),
    )

    assert len(rows) == 1
    assert rows[0]["qualified_type"] == "project_management.Requirement"
    assert rows[0]["lexical_rank"] == 1
    assert rows[0]["semantic_rank"] == 1


def test_hybrid_search_rejects_cross_family_scope():
    with pytest.raises(ValueError, match="one knowledge family"):
        hybrid_search(
            "checkout", ["project_management.Requirement", "strategy.Goal"],
            graph=FakeGraph(), vector_store=FakeVectorStore(), llm=FakeLlm(),
        )


def test_hybrid_search_requires_an_explicit_scope():
    with pytest.raises(ValueError, match="At least one entity type"):
        hybrid_search("checkout", [], graph=FakeGraph(), llm=FakeLlm())


def test_hybrid_search_rejects_unknown_types():
    with pytest.raises(ValueError, match="Unsupported search entity type"):
        hybrid_search("checkout", ["everything.Anything"], graph=FakeGraph(), llm=FakeLlm())


def test_hybrid_search_property_filters_remove_non_matching_results():
    assert hybrid_search(
        "guest checkout", ["project_management.Requirement"], {"status": "draft"},
        graph=FakeGraph(), vector_store=FakeVectorStore(), llm=FakeLlm(),
    ) == []


def test_hybrid_search_sanitizes_lucene_query_syntax():
    graph = FakeGraph()

    hybrid_search(
        "project: noesis-startup (AI)?", ["project_management.Requirement"],
        graph=graph, vector_store=FakeVectorStore(), llm=FakeLlm(),
    )

    assert graph.fulltext_queries == ["project noesis startup AI"]


class EmptyVectorStore:
    def search_vectors(self, collection, embedding, top_k):
        return []


class DeprecatedAwareGraph:
    """Full-text hit is tagged :Deprecated; hydration query is inspected to
    confirm the exclusion guard is (or isn't) present."""

    def __init__(self):
        self.hydrate_queries = []

    def run(self, query, **params):
        if "queryNodes" in query:
            return [{
                "labels": ["BmadEntity", "Requirement", "Deprecated"],
                "properties": {
                    "id": "req-1", "project_id": "project-1",
                    "key": "REQ-001", "search_text": "Old requirement",
                },
                "score": 2.0,
            }]
        self.hydrate_queries.append(query)
        return [{
            "identity": "req-1", "project_id": "project-1", "key": "REQ-001",
            "properties": {"entity_type": "Requirement"},
            "text": "Old requirement",
        }]


def test_hybrid_search_excludes_deprecated_nodes_by_default():
    graph = DeprecatedAwareGraph()

    rows = hybrid_search(
        "old requirement", ["project_management.Requirement"], graph=graph,
        vector_store=EmptyVectorStore(), llm=FakeLlm(),
    )

    assert rows == []


def test_hybrid_search_includes_deprecated_nodes_when_requested():
    graph = DeprecatedAwareGraph()

    rows = hybrid_search(
        "old requirement", ["project_management.Requirement"], graph=graph,
        vector_store=EmptyVectorStore(), llm=FakeLlm(),
        include_deprecated=True,
    )

    assert len(rows) == 1
    assert rows[0]["qualified_type"] == "project_management.Requirement"
    assert "NOT n:Deprecated" not in graph.hydrate_queries[0]


def test_hydrate_query_excludes_deprecated_nodes_by_default():
    from knowledge_bases.orchestration.hybrid_search import _hydrate, SEARCH_TYPES

    graph = DeprecatedAwareGraph()
    scope = [("project_management.Requirement", SEARCH_TYPES["project_management.Requirement"])]
    candidates = [{"qualified_type": "project_management.Requirement", "identity": "req-1", "project_id": "p"}]

    _hydrate(graph, scope, candidates)

    assert "NOT n:Deprecated" in graph.hydrate_queries[0]


def test_corpus_entity_types_have_a_real_retrieval_channel():
    """Component/Interface used to declare neither fulltext nor vector, and were
    not in semantic_search_kinds: hybrid_search could never return them. Pin the
    channel for the labels the docs corpus actually populates."""
    from knowledge_bases.orchestration.hybrid_search import SEARCH_TYPES
    from knowledge_bases.registry import semantic_search_kinds

    managed = semantic_search_kinds()
    for name in (
        "architecture.Component", "architecture.Interface",
        "project_management.Capability", "project_management.Specification",
        "business_dictionary.BusinessTerm",
    ):
        config = SEARCH_TYPES[name]
        assert (
            config.fulltext_index
            or config.vector_collection
            or (config.include_managed and config.label in managed)
        ), name


def test_metric_search_has_a_vector_collection():
    from knowledge_bases.orchestration.hybrid_search import SEARCH_TYPES

    config = SEARCH_TYPES["semantic_layer.Metric"]
    assert config.fulltext_index == "semantic_layer_text"
    assert config.vector_collection == "metric_entities"
