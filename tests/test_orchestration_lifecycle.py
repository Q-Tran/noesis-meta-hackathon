"""Unit tests for the generic, cross-domain deprecation/reactivation
commands (`knowledge_bases.orchestration.lifecycle`), which implement the
write side of `docs/architecture/lifecycle.md`'s "never delete, tag
instead" policy behind the `WriteProposal` propose -> confirm -> execute
gate. Exercised without a live Memgraph connection using the same recording
fake convention as `tests/test_lifecycle.py`/`tests/test_provenance.py`."""
import pytest

from knowledge_bases.orchestration.lifecycle import (
    DeprecateEntityRequest,
    ReactivateEntityRequest,
    _match_clause,
    deprecate_entity,
    preview_deprecation,
    reactivate_entity,
)


class RecordingGraph:
    def __init__(self, rows=None):
        self.calls = []
        self._rows = rows if rows is not None else [
            {"labels": ["Decision"], "properties": {"id": "d-1"}},
        ]
        self.mark_deprecated_calls = []
        self.reactivate_calls = []

    def run(self, query, **params):
        self.calls.append((query, params))
        if "count(DISTINCT n)" in query:
            return [{"affected": len(self._rows)}]
        return list(self._rows)

    def mark_deprecated(self, match_clause, params, superseded_by_key=None,
                        superseded_by_label=None):
        self.mark_deprecated_calls.append(
            (match_clause, params, superseded_by_key, superseded_by_label)
        )
        return []

    def reactivate(self, match_clause, params):
        self.reactivate_calls.append((match_clause, params))
        return list(self._rows)


class FakeVectorStore:
    def __init__(self):
        self.deleted = []

    def delete_vector(self, collection, id):
        self.deleted.append((collection, id))


# --- request validation --------------------------------------------------

def test_deprecate_request_rejects_unsupported_knowledge_base():
    with pytest.raises(ValueError, match="does not support deprecation"):
        DeprecateEntityRequest(knowledge_base="methodology", label="BmadEntity", key={"id": "x"})


def test_deprecate_request_rejects_label_not_owned_by_knowledge_base():
    with pytest.raises(ValueError, match="not owned by"):
        DeprecateEntityRequest(
            knowledge_base="project_management", label="BusinessTerm", key={"id": "x"},
        )


def test_deprecate_request_rejects_cascade_relationship_not_owned():
    with pytest.raises(ValueError, match="not exposed by"):
        DeprecateEntityRequest(
            knowledge_base="project_management", label="Task", key={"id": "x"}, cascade=["ALIAS_OF"],
        )


def test_deprecate_request_requires_a_nonempty_key():
    with pytest.raises(ValueError, match="key must identify"):
        DeprecateEntityRequest(knowledge_base="project_management", label="Task", key={})


def test_deprecate_request_requires_cascade_depth_when_cascade_given():
    with pytest.raises(ValueError, match="cascade_depth must be >= 1"):
        DeprecateEntityRequest(
            knowledge_base="project_management", label="Project", key={"id": "p-1"},
            cascade=["HAS_EPIC"], cascade_depth=0,
        )


def test_deprecate_request_validates_superseded_by_ownership():
    with pytest.raises(ValueError, match="not owned by"):
        DeprecateEntityRequest(
            knowledge_base="strategy", label="Decision", key={"id": "d-1"},
            superseded_by={"label": "Task", "key": {"id": "t-1"}},
        )


def test_deprecate_request_accepts_valid_superseded_by():
    request = DeprecateEntityRequest(
        knowledge_base="strategy", label="Decision", key={"id": "d-1"},
        superseded_by={"label": "Decision", "key": {"id": "d-2"}},
    )
    assert request.superseded_by.label == "Decision"


def test_reactivate_request_shares_the_same_ownership_rules():
    with pytest.raises(ValueError, match="does not support deprecation"):
        ReactivateEntityRequest(knowledge_base="tools", label="Tool", key={"key": "s-1"})


# --- match clause construction -------------------------------------------

def test_match_clause_without_cascade_matches_only_the_root():
    clause, params = _match_clause("Decision", {"id": "d-1"}, [], 0)
    assert clause == "MATCH (n:`Decision` {`id`: $key_id})"
    assert params == {"key_id": "d-1"}


def test_match_clause_with_cascade_walks_undirected_variable_length():
    clause, params = _match_clause(
        "Project", {"id": "p-1"}, ["HAS_EPIC", "HAS_STORY", "HAS_TASK"], 10,
    )
    assert clause == (
        "MATCH (root:`Project` {`id`: $key_id}) "
        "MATCH (root)-[:`HAS_EPIC`|`HAS_STORY`|`HAS_TASK`*0..10]-(n)"
    )
    assert params == {"key_id": "p-1"}


# --- preview / deprecate / reactivate ------------------------------------

def test_preview_deprecation_is_read_only_and_reports_affected_count():
    graph = RecordingGraph()
    request = DeprecateEntityRequest(
        knowledge_base="strategy", label="Decision", key={"id": "d-1"},
    )

    preview = preview_deprecation(graph, request)

    assert preview["affected_count"] == 1
    assert preview["sample"][0]["properties"]["id"] == "d-1"
    assert preview["sample_truncated"] is False
    assert not graph.mark_deprecated_calls


def test_preview_deprecation_marks_truncation_past_the_sample_limit():
    rows = [
        {"labels": ["Task"], "properties": {"id": f"t-{i}"}} for i in range(25)
    ]
    graph = RecordingGraph(rows=rows)
    request = DeprecateEntityRequest(
        knowledge_base="project_management", label="Project", key={"id": "p-1"},
        cascade=["HAS_TASK"], cascade_depth=5,
    )

    preview = preview_deprecation(graph, request)

    assert len(preview["sample"]) == 20
    assert preview["sample_truncated"] is True


def test_deprecate_entity_calls_mark_deprecated_with_the_built_clause():
    graph = RecordingGraph()

    result = deprecate_entity(
        {"knowledge_base": "strategy", "label": "Decision", "key": {"id": "d-1"}},
        graph, vector_store=FakeVectorStore(),
    )

    assert result == {"deprecated_count": 1, "labels": ["Decision"]}
    clause, params, superseded_key, superseded_label = graph.mark_deprecated_calls[0]
    assert clause == "MATCH (n:`Decision` {`id`: $key_id})"
    assert params == {"key_id": "d-1"}
    assert superseded_key is None
    assert superseded_label is None


def test_deprecate_entity_links_superseded_by_when_given():
    graph = RecordingGraph()

    deprecate_entity(
        {
            "knowledge_base": "strategy", "label": "Decision",
            "key": {"id": "d-1"},
            "superseded_by": {"label": "Decision", "key": {"id": "d-2"}},
        },
        graph, vector_store=FakeVectorStore(),
    )

    _, _, superseded_key, superseded_label = graph.mark_deprecated_calls[0]
    assert superseded_key == {"id": "d-2"}
    assert superseded_label == "Decision"


def test_deprecate_entity_deletes_vectors_for_labels_with_a_collection():
    graph = RecordingGraph(rows=[{"labels": ["Decision"], "properties": {"id": "d-1"}}])
    vector_store = FakeVectorStore()

    deprecate_entity(
        {"knowledge_base": "strategy", "label": "Decision", "key": {"id": "d-1"}},
        graph, vector_store=vector_store,
    )

    assert vector_store.deleted == [("strategy", "d-1")]


def test_deprecate_entity_skips_vector_cleanup_for_labels_without_a_collection():
    graph = RecordingGraph(rows=[{"labels": ["UserStory"], "properties": {"id": "s-1"}}])
    vector_store = FakeVectorStore()

    deprecate_entity(
        {"knowledge_base": "project_management", "label": "UserStory", "key": {"id": "s-1"}},
        graph, vector_store=vector_store,
    )

    assert vector_store.deleted == []


def test_deprecate_entity_tolerates_vector_store_errors():
    graph = RecordingGraph(rows=[{"labels": ["Decision"], "properties": {"id": "d-1"}}])

    class ExplodingVectorStore:
        def delete_vector(self, collection, id):
            raise RuntimeError("milvus is down")

    result = deprecate_entity(
        {"knowledge_base": "strategy", "label": "Decision", "key": {"id": "d-1"}},
        graph, vector_store=ExplodingVectorStore(),
    )

    assert result == {"deprecated_count": 1, "labels": ["Decision"]}


def test_reactivate_entity_calls_reactivate_with_the_built_clause():
    graph = RecordingGraph()

    result = reactivate_entity(
        {"knowledge_base": "strategy", "label": "Decision", "key": {"id": "d-1"}},
        graph,
    )

    assert result == {"reactivated_count": 1}
    clause, params = graph.reactivate_calls[0]
    assert clause == "MATCH (n:`Decision` {`id`: $key_id})"
    assert params == {"key_id": "d-1"}
