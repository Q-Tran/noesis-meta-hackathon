"""Session memory: durable, append-only, and ontology-bounded.

Memory never expires, so the tests that matter most are about *correction*
without destruction, and about the memory being unable to widen the ontology.
"""
from __future__ import annotations

import pytest

from knowledge_bases.orchestration.memory import SessionMemoryRepository
from knowledge_bases.orchestration.memory_models import (
    MemoryError_,
    MemoryKind,
    MemoryScope,
    SessionMemory,
    memory_key,
)


class FakeGraph:
    """Minimal graph double: records writes, answers existence checks."""

    def __init__(self, existing_ids=(), named=None):
        self.existing = set(existing_ids)
        # {label: {name: id}} for nodes that exist and can be matched by name.
        self.named = named or {}
        self.queries = []
        self.rels = []
        self.nodes = []
        self.sources = []

    def run(self, query, **params):
        self.queries.append((query, params))
        if "RETURN n.id AS id LIMIT 1" in query and "n.name" not in query:
            return [{"id": params["id"]}] if params.get("id") in self.existing else []
        if "WHERE n.name = $name" in query:
            for label, names in self.named.items():
                if f":`{label}`" in query:
                    for name, node_id in names.items():
                        if name.lower() == str(params.get("name", "")).lower():
                            return [{"id": node_id}]
            return []
        if "RETURN m.key AS key" in query and "MERGE" not in query:
            return [{"key": params.get("key")}]
        return []

    def upsert_node(self, label, key, props=None):
        from knowledge_bases.ontology import require_known_label

        require_known_label(label, actor="test")
        self.nodes.append((label, key, props or {}))
        return []

    def upsert_rel(self, start_label, start_key, rel_type, end_label, end_key, props=None):
        self.rels.append((start_label, start_key, rel_type, end_label, end_key))
        return []

    def attach_source(self, label, key, source_id, source_props=None, role=None):
        self.sources.append(source_id)
        return []


def _memory(**overrides):
    base = dict(
        title="Check the container first",
        body="Start noesis-neo4j before assuming a code bug.",
        kind=MemoryKind.pitfall, scope=MemoryScope.repo, repo="noesis",
    )
    base.update(overrides)
    return SessionMemory(**base)


# -- identity ----------------------------------------------------------------------

def test_the_same_lesson_in_the_same_scope_gets_one_key():
    """Re-recording a reworded lesson must update one node, not accumulate
    near-duplicates that each look authoritative."""
    assert _memory().key == _memory(body="Different wording entirely.").key


def test_the_same_title_in_a_different_scope_is_a_different_memory():
    assert _memory(repo="noesis").key != _memory(repo="other").key


def test_scope_qualifiers_are_part_of_identity():
    a = memory_key("t", "expert", None, "sql_expert")
    b = memory_key("t", "expert", None, "methodology_expert")
    assert a != b


# -- validation --------------------------------------------------------------------

def test_a_memory_with_an_empty_body_is_refused():
    """A titled memory with no content looks present in a roster but teaches
    nothing when opened -- worse than absent, because it stops the search."""
    with pytest.raises(MemoryError_, match="empty body"):
        _memory(body="   ").validate_for_storage()


def test_a_repo_scoped_memory_must_name_its_repo():
    with pytest.raises(MemoryError_, match="repo"):
        _memory(repo=None).validate_for_storage()


def test_an_expert_scoped_memory_must_name_its_expert():
    with pytest.raises(MemoryError_, match="expert"):
        SessionMemory(
            title="t", body="b", scope=MemoryScope.expert,
        ).validate_for_storage()


# -- ontology bounds ---------------------------------------------------------------

def test_a_memory_cannot_invent_a_relationship_type():
    """The whole point of the two-tier design: prose is free, edges are not."""
    graph = FakeGraph(existing_ids={"core"})
    result = SessionMemoryRepository(graph).record(_memory(entity_links=[
        {"predicate": "MUST_BE_RUNNING_BEFORE", "target_id": "core",
         "target_label": "CodeRepository"},
    ]))

    assert result["links_written"] == 0
    assert graph.rels == []
    assert "undeclared relationship type" in result["rejected"][0]["reason"]


def test_a_memory_creates_an_entity_the_graph_does_not_have_yet():
    """Memories drive entity creation: a lesson about something not yet in the
    graph should bring it into the graph, not be silently half-stored."""
    graph = FakeGraph(existing_ids=set())
    result = SessionMemoryRepository(graph).record(_memory(entity_links=[
        {"target_id": "new-thing", "target_label": "CodeRepository",
         "target_name": "New Thing"},
    ]))

    assert result["entities_created"] == 1
    # A lesson about a repository is scope, not a stance, so no edge is written
    # and the scope lands on the memory node.
    assert result["links_written"] == 0
    assert result["applies_to"] == [
        {"label": "CodeRepository", "id": "new-thing", "name": "New Thing"},
    ]
    assert ("CodeRepository", {"id": "new-thing"}) in [
        (label, key) for label, key, _ in graph.nodes
    ]


def test_a_created_entity_is_marked_provisional_with_its_source():
    """It was asserted by a memory, not curated or ingested, and a reader has
    to be able to tell the difference."""
    graph = FakeGraph(existing_ids=set())
    SessionMemoryRepository(graph).record(_memory(entity_links=[
        {"target_id": "new-thing", "target_label": "CodeRepository",
         "target_name": "New Thing"},
    ]))

    _, _, props = graph.nodes[0]
    assert props["provisional"] is True
    assert props["created_by"] == "session_memory"
    assert any(src.startswith("session-memory::") for src in graph.sources)


def test_an_entity_is_not_created_without_a_name_to_deduplicate_on():
    """An id with no name is an unmergeable orphan: nothing later can tell it
    is the same thing as the properly-named node someone ingests tomorrow."""
    graph = FakeGraph(existing_ids=set())
    result = SessionMemoryRepository(graph).record(_memory(entity_links=[
        {"target_id": "ghost", "target_label": "CodeRepository"},
    ]))

    assert result["entities_created"] == 0
    assert "target_name" in result["rejected"][0]["reason"]


def test_creation_can_be_turned_off_per_memory():
    graph = FakeGraph(existing_ids=set())
    result = SessionMemoryRepository(graph).record(_memory(
        create_missing_entities=False,
        entity_links=[{"target_id": "ghost",
                       "target_label": "CodeRepository", "target_name": "Ghost"}],
    ))

    assert result["entities_created"] == 0
    assert "create_missing_entities is off" in result["rejected"][0]["reason"]


def test_an_existing_entity_is_reused_rather_than_duplicated_by_name():
    """The failure mode that would make creation worse than useless: minting a
    second node that differs only in capitalisation."""
    graph = FakeGraph(existing_ids=set(), named={"CodeRepository": {"Routing": "routing"}})
    result = SessionMemoryRepository(graph).record(_memory(entity_links=[
        {"target_id": "routing-2", "target_label": "CodeRepository",
         "target_name": "routing"},
    ]))

    assert result["entities_created"] == 0
    # Resolved to the node that already existed, not the id the model guessed.
    assert result["applies_to"] == [
        {"label": "CodeRepository", "id": "routing", "name": "routing"},
    ]


def test_a_memory_still_cannot_create_a_label():
    """Creating a node is data; creating a label is schema.

    Caught by the triplet check before any write is attempted, so it is
    reported as a rejection rather than an exception -- one bad link must not
    lose the lesson. `upsert_node` refuses the same label independently, so the
    guarantee does not rest on this check alone.
    """
    graph = FakeGraph(existing_ids=set())
    result = SessionMemoryRepository(graph).record(_memory(entity_links=[
        {"target_id": "x", "target_label": "MadeUpLabel", "target_name": "X"},
    ]))

    assert result["entities_created"] == 0
    assert graph.nodes == []
    assert "undeclared object label" in result["rejected"][0]["reason"]
    assert result["key"]  # the prose survived


def test_upsert_node_independently_refuses_an_undeclared_label():
    """Defence in depth: the write boundary refuses it even if a caller skips
    the triplet check."""
    from knowledge_bases.ontology import OntologyError
    from graph import Graph

    graph = Graph.__new__(Graph)
    graph.run = lambda *a, **k: []

    with pytest.raises(OntologyError, match="MadeUpLabel"):
        graph.upsert_node("MadeUpLabel", {"id": "x"}, {"name": "X"})


def test_a_valid_link_to_an_existing_entity_is_written():
    graph = FakeGraph(existing_ids={"routing"})
    result = SessionMemoryRepository(graph).record(_memory(entity_links=[
        {"predicate": "SUPPORTS", "target_id": "routing", "target_label": "Decision"},
    ]))

    assert result["links_written"] == 1
    assert graph.rels[0][2] == "SUPPORTS"


def test_a_rejected_link_does_not_block_the_memory_itself():
    """The prose is the authoritative tier; losing an edge must not lose the
    lesson."""
    graph = FakeGraph(existing_ids=set())
    result = SessionMemoryRepository(graph).record(_memory(entity_links=[
        {"predicate": "WORKS_FOR", "target_id": "x", "target_label": "CodeRepository"},
    ]))

    assert result["key"]
    assert any("MERGE (m:SessionMemory" in q for q, _ in graph.queries)


def test_rejected_links_are_recorded_on_the_memory_for_review():
    graph = FakeGraph(existing_ids=set())
    SessionMemoryRepository(graph).record(_memory(entity_links=[
        {"predicate": "WORKS_FOR", "target_id": "x", "target_label": "CodeRepository"},
    ]))

    assert any("unmapped_claims" in q for q, _ in graph.queries)


# -- append-only correction --------------------------------------------------------

def test_superseding_creates_a_new_node_and_never_deletes():
    """Memory never expires; a correction is a new node plus an edge."""
    graph = FakeGraph()
    replacement = _memory(body="Corrected guidance.")
    replacement.key = "replacement-key"

    SessionMemoryRepository(graph).supersede("old-key", replacement)

    written = " ".join(q for q, _ in graph.queries)
    assert "SUPERSEDES" in written
    assert "SET old:Deprecated" in written
    assert "DELETE" not in written.upper()


def test_superseding_a_missing_memory_is_refused():
    class Empty(FakeGraph):
        def run(self, query, **params):
            self.queries.append((query, params))
            return []

    with pytest.raises(ValueError, match="supersede"):
        SessionMemoryRepository(Empty()).supersede("nope", _memory())


def test_search_hides_superseded_memories_by_default():
    graph = FakeGraph()
    SessionMemoryRepository(graph).search(repo="noesis")

    assert "NOT m:Deprecated" in graph.queries[0][0]


def test_search_can_include_superseded_memories_as_history():
    graph = FakeGraph()
    SessionMemoryRepository(graph).search(repo="noesis", include_superseded=True)

    assert "NOT m:Deprecated" not in graph.queries[0][0]


# -- scope -------------------------------------------------------------------------

def test_search_unions_repo_expert_and_global_scopes():
    """A session in a repo working through an expert should see all three."""
    graph = FakeGraph()
    SessionMemoryRepository(graph).search(repo="noesis", expert_key="sql_expert")

    written = graph.queries[0][0]
    assert "m.scope = 'global'" in written
    assert "m.repo = $repo" in written
    assert "m.expert_key = $expert_key" in written


def test_search_orders_most_specific_scope_first():
    graph = FakeGraph()
    SessionMemoryRepository(graph).search(repo="noesis")

    written = graph.queries[0][0]
    assert written.index("WHEN 'expert' THEN 0") < written.index("WHEN 'repo' THEN 1")


def test_a_text_filter_does_not_collide_with_the_cypher_argument():
    """`Graph.run(query, **params)` takes `query` itself, so a parameter of
    that name raises `got multiple values for argument`."""
    graph = FakeGraph()
    SessionMemoryRepository(graph).search(repo="noesis", query="container")

    _, params = graph.queries[0]
    assert "query" not in params
    assert params["text"] == "container"
