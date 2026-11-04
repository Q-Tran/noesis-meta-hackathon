"""`knowledge_bases.mcp`'s generic entity/relationship browsing tools exclude
`:Deprecated` nodes by default (see `docs/architecture/lifecycle.md`) and
accept `include_deprecated=True` to opt back in."""
from knowledge_bases.mcp import get_entity, list_entities, list_relationships
from knowledge_bases.registry import knowledge_base


class FakeGraph:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))
        return []


def test_list_entities_excludes_deprecated_nodes_by_default():
    graph = FakeGraph()

    list_entities(graph, knowledge_base("semantic_layer"), "Table")

    query, _ = graph.calls[0]
    assert "NOT n:Deprecated" in query


def test_list_entities_includes_deprecated_nodes_when_requested():
    graph = FakeGraph()

    list_entities(graph, knowledge_base("semantic_layer"), "Table", include_deprecated=True)

    query, _ = graph.calls[0]
    assert "Deprecated" not in query


def test_get_entity_excludes_deprecated_nodes_by_default():
    graph = FakeGraph()

    get_entity(graph, knowledge_base("semantic_layer"), "Table", "orders")

    query, _ = graph.calls[0]
    assert "NOT n:Deprecated" in query


def test_get_entity_includes_deprecated_nodes_when_requested():
    graph = FakeGraph()

    get_entity(
        graph, knowledge_base("semantic_layer"), "Table", "orders",
        include_deprecated=True,
    )

    query, _ = graph.calls[0]
    assert "Deprecated" not in query


def test_list_relationships_excludes_deprecated_endpoints_by_default():
    graph = FakeGraph()

    list_relationships(graph, knowledge_base("semantic_layer"), "DEPENDS_ON")

    query, _ = graph.calls[0]
    assert "NOT a:Deprecated" in query
    assert "NOT b:Deprecated" in query


def test_list_relationships_includes_deprecated_endpoints_when_requested():
    graph = FakeGraph()

    list_relationships(
        graph, knowledge_base("semantic_layer"), "DEPENDS_ON",
        include_deprecated=True,
    )

    query, _ = graph.calls[0]
    assert "Deprecated" not in query
