"""Decision archival now uses the shared `:Deprecated` label (see
`docs/architecture/lifecycle.md`) instead of `DecisionStatus.archived`, which
no longer exists. `evaluate_decisions`/`open_world_review` only ever
consider live (non-deprecated) decisions."""
import pytest

from knowledge_bases.strategy.models import Decision, DecisionStatus
from recompute import evaluate_decisions
from review import open_world_review


class FakeGraph:
    def __init__(self, decisions):
        self._decisions = decisions
        self.calls = []
        self.upserted = []

    def run(self, query, **params):
        self.calls.append((query, params))
        if "MATCH (d:Decision) WHERE NOT d:Deprecated" in query:
            return list(self._decisions)
        return []

    def upsert_node(self, label, key, props=None):
        self.upserted.append((label, key, props))
        return []


def test_decision_status_no_longer_has_an_archived_value():
    assert set(DecisionStatus) == {DecisionStatus.active, DecisionStatus.review}
    with pytest.raises(ValueError):
        DecisionStatus("Archived")


def test_decision_model_defaults_to_active_without_archived_option():
    decision = Decision(id="d-1", description="Ship the thing")
    assert decision.status == DecisionStatus.active


def test_evaluate_decisions_only_queries_non_deprecated_decisions():
    graph = FakeGraph([{"id": "d-1", "description": "Ship the thing"}])

    evaluate_decisions(graph)

    query, _ = graph.calls[0]
    assert "NOT d:Deprecated" in query
    assert "Archived" not in query


def test_open_world_review_only_scans_non_deprecated_decisions_and_noops_without_matches():
    graph = FakeGraph([])

    result = open_world_review(["some news text"], graph=graph, llm=object())

    query, _ = graph.calls[0]
    assert "NOT d:Deprecated" in query
    assert "Archived" not in query
    assert result is True


def test_every_query_evaluate_decisions_issues_is_parsable_and_ontology_legal():
    """The checks in `recompute.py` are only exercised against a fake graph that
    returns [] for every query, so a query that Neo4j would refuse to parse has
    never been caught. Two real bugs survived exactly this way: an unclosed
    `(c:Claim` paren, and `RETURN i.statement` where the node was bound as `c`.

    This asserts the three things a real database would check: balanced
    parentheses, every returned alias names a variable the query binds, and
    every relationship used is declared with legal endpoints.
    """
    import re

    from knowledge_bases.ontology import allowed_labels, allowed_relationships
    from knowledge_bases.registry import relationship_types

    graph = FakeGraph([{"id": "d-1", "description": "Ship the thing"}])
    evaluate_decisions(graph)

    labels = allowed_labels()
    relationships = allowed_relationships()
    queries = [q for q, _ in graph.calls]
    assert len(queries) >= 4, "the four health checks must all run"

    for query in queries:
        assert query.count("(") == query.count(")"), f"unbalanced parens: {query}"

        # Every `x.field AS y` must name a variable the query binds.
        bound = set(re.findall(r"\(\s*([a-zA-Z]+)\s*[:(]", query))
        bound |= set(re.findall(r"\[([a-zA-Z]+):", query))
        for alias in re.findall(r"RETURN\s+([a-zA-Z]+)\.", query):
            assert alias in bound, f"{alias!r} returned but never bound: {query}"
        for alias in re.findall(r"([a-zA-Z]+)\.\w+ AS ", query):
            assert alias in bound, f"{alias!r} used but never bound: {query}"

        for rel in re.findall(r"\[:([A-Z_]+)\]", query):
            assert rel in relationships, f"{rel} is not declared: {query}"

        for label in re.findall(r"\(\s*[a-zA-Z]+\s*:\s*([A-Za-z]+)", query):
            assert label in labels, f"{label} is not declared: {query}"


def test_outcome_check_traverses_the_goal_and_reads_the_measured_claim():
    """`MEASURED_BY` is Goal -> Metric, so a check that asked for
    Decision -[:MEASURED_BY]-> Claim was asking for an edge the ontology
    forbids. The reading lives on a Claim carrying metric_id and value, and the
    goal names the bar, so the traversal goes through the goal."""
    graph = FakeGraph([{"id": "d-1", "description": "Ship the thing"}])
    graph.rows = [(
        "MATCH (d:Decision {id: $did})-[:INTENDED_FOR]->(g:Goal) "
        "WHERE g.metric_id IS NOT NULL AND g.comparator IS NOT NULL "
        "  AND g.target_value IS NOT NULL "
        "MATCH (c:Claim) "
        "WHERE c.metric_id = g.metric_id AND c.value IS NOT NULL "
        "  AND coalesce(c.status, 'active') <> 'refuted' "
        "WITH g, c ORDER BY c.computed_at DESC "
        "RETURN g.target AS gtext, g.comparator AS comp, g.target_value AS target, "
        "  c.statement AS stmt, c.value AS value, c.computed_at AS at "
        "LIMIT 1",
        {"did": "d-1"},
    )]

    def run(query, **params):
        graph.calls.append((query, params))
        if "NOT d:Deprecated" in query:
            return [{"id": "d-1", "description": "Ship the thing"}]
        if "INTENDED_FOR" in query:
            return [{"gtext": "cut churn below 5%", "comp": "<=",
                     "target": 5.0, "stmt": "churn measured", "value": 7.2,
                     "at": "2026-10-01"}]
        return []

    graph.run = run
    results = evaluate_decisions(graph)
    breach = [r for r in results if r["status"] == "ReviewNeeded"]
    assert breach, "a measurement above the bar must flag the decision"
    assert any("Outcome breach" in reason for reason in breach[0]["reasons"])
