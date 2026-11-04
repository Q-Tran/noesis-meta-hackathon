from knowledge_bases.orchestration.experts_catalog import bootstrap_profiles
from knowledge_bases.orchestration.repository import RoutingRepository
from knowledge_bases.orchestration.routing import plan_route


class FakeResult:
    def consume(self):
        return None


class FakeTransaction:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))
        return FakeResult()


class FakeGraph:
    def __init__(self, rows=None):
        self.tx = FakeTransaction()
        self.rows = rows or []
        self.run_calls = []

    def execute_write(self, work):
        return work(self.tx)

    def run(self, query, **params):
        self.run_calls.append((query, params))
        return self.rows


def test_routing_trace_stores_verbatim_question_plus_hash_and_summary():
    """The routing trace is the graph's single store of what the user said:
    claims are substantiated by `(Claim)-[:FOUND_IN {role: "origin"}]->(RoutingTrace)`,
    so the full question is written (the tracer's `record_turn_start` does the
    same for harness turns); the hash correlates retries and the summary
    bounds list views. Duplicating the text into chunks or a conversation
    SourceNode is what this replaced -- see tests/test_conversation_provenance.py."""
    graph = FakeGraph()
    repository = RoutingRepository(graph)
    question = "revenue recognition question " + "x" * 400
    plan = plan_route(question, bootstrap_profiles())

    repository.record_plan(plan, question)

    trace_call = graph.tx.calls[0][1]
    assert trace_call["request_hash"]
    assert trace_call["summary"] == question[:240]
    assert trace_call["question"] == question


def test_routing_gaps_passes_min_traces_through_to_the_query():
    graph = FakeGraph(rows=[
        {
            "knowledge_base": "methodology", "low_confidence_count": 10,
            "clarification_needed_count": 6, "sample_questions": ["q1"],
        },
    ])
    repository = RoutingRepository(graph)

    gaps = repository.routing_gaps(min_traces=7)

    assert gaps == [
        {
            "knowledge_base": "methodology", "low_confidence_count": 10,
            "clarification_needed_count": 6, "sample_questions": ["q1"],
        },
    ]
    query, params = graph.run_calls[0]
    assert params["min_traces"] == 7
    assert "RoutingTrace" in query
    assert "clarification_needed" in query


def test_routing_metrics_for_aggregates_one_knowledge_base_with_no_minimum_traffic_threshold():
    """Unlike `routing_gaps()`, `routing_metrics_for()` is for re-evaluating
    one specific knowledge base self-improvement already applied a
    capability-wording change to -- it must never require a minimum trace
    count or a low-confidence-share threshold the way `routing_gaps()` does."""
    graph = FakeGraph(rows=[
        {
            "trace_count": 3, "low_confidence_count": 1,
            "clarification_needed_count": 1, "avg_confidence": 0.42,
        },
    ])
    repository = RoutingRepository(graph)

    metrics = repository.routing_metrics_for("methodology", since="2026-08-17T00:00:00")

    assert metrics == {
        "trace_count": 3, "low_confidence_count": 1, "clarification_needed_count": 1,
        "low_confidence_share": (1 + 1) / 3, "avg_confidence": 0.42,
    }
    query, params = graph.run_calls[0]
    assert params["knowledge_base"] == "methodology"
    assert params["since"] == "2026-08-17T00:00:00"
    assert "primary_expert: $knowledge_base" in query
    assert "min_traces" not in query


def test_routing_metrics_for_returns_zeroed_metrics_when_there_are_no_traces():
    graph = FakeGraph(rows=[])
    repository = RoutingRepository(graph)

    metrics = repository.routing_metrics_for("project_management", since="2026-08-17T00:00:00")

    assert metrics == {
        "trace_count": 0, "low_confidence_count": 0, "clarification_needed_count": 0,
        "low_confidence_share": 0.0, "avg_confidence": None,
    }
