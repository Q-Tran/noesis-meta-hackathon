from neo4j.time import DateTime
from pydantic import BaseModel

from knowledge_bases.orchestration.models import Delegation
from knowledge_bases.orchestration.models import RoutingPlan
from knowledge_bases.orchestration.experts_catalog import bootstrap_profiles
from knowledge_bases.orchestration.routing import plan_route
from knowledge_bases.orchestration.service import ExpertOrchestrator
import knowledge_bases.tools.tracing as tracing


class FakeRepository:
    def __init__(self):
        self.completed = None

    def load_experts(self):
        return bootstrap_profiles()

    def record_plan(self, plan, question):
        self.plan = plan
        self.question = question

    def complete_trace(self, trace_id, results, status):
        self.completed = (trace_id, results, status)


class DispatcherResult(BaseModel):
    data: object


def test_expert_failure_produces_partial_result_and_does_not_stop_delegation():
    repository = FakeRepository()
    plan = plan_route("target metric evidence", bootstrap_profiles())
    dispatchers = {
        delegation.knowledge_base: (
            (lambda _name, _args, _graph, **_kwargs: (_ for _ in ()).throw(RuntimeError("down")))
            if delegation.knowledge_base == "business_dictionary"
            else (lambda name, _args, _graph, **_kwargs: {"tool": name})
        )
        for delegation in plan.delegations
    }
    orchestrator = ExpertOrchestrator(
        graph=object(), dispatchers=dispatchers, repository=repository
    )

    result = orchestrator.execute(plan)

    assert result.status == "partial"
    assert result.expert_results[0].status == "failed"
    assert any(expert.status == "completed" for expert in result.expert_results[1:])
    assert repository.completed[2] == "partial"


def test_dispatcher_datetime_is_json_serializable_in_orchestration_result():
    repository = FakeRepository()
    plan = RoutingPlan(
        trace_id="trace-1",
        question_summary="When was this created?",
        primary_expert="epistemology",
        confidence=1.0,
        candidates=[],
        delegations=[Delegation(
            knowledge_base="epistemology",
            operation="lookup",
            arguments={},
            sequence=0,
        )],
    )
    orchestrator = ExpertOrchestrator(
        graph=object(),
            dispatchers={
                "epistemology": lambda _name, _args, _graph, **_kwargs: DispatcherResult(data={
                    "nested": [{"created_at": DateTime(2026, 8, 5, 12, 30)}]
                })
            },
        repository=repository,
        llm=object(),
    )

    result = orchestrator.execute(plan)

    assert result.expert_results[0].data["data"]["nested"][0]["created_at"].startswith(
        "2026-08-05T12:30:00"
    )
    result.model_dump(mode="json")


def test_execute_threads_the_routing_trace_id_into_every_dispatched_skill_call(monkeypatch):
    class FakePublisher:
        def __init__(self):
            self.spans = []

        def publish_span(self, event):
            self.spans.append(event)

    fake_publisher = FakePublisher()
    monkeypatch.setattr(tracing, "_get_sink", lambda: fake_publisher)

    repository = FakeRepository()
    seen_trace_ids = []

    def recording_dispatcher(operation, arguments, graph, trace_id=None, **_kwargs):
        seen_trace_ids.append(trace_id)
        return {"operation": operation}

    plan = RoutingPlan(
        trace_id="routing-trace-99",
        question_summary="What is ARR?",
        primary_expert="business_dictionary",
        confidence=1.0,
        candidates=[],
        delegations=[Delegation(
            knowledge_base="business_dictionary",
            operation="lookup_business_term",
            arguments={"term": "ARR"},
            sequence=0,
        )],
    )
    orchestrator = ExpertOrchestrator(
        graph=object(),
        dispatchers={"business_dictionary": recording_dispatcher},
        repository=repository,
        llm=object(),
    )

    orchestrator.execute(plan)

    assert seen_trace_ids == ["routing-trace-99"]


def test_execute_correlates_hybrid_search_span_with_the_routing_trace_id(monkeypatch):
    class FakePublisher:
        def __init__(self):
            self.spans = []

        def publish_span(self, event):
            self.spans.append(event)

    fake_publisher = FakePublisher()
    monkeypatch.setattr(tracing, "_get_sink", lambda: fake_publisher)
    monkeypatch.setattr(
        "knowledge_bases.orchestration.service.hybrid_search",
        lambda **_kwargs: {"results": []},
    )

    repository = FakeRepository()
    plan = RoutingPlan(
        trace_id="routing-trace-hybrid",
        question_summary="Find things",
        primary_expert="epistemology",
        confidence=1.0,
        candidates=[],
        delegations=[Delegation(
            knowledge_base="epistemology",
            operation="hybrid_search",
            arguments={"query": "things", "entity_types": ["claims.Claim"]},
            sequence=0,
        )],
    )
    orchestrator = ExpertOrchestrator(
        graph=object(), dispatchers={}, repository=repository, llm=object(),
    )

    orchestrator.execute(plan)

    assert len(fake_publisher.spans) == 1
    assert fake_publisher.spans[0].trace_id == "routing-trace-hybrid"
    assert fake_publisher.spans[0].name == "hybrid_search"
