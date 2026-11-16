"""Worker, reflection, eval-gate, ontology, and connector behavior.

The new control plane is tested without Neo4j: repositories are exercised
through query-recording fakes and the reflection service through the same
`InMemoryTraceStore` telemetry fake the trace tests use.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from knowledge_bases.orchestration.models import (
    ChangeProposal,
    WorkerDefinition,
    WorkerVersion,
)
from knowledge_bases.tools.models import OutcomeSource, SpanEvent, SpanKind, SpanStatus, TraceOutcome
from knowledge_bases.tools.scoring import (
    SCORER_VERSION,
    compare_scores,
    record_trace_score,
    run_calibration,
    score_trace,
)
from tests.trace_store_fakes import InMemoryTraceStore


class QueryGraph:
    def __init__(self, *, workers=None, versions=None, tools=None):
        self.queries = []
        self.workers = workers or {}
        self.versions = versions or {}
        self.tools = tools or {"tools.search": {}, "tools.read": {}}

    def run(self, query, **params):
        self.queries.append((query, params))
        if "MATCH (t:Tool {active: true})" in query:
            return [{"key": key} for key in self.tools]
        if "MATCH (w:Worker {key: $key})" in query and "ACTIVE_VERSION" in query:
            worker = self.workers.get(params.get("key"))
            if worker is None:
                return []
            version = self.versions.get(worker["version"]) if worker else None
            return [{"worker": worker or {}, "version": version or {}}]
        if "MATCH (v:WorkerVersion {key: $key})" in query:
            return [{"version": v} for v in self.versions.values() if v.get("key") == params.get("key")]
        return []

    def execute_write(self, work):
        return work(self)


def _worker(key="invoice-triage", **overrides):
    data = {
        "key": key,
        "name": "Invoice Triage",
        "description": "Classify invoices and route exceptions.",
        "mission": "Keep invoice exceptions moving.",
        "capabilities": ["classify invoices"],
        "priority": 100,
        "active": True,
        "kind": "custom",
        "version": 1,
        "autonomy": "auto",
        "system_prompt": "Classify each invoice exactly once.",
        "entry_tools": ["tools.search"],
        "allowed_tools": ["tools.search"],
        "skills": [],
        "model_hint": None,
        "routable": True,
    }
    data.update(overrides)
    return WorkerDefinition.model_validate(data)


def test_worker_creation_validates_tools_and_writes_worker_and_version():
    from knowledge_bases.orchestration.workers import WorkerRepository

    graph = QueryGraph()
    definition = _worker()
    repo = WorkerRepository(graph)
    created = repo.upsert_worker(definition, "test")

    assert created.version == 1
    queries = [q for q, _ in graph.queries]
    assert any("MERGE (w:Worker {key: $key})" in q for q in queries)
    assert any("MERGE (v:WorkerVersion {id: $id})" in q for q in queries)
    assert any("ACTIVE_VERSION" in q for q in queries)


def test_worker_creation_rejects_unknown_tool():
    from knowledge_bases.orchestration.workers import WorkerRepository

    graph = QueryGraph()
    with pytest.raises(ValueError, match="unknown or unregistered tools"):
        WorkerRepository(graph).upsert_worker(_worker(allowed_tools=["missing.tool"]), "test")


def test_worker_version_rows_are_read_as_definitions():
    from knowledge_bases.orchestration.workers import WorkerRepository

    graph = QueryGraph(
        workers={"invoice-triage": {
            "key": "invoice-triage", "name": "Invoice Triage", "description": "Classify",
            "mission": "Keep moving", "active": True, "kind": "custom", "version": 2,
            "autonomy": "review", "routable": True, "priority": 100, "model_hint": None,
        }},
        versions={
            2: {
                "id": "v2", "key": "invoice-triage", "version": 2, "status": "active",
                "system_prompt": "Only route one exception.", "capabilities": ["route one"],
                "entry_tools": ["tools.search"], "allowed_tools": ["tools.search"],
                "skills": [], "model_hint": None, "reason": "reviewed",
            },
        },
    )
    worker = WorkerRepository(graph).load_worker("invoice-triage")
    assert worker is not None
    assert worker.version == 2
    assert worker.system_prompt == "Only route one exception."
    assert worker.autonomy == "review"


def test_draft_version_does_not_activate_it():
    from knowledge_bases.orchestration.workers import WorkerRepository

    graph = QueryGraph(
        workers={"invoice-triage": {
            "key": "invoice-triage", "name": "Invoice Triage", "description": "Classify",
            "mission": "", "active": True, "kind": "custom", "version": 1,
            "autonomy": "auto", "routable": True, "priority": 100, "model_hint": None,
        }},
        versions={1: {
            "id": "v1", "key": "invoice-triage", "version": 1, "status": "active",
            "system_prompt": "Classify.", "capabilities": ["classify"],
            "entry_tools": ["tools.search"], "allowed_tools": ["tools.search"],
            "skills": [], "model_hint": None,
        }},
    )
    draft = WorkerRepository(graph).create_draft_version(
        "invoice-triage", {"system_prompt": "Classify carefully."}, "draft for review",
    )
    assert draft.version == 2
    assert draft.status == "draft"
    assert not any("MERGE (w)-[a:ACTIVE_VERSION]->(v)" in q for q, _ in graph.queries)


def _store(trace_specs):
    store = InMemoryTraceStore()
    now = datetime.now(timezone.utc)
    spec_index = 0
    for trace_id, success, error_count, empty_count in trace_specs:
        for i in range(max(1, error_count + empty_count)):
            status = SpanStatus.error if i < error_count else SpanStatus.ok
            store.insert_span(SpanEvent(
                trace_id=trace_id,
                name="tool",
                kind=SpanKind.tool_call,
                knowledge_base="tools",
                skill="tools.search",
                started_at=now,
                ended_at=now,
                duration_ms=10 * (i + 1),
                status=status,
                result_count=0 if i >= error_count and i < error_count + empty_count else 1,
                expert_key="invoice-triage",
            ))
        if success is not None:
            store.insert_outcome(TraceOutcome(
                trace_id=trace_id,
                success=success,
                source=OutcomeSource.user_correction if not success else OutcomeSource.self_assessed,
                reason="user corrected" if not success else "accepted",
            ))
        spec_index += 1
    return store


def test_automated_scoring_penalties_and_calibration():
    store = _store([("t-ok", True, 0, 0), ("t-bad", False, 2, 1)])
    good = score_trace("t-ok", store=store)
    bad = score_trace("t-bad", store=store)
    assert good.composite > bad.composite
    assert bad.signals["corrected"] is True

    calibration = run_calibration(
        trace_ids=["t-ok", "t-bad"],
        expected={"t-ok": good.composite, "t-bad": bad.composite},
        store=store,
    )
    assert calibration["passed"] is True
    assert calibration["scorer_version"] == SCORER_VERSION

    recorded = record_trace_score("t-ok", store=store)
    assert recorded.score == good.composite
    assert recorded.source == OutcomeSource.automated


def test_eval_comparison_rejects_grounding_regression():
    from knowledge_bases.tools.scoring import ScoreBreakdown

    baseline = ScoreBreakdown(90, 90, 90, 90)
    candidate = ScoreBreakdown(95, 80, 95, 95)
    assert compare_scores(baseline, candidate)["passed"] is False


class FakeWorkerRepo:
    def __init__(self, worker):
        self.worker = worker
        self.drafts = []
        self.activations = []

    def load_worker(self, key):
        return self.worker

    def create_draft_version(self, key, changes, reason):
        self.drafts.append((key, changes, reason))
        return WorkerVersion(
            id="v2", key=key, version=2, status="draft",
            system_prompt=changes.get("system_prompt", self.worker.system_prompt),
            capabilities=changes.get("capabilities", self.worker.capabilities),
            entry_tools=changes.get("entry_tools", self.worker.entry_tools),
            allowed_tools=changes.get("allowed_tools", self.worker.allowed_tools),
            skills=changes.get("skills", self.worker.skills),
            model_hint=changes.get("model_hint", self.worker.model_hint),
        )

    def activate_version(self, key, version, reason):
        self.activations.append((key, version, reason))
        return WorkerVersion(id="v2", key=key, version=version, status="active", system_prompt="new")


class FakeProposalRepo:
    def __init__(self, proposal):
        self.proposal = proposal
        self.evals = []
        self.statuses = []

    def get(self, proposal_id):
        return self.proposal

    def set_status(self, proposal_id, status, extra=None):
        self.statuses.append((status, extra or {}))
        self.proposal.status = status

    def record_eval(self, report):
        self.evals.append(report)
        return report


def _reflection_service(store, worker=None, proposal=None):
    from knowledge_bases.orchestration.workers import ReflectionService

    service = ReflectionService(QueryGraph(), store=store)
    service.workers = FakeWorkerRepo(worker or _worker())
    if proposal is not None:
        service.proposals = FakeProposalRepo(proposal)
    service.sample_worker_traces = lambda key, limit=40: [
        {"trace_id": f"t-{i}", "score": 50.0, "breakdown": {"correctness": 50, "groundedness": 50, "behavior": 50, "cost": 50}}
        for i in range(limit)
    ]
    return service


def test_reflection_creates_a_proposal_from_failed_tools():
    store = _store([("t-1", False, 2, 0), ("t-2", False, 1, 0), ("t-3", True, 0, 0)])
    service = _reflection_service(store)
    result = service.reflect_worker("invoice-triage")
    assert result["proposal"] is not None
    assert "tools.search" in result["proposal"].changes["system_prompt"]
    assert result["proposal"].evidence


def test_eval_gate_rejects_tool_removal_but_prompts_pass():
    store = _store([("t-1", True, 0, 0)])
    service = _reflection_service(store)
    good = ChangeProposal(
        id="p-good", kind="worker_version", target_key="invoice-triage",
        changes={"system_prompt": "Better guidance."},
    )
    service.proposals = FakeProposalRepo(good)
    assert service.evaluate_proposal("p-good", stage="replay").passed is True

    bad = ChangeProposal(
        id="p-bad", kind="worker_version", target_key="invoice-triage",
        changes={"remove_tools": ["tools.search"]},
    )
    service.proposals = FakeProposalRepo(bad)
    assert service.evaluate_proposal("p-bad", stage="replay").passed is False


def test_auto_apply_applies_after_eval_but_review_waits_for_confirmation():
    store = _store([("t-1", True, 0, 0)])
    service = _reflection_service(store, worker=_worker(autonomy="auto"))
    proposal = ChangeProposal(
        id="p-auto", kind="worker_version", target_key="invoice-triage", status="eval_passed",
        changes={"system_prompt": "Better."},
    )
    service.proposals = FakeProposalRepo(proposal)
    applied = service.apply_proposal("p-auto")
    assert applied.status == "applied"
    assert service.workers.activations

    review = _worker(autonomy="review", key="invoice-review")
    proposal = ChangeProposal(
        id="p-review", kind="worker_version", target_key="invoice-review", status="eval_passed",
        changes={"system_prompt": "Better."},
    )
    service = _reflection_service(store, worker=review)
    service.proposals = FakeProposalRepo(proposal)
    held = service.apply_proposal("p-review")
    assert held.status == "approved"
    assert not service.workers.activations


def test_connectors_catalog_lists_real_sources_and_instances_are_graph_data():
    from knowledge_bases.orchestration.connectors import ConnectorRepository

    graph = QueryGraph()
    repo = ConnectorRepository(graph)
    keys = {c.key for c in repo.list_connectors()}
    assert "claims.microsoft_graph.teams_chat" in keys
    assert "claims.microsoft_graph.email" in keys
    assert "development.repository" in keys
    instance = repo.create_instance("claims.microsoft_graph.teams_chat", "My Teams", {"chat": "team-a"}, "daily")
    assert instance.connector_key.endswith("teams_chat")
    assert any("CONFIGURES" in q for q, _ in graph.queries)


def test_ontology_reflection_uses_graph_pressure_and_defaults_to_review():
    from knowledge_bases.orchestration.ontology_reflection import OntologyReflector

    graph = QueryGraph()
    # Override the generic fake for the pressure queries.
    orig = graph.run

    def run(query, **params):
        if "RETURN labels(n) AS labels" in query:
            return [
                {"labels": ["Worker"], "degree": 0},
                {"labels": ["Worker"], "degree": 0},
                {"labels": ["Worker"], "degree": 3},
            ]
        if "s.status = 'error'" in query:
            return [{"name": "extract", "failed": 3}]
        if "unmapped_chunks" in query:
            return [{"unmapped_chunks": 12}]
        return orig(query, **params)

    graph.run = run
    reflector = OntologyReflector(graph)
    pressure = reflector.pressure("orchestration")
    assert any(row["label"] == "Worker" and row["orphans"] == 2 for row in pressure["labels"])
    proposal = reflector.propose("orchestration", kind="add_label")
    assert proposal.status == "proposed"
    assert proposal.evidence


def test_worker_runtime_runs_a_task_with_only_the_worker_prompt():
    from knowledge_bases.orchestration.workers import WorkerRuntime

    class StubLLM:
        def __init__(self):
            self.calls = []

        def complete(self, prompt, system=None):
            self.calls.append((prompt, system))
            return f"Worker answered: {prompt}"

    worker = _worker()
    graph = QueryGraph()
    runtime = WorkerRuntime.__new__(WorkerRuntime)
    runtime.graph = graph
    runtime.store = _store([("t-runtime", True, 0, 0)])
    llm = StubLLM()
    runtime.llm = llm

    class Repo:
        def load_worker(self, key):
            return worker

    runtime.repository = Repo()
    result = runtime.run("invoice-triage", "Classify invoice INV-1")
    assert result["worker_key"] == "invoice-triage"
    assert result["answer"].startswith("Worker answered:")
    assert llm.calls and llm.calls[0][1] == worker.system_prompt
    outcomes = [o for o in runtime.store.outcomes.values() if o.source == OutcomeSource.automated]
    assert outcomes and outcomes[0].score == 100.0


def test_shadow_run_executes_candidate_instead_of_estimating_it():
    from knowledge_bases.orchestration.workers import ReflectionService

    store = _store([("t-shadow", True, 0, 0)])
    worker = _worker(allowed_tools=["tools.search"])
    proposal = ChangeProposal(
        id="p-shadow", kind="worker_version", target_key="invoice-triage",
        changes={"system_prompt": worker.system_prompt + "\n\nHandle invoice exceptions first."},
    )
    service = ReflectionService(QueryGraph(), store=store)
    service.workers = FakeWorkerRepo(worker)
    service.proposals = FakeProposalRepo(proposal)
    service.sample_worker_traces = lambda key, limit=40: [
        {"trace_id": "t-shadow", "score": 90.0, "breakdown": {"correctness": 90, "groundedness": 90, "behavior": 90, "cost": 90}}
    ]
    service._trace_task = lambda trace_id: "Classify invoice INV-1"
    service._shadow_tool_calls = lambda trace_id: []

    class StubLLM:
        def complete(self, prompt, system=None):
            if system == worker.system_prompt:
                return "baseline answer"
            return "candidate answer with the new rule"

    service._shadow_llm = StubLLM()
    service._shadow_dispatcher = lambda tool_key, args, graph, trace_id: [{"tool": tool_key}]
    report = service.evaluate_proposal("p-shadow")  # default is shadow
    assert report.stage == "shadow"
    assert report.candidate_score is not None
    assert report.baseline_score is not None
    assert report.samples_json


def test_shadow_run_rejects_candidate_that_fails_the_same_tasks():
    from knowledge_bases.orchestration.workers import ReflectionService

    store = _store([("t-shadow-bad", True, 0, 0)])
    worker = _worker(allowed_tools=["tools.search"])
    proposal = ChangeProposal(
        id="p-shadow-bad", kind="worker_version", target_key="invoice-triage",
        changes={"remove_tools": ["tools.search"]},
    )
    service = ReflectionService(QueryGraph(), store=store)
    service.workers = FakeWorkerRepo(worker)
    service.proposals = FakeProposalRepo(proposal)
    service.sample_worker_traces = lambda key, limit=40: [
        {"trace_id": "t-shadow-bad", "score": 90.0, "breakdown": {"correctness": 90, "groundedness": 90, "behavior": 90, "cost": 90}}
    ]
    service._trace_task = lambda trace_id: "Classify invoice INV-2"
    service._shadow_tool_calls = lambda trace_id: [
        {"tool_key": "tools.search", "arguments": {"query": "Classify invoice INV-2"}},
    ]
    service._shadow_llm = type("StubLLM", (), {"complete": lambda self, prompt, system=None: "answer"})()
    service._shadow_dispatcher = lambda tool_key, args, graph, trace_id: [{"tool": tool_key}]

    report = service.evaluate_proposal("p-shadow-bad")
    assert report.passed is False
