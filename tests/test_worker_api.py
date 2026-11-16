"""HTTP surface for the workers/proposals/ontologies/connectors console."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app as web_app
from knowledge_bases.orchestration.models import (
    ChangeProposal,
    ConnectorDefinition,
    ConnectorInstance,
    OntologyProposal,
    WorkerDefinition,
)


class FakeGraph:
    def __init__(self):
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))
        return [{"labels": []}]


@pytest.fixture
def client(monkeypatch):
    from fastapi import FastAPI

    fake = FakeGraph()
    web_app.app.dependency_overrides[web_app.get_graph] = lambda: fake
    yield TestClient(web_app.app)
    web_app.app.dependency_overrides.clear()


def _worker():
    return WorkerDefinition.model_validate({
        "key": "invoice-triage",
        "name": "Invoice Triage",
        "description": "Classify invoices.",
        "mission": "Keep exceptions moving.",
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
    })


def test_workers_round_trip_through_api(client, monkeypatch):
    import knowledge_bases.orchestration.workers as workers_module

    class FakeRepo:
        def __init__(self, graph):
            pass

        def load_workers(self, include_inactive=False):
            return [_worker()]

    monkeypatch.setattr(workers_module, "WorkerRepository", FakeRepo)
    response = client.get("/api/workers")
    assert response.status_code == 200
    assert response.json()["workers"][0]["key"] == "invoice-triage"


def test_worker_creation_validates_through_api(client, monkeypatch):
    import knowledge_bases.orchestration.workers as workers_module

    class FakeRepo:
        def __init__(self, graph):
            pass

        def upsert_worker(self, definition, reason=""):
            if definition.key == "bad key":
                raise ValueError("Invalid worker key")
            return definition

    monkeypatch.setattr(workers_module, "WorkerRepository", FakeRepo)
    bad = client.post("/api/workers", json={"key": "bad key", "name": "Bad"})
    assert bad.status_code == 422
    good = client.post("/api/workers", json={
        "key": "good-worker", "name": "Good", "description": "Does things.",
        "allowed_tools": ["tools.search"],
    })
    assert good.status_code == 200
    assert good.json()["key"] == "good-worker"


def test_proposal_actions_are_separated_by_gate_state(client, monkeypatch):
    import knowledge_bases.orchestration.workers as workers_module

    proposal = ChangeProposal(
        id="p-1", kind="worker_version", target_key="invoice-triage",
        status="proposed", changes={"system_prompt": "Better."},
    )

    class FakeProposals:
        def __init__(self, graph):
            pass

        def list(self, status=None):
            return [proposal]

        def get(self, proposal_id):
            return proposal

    class FakeRepo:
        def __init__(self, graph):
            pass

    class FakeService:
        def __init__(self, graph):
            self.proposals = FakeProposals(graph)

        def evaluate_proposal(self, proposal_id):
            from knowledge_bases.orchestration.models import EvalReport
            proposal.status = "eval_passed"
            return EvalReport(id="e-1", proposal_id=proposal_id, stage="replay", passed=True, baseline_score=80, candidate_score=86)

        def approve_proposal(self, proposal_id):
            proposal.status = "approved"
            return proposal

        def apply_proposal(self, proposal_id, confirm=False):
            if proposal.status == "eval_passed":
                raise ValueError("proposal must be approved before application in review mode")
            proposal.status = "applied"
            return proposal

    monkeypatch.setattr(workers_module, "ReflectionService", FakeService)
    monkeypatch.setattr(workers_module, "ProposalRepository", FakeRepo)
    assert client.post("/api/proposals/p-1/evaluate").json()["passed"] is True
    assert client.post("/api/proposals/p-1/apply", json={"confirm": True}).status_code == 422
    assert client.post("/api/proposals/p-1/approve").json()["status"] == "approved"
    assert client.post("/api/proposals/p-1/apply", json={"confirm": True}).json()["status"] == "applied"


def test_connectors_and_ontology_proposals_round_trip(client, monkeypatch):
    import knowledge_bases.orchestration.connectors as connectors_module
    import knowledge_bases.orchestration.ontology_reflection as ontology_module

    class FakeConnectorRepo:
        def __init__(self, graph):
            pass

        def list_connectors(self):
            return [ConnectorDefinition(key="x", name="X", description="", source_system="x", produces_labels=[], handler="x:run")]

        def list_instances(self):
            return [ConnectorInstance(id="i-1", connector_key="x", name="Mine")]

    class FakeReflector:
        def __init__(self, graph):
            pass

        def list(self, status=None):
            return [OntologyProposal(id="o-1", package_key="orchestration", kind="add_label", target="NewThing", status="proposed")]

    monkeypatch.setattr(connectors_module, "ConnectorRepository", FakeConnectorRepo)
    monkeypatch.setattr(ontology_module, "OntologyReflector", FakeReflector)
    connectors = client.get("/api/connectors")
    assert connectors.status_code == 200
    assert connectors.json()["connectors"][0]["key"] == "x"
    assert client.get("/api/ontologies/proposals").json()["proposals"][0]["id"] == "o-1"
