import json

import pytest

from knowledge_bases.orchestration.commands import WRITE_COMMANDS, WriteCommand, command_for
from knowledge_bases.orchestration.write_service import KnowledgeWriteService


class Repository:
    def __init__(self):
        self.created = None
        self.reserved = None
        self.completed = []
        self.rejected = None

    def create(self, command, knowledge_base, operation, arguments, summary, preview=None):
        self.created = (command, knowledge_base, operation, arguments, summary, preview)
        result = {"proposal_id": "proposal-1", "command": command, "arguments": arguments}
        if preview is not None:
            result["preview"] = preview
        return result

    def reserve(self, proposal_id):
        assert proposal_id == "proposal-1"
        return self.reserved

    def complete(self, proposal_id, status, error=None):
        self.completed.append((proposal_id, status, error))

    def reject(self, proposal_id, reason=None):
        self.rejected = (proposal_id, reason)


def test_unsupported_write_command_is_rejected():
    with pytest.raises(ValueError, match="Unsupported Noesis write command"):
        command_for("project_management.generic_upsert")


def test_resolve_review_finding_is_a_registered_write_command():
    """The human-review path for architecture-kind self-improvement
    findings (and any other ReviewFinding) goes through the normal
    propose-confirm-execute WriteCommand machinery, not a bespoke path."""
    assert "methodology.resolve_review_finding" in WRITE_COMMANDS
    command = command_for("methodology.resolve_review_finding")
    assert command.knowledge_base == "methodology"
    assert command.operation == "resolve_review_finding"


def test_resolve_review_finding_validator_rejects_an_invalid_status():
    command = command_for("methodology.resolve_review_finding")

    with pytest.raises(Exception):
        command.validate({
            "project_id": "noesis-self", "finding_key": "x", "status": "not_a_real_status",
        })


def test_resolve_review_finding_validator_accepts_resolved_and_rejected():
    command = command_for("methodology.resolve_review_finding")

    validated = command.validate({
        "project_id": "noesis-self", "finding_key": "x", "status": "resolved", "note": "fixed manually",
    })
    assert validated["status"] == "resolved"

    validated = command.validate({
        "project_id": "noesis-self", "finding_key": "x", "status": "rejected",
    })
    assert validated["status"] == "rejected"


def test_proposal_validates_required_arguments_without_dispatching():
    repository = Repository()
    service = KnowledgeWriteService(object(), repository)

    with pytest.raises(ValueError, match="canonical_name"):
        service.propose(
            "business_dictionary.add_business_term_alias",
            {"alias": "ARR"},
            "Add ARR alias",
        )

    assert repository.created is None
    result = service.propose(
        "business_dictionary.add_business_term_alias",
        {"alias": "ARR", "canonical_name": "Annual Recurring Revenue"},
        "Add ARR alias",
    )
    assert result["proposal_id"] == "proposal-1"
    assert repository.created[1] == "business_dictionary"


def test_execution_requires_confirmation_and_uses_stored_arguments(monkeypatch):
    repository = Repository()
    repository.reserved = {
        "command": "business_dictionary.add_business_term_alias",
        "arguments": {"alias": "ARR", "canonical_name": "Annual Recurring Revenue"},
    }
    calls = []
    fake = WriteCommand(
        knowledge_base="business_dictionary",
        operation="add_business_term_alias",
        description="test",
        validate=lambda arguments: arguments,
        dispatch=lambda arguments, graph: calls.append(arguments) or {"written": True},
    )
    monkeypatch.setattr(
        "knowledge_bases.orchestration.write_service.command_for", lambda _key: fake
    )
    service = KnowledgeWriteService(object(), repository)

    with pytest.raises(ValueError, match="Explicit confirmation"):
        service.execute("proposal-1", False)
    assert calls == []

    result = service.execute("proposal-1", True)
    assert calls == [repository.reserved["arguments"]]
    assert result["status"] == "completed"
    assert repository.completed == [("proposal-1", "completed", None)]


class Result:
    def __init__(self, row=None):
        self.row = row

    def single(self):
        return self.row


class ProposalGraph:
    def __init__(self):
        self.proposal = None

    def run(self, query, **params):
        if "CREATE (p:WriteProposal" in query:
            self.proposal = dict(params, status="proposed")
        elif "SET p.status = $status" in query:
            self.proposal["status"] = params["status"]
        return []

    def execute_write(self, work):
        graph = self

        class Transaction:
            def run(self, query, **params):
                if graph.proposal is None or graph.proposal["status"] != "proposed":
                    return Result()
                if "status = 'executing'" in query:
                    graph.proposal["status"] = "executing"
                    return Result({
                        "command": graph.proposal["command"],
                        "arguments_json": graph.proposal["arguments_json"],
                    })
                if "status = 'rejected'" in query:
                    graph.proposal["status"] = "rejected"
                    graph.proposal["rejection_reason"] = params.get("reason")
                    return Result({"id": params["id"]})
                return Result()

        return work(Transaction())


def test_proposal_is_one_time_use():
    from knowledge_bases.orchestration.write_repository import WriteProposalRepository

    graph = ProposalGraph()
    repository = WriteProposalRepository(graph)
    proposal = repository.create(
        "business_dictionary.add_business_term_alias",
        "business_dictionary",
        "add_business_term_alias",
        {"alias": "ARR", "canonical_name": "Annual Recurring Revenue"},
        "Add alias",
    )

    reserved = repository.reserve(proposal["proposal_id"])
    assert json.loads(graph.proposal["arguments_json"]) == reserved["arguments"]
    with pytest.raises(ValueError, match="already used"):
        repository.reserve(proposal["proposal_id"])


def test_orchestrated_methodology_actor_is_runtime_owned():
    start = command_for("methodology.start_workflow").validate({
        "project_id": "project-1",
        "workflow": "discovery",
        "actor": "claimed-employee",
    })

    assert start["actor"] == "unknown"


def test_proposal_includes_preview_when_command_declares_one(monkeypatch):
    repository = Repository()
    fake = WriteCommand(
        knowledge_base="orchestration",
        operation="deprecate_entity",
        description="test",
        validate=lambda arguments: arguments,
        dispatch=lambda arguments, graph: {"deprecated_count": 1},
        preview=lambda arguments, graph: {"affected_count": 3},
    )
    monkeypatch.setattr(
        "knowledge_bases.orchestration.write_service.command_for", lambda _key: fake
    )
    service = KnowledgeWriteService(object(), repository)

    result = service.propose("orchestration.deprecate_entity", {}, "Deprecate node")

    assert result["preview"] == {"affected_count": 3}
    assert repository.created[-1] == {"affected_count": 3}


def test_proposal_omits_preview_when_command_declares_none():
    repository = Repository()
    service = KnowledgeWriteService(object(), repository)

    result = service.propose(
        "business_dictionary.add_business_term_alias",
        {"alias": "ARR", "canonical_name": "Annual Recurring Revenue"},
        "Add ARR alias",
    )

    assert "preview" not in result
    assert repository.created[-1] is None


def test_rejecting_a_proposal_delegates_to_the_repository():
    repository = Repository()
    service = KnowledgeWriteService(object(), repository)

    result = service.reject("proposal-1", "no longer needed")

    assert result == {"proposal_id": "proposal-1", "status": "rejected", "reason": "no longer needed"}
    assert repository.rejected == ("proposal-1", "no longer needed")


def test_proposal_reject_is_one_time_use_and_blocks_later_reservation():
    from knowledge_bases.orchestration.write_repository import WriteProposalRepository

    graph = ProposalGraph()
    repository = WriteProposalRepository(graph)
    proposal = repository.create(
        "business_dictionary.add_business_term_alias",
        "business_dictionary",
        "add_business_term_alias",
        {"alias": "ARR", "canonical_name": "Annual Recurring Revenue"},
        "Add alias",
    )

    repository.reject(proposal["proposal_id"], "changed my mind")
    assert graph.proposal["status"] == "rejected"
    assert graph.proposal["rejection_reason"] == "changed my mind"

    with pytest.raises(ValueError, match="already used"):
        repository.reserve(proposal["proposal_id"])

    with pytest.raises(ValueError, match="already used"):
        repository.reject(proposal["proposal_id"])
