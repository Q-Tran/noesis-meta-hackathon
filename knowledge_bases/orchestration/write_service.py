"""Propose and execute allowlisted domain-owned knowledge writes."""
from __future__ import annotations

from typing import Any

from graph import Graph
from knowledge_bases.orchestration.commands import command_for, list_write_commands
from knowledge_bases.orchestration.write_repository import WriteProposalRepository


class KnowledgeWriteService:
    def __init__(self, graph: Graph, repository: WriteProposalRepository | None = None):
        self.graph = graph
        self.repository = repository or WriteProposalRepository(graph)

    def list_commands(self) -> list[dict[str, str]]:
        return list_write_commands()

    def propose(
        self, command_key: str, arguments: dict[str, Any], summary: str
    ) -> dict[str, Any]:
        command = command_for(command_key)
        validated = command.validate(arguments)
        preview = command.preview(validated, self.graph) if command.preview else None
        return self.repository.create(
            command.key,
            command.knowledge_base,
            command.operation,
            validated,
            summary,
            preview=preview,
        )

    def reject(self, proposal_id: str, reason: str | None = None) -> dict[str, Any]:
        self.repository.reject(proposal_id, reason)
        return {"proposal_id": proposal_id, "status": "rejected", "reason": reason}

    def execute(self, proposal_id: str, confirmed: bool) -> dict[str, Any]:
        if confirmed is not True:
            raise ValueError("Explicit confirmation is required")
        proposal = self.repository.reserve(proposal_id)
        command = command_for(proposal["command"])
        arguments = command.validate(proposal["arguments"])
        try:
            result = command.dispatch(arguments, self.graph)
        except Exception as exc:
            self.repository.complete(proposal_id, "failed", str(exc))
            raise
        self.repository.complete(proposal_id, "completed")
        if hasattr(result, "model_dump"):
            result = result.model_dump(mode="json")
        return {
            "proposal_id": proposal_id,
            "command": command.key,
            "status": "completed",
            "result": result,
        }
