"""Durable proposal and confirmation state for bounded knowledge writes."""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from graph import Graph


class WriteProposalRepository:
    def __init__(self, graph: Graph):
        self.graph = graph

    def create(
        self,
        command: str,
        knowledge_base: str,
        operation: str,
        arguments: dict[str, Any],
        summary: str,
        preview: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        proposal_id = str(uuid4())
        self.graph.run(
            "CREATE (p:WriteProposal {id: $id, command: $command, "
            "knowledge_base: $knowledge_base, operation: $operation, "
            "arguments_json: $arguments_json, summary: $summary, "
            "preview_json: $preview_json, "
            "status: 'proposed', created_at: datetime(), "
            "expires_at: datetime() + duration({minutes: 30})})",
            id=proposal_id,
            command=command,
            knowledge_base=knowledge_base,
            operation=operation,
            arguments_json=json.dumps(arguments, sort_keys=True),
            summary=summary.strip()[:500],
            preview_json=json.dumps(preview, sort_keys=True, default=str) if preview else None,
        )
        result = {
            "proposal_id": proposal_id,
            "command": command,
            "arguments": arguments,
            "summary": summary.strip()[:500],
            "status": "proposed",
            "expires_in_minutes": 30,
        }
        if preview is not None:
            result["preview"] = preview
        return result

    def reserve(self, proposal_id: str) -> dict[str, Any]:
        def work(tx: Any) -> dict[str, Any]:
            result = tx.run(
                "MATCH (p:WriteProposal {id: $id, status: 'proposed'}) "
                "WHERE p.expires_at > datetime() "
                "SET p.status = 'executing', p.confirmed_at = datetime() "
                "RETURN p.command AS command, p.arguments_json AS arguments_json",
                id=proposal_id,
            )
            row = result.single()
            if row is None:
                raise ValueError("Write proposal is missing, expired, or already used")
            return {
                "command": row["command"],
                "arguments": json.loads(row["arguments_json"]),
            }

        return self.graph.execute_write(work)

    def complete(self, proposal_id: str, status: str, error: str | None = None) -> None:
        self.graph.run(
            "MATCH (p:WriteProposal {id: $id}) "
            "SET p.status = $status, p.completed_at = datetime(), p.error = $error",
            id=proposal_id,
            status=status,
            error=error,
        )

    def reject(self, proposal_id: str, reason: str | None = None) -> None:
        """Explicitly cancel a proposal before it's confirmed.

        One-time-use, mirroring `reserve()`'s atomicity: only a proposal
        still in `status: 'proposed'` can be rejected, so a proposal that's
        already reserved/completed/failed can't be rejected out from under
        an in-flight execution, and a proposal can't be rejected twice.
        """
        def work(tx: Any) -> None:
            result = tx.run(
                "MATCH (p:WriteProposal {id: $id, status: 'proposed'}) "
                "SET p.status = 'rejected', p.rejected_at = datetime(), "
                "p.rejection_reason = $reason "
                "RETURN p.id AS id",
                id=proposal_id,
                reason=reason,
            )
            if result.single() is None:
                raise ValueError("Write proposal is missing, expired, or already used")

        self.graph.execute_write(work)
