"""Ontology reflection: proposals for schema evolution, argued from graph
pressure evidence and reviewed before they graduate into code.

Ontology reflection never installs schema at runtime. It writes an
`OntologyProposal` (and a linked `ChangeProposal`) with evidence from
ingestion failures, unmapped chunks, orphan labels, and connector health.
Approval records a human decision; schema changes still ship as reviewed
code, and the graph keeps the rationale permanently.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from graph import Graph
from knowledge_bases.orchestration.connectors import ConnectorRepository
from knowledge_bases.orchestration.models import ChangeProposal, OntologyProposal
from knowledge_bases.orchestration.workers import ProposalRepository, _id, _now_iso

DEFAULT_ONTOLOGY_AUTONOMY = "review"  # reviewed by default; opt-in per ontology


class OntologyReflector:
    def __init__(self, graph: Graph, proposal_repo: ProposalRepository | None = None):
        self.graph = graph
        self.proposals = proposal_repo or ProposalRepository(graph)
        self.connectors = ConnectorRepository(graph)

    def _package_labels(self, package_key: str) -> list[str]:
        from knowledge_bases.registry import installed_packages

        for package in installed_packages():
            if package.key == package_key:
                labels = [label for kb in package.knowledge_bases for label in kb.labels]
                return sorted(set(labels))
        raise ValueError(f"Unknown ontology package {package_key!r}")

    def pressure(self, package_key: str) -> dict[str, Any]:
        labels = self._package_labels(package_key)
        label_counts = Counter()
        orphan_counts = Counter()
        try:
            rows = self.graph.run(
                "MATCH (n) RETURN labels(n) AS labels, "
                "count { (n)--() } AS degree"
            )
            for row in rows:
                for label in (row.get("labels") or []):
                    label_counts[label] += 1
                    if int(row.get("degree") or 0) == 0:
                        orphan_counts[label] += 1
        except Exception:  # noqa: BLE001 -- empty graph is valid pressure data
            pass
        try:
            failures = self.graph.run(
                "MATCH (s:Span) WHERE s.status = 'error' "
                "AND (s.name CONTAINS 'extract' OR s.name CONTAINS 'ingest' OR s.name CONTAINS 'chunk') "
                "RETURN s.name AS name, count(s) AS failed ORDER BY failed DESC LIMIT 10"
            )
        except Exception:  # noqa: BLE001
            failures = []
        try:
            unmapped = self.graph.run(
                "MATCH (c:Chunk) WHERE NOT EXISTS { (c)-[:EXTRACTED_FROM]-() } "
                "RETURN count(c) AS unmapped_chunks"
            )
        except Exception:  # noqa: BLE001
            unmapped = [{"unmapped_chunks": 0}]
        return {
            "package_key": package_key,
            "labels": [
                {
                    "label": label,
                    "count": label_counts.get(label, 0),
                    "orphans": orphan_counts.get(label, 0),
                }
                for label in labels
            ],
            "extraction_failures": [dict(row) for row in failures],
            "unmapped_chunks": int(unmapped[0].get("unmapped_chunks") or 0) if unmapped else 0,
            "connector_health": self.connectors.health(),
        }

    def _evidence(self, pressure: dict[str, Any]) -> list[dict]:
        evidence = []
        for row in pressure["labels"]:
            if row["orphans"] >= 10:
                evidence.append({
                    "id": _id("evidence"),
                    "kind": "orphan_label",
                    "summary": f"{row['label']}: {row['orphans']} orphan nodes of {row['count']}",
                    "trace_ids": [],
                    "score": row["orphans"],
                })
        for row in pressure["extraction_failures"]:
            evidence.append({
                "id": _id("evidence"),
                "kind": "extraction_failure",
                "summary": f"{row['name']}: {row['failed']} failed spans",
                "trace_ids": [],
                "score": row["failed"],
            })
        if pressure["unmapped_chunks"]:
            evidence.append({
                "id": _id("evidence"),
                "kind": "unmapped_chunk",
                "summary": f"{pressure['unmapped_chunks']} chunks have no extracted entity",
                "trace_ids": [],
                "score": pressure["unmapped_chunks"],
            })
        return evidence

    def propose(self, package_key: str, kind: str = "add_label", target: str = "") -> OntologyProposal:
        pressure = self.pressure(package_key)
        evidence = self._evidence(pressure)
        if kind == "deprecate_label":
            orphans = [row for row in pressure["labels"] if row["orphans"] >= 10]
            target = target or (orphans[0]["label"] if orphans else (pressure["labels"][0]["label"] if pressure["labels"] else ""))
            rationale = f"{target} is dominated by orphan nodes; deprecation keeps history while stopping new writes."
        elif kind in ("add_label", "add_property"):
            unmapped = pressure["unmapped_chunks"]
            rationale = (
                f"{unmapped} chunks could not be mapped to the current ontology; "
                f"extraction failures remain in {len(pressure['extraction_failures'])} tool(s)."
            )
        else:
            rationale = "Ontology reflection from graph pressure evidence."
        proposal = OntologyProposal(
            id=_id("ontologyproposal"),
            package_key=package_key,
            kind=kind,
            target=target,
            status="proposed",
            rationale=rationale,
            evidence=evidence,
            proposed_declaration={
                "package_key": package_key,
                "kind": kind,
                "target": target,
                "pressure": pressure,
                "note": "Approval is recorded here; schema lands as reviewed code.",
            },
        )
        self.graph.run(
            "MERGE (p:OntologyProposal {id: $id}) "
            "ON CREATE SET p.id = $id, p.package_key = $package, p.kind = $kind, "
            "p.target = $target, p.status = $status, p.rationale = $rationale, "
            "p.proposed_declaration_json = $declaration, p.created_at = datetime(), "
            "p.updated_at = datetime()",
            id=proposal.id, package=package_key, kind=kind, target=target,
            status=proposal.status, rationale=rationale,
            declaration=json.dumps(proposal.proposed_declaration, sort_keys=True),
        )
        for item in evidence:
            evidence_id = item.get("id") or _id("evidence")
            self.graph.run(
                "MERGE (e:ProposalEvidence {id: $id}) "
                "ON CREATE SET e.id = $id, e.kind = $kind, e.summary = $summary, "
                "e.trace_ids = $trace_ids, e.score = $score, e.created_at = datetime() "
                "WITH e "
                "MATCH (p:OntologyProposal {id: $proposal_id}) "
                "MERGE (p)-[:EVIDENCED_BY]->(e) "
                "ON CREATE SET e.created_at = datetime()",
                id=evidence_id, kind=item.get("kind", "score"),
                summary=item.get("summary", ""), trace_ids=item.get("trace_ids", []),
                score=item.get("score"), proposal_id=proposal.id,
            )
        return proposal

    def _to_proposal(self, props: dict[str, Any], evidence: list[dict]) -> OntologyProposal:
        declaration = props.get("proposed_declaration_json")
        if isinstance(declaration, str):
            try:
                declaration = json.loads(declaration)
            except ValueError:
                declaration = {}
        return OntologyProposal.model_validate({
            **props,
            "evidence": evidence,
            "proposed_declaration": declaration or {},
        })

    def get(self, proposal_id: str) -> OntologyProposal | None:
        rows = self.graph.run(
            "MATCH (p:OntologyProposal {id: $id}) "
            "OPTIONAL MATCH (p)-[:EVIDENCED_BY]->(e:ProposalEvidence) "
            "WITH p, collect(properties(e)) AS evidence "
            "RETURN properties(p) AS proposal, evidence",
            id=proposal_id,
        )
        if not rows:
            return None
        return self._to_proposal(rows[0].get("proposal") or {}, rows[0].get("evidence") or [])

    def list(self, status: str | None = None) -> list[OntologyProposal]:
        rows = self.graph.run(
            "MATCH (p:OntologyProposal) WHERE $status IS NULL OR p.status = $status "
            "OPTIONAL MATCH (p)-[:EVIDENCED_BY]->(e:ProposalEvidence) "
            "WITH p, collect(properties(e)) AS evidence "
            "RETURN properties(p) AS proposal, evidence "
            "ORDER BY p.created_at DESC",
            status=status,
        )
        return [self._to_proposal(row.get("proposal") or {}, row.get("evidence") or []) for row in rows]

    def set_status(self, proposal_id: str, status: str, actor: str = "user") -> OntologyProposal:
        proposal = self.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Unknown ontology proposal {proposal_id!r}")
        self.graph.run(
            "MATCH (p:OntologyProposal {id: $id}) "
            "SET p.status = $status, p.updated_at = datetime(), p.decided_by = $actor",
            id=proposal_id, status=status, actor=actor,
        )
        return self.get(proposal_id) or proposal

    def approve(self, proposal_id: str) -> OntologyProposal:
        return self.set_status(proposal_id, "approved")

    def reject(self, proposal_id: str) -> OntologyProposal:
        return self.set_status(proposal_id, "rejected")

    def reflect(self, package_key: str, kind: str = "add_label", target: str = "", auto_approve: bool = False) -> OntologyProposal:
        """One reflection pass. Default policy is review: approval is a
        separate, explicit act. `auto_approve=True` is the per-ontology
        opt-in; it still never installs schema at runtime."""
        proposal = self.propose(package_key, kind=kind, target=target)
        if auto_approve:
            return self.approve(proposal.id)
        return proposal
