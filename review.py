"""Open-world review: the unknown-unknowns catch.

Scans unstructured text (emails, news, tickets) against active decisions and
emits opposing evidence: a `SourceNode` for the external material, an external
`Claim` node linked to it via `FOUND_IN {role: "origin"}`, and a `CHALLENGES`
edge to the threatened decision. With the `null` LLM provider this is a safe
no-op.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from graph import Graph
from knowledge_bases.provenance.queries import register_source
from llm import BaseLLM, get_llm


class Threat(BaseModel):
    statement: str
    decision_id: str
    weight: float = 0.5
    kind: str = "rebutting"  # rebutting | undercutting


class ThreatList(BaseModel):
    threats: list[Threat] = Field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _claim_id(statement: str) -> str:
    h = hashlib.sha1(statement.encode("utf-8")).hexdigest()[:12]
    return f"claim::external::{h}"


def open_world_review(texts: list[str], graph: Graph | None = None,
                      llm: BaseLLM | None = None) -> bool:
    graph = graph or Graph()
    llm = llm or get_llm()

    decisions = graph.run(
        "MATCH (d:Decision) WHERE NOT d:Deprecated "
        "RETURN d.id AS id, d.description AS desc"
    )
    if not decisions:
        return True
    ctx = "\n".join(f"- {d['id']}: {d['desc']}" for d in decisions)

    for text in texts:
        prompt = (
            f"Active business decisions:\n{ctx}\n\n"
            f"New information:\n{text}\n\n"
            "Identify any information that challenges an active decision. "
            "For each, give the decision_id, a one-line statement of the threat, "
            "a weight 0-1 (severity), and kind ('rebutting' attacks the decision "
            "directly, 'undercutting' attacks a premise)."
        )
        try:
            result = llm.extract(prompt, ThreatList)
        except Exception as e:
            print(f"[review] extraction failed: {e}")
            continue

        valid_ids = {d["id"] for d in decisions}
        for t in result.threats:
            if t.decision_id not in valid_ids:
                continue
            cid = _claim_id(t.statement)
            source_id = f"source::external::{iid.split('::')[-1]}"
            register_source(
                graph, source_id, name=t.statement, type="external",
                observed_at=_now(),
            )
            graph.upsert_node("Claim", {"id": cid}, {
                "statement": t.statement,
                "computed_at": _now(),
            })
            graph.attach_source("Claim", {"id": cid}, source_id, role="origin")
            graph.upsert_rel("Claim", {"id": cid}, "CHALLENGES",
                             "Decision", {"id": t.decision_id},
                             {"weight": float(t.weight), "kind": t.kind, "status": "active"})
    return True


if __name__ == "__main__":
    open_world_review([
        "Supplier reports a labor strike that may disrupt DHL shipments next quarter."
    ])
