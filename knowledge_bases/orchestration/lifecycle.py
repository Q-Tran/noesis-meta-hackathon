"""Generic, cross-domain deprecation/reactivation for graph nodes.

Implements the write side of `docs/architecture/lifecycle.md`'s "never
delete, tag instead" policy: a single, domain-parameterized pair of
operations (`deprecate_entity`/`reactivate_entity`) rather than one
hand-written command per knowledge base. Both are dispatched exclusively
through the `WriteProposal` propose -> confirm -> execute flow
(`knowledge_bases.orchestration.write_service.KnowledgeWriteService`), so a
human always sees and confirms exactly what will be tagged before
`Graph.mark_deprecated`/`Graph.reactivate` (`graph.py`) ever runs.

Ownership is checked against the same `KnowledgeBase.labels`/`relationships`
allowlists `knowledge_bases.mcp`'s generic entity/relationship browsing
already enforces (`knowledge_bases/registry.py`) -- there is no raw-Cypher
or unbounded-label path here, matching every other write command's
invariant (see `docs/architecture/knowledge_write_orchestration.md`).

Deliberately excluded knowledge bases:
  - `methodology`: already has its own append-only revisioning
    (`SUPERSEDED_BY`/`SUPERSEDES` via
    `knowledge_bases.methodology.repository._revise_entity`), and its
    machinery labels (`BmadEntity`, `Revision`, ...) are not business
    knowledge a caller should retire.
  - `skills`: already has `Tool.active` as its lifecycle flag.
  - `orchestration`/`provenance`: cross-cutting infrastructure, not
    user-owned business knowledge.
  - `development`: regenerated wholesale by `scripts/dev_catalog.py`, which
    tags absent entities `:Deprecated` itself; a manual deprecation wouldn't
    survive the next sync.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from graph import Graph
from knowledge_bases.registry import deprecatable_knowledge_bases, knowledge_base

# Derived from each knowledge base's `deprecatable` declaration in its
# `package.py` (see `knowledge_bases/base.py`).
DEPRECATABLE_KNOWLEDGE_BASES = deprecatable_knowledge_bases()

PREVIEW_SAMPLE_LIMIT = 20


class EntityRef(BaseModel):
    label: str
    key: dict[str, Any] = Field(default_factory=dict)


def _validate_ownership(knowledge_base_key: str, label: str, cascade: list[str]) -> None:
    if knowledge_base_key not in DEPRECATABLE_KNOWLEDGE_BASES:
        raise ValueError(
            f"{knowledge_base_key!r} does not support deprecation "
            f"(allowed: {', '.join(DEPRECATABLE_KNOWLEDGE_BASES)})"
        )
    kb = knowledge_base(knowledge_base_key)
    if label not in kb.labels:
        raise ValueError(f"{label!r} is not owned by {kb.name}")
    unknown = [rel for rel in cascade if rel not in kb.relationships]
    if unknown:
        raise ValueError(
            f"{', '.join(unknown)} not exposed by {kb.name}; "
            f"allowed: {', '.join(kb.relationships)}"
        )


class DeprecateEntityRequest(BaseModel):
    knowledge_base: str
    label: str
    key: dict[str, Any]
    cascade: list[str] = Field(default_factory=list)
    cascade_depth: int = Field(default=0, ge=0, le=20)
    superseded_by: EntityRef | None = None

    @model_validator(mode="after")
    def _check_ownership(self) -> "DeprecateEntityRequest":
        if not self.key:
            raise ValueError("key must identify at least one property")
        _validate_ownership(self.knowledge_base, self.label, self.cascade)
        if self.superseded_by is not None:
            _validate_ownership(self.knowledge_base, self.superseded_by.label, [])
            if not self.superseded_by.key:
                raise ValueError("superseded_by.key must identify at least one property")
        if self.cascade and self.cascade_depth < 1:
            raise ValueError("cascade_depth must be >= 1 when cascade relationships are given")
        return self


class ReactivateEntityRequest(BaseModel):
    knowledge_base: str
    label: str
    key: dict[str, Any]
    cascade: list[str] = Field(default_factory=list)
    cascade_depth: int = Field(default=0, ge=0, le=20)

    @model_validator(mode="after")
    def _check_ownership(self) -> "ReactivateEntityRequest":
        if not self.key:
            raise ValueError("key must identify at least one property")
        _validate_ownership(self.knowledge_base, self.label, self.cascade)
        if self.cascade and self.cascade_depth < 1:
            raise ValueError("cascade_depth must be >= 1 when cascade relationships are given")
        return self


def _match_clause(
    label: str, key: dict[str, Any], cascade: list[str], cascade_depth: int,
) -> tuple[str, dict[str, Any]]:
    """Build the caller-supplied match clause `Graph.mark_deprecated`/
    `Graph.reactivate` expect (a Cypher fragment binding `n`).

    Without `cascade`, this matches exactly the root node. With `cascade`,
    it walks an *undirected* variable-length pattern from the root, since
    containment edges point in different directions across domains
    (`HAS_TASK` parent->child vs. `COLUMN_OF`/`BELONGS_TO` child->parent) --
    undirected is the only generically-safe choice for a cross-domain
    primitive. `n` includes the root itself (`*0..depth`).
    """
    key_match = ", ".join(f"`{k}`: $key_{k}" for k in key)
    params = {f"key_{k}": v for k, v in key.items()}
    if not cascade:
        return f"MATCH (n:`{label}` {{{key_match}}})", params
    rel_pattern = "|".join(f"`{rel}`" for rel in cascade)
    return (
        f"MATCH (root:`{label}` {{{key_match}}}) "
        f"MATCH (root)-[:{rel_pattern}*0..{cascade_depth}]-(n)",
        params,
    )


def preview_deprecation(graph: Graph, request: DeprecateEntityRequest) -> dict[str, Any]:
    """Read-only: report how many/which nodes a deprecation would affect,
    without mutating anything, so a human can see the blast radius before
    confirming the proposal.
    """
    match_clause, params = _match_clause(
        request.label, request.key, request.cascade, request.cascade_depth,
    )
    # `DISTINCT n` guards against duplicate bindings when an undirected
    # variable-length cascade reaches the same node via more than one path.
    rows = graph.run(
        f"{match_clause} WITH DISTINCT n "
        "RETURN labels(n) AS labels, properties(n) AS properties "
        f"LIMIT {PREVIEW_SAMPLE_LIMIT + 1}",
        **params,
    )
    truncated = len(rows) > PREVIEW_SAMPLE_LIMIT
    sample = rows[:PREVIEW_SAMPLE_LIMIT]
    count_rows = graph.run(f"{match_clause} RETURN count(DISTINCT n) AS affected", **params)
    return {
        "affected_count": count_rows[0]["affected"] if count_rows else 0,
        "sample": sample,
        "sample_truncated": truncated,
    }


def _delete_vectors(label: str, rows: list[dict[str, Any]], vector_store: Any) -> None:
    """Best-effort: drop any Milvus embedding for each deprecated node, per
    `docs/architecture/lifecycle.md` (deprecation deletes derived vector
    entries; the graph node itself is never deleted). Failures never block
    the graph mutation that already happened.
    """
    from knowledge_bases.orchestration.hybrid_search import SEARCH_TYPES

    search_type = next(
        (config for config in SEARCH_TYPES.values()
         if config.label == label and config.vector_collection),
        None,
    )
    if search_type is None:
        return
    for row in rows:
        identity = row.get("properties", {}).get(search_type.identity_property)
        if identity is None:
            continue
        try:
            vector_store.delete_vector(search_type.vector_collection, identity)
        except Exception:
            continue


def deprecate_entity(
    arguments: dict[str, Any], graph: Graph, vector_store: Any | None = None,
) -> dict[str, Any]:
    request = DeprecateEntityRequest.model_validate(arguments)
    match_clause, params = _match_clause(
        request.label, request.key, request.cascade, request.cascade_depth,
    )
    affected = graph.run(
        f"{match_clause} WITH DISTINCT n "
        "RETURN labels(n) AS labels, properties(n) AS properties",
        **params,
    )
    superseded_by_key = None
    superseded_by_label = None
    if request.superseded_by is not None:
        superseded_by_key = request.superseded_by.key
        superseded_by_label = request.superseded_by.label
    graph.mark_deprecated(
        match_clause, params,
        superseded_by_key=superseded_by_key, superseded_by_label=superseded_by_label,
    )
    if vector_store is None:
        try:
            from vector_store import VectorStore
            vector_store = VectorStore()
        except Exception:
            vector_store = None
    if vector_store is not None:
        by_label: dict[str, list[dict[str, Any]]] = {}
        for row in affected:
            for label in row["labels"]:
                by_label.setdefault(label, []).append(row)
        for label, rows in by_label.items():
            _delete_vectors(label, rows, vector_store)
    return {
        "deprecated_count": len(affected),
        "labels": sorted({label for row in affected for label in row["labels"]}),
    }


def reactivate_entity(arguments: dict[str, Any], graph: Graph) -> dict[str, Any]:
    request = ReactivateEntityRequest.model_validate(arguments)
    match_clause, params = _match_clause(
        request.label, request.key, request.cascade, request.cascade_depth,
    )
    rows = graph.reactivate(match_clause, params)
    return {"reactivated_count": len(rows)}
