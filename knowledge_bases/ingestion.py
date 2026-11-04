"""Append-only ingestion: the shared write primitive every bounded create uses.

Noesis never overwrites knowledge. When a new fact arrives that contradicts or
restates an existing one, the old node stays exactly as it was and the new node
is *related* to it -- `SUPERSEDES` for "this replaces that", `CONFLICTS_WITH`
for "these disagree and neither wins yet". That is what makes a changeset
unnecessary for ordinary ingestion: the graph is its own history, so there is
nothing to version around. The methodology engine still exists for work that
needs phase gates and baselines; it is no longer the price of writing a fact.

Two consequences worth stating, because both are load-bearing:

- **Reads must exclude superseded nodes,** or every query returns each fact once
  per revision. `current_filter()` is the clause to add; `SUPERSEDES` points
  from new to old, so "current" means "nothing supersedes me".
- **Identity is the caller's decision, not this module's.** `MERGE` on a natural
  key (a term's name, a metric's name) updates in place and is right for
  re-ingesting the same fact; a fresh id plus `SUPERSEDES` is right for a
  genuinely new version of it. `create_entity` takes `supersedes` explicitly so
  that choice is always visible at the call site rather than guessed here.

Every create funnels through `graph.upsert_node`/`upsert_rel`, so labels and
relationship types are validated against the composed ontology
(`knowledge_bases/ontology.py`) at the single write boundary. A bounded create
can therefore only write shapes its package already declared: ingestion widens
what is *in* the graph, never what *kinds of thing* the graph admits.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from graph import Graph

# Relationship types this module writes. Both are declared by the packages that
# use them (see each `package.py`); named here so a caller cannot typo one into
# a silently different edge.
SUPERSEDES = "SUPERSEDES"
CONFLICTS_WITH = "CONFLICTS_WITH"


def new_id(prefix: str) -> str:
    """A stable, readable id for an ingested entity."""
    return f"{prefix}-{uuid4().hex[:12]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_entity(
    graph: Graph,
    label: str,
    key: dict[str, Any],
    props: dict[str, Any] | None = None,
    *,
    supersedes: dict[str, Any] | None = None,
    conflicts_with: dict[str, Any] | None = None,
    source_id: str | None = None,
    source_props: dict[str, Any] | None = None,
    actor: str = "ingestion",
) -> dict[str, Any]:
    """Create or update one ontology entity, relating it to what it replaces.

    `supersedes`/`conflicts_with` are the key dicts of existing nodes of the
    same label. Neither is modified: the old node keeps its properties and its
    edges, and only gains an incoming `SUPERSEDES`/`CONFLICTS_WITH` from the new
    one. A caller that passes neither is asserting the fact is new, not that it
    is authoritative.

    Returns the ingested entity's key plus what it was related to, so a tool can
    report "recorded, superseding X" rather than a bare success.
    """
    props = dict(props or {})
    props.setdefault("ingested_at", _now())
    props.setdefault("ingested_by", actor)

    graph.upsert_node(label, key, props)

    related: dict[str, Any] = {}
    if supersedes:
        graph.upsert_rel(label, key, SUPERSEDES, label, supersedes)
        related["supersedes"] = supersedes
    if conflicts_with:
        graph.upsert_rel(label, key, CONFLICTS_WITH, label, conflicts_with,
                         props={"status": "active"})
        related["conflicts_with"] = conflicts_with
    if source_id:
        graph.attach_source(label, key, source_id, source_props, role="origin")
        related["source_id"] = source_id

    return {"label": label, "key": key, **related}


def relate(
    graph: Graph,
    start_label: str,
    start_key: dict[str, Any],
    rel_type: str,
    end_label: str,
    end_key: dict[str, Any],
    props: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Relate two existing entities, validated against the composed ontology."""
    graph.upsert_rel(start_label, start_key, rel_type, end_label, end_key, props)
    return {
        "from": {"label": start_label, "key": start_key},
        "relationship": rel_type,
        "to": {"label": end_label, "key": end_key},
    }


def current_filter(variable: str = "n") -> str:
    """Cypher predicate selecting only non-superseded, non-deprecated nodes.

    Append-only storage means a read that forgets this returns every historical
    version of a fact alongside the current one. `SUPERSEDES` points from the
    newer node to the older, so the current node is the one nothing supersedes.
    """
    return (
        f"NOT ({variable})<-[:{SUPERSEDES}]-() "
        f"AND NOT {variable}:Deprecated"
    )
