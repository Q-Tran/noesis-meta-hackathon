"""Durable session memory: what was learned, kept in the graph.

Memory never expires. That is a deliberate choice and it has a consequence:
correction must be *append-only*. A memory that turns out to be wrong is
superseded by a new node (`(:SessionMemory)-[:SUPERSEDES]->(:SessionMemory)`),
never edited in place -- the same shape `TraceOutcome` already uses for a
corrected verdict. Retrieval hides superseded nodes; history stays intact.

Two tiers, with different rules:

- The **body** is prose, and prose is what gets read back. A lesson like "check
  the container before editing code" is conditional and ordered; shredding it
  into triplets would lose the condition and the ordering that make it usable.
- The **edges** are triplets, and they are what make a memory findable by
  traversal rather than only by text search. They are restricted to ontology
  the registry already declares (see `knowledge_bases/ontology.py`), so writing
  a memory can never add a label or relationship type at runtime.

Scope is both axes at once: a memory is scoped to a repo, an expert, both, or
globally, and retrieval unions them with the most specific winning.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryScope(str, Enum):
    """How widely a memory applies.

    `session` exists so a memory can be captured before it is known to
    generalize; promoting it later is a new node that supersedes it.
    """

    session = "session"
    repo = "repo"
    expert = "expert"
    global_ = "global"


class MemoryKind(str, Enum):
    """What the memory is about, mirroring the four things worth persisting.

    Deliberately parallel to a harness's own refinement kinds, so a trajectory
    analyzer can classify into a vocabulary that already has graph homes --
    but `fact` is the common case and the others are for durable behaviour.
    """

    fact = "fact"
    procedure = "procedure"
    pitfall = "pitfall"
    preference = "preference"


def memory_key(title: str, scope: str, repo: str | None, expert_key: str | None) -> str:
    """A stable key so re-recording the same lesson MERGEs instead of duplicating.

    Derived from the title and the scope qualifiers rather than the body: a
    reworded body for the same lesson should update one memory, not accumulate
    near-duplicates.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
    digest = hashlib.sha1(
        "|".join([title, scope, repo or "", expert_key or ""]).encode("utf-8")
    ).hexdigest()[:10]
    return f"{slug}::{digest}" if slug else f"memory::{digest}"


class SessionMemory(BaseModel):
    """One durable lesson.

    `body` is the authoritative content. `entity_links` are ontology-checked
    triplets to entities that ALREADY exist -- a memory may link, never invent.
    `unmapped_claims` are extracted claims the ontology could not express, kept
    as reviewable evidence rather than dropped (a recurring one is a signal the
    ontology needs a reviewed change).
    """

    key: str = ""
    title: str
    body: str
    kind: MemoryKind = MemoryKind.fact
    scope: MemoryScope = MemoryScope.repo
    repo: str | None = None
    expert_key: str | None = None
    session_id: str | None = None
    trace_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Set on the node this one replaces; retrieval filters superseded memories.
    supersedes: str | None = None
    superseded_by: str | None = None
    entity_links: list[dict[str, Any]] = Field(default_factory=list)
    # Memories drive entity creation: an entity a memory refers to but the
    # graph lacks is created (tagged `provisional`, with a `SourceNode`), not
    # refused. Creating a node is data; creating a label is still schema, so
    # `upsert_node` rejects any label the registry has not declared.
    create_missing_entities: bool = True
    # Try embedding search when id and name both miss, so a paraphrase of an
    # existing entity resolves to it instead of creating a near-duplicate.
    # Turned off, resolution is exactly the deterministic name matching --
    # which is also what happens automatically when Milvus is unavailable.
    semantic_resolution: bool = True
    # Which entities the lesson was learned against. Scope, not stance: a
    # claim is a proposition and can only argue with a proposition, so the
    # subject matter of a lesson is recorded here rather than as an edge.
    applies_to: list[dict[str, str]] = Field(default_factory=list)
    unmapped_claims: list[dict[str, Any]] = Field(default_factory=list)

    def model_post_init(self, _context: Any) -> None:
        if not self.key:
            self.key = memory_key(
                self.title, self.scope.value, self.repo, self.expert_key,
            )

    def validate_for_storage(self) -> SessionMemory:
        """Raise unless this memory is worth storing and retrievable.

        An empty body is the failure mode that matters: a titled memory with no
        content looks present in a roster but teaches nothing when opened.
        """
        if not self.title.strip():
            raise MemoryError_("A session memory needs a title.")
        if not self.body.strip():
            raise MemoryError_(
                f"Session memory {self.title!r} has an empty body; a memory with "
                "no content is indistinguishable from a missing one when read."
            )
        if self.scope is MemoryScope.repo and not self.repo:
            raise MemoryError_("A repo-scoped memory must name its repo.")
        if self.scope is MemoryScope.expert and not self.expert_key:
            raise MemoryError_("An expert-scoped memory must name its expert.")
        return self


class MemoryError_(ValueError):
    """A session memory that could not be stored as given."""
