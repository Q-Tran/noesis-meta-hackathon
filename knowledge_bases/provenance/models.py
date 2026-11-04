"""Provenance model ownership.

`SourceNode` is the canonical provenance node for the *entire* graph, not just
documents or users. Any entity in any other knowledge base records where it
came from via `(:Entity)-[:FOUND_IN {role}]->(:SourceNode)` (see
`graph.attach_source`/`graph.trace_sources` and
`docs/architecture/provenance.md`) rather than a flat string property.
Domain knowledge bases point outward at this node; they don't own it.

This is unrelated to `Claim` provenance (`(SourceNode)-[:ASSERTS]->(Claim)`,
`(SourceNode)-[:SUPPORTS|CHALLENGES]->(Claim)`), which is owned by the
Evidence knowledge base (`knowledge_bases.epistemology`) — `SourceNode` is simply
the origin of both mechanisms.

`SourceNode` intentionally holds only *metadata about* a source (name, type,
where to find it, when it was observed/published), never the source's actual
content/excerpt. Noesis deliberately keeps document/excerpt text out of the
graph database; that previously lived on the now-removed `ExternalInfo.content`
field, which was write-only and never read back.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SourceReliability(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class SourceNode(BaseModel):
    id: str
    name: str
    type: str
    reliability: SourceReliability = SourceReliability.medium
    url: Optional[str] = None
    observed_at: Optional[str] = None
    published_at: Optional[str] = None


class Chunk(BaseModel):
    id: str
    text: str
    source_id: Optional[str] = None
    document_id: Optional[str] = None
    index: Optional[int] = None
    created_at: Optional[str] = None

