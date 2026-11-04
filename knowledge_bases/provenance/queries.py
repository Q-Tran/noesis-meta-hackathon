"""Queries owned by Provenance."""

from typing import Any
from graph import Graph
from vector_store import VectorStore
from llm import get_llm


def trace_sources(graph: Graph, label: str, key: dict) -> list[dict]:
    """Return every `SourceNode` an entity is provenance-linked to, with role."""
    return graph.trace_sources(label, key)


def register_source(
    graph: Graph, source_id: str, name: str, type: str,
    reliability: str = "medium", url: str | None = None,
    observed_at: str | None = None, published_at: str | None = None,
) -> str:
    """Upsert a standalone `SourceNode` (metadata only — no content/excerpt
    text; see `knowledge_bases/provenance/models.py`). Returns `source_id`.

    Use `graph.attach_source(...)` to also link an entity to it via
    `FOUND_IN`; this just creates/updates the `SourceNode` itself.
    """
    graph.upsert_node("SourceNode", {"id": source_id}, {
        "name": name,
        "type": type,
        "reliability": reliability,
        "url": url,
        "observed_at": observed_at,
        "published_at": published_at,
    })
    return source_id


def register_chunk(
    graph: Graph,
    chunk_id: str,
    text: str,
    source_id: str | None = None,
    document_id: str | None = None,
    index: int | None = None,
    created_at: str | None = None,
) -> str:
    """Upsert a `Chunk` node and link it to its `SourceNode` if supplied."""
    props = {
        "text": text,
        "source_id": source_id,
        "document_id": document_id,
        "index": index,
        "created_at": created_at,
    }
    graph.upsert_node("Chunk", {"id": chunk_id}, props)
    if source_id:
        graph.upsert_rel("Chunk", {"id": chunk_id}, "FOUND_IN", "SourceNode",
                         {"id": source_id}, {"role": "origin"})
    return chunk_id


def trace_chunks(graph: Graph, label: str, key: dict) -> list[dict]:
    """Return every `Chunk` an entity is provenance-linked to."""
    return graph.trace_chunks(label, key)


def trace_chunk_provenance(graph: Graph, chunk_id: str) -> dict:
    """Full provenance for a chunk: source, extracted entities, and relationships."""
    return graph.trace_chunk_provenance(chunk_id)


def trace_relationship_chunk(
    graph: Graph,
    start_label: str,
    start_key: dict,
    rel_type: str,
    end_label: str,
    end_key: dict,
) -> dict | None:
    """Return the `Chunk` that backs a given relationship, if any."""
    return graph.trace_relationship_chunk(start_label, start_key, rel_type, end_label, end_key)


def search_provenance(
    query: str,
    top_k: int = 5,
    graph: Graph | None = None,
    vector_store: VectorStore | None = None,
    llm: Any | None = None,
) -> list[dict]:
    """Search vectorized chunks in Milvus via DiskANN and hydrate their graph provenance.

    Returns the top-k semantically relevant chunks alongside the entities and
    relationships grounded in each chunk.
    """
    graph = graph or Graph()
    vector_store = vector_store or VectorStore()
    llm = llm or get_llm()

    query_emb = llm.embed(query)
    hits = vector_store.search_chunks(query_emb, top_k=top_k)

    results = []
    for hit in hits:
        chunk_id = hit["id"]
        prov = graph.trace_chunk_provenance(chunk_id)
        results.append({
            "chunk_id": chunk_id,
            "similarity": hit.get("similarity", 0.0),
            "text": hit.get("text") or (prov.get("chunk") or {}).get("text"),
            "chunk": prov.get("chunk"),
            "source": prov.get("source"),
            "entities": prov.get("entities", []),
            "relationships": prov.get("relationships", []),
        })
    return results


def repoint_entity_reference(
    graph: Graph,
    chunk_id: str,
    old_entity_id: str,
    new_entity_id: str,
    new_entity_label: str | None = None,
    reason: str = "user_feedback",
    ref_id: str | None = None,
) -> dict:
    """Repoint an entity reference grounded in a chunk to a different entity.

    Promotes runner-up arguments, creates an audit attack edge, rewires canonical
    active edges (supersedes / superseded_by), rolls back properties contributed
    by the chunk, and marks orphan old entities as :Deprecated.
    """
    return graph.repoint_entity_reference(
        chunk_id=chunk_id,
        old_entity_id=old_entity_id,
        new_entity_id=new_entity_id,
        new_entity_label=new_entity_label,
        reason=reason,
        ref_id=ref_id,
    )


def trace_relationship_history(graph: Graph, rel_id: str) -> dict:
    """Trace the complete version lineage and argumentation audit trail for an edge."""
    return graph.trace_relationship_history(rel_id)


def trace_argument_lineage(graph: Graph, argument_id: str) -> dict:
    """Trace an argument's status, concluded claims, and attack relations."""
    return graph.trace_argument_lineage(argument_id)


__all__ = [
    "trace_sources",
    "register_source",
    "register_chunk",
    "trace_chunks",
    "trace_chunk_provenance",
    "trace_relationship_chunk",
    "search_provenance",
    "repoint_entity_reference",
    "trace_relationship_history",
    "trace_argument_lineage",
]
