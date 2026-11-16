"""Semantic entity resolution for session memories.

A memory that says "the vector store fallback" and one that says "Vector Store
Fallback" mean the same component; a memory that says "the fallback path for
embeddings" also does, and no amount of string comparison will see it. That
last case is what this module adds.

Layered deliberately, cheapest and most certain first:

1. **id** -- exact, free.
2. **name** -- exact, then case-insensitive. Deterministic.
3. **embedding** -- nearest neighbours from Milvus, reranked, with the same
   0.94 auto-merge / 0.85 LLM-arbiter thresholds `extract.py` already uses for
   document ingestion. Two thresholds in two places for one question would be
   two things to keep in sync.

Milvus is an accelerator, never a dependency: every step degrades to the one
above it. With Milvus down, resolution is exactly the deterministic name
matching it was before -- correct, just blind to paraphrase. So this module
never raises on an unavailable vector store, and callers need no fallback of
their own.

Only labels with a vector collection can be resolved semantically
(`VectorStore.COLLECTIONS`); everything else stops at name matching.
"""
from __future__ import annotations

from dataclasses import dataclass

from graph import Graph

# Graph label -> (Milvus collection, the property holding its human text).
# A label absent here is name-matched only. Mirrors the call sites in
# `extract.py`'s `extract_knowledge_graph`, which is where these pairings are
# established for ingestion.
SEMANTIC_COLLECTIONS: dict[str, tuple[str, str]] = {
    "Claim": ("claims", "statement"),
    "Decision": ("strategy", "description"),
    "Task": ("tasks", "name"),
    "Project": ("projects", "name"),
    "Epic": ("epics", "name"),
    "BusinessTerm": ("business_terms", "name"),
}

# Same thresholds as `extract.py::_resolve_fuzzy_identity`: above AUTO_MERGE a
# match is taken outright; between ARBITER and AUTO_MERGE an LLM is asked
# whether the two texts denote the same entity; below ARBITER nothing matches.
AUTO_MERGE_SIMILARITY = 0.94
ARBITER_SIMILARITY = 0.85


@dataclass(frozen=True)
class Resolution:
    """How an entity reference was resolved, and on what evidence.

    `method` is recorded on the resulting link so a merge made by an embedding
    (which can be wrong) is distinguishable later from one made by an exact id
    (which cannot).
    """

    entity_id: str | None
    method: str  # "id" | "name" | "embedding" | "arbiter" | "none"
    similarity: float | None = None
    matched_text: str | None = None

    @property
    def found(self) -> bool:
        return self.entity_id is not None


def resolve_by_id(graph: Graph, label: str, entity_id: str) -> Resolution:
    rows = graph.run(
        f"MATCH (n:`{label}` {{id: $id}}) RETURN n.id AS id LIMIT 1", id=entity_id,
    )
    if rows:
        return Resolution(rows[0]["id"], "id")
    return Resolution(None, "none")


def resolve_by_name(graph: Graph, label: str, name: str) -> Resolution:
    """Exact name first, then case-insensitive.

    Ordered so "Routing" never resolves to "routing" when an exact "Routing"
    exists; without the ORDER BY the choice would depend on storage order.
    """
    if not name.strip():
        return Resolution(None, "none")
    rows = graph.run(
        f"MATCH (n:`{label}`) WHERE n.name = $name OR toLower(n.name) = toLower($name) "
        "RETURN n.id AS id, n.name AS name "
        "ORDER BY CASE WHEN n.name = $name THEN 0 ELSE 1 END LIMIT 1",
        name=name,
    )
    if rows:
        return Resolution(rows[0]["id"], "name", matched_text=rows[0].get("name"))
    return Resolution(None, "none")


def resolve_by_embedding(
    graph: Graph, label: str, text: str, *, vector_store=None, llm=None,
) -> Resolution:
    """Nearest neighbour in the label's vector collection, reranked and gated.

    Returns an unresolved `Resolution` -- never raises -- when Milvus is down,
    the label has no collection, or nothing clears the threshold. The caller
    cannot tell "no match" from "no Milvus", and should not need to: both mean
    fall back to the deterministic path.
    """
    if label not in SEMANTIC_COLLECTIONS or not text.strip():
        return Resolution(None, "none")

    collection, text_property = SEMANTIC_COLLECTIONS[label]

    try:
        if vector_store is None:
            from vector_store import VectorStore

            vector_store = VectorStore()
        if llm is None:
            from llm import get_llm

            llm = get_llm()

        embedding = llm.embed(text)
        candidate_k = llm.rerank_candidate_count(1, dedupe=True)
        nearest = vector_store.search_vectors(collection, embedding, top_k=candidate_k)
    except Exception:
        # Milvus unreachable, or no local embedding model. Both are "the
        # accelerator isn't there", which is not an error for the caller.
        return Resolution(None, "none")

    candidates = []
    for hit in nearest:
        if not hit.get("similarity"):
            continue
        rows = graph.run(
            f"MATCH (n:`{label}` {{id: $id}}) RETURN n.`{text_property}` AS text",
            id=hit["id"],
        )
        if rows and rows[0].get("text"):
            candidate = dict(hit)
            candidate["text"] = rows[0]["text"]
            candidates.append(candidate)

    if not candidates:
        return Resolution(None, "none")

    try:
        ranked = llm.rerank(text, candidates, text_key="text")
    except Exception:
        ranked = candidates

    for candidate in ranked:
        similarity = candidate.get("similarity") or 0.0
        if similarity >= AUTO_MERGE_SIMILARITY:
            return Resolution(
                candidate["id"], "embedding", similarity, candidate["text"],
            )
        if similarity >= ARBITER_SIMILARITY:
            # Near-miss: similar wording is not the same referent ("Q3 revenue
            # target" vs "Q4 revenue target" score alike), so ask rather than
            # guess. A failed arbiter call means "not proven the same".
            try:
                answer = llm.complete(
                    f"Entity A: '{candidate['text']}'. Entity B: '{text}'. "
                    "Do these refer to the exact same entity? Answer ONLY 'Yes' or 'No'."
                ).strip().lower()
            except Exception:
                continue
            if "yes" in answer:
                return Resolution(
                    candidate["id"], "arbiter", similarity, candidate["text"],
                )

    return Resolution(None, "none")


def resolve_entity(
    graph: Graph,
    label: str,
    entity_id: str,
    name: str = "",
    *,
    semantic: bool = True,
    vector_store=None,
    llm=None,
) -> Resolution:
    """Resolve a reference through id, then name, then embedding."""
    by_id = resolve_by_id(graph, label, entity_id)
    if by_id.found:
        return by_id

    by_name = resolve_by_name(graph, label, name)
    if by_name.found:
        return by_name

    if semantic:
        return resolve_by_embedding(
            graph, label, name or entity_id, vector_store=vector_store, llm=llm,
        )
    return Resolution(None, "none")
