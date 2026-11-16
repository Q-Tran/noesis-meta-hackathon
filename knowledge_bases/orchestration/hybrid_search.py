"""Entity-scoped hybrid retrieval across bounded knowledge families."""
from __future__ import annotations

import re
from typing import Any

from graph import Graph
from knowledge_bases.base import SearchType
from knowledge_bases.registry import search_types, semantic_search_kinds


# Methodology-managed entities (`(:Label:BmadEntity)`, whose content lives on
# the current `Revision`) are indexed in their own fulltext index and vector
# collection regardless of which domain knowledge base owns the label. The
# names are data -- renaming them would orphan the stored vectors.
#
# WHICH labels those are is composed from the installed packages, not named here:
# this module ships with the engine, and an engine that hardcodes eleven domain
# label names cannot be distributed without them.
MANAGED_FULLTEXT_INDEX = "bmad_entity_text"
MANAGED_VECTOR_COLLECTION = "bmad_entities"


# The searchable type allowlist is composed from each installed ontology
# package's `search_types` contribution (see `knowledge_bases/base.py` and
# `knowledge_bases/<key>/package.py`); declaring a type in the owning
# package's `package.py` is the only way to widen it.
SEARCH_TYPES: dict[str, SearchType] = search_types()


def _validate_scope(entity_types: list[str]) -> list[tuple[str, SearchType]]:
    if not entity_types:
        raise ValueError("At least one entity type is required")
    try:
        scope = [(name, SEARCH_TYPES[name]) for name in dict.fromkeys(entity_types)]
    except KeyError as exc:
        raise ValueError(f"Unsupported search entity type: {exc.args[0]}") from exc
    families = {config.family for _, config in scope}
    if len(families) != 1:
        raise ValueError("Hybrid search entity types must belong to one knowledge family")
    return scope


def _matches_filters(entity: dict[str, Any], filters: dict[str, Any]) -> bool:
    properties = entity["properties"]
    for name, expected in filters.items():
        actual = entity.get(name, properties.get(name))
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _fulltext_candidates(
    graph: Graph, query: str, scope: list[tuple[str, SearchType]], candidate_k: int,
    include_deprecated: bool = False,
) -> list[dict[str, Any]]:
    # Natural-language prompts commonly contain Lucene operators and punctuation.
    query_text = " ".join(re.findall(r"\w+", query, flags=re.UNICODE))
    if not query_text:
        return []
    index_names = dict.fromkeys(
        index
        for _, config in scope
        for index in (
            (config.fulltext_index, MANAGED_FULLTEXT_INDEX)
            if config.include_managed
            else (config.fulltext_index,)
        )
        if index
    )
    candidates = []
    for index_name in index_names:
        rows = graph.run(
            "CALL db.index.fulltext.queryNodes($index_name, $query_text) "
            "YIELD node, score RETURN labels(node) AS labels, "
            "properties(node) AS properties, score ORDER BY score DESC LIMIT $limit",
            index_name=index_name, query_text=query_text, limit=candidate_k,
        )
        for rank, row in enumerate(rows, 1):
            labels = set(row["labels"])
            if not include_deprecated and "Deprecated" in labels:
                continue
            for qualified_type, config in scope:
                is_match = config.label in labels
                # A managed hit ((:Label:BmadEntity)) only counts for a type
                # that opted into managed entities; a domain index can return
                # the managed node's sparse entity row too, so check both ways.
                if "BmadEntity" in labels and not config.include_managed:
                    is_match = False
                if not is_match:
                    continue
                identity = row["properties"].get(config.identity_property)
                if identity is not None:
                    candidates.append({
                        "qualified_type": qualified_type,
                        "identity": identity,
                        "project_id": row["properties"].get("project_id"),
                        "lexical_rank": rank,
                    })
                break
    return candidates


def _vector_candidates(
    vector_store: Any, embedding: list[float],
    scope: list[tuple[str, SearchType]], candidate_k: int,
) -> list[dict[str, Any]]:
    candidates = []
    for collection in dict.fromkeys(
        config.vector_collection for _, config in scope if config.vector_collection
    ):
        hits = vector_store.search_vectors(collection, embedding, top_k=candidate_k)
        for rank, hit in enumerate(hits, 1):
            for qualified_type, config in scope:
                if config.vector_collection != collection:
                    continue
                if config.identity_property == "name":
                    identity = hit.get("name")
                else:
                    identity = hit.get("id")
                if identity is not None:
                    candidates.append({
                        "qualified_type": qualified_type,
                        "identity": identity,
                        "project_id": hit.get("project_id"),
                        "semantic_rank": rank,
                        "similarity": hit.get("similarity"),
                    })
                break
    managed_types = [
        (qualified_type, config) for qualified_type, config in scope
        if config.include_managed and config.label in semantic_search_kinds()
    ]
    if managed_types:
        hits = vector_store.search_vectors(
            MANAGED_VECTOR_COLLECTION, embedding, top_k=candidate_k,
        )
        for rank, hit in enumerate(hits, 1):
            for qualified_type, config in managed_types:
                if hit.get("entity_type") != config.label:
                    continue
                identity = hit.get("entity_id")
                if identity is not None:
                    candidates.append({
                        "qualified_type": qualified_type,
                        "identity": identity,
                        "project_id": hit.get("project_id"),
                        "semantic_rank": rank,
                        "similarity": hit.get("similarity"),
                    })
                break
    return candidates


def _hydrate(
    graph: Graph, scope: list[tuple[str, SearchType]], candidates: list[dict[str, Any]],
    include_deprecated: bool = False,
) -> list[dict[str, Any]]:
    deprecated_guard = "" if include_deprecated else "AND NOT n:Deprecated "
    hydrated = []
    for qualified_type, config in scope:
        selected = [c for c in candidates if c["qualified_type"] == qualified_type]
        if not selected:
            continue
        scoped_identities = [
            {"identity": c["identity"], "project_id": c.get("project_id")}
            for c in selected
        ]
        # One query covers both plain and managed nodes: a managed entity
        # ((:Label:BmadEntity)) hydrates its content from the current
        # Revision, a plain node from its own properties.
        rows = graph.run(
            "UNWIND $scoped_identities AS selected "
            f"MATCH (n:`{config.label}`) "
            f"WHERE n.`{config.identity_property}` = selected.identity "
            "AND (selected.project_id IS NULL OR n.project_id = selected.project_id) "
            f"{deprecated_guard}"
            "OPTIONAL MATCH (n)-[:CURRENT]->(r:Revision) "
            f"RETURN n.`{config.identity_property}` AS identity, "
            "n.project_id AS project_id, n.key AS key, "
            "coalesce(properties(r), properties(n)) AS properties, "
            "n.search_text AS text",
            scoped_identities=scoped_identities,
        )
        for row in rows:
            row["qualified_type"] = qualified_type
            if row.get("project_id") is None:
                row["project_id"] = row["properties"].get("project_id")
            if not row.get("text"):
                row["text"] = " ".join(
                    str(value) for value in row["properties"].values()
                    if isinstance(value, (str, int, float, bool))
                )
            hydrated.append(row)
    return hydrated


def hybrid_search(
    query: str,
    entity_types: list[str],
    filters: dict[str, Any] | None = None,
    limit: int = 10,
    graph: Graph | None = None,
    vector_store: Any | None = None,
    llm: Any | None = None,
    include_deprecated: bool = False,
) -> list[dict[str, Any]]:
    """Fuse lexical and semantic candidates within one explicit entity scope.

    Deprecated nodes (see `docs/architecture/lifecycle.md`) are excluded by
    default from both the full-text channel and the final hydration step;
    pass `include_deprecated=True` to include them (e.g. for an explicit
    "show history" request). Vector candidates need no filtering here since
    a deprecated node's embedding is removed from Milvus at deprecation time
    (`VectorStore.delete_vector`) rather than filtered per-query.
    """
    scope = _validate_scope(entity_types)
    graph = graph or Graph()
    if llm is None:
        from llm import get_llm
        llm = get_llm()
    candidate_k = llm.rerank_candidate_count(limit)
    candidates = _fulltext_candidates(
        graph, query, scope, candidate_k, include_deprecated=include_deprecated,
    )
    has_vector_channel = any(
        config.vector_collection
        or (config.include_managed and config.label in semantic_search_kinds())
        for _, config in scope
    )
    if has_vector_channel:
        if vector_store is None:
            from vector_store import VectorStore
            vector_store = VectorStore()
        candidates.extend(_vector_candidates(
            vector_store, llm.embed(query), scope, candidate_k
        ))

    fused: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for candidate in candidates:
        key = (
            candidate["qualified_type"], str(candidate["identity"]),
            candidate.get("project_id"),
        )
        current = fused.setdefault(key, {
            "qualified_type": candidate["qualified_type"],
            "identity": candidate["identity"],
            "project_id": candidate.get("project_id"),
            "score": 0.0,
        })
        for channel in ("lexical", "semantic"):
            rank = candidate.get(f"{channel}_rank")
            if rank:
                current["score"] += 1.0 / (60 + rank)
                current[f"{channel}_rank"] = rank
        if candidate.get("similarity") is not None:
            current["similarity"] = candidate["similarity"]

    entities = _hydrate(
        graph, scope, list(fused.values()), include_deprecated=include_deprecated,
    )
    filters = filters or {}
    entities = [entity for entity in entities if _matches_filters(entity, filters)]
    scores = {
        (item["qualified_type"], str(item["identity"]), item.get("project_id")): item
        for item in fused.values()
    }
    for entity in entities:
        candidate = scores.get((
            entity["qualified_type"], str(entity["identity"]),
            entity.get("project_id"),
        )) or scores.get((entity["qualified_type"], str(entity["identity"]), None))
        if candidate is None:
            continue
        entity.update({key: value for key, value in candidate.items() if key not in {
            "qualified_type", "identity", "project_id",
        }})
    ranked = llm.rerank(query, entities, text_key="text")[:max(1, min(limit, 50))]
    for entity in ranked:
        entity.pop("text", None)
    return ranked
