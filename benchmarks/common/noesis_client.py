"""Noesis memory client.

Plays the same role as memory-benchmarks' `Mem0Client`, but talks to Noesis's
own graph-native evidence pipeline directly (in-process Python calls, not a
REST API -- Noesis has no standalone memory server):

  add(messages, user_id)  -> knowledge_bases.epistemology.feedback.FeedbackRepository.record
  search(query, user_id)  -> knowledge_bases.orchestration.hybrid_search.hybrid_search
  delete_user(user_id)    -> DETACH DELETE every Claim + SourceNode for that user

Each conversation turn is stored as its own `Claim` node (no LLM-based fact
extraction/dedup is performed -- see benchmarks/README.md), tagged with a
`user_id` property so `hybrid_search`'s `filters` parameter can scope a
query to one conversation. Retrieval fuses Neo4j full-text search
(`evidence_claim_text`) with a per-query Milvus vector search over
`claim`-shaped embeddings of each claim, then reranks with the same
cross-encoder used by every other Noesis knowledge base.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

CLAIMS_VECTOR_COLLECTION = "benchmark_claims"


class NoesisClient:
    """Synchronous-under-the-hood client exposing the same async interface
    the benchmark runners expect (add/search/delete_user), so the rest of
    the LOCOMO runner can stay close to the upstream mem0ai/memory-benchmarks
    structure.
    """

    def __init__(self, max_workers: int = 4):
        from graph import Graph
        from vector_store import VectorStore
        from llm import get_llm

        self.graph = Graph()
        self.vector_store = VectorStore()
        self.llm = get_llm()
        self._ensure_claims_collection()
        self._sem = asyncio.Semaphore(max_workers)

    def _ensure_claims_collection(self) -> None:
        from pymilvus import DataType

        client = self.vector_store.client
        if not client.has_collection(collection_name=CLAIMS_VECTOR_COLLECTION):
            from config import settings

            schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
            schema.add_field(
                field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=settings.embedding_dimension
            )
            index_params = client.prepare_index_params()
            index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="COSINE")
            client.create_collection(
                collection_name=CLAIMS_VECTOR_COLLECTION, schema=schema, index_params=index_params
            )

    async def close(self) -> None:
        self.graph.close()

    async def __aenter__(self) -> "NoesisClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # -------------------------------------------------------------------
    # Add
    # -------------------------------------------------------------------

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        timestamp: int | None = None,
    ) -> dict | None:
        """Store each message as a `Claim`, provenance-linked to
        `SourceNode {id: f"user::{user_id}"}` and tagged `user_id` for
        per-conversation scoping. Embeds each claim's text into a
        benchmark-only Milvus collection so search can do vector + fulltext
        fusion the same way `hybrid_search` does for every other Noesis
        entity type.
        """
        async with self._sem:
            return await asyncio.to_thread(self._add_sync, messages, user_id, timestamp)

    def _add_sync(self, messages, user_id, timestamp) -> dict:
        created_at = None
        if timestamp is not None:
            created_at = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

        source_id = f"user::{user_id}"
        results = []

        def work(tx: Any) -> None:
            tx.run(
                "MERGE (s:SourceNode {id: $source_id}) "
                "ON CREATE SET s.name = $user_id, s.type = 'user', "
                "s.reliability = 'medium', s.created_at = datetime() "
                "SET s.updated_at = datetime()",
                source_id=source_id,
                user_id=user_id,
            ).consume()
            for message in messages:
                text = message.get("content", "").strip()
                if not text:
                    continue
                claim_id = str(uuid4())
                tx.run(
                    "MATCH (s:SourceNode {id: $source_id}) "
                    "CREATE (c:Claim {id: $claim_id, claim_type: 'fact', "
                    "statement: $statement, status: 'cataloged', user_id: $user_id, "
                    "role: $role, created_at: coalesce($created_at, datetime())}) "
                    "CREATE (s)-[:ASSERTS {recorded_at: datetime()}]->(c)",
                    source_id=source_id,
                    claim_id=claim_id,
                    statement=text,
                    user_id=user_id,
                    role=message.get("role", ""),
                    created_at=created_at,
                ).consume()
                results.append({"id": claim_id, "memory": text, "event": "ADD"})

        self.graph.execute_write(work)

        for item in results:
            embedding = self.llm.embed(item["memory"])
            self.vector_store.upsert_vector(
                CLAIMS_VECTOR_COLLECTION,
                item["id"],
                embedding,
                metadata={"user_id": user_id},
            )
        return {"results": results}

    # -------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------

    async def search(self, query: str, user_id: str, top_k: int = 200) -> list[dict]:
        """Hybrid (fulltext + vector) search scoped to one conversation,
        via the same `hybrid_search` primitive every Noesis knowledge base
        uses, filtered by the `user_id` property set at ingest time.
        """
        async with self._sem:
            return await asyncio.to_thread(self._search_sync, query, user_id, top_k)

    def _search_sync(self, query: str, user_id: str, top_k: int) -> list[dict]:
        from knowledge_bases.orchestration import hybrid_search as hs_module

        # Route the Claim vector channel to our benchmark-only collection
        # rather than mutating the shared SEARCH_TYPES registry globally.
        original = hs_module.SEARCH_TYPES["claims.Claim"]
        patched = hs_module.SearchType(
            original.family, original.label, original.fulltext_index,
            CLAIMS_VECTOR_COLLECTION, original.identity_property, original.include_managed,
        )
        hs_module.SEARCH_TYPES["claims.Claim"] = patched
        try:
            hits = hs_module.hybrid_search(
                query,
                ["claims.Claim"],
                filters={"user_id": user_id},
                limit=min(top_k, 50),
                graph=self.graph,
                vector_store=self.vector_store,
                llm=self.llm,
            )
        finally:
            hs_module.SEARCH_TYPES["claims.Claim"] = original

        normalised = []
        for hit in hits:
            props = hit.get("properties", {})
            normalised.append({
                "id": hit.get("identity", ""),
                "memory": props.get("statement", ""),
                "score": hit.get("rerank_score", hit.get("score", 0.0)),
                "created_at": props.get("created_at"),
            })
        normalised.sort(key=lambda x: x.get("score", 0), reverse=True)
        return normalised

    # -------------------------------------------------------------------
    # Delete
    # -------------------------------------------------------------------

    async def delete_user(self, user_id: str) -> bool:
        async with self._sem:
            return await asyncio.to_thread(self._delete_user_sync, user_id)

    def _delete_user_sync(self, user_id: str) -> bool:
        try:
            rows = self.graph.run(
                "MATCH (s:SourceNode {id: $source_id})-[:ASSERTS]->(c:Claim) "
                "RETURN c.id AS id",
                source_id=f"user::{user_id}",
            )
            ids = [r["id"] for r in rows]
            for claim_id in ids:
                try:
                    self.vector_store.delete_vector(CLAIMS_VECTOR_COLLECTION, claim_id)
                except Exception:
                    pass
            self.graph.run(
                "MATCH (s:SourceNode {id: $source_id})-[:ASSERTS]->(c:Claim) "
                "DETACH DELETE c",
                source_id=f"user::{user_id}",
            )
            self.graph.run(
                "MATCH (s:SourceNode {id: $source_id}) DETACH DELETE s",
                source_id=f"user::{user_id}",
            )
            return True
        except Exception as exc:
            logger.warning("Failed to delete user %s: %s", user_id, exc)
            return False


def format_search_results(search_results: list[dict]) -> list[dict]:
    """Normalize search results into the format the LOCOMO prompts expect."""
    sorted_results = sorted(search_results, key=lambda x: x.get("score", 0), reverse=True)
    formatted = []
    for r in sorted_results:
        entry: dict[str, Any] = {
            "memory": r.get("memory", ""),
            "score": r.get("score", 0),
            "id": r.get("id", ""),
        }
        if r.get("created_at"):
            entry["created_at"] = r["created_at"]
        formatted.append(entry)
    return formatted
