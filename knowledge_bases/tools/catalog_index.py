"""Semantic index over the graph's Tool and Expert catalogs.

Keyword overlap alone cannot match "where do this metric's numbers come
from" to `trace_metric_lineage`, or "who is on the hook for this blocked
epic" to the delivery expert. This module embeds every active `Tool`
(`noesis_skills` collection) and every expert capability phrase +
description (`noesis_experts` collection) into Milvus, so `search_noesis_tools`
and `route_noesis_question` can retrieve by meaning and then let the
cross-encoder / the harness LLM make the final choice.

The index is rebuilt by the same syncs that write the graph (`sync_catalog`,
`ExpertOrchestrator.sync_experts`, expert revisions) and is strictly an
accelerator: every entry point degrades to keyword-only when Milvus or the
embedding model is unavailable, logging once rather than failing a call.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

SKILLS_COLLECTION = "noesis_skills"
EXPERTS_COLLECTION = "noesis_experts"

logger = logging.getLogger(__name__)


def skill_text(skill: Any) -> str:
    return f"{skill.name}. {skill.description}".strip()


def expert_texts(expert: Any) -> list[tuple[str, str]]:
    """`(vector_id, text)` rows for one expert: its description plus each
    capability phrase separately, so a question only has to resemble one
    phrase to surface the expert."""
    rows = [(f"{expert.key}:description", f"{expert.name}. {expert.description}")]
    rows += [
        (f"{expert.key}:{position}", statement)
        for position, statement in enumerate(expert.effective_capabilities)
    ]
    return rows


class CatalogIndex:
    def __init__(self, vector_store: Any | None = None, llm: Any | None = None):
        self._vector_store = vector_store
        self._llm = llm
        self._unavailable = False

    # --- lazy dependencies ----------------------------------------------
    @property
    def vector_store(self) -> Any | None:
        if self._unavailable:
            return None
        if self._vector_store is None:
            try:
                from vector_store import VectorStore

                self._vector_store = VectorStore()
                self._vector_store.ensure_collections()
            except Exception as exc:
                # Expected whenever Milvus isn't running: every consumer
                # degrades to keyword-only. Log the reason, not a traceback --
                # this fires on every catalog write, and a stack trace per
                # write reads like a failure when it is a documented fallback.
                logger.warning(
                    "Catalog vector index unavailable (%s); using keyword-only ranking",
                    type(exc).__name__,
                )
                logger.debug("Vector index connect failed", exc_info=True)
                self._unavailable = True
                return None
        return self._vector_store

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from llm import get_llm

            self._llm = get_llm()
        return self._llm

    # --- writes ---------------------------------------------------------
    def _replace(self, collection: str, rows: Iterable[tuple[str, str, dict]]) -> int:
        store = self.vector_store
        if store is None:
            return 0
        count = 0
        for vector_id, text, metadata in rows:
            try:
                store.upsert_vector(collection, vector_id, self.llm.embed(text), metadata)
                count += 1
            except Exception:
                logger.warning("Failed to index %s into %s", vector_id, collection, exc_info=True)
        return count

    def index_skills(self, skills: list[Any]) -> int:
        return self._replace(
            SKILLS_COLLECTION,
            (
                (skill.key, skill_text(skill), {"knowledge_base": skill.knowledge_base})
                for skill in skills if skill.active
            ),
        )

    def index_experts(self, experts: list[Any]) -> int:
        rows = []
        for expert in experts:
            if not expert.active:
                continue
            rows += [
                (vector_id, text, {"expert_key": expert.key})
                for vector_id, text in expert_texts(expert)
            ]
        self.drop_expert_vectors([e.key for e in experts], keep=[r[0] for r in rows])
        return self._replace(EXPERTS_COLLECTION, rows)

    def drop_expert_vectors(self, expert_keys: list[str], keep: list[str] = ()) -> None:
        """Remove stale capability rows for `expert_keys` (e.g. a phrase
        deleted by a revision) that are not in `keep`."""
        store = self.vector_store
        if store is None:
            return
        try:
            for key in expert_keys:
                hits = store.client.query(
                    collection_name=EXPERTS_COLLECTION,
                    filter=f'expert_key == "{key}"', output_fields=["id"],
                )
                stale = [hit["id"] for hit in hits if hit["id"] not in set(keep)]
                if stale:
                    store.client.delete(collection_name=EXPERTS_COLLECTION, ids=stale)
        except Exception:
            logger.debug("Could not prune expert vectors", exc_info=True)

    # --- reads ----------------------------------------------------------
    def search_skills(self, query: str, top_k: int = 20) -> dict[str, float]:
        """`{skill_key: cosine similarity}` for the closest skills."""
        store = self.vector_store
        if store is None:
            return {}
        try:
            hits = store.search_vectors(SKILLS_COLLECTION, self.llm.embed(query), top_k)
        except Exception:
            return {}
        return {hit["id"]: float(hit["similarity"]) for hit in hits}

    def search_experts(self, query: str, top_k: int = 30) -> dict[str, dict[str, Any]]:
        """`{expert_key: {"similarity": best, "matched": vector_id}}` --
        the best-matching capability row per expert."""
        store = self.vector_store
        if store is None:
            return {}
        try:
            hits = store.search_vectors(EXPERTS_COLLECTION, self.llm.embed(query), top_k)
        except Exception:
            return {}
        best: dict[str, dict[str, Any]] = {}
        for hit in hits:
            key = hit.get("expert_key") or str(hit["id"]).split(":", 1)[0]
            similarity = float(hit["similarity"])
            if key not in best or similarity > best[key]["similarity"]:
                best[key] = {"similarity": similarity, "matched": hit["id"]}
        return best
