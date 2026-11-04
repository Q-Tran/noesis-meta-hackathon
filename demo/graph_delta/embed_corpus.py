"""Embed the docs chunks into Milvus for graph vector search.

The extraction/claims live in Neo4j; this adds the retrieval vectors for the
`chunks` collection so a question can be answered from the corpus by semantic
search, then walked into the graph (chunk -> claims -> entities -> pages).

    set -a && source .env && set +a
    python3 demo/graph_delta/embed_corpus.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from config import settings
from graph import Graph
from vector_store import VectorStore

BATCH = 64
UPSERT = 256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with Graph() as graph:
        rows = graph.run(
            "MATCH (c:Chunk)-[:FOUND_IN]->(s:SourceNode) "
            "WHERE c.id STARTS WITH 'metadocs-chunk-' "
            "RETURN c.id AS id, c.text AS text, c.index AS idx, "
            "       s.name AS title, s.url AS url "
            "ORDER BY c.id"
        )
    if args.limit:
        rows = rows[: args.limit]
    print(f"chunks: {len(rows)}")

    model_path = Path(settings.embedding_model_path).expanduser().resolve()
    model = SentenceTransformer(str(model_path), local_files_only=True)
    store = VectorStore(timeout=30)
    store.ensure_collections()

    started = time.time()
    for start in range(0, len(rows), UPSERT):
        batch = rows[start:start + UPSERT]
        vectors = model.encode(
            [r["text"] for r in batch],
            batch_size=BATCH,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        payload = []
        for row, vector in zip(batch, vectors):
            payload.append({
                "id": row["id"],
                "embedding": vector.tolist(),
                "text": row["text"][:3000],
                "source_id": row["id"],
                "title": row["title"],
                "url": row["url"],
                "chunk_index": row["idx"],
            })
        store.client.upsert(collection_name="chunks", data=payload)
        done = min(start + UPSERT, len(rows))
        print(f"  {done}/{len(rows)}  {time.time() - started:.1f}s", flush=True)

    print("chunks collection:", store.client.get_collection_stats("chunks"))


if __name__ == "__main__":
    main()
