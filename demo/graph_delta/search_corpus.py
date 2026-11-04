"""Ask the docs corpus a question with graph vector search.

Embeds the question with the same MiniLM model used for the corpus, searches
the Milvus `chunks` collection, then walks each hit into the graph:
chunk -> source page + claims + entities. The answer is grounded in those
claims, not in the model's memory.

    set -a && source .env && set +a
    python3 demo/graph_delta/search_corpus.py "What capabilities does Muse have?"
"""
from __future__ import annotations

import argparse
import json

from sentence_transformers import SentenceTransformer

from config import settings
from graph import Graph
from vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    question = " ".join(args.query)

    model = SentenceTransformer(
        str(settings.embedding_model_path), local_files_only=True
    )
    vector = model.encode(question, normalize_embeddings=True).tolist()
    try:
        store = VectorStore(timeout=10)
        hits = store.search_vectors("chunks", vector, top_k=args.top * 3)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"vector search unavailable ({exc}). Start Milvus and run "
            "demo/graph_delta/embed_corpus.py first."
        )
    hits = [h for h in hits if str(h.get("id", "")).startswith("metadocs-chunk-")]
    hits = hits[: args.top]

    results = []
    with Graph() as graph:
        for hit in hits:
            chunk_id = hit["id"]
            rows = graph.run(
                "MATCH (c:Chunk {id: $id})-[:FOUND_IN]->(s:SourceNode) "
                "OPTIONAL MATCH (cl:Claim)-[:FOUND_IN]->(c) "
                "OPTIONAL MATCH (c)-[:MENTIONS]->(e) "
                "RETURN s.name AS page, s.url AS url, "
                "       collect(DISTINCT cl.statement)[0..5] AS claims, "
                "       collect(DISTINCT e.name)[0..8] AS entities",
                id=chunk_id,
            )
            if not rows:
                continue
            row = rows[0]
            results.append({
                "similarity": round(float(hit.get("similarity", 0)), 3),
                "chunk_id": chunk_id,
                "page": row["page"],
                "url": row["url"],
                "claims": [c for c in row["claims"] if c],
                "entities": [e for e in row["entities"] if e],
            })

    if args.json:
        print(json.dumps({"question": question, "results": results}, indent=1))
        return

    print(f"Q: {question}\n")
    for index, result in enumerate(results, 1):
        print(f"{index}. {result['similarity']:.3f}  {result['page']}")
        print(f"   {result['url']}")
        if result["entities"]:
            print(f"   entities: {', '.join(result['entities'])}")
        for claim in result["claims"]:
            print(f"   - {claim}")
        print()


if __name__ == "__main__":
    main()
