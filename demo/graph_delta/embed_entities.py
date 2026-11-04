"""Embed canonical entities for hybrid (lexical + vector) entity search.

Claims are deliberately not embedded: a claim closely quotes its chunk, and the
chunk already owns the evidence vector. Entities are the canonical, deduplicated
concept (name + description + aliases), so they are the useful second index.

Collections match the declared `SearchType`s:

- `architecture_entities`  Component, Interface
- `bmad_entities`          Architecture, Capability, Specification, Decision,
                           Goal, Epic, UserStory, Requirement, Problem,
                           ProductPlan, ReviewFinding
- `business_terms`         BusinessTerm (identity is `name`)
- `code_entities`          CodeFunction, CodeMethod, CodeClass

    set -a && source .env && set +a
    python3 demo/graph_delta/embed_entities.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from config import settings
from graph import Graph
from vector_store import VectorStore

COLLECTIONS = {
    "architecture_entities": ("Component", "Interface"),
    "bmad_entities": (
        "Architecture", "Capability", "Specification", "Decision", "Goal",
        "Epic", "UserStory", "Requirement", "Problem", "ProductPlan",
        "ReviewFinding",
    ),
    "business_terms": ("BusinessTerm",),
    "metric_entities": ("Metric",),
    "code_entities": ("CodeFunction", "CodeMethod", "CodeClass"),
}

TEXT_KEYS = (
    "name", "description", "definition", "target", "statement", "context",
    "qualname", "signature", "docstring", "search_text", "unit", "status",
    "calculation_summary", "business_term", "normalized_name", "grain",
)


def entity_text(props: dict) -> str:
    parts = []
    for key in TEXT_KEYS:
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " \n".join(dict.fromkeys(parts))[:2000]


def fetch_rows(graph: Graph, label: str, per_label: int = 0) -> list[dict]:
    limit = f" LIMIT {int(per_label)}" if per_label else ""
    # BusinessTerm identity is `name`, not `id`; a generic id filter silently
    # skipped every term and left the vector collection without the demo ARR.
    if label == "BusinessTerm":
        return graph.run(
            f"MATCH (n:`{label}`) WHERE n.name IS NOT NULL "
            f"RETURN n.name AS id, properties(n) AS props{limit}"
        )
    return graph.run(
        f"MATCH (n:`{label}`) WHERE n.id IS NOT NULL "
        f"RETURN n.id AS id, properties(n) AS props{limit}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-label", type=int, default=0,
                        help="only this many nodes per label (for a probe)")
    parser.add_argument("--collection", choices=sorted(COLLECTIONS),
                        help="only refresh one collection")
    args = parser.parse_args()

    model = SentenceTransformer(
        str(Path(settings.embedding_model_path).expanduser().resolve()),
        local_files_only=True,
    )
    store = VectorStore(timeout=30)
    store.ensure_collections()

    started = time.time()
    total = 0
    with Graph() as graph:
        for collection, labels in COLLECTIONS.items():
            if args.collection and collection != args.collection:
                continue
            rows = []
            for label in labels:
                for row in fetch_rows(graph, label, args.per_label):
                    text = entity_text(row["props"])
                    if not text:
                        continue
                    rows.append({"label": label, "id": row["id"], "text": text,
                                 "props": row["props"]})
            print(f"{collection:22} {len(rows)} entities", flush=True)
            for start in range(0, len(rows), 256):
                batch = rows[start:start + 256]
                vectors = model.encode(
                    [r["text"] for r in batch], batch_size=64,
                    normalize_embeddings=True, show_progress_bar=False,
                )
                payload = []
                for row, vector in zip(batch, vectors):
                    props = row["props"]
                    if collection == "business_terms":
                        meta = {"name": props.get("name") or row["id"]}
                        vector_id = meta["name"]
                    elif collection == "bmad_entities":
                        meta = {
                            "entity_id": row["id"],
                            "entity_type": row["label"],
                            "project_id": props.get("project_id"),
                            "name": props.get("name"),
                            "search_text": row["text"],
                        }
                        vector_id = f"bmad:{row['label']}:{row['id']}"
                    elif collection == "code_entities":
                        meta = {
                            "id": row["id"],
                            "repo": props.get("repo"),
                            "path": props.get("path"),
                            "qualname": props.get("qualname"),
                            "name": props.get("name"),
                            "search_text": row["text"],
                        }
                        vector_id = row["id"]
                    else:
                        meta = {"id": row["id"], "name": props.get("name")}
                        vector_id = row["id"]
                    payload.append({"id": vector_id,
                                    "embedding": vector.tolist(), **meta})
                store.client.upsert(collection_name=collection, data=payload)
                total += len(payload)
                print(f"  {min(start+256, len(rows))}/{len(rows)}  "
                      f"{time.time()-started:.1f}s", flush=True)

    print(f"embedded {total} entities in {time.time()-started:.1f}s")
    for collection in COLLECTIONS:
        print(f"  {collection:22}", store.client.get_collection_stats(collection))


if __name__ == "__main__":
    main()
