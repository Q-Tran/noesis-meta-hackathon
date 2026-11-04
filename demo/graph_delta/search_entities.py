"""Search canonical entities with hybrid (lexical + vector) retrieval.

`hybrid_search` deliberately scopes one knowledge family per call. This CLI
groups the requested types by family, runs each family, and prints the fused
results together. Entities, not claims: claims quote chunks and are reached
through their chunk evidence.

    set -a && source .env && set +a
    python3 demo/graph_delta/search_entities.py "Muse Wonder agentic coding" \
        --types architecture.Component architecture.Interface \
                project_management.Capability business_dictionary.BusinessTerm
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from graph import Graph
from knowledge_bases.orchestration.hybrid_search import SEARCH_TYPES, hybrid_search
from llm import get_llm
from vector_store import VectorStore

DEFAULT_TYPES = [
    "architecture.Component",
    "architecture.Interface",
    "project_management.Capability",
    "project_management.Specification",
    "business_dictionary.BusinessTerm",
    "semantic_layer.Metric",
    "development.CodeFunction",
    "development.CodeMethod",
    "development.CodeClass",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+")
    parser.add_argument("--types", nargs="+", default=DEFAULT_TYPES)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    query = " ".join(args.query)

    unknown = [t for t in args.types if t not in SEARCH_TYPES]
    if unknown:
        raise SystemExit(f"unknown search types: {unknown}")

    groups: dict[str, list[str]] = defaultdict(list)
    for name in args.types:
        groups[SEARCH_TYPES[name].family].append(name)

    llm = get_llm()
    store = VectorStore(timeout=10)
    results = []
    with Graph() as graph:
        for family, types in groups.items():
            rows = hybrid_search(
                query, types, limit=args.limit, graph=graph,
                vector_store=store, llm=llm,
            )
            for row in rows:
                props = row.get("properties") or {}
                results.append({
                    "family": family,
                    "type": row["qualified_type"],
                    "identity": row.get("identity"),
                    "name": props.get("name") or props.get("qualname"),
                    "description": props.get("description") or props.get("docstring"),
                    "score": round(row.get("score", 0.0), 4),
                    "semantic_rank": row.get("semantic_rank"),
                    "lexical_rank": row.get("lexical_rank"),
                    "similarity": row.get("similarity"),
                })

    results.sort(key=lambda r: r["score"], reverse=True)
    if args.json:
        print(json.dumps({"query": query, "results": results}, indent=1))
        return

    print(f"Q: {query}\n")
    for index, result in enumerate(results, 1):
        channels = []
        if result["semantic_rank"]:
            channels.append(f"vector #{result['semantic_rank']}")
        if result["lexical_rank"]:
            channels.append(f"text #{result['lexical_rank']}")
        print(f"{index}. {result['score']:.4f}  {result['type']}  "
              f"({', '.join(channels) or '?'})")
        print(f"   {result['name'] or result['identity']}")
        if result["description"]:
            print(f"   {result['description'][:180]}")
        print()


if __name__ == "__main__":
    main()
