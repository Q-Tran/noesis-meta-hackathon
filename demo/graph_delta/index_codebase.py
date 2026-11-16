"""Index this repository with the development (codebase) ontology.

The docs corpus describes APIs; it is not a repository. This script is the
honest use of the `Code*` labels: point `open_codebase` at a real checkout and
it measures files, functions, classes, methods, calls and imports.

    set -a && source .env && set +a
    python3 demo/graph_delta/index_codebase.py

Idempotent: `reindex_when_behind=False` would skip a current repo, so the
default re-checks git and reindexes when the checkout moved. Embeddings and
the analytics phases are off; this is a local demo graph.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from graph import Graph
from knowledge_bases.development.open_codebase import on_open

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    with Graph() as graph:
        result = on_open(
            graph,
            str(REPO_ROOT),
            repo="noesis",
            reindex_when_behind=True,
            embed=False,
            skip_analytics=True,
        )
        print(asdict(result))


if __name__ == "__main__":
    main()
