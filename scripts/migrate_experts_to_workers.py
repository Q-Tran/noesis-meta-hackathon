"""Migrate graph-defined experts to fine-grained workers.

Experts remain readable for compatibility; workers are the canonical
product surface. The migration copies each KnowledgeBaseExpert definition
into Worker + WorkerVersion(v1) without changing expert behavior.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph import Graph
from knowledge_bases.orchestration.workers import WorkerRepository


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Count experts without writing workers")
    args = parser.parse_args(argv)
    graph = Graph()
    try:
        if args.dry_run:
            rows = graph.run("MATCH (e:KnowledgeBaseExpert) RETURN count(e) AS c")
            print(json.dumps({"experts": rows[0]["c"] if rows else 0, "dry_run": True}, indent=2))
            return 0
        workers = WorkerRepository(graph).migrate_experts()
        print(json.dumps({"workers": [worker.model_dump(mode="json") for worker in workers], "count": len(workers)}, indent=2))
        return 0
    finally:
        graph.close()


if __name__ == "__main__":
    raise SystemExit(main())
