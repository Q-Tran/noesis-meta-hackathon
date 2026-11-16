"""Run worker reflection for one worker.

By default reflection samples traces, creates a proposal, and evaluates it.
`--apply` is required to activate an auto-apply proposal; review-mode
workers leave the proposal in the inbox for explicit approval.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph import Graph
from knowledge_bases.orchestration.workers import ReflectionService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker_key")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--apply", action="store_true", help="Apply if the worker is auto-apply and the gate passes")
    args = parser.parse_args(argv)
    graph = Graph()
    try:
        service = ReflectionService(graph)
        result = service.reflect_and_apply(args.worker_key, limit=args.limit) if args.apply else service.reflect_worker(args.worker_key, limit=args.limit)
        if not args.apply and result.get("proposal") is not None:
            evaluation = service.evaluate_proposal(result["proposal"].id, stage="shadow")
            result = {**result, "evaluation": evaluation, "action": "evaluated"}
        print(json.dumps({
            "proposal": result.get("proposal").model_dump(mode="json") if result.get("proposal") else None,
            "evaluation": result.get("evaluation").model_dump(mode="json") if result.get("evaluation") else None,
            "action": result.get("action"),
            "run": result.get("run").model_dump(mode="json") if result.get("run") else None,
        }, indent=2))
        return 0
    finally:
        graph.close()


if __name__ == "__main__":
    raise SystemExit(main())
