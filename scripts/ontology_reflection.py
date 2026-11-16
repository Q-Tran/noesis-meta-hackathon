"""Create an ontology reflection proposal from graph pressure evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph import Graph
from knowledge_bases.orchestration.ontology_reflection import OntologyReflector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_key")
    parser.add_argument("--kind", default="add_label")
    parser.add_argument("--target", default="")
    parser.add_argument("--auto-approve", action="store_true", help="Opt this ontology into auto-approved proposals (never auto-installs schema)")
    args = parser.parse_args(argv)
    graph = Graph()
    try:
        proposal = OntologyReflector(graph).reflect(
            args.package_key, kind=args.kind, target=args.target, auto_approve=args.auto_approve,
        )
        print(json.dumps(proposal.model_dump(mode="json"), indent=2))
        return 0
    finally:
        graph.close()


if __name__ == "__main__":
    raise SystemExit(main())
