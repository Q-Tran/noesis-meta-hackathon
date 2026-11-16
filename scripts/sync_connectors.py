"""Sync the default connector catalog into the graph."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph import Graph
from knowledge_bases.orchestration.connectors import ConnectorRepository


def main() -> int:
    graph = Graph()
    try:
        connectors = ConnectorRepository(graph).sync_catalog("scripts/sync_connectors.py")
        print(json.dumps({"connectors": [c.model_dump(mode="json") for c in connectors], "count": len(connectors)}, indent=2))
        return 0
    finally:
        graph.close()


if __name__ == "__main__":
    raise SystemExit(main())
