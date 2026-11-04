"""Compatibility entry point for canonical Power BI Semantic Layer ingestion.

The old cross-repository glossary JSON ingestion is retired. Use a PBIP root or
Fabric-synced definition directory directly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_bases.semantic_layer.powerbi.cli import main


if __name__ == "__main__":
    main()
