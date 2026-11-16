"""Compatibility entry point for the graph mixture-of-experts orchestrator."""

from knowledge_bases.orchestration.agent import answer, main

__all__ = ["answer", "main"]


if __name__ == "__main__":
    main()
