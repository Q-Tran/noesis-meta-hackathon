"""MCP server for graph-backed mixture-of-experts orchestration.

The tool surface (list + dispatch) is resolved live from graph-registered
`Tool` nodes owned by the `orchestration` knowledge base; see
`knowledge_bases.mcp` and `knowledge_bases/tools/catalog.py`. This module
only wires up the server; `text_json` is re-exported as `_json` for
backward compatibility with existing imports/tests.
"""
from __future__ import annotations

from knowledge_bases.mcp import create_domain_server, run_server, text_json as _json
from knowledge_bases.registry import knowledge_base

app, dispatch_tool = create_domain_server(knowledge_base("orchestration"))


def main() -> None:
    run_server(app)


if __name__ == "__main__":
    main()
