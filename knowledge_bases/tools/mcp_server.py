"""Skills/Tools/Traces MCP server.

The tool surface (list + dispatch) is resolved live from graph-registered
`Tool` nodes owned by this knowledge base; see `knowledge_bases.mcp` and
`knowledge_bases/tools/catalog.py`. This module only wires up the server.
"""
from __future__ import annotations

from knowledge_bases.mcp import create_domain_server, run_server
from knowledge_bases.registry import knowledge_base

app, dispatch_tool = create_domain_server(knowledge_base("tools"))


def main() -> None:
    run_server(app)


if __name__ == "__main__":
    main()
