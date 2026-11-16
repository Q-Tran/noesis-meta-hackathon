"""Neo4j persistence for Tool definitions.

Deliberately thin: versioned, low-cardinality definitions. Execution spans
and trace outcomes are written by `trace_store.py` and link back to these
nodes via `(:Span)-[:CALLED]->(:Tool)`.

Skills are the graph's MCP tool registry: `description`/`input_schema`
carry everything a harness needs to build a live MCP `Tool` definition, and
`handler` points at the Python callable that still does the work. Neo4j
properties cannot store nested maps, so `input_schema` round-trips as a JSON
string on the node and as a `dict` on the `Tool` model.
"""
from __future__ import annotations

import json
from typing import Any

from graph import Graph
from knowledge_bases.tools.models import Tool


class ToolRepository:
    def __init__(self, graph: Graph):
        self.graph = graph

    def sync_tool(self, tool: Tool) -> None:
        """MERGE the seeded definition. A graph-side `description_override`
        (set by self-improvement, see `override_description`) survives
        re-syncs: the seed only ever writes the base `description`."""
        def work(tx: Any) -> None:
            tx.run(
                "MERGE (s:Tool {key: $key}) "
                "ON CREATE SET s.created_at = datetime() "
                "SET s.name = $name, s.version = $version, "
                "s.knowledge_base = $knowledge_base, s.active = $active, "
                "s.description = $description, s.input_schema = $input_schema, "
                "s.handler = $handler, s.updated_at = datetime()",
                key=tool.key,
                name=tool.name,
                version=tool.version,
                knowledge_base=tool.knowledge_base,
                active=tool.active,
                description=tool.description,
                input_schema=json.dumps(tool.input_schema),
                handler=tool.handler,
            ).consume()

        self.graph.execute_write(work)

    def override_input_schema(
        self, key: str, input_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Set (or clear, with `None`) the graph-side `input_schema` override.

        Same shape as `override_description`: the seeded base schema is left
        alone so `sync_catalog()` cannot clobber the override, and the
        previous value is returned so a caller can revert exactly.

        Validation lives in `validate_input_schema_override` and must have run
        already -- a schema that drops an argument the handler reads would turn
        every call into a `KeyError` at dispatch.
        """
        rows = self.graph.run(
            "MATCH (s:Tool {key: $key}) "
            "WITH s, s.input_schema_override AS previous_override "
            "SET s.input_schema_override = $input_schema, s.updated_at = datetime() "
            "RETURN previous_override",
            key=key,
            input_schema=None if input_schema is None else json.dumps(input_schema),
        )
        if not rows:
            raise ValueError(f"Unknown skill {key!r}")
        previous = rows[0].get("previous_override")
        return {"previous_override": json.loads(previous) if previous else None}

    def set_active(self, key: str, active: bool) -> dict[str, Any]:
        """Retire or reactivate one skill.

        `load_tools()` filters on `active`, and both tool discovery and
        dispatch go through it (`knowledge_bases.mcp.load_tool_definitions`/
        `dispatch_tool`), so this is the whole mechanism: a retired skill
        stops being listed and stops being callable, while its node, its
        history, and every `Span` that ever called it stay intact.
        """
        rows = self.graph.run(
            "MATCH (s:Tool {key: $key}) "
            "WITH s, s.active AS previous "
            "SET s.active = $active, s.updated_at = datetime() "
            "RETURN previous",
            key=key, active=active,
        )
        if not rows:
            raise ValueError(f"Unknown skill {key!r}")
        return {"key": key, "active": active, "previously_active": rows[0]["previous"]}

    def override_description(self, key: str, description: str | None) -> dict[str, Any]:
        """Set (or clear, with `None`) the graph-side description override
        for one skill. Returns the previous override and previous effective
        description so a caller can record them for revert."""
        rows = self.graph.run(
            "MATCH (s:Tool {key: $key}) "
            "WITH s, s.description_override AS previous_override, "
            "coalesce(s.description_override, s.description) AS previous "
            "SET s.description_override = $description, s.updated_at = datetime() "
            "RETURN previous_override, previous",
            key=key, description=description,
        )
        if not rows:
            raise ValueError(f"Unknown skill {key!r}")
        return dict(rows[0])

    @staticmethod
    def _to_skill(row: dict[str, Any]) -> Tool:
        row = dict(row)
        raw_schema = row.get("input_schema")
        row["input_schema"] = json.loads(raw_schema) if raw_schema else {
            "type": "object", "properties": {},
        }
        return Tool.model_validate(row)

    def load_tools(self, knowledge_base: str | None = None) -> list[Tool]:
        rows = self.graph.run(
            "MATCH (s:Tool {active: true}) "
            "WHERE $knowledge_base IS NULL OR s.knowledge_base = $knowledge_base "
            "RETURN s.key AS key, s.name AS name, s.version AS version, "
            "s.knowledge_base AS knowledge_base, s.active AS active, "
            "coalesce(s.description_override, s.description) AS description, "
            "coalesce(s.input_schema_override, s.input_schema) AS input_schema, "
            "s.handler AS handler "
            "ORDER BY s.key",
            knowledge_base=knowledge_base,
        )
        return [self._to_skill(row) for row in rows]
