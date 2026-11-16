"""Shared restricted MCP machinery for bounded Noesis knowledge bases.

Tool surface is resolved live from graph-registered `Tool` nodes (see
`knowledge_bases.tools`), not hardcoded per server: each domain MCP server
is a thin connector that lists whatever the graph currently says it owns
and dispatches by dynamically importing the registered handler. Adding,
versioning, or retiring a tool means editing the `Tool` catalog (see
`knowledge_bases/tools/catalog.py`) and re-running `sync_catalog()`; it
does not require touching this module or any `mcp_server.py`.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Callable
from uuid import uuid4

import mcp.server.stdio
from mcp.server import NotificationOptions, Server
from mcp.types import GetPromptResult, Prompt, PromptMessage, TextContent, Tool

from graph import Graph, json_safe
from knowledge_bases.registry import KnowledgeBase
from knowledge_bases.tools.active_trace import (
    bind_connection,
    release_connection,
    resolve_identity,
    resolve_trace_context,
)
from knowledge_bases.tools.catalog_notify import register_session
from knowledge_bases.tools.models import SpanKind
from knowledge_bases.tools.repository import ToolRepository
from knowledge_bases.tools.resolver import invoke_tool
from knowledge_bases.tools.tracing import record_tool_span


def _server_knowledge_bases() -> str:
    """The callable knowledge bases named in `NOESIS_SERVER_INSTRUCTIONS`.

    Composed from the installed ontology packages' `call_noesis_tool`
    declarations so the instruction text tracks whatever a product ships
    rather than enumerating one distribution's set.
    """
    from knowledge_bases.registry import call_noesis_tool_knowledge_bases

    return ", ".join(call_noesis_tool_knowledge_bases())


NOESIS_SERVER_INSTRUCTIONS = (
    f"Every Noesis knowledge base ({_server_knowledge_bases()}) exposes bounded, "
    "graph-registered tools -- never guess a tool name, a table/column name, or "
    "hand-parse a DAX/SQL definition yourself when a bounded tool can resolve it. "
    "Call `search_noesis_tools` with a short plain-language description of what "
    "you need whenever the right operation isn't already obvious; it searches the "
    "full tool catalog across every knowledge base and returns each match's real "
    "name, knowledge base, and arguments, ready to call immediately via "
    "`call_noesis_tool`. For a metric or business-term question, resolve Business "
    "Dictionary language (`business_dictionary`/`resolve_business_language`) "
    "before Semantic Layer metric selection (`semantic_layer`/"
    "`resolve_metric_question`), then trace lineage "
    "(`semantic_layer`/`trace_metric_lineage`) before assuming a metric's source "
    "tables aren't captured in the graph. Use `hybrid_search` directly for "
    "exploratory/cross-domain lookups across entity types. Every call is "
    "automatically correlated with the active trace/session; never set "
    "`trace_id`/`session_id` yourself unless deliberately overriding that."
)


# Declared server capabilities. `tools_changed`/`prompts_changed` tell the
# client "this list can change while you are connected, and I will tell you" --
# which is true of every Noesis surface, since tools are `Tool` nodes and
# prompts are `KnowledgeBaseExpert` nodes read live from the graph. A client
# that advertises the capability re-fetches on
# `notifications/tools/list_changed` (OpenCode does); one that does not is
# skipped by the SDK, so declaring this is safe either way. Without it a
# mid-session catalog change is live server-side but invisible to the running
# client until it reconnects (see `knowledge_bases.tools.catalog_notify`).
NOESIS_NOTIFICATION_OPTIONS = NotificationOptions(
    prompts_changed=True, resources_changed=False, tools_changed=True,
)

Handler = Callable[[dict[str, Any], Graph], Any]


def trace_context(arguments: dict[str, Any]) -> tuple[str | None, str]:
    """Resolve the `trace_id`/`session_id` to correlate this call with.

    The calling model is never expected to set either: this always falls
    back to `knowledge_bases.tools.active_trace`'s process-local state --
    the `trace_id` most recently minted by `route_noesis_question`/
    `delegate_noesis_question` in this process, and a `session_id`
    generated once per process -- so correlation happens automatically for
    *any* MCP harness. An explicit `trace_id`/`session_id` in `arguments`
    (ordinary top-level keys, not a separate MCP envelope) only exists as a
    deliberate override of that automatic correlation, and wins when
    present. Handlers that also read `trace_id` as functional input (e.g.
    `get_trace_tool`, `get_noesis_routing_trace_tool`) are unaffected, since
    this never mutates or pops the dict.
    """
    trace_id = arguments.get("trace_id")
    session_id = arguments.get("session_id")
    return resolve_trace_context(
        trace_id if isinstance(trace_id, str) and trace_id else None,
        session_id if isinstance(session_id, str) and session_id else None,
    )


def _short_name(skill_key: str, knowledge_base: str) -> str:
    prefix = f"{knowledge_base}."
    return skill_key[len(prefix):] if skill_key.startswith(prefix) else skill_key


def _require_schema_fields(tool: Any, arguments: dict[str, Any]) -> None:
    """Enforce the `required` list a Tool already declares in its input_schema.

    Every tool publishes its required fields, and nothing checked them, so a
    missing argument reached the handler and surfaced as a bare `KeyError:
    'trace_id'` -- which names the key but not the tool, the caller, or the
    fact that it was *missing* rather than wrong. Callers then retried the
    identical call, because the error gave them nothing to act on.

    Checked here, at the one point every dispatch passes through, so no handler
    has to repeat it.
    """
    schema = getattr(tool, "input_schema", None) or {}
    required = schema.get("required") or []
    missing = [field for field in required if field not in arguments]
    if missing:
        known = ", ".join(sorted(schema.get("properties", {}))) or "none"
        raise ValueError(
            f"{tool.key} requires {', '.join(missing)}, which "
            f"{'was' if len(missing) == 1 else 'were'} not supplied. "
            f"Accepted arguments: {known}."
        )


def _result_count(value: Any) -> int | None:
    """How many results a retrieval call returned, or `None` if not a retrieval.

    A search that returns zero rows has `status = ok` and no error, so nothing
    that counts failures can see it -- yet an empty result is precisely what
    sends a caller round the loop again. Counting it here, at the one dispatch
    point every tool passes through, means no tool has to opt in.
    """
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("results", "matches", "entities", "related_by"):
            inner = value.get(key)
            if isinstance(inner, list):
                return len(inner)
            if isinstance(inner, dict):
                return sum(
                    len(v) if isinstance(v, list) else 1 for v in inner.values()
                )
        if "total" in value and isinstance(value["total"], int):
            return value["total"]
    return None


def _summarize_for_span(value: Any) -> Any:
    """Best-effort JSON-safe capture of a handler's return value for
    `SpanEvent.raw_output`. Never raises: a value tracing can't serialize
    is simply recorded as its string repr rather than failing the dispatch.
    """
    try:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="python")
        return json_safe(value)
    except Exception:
        return repr(value)


def load_tool_definitions(graph: Graph, knowledge_base: str) -> list[Tool]:
    """Build live MCP `Tool` definitions from graph `Tool` nodes.

    This is the dynamic-discovery half of the connector: a harness (or this
    process) calling `list_tools()` gets back whatever is currently active
    in the graph, with no locally hardcoded tool list.
    """
    tools = ToolRepository(graph).load_tools(knowledge_base)
    return [
        Tool(
            name=_short_name(tool.key, knowledge_base),
            description=tool.description,
            inputSchema=tool.input_schema,
        )
        for tool in tools
    ]


def dispatch_tool(
    graph: Graph,
    knowledge_base: str,
    name: str,
    arguments: dict[str, Any],
    *,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
    session_id: str | None = None,
) -> Any:
    """Resolve a Tool's graph-registered handler and invoke it.

    Looks the skill up by its short (unprefixed) name within one knowledge
    base's active catalog, then dynamically imports and calls the handler
    named on that `Tool` node (see `knowledge_bases.tools.resolver`). Every
    call is wrapped in a `SpanEvent` (see `knowledge_bases.tools.tracing`);
    passing the orchestrator's `RoutingTrace.trace_id` correlates the span
    with the routing decision that produced this dispatch.

    `trace_id`/`session_id` are resolved independently when an internal
    caller (e.g. `ExpertOrchestrator.execute`, which always passes its own
    `trace_id` but never a `session_id`) doesn't supply one: each missing
    value falls back to whatever the MCP caller put in `arguments`, and
    ultimately to this process's active trace/session (see `trace_context`,
    `knowledge_bases.tools.active_trace`) -- so a client-side agent driving
    its own tool-calling loop, not just `delegate_noesis_question`, gets
    every call correlated with zero id bookkeeping of its own.
    """
    if trace_id is None or session_id is None:
        fallback_trace_id, fallback_session_id = trace_context(arguments)
        trace_id = trace_id if trace_id is not None else fallback_trace_id
        session_id = session_id if session_id is not None else fallback_session_id
    identity = resolve_identity(arguments)
    tools = ToolRepository(graph).load_tools(knowledge_base)
    for tool in tools:
        if _short_name(tool.key, knowledge_base) == name:
            if not tool.handler:
                raise ValueError(f"Tool {tool.key!r} has no registered handler")
            _require_schema_fields(tool, arguments)
            with record_tool_span(
                knowledge_base, name, arguments,
                skill=tool.key, skill_version=tool.version,
                trace_id=trace_id, parent_span_id=parent_span_id,
                session_id=session_id, **identity,
            ) as span:
                result = invoke_tool(tool.handler, arguments, graph)
                span.raw_output = _summarize_for_span(result)
                span.result_count = _result_count(result)
                return result
    raise ValueError(f"Unknown {knowledge_base} tool: {name}")


def _load_experts(graph: Graph) -> list[Any]:
    from knowledge_bases.orchestration.repository import RoutingRepository

    return RoutingRepository(graph).load_experts(include_unroutable=True)


def text_json(value: Any) -> list[TextContent]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    return [
        TextContent(type="text", text=json.dumps(json_safe(value), indent=2, default=str))
    ]


def list_entities(
    graph: Graph, kb: KnowledgeBase, label: str, limit: int = 50,
    include_deprecated: bool = False,
) -> list[dict]:
    """List entities of `label`. Deprecated nodes (see
    `docs/architecture/lifecycle.md`) are excluded unless `include_deprecated`
    is set, e.g. for an explicit "show history" request."""
    if label not in kb.labels:
        raise ValueError(f"{label} is not owned by {kb.name}")
    limit = max(1, min(limit, 200))
    deprecated_guard = "" if include_deprecated else "WHERE NOT n:Deprecated "
    return graph.run(
        f"MATCH (n:`{label}`) {deprecated_guard}"
        "RETURN properties(n) AS entity LIMIT $limit",
        limit=limit,
    )


def get_entity(
    graph: Graph, kb: KnowledgeBase, label: str, key: str,
    include_deprecated: bool = False,
) -> dict | None:
    if label not in kb.labels:
        raise ValueError(f"{label} is not owned by {kb.name}")
    deprecated_guard = "" if include_deprecated else "AND NOT n:Deprecated "
    rows = graph.run(
        f"MATCH (n:`{label}`) WHERE (n.id = $key OR n.name = $key) "
        f"{deprecated_guard}"
        "RETURN properties(n) AS entity LIMIT 1",
        key=key,
    )
    return rows[0]["entity"] if rows else None


def list_relationships(
    graph: Graph, kb: KnowledgeBase, relationship: str, limit: int = 50,
    include_deprecated: bool = False,
) -> list[dict]:
    if relationship not in kb.relationships:
        raise ValueError(f"{relationship} is not exposed by {kb.name}")
    limit = max(1, min(limit, 200))
    deprecated_guard = "" if include_deprecated else "AND NOT a:Deprecated AND NOT b:Deprecated "
    return graph.run(
        f"MATCH (a)-[r:`{relationship}`]->(b) "
        "WHERE (any(label IN labels(a) WHERE label IN $labels) "
        "OR any(label IN labels(b) WHERE label IN $labels)) "
        f"{deprecated_guard}"
        "RETURN labels(a) AS from_labels, properties(a) AS from, "
        "type(r) AS relationship, properties(r) AS properties, "
        "labels(b) AS to_labels, properties(b) AS to LIMIT $limit",
        labels=list(kb.labels),
        limit=limit,
    )


def create_domain_server(
    kb: KnowledgeBase,
) -> tuple[Server, Callable[[str, dict[str, Any], Graph], Any]]:
    """Build an MCP server whose entire tool surface is read live from the
    graph's `Tool` catalog for `kb`, plus the three generic entity/
    relationship browsing tools every knowledge base gets by ownership
    (`registry.py`). No domain-specific tool list is hardcoded here or in
    any `mcp_server.py`; call `sync_catalog()` to change what a harness sees.
    """
    server = Server(f"noesis-{kb.key.replace('_', '-')}")

    def definitions(graph: Graph) -> list[Tool]:
        base = [
            Tool(
                name=f"list_{kb.key}_entities",
                description=(
                    f"List entities owned by the {kb.name} knowledge base. "
                    "Deprecated entities are excluded by default."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": list(kb.labels)},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                        "include_deprecated": {
                            "type": "boolean",
                            "description": "Include deprecated entities (default false).",
                        },
                    },
                    "required": ["label"],
                },
            ),
            Tool(
                name=f"get_{kb.key}_entity",
                description=(
                    f"Get one entity owned by the {kb.name} knowledge base. "
                    "Deprecated entities are excluded by default."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": list(kb.labels)},
                        "key": {"type": "string"},
                        "include_deprecated": {
                            "type": "boolean",
                            "description": "Include deprecated entities (default false).",
                        },
                    },
                    "required": ["label", "key"],
                },
            ),
            Tool(
                name=f"list_{kb.key}_relationships",
                description=(
                    f"List approved relationships exposed by {kb.name}. "
                    "Relationships touching a deprecated entity are excluded by default."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "relationship": {
                            "type": "string", "enum": list(kb.relationships)
                        },
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                        "include_deprecated": {
                            "type": "boolean",
                            "description": "Include deprecated entities (default false).",
                        },
                    },
                    "required": ["relationship"],
                },
            ),
        ]
        if kb.key == "orchestration" and os.environ.get("NOESIS_PRUNED_SURFACE", "true").lower() in ("1", "true", "yes"):
            # Minimal startup surface for agent harnesses: search discovery, query routing,
            # and universal dispatch. All other capabilities remain in the graph and are
            # discovered via search and invoked via call_noesis_tool.
            primary_tools = {"search_noesis_tools", "route_noesis_question", "call_noesis_tool"}
            all_tools = load_tool_definitions(graph, kb.key)
            return [t for t in all_tools if t.name in primary_tools]
        return base + load_tool_definitions(graph, kb.key)

    def dispatch(name: str, arguments: dict[str, Any], graph: Graph) -> Any:
        trace_id, session_id = trace_context(arguments)
        if name == f"list_{kb.key}_entities":
            with record_tool_span(
                kb.key, name, arguments, kind=SpanKind.tool_call,
                trace_id=trace_id, session_id=session_id,
            ) as span:
                span.raw_output = list_entities(
                    graph, kb, arguments["label"], arguments.get("limit", 50),
                    include_deprecated=arguments.get("include_deprecated", False),
                )
                return span.raw_output
        if name == f"get_{kb.key}_entity":
            with record_tool_span(
                kb.key, name, arguments, kind=SpanKind.tool_call,
                trace_id=trace_id, session_id=session_id,
            ) as span:
                span.raw_output = get_entity(
                    graph, kb, arguments["label"], arguments["key"],
                    include_deprecated=arguments.get("include_deprecated", False),
                )
                return span.raw_output
        if name == f"list_{kb.key}_relationships":
            with record_tool_span(
                kb.key, name, arguments, kind=SpanKind.tool_call,
                trace_id=trace_id, session_id=session_id,
            ) as span:
                span.raw_output = list_relationships(
                    graph, kb, arguments["relationship"], arguments.get("limit", 50),
                    include_deprecated=arguments.get("include_deprecated", False),
                )
                return span.raw_output
        return dispatch_tool(graph, kb.key, name, arguments)

    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        with Graph() as graph:
            return definitions(graph)

    if kb.key == "orchestration":
        # Graph-defined experts are also served through MCP's prompts
        # primitive: a harness that supports prompts can instantiate an
        # expert subagent from `prompts/get <expert.key>` with no Noesis-
        # specific client code; the tool form (`get_noesis_expert`) carries
        # the same data plus the callable skill schemas.
        @server.list_prompts()
        async def handle_list_prompts() -> list[Prompt]:
            with Graph() as graph:
                return [
                    Prompt(name=expert.key, description=expert.description)
                    for expert in _load_experts(graph)
                ]

        @server.get_prompt()
        async def handle_get_prompt(
            name: str, arguments: dict[str, str] | None
        ) -> GetPromptResult:
            with Graph() as graph:
                expert = next((e for e in _load_experts(graph) if e.key == name), None)
            if expert is None:
                raise ValueError(f"Unknown expert prompt {name!r}")
            return GetPromptResult(
                description=expert.description,
                messages=[PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=expert.effective_prompt),
                )],
            )

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> list[TextContent]:
        # Remember this connection's session so a catalog change made *by* this
        # call (retiring a skill, creating an expert) can tell the harness to
        # re-fetch its tool list -- otherwise the change is live in the graph
        # but invisible to the running client (see
        # `knowledge_bases.tools.catalog_notify`).
        _register_notifiable_session(server)
        try:
            with Graph() as graph:
                return text_json(dispatch(name, arguments or {}, graph))
        except Exception as exc:
            return text_json({"error": type(exc).__name__, "message": str(exc)})

    return server, dispatch


def _register_notifiable_session(server: Server) -> None:
    """Best-effort capture of the live session from the request context.

    The low-level server only exposes the session mid-request, so this is
    where it is reachable. Never raises: no session simply means no
    notification (e.g. a direct in-process dispatch, or a test).
    """
    try:
        register_session(server.request_context.session)
    except Exception:  # noqa: BLE001 -- no request context / no session
        pass


async def run_stdio(server: Server) -> None:
    """Serve one stdio connection.

    Binds a correlation scope for it (see
    `knowledge_bases.tools.active_trace`): a stdio server is spawned per
    harness connection, so the process's single scope *is* this conversation.
    Binding explicitly rather than relying on the implicit default means the
    ownership is stated, and a future transport that multiplexes connections
    only has to call `bind_connection` per connection to get isolation.
    """
    connection = os.getenv("NOESIS_CONNECTION_ID") or f"stdio:{uuid4()}"
    bind_connection(connection)
    try:
        async with mcp.server.stdio.stdio_server() as streams:
            init_options = server.create_initialization_options(
                notification_options=NOESIS_NOTIFICATION_OPTIONS,
            )
            init_options.instructions = NOESIS_SERVER_INSTRUCTIONS
            await server.run(streams[0], streams[1], init_options)
    finally:
        release_connection(connection)


def run_server(server: Server) -> None:
    asyncio.run(run_stdio(server))
