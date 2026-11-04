# Graph-Driven MCP Connectors

Every `knowledge_bases.<domain>.mcp_server` is a **connector to the graph**,
not a local tool registry. `list_tools()` and dispatch both resolve live
against graph `Tool` nodes owned by `knowledge_bases.tools`; no domain
`mcp_server.py` hardcodes a `{name: (description, schema, handler)}` dict.
A harness talking to any Noesis MCP server always sees exactly what is
currently active in the graph — restart the graph's data, and every
connected harness's tool surface changes with it, with no code deploy.

## Why

Before this, each `mcp_server.py` declared a static Python `TOOLS` dict.
`knowledge_bases.tools` already stored `Tool`/`Tool` nodes in Neo4j, but
only as thin telemetry/governance metadata (`key`, `version`,
`knowledge_base`, `active`) — not enough to reconstruct a callable MCP tool.
The tool surface and the graph's record of it had drifted into two separate
sources of truth. This closes that gap: the graph is now the *only* place a
tool's shape (name, description, JSON schema) and its binding to code
(`handler`) are declared for runtime purposes.

## Graph Model

- `Skill {key, name, version, knowledge_base, active, description,
  input_schema, handler}` (`knowledge_bases/tools/models.py`) — one node
  per MCP-invokable capability. `input_schema` is stored as a JSON string
  property (Neo4j properties can't nest maps) and round-trips to a `dict`
  through `ToolRepository`. `handler` is a dotted `"module:function"`
  path to a Python callable with signature `(arguments: dict, graph: Graph)
  -> Any`; only the *registration* lives in the graph, the implementation
  code does not move.
- `key` is namespaced `"<knowledge_base>.<short_name>"` (e.g.
  `"business_dictionary.lookup_business_term"`); the short name (after the
  prefix) is what the MCP tool is actually called as, so multiple knowledge
  bases can each have their own `list_*`/`get_*`-shaped tool without
  colliding.
- `Tool` nodes (unchanged shape, now also carrying `description`) and the
  `(:Tool)-[:USES]->(:Tool)` edge remain a separate, coarser-grained record
  of which underlying tool (e.g. a shared warehouse/LLM capability) a skill
  depends on; they are not resolved into MCP tool definitions themselves.

## Runtime Path

1. **List** — `knowledge_bases.mcp.load_tool_definitions(graph, kb_key)` runs
   `ToolRepository.load_tools(kb_key)` and builds one `mcp.types.Tool` per
   active `Tool`, using its stored `description`/`input_schema` verbatim.
   `create_domain_server()` prepends the three generic entity/relationship
   browsing tools every knowledge base gets from `registry.py` ownership,
   then returns this as the server's `list_tools()` response.
2. **Dispatch** — `knowledge_bases.mcp.dispatch_tool(graph, kb_key, name,
   arguments)` re-loads that knowledge base's skills, matches the tool's
   short name, and calls `knowledge_bases.tools.resolver.invoke_tool` on
   its `handler` path. The resolver (`importlib.import_module` + `getattr`,
   `functools.lru_cache`d per path) is the one place a string becomes a live
   callable.
3. **Seed** — `knowledge_bases/tools/catalog.py`'s `bootstrap_catalog()` is
   the only place tool definitions still exist in Python, as declarative
   seed data (mirroring `orchestration/experts_catalog.py`'s `CAPABILITIES`
   pattern). `sync_catalog()` `MERGE`s each entry via
   `ToolRepository.sync_tool`. Run `python scripts/sync_tools_catalog.py`
   after editing the catalog; nothing reads `catalog.py` at MCP-serving
   time — only `list_skill_tools`/`dispatch_tool` do, straight from Neo4j.

```
catalog.py (seed data, Python)
      │  sync_catalog()
      ▼
Neo4j :Tool nodes  ──── one source of truth for every harness ────┐
      │  load_tools(kb)                                            │
      ▼                                                              │
mcp_server.py  list_tools()/call_tool()  ◄────────────────────── resolver.py
      │                                                    importlib.import_module
      ▼
knowledge_bases/<domain>/mcp_tools.py  (unchanged implementation code)
```

## What did not change

- Business logic: every `*_tool` function in the new
  `knowledge_bases/<domain>/mcp_tools.py` modules is a thin adapter that
  calls the exact same `knowledge_bases/<domain>/queries.py` (or
  equivalent) functions the old hardcoded lambdas called. No query, no
  validation rule, and no write-governance behavior changed.
- The orchestrator's propose/confirm/execute write flow
  (`docs/architecture/knowledge_write_orchestration.md`): commands still
  resolve through the same per-domain dispatch, which now itself resolves
  through `dispatch_tool` — one dispatch path shared by reads and writes.
- `knowledge_bases.orchestration` is registered as its own knowledge base
  in `registry.py` (owning `KnowledgeBaseExpert`/`ExpertCapability`/
  `RoutingTrace`/`WriteProposal`) so its routing/feedback/write tools go
  through this same mechanism instead of a bespoke static list.

## Adding, changing, or retiring a tool

1. Add or edit the corresponding function in `knowledge_bases/<domain>/
   mcp_tools.py` (or reuse an existing `queries.py` function directly).
2. Add or edit its `Tool` entry in `knowledge_bases/tools/catalog.py`
   (`description`, `input_schema`, `handler` path).
3. Run `python scripts/sync_tools_catalog.py`.
4. To retire a tool without deleting history, set `active=false` on its
   bootstrap entry and re-sync — inactive skills are still queryable via
   `knowledge_bases.tools.mcp_server` (`get_span`/`get_trace` etc.) but
   `load_tools()` (and therefore every connector's `list_tools()`) filters
   `{active: true}`.

There is deliberately no MCP *client* code added by this change — Noesis
still only hosts servers (see `knowledge_bases/mcp.py`); an external harness
(Prime Agent, via `prime-agent mcp add noesis`) is the client
that benefits from the dynamic surface.

## Every dispatch is a recorded span

`dispatch_tool()` doesn't just resolve and invoke a handler — it wraps that
invocation in `knowledge_bases.tools.tracing.record_tool_span`, which
writes a `Span` node (`kind=skill_call`, name, timing, status/error,
`raw_input`/`raw_output`, plus `user_id`/`client`/`expert_key`) straight
into the graph via `knowledge_bases.tools.trace_store.TraceStore`. The
span is linked `-[:CALLED]->(:Tool)` to the skill it ran and, when the
`trace_id` came from `route_noesis_question`, `-[:PART_OF]->(:RoutingTrace)`,
so the routing decision (which experts were considered/delegated to) and
what each delegation actually did are one connected subgraph.
`get_noesis_routing_trace_tool` returns both plus the aggregate
`get_trace_metrics` verdict, so one MCP call is the full trace of one
routed question. Turn completion and outcomes (`TraceOutcome`) are appended
automatically by the harness or via `TraceStore.insert_outcome`.

Telemetry lives in the graph — not a per-machine file behind a message
queue — so every user's harness, every MCP server process, and the
self-improvement loop see the same data concurrently with no worker to
deploy. Tracing is best-effort and can never fail a real dispatch: a
missing driver or an unreachable graph is logged and swallowed
(`record_tool_span`'s `finally`). Set `ENABLE_SKILL_TRACING=false` to
disable span writes entirely.

A domain MCP server invoked directly by a harness — not through
`noesis-orchestrator` — still gets a span per dispatch; `dispatch_tool`
generates a fresh `trace_id` when none is passed, so every skill call is
traceable on its own, not only when it's part of a routed question.

## Experts are graph data too

The same principle covers the agents that *use* the tools:
`KnowledgeBaseExpert` nodes carry each expert's `system_prompt`,
`capabilities` (routing vocabulary), `entry_tools`, and `allowed_tools`
(→ `CAN_USE` edges), versioned as `ExpertRevision`s. The orchestration
server serves them via `list_noesis_experts`/`get_noesis_expert` and the
MCP prompts primitive, and `scripts/export_prime_context.py` renders them
into harness agent files. See `docs/architecture/experts.md`.
