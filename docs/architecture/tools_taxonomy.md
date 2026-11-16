# Noesis Tool Taxonomy & Architecture

Noesis tools are organized into a three-tier architecture that mirrors the division of responsibility between the core context engine (**`noesis-core`**) and the specialized domain packages (**`noesis-ontologies`**).

```
┌────────────────────────────────────────────────────────────────────────┐
│ noesis-core                                                            │
│                                                                        │
│   Tier 1: Minimal MCP Gateway Surface                                  │
│   - Directly exposed on primary MCP connection (~400 tokens):          │
│     1. search_noesis_tools (intent discovery over all graph tools)     │
│     2. route_noesis_question (returns expert persona, prompt, & tools) │
│     3. call_noesis_tool (universal execution gateway)                  │
│   - Gated utility tools (callable via call_noesis_tool or search):     │
│     register_tool, get_noesis_context, list_noesis_sessions, writes    │
│                                                                        │
│   Tier 2: Tool Governance & Telemetry (tools.*)                        │
│   - Schema overrides, retiring/reactivating tools                      │
│   - Execution spans, traces, self-improvement loop                     │
└────────────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │ Dispatches / Discovers on demand
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ noesis-ontologies                                                      │
│                                                                        │
│   Tier 3: Specialized Domain Ontologies                                │
│   - semantic_layer.*        (Metrics, dimensional models, lineage)     │
│   - business_dictionary.*   (Glossary terms, business language)        │
│   - project_management.*    (Blockers, epics, tasks)                   │
│   - strategy.*              (Decisions, health, compromised objectives)│
│   - methodology.*           (Rules, workflows, changesets)             │
│   - development.*           (Code intelligence, architectural context) │
│   - epistemology.*          (Claims, provenance, feedback)             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Why a Three-Tier Taxonomy?

1. **Zero-Overhead Token Efficiency**: Connected harnesses (e.g. OpenCode, Prime) only load the minimal **3-tool gateway** (`search_noesis_tools`, `route_noesis_question`, `call_noesis_tool`) into the primary prompt context (~400 tokens). If an engineer is editing CSS or fixing a unit test, Noesis sits completely inert and costs virtually nothing.
2. **One-Stop Subagent Briefing**: `route_noesis_question` bundles the expert's `system_prompt` and search-ranked `suggested_tools` (with full argument schemas). OpenCode can spawn its native subagents immediately with zero redundant round trips.
3. **Progressive Disclosure & Gated Utilities**: All other tools (`register_tool`, `get_noesis_context`, domain queries, writes) live in the Neo4j graph and Milvus vector index. When needed, the agent searches for them and invokes them through `call_noesis_tool`.
4. **Clean Packaging Boundaries**:
   - `noesis-core` owns tool cataloging, search ranking, routing, traces, versioning, and governance.
   - `noesis-ontologies` owns domain models, vocabularies, and domain-specific handlers.
5. **Deferred Namespace for Small-Context Models**: For a provider with hosted tool search (Meta's Muse), the OpenCode V2 plugin `.opencode/plugins/muse-tool-search/index.ts` collapses the ~60 `noesis_*` function schemas into a single namespace description plus a `tool_search` call, so the model loads the schemas it needs on demand. The plugin replays the resulting `tool_search_call`/`tool_search_output` items in history so loaded tools stay callable and the prompt-cache prefix survives. Scoped to provider `meta` on the primary agent loop; nothing changes for other providers.

---

## Tier 1: Core Gateway Surface (`orchestration.*`)

**Repository**: [`noesis-core`](https://github.com/Q-Tran/noesis-core)  
**Surface**: Exposed directly over the primary MCP connection (`knowledge_bases.orchestration.mcp_server`).

### The 3 Gateway Tools
- **`search_noesis_tools`**: Plain-language semantic and keyword search across all 95 active tools in the graph (97 registered; retired tools keep their nodes as history and telemetry and drop out of discovery). Ranks matches using Milvus vector similarity and BM25 token overlap. Returns matching tool keys, descriptions, and JSON schemas.
- **`route_noesis_question`**: Retrieval-based routing that scores graph-defined experts against a user query. **Directly returns**:
  - `primary_expert`: Canonical knowledge base key (e.g. `semantic_layer`).
  - `primary_expert_prompt`: The complete, live system prompt with overlays and rules.
  - `suggested_tools`: Top matching tools complete with their input JSON schemas.
  - `delegations`: Ordered sequence of starting operations with pre-populated arguments.
  - `trace_id`: Observability correlation ID.
- **`call_noesis_tool`**: The universal dispatch gateway to call any domain or orchestration tool across any knowledge base without needing separate MCP connections.

### Gated Control-Plane & Admin Tools (Reachable via `search_noesis_tools` + `call_noesis_tool`)
- **`register_tool`**: Simple runtime endpoint allowing any harness to register a new tool directly into the graph. Persists as a `:Tool` node, updates the vector index, and emits `notifications/tools/list_changed`.
- **`get_noesis_context`**: Complete context plane in one call (active experts, callable tool surface, durable instructions, resumable sessions).
- **`get_noesis_expert`**: Fetches an individual expert's live system prompt, allowed tools, and entry skills.
- **`list_noesis_sessions` & `resume_noesis_session`**: Cross-session memory recovery, deriving conversation digests from recorded traces and delegation results.
- **`get_delegation_results`**: Structured retrieval of expert findings derived automatically from execution trace spans.
- **`create_expert`, `revise_expert`, `retire_expert`**: Runtime administration of graph-defined experts without code deploys.
- **`list_noesis_write_commands`, `propose_noesis_write`, `execute_noesis_write`, `reject_noesis_write`**: Governed propose-confirm-execute pipeline for graph mutations.
- **`catalog_noesis_feedback`**: Records user-provided facts and corrections as `Claim` nodes.

---

## Tier 2: Core Tool Governance & Telemetry (`tools.*`)

**Repository**: [`noesis-core`](https://github.com/Q-Tran/noesis-core)  
**Surface**: Seeded by `knowledge_bases/tools/catalog.py`, callable through `call_noesis_tool` or directly via `tools.mcp_server`.

These tools govern the lifecycle, performance, and self-improvement of all tools in the graph.

### Lifecycle & Schema Overrides
- **`tools.override_skill_input_schema`**: Corrects a tool's JSON argument schema directly in the graph when the seeded schema is incomplete, without requiring a code deploy or restart.
- **`tools.retire_skill`**: Deactivates a misfiring or obsolete tool. Retains the node, history, and telemetry, but removes it from active discovery.
- **`tools.reactivate_skill`**: Restores a previously retired tool to active discovery and dispatch.

### Observability & Telemetry
- **`tools.get_span`**: Retrieves raw input, output, duration, and status for a single tool execution span.
- **`tools.get_trace`**: Fetches every span belonging to one trace, ordered chronologically.
- **`tools.get_trace_metrics`**: Aggregates latency, tool call counts, error rates, and outcome verdicts for a trace.
- *Automated Harness Tracing*: Spans, traces, and session turns are logged automatically by the harness (e.g. OpenCode V2 plugin `.opencode/plugins/noesis-tracer/index.ts` via `knowledge_bases/tools/harness_tracer.py`), creating `:RoutingTrace`, `:Span`, and `:Session` directly in the graph with zero model overhead. Each turn's prompt is recorded verbatim on `RoutingTrace.question` (the plugin decodes the hook's JSON-encoded text, so the stored words are the user's own) -- that node is the graph's single store of what was said, and user claims are grounded in it via `FOUND_IN {role: "origin"}` (see `docs/architecture/provenance.md`).

### Self-Improvement Loop
- **`tools.propose_self_improvements`**: Analyzes execution traces and search retry patterns to propose concrete tool wording fixes, schema adjustments, or new tool drafts (`GeneratedToolDraft`).
- **`tools.apply_self_improvement` & `tools.revert_self_improvement`**: Applies reviewable self-improvement edits or rolls them back if post-apply metrics degrade.

---

## Tier 3: Specialized Domain Ontologies (`<domain>.*`)

**Repository**: [`noesis-ontologies`](https://github.com/Q-Tran/noesis-ontologies)  
**Surface**: Dispatched on demand via `call_noesis_tool` or delegated via `spawn_ephemeral_subagent`.

Each domain package contributes its own bounded operations through its `package.py` and `tools_seed`.

### 1. Semantic Layer (`semantic_layer.*`)
*Tables, columns, metrics, calculations, lineage, and dimensional models.*
- `find_metrics`: Search metrics by keyword, business term, or domain.
- `get_metric`: Fetch a metric's formal calculation, grain, and source expressions.
- `describe_metric`: Full metric metadata with dimensions, owners, and upstream dependencies.
- `trace_metric_lineage`: Trace metric dependencies back to source warehouse tables and columns.
- `audit_semantic_metrics`: Detect orphaned metrics, missing dimensions, or circular dependencies.
- `ingest_powerbi_directory`: Ingest PowerBI/Tabular semantic models into the graph.

### 2. Business Dictionary (`business_dictionary.*`)
*Canonical business terminology, definitions, governance, and metric mappings.*
- `lookup_business_term`: Retrieve official definitions and synonyms for business terms.
- `resolve_business_language`: Map conversational phrases to canonical business terms.
- `metrics_for_business_term`: Find all metrics associated with a business term.
- `map_business_term`: Link terms to technical entities.
- `list_provisional_business_terms`: Review newly ingested or unverified terminology.

### 3. Project Management (`project_management.*`)
*Delivery plans, requirements, epics, stories, and iterations.*
- `list_project_blockers`: Identify unresolved blockers and dependencies across epics and tasks.
- `get_project_hierarchy`: Inspect project breakdown from milestones to deliverables.
- `list_ready_tasks`: Retrieve unblocked tasks ready for execution.

### 4. Development & Code Intelligence (`development.*`)
*Code intelligence, components, call graphs, file trees, and architecture rationales.*
- `noesis_self_overview`: High-level structural overview of the system architecture.
- `list_noesis_components` & `describe_noesis_component`: Inspect modular software components and interfaces.
- `noesis_architectural_rationale`: Retrieve historical architectural decision records (ADRs).
- `inspect_code` & `list_code_repos`: Code navigation, function signatures, and call hierarchy.

### 5. Methodology (`methodology.*`)
*Phase-gated workflows, process rules, and changeset versioning.*
- `consult_workflow`: Query process rules and exit criteria for project phases.
- `list_projects` & `create_project`: Manage methodology tracking projects.
- `start_workflow` & `submit_changeset`: Drive governed changes through review gates.

### 6. Strategy (`strategy.*`)
*Objectives, problems, goals, decisions, and alternative evaluations.*
- `list_compromised_decisions`: Surface decisions whose core assumptions have been invalidated.
- `get_decision_detail`: Inspect alternatives considered, trade-offs, and stakeholder rationale.
- `evaluate_decision_health`: Monitor live health and risk signals for active decisions.

### 7. Epistemology & Claims (`epistemology.*`)
*Statements, evidence, observations, and provenance.*
- `get_claim_provenance`: Trace the source evidence and citations backing a claim.
- `catalog_user_feedback`: Ingest user assertions into the feedback loop.
- `describe_ontology`: Inspect active labels, relationships, and schema constraints.

---

## Summary Matrix

| Tier | Namespace | Owned By | Discovery / Surface | Dispatch Mechanism |
|---|---|---|---|---|
| **Tier 1: Gateway** | `orchestration.*` | `noesis-core` | Direct MCP Server (`noesis`): `search`, `route`, `call` | Native MCP tool call |
| **Tier 2: Governance** | `tools.*` | `noesis-core` | Gated: `search_noesis_tools("tools.*")` | `call_noesis_tool("tools", ...)` |
| **Tier 3: Domain Ontologies** | `<domain>.*` | `noesis-ontologies` | Gated: `search_noesis_tools` / `route_noesis_question` | `call_noesis_tool("<domain>", ...)` |

---

## OpenCode Integration: Toggling & Agent Modes

In OpenCode, Noesis can be configured at two levels:

### 1. Toggleable Startup MCP Tool
In OpenCode's configuration (`opencode.json` or `~/.config/opencode/settings.json`), Noesis is registered as an MCP server:
```json
{
  "mcpServers": {
    "noesis": {
      "command": "python",
      "args": ["-m", "knowledge_bases.orchestration.mcp_server"],
      "enabled": true
    }
  }
}
```
* **Enabled by default with zero penalty**: Because the startup surface is pruned to 3 tools (`search_noesis_tools`, `route_noesis_question`, `call_noesis_tool`), it consumes only ~400 tokens.
* **Turn On/Off on demand**: Can be disabled by toggling `"enabled": false` when working purely on disconnected local code.

### 2. Dedicated Agent Mode (`plan`, `build`, `noesis`)
Harnesses like OpenCode support custom agent modes. You can define a dedicated **`noesis` mode** alongside `plan` and `build`:

* **`build` mode**: Code generation, command execution, and file edits. Noesis MCP remains inert in the background.
* **`plan` mode**: Architectural analysis and roadmap design.
* **`noesis` mode**: Specialized knowledge graph exploration. Its mode instructions direct the agent:
  > *"You are acting in Noesis Mode. Your source of truth is the enterprise knowledge graph. Use `route_noesis_question` to determine expert ownership and retrieve subagent briefs, use `search_noesis_tools` to find domain operations, and dispatch them via `call_noesis_tool`."*

