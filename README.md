# Noesis

## The truth isn't enough. You need to know *why* it's the way it is.

Noesis is a graph knowledge engine. In one Neo4j corpus it keeps **what your
organization believes** — the canonical truth — together with **why each belief
is justified** and **where every piece of it came from**. You can ask it
questions, browse it as a map, and correct it.

**1. There is no single truth — only approximations of it.** Even inside one
organization, every team holds its own version. Noesis treats beliefs the way
an argumentative framework does: claims with premises, arguments that attack
and defend, outcomes that can flip. The standing picture is the canonical
truth.

**2. And the canonical truth is always changing.** Data updates, docs change,
strategies shift. The correction almost always arrives at the level of an
individual — then takes months or years to become what the organization
believes. Noesis makes that correction a **first-class, provable act** the
moment one person knows better: corrections are epistemic — challenged or
repointed, never deleted, never re-ingested.

### Live demo (Meta hackathon)

**<http://104.185.152.161:8000>** — landing page at `/`, the corpus explorer +
chat at `/app`. The demo corpus is the
[dev.meta.ai](https://dev.meta.ai) documentation (34k+ nodes, 63k+ edges):
ask it *"Which metrics cover token usage?"*, watch the agent's tool calls
stream in, then try *"Test a correction"* and see the graph change — new nodes
in green, the defeated argument in red, nothing deleted. (The author's Pi,
via a router port-forward of TCP 8000 → the app. If it stops answering, the
Pi is offline or its ISP address changed; everything below reproduces it
locally.)

### Why it matters

| | |
|---|---|
| **$600B+** | spent every year on establishing, verifying and querying business truth — ~$180B in software, and ~$450B+ in payroll for 3.5–5M data professionals, the hidden majority of the market |
| **$3.1T** | the annual cost of bad data to the U.S. economy (IBM) |
| **$12.9M** | what poor data quality costs the average enterprise, per year (Gartner) |

### The correction, step by step

This is the replay on the homepage, over the dev.meta.ai corpus:

1. Today: the docs page says tool search is not available on Chat Completions
   — and the graph believes it, on that evidence.
2. The user says otherwise. A new argument **ATTACKS** the stale one's
   warrant: the docs page predates the launch.
3. The updated fact arrives as a claim — grounded in this conversation, not in
   any docs chunk.
4. A new argument concludes the capability with the update. Nothing deleted,
   nothing re-ingested.
5. The stale claim stays as history — defeated on record, by an argument with
   its own provenance.
6. What the average user actually sees: the canonical truth. The layers
   underneath stay there for whoever asks why.

### Three layers, two models

| Layer | What it holds |
|---|---|
| **Canonical** | what the organization believes: components, capabilities, interfaces, metrics, terms |
| **Epistemology** | why: `Claim`s with premises, `Argument`s that attack/defend and conclude |
| **Provenance** | where it came from: source pages, chunks, routing traces, sessions, spans |

| Job | Model |
|---|---|
| Graph ingestion (docs → claims/arguments) | **Muse Spark 1.3** (`muse-spark-1.3`) |
| The chat agent (bounded tools, streamed traces) | **Muse Wonder** (`muse-wonder-1.0`) |

The chat agent picks its tools through **Meta's tool search** itself: all
tools are sent deferred to the Responses API, the model asks tool search which
ones apply, and only those load — the token-saving feature the corpus is
about, dogfooded by the demo that explains it.

### Host it yourself

```bash
git clone https://github.com/Q-Tran/noesis-meta-hackathon.git
cd noesis-meta-hackathon

docker compose up -d neo4j                 # Neo4j at :7474 / :7687 (neo4j/password)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                       # defaults match compose; no real credentials in this repo
python scripts/download_embedding_model.py # local retrieval weights (~100MB, git-ignored)
python scripts/download_reranker_model.py
python scripts/create_indexes.py

python app.py                              # http://localhost:8000  (landing /, explorer+chat /app)
.venv/bin/python -m pytest tests -q        # 1101 tests, no Neo4j needed for most
```

An empty Neo4j gives you the explorer over an empty corpus; the demo corpus
is built by the ingestion pipeline in `demo/graph_delta/`
(`crawl_meta_docs.py` → `extract_meta_claims.py` → `embed_corpus.py`) — or
just use the live demo above, which runs on the author's machine with the
author's Meta Model API key. **This public snapshot is a toy: it contains no
credentials, no `.env`, no API keys, and no git history — one clean commit.**
The standalone replay player (`demo/graph_delta/index.html`, no server
needed) is in `demo/graph_delta/README.md`.

### What this is, technically

A universal knowledge graph (Neo4j) made of bounded knowledge bases, each
owning its ontology, services, and MCP tool surface — plus a set of
**graph-defined experts** that an external agent harness runs as subagents.
Everything an agent needs is graph data: the tools (`Tool` nodes), the experts
(`KnowledgeBaseExpert` nodes with prompt, allowed tools, routing vocabulary),
the routing decisions (`RoutingTrace`), the conversations themselves
(`Session`), what each expert established (`DelegationResult`), and every tool
call that was made (`Span`/`TraceOutcome`). Because all users share one graph,
an improvement to a tool description or an expert's prompt reaches every user
at once — on the next delegation, with no redeploy and no restart —
and the telemetry that drives self-improvement is pooled.

What may change at runtime and what requires a reviewed code change is
declared once in `knowledge_bases/boundary.py` and enforced by
`tests/test_architecture_boundary.py`: tools, experts, and prompts are data;
the graph ontology, migrations, routing mathematics, write allowlist, and the
generated context export are architecture.

## Contents

| Section | What it covers |
|---|---|
| [Architecture](#architecture) | System topology, then an expansion diagram per box: knowledge bases, orchestration, tools, root pipeline, metrics, ontology, scripts |
| [What Noesis Does](#what-noesis-does) | The four capabilities, with the code behind each |
| [Knowledge Bases and Experts](#knowledge-bases-and-experts) | The 11 knowledge bases, tools vs agent skills |
| [Setup](#setup) | Containers, Python env, schema, catalog sync, Prime registration |
| [Run the demo](#run-the-demo) | End-to-end walkthrough on seeded data |
| [Files](#files) | Repo layout and what each script does |
| [Versioned product development](#versioned-product-development) | Revisions, changesets, review findings |
| [Mixture of Experts](#mixture-of-experts) | Routing, delegation, self-improvement |
| [Ingestion](#ingestion) | Arbitrary information against the described ontology, the automatic reasoning cascade, Power BI semantic models, Microsoft Graph evidence |
| [Operations](#operations) | LLM providers, reranking, adding a data source, deprecations |
| [Self-Modeling](#self-modeling-noesis-about-noesis) | Noesis catalogs its own architecture |

## Architecture

Three things are worth understanding before the diagram: **the graph is the
source of truth for context** (tools, experts, instructions, memory), **Prime
Agent owns execution** (the model, the loop, the terminal), and **the boundary
between data and architecture is enforced in code**
(`knowledge_bases/boundary.py`), not by convention.

### System topology

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ CLIENT                                                                       │
│                                                                              │
│  Prime Agent  —  model, tool-calling loop, terminal                          │
│                                                                              │
│    .prime/agent/skills/noesis-*/SKILL.md            10 stubs                 │
│      frontmatter (name, description)  ··  cached, read at startup            │
│      body                             ··  fetched live from the graph        │
│                                                                              │
└─────────────────────────────┬────────────────────────────────────────────────┘
                             │  MCP over stdio (JSON-RPC)
                             │  tools/list  ·  tools/call
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ NOESIS PYTHON RUNTIME                                                        │
│                                                                              │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ MCP SERVER                                  knowledge_bases/mcp.py       │ │
│ │                                                                          │ │
│ │   tools/list   built from :Tool nodes — never a static list              │ │
│ │   tools/call   dispatch_tool(graph, knowledge_base, name, arguments)     │ │
│ │                                                                          │ │
│ │     1. ToolRepository.load_tools(kb)      resolve tool from graph        │ │
│ │     2. _require_schema_fields(tool, args) enforce declared required      │ │
│ │     3. record_tool_span(...)              open a Span (+result_count)    │ │
│ │     4. invoke_tool(tool.handler, args)    dynamic import, then call      │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ ORCHESTRATION — 9 experts  routing.py · experts_catalog.py · memory.py   │ │
│ │                                                                          │ │
│ │   delegate_noesis_question(question)                                     │ │
│ │     ├─ score_experts(...)  keyword ⊕ vector evidence → fused score       │ │
│ │     │                      every candidate 0.0 → refuse, never guess     │ │
│ │     ├─ ExpertOrchestrator  the expert's own tool subset + prompt         │ │
│ │     └─ writes RoutingTrace + Spans back for self-improvement             │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ 11 KNOWLEDGE BASES → 95 tools, each dispatched by a graph-registered     │ │
│ │ handler string (module:function), resolved at call time                  │ │
│ │                                                                          │ │
│ │   orchestration 33   development 18   tools 10   methodology 10          │ │
│ │   semantic_layer 8   business_dictionary 6   strategy 3                  │ │
│ │   epistemology 4   project_management 3                                  │ │
│ │   architecture · provenance                                              │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ WRITE PATH — the only way domain knowledge changes                       │ │
│ │                                                                          │ │
│ │   18 WRITE_COMMANDS, each propose → confirm → execute                    │ │
│ │   the allowlist itself is an immutable surface (human review only)       │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────┬────────────────────────────────────────────────┘
                             │  Bolt — graph.py · Graph
                             ▼
┌───────────────────┐  ┌──────────────────┐  ┌─────────────────┐  ┌────────────┐
│ Neo4j :7687       │  │ Milvus :19530    │  │ Snowflake       │  │ Kafka :9092│
├───────────────────┤  ├──────────────────┤  ├─────────────────┤  ├────────────┤
│ 166k nodes        │  │ term + insight   │  │ PROD_BDI        │  │ knowledge- │
│ 578k rels         │  │ vectors          │  │ warehouse       │  │ worker     │
│                   │  │                  │  │                 │  │            │
│ SOURCE OF TRUTH   │  │ ACCELERATOR ONLY │  │ read-only SQL   │  │ async      │
│                   │  │ never a          │  │ via 3 bounded   │  │ ingestion  │
│ catalog · ontology│  │ dependency; every│  │ semantic_layer  │  │            │
│ experts · tools   │  │ path degrades if │  │ tools; never    │  │            │
│ skills · memory   │  │ down (warns, no  │  │ mirrored        │  │            │
│ spans             │  │ fail)            │  │                 │  │            │
└───────────────────┘  └──────────────────┘  └─────────────────┘  └────────────┘

                             ▲  reads Spans, writes ReviewFindings
                             │
┌──────────────────────────────────────────────────────────────────────────────┐
│ SELF-IMPROVEMENT LOOP        self_improvement.py · auto_improve.py           │
│                                                                              │
│   4 signals ──► draft (LLM) ──► ReviewFinding ──► propose only, by default   │
│                                                                              │
│     error rate · user corrections · routing confidence · search retries      │
│                                                                              │
│   trigger   any Span with status=error, plus TraceOutcome success=false      │
│   restraint backgrounded (never inline), debounced to 1 run/day, off         │
│             under pytest, and it never edits an immutable surface            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component index

Every component is either in the topology above or in one of these expansions.
Each expansion opens up a single box from that diagram.

| Expansion | Opens up | Covers |
|---|---|---|
| [Knowledge bases](#expands-the-11-knowledge-bases) | the 11-KB box | all 11 packages, 95 tools |
| [orchestration](#expands-orchestration) | the orchestration box | 29 modules, 33 tools |
| [tools](#expands-tools) | part of the 11-KB box | 25 modules, the self-management KB |
| [Root pipeline](#expands-root-modules) | feeds ingestion + reasoning | 14 root modules |
| [Metrics](#expands-metrics) | part of semantic_layer | registration → resolution → audit |
| [Ontology](#expands-the-ontology) | the Neo4j box | 61 labels, 90 relationship types |
| [Scripts & containers](#expands-scripts--operations) | how it is operated | 18 scripts, 6 containers |

### EXPANDS: the 11 knowledge bases

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 11 KNOWLEDGE BASES registered, 95 tools                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ Each package owns its ontology slice, its read queries, and its MCP tool     │
│ surface. The pattern repeats, which is why one diagram covers all eleven:    │
│                                                                              │
│    package.py     the OntologyPackage: labels, rels, search types, seeds     │
│    catalog.py     this KB's seeded Tool definitions                          │
│    ontology_types.py   descriptions + endpoint types for labels/edges        │
│    experts.py     this KB's seeded expert definition                         │
│    kinds.py       the nouns methodology may version, + their rules (if any)  │
│    models.py      pydantic shapes for this domain                            │
│    queries.py     read Cypher (never writes)                                 │
│    mcp_tools.py   thin handlers, named by :Tool.handler in the graph         │
│    mcp_server.py  standalone server, for running this KB on its own          │
│                                                                              │
│   KB                     tools   deviations from that pattern                │
│   ────────────────────── ─────   ─────────────────────────────────────────   │
│   orchestration             33   no queries.py; +23 modules (own diagram)    │
│   tools                     10   +17 modules (own diagram)                   │
│   methodology               10   no queries.py; +repository, workflow,       │
│                                  validation, review_findings, search;        │
│                                  kinds.py composes the versionable nouns     │
│                                  other KBs contribute, phases.py is the      │
│                                  product's phase/persona/gate profile and    │
│                                  profile.py is lookup over it                │
│   business_dictionary        6   exactly the 4-module pattern                │
│   semantic_layer             8   +import_powerbi, lineage, resolution,       │
│                                  metric_registry, snowflake_{session,        │
│                                  queries}, powerbi/ subpackage               │
│   strategy                   3   exactly the 4-module pattern                │
│   development               18   +ir, pipeline, phases, resolution, persist, │
│                                  languages/ — GitNexus-style phase-DAG code  │
│                                  indexing; populated by scripts/dev_catalog. │
│                                  self/ holds the Noesis-about-Noesis reads   │
│   epistemology                 2   +feedback.py, connectors/ subpackage      │
│   project_management         3   +task_runner (bounded unattended runs)      │
│   architecture               0   models + mcp_server only — entities read    │
│                                  via shared search, versioned by methodology │
│   provenance                 0   models + queries only — no tool surface,    │
│                                  read by other KBs rather than dispatched    │
│                                                                              │
│  Shared across all of them, not owned by any:                                │
│    base.py        KnowledgeBase / OntologyPackage / ProductProfile / kinds   │
│    product.py     the installed product: which packages, name, tool target   │
│    registry.py    composes each package.py into the declared ontology        │
│    ontology.py    validation at the single write boundary                    │
│    boundary.py    which surfaces are data and which are architecture         │
│    common.py      shared helpers                                             │
│    mcp.py         dispatch, schema enforcement, span recording               │
│    bridges/       cross-KB edge contracts, declared by describers            │
│    conformance/   the kit an ontology package runs against itself to prove   │
│                   it is legal — ships with core, owns no ontology            │
│    tasks/         personal-scale delivery: Task · TaskList · Project ·       │
│                   Person. DECLARED here, NOT installed by this product —     │
│                   it owns Task, so it cannot compose alongside               │
│                   project_management. The noesis-individual profile          │
│                   installs it instead (tests/test_tasks_ontology.py)         │
│                                                                              │
│  Ingestion and automatic reasoning (ontology-driven, no hardcoded types):    │
│    triplets.py    renders the extraction prompt from the composed registry;  │
│                   validates endpoints, enums, required props                 │
│    ingestion.py   append-only primitive: SUPERSEDES, current_filter()        │
│    ingestion_..py ingest_triplets — writes what fits, reports the rest       │
│    cascade.py     the chain: ingestion→claims→problems→options→              │
│                   strategy→structure→work, ordered by depends_on;            │
│                   starts anywhere. Steps in one wave may run                 │
│                   concurrently (max_workers); waves never do                 │
│    strategy/      strategy.py — the Goal→Decision hop, usable by an          │
│                   agent, by a person, or by a person steering the agent      │
│    development/   open_codebase.py — opening a checkout indexes it if the    │
│                   graph lacks it; on_generate also writes CATALOGS           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### EXPANDS: orchestration

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ EXPANDS: orchestration — 29 modules, 33 tools                                │
├──────────────────────────────────────────────────────────────────────────────┤
│  ROUTING & DELEGATION                                                        │
│    routing.py            score_experts: keyword ⊕ vector → fused score       │
│    routing_eval.py       the golden eval; gates capability wording changes   │
│    service.py            ExpertOrchestrator — runs the chosen expert         │
│    experts_catalog.py    composes the 9 experts (bootstrap_experts)          │
│    experts.py            the orchestrator's own expert definition            │
│    catalog.py            the orchestration tools' seed entries               │
│    ontology_types.py     what its 10 labels / 8 relationships MEAN           │
│    package.py            this KB's OntologyPackage declaration               │
│    expert_admin.py       create / retire / reword an expert at runtime       │
│    repository.py         load experts + write ExpertRevisions                │
│    delegation_results.py what each expert established, per trace             │
│    agent.py              the agent-facing entry point                        │
│    subagents.py          ephemeral subagents for isolated inquiries          │
│                                                                              │
│  SEARCH & RESOLUTION                                                         │
│    hybrid_search.py      vector + keyword + fulltext, ontology-bounded       │
│    resolution.py         entity resolution (exact, then embedding)           │
│    neighborhood.py       entity_neighborhood; MAX_DEPTH = 3                  │
│                                                                              │
│  SESSION & MEMORY                                                            │
│    sessions.py           Session nodes, the conversation record              │
│    memory.py             SessionMemory: prose body + ontology triplets       │
│    memory_models.py      the memory shapes                                   │
│    context_plane.py      what an agent is told, assembled from the graph     │
│                                                                              │
│  AGENT SKILLS (instructions, never dispatched)                               │
│    agent_skills.py           AgentSkill repository                           │
│    agent_skills_catalog.py   regenerate bundles from the tool catalog        │
│    prime_export.py           write the cached SKILL.md stubs                 │
│                                                                              │
│  WRITES & LIFECYCLE                                                          │
│    commands.py           18 WRITE_COMMANDS — the allowlist (immutable)       │
│    write_dispatch.py     WriteCommand + builders packages reuse              │
│    write_service.py      propose → confirm → execute                         │
│    write_repository.py   the actual graph writes                             │
│    lifecycle.py          :Deprecated, SUPERSEDES / SUPERSEDED_BY             │
│                                                                              │
│  models.py · mcp_tools.py · mcp_server.py  (the standard trio)               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### EXPANDS: tools

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ EXPANDS: tools — 23 modules, 10 tools (the self-management KB)               │
├──────────────────────────────────────────────────────────────────────────────┤
│  THE CATALOG (tools describing tools)                                        │
│    catalog.py            bootstrap_catalog: all 95 Tool definitions          │
│    ontology_types.py     what Tool / Span / TraceOutcome MEAN                │
│    package.py            this KB's OntologyPackage declaration               │
│    repository.py         load_tools / sync_tool — the graph is the registry  │
│    resolver.py           invoke_tool: import 'module:function', then call    │
│    catalog_index.py      vector index over tool descriptions, for routing    │
│    catalog_notify.py     push tools/list_changed to live MCP sessions        │
│    search.py             search_noesis_tools                                 │
│    tool_admin.py         activate / deactivate / reword a tool at runtime    │
│                                                                              │
│  TELEMETRY (what happened)                                                   │
│    tracing.py            record_tool_span; triggers improvement on error     │
│    trace_store.py        Span / TraceOutcome writes + signal queries         │
│    active_trace.py       ambient trace + session id, so callers need none    │
│    harness_tracer.py     bridges harness events into graph telemetry         │
│                                                                              │
│  SELF-IMPROVEMENT (what to do about it)                                      │
│    self_improvement.py       4 signals → LLM draft → ReviewFinding           │
│    self_improvement_apply.py apply, re-evaluate, and retire stale findings   │
│    auto_improve.py           the automatic trigger: debounced, background    │
│    source_editor.py          edits its own source, within limits             │
│    git_ops.py                commits an applied change; can revert it        │
│    graph_edits.py            the graph-side equivalent of an edit            │
│    versioning.py             reaches the installed versioning engine by      │
│                              capability, so tools/ ships without methodology │
│                                                                              │
│  store_pressure.py     store size vs page cache — the tiering decision       │
│  models.py · queries.py · mcp_tools.py · mcp_server.py                       │
│                                                                              │
│  Note: source_editor + git_ops are the only modules in the system that       │
│  write code. They are bounded by boundary.py — an immutable surface can      │
│  never be their target, and by default the loop only proposes.               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### EXPANDS: root modules

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ROOT MODULES — 14 files, the pipeline layer                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  INGESTION & EXTRACTION                                                      │
│    extract.py         LLM extraction → entities + ontology-checked triplets  │
│    ingest.py          create a Decision and wire its edges                   │
│    worker.py          Kafka consumer; extraction off the hot path            │
│    review.py          open-world review — the unknown-unknowns catch         │
│                                                                              │
│  REASONING                                                                   │
│    recompute.py       the 4 decision-health signals → ReviewNeeded           │
│    query.py           read paths for decisions, lineage, ownership           │
│                                                                              │
│  INFRASTRUCTURE                                                              │
│    graph.py           Graph — the single Neo4j write boundary                │
│    vector_store.py    Milvus connection + indexing                           │
│    llm.py             provider gateway, retries, redaction                   │
│    config.py          settings                                               │
│    migrations.py      schema: 83 idempotent IF NOT EXISTS statements         │
│                                                                              │
│  ENTRY POINTS / FACADES                                                      │
│    app.py             the web UI: corpus explorer + ask                      │
│    corpus.py          read-only corpus queries powering the web UI           │
│    agent.py           facade → orchestration.agent                           │
│    schema.py          facade → knowledge_bases.<domain>.models               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### EXPANDS: metrics

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ EXPANDS: metrics — registration, resolution, redundancy                      │
├──────────────────────────────────────────────────────────────────────────────┤
│  semantic_layer/metric_registry.py · import_powerbi.py · lineage.py          │
│                                                                              │
│  1. REGISTER          register_metric(...)                                   │
│       fingerprints    normalized_name · definition_fingerprint ·             │
│                       lineage_fingerprint · unit                             │
│       Metric node     72,418 live, keyed by snapshot_id so a reimport        │
│                       replaces rather than accumulates                       │
│                                                                              │
│  2. RESOLVE TO A BUSINESS TERM      the canonical grouping                   │
│       a. exact match on an existing BusinessTerm  (13,176 live)              │
│       b. else vector search over term embeddings  (Milvus)                   │
│       c. else cross-encoder rerank of candidates                             │
│       d. else create a provisional title-cased term, flagged for review      │
│       edge            (BusinessTerm)-[:MAPS_TO]->(Metric)   58,232 live      │
│       aliases         (AlternateBusinessTerm)-[:ALIAS_OF]->(BusinessTerm)    │
│                                                                              │
│  3. AUDIT REDUNDANCY  audit_metric(metric_id, business_term)                 │
│     Compares the new metric against every other metric under the same term:  │
│                                                                              │
│       same definition + same unit      → POSSIBLE_DUPLICATE_OF  score 0.95   │
│       same definition                  → POSSIBLE_DUPLICATE_OF  score 0.85   │
│       same name, different definition  → CONFLICTS_WITH                      │
│       different unit                   → CONFLICTS_WITH                      │
│                                                                              │
│     Live: POSSIBLE_DUPLICATE_OF 155,091 · CONFLICTS_WITH 42,567              │
│           DUPLICATE_OF 0 · VARIANT_OF 0  (declared, promoted by review)      │
│                                                                              │
│  4. LINEAGE           (Metric)-[:COMPUTED_FROM]->(Table)   22,781 live       │
│                       trace_metric_lineage walks it back to columns          │
│                                                                              │
│  Redundancy is represented, never auto-resolved: two teams disagreeing about │
│  a metric is a fact about the business, so it is recorded as an edge for a   │
│  human to adjudicate rather than silently merged.                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

### EXPANDS: the ontology

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ONTOLOGY — 61 labels, 91 relationship types, all described, by owner         │
├──────────────────────────────────────────────────────────────────────────────┤
│  Declared in registry.py, validated in ontology.py at graph.upsert_rel.      │
│  A label or relationship type not listed here cannot be written at runtime.  │
│  Each is DESCRIBED in its package's ontology_types.py — descriptions are     │
│  load-bearing, not docs: extraction targets them. 33 labels / 66 rels are    │
│  extractable; the rest are written by repository code only.                  │
│                                                                              │
│  semantic_layer   Column · Metric · Table · Value                            │
│    BELONGS_TO · COLUMN_OF · COMPUTED_FROM · CONFLICTS_WITH · DEPENDS_ON      │
│    DERIVED_FROM · DUPLICATE_OF · FOUND_IN · JOINS_TO · POSSIBLE_DUPLICATE_OF │
│    RELATED_TO · USES · VALUE_OF · VARIANT_OF                                 │
│                                                                              │
│  business_dictionary   BusinessTerm · AlternateBusinessTerm                  │
│    ALIAS_OF · MAPS_TO                                                        │
│                                                                              │
│  strategy    Decision · Goal · Alternative · Idea · Problem · SuccessMeasure │
│    ADDRESSES · ADVANCES · AFFECTS · CHALLENGES · CHOSEN_OVER · HELD_BY ·     │
│    INTENDED_FOR · MADE_BY · MEASURED_BY · MEASURES · SUPPORTS                │
│                                                                              │
│  epistemology Argument · Claim · DocumentNode · IngestionEvent               │
│ ANSWERS · APPLIES_TO · ASSERTS · AUTHORED · ATTACKS · CHALLENGES             │
│    CONFLICTS_WITH · CONCLUDES · SUPPORTS · USES_PREMISE · WARRANT_OF         │
│                                                                              │
│  provenance   SourceNode · Chunk                                             │
│    FOUND_IN · EXTRACTED_FROM · MENTIONS                                      │
│                                                                              │
│  architecture   Architecture · Component · Interface                         │
│    CONSUMES · DEPENDS_ON · EXPOSES · IMPLEMENTS · REALIZES                   │
│                                                                              │
│  project_management   ProductPlan · Requirement · Capability · Constraint ·  │
│    Specification · AcceptanceCriterion · Person · Project · Epic ·           │
│    UserStory · Task · Iteration · TestCase · ReviewFinding                   │
│    21 rels incl. ASSIGNED_TO · BLOCKED_BY · DELIVERS · HAS_EPIC ·            │
│    HAS_STORY · HAS_TASK · IMPLEMENTS · MUST_SATISFY · SATISFIES · VERIFIES   │
│                                                                              │
│  methodology   versioning machinery: BmadEntity · Revision · WorkflowRun ·   │
│    ChangeSet · Baseline · BaselineRelationship                               │
│ 17 structural rels incl. CURRENT · SUPERSEDES · HAS_BASELINE · HAS_RUN ·     │
│    RECORDED -- the domain nouns it versions are owned above                  │
│                                                                              │
│  orchestration   KnowledgeBaseExpert · ExpertRevision · ExpertCapability ·   │
│    AgentSkill · RoutingTrace · Session · SessionMemory · DelegationResult ·  │
│    WriteProposal                                                             │
│    CAN_USE · CONSIDERED · CURRENT · DELEGATED_TO · DOCUMENTS ·               │
│    HAS_CAPABILITY · HAS_TRACE · RESULTED_IN · SUPERSEDES                     │
│                                                                              │
│  tools   Tool · Span · TraceOutcome            CALLED · JUDGES · PART_OF     │
│                                                                              │
│  development   CodeRepository · CodeFile · CodeFunction · CodeClass ·        │
│    CodeMethod · CodeCommunity · CodeProcess · CodeRoute                      │
│    CALLS · CATALOGS · CONTAINS · FOUND_IN · HANDLES_ROUTE · IMPORTS ·        │
│    INHERITS · MEMBER_OF · OVERRIDES · STEP_IN_PROCESS                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### EXPANDS: scripts & operations

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ SCRIPTS & OPERATIONS — 18 scripts, 6 containers                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  SCHEMA & CATALOG SYNC   (this order, after any catalog change)              │
│    create_indexes.py        indexes + constraints (Revision.id uniqueness)   │
│    sync_tools_catalog.py    tools + experts + agent skills → graph; then     │
│                             re-exports + /reloads Prime if the stubs drifted │
│    export_prime_context.py  regenerate the cached SKILL.md stubs by hand     │
│                                                                              │
│  SEEDING & SELF-MODEL                                                        │
│    seed_self.py             the self-model's reasoning layer                 │
│    dev_catalog.py           index repos, incl. Noesis itself (repo='self')   │
│    produce_mock.py          push mock messages onto Kafka                    │
│                                                                              │
│  INGESTION                                                                   │
│    dev_catalog.py                    index repos into the code graph         │
│    ingest_semantic_models.py         Power BI import                         │
│    ingest_bert_full.py               BERT paper ingest                       │
│                                                                              │
│  SELF-IMPROVEMENT                                                            │
│    self_improve.py    full pass: collect → draft → file → retire stale       │
│    self_recompute.py  decision health for the self-model                     │
│                                                                              │
│  MODELS & LIFECYCLE                                                          │
│    download_embedding_model.py · download_reranker_model.py                  │
│    start_local.sh · stop_local.sh                                            │
│                                                                              │
│  CONTAINERS (docker-compose.yml)                                             │
│    default         neo4j · milvus (+etcd +minio) · kafka (+zookeeper)        │
│    --profile app   knowledge-worker · microsoft-graph-ingestion              │
│                    (image builds fail behind a corporate proxy; infra-only   │
│                     is the normal local mode)                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### What a request actually does

A question arriving from Prime Agent takes this path. Each step names the code
that runs it, so the diagram above stays checkable against reality.

| # | Step | Code |
|---|---|---|
| 1 | Prime calls an MCP tool | `.prime/agent/skills/noesis-*/SKILL.md` (stub) |
| 2 | Server resolves the tool from the graph, not a static list | `mcp.py` `dispatch_tool` |
| 3 | Declared `required` fields are enforced before the handler runs | `mcp.py` `_require_schema_fields` |
| 4 | A `Span` opens, correlated to the active trace/session | `tools/tracing.py` |
| 5 | Routing scores experts on fused keyword + vector evidence | `orchestration/routing.py` `score_experts` |
| 6 | Zero evidence for every expert → refuse rather than guess | same |
| 7 | The chosen expert runs its own tool loop over its tool subset | `orchestration/service.py` `ExpertOrchestrator` |
| 8 | Handler is dynamically imported from the `Tool` node | `tools/resolver.py` `invoke_tool` |
| 9 | Any write goes through propose → confirm → execute | `orchestration/commands.py` |
| 10 | `RoutingTrace`, `Span`, and result counts are persisted | `tools/trace_store.py` |
| 11 | An error span triggers a debounced background improvement pass | `tools/auto_improve.py` |

### The data / architecture boundary

`knowledge_bases/boundary.py` is the file that makes "the graph is the source of
truth" safe. It splits every surface in two, and the split is asserted by tests
rather than documented and hoped for.

**Runtime-mutable (data — the system may change these itself):**
`agent_skill_content`, `domain_knowledge`, `expert_lifecycle`, `expert_prompt`,
`expert_routing_capability`, `new_bounded_tool`, `session_memory`,
`tool_activation`, `tool_description`, `tool_input_schema`.

**Immutable (architecture — human-written code and a reviewed commit only):**
`generated_code_guardrails`, `generated_context_export`, `graph_ontology`,
`knowledge_base_packages`, `memory_ontology_mapping`, `package_composition`,
`product_profile`, `routing_mathematics`, `schema_migrations`,
`search_type_registry`, `versioned_kind_registry`, `workflow_profile`,
`write_command_allowlist`.

The line is drawn at **labels and relationship types**. Adding a node is data;
adding a *kind* of node is schema. So the self-improvement loop may reword a
tool description, but it can never add a relationship type, change the routing
mathematics, or extend the write-command allowlist — and
`require_runtime_mutable` raises if any future code path tries.

`product_profile` and `package_composition` are the widest of them, and the
reason they are listed at all: a product *is* its set of installed ontology
packages (declared once in `knowledge_bases/product.py`), and one package
contributes labels,
relationships, search types, tools, experts, and versionable kinds together. A
runtime path that could install a package would therefore cross every other
immutable line at once. `versioned_kind_registry` and `workflow_profile` are
the methodology's half of that: the creation-traceability rules and the phase
gates `approve_baseline` refuses to pass.

### The core reasoning path

The decision-intelligence spine, drawn from the ontology above. This is the
subgraph the whole system exists to serve.

```
  DATA LAYER — what exists
  ┌──────────┐   :VALUE_OF   ┌──────────┐  :COLUMN_OF  ┌─────────┐
  │  Value   │──────────────►│  Column  │─────────────►│  Table  │
  └──────────┘               └──────────┘              └─────────┘
                                   ▲
                                   │ :USES
                             ┌──────────┐
                             │  Metric  │
                             └──────────┘
                                   │ :SUPPORTS          ┌────────────┐
  KNOWLEDGE LAYER — what it means  │   :COMPUTED_FROM   │ SourceNode │
                                   ▼                    └────────────┘
                            ┌─────────────┐                    │
                            │   Insight   │◄───────────────────┘
                            └─────────────┘  :FOUND_IN {role:"origin"}
                                   │
                    :SUPPORTS  ·  :CHALLENGES   (evidence is an EDGE,
                                   │             never a node)
  DECISION LAYER — what we did     ▼
                            ┌─────────────┐  :INTENDED_FOR  ┌────────┐
                            │  Decision   │────────────────►│  Goal  │
                            └─────────────┘                 └────────┘
                              │  :MEASURED_BY  ──► Insight       │
                              │  :CHOSEN_OVER  ──► Alternative   │ :HELD_BY
                              │  :MADE_BY      ──► Person        ▼
                              └────────────────────────────► ┌────────┐
                                                             │ Person │
                                                             └────────┘
```

- **Neo4j owns the catalog + reasoning layer.** It never mirrors raw rows.
- **Snowflake owns raw rows + aggregation**, reached through the bounded
  read-only `semantic_layer` tools (`run_snowflake_query`,
  `describe_snowflake_table`, `list_snowflake_tables`) once a metric's
  lineage names the table, columns, and filters to query.
- **Evidence is an edge**, not a node: an `Insight` `SUPPORTS` or `CHALLENGES` a
  `Decision`. Assumptions = a `SUPPORTS` edge with a `valid_if` condition.
- **Metrics and external sources both back insights.** Metrics back computed
  insights; a `SourceNode {type: "external"}` (metadata only — no excerpt
  content, see `docs/architecture/provenance.md`) backs imported/open-world
  insights via `FOUND_IN {role: "origin"}`.
- **Business terms are the canonical metric grouping.** Metric insertion maps each
  concrete implementation to a `BusinessTerm`; with no exact match Noesis
  vector-searches, reranks with a cross-encoder, and otherwise creates a
  provisional title-cased term for review.
- **Metric redundancy is in the graph**: `POSSIBLE_DUPLICATE_OF`, `DUPLICATE_OF`,
  `CONFLICTS_WITH`, `VARIANT_OF` between implementations of one business term.
- **Decisions and goals are owned by people.** `Decision` carries `made_at` and
  `MADE_BY`; `Goal` carries `created_at` and `HELD_BY`. **Known bug:** all 20
  `MADE_BY` edges in the live graph point `(Person)->(Decision)`, the reverse of
  what `query.py` reads, so decision ownership silently resolves to nothing. The
  ontology guard validates relationship *type* and *labels* but not *direction*.
  See `docs/architecture/made_by_direction_bug.md`.
- **RAG retrieval is two-stage.** Vector search returns a broad pool, then a
  cross-encoder reranks against the query before returning or accepting a match.

### The 4 decision-health signals (`recompute.py`)

A `Decision` flips to `ReviewNeeded` if any fire:

1. **Premise** — a `SUPPORTS` edge's `valid_if` no longer holds vs its insight.
2. **Outcome** — the `MEASURED_BY` insight breaches the `INTENDED_FOR` goal.
3. **Alternative** — a `CHOSEN_OVER` alternative now out-scores the decision.
4. **Counter-evidence** — a high-weight `CHALLENGES` edge, or net opposing weight
   crosses the erosion limit. Fed by the open-world review (`review.py`, repo root).

## Knowledge Bases and Experts

| Knowledge base | Owned entities | MCP module | Expert (graph key) |
|---|---|---|---|
| Semantic Layer | `Table`, `Column`, `Value`, `Metric` | `knowledge_bases.semantic_layer.mcp_server` | `semantic_layer` |
| Business Dictionary | `BusinessTerm`, `AlternateBusinessTerm` | `knowledge_bases.business_dictionary.mcp_server` | `business_dictionary` |
| Epistemology | `DocumentNode`, `Claim` | `knowledge_bases.epistemology.mcp_server` | `epistemology` |
| Strategy | `Decision`, `Goal`, `Alternative`, `Idea`, `Problem`, `SuccessMeasure` | `knowledge_bases.strategy.mcp_server` | `strategy` |
| Architecture | `Architecture`, `Component`, `Interface` | `knowledge_bases.architecture.mcp_server` | `architecture` |
| Project Management | `ProductPlan`, `Requirement`, `Capability`, `Constraint`, `Specification`, `AcceptanceCriterion`, `Person`, `Project`, `Epic`, `UserStory`, `Task`, `Iteration`, `TestCase`, `ReviewFinding` | `knowledge_bases.project_management.mcp_server` | `project_management` |
| Methodology | versioning machinery + the workflow itself (the domain nouns above are versioned here, not owned) | `knowledge_bases.methodology.mcp_server` | `methodology` |

The user-facing `knowledge_bases.orchestration` layer is the mixture-of-experts
control plane. Neo4j stores `KnowledgeBaseExpert` nodes (each with
`system_prompt`, `capabilities`, `entry_tools`, `allowed_tools` →
`CAN_USE` edges to `Tool`s, and a versioned `ExpertRevision` chain),
`RoutingTrace` nodes, and `CONSIDERED`/`DELEGATED_TO` edges. Experts are
seeded from `knowledge_bases/orchestration/experts_catalog.py` and served live
over MCP (`list_noesis_experts`, `get_noesis_expert`, and the MCP prompts
primitive). See "Mixture of Experts" below for how a harness runs the loop.

The `development` expert owns Noesis's model of *itself*: Noesis's own source is
indexed as one repository in the code graph (`repo='self'`) -- so "which module
implements routing?" and "what breaks if I change this?" are answered from the graph
rather than from a model's recollection of a codebase it read weeks ago. It is
the lowest-priority routable expert, because questions are about the business
far more often than about Noesis.

`knowledge_bases.provenance` (owns `SourceNode`) and `knowledge_bases.tools`
(owns `Tool` plus the `Span`/`TraceOutcome` execution telemetry) are
cross-cutting infrastructure knowledge bases rather than user-routable
experts, so they're not in the table above. Any knowledge base may attach
provenance to its entities via `(:Entity)-[:FOUND_IN {role}]->(:SourceNode)`
— the same edge is the one verbatim-provenance edge, so it also aims at a `Chunk`
(document text) or a `RoutingTrace` (the conversation turn a claim was made on)
— see `docs/architecture/provenance.md`. Every node and edge written through
`graph.upsert_node`/`upsert_rel` carries `created_at`, stamped at that boundary
(community-edition Neo4j has no triggers), so provenance is datable without each
package declaring a timestamp of its own.

`schema.py` remains a compatibility facade. New code should import models from
the owning `knowledge_bases.<domain>.models` module.

### MCP tool discovery is graph-driven

Every `knowledge_bases.<domain>.mcp_server` is a thin connector, not a tool
registry: `list_tools()` and dispatch both resolve live against graph
`Tool` nodes (see `knowledge_bases.mcp.create_domain_server`), not a
hardcoded Python dict. A `Tool` node carries everything needed to build a
real MCP tool -- `description`, `input_schema`, and a `handler` dotted
`module:function` path -- and `knowledge_bases.tools.resolver` dynamically
imports and invokes that handler at call time.

The seed catalog (name, description, schema, handler for every current
tool) lives in `knowledge_bases/tools/catalog.py`. Editing a tool's
definition means editing that file, then re-running
`python scripts/sync_tools_catalog.py` to `MERGE` it into Neo4j -- the
graph, not the Python module, is what any connected harness reads at
runtime. `knowledge_bases.orchestration` is registered as a knowledge base
too and exposes its routing/feedback/write tools through the same
mechanism, so there is exactly one path -- the graph -- from which every
MCP server derives its tool surface.

### Tools vs Agent Skills

Two different things used to share the word "skill". They are now separate
labels, because conflating them is what made the old naming confusing:

| | `Tool` (`tools` KB) | `AgentSkill` (`orchestration` KB) |
|---|---|---|
| What it is | one callable operation | an instruction bundle |
| Has | `handler` + `input_schema` | `description` + `body` |
| Used by | `call_noesis_tool` dispatch | a harness's progressive disclosure |
| Discovered with | `search_noesis_tools` | `list_noesis_agent_skills` |

"Skill" now means what it means in a harness: instructions a model is *given*,
never an operation it can call. An `AgentSkill` has no handler and is never
dispatched, so rewording one cannot widen what any expert can do.

The containment is explicit in the graph: catalog sync writes
`(AgentSkill)-[:DOCUMENTS]->(Tool)` edges for every tool in the skill's
knowledge base, so "a skill contains its tools" is queryable, not just a
convention of the generated body.

### Agent skills are looked up dynamically

A harness loads a skill's frontmatter at startup but reads its **body on
demand**, when the model decides the skill is relevant (verified against Prime
Agent's loader — see `docs/architecture/prime_skill_loading.md`). Noesis uses
that seam: the generated `SKILL.md` is a stub whose body says "fetch the real
instructions from the graph", and

```python
call_noesis_tool("orchestration", "get_noesis_agent_skill",
                 {"key": "noesis-semantic-layer"})
```

renders the body from the live `Tool` catalog at the moment of the call. So a
retired tool or a reworded description is reflected on the next lookup, with no
re-export and no restart.

Only the frontmatter stays a cache, and it has to: a harness needs a name and a
description *before* a session starts and cannot resolve either at runtime. Its
constraints (lowercase-hyphen name matching the directory, non-empty
description within 1024 characters) are validated on write, because a harness
silently drops a skill that violates them.

## What Noesis Does

Four capabilities, each built on the same idea: the graph is the system of
record, and everything else is a cache of it.

### 1. It improves itself from its own telemetry

Every tool call writes a `Span`; every routing decision writes a
`RoutingTrace`; a user correcting an answer writes a `TraceOutcome`. Those are
not logs -- they are the input to `scripts/self_improve.py`, which reads a
7-day window and files `ReviewFinding`s in the reflexive `noesis-self` methodology
project.

Four things make a tool or expert a candidate:

| Signal | Threshold | What it usually means |
|---|---|---|
| Error rate | ≥ 20% over ≥ 5 calls | The tool is broken or misused |
| User corrections | ≥ 2 distinct traces | The answer was wrong, not crashed |
| Routing confidence | low/ambiguous over ≥ 5 traces | Expert capability wording is off |
| **Search retries** | **≥ 3 calls in one turn, in ≥ 2 turns** | **The caller could not express what they wanted** |

The retry signal is the one worth explaining. A caller searching three times
for a single turn never errored, and the user may never have complained -- so
`skill_health`, which sees only errors and corrections, is blind to it. Yet it
is the clearest evidence that something is wrong: they kept asking because they
kept not getting it.

**Which level failed decides the fix**, so the two are separated:

- **Returned nothing** (`empty_calls` majority) -- the caller's vocabulary does
  not match the data. The *tool's description* misdescribes what it covers, so
  people reach for it with the wrong words.
- **Returned the wrong things** -- the search worked and they searched again
  anyway. That points at *scope wording* -- which entities it covers, which
  question it answers -- not at the index.

Each level gets its own drafting prompt, because "improve this description"
without knowing which way it was wrong is not an actionable instruction. This
needed a new `Span.result_count`: a search returning zero rows has
`status = ok` and no error, so nothing that counts failures could see it.

Five kinds apply automatically -- tool wording, routing capability wording,
expert prompt wording, prompt prefixes, and brand-new bounded tools. Each is
scored against the golden routing eval *before* applying, re-measured after a
3-day cooldown, and **auto-reverted if it got worse**. Anything that touches
ontology, migrations, routing mathematics, or the write allowlist is
architecture and routes to a human (`knowledge_bases/boundary.py`).

**A failure triggers a pass automatically.** Two things count as a failure:
a tool call that raises (`Span.status = error`), and a caller explicitly
reporting a bad result (`TraceOutcome.success = false`). The first is the
common case by far -- outcomes are only written when someone reports one.

Two deliberate restraints:

- **Never mid-task.** The pass runs after the failing call has already
  returned, and it files findings rather than editing anything, so nothing
  changes under you while you work.
- **Debounced, and proposes only.** At most one pass a day (the signals use a
  7-day lookback, so more often surfaces nothing new), and an automatic pass
  *files findings without applying them*. Applying edits source and commits to
  git; doing that as a side effect of your task failing is a surprise no
  threshold tuning makes acceptable. `apply_self_improvement` is one call.

The debounce marker lives on the `noesis-self` project, not a local file, so it
is shared across every process using the graph. Set
`NOESIS_AUTO_SELF_IMPROVE=0` to disable. `python scripts/self_improve.py` still
runs the full propose-apply-reevaluate pass on demand.

**Findings expire.** A finding is a claim about the present ("this tool is
failing"), so an open one is re-checked and retired when its signal no longer
holds: the target recovered over enough recent calls, or -- the case that can
never resolve on its own -- the failures sit under a tool key that no longer
exists, so no traffic will ever arrive to dilute the metric. Retirement is a
new revision carrying a `retired_reason`, never a delete, so the history of
what was once wrong stays intact. Only `open` findings are retired; an
`applied` one stays owned by `reevaluate_self_improvements`, which can revert
its source edit.

### 2. Skills live in the graph, not in files

An **Agent Skill** is an instruction bundle a harness gives a model. A **Tool**
is a callable operation with a handler and an input schema. They are different
things, so they are different labels -- which makes it structurally impossible
to confuse them again.

Both are fetched live:

- `SKILL.md` files on disk are **stubs**. They carry frontmatter (`name`,
  `description`) because a harness needs that before a session starts, and a
  pointer that says *fetch the body from the graph*. The body is rendered from
  the current tool catalog at the moment of the call, so a tool added or
  retired since that file was written is reflected immediately -- no
  re-export, no restart.
- Tool descriptions, input schemas, and activation are read live from `Tool`
  nodes on every dispatch (`ToolRepository.load_tools`).
- Handler *code* reloads too: `resolver.py` watches each module's mtime and
  re-imports on change, so editing a handler takes effect on the next call.

The consequence is the point: one central catalog, shared by every user of the
graph. Reword a tool once and every session everywhere sees it on the next
call. Skills are stored centrally and executed programmatically, the same way
the harness runs its own.

### 3. Everything lives on the graph

If the harness's local state were deleted, nothing that matters would be lost:

| What | Where | Written by |
|---|---|---|
| Conversations | `Session` | orchestration |
| Tool calls | `Span` (name, duration, status, error, user, client, expert) | `tracing.py` |
| Routing decisions | `RoutingTrace` + `CONSIDERED`/`DELEGATED_TO` | `routing.py` |
| Outcomes & corrections | `TraceOutcome` | `record_trace_outcome` |
| What each expert established | `DelegationResult` | orchestration |
| Durable lessons | `SessionMemory` (never expires) | `record_noesis_session_memory` |
| Long-term facts | Typed entities + `ExRelationship` triplets | `extract.py` |
| Noesis's own code | `CodeRepository` (`is_self`) + `CodeFile`/`CodeFunction`/... | `scripts/dev_catalog.py` |

`resume_noesis_session` reconstructs a prior conversation from those traces --
a *derived* digest, recomputed from what was recorded, not a saved transcript.

Session memory is deliberately two-tier. The **body is prose**, because a
lesson like "check the container before editing code" is conditional and
ordered, and shredding it into triplets loses exactly what makes it usable. The
**edges are triplets**, so the memory is findable by traversal. Memory never
expires, so correction is append-only: a new node `SUPERSEDES` the old, which
is tagged `:Deprecated` and drops out of retrieval while staying readable as
history.

### 4. GraphRAG search, bounded by the ontology

Two steps, and the second is the reason to use a graph at all.

**Find by meaning** -- `hybrid_search` fuses a lexical channel (Neo4j
full-text) and a semantic one (Milvus embeddings) with reciprocal-rank fusion,
then reranks with a cross-encoder. Scope is always explicit: a caller names the
entity types to search, so results cannot silently span knowledge bases.

**Then find what is related** -- `entity_neighborhood` walks out from a result
and returns its neighbours *grouped by relationship type*, so the answer says
**how** things connect rather than returning an undifferentiated blob:

```json
{"related_by": {"DEPENDS_ON": ["methodology", "core", "epistemology", "tools"],
                "PART_OF":    ["knowledge_bases.orchestration", "..."]}}
```

Both are bounded by the declared ontology, and that bound is enforced at the
write boundary too. `graph.upsert_rel` and `graph.upsert_node` validate their
relationship type and label against `knowledge_bases/registry.py`, because both
interpolate into Cypher and both are reachable from LLM extraction -- which
means an unchecked path could add ontology at runtime. Extraction proposes;
the registry disposes. A triplet naming an undeclared relationship is refused
and returned with a reason, not silently dropped: an extractor repeatedly
reaching for an edge the ontology lacks is precisely the signal that should
reach a human.

Entity resolution runs **id → name → embedding**, so a memory saying
"quarterly forecast of revenue" resolves to an existing "Quarterly Revenue
Forecast" instead of creating a near-duplicate, using the same 0.94 auto-merge
/ 0.85 LLM-arbiter thresholds `extract.py` applies to document ingestion.

Milvus is an accelerator, never a dependency: with it down, search falls back
to keyword-only and resolution to deterministic name matching. Both warn rather
than fail -- which means **a degraded Noesis looks like a working one**, so
check for it explicitly (`docs/operations/local_services.md`).

## Setup

```bash
# 1. Start Neo4j, Milvus, Kafka (+ the knowledge worker)
./scripts/start_local.sh          # or: docker compose up -d --build

# 2. Python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # requirements.txt + pytest

# 3. Config (optional - defaults work offline with LLM_PROVIDER=null)
cp .env.example .env

# 4. Graph schema, then the tool + expert catalogs, then the Prime context export
python scripts/create_indexes.py
python scripts/sync_tools_catalog.py
python scripts/export_prime_context.py   # prints the `prime-agent mcp add` command

# 5. Install Prime Agent and register Noesis (one time)
curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh
prime-agent mcp add noesis --cwd "$PWD" -- "$PWD/.venv/bin/python" \
  -m knowledge_bases.orchestration.mcp_server

# 6. The self-model (optional; needed for self-improvement findings)
python scripts/seed_self.py
```

If the Prime Agent installer fails with `EACCES ... mkdir '/usr/lib/node_modules'`,
npm's global prefix points at a root-owned directory. Point it somewhere you own
rather than installing as root:

```bash
npm config set prefix ~/.local        # ensure ~/.local/bin is on PATH
```

(Note the published command is `prime-agent` via the installer above; there is no
`prime-agent` package on the public npm registry, so `npm i -g prime-agent` 404s.)

Behind a TLS-inspecting proxy, also set `NODE_EXTRA_CA_CERTS` -- Node reads
neither `SSL_CERT_FILE` nor `REQUESTS_CA_BUNDLE`, and without it every HTTPS
call fails as a bare `Connection error.` See
`docs/architecture/prime_agent_setup.md` for that and for the Copilot/Databricks
provider configuration.

Milvus is optional at runtime: routing and tool discovery fall back to
keyword matching when the semantic catalog index is unreachable. Kafka is
only needed for document ingestion (`worker.py`), not for tracing.

For local startup, use the project-owned launchers:

```bash
./scripts/start_local.sh
./scripts/stop_local.sh
```

## Visualizing the Graph

In Neo4j Browser (at `http://localhost:7474`):
1. Go to the **Query Execution** tab on the left sidebar (the console/play icon).
2. Paste any query below and click **Run**.
3. Ensure the results view is set to **Graph** (top right of the results pane).

**See the entire graph:**
```cypher
MATCH (n)-[r]->(m) RETURN n, r, m;
```

**The strategy core (claims vs strategy):**
```cypher
MATCH (i:Insight)-[r:SUPPORTS|CHALLENGES]->(d:Decision)
RETURN i, r, d;
```

**Decision ownership and goals:**
```cypher
MATCH p=(d:Decision)-[:MADE_BY]->(person:Person)
OPTIONAL MATCH p2=(d)-[:INTENDED_FOR]->(g:Goal)-[:HELD_BY]->(owner:Person)
RETURN p, p2;
```

**Insight provenance from metrics and external sources:**
```cypher
MATCH p=(m:Metric)-[:SUPPORTS]->(i:Insight)
RETURN p
UNION
MATCH p=(i:Insight)-[:FOUND_IN {role: "origin"}]->(s:SourceNode)
RETURN p;
```

**Metric audit findings:**
```cypher
MATCH (a:Metric)-[r:POSSIBLE_DUPLICATE_OF|DUPLICATE_OF|CONFLICTS_WITH|VARIANT_OF]->(b:Metric)
OPTIONAL MATCH (term:BusinessTerm)-[:MAPS_TO]->(a)
RETURN term.name AS business_term, a.name AS metric, type(r) AS finding,
       r.reason AS reason, b.name AS related_metric;
```

**Provisional business terms created during metric insertion:**
```cypher
MATCH (t:BusinessTerm)-[:MAPS_TO]->(m:Metric)
WHERE t.review_status = 'needs_review' OR t.created_by = 'metric_registry'
RETURN t.name AS provisional_term, collect(m.name) AS metrics;
```

**The full chain for the compromised DHL decision:**
```cypher
MATCH p = (d:Decision {id:'d_dhl'})-[*1..2]-(x)
RETURN p;
```

**The execution layer (Projects & Blockers):**
```cypher
MATCH p=(person:Person)-[*1..2]-(task:Task)<-[*1..2]-(proj:Project)-[:IMPLEMENTS]->(d:Decision)
RETURN p;
```

**The catalog layer (Tables → Columns → categorical Values):**
```cypher
MATCH (t:Table)<-[:COLUMN_OF]-(c:Column)
OPTIONAL MATCH (c)<-[:VALUE_OF]-(v:Value)
RETURN t, c, v;
```

Ask a question through a harness (see "Mixture of Experts"), or run the
legacy one-shot CLI, which routes and calls each expert's entry skill once
without looping:

```bash
python agent.py "Which decisions need review?"
```

### The graph delta demo

`demo/graph_delta/` replays a recorded session as a graph delta: every node
created, every edge created, every property changed and every edge re-weighted,
in the order the write boundary stamped them, coloured by ontology with
provenance pinned first in the legend. The recording also shows the graph's
justification shape: every knowledge entity is concluded by an `Argument` whose
premises are `Claim`s, and every claim is `FOUND_IN` exactly one `Chunk` or one
`RoutingTrace`. Entities are recorded as atomic bundles — the entity, its
claims, grounds, argument and stance edges are one beat — so the player never
shows an entity before the reason it exists. The recording separates the
**canonical** layer (the clean, 1-hop entities and active edges) from the
**truth** layer (evidence, resolution arguments and claims), and includes a real
`repoint_entity_reference` correction. The corpus is the public Meta Model API
documentation (`dev.meta.ai`): source pages and verbatim chunks in
`demo/graph_delta/sources/`, claims grounded in those chunks, and the docs' own
Muse Wonder vs Muse Wonder Full distinction as the correction — a "Muse Wonder"
mention is repointed from the subscription-only variant to the hosted model,
superseding the wrong `EXPOSES` edge and transferring its property while the
truth layer keeps the rejected resolution as history. A **Corpus** toggle in
the player switches from the focused scene to a map of the full corpus: 204
documentation pages and their 3,194 chunks, grouped by section, with
click-through page details. The full corpus also runs an automated Muse Spark
claim/argument/entity extraction over every chunk (`extract_meta_claims.py`):
**16,240 claims, 8,014 arguments and 3,450 canonical entities**, with 97.6% of
chunks producing at least one claim. Separately, the real Noesis codebase is
indexed through the development ontology (206 files, 1,804 functions, 262
classes, 3,743 calls), and canonical entities are embedded for hybrid entity
search (components, interfaces, capabilities, specifications, terms, metrics,
code). Open `demo/graph_delta/index.html` directly — it is
self-contained, no server or network — or rebuild it:

```bash
set -a && source .env && set +a
python3 demo/graph_delta/record.py   # runs the scenario, diffs the live graph
python3 demo/graph_delta/render.py    # embeds the recording into index.html
```

`record.py` drives the real repositories (`TraceStore`, `graph.upsert_node`,
`graph.upsert_rel`, `ground_claim_in_trace`); the recording is a diff of the
live graph, not authored events. See `demo/graph_delta/README.md` for the
controls and the event model.

## Files

Root-level modules are the shared library code — imported across
`knowledge_bases/`, `scripts/`, and `tests/`. Entry points that are only ever
*run*, not imported, live in `scripts/` (see the table after this one).

| File | Role |
| :--- | :--- |
| `config.py` | env/settings |
| `graph.py` | Neo4j connection + MERGE upsert helpers |
| `schema.py` | compatibility facade re-exporting the `knowledge_bases.<domain>.models` ontology |
| `llm.py` | pluggable LLM (`null` / `openai` / `ollama`) |
| `ingest.py` | define metrics, external info, decisions/goals, and stance edges |
| `extract.py` | Kafka claim-extraction pipeline: raw message → `DocumentNode`/`Insight`/`Claim` |
| `recompute.py` | evaluate the 4 decision-health signals → `ReviewNeeded` |
| `review.py` | open-world scan → external info + external insights + `CHALLENGES` edges |
| `query.py` | graph read paths for decisions, lineage, ownership |
| `vector_store.py` | Milvus vector collections for hybrid search + reranking |
| `agent.py` | legacy one-shot CLI: route → call each expert's entry skill once → narrate (no loop) |
| `app.py` | FastAPI web UI: corpus explorer over `corpus.py` + the ask chat, which serves the demo graph-agent loop (`demo/graph_delta/server.py`) at `/api/ask` (`web/index.html`) |
| `corpus.py` | read-only corpus queries: overview, keyword search, node neighborhoods, recent activity |
| `worker.py` | Kafka worker: consumes raw messages, runs `extract.py`, writes to the graph |
| `migrations.py` | ordered Neo4j constraints and schema migrations |
| `knowledge_bases/ontology.py` | the registry-derived allowlist every runtime write is validated against |

The domain ontology packages (`semantic_layer`, `business_dictionary`, `epistemology`,
`strategy`, `architecture`, `project_management`, `methodology`, `development`,
`tasks`) are NOT in this tree: they ship in `noesis-ontologies`, which declares
its own `noesis.ontologies` entry points. `knowledge_bases.<key>` remains the
import path — the contract is the import name, not the installation directory.
This distribution owns the engine, the control plane (`tools`, `orchestration`,
`provenance`), and the product profile.

The versioned product development workflow lives entirely under
`knowledge_bases/methodology/` like every other knowledge base (source in
`noesis-ontologies/packages/methodology`) — see the table in that section below.

### `scripts/`

CLI entry points — invoked directly (`python scripts/<name>.py`), never
imported as a library elsewhere.

| Script | Role |
| :--- | :--- |
| `scripts/seed_self.py` | seeds the reflexive `noesis-self` methodology project (worked example: a graph-tiering decision) |
| `scripts/self_recompute.py` | files `ReviewFinding`s when the indexed code graph drifts from recorded architecture rationale |
| `scripts/self_improve.py` | files `ReviewFinding`s when execution traces show a skill failing/being corrected or a knowledge base being routed to with low confidence |
| `scripts/ingest_semantic_models.py` | compatibility entry point delegating to `knowledge_bases.semantic_layer.powerbi.cli` |
| `scripts/create_indexes.py` | apply graph constraints/migrations + build vector collections |
| `scripts/sync_tools_catalog.py` | MERGEs the Tool catalog, the expert catalog, and the AgentSkill instruction bundles into Neo4j; then, if `AGENTS.md` or a `SKILL.md` stub has drifted from the code, re-exports them and asks any running Prime agent to `/reload` (`--no-auto-export` to skip) |
| `scripts/export_prime_context.py` | render `.prime/agent/` (`AGENTS.md`, per-KB agent-skill stubs, settings) from the graph, then ask running Prime agents to `/reload` (`--no-reload` to skip) |
| `scripts/download_embedding_model.py` / `scripts/download_reranker_model.py` | fetch/verify local embedding + reranker model weights |
| `scripts/produce_mock.py` | publish mock messages onto the Kafka topics `worker.py` consumes |
| `scripts/start_local.sh` / `scripts/stop_local.sh` | start/stop the local Docker Compose stack |
| `scripts/ingest_bert_full.py` | ingest fulltext, figures, and benchmark scores for the BERT paper |

## Versioned product development

Noesis runs a persona-led product development process (analyst, product
manager, architect, scrum-master, developer -- brainstorm through build) while
persisting its artifacts and workflow state directly in Neo4j. The graph
replaces artifact files; it does not replace the conversations that create and
refine those artifacts.

The split that matters: `knowledge_bases/methodology` owns the *process* --
the versioning machinery (`BmadEntity`/`Revision`/`ChangeSet`/`Baseline`),
phase gates, and workflow facilitation -- while the *nouns* it versions are
owned by the domain knowledge bases that already exist: discovery artifacts
(`Idea`, `Problem`, `Goal`, `SuccessMeasure`) live under `strategy`, solution
structure (`Architecture`, `Component`, `Interface`) under `architecture`, and
everything from product plan to delivery (`ProductPlan`, `Requirement`,
`Specification`, `Epic`, `UserStory`, `TestCase`, `ReviewFinding`, ...) under
`project_management`. A managed entity carries its domain label plus
`:BmadEntity`, so `(:Requirement:BmadEntity)` is a `project_management`
entity whose content lives on its current `Revision`.

| File | Role |
| :--- | :--- |
| `knowledge_bases/methodology/models.py` | versioned workflow request and domain contracts |
| `knowledge_bases/methodology/repository.py` | transactional changesets, claims, revisions, and baselines |
| `knowledge_bases/methodology/validation.py` | phase-specific traceability and readiness gates |
| `knowledge_bases/methodology/workflow.py` | phase/persona workflow state machine |
| `knowledge_bases/methodology/search.py` | vector search + reindexing over managed entities |
| `knowledge_bases/methodology/mcp_server.py` | MCP tool surface, resolved live from graph-registered `Tool` nodes |
| `knowledge_bases/methodology/mcp_tools.py` | MCP-callable handlers adapting the repository/validator/facilitator |

Apply graph migrations after starting Neo4j:

```bash
python scripts/create_indexes.py
```

Start a local stdio MCP server:

```bash
python -m knowledge_bases.semantic_layer.mcp_server
python -m knowledge_bases.business_dictionary.mcp_server
python -m knowledge_bases.epistemology.mcp_server
python -m knowledge_bases.strategy.mcp_server
python -m knowledge_bases.architecture.mcp_server
python -m knowledge_bases.project_management.mcp_server
python -m knowledge_bases.methodology.mcp_server
python -m knowledge_bases.orchestration.mcp_server
```

The MCP surface deliberately exposes typed workflow operations rather than raw
Cypher or generic graph upserts. Material changes create immutable revisions in
an atomic `ChangeSet`. Claims are immutable, phase approval runs validation, and
each `Baseline` snapshots both exact revisions and the relationships among them.

All current managed entities are keyword-searchable through Neo4j full-text
search. Semantic search uses the lean `bmad_entities` Milvus collection for
Problems, Goals, Product Plans, Requirements, Capabilities, Specifications,
Architectures, Decisions, Epics, User Stories, and Review Findings. A stable
vector identity overwrites the previous embedding on revision; immutable
revisions remain auditable through `SUPERSEDES`/`SUPERSEDED_BY` and baseline
relationships.

Cross-domain retrieval is exposed through one orchestration-owned `hybrid_search`
tool. Workflow requests instead use `consult_workflow`, which reconstructs the
active phase, persona, owning domain, approved baselines, complete relevant
artifact context, relationships, and validation findings before selecting the
next facilitation step. Every hybrid-search request
must name one or more qualified entity types from a single knowledge
family, such as `project_management.Requirement` or the related set
`project_management.Project`, `project_management.Epic`,
`project_management.UserStory`, and `project_management.Task`. Domain types
transparently cover both plain and managed (`:BmadEntity`-co-labelled) nodes.
Keyword and semantic candidates are fused only inside that scope, then
property filters and reranking are applied before results are returned.

Retrieval models load from local disk only. Prefetch the reranker with
`python scripts/download_reranker_model.py`; corporate environments can provide
`--endpoint` and `--ca-bundle`, or import an approved package with
`--archive /path/to/model.zip`. Containers see the persisted workspace model at
`/app/models/ms-marco-MiniLM-L-6-v2`.

The local reranker currently uses the canonical `model.safetensors` mirrored by
`Fides-Connect/linkora` on GitHub. Its verified SHA-256 is
`821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`;
the import script rejects a different safetensors weight.

## Mixture of Experts

Noesis does not run an agent loop, and does not try to. It targets **Prime
Agent** exclusively, on a clean split:

| Prime owns execution | Noesis owns context |
|---|---|
| agent loop, Python kernel | which experts exist and what each knows |
| `rlm.spawn` sub-agents | which tools are callable, and how |
| daemon continuity, detach/reattach | durable instructions |
| compaction, heartbeats, schedules | cross-session memory |
| autonomous budgets | what is unresolved, and what improved |

Prime already has sub-agents, a daemon, heartbeats, and bounded autonomous mode.
What it lacks is memory that outlives a checkout -- its harness state is a local
file, per session, per machine. Noesis supplies that half.

```text
prime-agent (owns the loop)
  │  await mcp.call_tool("noesis", "get_noesis_context", {})   # experts, tools,
  │                                              # instructions, memory
  │  route_noesis_question(question)
  │    -> trace_id, ranked candidates, one starting delegation per expert
  │  for each delegation: rlm.spawn(..., name=<expert_key>)
  ▼
child (adopts an expert)
  get_noesis_expert(expert_key)                 # live prompt + allowed tools
  loop until resolved:
    search_noesis_tools / call_noesis_tool / hybrid_search   # -> Span nodes
  record_delegation_result(trace_id, expert_key, ...)
  ▼
parent: get_delegation_results(trace_id) -> synthesize
        record_noesis_outcome(trace_id, success)
```

- `get_noesis_context` is the whole context plane in one call at session start.
  Because it is graph data, an expert created an hour ago and a tool retired this
  morning are already in effect -- no restart, no re-export.
- Setup is one command plus one export. `prime-agent mcp add noesis -- ...`
  registers the server (user-level, because Prime ignores project-level
  `mcpServers` by design), and `scripts/export_prime_context.py` writes
  `.prime/agent/` -- `AGENTS.md`, one Agent Skill per knowledge base, settings.
  Those files are a *cache* of the graph: they change only when the
  orchestrator's immutable base prompt or a knowledge base's registry
  name/description does, and `sync_tools_catalog.py` re-exports them and asks
  any running Prime agent to `/reload` when it detects either -- Prime reads
  skill frontmatter only at startup or on that command.
- The expert roster is editable at runtime: `create_expert` / `revise_expert` /
  `retire_expert` are bounded write commands, gated by the capability-pollution
  guard and the golden routing eval.
- A conversation is a `:Session` node, so `resume_noesis_session` recovers what
  an earlier session established -- derived from its traces, not a saved
  transcript or serialized process state.
- Delegation results are **derived from spans** (`get_delegation_results`), so
  the record of what each expert called and resolved exists whether or not the
  expert reports it; `record_delegation_result` adds only the judgement spans
  cannot hold.
- Every dispatch is a `Span` linked `-[:CALLED]->(:Tool)` and, for routed
  questions, `-[:PART_OF]->(:RoutingTrace)`, tagged with `user_id`,
  `client`, and `expert_key`. `get_noesis_routing_trace` returns routing +
  spans for one `trace_id`.

Metric requests normally traverse:

```text
Business Dictionary -> Semantic Layer -> optional Claims / Strategy / Project Management
```

### Self-improvement

`tools.propose_self_improvements` mines the pooled telemetry (tool error/
correction rates, low-confidence routing, per-expert corrections) and files
`ReviewFinding`s; `apply_self_improvement` applies them and
`reevaluate_self_improvements` keeps or reverts them after a cooldown.
Wording improvements (`tool_wording`, `routing_capability_wording`,
`expert_prompt_wording`) are **graph edits** -- a `Tool.description_override`
or a new `ExpertRevision` carrying one more overlay -- so a catalog re-sync
never clobbers them and every user sees them immediately. Routing wording
additionally passes a vocabulary-pollution guard and a golden routing-eval
gate (`knowledge_bases/orchestration/routing_eval.jsonl`) before it is
written. `new_tool` is the one kind that becomes a test-gated git commit.
See `docs/architecture/self_improvement.md`.

## Ingestion

### Arbitrary information, against the described ontology

The general path. Anything that fits the ontology can be recorded, and nothing
else can:

```text
text -> describe_ontology (the vocabulary, with meanings)
     -> extract triplets against it
     -> ingest_information (writes what fits, returns what did not)
```

`describe_ontology` returns the installed vocabulary — every label and
relationship with what it means, what it is *not* to be confused with, its legal
endpoints, and its properties' value sets. It is returned rather than hardcoded
because it depends on which ontology packages are installed: a narrower product
must be told a narrower ontology, or it extracts types it cannot store. Scope it
with `knowledge_bases` to pay for less of it (~38k characters for everything,
0.3k–14k for one), and note that scoping labels also prunes relationships no
offered label could legally use — relationship descriptions are about 42% of the
bytes, so label-only scoping saves far less than it appears to.

`ingest_information` **writes immediately**. There is no propose/confirm step and
no write-intent detection, because fitting the described ontology *is* the
authorization: the ontology surface is immutable at runtime, so ingestion can add
unlimited data and no schema, and writes are append-only, so a fact that
contradicts an earlier one is related to it rather than overwriting it.

Rejections come back as data instead of raising — one unusable sentence must not
discard a document, and a repeated rejection is the signal that a type is
genuinely missing, which is what `catalog_noesis_feedback` records. The response
also carries `incomplete`: kinds written without the companion edge their own
rules require (a Goal nothing measures, a Decision affecting no architecture).
`submit_changeset` refuses those outright; ingestion records them and says so,
because an unmeasured goal somebody actually stated is still true.

### What the extractor is allowed to extract

The extraction vocabulary is composed, not hardcoded. `extract.py` used to keep
ten Pydantic classes in the engine and `worker.py` a list of three Kafka topics;
both described vocabularies the engine does not own, and both failed quietly -- a
narrower product was offered shapes it could not store (the model filled them, the
write was skipped, and the model call was paid for nothing), and a package that
wanted a new shape had to add a class to engine code.

Each package now declares its own shapes (`extraction_shapes_seed`) and its own
sources (`extraction_sources_seed`), by the same lazy `"module.path:ATTR"` seed
mechanism it already uses for tools, experts, kinds, and capabilities.
`registry.extraction_shapes()` builds the schema `extract.py` offers, and
`registry.extraction_sources()` builds the topic list `worker.py` subscribes to.
Installing a package extends extraction; uninstalling one narrows it.

Composition refuses, rather than degrading: a shape for a label no installed
package declares, a shape with no merge rule, two shapes claiming one label, a
source with no parser, two sources claiming one topic, and a worker with no
streams at all. `tasks` and `project_management` are mutually exclusive, and their
shapes inherit that.

Two write paths, one check: prose goes through a package's parser to a
`NormalizedDocument` and the LLM; a deterministic format (a Power BI TMDL table, a
code scan) emits typed triplets straight to `ingest_triplets`. Both pass the same
`partition()` ontology check. See `docs/architecture/extraction_composition.md`.

### Automatic reasoning (the cascade)

Recording facts is the first hop of a chain that continues on its own:

```text
SourceNode ─► Claim ─► Problem ─► Idea ─┤ human picks ├─► Decision ─► Epic/Task
             epistemology strategy    strategy             project_management
```

Each hop is a `CascadeStep` declared by the package that owns what it *produces*,
so uninstalling a package removes the reasoning that fabricates its entities.
Order comes from `depends_on` rather than a fixed list, and a dependency on an
uninstalled step is ignored — a product installing a subset still gets a runnable
chain.

You can enter anywhere. Handing the planner a bare `SourceNode` plans the whole
chain; handing it an existing `Decision` plans only the work hop.

Two rules make this automation rather than amplification:

- **A step never reads output from its own wave.** Otherwise one wrong Insight
  becomes a Problem, then an Idea, then committed work — a single mistake
  converted into a structure that looks deliberate.
- **Inferences need confirmation by default**, the opposite of
  `ingest_information`. Ingestion records what somebody said; a cascade step
  asserts something nobody said. Writes are attributed to `cascade::<step>` so a
  later pass can tell inference from observation.

The chain deliberately stops before creating a `Decision`. A Decision is a choice
somebody is accountable for — `MADE_BY` points at a Person — and the next hop
turns Decisions into committed work, so fabricating them would fill the graph with
plans traceable to no one. The cascade lays out the options; a human chooses; that
choice immediately re-triggers the work hop.

### Power BI Semantic Models

Power BI acquisition and parsing now live inside the Semantic Layer. The
canonical path is:

```text
PBIP directory or Fabric getDefinition
  -> deterministic TMDL parser
  -> semantic snapshot v2
  -> one Neo4j transaction
```

Ingest local PBIP definitions:

```bash
python -m knowledge_bases.semantic_layer.powerbi.cli local data/powerbi
```

Acquire definitions from Fabric into `data/powerbi` without importing them into
Neo4j:

```bash
python -m knowledge_bases.semantic_layer.powerbi.cli fabric --dry-run
```

The command loads `FABRIC_TENANT_ID`, `FABRIC_CLIENT_ID`, and an optional
`FABRIC_CLIENT_SECRET` from `.env` and uses delegated user authentication. A
confidential web-app registration requires the secret for its authorization-code
exchange; it is not used for app-only authentication. Omit `--dry-run` to also
import the downloaded definitions into Neo4j, or use `--output` to choose a
different download directory.

Fabric does not expose semantic-model modification timestamps through these
item APIs. Normal syncs compare downloaded definition content and leave
unchanged local models untouched. Use `--resume` after an interrupted initial
pull to skip models that already exist without requesting their definitions.

Delegated sign-in opens a browser on the first run and persists an MSAL cache
at `.fabric-token-cache.json` with mode `0600`. Subsequent syncs use the cache.

The merged path does not use the deprecated Snowflake semantic catalog,
Snowpark session, destructive catalog table, or Snowflake-backed MCP search.
Neo4j is authoritative. Snapshot imports include physical sources, model and
column relationships, renames, static rows, explicit measures, and resolved or
unresolved dependencies.

### Microsoft Graph Evidence

Outlook email, Teams chat, and meeting transcripts flow through normalized,
idempotent Kafka events:

```text
Microsoft Graph -> document event -> Kafka -> KnowledgeWorker -> Neo4j
```
The three topics are declared by `epistemology` (`extraction_sources_seed`), not
hardcoded in `worker.py`: the package that owns the labels a stream produces
declares the topic and the parser, and the worker subscribes to what is installed.
The publisher's topic map is derived from the same declaration, so publisher and
worker cannot disagree.

Configure `GRAPH_TENANT_ID` and `GRAPH_CLIENT_ID`, then run. User ID and email
default to the authenticated Microsoft Graph `/me` profile:

```bash
docker compose --profile microsoft-graph up -d
```

OAuth tokens are stored in a permission-restricted MSAL cache and stream
cursors in SQLite. Neither tokens nor continuation URLs are stored in Neo4j or
Snowflake. The old SSE `extract_knowledge` transport and Snowflake checkpoint
table are retired.

## Operations

### LLM providers
Set `LLM_PROVIDER` in `.env`:
- `null` — fully offline; descriptions/extraction degrade to heuristics/no-ops.
- `openai` — set `OPENAI_API_KEY` (+ `OPENAI_MODEL`).
- `ollama` — set `OLLAMA_HOST` / `OLLAMA_MODEL`.

### Reranking
Reranking is enabled by default for insight search, business-term matching,
fuzzy entity resolution, and glossary mapping. Defaults retrieve up to 50 vector
candidates for normal `top_k=5` searches, rerank with
`cross-encoder/ms-marco-MiniLM-L-6-v2`, then return the requested final results.

Environment overrides:
- `ENABLE_RERANKING=true`
- `RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`
- `RERANK_CANDIDATE_MULTIPLIER=10`
- `RERANK_MIN_CANDIDATES=50`
- `RERANK_MAX_CANDIDATES=100`
- `DEDUPE_RERANK_CANDIDATES=25`

### Adding a real data source
Raw-row access is a bounded tool, not a pipeline module: add a read-only
handler next to `knowledge_bases/semantic_layer/snowflake_queries.py`
(Snowflake already reaches the PROD_BDI warehouse), register it in
`knowledge_bases/tools/catalog.py`, and run `scripts/sync_tools_catalog.py`.
Documents flow through the Kafka → `worker.py` → `extract.py` path instead.

### Proposing a Deprecation

Tagging a node (or a cascaded block of nodes) `:Deprecated` — or reactivating
one — goes through the same propose/confirm/execute gate as every other
Noesis write (`docs/architecture/knowledge_write_orchestration.md`): the
`orchestration.deprecate_entity`/`orchestration.reactivate_entity` commands
(`knowledge_bases/orchestration/lifecycle.py`) are generic across every
applicable knowledge base rather than one command per domain, and
`deprecate_entity` additionally stages a read-only preview (affected-node
count + sample) on the `WriteProposal` before anyone confirms it. See
`docs/architecture/lifecycle.md`.

## Self-Modeling (Noesis-about-Noesis)

Noesis catalogs its own architecture and the decisions behind it, using two
bridged paradigms rather than new machinery. Crucially, neither is
self-specific: the structure tier is the same code graph used for every other
repository, and the reasoning tier is the same methodology system used for
every other project. Being "about Noesis" is a *marker*, not an ontology.

- **Structure tier** — `knowledge_bases.development`. Noesis's own source is
  indexed like any other repo, as `CodeRepository {id: "noesis", is_self:
  true}` with `CodeFile`/`CodeFunction`/`CodeClass` children and real `CALLS`
  edges. Pass `repo="self"` to any development tool to scope it here; the
  self-specific reads (components, rationale, overview) live in
  `knowledge_bases/development/self/`, a code boundary rather than an
  ontology boundary. Populated by:
  ```bash
  python scripts/dev_catalog.py sync --repo noesis
  ```
  This replaced a separate `self_catalog` knowledge base
  (`SelfModule`/`SelfComponent`/`SelfInterface`) in migration 23. That tier
  was a strict subset of what `development` already had, and it rotted: 99 of
  its 282 module nodes pointed at deleted files while 76 live files were
  missing entirely. An expert whose one promise is "answer from the committed
  source, never from recollection" was answering from a snapshot of a codebase
  that no longer existed — worse than the gap it was built to close.
- **Reasoning tier** — the existing versioned methodology system, pointed at a
  reflexive `noesis-self` project. `Architecture`/`Decision`/rationale `Claim`
  entities capture *why*, reusing the methodology's existing validation gates
  (`DECISION_WITHOUT_RATIONALE`, etc.) unchanged. Seed a worked example (a
  hot/cold graph-tiering decision) with:
  ```bash
  python scripts/seed_self.py
  ```
- **Bridge** — `(:BmadEntity:Architecture)-[:CATALOGS]->(:CodeRepository)`,
  written with the plain `Graph.upsert_rel` helper (not the methodology changeset
  API, since `CodeRepository` isn't a `BmadEntity`) — the same pattern
  `graph.attach_source`/`FOUND_IN` already uses to bridge across knowledge
  bases. This single edge is the *only* self-specific piece of ontology; the
  two tiers stay in their own paradigms on either side of it.
- **Feedback loop** — `python scripts/self_recompute.py` diffs the current
  code graph against recorded rationale and files an advisory
  `ReviewFinding` when a component's dependencies drift from what any
  `Decision`'s rationale accounted for. It only ever files findings for
  review; it never supersedes a `Decision` automatically (propose-confirm-
  execute, per `docs/architecture/user_feedback_loop.md`).
- **Trace-driven self-improvement** — a sibling feedback loop,
  `python scripts/self_improve.py`, analyzes *execution* traces (`SpanEvent`,
  `TraceOutcome`, `RoutingTrace`) instead of code drift: it flags tools
  with high error/user-correction rates and knowledge bases with
  disproportionately low-confidence routing, has an LLM draft a concrete
  wording/schema suggestion for each, and files it as the same kind of
  advisory `ReviewFinding` in `noesis-self`. It never edits
  `knowledge_bases/tools/catalog.py` or `orchestration/experts_catalog.py`
  automatically. See `docs/architecture/self_improvement.md` and the
   `propose_self_improvements` MCP tool
   (`knowledge_bases.tools.mcp_server`).
