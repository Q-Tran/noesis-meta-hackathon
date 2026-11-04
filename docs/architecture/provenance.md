# Provenance / Epistemology Layer

Noesis models "where did we learn this" as a graph relationship, not a flat
string property, so provenance can be traced and queried like any other
knowledge. This unifies a pattern that already existed for unstructured
extraction (`extract.py`) and methodology claims (`knowledge_bases/methodology/repository.py`) and extends
it to the semantic layer (`Table`/`Column`/`Value`/`Metric`) and claims
(`Insight`).

`SourceNode` deliberately holds only *metadata about* a source (name, type,
`url`, `observed_at`, `published_at`) — never the source's actual
content/excerpt text. Noesis keeps document/excerpt content out of the graph
database generally; that's why the old `ExternalInfo` node (which had a
`content` field) was folded into `SourceNode` rather than kept as a separate,
content-bearing evidence type (see "What changed" below).

## Graph Model

- `SourceNode {id, name, type, reliability}` (`knowledge_bases.provenance`,
  its own knowledge base — see `knowledge_bases/provenance/models.py`) is the
  canonical provenance node for the *entire* graph, not just documents/users.
  Like `skills`, `provenance` is cross-cutting infrastructure rather than a
  user-routable expert (see `knowledge_bases/orchestration/experts_catalog.py`).
- `(:AnyEntity)-[:FOUND_IN {role}]->(:SourceNode)` is the cross-cutting
  provenance relationship. Any knowledge base may use it to point outward at
  a `SourceNode`; it's owned by the provenance knowledge base but usable
  everywhere (see `knowledge_bases/registry.py`). Use
  `graph.attach_source()`/`graph.trace_sources()` (on `Graph`, alongside
  `upsert_node`/`upsert_rel`) or the thin wrappers in
  `knowledge_bases/provenance/queries.py` to write/read it.
- The same edge is the spine's ONE verbatim-provenance relationship, so its
  target is whatever record holds the words: a `SourceNode` (origin metadata),
  a `Chunk` (verbatim document text — `(Claim)-[:FOUND_IN {role: "origin"}]->(Chunk)`
  from `extract.py`), or a `RoutingTrace` (the verbatim prompt of the
  conversation turn a user claim was made on — see "User conversations as
  provenance" below). There is deliberately no bespoke grounding edge per
  source kind.
- `role` distinguishes which aspect of the entity a given source backs, since
  one node can have more than one source:
  - `"hosted_by"` — which connector/system this data physically lives in
    (`Table`, `Metric` → a `SourceNode {type: "database_connector"}`).
  - `"description"` — how a `Table`/`Column`/`Value`'s description text was
    produced (`SourceNode {type: "llm_description" | "human_edit" | "imported"}`).
  - `"definition"` — whether a `Metric`'s definition is evaluated in-house or
    reported externally (`SourceNode {type: "internal_sql" | "external"}`).
  - `"origin"` — the internal/external classification for `Insight` (an
    externally-sourced `Insight` points at a `SourceNode {type: "external"}`
    carrying that document's `name`/`url`/`observed_at` — this replaced the
    old `ExternalInfo` node).
  - `"extracted"` — provenance from unstructured-text extraction
    (`extract.py`); every entity type it produces gets this edge to the
    `SourceNode` representing the source document/feed.
- Creation time is stamped at the write boundary, not by each package.
  `graph.upsert_node`/`graph.upsert_rel` add `created_at` on `ON CREATE SET`,
  beside `weight`. Neo4j runs in community edition here, so there are no
  database triggers and those two functions are the only place it can be
  recorded once for every writer. Writers that build their Cypher by hand —
  the tool registry, the expert catalog and its `CAN_USE`/`HAS_CAPABILITY`
  edges, the telemetry store's `CALLED`/`PART_OF`/`HAS_TRACE`, and
  `mark_deprecated`'s `SUPERSEDES`/`SUPERSEDED_BY` pair — spell the stamp out
  at each site; `tests/test_creation_time_stamps.py` scans every
  relationship-creating clause under `knowledge_bases/` and fails on one that
  carries no time. A repository that supplies its own `created_at` keeps it —
  the caller's `SET` runs after `ON CREATE SET` — and a re-touch moves only
  `weight`, never the birth time. For the spine that means the grounding edge
  carries `{role, created_at}`: "when did this become known" is answerable for
  a conversation grounding edge as well as a document one. No package declares
  a timestamp on `FOUND_IN`, so the boundary stays the single source (pinned by
  `tests/test_creation_time_full_profile.py`, and the cross-profile stamp by
  `noesis-core/tests/test_live_graph_writes.py`).
- Use `graph.attach_source(label, key, source_id, source_props, role=...)` to
  write one of these edges (it `MERGE`s the `SourceNode` and the edge), and
  `graph.trace_sources(label, key)` to read every source linked to a node.

This is unrelated to `Claim` provenance (`(SourceNode)-[:ASSERTS]->(Claim)`,
`(SourceNode)-[:SUPPORTS|CHALLENGES]->(Claim)`, `(Claim)-[:ABOUT]->(entity)`),
which already modeled user/methodology-rationale provenance correctly — see
`docs/architecture/user_feedback_loop.md`.

## Searchable Chunk Provenance Layer

In addition to metadata-level origin tracking (`SourceNode`), Noesis maintains a verbatim, searchable provenance layer linking extracted entities and relationships back to the source chunks:

- **`Chunk {id, text, source_id, document_id, index, created_at}`** (`knowledge_bases.provenance.models.Chunk`): Represents a discrete excerpt or segment of unstructured text.
- **Milvus Vector Indexing (DiskANN)**: Each chunk is vectorized with the embedding model (`llm.embed(text)`) and indexed into the `"chunks"` collection in Milvus using DiskANN with cosine similarity (`index_type: "DISKANN"`, `metric_type: "COSINE"`), mirroring how entities are embedded and searched.
- **Graph Lineage Edges**:
  - `(:Chunk)-[:FOUND_IN {role: "origin"}]->(:SourceNode)` links the chunk to its parent source metadata.
  - `(:AnyEntity)-[:EXTRACTED_FROM {role: "extracted"}]->(:Chunk)` grounds an entity directly in the chunk text.
  - `(:Chunk)-[:MENTIONS]->(:AnyEntity)` enables forward traversal from chunks to extracted entities.
  - Relationships extracted from unstructured text carry `chunk_id` as an edge property: `(a)-[r {chunk_id: "..."}]->(b)`.
- **Search and Tracing API**:
  - `vector_store.upsert_chunk(id, embedding, metadata)`: Stores chunk embedding and text metadata in Milvus.
  - `vector_store.search_chunks(embedding, top_k)`: Performs semantic vector search over chunks.
  - `graph.attach_chunk(label, key, chunk_id, ...)`: Upserts Chunk node and `EXTRACTED_FROM` / `MENTIONS` edges.
  - `graph.trace_chunks(label, key)`: Finds all chunks grounding an entity.
  - `graph.trace_chunk_entities(chunk_id)`: Traces entities grounded in a chunk.
  - `graph.trace_chunk_relationships(chunk_id)`: Traces relationships backed by a chunk.
  - `graph.trace_relationship_chunk(...)`: Identifies the chunk backing a specific relationship edge.
  - `knowledge_bases.provenance.queries.search_provenance(query, top_k=5)`: Semantic vector search over chunks via DiskANN, hydrated with full graph provenance (chunk node, source node, extracted entities, and backed relationships).

## User conversations as provenance

The conversation itself is already graph data, kept by the tracer:
`(:Session)-[:HAS_TRACE]->(:RoutingTrace)`, where `RoutingTrace.question`
carries the verbatim user prompt (the trace store writes it on every turn
start, including traces first created by a tool-call span merge, and
`RoutingRepository.record_plan` writes it on every routed plan). A
user-sourced claim is substantiated by that turn with exactly one edge, from
`knowledge_bases/provenance/conversation.py`:

- `(:Claim)-[:FOUND_IN {role: "origin"}]->(:RoutingTrace {question})` -- the same
  edge, same role, that `extract.py` gives an ingested claim pointing at its
  verbatim `Chunk`. One hop from the claim to the user's own words.

No `Chunk` copies, no conversation `SourceNode`, no transcript embedding:
the words live once, on the turn. "Why is this data the way it is?" resolves
`(claim)-[:FOUND_IN {role: "origin"}]->(trace.question)`.

The role mapping to ingestion is exact. In ingestion, a claim is substantiated
by the verbatim record -- `(Claim)-[:FOUND_IN {role: "origin"}]->(Chunk)`
(`extract.py`) -- and the chunk's `FOUND_IN` to its `SourceNode` names the
feed. The trace is the conversation's chunk: the verbatim record of one turn.
The conversation's source is the `Session` (`HAS_TRACE`). So the trace plays
the *chunk* role, not the source-node role (`SourceNode` is metadata-only by
its own ontology description), and the spine's one edge carries it: the
`FOUND_IN` contract targets `SourceNode`, `Chunk`, and `RoutingTrace`. A
bespoke edge per source was the first design and was dropped -- one verbatim
provenance edge, aimed at whatever record holds the words, is the rule.

The composition cost of that unification is real and accepted: a contract is
inert unless every endpoint label it names is installed, so a product that
shipped `provenance` without `orchestration` would lose the entire `FOUND_IN`
spine. The shipped profile composes both (see `knowledge_bases/product.py`),
and `tests/test_edge_contracts.py` pins the narrowed-product behaviour as the
place a future product must re-take this trade (`MEASURED_BY`/`REALIZES` are
the inert-by-composition precedent).

`catalog_noesis_feedback` performs the grounding automatically: the trace
resolves from trusted runtime context (the `trace_id` argument, else the
connection's active trace from the last route/delegate), never a
model-supplied identity. Filing succeeds with no trace context, but the
result then reports `grounded: null` with the reason, per claim --
provenance gaps are surfaced, never silent. The epistemology recorder's
`(SourceNode {user})-[:ASSERTS]->(Claim)` and stance edges
(`SUPPORTS`/`CHALLENGES`/`CONFLICTS_WITH`) are unchanged: the claim's
provenance is the turn; its argumentative position is toward the entities it
updates.

## Denormalized routing property

`Table.source` / `Metric.source` (e.g. `"powerbi"`, `"snowflake"`) remain plain
string properties carrying the connector name, mirrored by the
`FOUND_IN {role: "hosted_by"}` edge to the connector's `SourceNode`. There is
no longer a connector registry that resolves them at query time — raw-row
access goes through the bounded read-only tools
(`knowledge_bases/semantic_layer/snowflake_queries.py`) — so the property is
now descriptive provenance only, kept in sync on every write
(`register_metric` in `knowledge_bases.semantic_layer.metric_registry`).
Don't add new flat provenance properties elsewhere.

## What changed

`description_source` (`Table`/`Column`/`Value`), `Metric.source_type`/`url`,
and `Insight`/`ExternalInfo.source_type` were flat enum/string properties with
no linked node. They have been removed from the pydantic models
(`knowledge_bases/semantic_layer/models.py`,
`knowledge_bases/epistemology/models.py`) in favor of the `FOUND_IN` edges above.
`knowledge_bases/common.DescriptionSource` / `SourceType` still exist as the
canonical vocabularies for `SourceNode.type` values (e.g. `"llm"`/`"human"`
map to `SourceNode {type: "llm_description" | "human_edit"}`).

`SourceNode`/`SourceReliability` themselves used to be defined in
`knowledge_bases/epistemology/models.py`. They now live in their own
`knowledge_bases/provenance/` knowledge base (`models.py`, `__init__.py`,
`queries.py`), matching the other domain layers under `knowledge_bases/`
(see `knowledge_bases/registry.py`'s `"provenance"` entry and
`knowledge_bases/bridges/README.md`). `knowledge_bases.epistemology` and
`schema.py` still re-export `SourceNode` for backward compatibility, but new
code should import it from `knowledge_bases.provenance.models`. The claims KB
keeps ownership of everything built *on top of* `SourceNode` (`Claim` and its
`ASSERTS`/`ABOUT`/`ANSWERS`/`CONFLICTS_WITH` relationships) — only the
provenance node itself moved.

`ExternalInfo {id, title, content, source, observed_at, url, published_at}`
(claims knowledge base) was removed outright rather than kept alongside
`SourceNode`: it was a document-shaped `SourceNode` in all but name, and its
`content` field was write-only (never read back) — exactly the kind of
excerpt text that shouldn't be in the graph. `SourceNode` gained `url`,
`observed_at`, and `published_at` to cover what `ExternalInfo` legitimately
added (all metadata, no content). Callers now register a `SourceNode`
directly via `knowledge_bases.provenance.queries.register_source(...)` and
link it to what it backs with `graph.attach_source(..., role="origin")`,
instead of creating an `ExternalInfo` node and a separate `SUPPORTS` edge.

Existing graph data written before this change still carries the old flat
properties. The one-time `backfill_source_relationships.py` that rewired
those legacy properties and `ExternalInfo` nodes into `SourceNode`/`FOUND_IN`
edges has been retired: every deployment that needed it has run it, and the
`ExternalInfo` fold-in itself is part of the `core` migration stream. Legacy
properties left on kept labels are harmless and remain as an audit trail of
the migration.

## Power BI ingestion

`knowledge_bases/semantic_layer/import_powerbi.py` now participates in this
layer too: every `Table`/`Metric` it imports gets a
`FOUND_IN {role: "hosted_by"}` edge to a `SourceNode {name: <report/model
name>, type: "powerbi"}` — one `SourceNode` per Power BI report/semantic
model (`id: "powerbi::<model>"`), alongside the existing `source: "PowerBI"`
flat property (kept, same denormalized-routing-cache exception as above).

`knowledge_bases/semantic_layer/powerbi/catalog.py`'s `source_path`/`sources`
fields (where a measure's *query* references come from — a distinct,
PowerBI-internal concept) are unrelated and out of scope for this layer.

### Intermediate metrics / recomputation

A metric's DAX definition can reference other measures (`[Helper Measure]`).
`build_snapshot`/`import_catalog` already link those via `DEPENDS_ON`, but
previously only resolved references within the *same* semantic model. Power
BI composite/live-connection reports can reference an unqualified measure
brought in from a different, shared dataset — so resolution now falls back to
a cross-model lookup when a reference doesn't resolve locally, tagging the
resulting edge `cross_model: true` (`knowledge_bases/semantic_layer/lineage.py`'s
`populate_metric_dependencies`, the graph-based repair path, does the same).
This ensures every intermediate metric that's actually present in the graph
gets linked in, regardless of which report/model it lives in.

`trace_metric_lineage` walks the full transitive `DEPENDS_ON` chain (capped
at depth 20) plus every base table/column any step in that chain is computed
from, so a metric's complete calculation "recipe" — not just its immediate
dependencies — can be retrieved for exact recomputation.

If the underlying shared dataset a report live-connects to was never scanned
into the same ingestion run (e.g. only the thin report's folder was pointed
at, not the base dataset's), its measures simply don't exist anywhere to
link to — that's a scan-scope gap, not something this resolution logic can
paper over. Point `knowledge_bases/semantic_layer/powerbi/cli.py`'s ingestion
at a root that contains every model a report can reference.

## Argument-Driven Entity Resolution Layer

Natural-language text is inherently context-sensitive and ambiguous (e.g. "Tom" can refer to an existing `Person` or a newly introduced entity). To avoid irreversible premature merges and retain an auditable trail, entity resolution is modeled as structured argumentation (Pattern C) grounded in verbatim chunks:

- **Hypothesis Generation & Top-K Persistence**:
  - During chunk extraction, entity resolution evaluates candidate matches via embedding vector search in Milvus (DiskANN) followed by reranking.
  - The top-$K$ runner-up candidates ($K$ configurable via `RESOLUTION_TOP_K`, default `3`) plus the null hypothesis ("new entity") are persisted upfront as structured `Argument` nodes.
  - The winning candidate is assigned `status: "in"`; runner-up candidates and the null hypothesis are assigned `status: "out"`.
  - An attack edge `(Arg_Winner)-[:ATTACKS {target: "conclusion", reason: "higher_confidence"}]->(Arg_RunnerUp)` records the defeat of each competing hypothesis.
- **Derivation Grounding (`Claim`)**:
  - Each candidate match generates a `Claim` node (`claim_type: "derivation"`, `status: "active"` for winner, `"superseded"` for runner-up).
  - Linked to chunk via `(Claim)-[:FOUND_IN {role: "origin"}]->(Chunk)` and concluded by the argument via `(Argument)-[:CONCLUDES]->(Claim)`.
- **Property Attribution Ledger (`_property_origins`)**:
  - Properties contributed to canonical nodes carry granular attribution stored in a `_property_origins` JSON ledger:
    `{prop_name: {"argument_id": win_arg, "chunk_id": chunk_id, "status": "active"}}`
  - Subsequent ingestion passes merge new property origins without overwriting previous attributions.
- **Relationship Provenance**:
  - Extracted relationships carry deterministic `id`, `status: "active"`, `chunk_id`, `argument_id`, `subject_arg_id`, and `object_arg_id`.
- **Bidirectional Edge Linking & Non-Destructive Repointing (`repoint_entity_reference`)**:
  - When user feedback or downstream adjudication corrects an entity resolution:
    - The runner-up argument for `new_entity` is promoted to `status: "in"`.
    - A feedback argument `Arg_Feedback` is created, and both `Arg_Feedback` and `Arg_New` attack `Arg_Old` (`target: "conclusion"`), setting `Arg_Old.status = "out"`.
    - Canonical relationships are rewired bidirectionally: old edge is tagged `status: "superseded"`, `superseded_by: new_rel_id`, `superseded_at: <ts>`; new edge is tagged `status: "active"`, `supersedes: old_rel_id`, `created_at: <ts>`.
    - Property rollback: properties contributed to `old_entity` solely by the corrected chunk are removed and transferred to `new_entity`.
    - Chunk provenance edges (`EXTRACTED_FROM` / `MENTIONS`) are rewired.
    - If `old_entity` has no remaining active edges or properties, it is tagged `:Deprecated` following the lifecycle policy.
- **1-Hop Fast Path**:
  - Normal operational queries traverse active canonical edges directly without traversing mention webs, preserving 1-hop read performance.

## Out of scope

- `BusinessTerm.created_by`, `Decision.url`, `Task.url`, `Insight.url` are
  similar "who/where this came from" fields that were not folded into
  `FOUND_IN` in this pass; a future pass could extend the same pattern to
  them (e.g. `CREATED_BY` to a `Person`) if useful.

