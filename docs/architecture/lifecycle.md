# Lifecycle: Never Delete, Tag Instead

Noesis's graph never hard-deletes domain data. Nodes that are no longer
current are tagged so default traversals/search skip them, while remaining
fully queryable for anyone who explicitly asks for history. (The only hard
deletes in the codebase are full-database wipes for dev/test — `Graph.wipe()`
— and a handful of one-off re-ingestion scripts that delete-then-recreate a
node from the same source of truth, e.g. `import_powerbi.py`'s snapshot
cleanup. Neither is a "this is now historical"
signal; both are unrelated to the mechanism below.)

There are two mechanisms, chosen by how many nodes change at once.

## 1. Single-entity replacement — `SUPERSEDED_BY` / `SUPERSEDES`

When exactly one node is replaced by exactly one other node, record it as a
relationship, not a property:

```
(old)-[:SUPERSEDED_BY]->(new)
(new)-[:SUPERSEDES]->(old)
```

`old` is left otherwise untouched — no field to set, nothing to keep in sync.
A "give me the current one" traversal simply never follows `SUPERSEDED_BY`;
a "show me the history" traversal does `MATCH (n)-[:SUPERSEDED_BY*]->(old)`
explicitly.

This is already how methodology entity revisions work
(`knowledge_bases/methodology/repository.py:415-470` `_revise_entity`): revising an
entity deletes only the `CURRENT` edge from
the entity to its old `Revision` node, creates a new `Revision`, and links
`old-[:SUPERSEDED_BY]->new`/`new-[:SUPERSEDES]->old`. The old `Revision` node
itself is never deleted. Any other domain that replaces one node with
another (e.g. a redefined `Metric`, a renamed `BusinessTerm`) should use the
same two relationship types rather than inventing a new one.

## 2. Bulk / logical-block deprecation — the `:Deprecated` label

When a whole subgraph is retired at once (e.g. an entire `Project` and every
`Epic`/`UserStory`/`Task` under it), tag every affected node with a second
label instead of a property:

```cypher
MATCH (p:Project {id: $id})-[:HAS_EPIC|HAS_STORY|HAS_TASK*0..]->(n)
SET n:Deprecated
```

Use `Graph.mark_deprecated(match_clause, params, ...)` /
`Graph.reactivate(match_clause, params)` (`graph.py`) rather than writing
this by hand — `match_clause` is caller-supplied Cypher that must bind a
variable named `n` for the node(s) to tag, using whatever containment edges
that domain already has (`HAS_TASK`, `BELONGS_TO`, etc.). There is
deliberately no generic cross-domain "contains" relationship or registry
field for this: hot-path traversals (e.g. "who is working on project X")
keep using the specific edge types they already do, so a label scan for
deprecation never has to interpret a lossy generic edge.

`Graph.mark_deprecated` can optionally also create a `SUPERSEDED_BY`/
`SUPERSEDES` pair pointing the block's head node at whatever replaced it —
the two mechanisms compose (`:Deprecated` marks every member node as
excluded from default queries; `SUPERSEDED_BY` records what to follow for
lineage from the head node).

Why a label and not a boolean property:
- Native label scan, no `coalesce(n.deprecated, false)` null-handling.
- Presence/absence is unambiguous — a label carries no "was this ever set"
  ambiguity the way a missing vs. `false` property can.
- One bulk `SET`/`REMOVE` statement covers an arbitrarily large block in one
  write, regardless of size.

## Domain status fields are untouched (mostly)

Several labels already carry their own richer, domain-specific lifecycle
field — `WriteProposal.status`, `RoutingTrace.status`, `IngestionEvent.status`,
`Tool.active`, `Claim.status`, and `EdgeStatus` (`active`/`refuted`/
`superseded`) on `SUPPORTS`/`CHALLENGES` relationships. These are left
exactly as they are; `:Deprecated`/`SUPERSEDED_BY` is additive, for entities
that don't already have a bespoke field, plus the one exception below.

**`Decision.status`** previously had an `archived` value overloading two
different concerns: "is this decision still current" and "does this decision
need review." `DecisionStatus` now only has `active`/`review` —
archival is `Graph.mark_deprecated("MATCH (d:Decision {id: $id})", {"id": did})`
instead, and `recompute.py`/`review.py`'s "only evaluate live decisions"
guards check `WHERE NOT d:Deprecated` rather than `d.status <> 'Archived'`.

`EdgeStatus` on `SUPPORTS`/`CHALLENGES` **relationships** could not migrate to
this scheme even in principle: Cypher relationships cannot carry labels,
only nodes can. `coalesce(r.status, 'active') = 'active'` guards
(`recompute.py:110,163,169`) are unrelated to this change and remain a plain
relationship property.

## Search/index behavior

- **Neo4j full-text indexes** (`migrations.py`) are not rebuilt or filtered
  at index time — the index lives in the same storage engine as the node, so
  stripping entries would mean destroying the node's own indexed
  properties. Generic search/traversal entry points
  (`knowledge_bases/orchestration/hybrid_search.py`, `knowledge_bases/mcp.py`)
  instead add `WHERE NOT n:Deprecated` at query time, with an explicit
  `include_deprecated=True` opt-in for callers that want historical results.
- **Milvus vector collections** (`vector_store.py`) are a derived,
  rebuildable cache, not a source of truth — when a node is deprecated, its
  vector entry is deleted via `VectorStore.delete_vector(...)`. If reactivated,
  it is simply re-embedded and re-upserted. This is safe precisely because
  deleting a *derived* artifact never violates "never delete graph data."

## Proposing a deprecation (human approval required)

`Graph.mark_deprecated`/`Graph.reactivate` are never called directly from an
agent-facing tool. They're only reached through the same propose -> confirm
-> execute gate every other Noesis write goes through (see
`docs/architecture/knowledge_write_orchestration.md`): a user (or an
LLM acting on their behalf) proposes a change, sees exactly what would be
affected, and only *then* confirms it.

`knowledge_bases/orchestration/lifecycle.py` implements this as two generic,
domain-parameterized `WriteCommand`s (`knowledge_bases/orchestration/commands.py`)
rather than one hand-written command per knowledge base:

- **`orchestration.deprecate_entity`** — `{knowledge_base, label, key,
  cascade?, cascade_depth?, superseded_by?}`. `cascade` names the
  relationship types to walk outward from `key` (e.g. `["HAS_EPIC",
  "HAS_STORY", "HAS_TASK"]` for an entire `project_management.Project`); omitting it
  deprecates exactly the one matched node. `superseded_by` optionally links
  `SUPERSEDED_BY`/`SUPERSEDES` to an *already-existing* replacement node
  (creating a new node in the same step is out of scope — see below).
- **`orchestration.reactivate_entity`** — the inverse, same shape minus
  `superseded_by`.

Both validate `knowledge_base`/`label`/`cascade` against the exact same
`KnowledgeBase.labels`/`relationships` allowlists `knowledge_bases/mcp.py`'s
generic entity/relationship browsing already enforces
(`knowledge_bases/registry.py`) — there is no raw-label or raw-Cypher path
here, matching every other write command's invariant.

Supported knowledge bases: 
`claims`, `strategy`, `project_management`. Deliberately excluded:
`methodology` (already has its own append-only `SUPERSEDED_BY` revisioning),
`tools` (`Tool.active` is already its lifecycle flag), `orchestration`/
`provenance` (infrastructure, not user knowledge), `self_catalog`
(regenerated wholesale by AST introspection; a manual tag wouldn't survive
the next sync).

### Preview before confirming

Because a cascading deprecation's blast radius isn't obvious from its
arguments alone, `deprecate_entity` is the first `WriteCommand` to declare a
`preview` callback (`WriteCommand.preview`, `commands.py`): proposing it runs
a read-only `preview_deprecation()` (count + a capped sample of affected
nodes, no mutation) and stores the result on the `WriteProposal` itself, so
`propose_noesis_write`'s response shows "this will affect 14 nodes: [...]"
before anyone confirms. `reactivate_entity` has no preview since reactivating
an already-tagged block is inherently lower-risk (it only ever *removes* an
exclusion, never removes data).

### Explicit rejection

Every `WriteProposal` (not just deprecations) can now be explicitly rejected
via `orchestration.reject_noesis_write` (`proposal_id`, optional `reason`),
recording `status: 'rejected'`/`rejected_at`/`rejection_reason` — a one-time,
atomic transition exactly like `reserve()`'s. Previously a proposal that
wasn't executed just expired silently after 30 minutes; deprecation's larger
blast radius made an explicit audit trail worth adding for every proposal
type, not only deprecations.

### Vector cleanup

`deprecate_entity`'s dispatch also does the `VectorStore.delete_vector(...)`
cleanup this doc already described as the expected behavior (see "Search/index
behavior" above) — for every affected node whose label has a Milvus
collection (per `hybrid_search.SEARCH_TYPES`), best-effort, so a vector-store
outage never blocks the graph mutation that already happened.

## Out of scope for now

- No generic `contains`/`owns` field on `KnowledgeBase` (`knowledge_bases/registry.py`)
  for discovering block membership — the deprecation command's `cascade`
  argument names relationship types explicitly per call instead (see §2 and
  "Proposing a deprecation" above). This can be revisited if enough callers
  end up passing the same `cascade` list repeatedly.
- Creating a replacement node and deprecating+linking the old one in the same
  proposal — `deprecate_entity`'s `superseded_by` only links to an
  already-existing node (e.g. one created by a prior, separate proposal).
- Re-embedding on `reactivate_entity` — reactivating a node does not restore
  its Milvus vector; that would need re-running the domain's embedding text
  builder, which is domain-specific.
- Physically moving deprecated subgraphs to a slower-tier store is not
  implemented. `:Deprecated` is the trigger condition such a workflow would
  key off of.
