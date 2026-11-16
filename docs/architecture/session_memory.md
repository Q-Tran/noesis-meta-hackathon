# Session memory

Durable lessons from a session, in the graph rather than in a harness's local
files. Built because the existing self-improvement loop structurally cannot see
this class of learning, and because a harness that keeps its own refinement
state puts that state outside the graph.

## Why not the self-improvement loop

`run_self_improvement` gathers exactly three signals:

    skill_signals   = collect_skill_health_signals(store, since)   # Span error rates
    routing_signals = collect_routing_gap_signals(...)             # low-confidence routing
    expert_signals  = collect_expert_correction_signals(store, since)

All three are telemetry about *Noesis calling its own tools*. A lesson like
"a label rename doesn't rename the data" is learned in bash, git, and an editor
and produces zero Noesis spans, so the loop is blind to it. That is a different
instrument, not a broken one.

## Two tiers

**The body is prose.** It is what gets read back. "When the MCP server fails to
connect, check the container before editing code" is conditional and ordered;
decomposing it into triplets loses the condition and the ordering that make it
actionable, and scatters it so it cannot be reassembled into "what should I know
about working here".

**The edges are triplets**, and they exist so a memory is findable by traversal
rather than only by text match. They are constrained to ontology the registry
already declares. Two rules make that meaningful:

- *Closed vocabulary.* Predicate and both labels must be declared
  (`knowledge_bases/ontology.py`, derived from `registry.py`). There is no
  second list, so declaring ontology in the registry is the only way to widen
  what a memory may write.
- *Memories drive entity creation.* A lesson about something the graph does not
  know about yet brings it into the graph rather than being half-stored. The
  line is **node vs label**: creating a node is data, creating a label is a
  schema change, so `upsert_node` refuses any label the registry has not
  declared.

  Creation is guarded, because the failure mode that would make it worse than
  useless is silently minting a second `Project` differing only in
  capitalisation. Resolution runs three passes before creating -- id, exact
  name, case-insensitive name -- and a link with no `target_name` is refused
  outright, since an id with no name is an unmergeable orphan. Created entities
  are tagged `provisional: true`, `created_by: "session_memory"`, and given a
  `SourceNode` via `FOUND_IN` with role `asserted_by_memory`, so "where did this
  come from?" is answerable and a reader can weigh an asserted entity
  differently from a curated one.

  Resolution runs **id -> name -> embedding**
  (`knowledge_bases/orchestration/resolution.py`). The embedding pass is what
  catches paraphrase: "quarterly forecast of revenue" resolves to an existing
  "Quarterly Revenue Forecast" that no string comparison would match. It uses
  the same 0.94 auto-merge / 0.85 LLM-arbiter thresholds `extract.py` already
  applies to document ingestion -- two thresholds for one question in two
  places would be two things to keep in sync.

  A newly created entity is embedded into its collection immediately, so the
  *next* memory's paraphrase resolves to it. Without that, memory-created
  entities would be invisible to embedding search and the second paraphrase
  would duplicate the first -- the exact failure this layer prevents.

  Milvus is an accelerator, never a dependency. Every step degrades to the one
  above it: with Milvus down, resolution is exactly the deterministic name
  matching, which is correct but blind to paraphrase. `resolve_by_embedding`
  never raises, so callers need no fallback of their own. `resolved_by` is
  recorded on each link, so a merge made by an embedding (which can be wrong)
  stays distinguishable from one made by an exact id (which cannot).

  Only labels with a vector collection can be resolved semantically
  (`SEMANTIC_COLLECTIONS`: Insight, Decision, Task, Project, Epic,
  BusinessTerm). `CodeRepository` has none, so it stops at name matching.

A triplet failing either rule is not dropped. It is returned in `rejected` and
stored as `unmapped_claims` on the node, because an extractor repeatedly
reaching for a relationship the ontology lacks is precisely the signal that
should reach a human -- the same reasoning as `catalog_noesis_feedback`.

## Memory never expires

So correction has to be append-only. `supersede` writes a *new* node,
`(:SessionMemory)-[:SUPERSEDES]->(:SessionMemory)`, and tags the old one
`:Deprecated` -- the lifecycle marker retrieval already filters
(`docs/architecture/lifecycle.md`). The old memory stays readable as history.
Nothing in the repository deletes.

The identity key is derived from the title and scope qualifiers, not the body,
so re-recording a reworded lesson updates one node instead of accumulating
near-duplicates that each look authoritative.

## Scope

`session | repo | expert | global`, unioned at read time with the most specific
first: a session in a repo working through an expert should see that expert's
memories, that repo's, and global ones together.

## Boundary

- `session_memory` is **mutable at runtime**: prose plus edges to existing
  entities adds no label and no relationship type, so it is data.
- `memory_ontology_mapping` is **immutable**: a runtime path that could widen
  the allowlist could mint schema from extracted prose.

## Tools

    record_noesis_session_memory     store one lesson; `supersedes_key` corrects
    search_noesis_session_memories   scoped retrieval, superseded hidden
    get_noesis_session_memory        one memory with links and history
