# Noesis's model of itself

Noesis can answer questions about its own code. It does so through the
`development` knowledge base, because its own source is just another indexed
repository.

## What this replaced

There used to be a `self_catalog` knowledge base owning `SelfModule`,
`SelfComponent` and `SelfInterface`, populated by `scripts/self_catalog.py`'s
static AST introspection, read by five tools, and owned by its own expert at
routing priority 70.

It was retired in migration 23, for two reasons.

**It was a duplicate.** `development` already indexed Noesis's own source, as
`CodeRepository {id: "noesis"}`, with a strict superset of the information:
2660 definitions against 1343 `SelfInterface` nodes, plus 3180 `CALLS` edges
with confidence tiers, communities, processes and a `:Deprecated` lifecycle
that `self_catalog` had none of.

**It had rotted.** Measured at retirement:

| | `self_catalog` | `development` (repo `noesis`) |
| :--- | ---: | ---: |
| nodes pointing at deleted files | 99 of 282 | 0 |
| live files missing from the index | 76 of 259 | 0 |
| last sync | 2026-09-11 | 2026-09-15 |

Six of its thirteen `SelfComponent` ids (`agents`, `bmad`,
`decision_intelligence`, `delivery`, `evidence`, `snowflake`) were keys of a
registry that no longer existed, while six knowledge bases that *did* exist had
no component at all. Its `self_catalog_text` fulltext index was referenced by
no `SEARCH_TYPES` entry, so it was maintained on every sync for no reader.

The expert's one promise was to answer from the committed source and never from
recollection. It was answering from a snapshot of a codebase that no longer
existed, which is worse than the gap it was built to close: a confidently wrong
answer about a file that does not look like that any more costs more than "no
expert owns this question".

## The shape now

**A marker, not an ontology.** `CodeRepository {id: "noesis"}` carries
`is_self: true`. There are no self-specific labels.

**The id stays `noesis`.** Renaming it to `self` looks tidier and is a trap: the
repo id is a literal prefix of every child id (`noesis:agent.py:<module>`), so a
rename rewrites thousands of ids, invalidates the `devcatalog://noesis@<sha>`
provenance URI, and -- worst -- `dev_catalog.py` discovers repos by *directory
name*, so the next sync would recreate `noesis` beside the renamed node and
split the graph in two silently. Instead `repo="self"` is an alias resolved in
`development/mcp_tools.py:_repo`, which makes it work across all twelve
repo-scoped tools at once.

**Components are derived, not stored.** A component is a
`knowledge_bases/<key>/` package; the registry already says which keys exist and
the graph already has `CodeFile -[:IMPORTS]-> CodeFile`. So
`development/self/components.py` computes the ownership view at read time. It
cannot go stale independently of the code, which is the failure mode that
retired the old nodes. Loose modules directly under `knowledge_bases/`
(`registry.py`, `ontology.py`, ...) group into `core`; grouping on
`split(path, '/')[1]` unguarded invents junk components named after files.

Note `CodeCommunity` is a different thing and both are useful: communities are
Louvain clusters of the call graph (emergent), components are ownership
boundaries (declared).

## The two tiers, and why they use different ontologies

Structure and reasoning stay in separate paradigms on purpose:

- **Structure** -- `development`. Regenerated wholesale from source, plain
  `MERGE` upserts, no revisioning. If it disagrees with the source, the source
  wins and the fix is a re-sync.
- **Reasoning** -- a normal methodology project (`project_id = "noesis-self"`)
  built from `strategy`/`architecture`/`project_management` labels with
  `methodology` revisioning. Human-authored, reviewed, versioned. If it
  disagrees with the code, that disagreement is *itself* the finding.

Those are opposite epistemics, so they are not one ontology. Exactly one edge
joins them:

    (:BmadEntity:Architecture)-[:CATALOGS]->(:CodeRepository {id: "noesis"})

That edge is the only self-specific piece of ontology in the system. It is
written with the plain `Graph.upsert_rel` helper rather than the methodology
changeset API, because `CodeRepository` is not a `BmadEntity` -- the same
pattern `graph.attach_source`/`FOUND_IN` already uses across knowledge bases.

`noesis_architectural_rationale` reads across it and returns an explicit
`rationale_recorded: false` when nothing is bridged, because "nothing is
recorded" and "the bridge is broken" are different answers and a bare `[]`
cannot tell them apart.

## Code layout

`development/self/` is a code boundary inside a shared ontology -- separate
files, same labels, one registry key, one expert:

    knowledge_bases/development/
    ├── models.py, queries.py, mcp_tools.py   # repo-agnostic
    └── self/
        ├── __init__.py       # SELF_REPO_ID, the "self" alias
        ├── components.py     # derived ownership view
        ├── queries.py        # self-specific reads + rationale bridge
        └── mcp_tools.py      # handlers, registered under `development`

**Always alias imports from it.** `from knowledge_bases.development import self`
binds the name `self` and shadows the instance parameter inside any method --
silently, no error. Write `from knowledge_bases.development.self import queries
as self_queries`. Pinned by `tests/test_development_self.py`.

## Routing

Retiring an expert perturbs routing for *every* question, so the golden eval
(`routing_eval.jsonl`) is the gate -- the same one `create_expert` enforces at
runtime.

Merging naively **fails it**: accuracy drops to 0.94 because "which module
implements expert routing in Noesis?" and "why does Noesis separate Tool from
AgentSkill?" fall through to `claims`. The `development` expert therefore
carries the absorbed capability wording, including "what breaks if I change
this" -- without that phrase the registry question scored 0.0 against every
expert and routing refused to answer at all, which is precisely the original
gap.

With the merged wording accuracy is back to **1.0, now over 53 cases**: the four
self-referential cases moved to `development`, and two generic code-graph cases
were added because `development` previously had *no* golden coverage of its own.

Priority stays 65, below every domain expert: questions are about the business
far more often than about Noesis, so ambiguous wording should favour a domain
expert. Verified on held-out questions that `architecture` still wins "which
components realize the checkout capability?" -- design intent was never in
scope for the code graph.

## Drift detection

`scripts/self_recompute.py` compares what the self repo's components actually
depend on against what any recorded rationale accounts for, and files advisory
`ReviewFinding`s. It never supersedes a `Decision` automatically.

Deriving components from the live import graph made this check see every real
edge rather than the seven stale stored ones, so it is capped at five findings
per run, heaviest dependency first. A queue of fifteen findings is a queue
nobody reads.

Migration 23 also deprecated (never deleted) the four findings this check had
filed against knowledge bases that no longer existed.

## What Noesis still may not do

Retiring a knowledge base touches `registry.py`, `migrations.py` and the
knowledge-base package set -- all IMMUTABLE surfaces under
`knowledge_bases/boundary.py`. This was a human-reviewed commit and a
migration, by design. Noesis cannot do this to itself, and the `development`
expert is instructed to report what exists and what would be affected, never to
propose edits as if it could apply them.
