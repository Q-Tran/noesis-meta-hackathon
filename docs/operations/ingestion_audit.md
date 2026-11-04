# Ingestion audit — orchestrator write path (pre-Phase 2)

Verified on the current tree (831 tests green, routing eval 1.0).

## What works
- 15 ingestion-capable tools exist; 17 bounded write commands in the allowlist.
- `methodology.submit_changeset` is a genuine generic ingestion path: 20
  versionable kinds, 24 changeset relationships, propose/confirm/execute gated.
- `claims.catalog_user_feedback` creates `Claim` + `SourceNode` with user
  provenance (a deliberate sourced-data path, not a bypass).
- The propose/confirm/execute flow itself is sound and boundary-enforced.

## Gap 1 — routing has no write-intent concept (root cause)
`plan_route` models `metric_intent` only (`routing.py:265`). Ingestion-phrased
questions therefore always start at a READ operation:
  0 of 9 ingestion-phrased questions produced a write starting operation,
  despite 17 write operations being available.
  "Create a project for the mobile app rewrite" -> `list_project_blockers`.
Expert capability wording is phrased as questions, so ingestion verbs score
nothing: 7 of 9 experts have zero ingestion verbs in their capabilities.
3 of 12 ingestion phrasings were refused outright (no expert matched).

## Gap 2 — no create path for business terms or metrics
`business_dictionary` has exactly one write: `add_business_term_alias`, which
requires an EXISTING canonical term. There is no `create_business_term`.
`semantic_layer` has 0 write commands and 0 versionable kinds: a user-supplied
metric definition cannot be persisted as a metric at all.
`strategy.evaluate_decision_health` is a recompute, not an ingestion path.

## Gap 3 — changeset-only experts get no methodology handoff
`project_management` (12 kinds) and `architecture` (3 kinds) own no direct
write, so they can only persist through `methodology.submit_changeset`. Routing
does not add that delegation, so the plan stops at a read tool.

## Gap 4 — consumer profile loses ingestion disproportionately
A narrowed consumer profile (drop project_management + architecture) keeps only
5 of 20 ingestable kinds and loses the sole business_dictionary write.
Phase 6's `noesis-individual` would ship with a very thin write surface.

## Suggested sequencing
These are ontology/capability-surface gaps, not engine gaps, and the fix is
mostly graph-data (capability wording) plus a few bounded commands. Both are
cheaper after the Phase 2/3 split, EXCEPT the routing write-intent change,
which is engine code and belongs in `noesis-core` before extraction.
