# MADE_BY edges are written backwards by LLM extraction

**Found while documenting the ontology. Not yet fixed — needs a decision.**

## The bug

`query.py:282` reads:

```cypher
MATCH (d:Decision {id: $id})-[:MADE_BY]->(p:Person)
```

`ingest.py:69` writes that same direction. But every `MADE_BY` edge in the live
graph points the other way:

| Direction | Count |
|---|---|
| `(Decision)-[:MADE_BY]->(Person)` — what readers expect | **0** |
| `(Person)-[:MADE_BY]->(Decision)` — what exists | **20** |

So the decision-owner lookup in `query.py` silently returns nothing. It does not
error; a decision just looks unowned.

## Why it happened

`extract.py` asks an LLM for `relationship_type` and passes the result to
`graph.upsert_rel(subject, predicate, object)`. The ontology guard
(`knowledge_bases/ontology.py` `require_known_relationship`) validates the
relationship *type* and both *labels* — but **not the direction**. `MADE_BY`
with `Person` as subject and `Decision` as object is three valid parts in an
invalid arrangement, so it passes.

Read semantically, the model was not unreasonable: "person made decision" reads
naturally as `Person -MADE_BY-> Decision`. The declared convention is the
opposite (`Decision -MADE_BY-> Person`, i.e. "decision was made by person").
The guard has no way to express that.

## Scope

Only `MADE_BY` is currently affected. Every other declared relationship is
either absent or correctly oriented (checked all 12):

- Correct: `SUPPORTS` (7), `COMPUTED_FROM` (22,781), `FOUND_IN` (165),
  `COLUMN_OF` (45,447).
- Absent, so unverifiable from data: `HELD_BY`, `INTENDED_FOR`, `MEASURED_BY`,
  `CHOSEN_OVER`, `CHALLENGES`, `VALUE_OF`, `USES`.

The absent ones are the risk: nothing stops the same reversal happening to them
on the next extraction run.

## Options

1. **Declare direction in the ontology and enforce it at `upsert_rel`.** Turns
   the class of bug into a rejected triplet with a reason, consistent with how
   off-ontology relationship *types* are already handled (skip and report). Also
   makes the remaining 7 relationships safe before they are ever written.
2. **Normalise on write** — flip a reversed edge to the declared direction
   instead of rejecting it. Fixes more silently; risks masking a genuine
   modelling disagreement.
3. **Migrate the existing 20 edges** either way. Needed regardless of 1 or 2,
   since fixing the guard does not fix data already written.

Option 1 plus 3 is the smaller, more honest change. Note that adding a direction
to the ontology is an **architecture surface** (`graph_ontology` /
`memory_ontology_mapping` are immutable per `knowledge_bases/boundary.py`), so it
requires a human-reviewed commit — this is not something the self-improvement
loop may do.
