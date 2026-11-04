# Migration streams: what moved, and what a deployment must know

## Why

Every migration used to live in one stream, `"core"`. The framework for per-package
streams existed and held no data. That blocks the split in two ways:

- A product that installs 6 of 9 ontologies runs migrations for labels it does not
  have, creating constraints and indexes for entities nothing can write.
- An ontology package distributed separately **cannot ship a new migration at all**
  if the only stream belongs to the engine.

## What moved

Bookkeeping keys on `(stream, version)`. Four migrations moved:

| core version | new stream | new version | name |
|---|---|---|---|
| 3 | `methodology` | 1 | `bmad_baseline_relationships` |
| 5 | `claims` | 1 | `ingestion_events` |
| 7 | `claims` | 2 | `user_feedback_claims` |
| 22 | `development` | 1 | `development_kb` |

Only four of 23, because the eligibility rule is strict. A migration may leave
`"core"` only if **all** of these hold:

1. every label it touches is owned by exactly **one** ontology package;
2. it touches no control-plane label (`tools`, `orchestration`, `provenance`), which
   ship with the engine;
3. it names no retired label (`SelfComponent`, `ExternalInfo`, `Skill`, `Counter`) --
   nothing owns those, so nothing can carry their migration;
4. every statement is **idempotent DDL**.

## The re-run problem, and why (4) is the whole answer

A moved migration has a `(stream, version)` key **no existing deployment has
recorded**, so it *will* run again on the next boot. No backfill fixes this in
general: nothing can distinguish "already applied under the old key" from "never
applied".

So the safety property is not about the move, it is about the **content**. Every
moved migration consists only of `CREATE CONSTRAINT/INDEX ... IF NOT EXISTS` and
`DROP ... IF EXISTS`, so the second application is a no-op.

This is why migrations containing data statements stay in `"core"` **even when they
name domain labels** -- `rename_skill_label_to_tool` (v18),
`rekey_skills_knowledge_base_to_tools` (v19),
`drop_superseded_expert_skill_properties` (v20), and
`retire_self_catalog_into_development` (v23). Re-running a rename or a
`DETACH DELETE` is not a no-op. `tests/test_migrations.py::
test_a_migration_that_changed_stream_is_safe_to_apply_twice` enforces this, so a
future data migration cannot be moved by accident.

`lexical_search` (v6) stays for a different reason: one fulltext index spans six
packages' labels at once. It cannot belong to a single owner, and splitting a
fulltext index per package would change search behaviour, not just packaging. That is
a real open item, recorded below.

## What a deployment sees

- **Fresh graph:** nothing to know. 23 migrations across four streams.
- **Existing graph:** four migrations re-run as no-ops and four new `SchemaMigration`
  rows appear (`claims` v1/v2, `methodology` v1, `development` v1). The old
  `("core", 3/5/7/22)` rows stay, as the record of what that deployment did.
- **No action required.** No downtime, no manual backfill.

Core version numbers 3, 5, 7 and 22 are now permanent gaps. Reusing one would make a
*different* migration look already-applied on any older deployment, and skipping is
the silent path.

## A related bug this surfaced

`composed_migrations()` called `ontology_packages()` with no arguments, which
**re-loads every declared package** rather than returning the installed set. Three
other composition sites did the same: the tool catalog, the expert catalog, and
versioned kinds. So a narrow product profile was ignored -- it got the full set of
migrations, tools, and experts regardless of what it installed.

Found only because a test asserting "uninstalling a package removes its migrations"
could not be made to pass. `registry.installed_packages()` is now the single accessor,
and `test_no_composition_site_reloads_every_declared_package` prevents a regression.

## Open

- **v6 `lexical_search`** spans six packages' labels in one fulltext index. Needs a
  decision on per-package fulltext indexes before it can be split.
- **v1/v2/v9** span `claims` + `methodology`. Splittable statement-by-statement, but
  each half would be a NEW migration re-running against live data, so this is
  deliberately deferred until the packages are physically separated.
