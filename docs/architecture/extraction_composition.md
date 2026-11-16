# Extraction is composed from the product profile

Where the extraction vocabulary lives, and why it moved.

## The problem

Two constants in the engine described vocabularies the engine does not own:

- `extract.py` kept ten Pydantic classes -- `ExPerson`, `ExProject`, `ExEpic`,
  `ExTask`, `ExDecision`, `ExInsight`, `ExBusinessTerm`, `ExTable`, `ExColumn`,
  `ExValue` -- and built the extraction schema from them.
- `worker.py` kept `self.topics = ['raw-emails', 'teams-messages',
  'evidence-documents']` and subscribed to those.

Both failed in the same direction, and quietly:

- A product that installed no `project_management` was still offered `Epic`, and
  the model filled it. The write was then skipped at the ontology boundary, so
  extraction ran, cost a model call, and produced nothing.
- A product that installed `semantic_layer` had no shape for `Metric`, because
  adding a shape meant adding a class to engine code. The vocabulary of a package
  could only grow by editing the engine.
- A product with no `claims` package subscribed to three Kafka streams it had no
  parser for.

## The seam

`OntologyPackage` (`knowledge_bases/base.py`) already declares everything by lazy
`"module.path:ATTR"` seeds -- `tools_seed`, `experts_seed`, `kinds_seed`,
`entity_types_seed`, `relationship_types_seed`, `capabilities_seed`,
`edge_contracts`. Extraction is now declared the same way, by the package that
owns the labels:

- `extraction_shapes_seed` -- the entity shapes this package's labels may be
  extracted into. Each `ExtractionShape` names its field, label, model, a `props`
  callable, and either `fuzzy_text` (merge by embedding similarity) or `identity`
  (merge by a key the package declares).
- `extraction_sources_seed` -- the ingestion streams and pull sources this
  package can parse. Each `ExtractionSource` names a topic (or nothing, for a
  pull source), a source type, and a `parser` (raw -> `NormalizedDocument`, the
  LLM path) or a `triplets` emitter (raw -> `list[Triplet]`, the deterministic
  path).

`registry.extraction_shapes()` and `registry.extraction_sources()` compose them.
`extract.py` builds its schema with `create_model` over the composed shapes;
`worker.py` subscribes to the composed topics.

Ownership follows the label: `project_management` declares Person, Project, Epic,
Task; `tasks` declares Person, Project, Task (no Epic); `strategy` declares
Decision; `claims` declares Insight and the three Microsoft Graph topics;
`business_dictionary` declares BusinessTerm; `semantic_layer` declares Table,
Column, Value and the Power BI source; `development` declares the repository
source.

## What composition refuses

Four refusals, all at composition, next to the composition that decided the label
set:

1. A shape for a label no installed package declares. The model would fill it and
   the write would be skipped -- extraction that produces nothing.
2. A shape with neither `identity` nor `fuzzy_text`: a half-declared contract,
   whose graph id is never generated.
3. Two shapes claiming one field, or one label. `compose()` already refuses two
   owners for one label, so `tasks` and `project_management` are mutually
   exclusive, and their shapes inherit that.
4. A source with neither a parser nor a triplet emitter: a stream the product
   subscribes to and cannot read. And two sources claiming one topic: the same
   message normalized two ways.

A worker with no sources at all refuses to be built. A worker with pull sources
and no streams -- a code-and-schema product -- is constructible so
`run_pull_sources()` can be called, and `run()` refuses with a clear message
rather than looping on an empty subscription.

## Two write paths, one ontology check

- **Prose** goes `parser` -> `NormalizedDocument` -> `extract.py` -> `triplets.py`
  -> `ingestion_service.ingest_triplets`. The package normalizes; the shared path
  never learns where a document came from.
- **Deterministic formats** go `triplets` -> `ingestion_service.ingest_triplets`
  directly. A Power BI TMDL table states its columns exactly; asking a model to
  guess a schema it is handed is a way to invent ontology.

Both end at the same `partition()` check. The check is the point, not the model.

`code_authored` is the one relaxation: a code scan IS repository code, so the
`extractable` filter (which means "may be written from extracted information") is
lifted for it via `extractable_only=False`. The label, relationship, and endpoint
typing checks still run -- relaxing who may write is not relaxing what may be
written.

## Identity is the package's knowledge

The merge rules are the domain content, and they are not uniform:

- A Person merges on email, then `teams_id`, then slugified name. The order is the
  point: a meeting note saying "Alice" and an email from `alice@corp` are one
  person, so the contact rule runs first.
- Project, Epic, Task, Insight, Decision, BusinessTerm merge by embedding
  similarity, because "Project Phoenix" and "the Phoenix initiative" are one
  thing and no string rule finds that.
- Table, Column, Value merge on composite keys (`table::model.name`,
  `column::model.name.column`, `value::column=value`), because a semantic
  catalog's whole value is lineage, and an embedding search that merged `orders`
  with `order_lines` would corrupt it.

## Tests

- `noesis-ontologies/tests/test_extraction_profile.py` -- 19 tests: the enterprise
  profile composes exactly the ten labels the constant produced; the personal
  profile composes five and gets no `Epic`, `Table`, `Column`, `Value`,
  `BusinessTerm`; the four refusals; the identity rules; every declared seed
  resolves; the Graph parser normalizes a payload; a Power BI table becomes
  triplets the ontology accepts; a repository scan becomes 4,625 triplets; a pull
  source reads `DEV_CATALOG_REPOS`.
- `tests/test_extract_vocabulary.py` -- the guard that the relationship vocabulary
  the extractor is offered is unchanged by the refactor.
- `tests/test_worker_idempotency.py` -- the worker subscribes to the topics its
  packages declare, skips completed events, skips messages from uninstalled
  sources, and refuses to start with no sources.

## Known limits

- `Metric` has no extraction shape. A metric is registered through
  `define_metric` with a formula, which is a governed write; a model guessing a
  formula is not.
- `scripts/dev_catalog.py` keeps the full phase DAG, the analytics phases
  (communities, processes), and `persist`. The `development` source is the
  triplet path, which does not run analytics. Repo discovery is shared
  (`development.extraction.discover_repos`), so the two agree on
  `DEV_CATALOG_REPOS`.
- Snowflake is queried on demand through the MCP tool catalog; it is not an
  ingestion source, and declaring it as one would make the product poll a
  warehouse for facts it asks for.

## Properties are contributed by the sources that supply them

A connector knows the identifiers its payloads carry. The Microsoft Graph
connector knows an email and a Teams id; a Slack connector knows a Slack
username. `Person` belongs to the delivery ontology, so a connector cannot
describe it -- `registry.entity_types()` is single-owner, and that exclusivity is
what keeps one noun from having two competing definitions. It contributes.

`OntologyPackage.property_contributions_seed` carries `PropertyContribution`
tuples: `label`, `prop`, `prefix` (the graph-id prefix the key produces),
`priority` (its place in the merge ladder), `lowercase`, and a `description`.
`registry.property_contributions()` composes them and merges them into the label
owner's declaration, so one declaration makes a property:

- offered by `describe_ontology`,
- kept by `filter_properties`,
- stored as a queryable property,
- a rung of the merge ladder.

Installing the Graph connector makes `teams_id` all four. Uninstalling it removes
all four, and a product with no Graph connector never merges on a Teams id it
cannot receive.

Precedence is stated, not left to chance: when the label owner already declares
the property, the owner's description wins and the contribution supplies only the
prefix. `tasks` declares `email`; the Graph connector contributes `email` with the
`person_email` prefix, so the meaning stays with the ontology and the merge key
comes from the connector. `project_management` does not declare `email`, so the
connector's descriptor is added.

Refused at composition: a contribution to a label no installed package declares
(the property describes a node the product cannot write), two packages
contributing one property to one label, and two identity keys sharing one
graph-id prefix (two sources writing one merge key would merge unrelated
entities).

Casing is declared per key. An email is one address whatever its casing; a Teams
id is a server-assigned string whose casing a connector must not invent. So
`lowercase` is a field, not an assumption.

A Person with neither a declared contact key nor a name is refused. A merge key of
`person_name::` from an empty name would collapse every unidentifiable person
into one node.

**The bug this fixes.** `filter_properties` drops undeclared properties, so the
enterprise product extracted `email` from Microsoft Graph payloads and threw it
away: `project_management` declares `id`, `name`, `title`, `role`, and nothing
declared `email` or `teams_id`. Both were dropped on every write.

**The unsolved case, named.** The ladder converges only when one key is shared.
If Graph supplies `teams_id` and Slack supplies `slack_username` for the same
human, and neither payload carries email, you get two Person nodes. No ladder
fixes that. The precedent in this codebase is `BusinessTerm`'s `ALIAS_OF` edge,
resolved at query time (`query.py:88`: `coalesce(canonical, term)`). The same
shape works for Person: a canonical node, alias edges from connector-specific
nodes, resolved on read. That is not implemented.
