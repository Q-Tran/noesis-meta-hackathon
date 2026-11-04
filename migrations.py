"""Ordered Neo4j schema migrations for Noesis."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class QueryRunner(Protocol):
    def run(self, query: str, **params: object) -> list[dict]: ...


@dataclass(frozen=True)
class Migration:
    """One ordered schema migration inside a `stream`.

    Every ontology package owns its own migration stream (`stream` names the
    owning package; `"core"` is the engine's stream, which also carries the
    entire pre-split history). Streams version independently: version 4 of
    `semantic_layer` and version 4 of `development` are unrelated. A migration
    must mean the same thing forever -- never reorder or renumber a shipped
    stream; append.
    """

    version: int
    name: str
    statements: tuple[str, ...]
    stream: str = "core"


MIGRATIONS = (
    Migration(
        version=1,
        name="bmad_runtime",
        statements=(
            "CREATE CONSTRAINT bmad_entity_identity IF NOT EXISTS "
            "FOR (n:BmadEntity) REQUIRE (n.project_id, n.key) IS UNIQUE",
            "CREATE CONSTRAINT revision_id IF NOT EXISTS "
            "FOR (n:Revision) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT changeset_id IF NOT EXISTS "
            "FOR (n:ChangeSet) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT workflow_run_id IF NOT EXISTS "
            "FOR (n:WorkflowRun) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT baseline_id IF NOT EXISTS "
            "FOR (n:Baseline) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT counter_namespace IF NOT EXISTS "
            "FOR (n:Counter) REQUIRE (n.project_id, n.namespace) IS UNIQUE",
            "CREATE INDEX revision_entity IF NOT EXISTS "
            "FOR (n:Revision) ON (n.entity_id)",
            "CREATE INDEX workflow_project IF NOT EXISTS "
            "FOR (n:WorkflowRun) ON (n.project_id)",
            "CREATE INDEX claim_project IF NOT EXISTS "
            "FOR (n:Claim) ON (n.project_id)",
        ),
    ),
    Migration(
        version=2,
        name="bmad_claims_and_baselines",
        statements=(
            "CREATE CONSTRAINT claim_id IF NOT EXISTS "
            "FOR (n:Claim) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX baseline_project_phase IF NOT EXISTS "
            "FOR (n:Baseline) ON (n.project_id, n.phase)",
        ),
    ),
    # v3 ("bmad_baseline_relationships") moved to the 'methodology' stream
    # (`knowledge_bases/methodology/migrations.py`). The version number is left as a gap
    # rather than reused: a deployment that already applied core v3 has a
    # SchemaMigration row keyed ("core", 3), and renumbering would make a different
    # migration look already-applied.
    Migration(
        version=4,
        name="graph_expert_orchestration",
        statements=(
            "CREATE CONSTRAINT knowledge_base_expert_key IF NOT EXISTS "
            "FOR (n:KnowledgeBaseExpert) REQUIRE n.key IS UNIQUE",
            "CREATE CONSTRAINT expert_capability_id IF NOT EXISTS "
            "FOR (n:ExpertCapability) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT routing_trace_id IF NOT EXISTS "
            "FOR (n:RoutingTrace) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX routing_trace_status IF NOT EXISTS "
            "FOR (n:RoutingTrace) ON (n.status, n.started_at)",
        ),
    ),
    # v5 ("ingestion_events") moved to the 'epistemology' stream
    # (`knowledge_bases/epistemology/migrations.py`). The version number is left as a gap
    # rather than reused: a deployment that already applied core v5 has a
    # SchemaMigration row keyed ("core", 5), and renumbering would make a different
    # migration look already-applied.
    Migration(
        version=6,
        name="lexical_search",
        statements=(
            "DROP INDEX insight_statement IF EXISTS",
            "CREATE FULLTEXT INDEX bmad_entity_text IF NOT EXISTS "
            "FOR (n:BmadEntity) ON EACH [n.key, n.search_text]",
            "CREATE FULLTEXT INDEX business_dictionary_text IF NOT EXISTS "
            "FOR (n:BusinessTerm|AlternateBusinessTerm) "
            "ON EACH [n.name, n.definition, n.context]",
            "CREATE FULLTEXT INDEX semantic_layer_text IF NOT EXISTS "
            "FOR (n:Table|Column|Value|Metric) "
            "ON EACH [n.name, n.description, n.definition]",
            "CREATE FULLTEXT INDEX decision_text IF NOT EXISTS "
            "FOR (n:Decision|Goal|Alternative) "
            "ON EACH [n.name, n.description, n.target]",
            "CREATE FULLTEXT INDEX delivery_text IF NOT EXISTS "
            "FOR (n:Person|Project|Epic|UserStory|Task) "
            "ON EACH [n.name, n.title, n.description]",
            "CREATE FULLTEXT INDEX evidence_metadata_text IF NOT EXISTS "
            "FOR (n:SourceNode|DocumentNode|ExternalInfo) "
            "ON EACH [n.name, n.title, n.description, n.source]",
        ),
    ),
    # v7 ("user_feedback_claims") moved to the 'epistemology' stream
    # (`knowledge_bases/epistemology/migrations.py`). The version number is left as a gap
    # rather than reused: a deployment that already applied core v7 has a
    # SchemaMigration row keyed ("core", 7), and renumbering would make a different
    # migration look already-applied.
    Migration(
        version=8,
        name="knowledge_write_proposals",
        statements=(
            "CREATE CONSTRAINT write_proposal_id IF NOT EXISTS "
            "FOR (n:WriteProposal) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX write_proposal_status IF NOT EXISTS "
            "FOR (n:WriteProposal) ON (n.status, n.expires_at)",
        ),
    ),
    Migration(
        version=9,
        name="repair_lexical_search_indexes",
        statements=(
            "CREATE FULLTEXT INDEX bmad_entity_text IF NOT EXISTS "
            "FOR (n:BmadEntity) ON EACH [n.key, n.search_text]",
            "CREATE FULLTEXT INDEX evidence_metadata_text IF NOT EXISTS "
            "FOR (n:SourceNode|DocumentNode|ExternalInfo) "
            "ON EACH [n.name, n.title, n.description, n.source]",
        ),
    ),
    Migration(
        version=10,
        name="skill_tool_definitions",
        statements=(
            "CREATE CONSTRAINT skill_key IF NOT EXISTS "
            "FOR (n:Tool) REQUIRE n.key IS UNIQUE",
            "CREATE CONSTRAINT tool_key IF NOT EXISTS "
            "FOR (n:Tool) REQUIRE n.key IS UNIQUE",
        ),
    ),
    Migration(
        version=11,
        name="provenance_relationships",
        statements=(
            "CREATE CONSTRAINT source_node_id IF NOT EXISTS "
            "FOR (n:SourceNode) REQUIRE n.id IS UNIQUE",
        ),
    ),
    Migration(
        version=12,
        name="drop_external_info_label",
        statements=(
            # `ExternalInfo` was folded into `SourceNode` (see
            # docs/architecture/provenance.md); rebuild the fulltext index
            # without it and add `url`, now that SourceNode carries it.
            "DROP INDEX evidence_metadata_text IF EXISTS",
            "CREATE FULLTEXT INDEX evidence_metadata_text IF NOT EXISTS "
            "FOR (n:SourceNode|DocumentNode) "
            "ON EACH [n.name, n.title, n.description, n.url]",
        ),
    ),
    Migration(
        version=13,
        name="self_catalog",
        statements=(
            "CREATE CONSTRAINT self_module_id IF NOT EXISTS "
            "FOR (n:SelfModule) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT self_component_id IF NOT EXISTS "
            "FOR (n:SelfComponent) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT self_interface_id IF NOT EXISTS "
            "FOR (n:SelfInterface) REQUIRE n.id IS UNIQUE",
            "CREATE FULLTEXT INDEX self_catalog_text IF NOT EXISTS "
            "FOR (n:SelfModule|SelfComponent|SelfInterface) "
            "ON EACH [n.id, n.name, n.docstring, n.description]",
        ),
    ),
    Migration(
        version=14,
        name="graph_execution_telemetry",
        statements=(
            "CREATE CONSTRAINT span_id IF NOT EXISTS "
            "FOR (n:Span) REQUIRE n.span_id IS UNIQUE",
            "CREATE INDEX span_trace_id IF NOT EXISTS FOR (n:Span) ON (n.trace_id)",
            "CREATE INDEX span_skill_started IF NOT EXISTS "
            "FOR (n:Span) ON (n.skill, n.started_at)",
            "CREATE CONSTRAINT trace_outcome_id IF NOT EXISTS "
            "FOR (n:TraceOutcome) REQUIRE n.outcome_id IS UNIQUE",
            "CREATE INDEX trace_outcome_trace_id IF NOT EXISTS "
            "FOR (n:TraceOutcome) ON (n.trace_id)",
        ),
    ),
    Migration(
        version=15,
        name="graph_defined_experts",
        statements=(
            "CREATE CONSTRAINT expert_revision_id IF NOT EXISTS "
            "FOR (n:ExpertRevision) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX expert_revision_key IF NOT EXISTS "
            "FOR (n:ExpertRevision) ON (n.key, n.version)",
        ),
    ),
    Migration(
        version=16,
        name="durable_sessions",
        statements=(
            # A conversation becomes a first-class node so it can be resumed:
            # `session_id` was previously only a Span property, unindexed and
            # never queried, so closing the terminal lost the thread.
            "CREATE CONSTRAINT session_id IF NOT EXISTS "
            "FOR (n:Session) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX session_user_started IF NOT EXISTS "
            "FOR (n:Session) ON (n.user_id, n.started_at)",
            # Spans already carry session_id; this makes "everything in this
            # session" a real lookup rather than a scan.
            "CREATE INDEX span_session_id IF NOT EXISTS FOR (n:Span) ON (n.session_id)",
        ),
    ),
    Migration(
        version=17,
        name="delegation_results_and_agent_messages",
        statements=(
            # One expert's structured findings for one delegation, so the
            # orchestrator synthesizes from data rather than re-reading prose.
            "CREATE CONSTRAINT delegation_result_id IF NOT EXISTS "
            "FOR (n:DelegationResult) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX delegation_result_trace IF NOT EXISTS "
            "FOR (n:DelegationResult) ON (n.trace_id)",
            # Graph-mediated agent-to-agent messages (drain-on-read).
            "CREATE CONSTRAINT agent_message_id IF NOT EXISTS "
            "FOR (n:AgentMessage) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX agent_message_recipient IF NOT EXISTS "
            "FOR (n:AgentMessage) ON (n.recipient_expert, n.status)",
            "CREATE INDEX agent_message_session IF NOT EXISTS "
            "FOR (n:AgentMessage) ON (n.session_id)",
        ),
    ),
    Migration(
        version=18,
        name="rename_skill_label_to_tool",
        statements=(
            # The `Skill` label was a misnomer: these nodes are bounded,
            # callable operations (handler + input schema), which is what a
            # harness calls a *tool*. "Skill" means something else entirely in
            # a harness -- a progressive-disclosure instruction bundle -- and
            # that collision is now resolved in favour of the harness meaning
            # (see the `AgentSkill` label added below).
            #
            # Relabel in place so every existing `(:Span)-[:CALLED]->` edge and
            # all recorded telemetry keep pointing at the same nodes.
            "MATCH (n:Skill) SET n:Tool REMOVE n:Skill",
            # Bounded instruction bundles a harness discloses to a model. Owned
            # by `orchestration`, deliberately NOT by `tools`: an AgentSkill
            # carries no handler and is never dispatched.
            "CREATE CONSTRAINT agent_skill_key IF NOT EXISTS "
            "FOR (n:AgentSkill) REQUIRE n.key IS UNIQUE",
            "CREATE INDEX agent_skill_active IF NOT EXISTS "
            "FOR (n:AgentSkill) ON (n.active)",
        ),
    ),
    Migration(
        version=19,
        name="rekey_skills_knowledge_base_to_tools",
        statements=(
            # Relabelling the node was not enough: a tool is addressed by its
            # `"<kb>.<operation>"` key and by `knowledge_base`, both of which
            # still said "skills". Left alone, every already-synced tool in that
            # knowledge base became unreachable -- the registry no longer has a
            # `skills` entry to validate against.
            "MATCH (n:Tool) WHERE n.knowledge_base = 'skills' "
            "SET n.knowledge_base = 'tools'",
            "MATCH (n:Tool) WHERE n.key STARTS WITH 'skills.' "
            "SET n.key = 'tools.' + substring(n.key, 7)",
            # Spans record the tool key they called; keep telemetry joinable.
            "MATCH (s:Span) WHERE s.skill STARTS WITH 'skills.' "
            "SET s.skill = 'tools.' + substring(s.skill, 7)",
            "MATCH (s:Span) WHERE s.knowledge_base = 'skills' "
            "SET s.knowledge_base = 'tools'",
            # An expert's allowlist and entry points are stored keys too, and a
            # stale `skills.*` glob would silently stop matching anything.
            "MATCH (e:KnowledgeBaseExpert) WHERE e.allowed_skills IS NOT NULL "
            "SET e.allowed_skills = [x IN e.allowed_skills | "
            "  CASE WHEN x STARTS WITH 'skills.' "
            "       THEN 'tools.' + substring(x, 7) ELSE x END]",
            "MATCH (e:KnowledgeBaseExpert) WHERE e.entry_skills IS NOT NULL "
            "SET e.entry_skills = [x IN e.entry_skills | "
            "  CASE WHEN x STARTS WITH 'skills.' "
            "       THEN 'tools.' + substring(x, 7) ELSE x END]",
        ),
    ),
    Migration(
        version=20,
        name="drop_superseded_expert_skill_properties",
        statements=(
            # `allowed_skills`/`entry_skills` were renamed to
            # `allowed_tools`/`entry_tools`. A sync writes the new properties
            # but cannot remove the old ones, so both were left on every node
            # holding identical values -- two sources of truth for what an
            # expert may call, of which only one is read. Drop the dead pair
            # now that migration 19 has copied their contents forward.
            "MATCH (e:KnowledgeBaseExpert) "
            "REMOVE e.allowed_skills, e.entry_skills",
            "MATCH (r:ExpertRevision) "
            "REMOVE r.allowed_skills, r.entry_skills",
        ),
    ),
    Migration(
        version=21,
        name="session_memory_constraints",
        statements=(
            # Memory MERGEs on `key`, so a duplicate key would silently split
            # one lesson across two nodes and make supersession ambiguous.
            "CREATE CONSTRAINT session_memory_key IF NOT EXISTS "
            "FOR (m:SessionMemory) REQUIRE m.key IS UNIQUE",
            # Retrieval always filters by scope before anything else.
            "CREATE INDEX session_memory_scope IF NOT EXISTS "
            "FOR (m:SessionMemory) ON (m.scope)",
        ),
    ),
    # v22 ("development_kb") moved to the 'development' stream
    # (`knowledge_bases/development/migrations.py`). The version number is left as a gap
    # rather than reused: a deployment that already applied core v22 has a
    # SchemaMigration row keyed ("core", 22), and renumbering would make a different
    # migration look already-applied.
    Migration(
        version=23,
        name="retire_self_catalog_into_development",
        statements=(
            # The `self_catalog` knowledge base was a strict subset of
            # `development` that had rotted: 99 of its 282 SelfModule nodes
            # pointed at files deleted from disk, 76 live files were missing,
            # and 6 of its 13 SelfComponent ids were keys of a registry that
            # no longer existed. `development` already indexes Noesis's own
            # source as CodeRepository {id: "noesis"} -- same information,
            # current, with real CALLS edges. See
            # knowledge_bases/development/self/__init__.py.
            #
            # 1. Mark the self repo. The id stays "noesis" on purpose: it is a
            #    literal prefix of every child id, and dev_catalog.py
            #    discovers repos by directory name, so a rename would both
            #    rewrite thousands of ids and let the next sync recreate the
            #    old node. `repo="self"` is an alias resolved at query time.
            "MATCH (r:CodeRepository {id: 'noesis'}) SET r.is_self = true",
            # 2. Move the rationale bridge off the retired label. The reasoning
            #    layer itself is untouched -- it is a normal methodology
            #    project (project_id 'noesis-self') in a different, revisioned
            #    ontology, which is where design intent belongs. Only this one
            #    join edge was self-catalog-shaped.
            "MATCH (a:BmadEntity:Architecture)-[old:CATALOGS]->(:SelfComponent) "
            "MATCH (r:CodeRepository {id: 'noesis'}) "
            "MERGE (a)-[:CATALOGS]->(r) "
            "DELETE old",
            # 3. Deprecate the drift findings that self_recompute.py raised
            #    against knowledge bases which no longer exist (agents,
            #    decision_intelligence, delivery, evidence). Tagged, never
            #    deleted, per docs/architecture/lifecycle.md -- they are real
            #    history, they are just no longer actionable.
            # The dead keys are listed literally, not derived from
            # KNOWLEDGE_BASES: a migration must mean the same thing forever,
            # and a registry-derived list would silently re-scope this
            # statement every time a knowledge base is added or renamed.
            "MATCH (f:BmadEntity:ReviewFinding {project_id: 'noesis-self'}) "
            "WHERE f.key STARTS WITH 'drift::core::' "
            "  AND split(f.key, '::')[2] IN "
            "      ['agents', 'decision_intelligence', 'delivery', 'evidence'] "
            "SET f:Deprecated",
            # 4. Drop the retired nodes. Unlike the findings these are not
            #    history: they are a stale mirror of source that the code
            #    graph already holds correctly, so keeping them would leave
            #    two answers to "what does this module import" with no rule
            #    for which wins.
            "MATCH (n:SelfInterface) DETACH DELETE n",
            "MATCH (n:SelfModule) DETACH DELETE n",
            "MATCH (n:SelfComponent) DETACH DELETE n",
            # 5. Drop their constraints and the fulltext index, which no
            #    SEARCH_TYPES entry ever referenced -- it was dead weight
            #    kept current on every sync for no reader.
            "DROP CONSTRAINT self_module_id IF EXISTS",
            "DROP CONSTRAINT self_component_id IF EXISTS",
            "DROP CONSTRAINT self_interface_id IF EXISTS",
            "DROP INDEX self_catalog_text IF EXISTS",
            # 6. Retire the expert and its Tool rows; the `development` expert
            #    absorbs both (its capability wording was widened to keep the
            #    golden routing eval at 1.0).
            "MATCH (t:Tool) WHERE t.knowledge_base = 'self_catalog' DETACH DELETE t",
            "MATCH (e:KnowledgeBaseExpert {key: 'self_catalog'}) "
            "OPTIONAL MATCH (e)-[:CURRENT|SUPERSEDES*0..]->(rev:ExpertRevision) "
            "OPTIONAL MATCH (e)-[:HAS_CAPABILITY]->(c:ExpertCapability) "
            "DETACH DELETE e, rev, c",
            "MATCH (s:AgentSkill {key: 'noesis-self-catalog'}) DETACH DELETE s",
        ),
    ),
    Migration(
        version=24,
        name="retire_agent_messages_for_ephemeral_subagents",
        statements=(
            "DROP CONSTRAINT agent_message_id IF EXISTS",
            "DROP INDEX agent_message_recipient IF EXISTS",
            "DROP INDEX agent_message_session IF EXISTS",
            "MATCH (m:AgentMessage) DETACH DELETE m",
            "MATCH (t:Tool) WHERE t.key IN ['orchestration.send_noesis_agent_message', 'orchestration.read_noesis_agent_messages'] DETACH DELETE t",
        ),
    ),
)


# Stream name stamped on `SchemaMigration` rows written before streams
# existed: the whole pre-split history is the engine's.
LEGACY_STREAM = "core"


def composed_migrations() -> tuple[Migration, ...]:
    """The engine's stream plus every installed package's own stream.

    A package declares `migration_stream` (`knowledge_bases/base.py`) pointing
    at a `"module.path:ATTR"` tuple of `Migration`s in its own stream, so
    installing a package brings its constraints and indexes with it instead of
    requiring an edit here. Streams version independently, so packages can be
    developed and shipped without renumbering each other.
    """
    from knowledge_bases.registry import installed_packages, resolve_seed

    composed: list[Migration] = list(MIGRATIONS)
    for package in installed_packages():
        if not package.migration_stream:
            continue
        for migration in resolve_seed(package.migration_stream):
            if migration.stream != package.key:
                raise ValueError(
                    f"Package {package.key!r} contributed a migration in "
                    f"stream {migration.stream!r}: a package owns exactly its "
                    "own stream."
                )
            composed.append(migration)
    return tuple(composed)


_CONSTRAINT_PATTERN = re.compile(
    r"FOR\s*\(\w+:([`\w]+)\)\s*REQUIRE\s*(?:\w+\.([`\w]+)|\(([^)]+)\))\s+IS\s+UNIQUE",
    re.IGNORECASE,
)


def _drop_conflicting_indexes(graph: QueryRunner, statement: str) -> bool:
    """If `statement` is a CREATE CONSTRAINT statement blocked by an unowned plain index, drop that index."""
    match = _CONSTRAINT_PATTERN.search(statement)
    if not match:
        return False
    label = match.group(1).replace("`", "")
    if match.group(2):
        props = [match.group(2).replace("`", "")]
    else:
        props = [p.strip().split(".")[-1].replace("`", "") for p in match.group(3).split(",")]

    try:
        rows = graph.run(
            "SHOW INDEXES YIELD name, labelsOrTypes, properties, owningConstraint "
            "WHERE owningConstraint IS NULL"
        )
    except Exception:
        return False

    dropped = False
    for row in rows:
        labels = row.get("labelsOrTypes")
        properties = row.get("properties")
        if labels == [label] and properties == props:
            index_name = row.get("name")
            if index_name:
                graph.run(f"DROP INDEX `{index_name}`")
                dropped = True
    return dropped


def _run_migration_statement(graph: QueryRunner, statement: str) -> None:
    try:
        graph.run(statement)
    except Exception as exc:
        if "IndexAlreadyExists" in str(exc) or "already exists an index" in str(exc):
            if _drop_conflicting_indexes(graph, statement):
                graph.run(statement)
                return
        raise


def apply_migrations(
    graph: QueryRunner, migrations: tuple[Migration, ...] | None = None,
) -> list[tuple[str, int]]:
    """Apply pending migrations; return the (stream, version) pairs applied.

    Defaults to `composed_migrations()`, so a product applies the engine's
    stream plus the streams of exactly the packages it installs.

    Stream bookkeeping is framework code, not a migration: on a graph that
    predates streams, existing `SchemaMigration` rows are backfilled to
    `LEGACY_STREAM` before the composite `(stream, version)` constraint is
    created, and the old `version`-only constraint is dropped -- both are
    idempotent, so a fresh graph takes the same path harmlessly.

    RE-RUN SAFETY WHEN A MIGRATION MOVES STREAM. The bookkeeping key is
    `(stream, version)`, so a migration moved out of `"core"` into its owning
    package gets a key no existing deployment has, and WILL run again. That is why
    only migrations whose every statement is idempotent DDL were moved: each is
    `CREATE CONSTRAINT/INDEX ... IF NOT EXISTS` or `DROP ... IF EXISTS`, so the
    second application is a no-op against a graph that already has them.

    Migrations containing data statements (`MATCH ... SET`, `DETACH DELETE`) stay in
    `"core"` for exactly this reason -- re-running a rename or a delete against
    already-migrated data is not a no-op, and no backfill can distinguish "already
    applied under the old key" from "never applied". Their core version numbers are
    therefore permanent, and a moved migration's old core number is left as a gap in
    this file rather than reused.
    """
    graph.run(
        "MATCH (m:SchemaMigration) WHERE m.stream IS NULL "
        "SET m.stream = $stream",
        stream=LEGACY_STREAM,
    )
    _run_migration_statement(
        graph,
        "CREATE CONSTRAINT schema_migration_stream_version IF NOT EXISTS "
        "FOR (m:SchemaMigration) REQUIRE (m.stream, m.version) IS UNIQUE",
    )
    graph.run("DROP CONSTRAINT schema_migration_version IF EXISTS")
    if migrations is None:
        migrations = composed_migrations()
    rows = graph.run(
        "MATCH (m:SchemaMigration) "
        "RETURN m.stream AS stream, m.version AS version"
    )
    applied = {(row["stream"], row["version"]) for row in rows}
    completed: list[tuple[str, int]] = []

    for migration in migrations:
        key = (migration.stream, migration.version)
        if key in applied:
            continue
        for statement in migration.statements:
            _run_migration_statement(graph, statement)
        graph.run(
            "CREATE (m:SchemaMigration {"
            "stream: $stream, version: $version, name: $name, "
            "applied_at: datetime()})",
            stream=migration.stream,
            version=migration.version,
            name=migration.name,
        )
        completed.append(key)

    return completed
