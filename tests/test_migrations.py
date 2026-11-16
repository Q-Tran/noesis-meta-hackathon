import pytest

from migrations import (
    MIGRATIONS,
    Migration,
    apply_migrations,
    composed_migrations,
)


class FakeGraph:
    def __init__(self, applied=None):
        # applied: iterable of (stream, version) already recorded in the graph
        self.applied = applied or []
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))
        if "RETURN m.stream AS stream, m.version AS version" in query:
            return [
                {"stream": stream, "version": version}
                for stream, version in self.applied
            ]
        return []


def test_apply_migrations_records_pending_versions():
    """Compared against the COMPOSED set, not the engine's own stream.

    These two were identical until the ontology packages got streams of their own,
    which is exactly the confusion worth pinning: `apply_migrations` defaults to
    `composed_migrations()`, so asserting against `MIGRATIONS` would quietly stop
    checking any migration a package owns.
    """
    graph = FakeGraph()
    composed = composed_migrations()

    assert apply_migrations(graph) == [
        (migration.stream, migration.version) for migration in composed
    ]

    queries = [query for query, _ in graph.calls]
    assert composed[0].statements[0] in queries
    assert any("CREATE (m:SchemaMigration" in query for query in queries)


def test_apply_migrations_skips_completed_versions_and_applies_later_ones():
    graph = FakeGraph(applied=[("core", 1)])
    composed = composed_migrations()

    assert apply_migrations(graph) == [
        (migration.stream, migration.version)
        for migration in composed
        if (migration.stream, migration.version) != ("core", 1)
    ]

    queries = [query for query, _ in graph.calls]
    assert composed[0].statements[0] not in queries
    assert composed[1].statements[0] in queries


def test_migration_streams_version_independently():
    """The same version number in another stream is not 'already applied'."""
    from migrations import Migration

    foreign = Migration(
        version=1, name="pkg_init", statements=("CREATE INDEX x IF NOT EXISTS FOR (n:X) ON (n.id)",),
        stream="semantic_layer",
    )
    graph = FakeGraph(applied=[("core", 1)])

    assert apply_migrations(graph, migrations=(foreign,)) == [("semantic_layer", 1)]


def test_apply_migrations_backfills_legacy_rows_and_rekeys_the_constraint():
    graph = FakeGraph()
    apply_migrations(graph)

    queries = [query for query, _ in graph.calls]
    assert any(
        "m.stream IS NULL" in q and "SET m.stream" in q for q in queries
    )
    assert any("schema_migration_stream_version" in q for q in queries)
    assert any("DROP CONSTRAINT schema_migration_version" in q for q in queries)


def test_lexical_search_excludes_insight_statements():
    statements = next(
        migration.statements for migration in MIGRATIONS
        if migration.name == "lexical_search"
    )

    assert statements[0] == "DROP INDEX insight_statement IF EXISTS"
    assert any("bmad_entity_text" in statement for statement in statements)
    assert not any("Insight" in statement for statement in statements[1:])


def test_feedback_claims_are_indexed():
    # Owned by the `claims` package now, so it is only visible in the composed set.
    statements = next(
        migration.statements for migration in composed_migrations()
        if migration.name == "user_feedback_claims"
    )

    assert any("claim_id" in statement for statement in statements)
    assert any("evidence_claim_text" in statement for statement in statements)


def test_write_proposals_are_constrained_and_expirable():
    statements = next(
        migration.statements for migration in MIGRATIONS
        if migration.name == "knowledge_write_proposals"
    )

    assert any("write_proposal_id" in statement for statement in statements)
    assert any("expires_at" in statement for statement in statements)


def test_missing_lexical_indexes_are_repaired_forward_only():
    statements = next(
        migration.statements for migration in MIGRATIONS
        if migration.name == "repair_lexical_search_indexes"
    )

    assert any("bmad_entity_text" in statement for statement in statements)
    assert any("evidence_metadata_text" in statement for statement in statements)


def test_provenance_relationships_constrain_source_node_identity():
    statements = next(
        migration.statements for migration in MIGRATIONS
        if migration.name == "provenance_relationships"
    )

    assert any("source_node_id" in statement for statement in statements)
    assert any("SourceNode" in statement for statement in statements)


def test_drop_external_info_label_rebuilds_evidence_index_without_it():
    statements = next(
        migration.statements for migration in MIGRATIONS
        if migration.name == "drop_external_info_label"
    )

    assert any("DROP INDEX evidence_metadata_text" in statement for statement in statements)
    assert not any("ExternalInfo" in statement for statement in statements)
    assert any("evidence_metadata_text" in statement and "SourceNode" in statement
               for statement in statements)


def test_composed_migrations_include_the_engine_stream_unchanged():
    """The engine's own history must survive composition exactly: whatever
    packages contribute, every `core` migration is still applied, in order."""
    core = tuple(m for m in composed_migrations() if m.stream == "core")
    assert core == MIGRATIONS


def test_a_package_may_only_contribute_to_its_own_stream():
    """Streams are per-package so versions never collide. A package writing
    into another's stream could mark a foreign version applied, silently
    skipping that package's real migration."""
    import knowledge_bases.registry as registry
    from knowledge_bases.base import OntologyPackage

    intruder = OntologyPackage(
        key="intruder", migration_stream="tests.test_migrations:_FOREIGN_STREAM"
    )
    # Patched at `_PACKAGES`, which is what `installed_packages()` returns and what
    # composition now reads. It used to patch `ontology_packages`, and that call
    # RE-LOADED every declared package regardless of what the product installs -- so
    # composition ignored the installed set entirely. Patching the old seam now
    # leaves the test green while exercising nothing.
    real = registry._PACKAGES
    registry._PACKAGES = (intruder,)
    try:
        with pytest.raises(ValueError, match="owns exactly its own stream"):
            composed_migrations()
    finally:
        registry._PACKAGES = real


_FOREIGN_STREAM = (
    Migration(version=1, name="not_mine", statements=(), stream="semantic_layer"),
)


def test_every_declared_migration_stream_resolves_to_migrations():
    """A package naming a stream that does not import, or that holds something
    other than `Migration`s, would fail only at deploy time."""
    from knowledge_bases.registry import installed_packages, resolve_seed

    for package in installed_packages():
        if not package.migration_stream:
            continue
        contributed = resolve_seed(package.migration_stream)
        assert contributed, f"{package.key} declares an empty stream"
        assert all(isinstance(m, Migration) for m in contributed)


def test_applying_migrations_defaults_to_the_composed_set():
    """`apply_migrations(graph)` with no explicit tuple must apply what the
    product actually installs, not just the engine's stream."""
    graph = FakeGraph()

    applied = apply_migrations(graph)

    assert applied == [
        (migration.stream, migration.version)
        for migration in composed_migrations()
    ]


# -- stream ownership ----------------------------------------------------------


def _is_idempotent_ddl(statement: str) -> bool:
    """Whether re-applying `statement` to an already-migrated graph is a no-op.

    Only schema DDL guarded by `IF NOT EXISTS` / `IF EXISTS` qualifies. A data
    statement does not: re-running `MATCH (n:Skill) SET n:Tool REMOVE n:Skill` after
    it already ran is harmless only by luck, and `DETACH DELETE` is not recoverable
    at all.
    """
    import re

    return bool(re.match(
        r"^(CREATE (CONSTRAINT|INDEX|FULLTEXT INDEX|VECTOR INDEX)|DROP (CONSTRAINT|INDEX))\b",
        statement.strip(),
    ))


def test_a_migration_that_changed_stream_is_safe_to_apply_twice():
    """The one way moving a migration can corrupt a live graph.

    Bookkeeping keys on `(stream, version)`. A migration moved out of `"core"` into
    its owning package therefore has a key no existing deployment has recorded, so it
    WILL run again on the next boot. There is no backfill that fixes this in general:
    nothing can distinguish "already applied under the old key" from "never applied".

    So the safety property is not about the move, it is about the CONTENT: every
    migration outside `"core"` must consist only of idempotent DDL. This is the test
    that stops someone moving a data migration -- which is why `rename_skill_label_to_tool`
    and `retire_self_catalog_into_development` stay in the engine's stream despite
    naming domain labels.
    """
    from migrations import composed_migrations

    for migration in composed_migrations():
        if migration.stream == "core":
            continue
        for statement in migration.statements:
            assert _is_idempotent_ddl(statement), (
                f"{migration.stream} v{migration.version} ({migration.name}) contains "
                f"a non-idempotent statement: {statement[:80]!r}. A migration outside "
                "'core' re-runs on deployments that applied it as core, so it must be "
                "safe to apply twice."
            )


def test_a_package_declares_only_its_own_stream():
    """Already enforced by `composed_migrations`; asserted so the message is findable.

    A package declaring `stream="core"` would inject migrations into the engine's
    history and collide with its version numbers.
    """
    from knowledge_bases.registry import installed_packages, resolve_seed

    for package in installed_packages():
        if not package.migration_stream:
            continue
        for migration in resolve_seed(package.migration_stream):
            assert migration.stream == package.key, (
                f"package {package.key!r} declares a migration in stream "
                f"{migration.stream!r}"
            )


def test_uninstalling_a_package_removes_exactly_its_migrations():
    """The whole point of streams, and it was untestable while every migration was core.

    A personal product without `development` must not create eight CodeFunction-shaped
    constraints, and dropping it must not disturb the engine's stream or any other
    package's numbering.
    """
    import knowledge_bases.registry as registry
    from migrations import composed_migrations

    before = {(m.stream, m.version): m.name for m in composed_migrations()}
    original = registry._PACKAGES
    try:
        registry._PACKAGES = tuple(p for p in original if p.key != "development")
        after = {(m.stream, m.version): m.name for m in composed_migrations()}
    finally:
        registry._PACKAGES = original

    removed = set(before) - set(after)
    assert removed, "dropping 'development' removed no migration, so it owned none"
    assert all(stream == "development" for stream, _ in removed), (
        f"dropping 'development' removed migrations it does not own: {sorted(removed)}"
    )
    assert {k: v for k, v in before.items() if k in after} == after, (
        "an unrelated migration was renumbered or renamed"
    )


def test_core_version_numbers_of_moved_migrations_are_never_reused():
    """A reused number makes a DIFFERENT migration look already-applied.

    Four migrations left `"core"`, and their old numbers are deliberately gaps. A
    deployment holding a `("core", 5)` row would skip whatever new migration claimed
    5 -- silently, since skipping is the normal path.
    """
    from migrations import MIGRATIONS

    moved_core_versions = {3, 5, 7, 22}
    present = {migration.version for migration in MIGRATIONS}

    assert not (present & moved_core_versions), (
        f"core reused the version number of a moved migration: "
        f"{sorted(present & moved_core_versions)}"
    )


def test_every_stream_numbers_its_migrations_from_one_without_gaps():
    """A package's own stream is new, so it has no history to preserve.

    `core` is exempt: its gaps are the moved migrations and must stay.
    """
    import collections

    from migrations import composed_migrations

    by_stream = collections.defaultdict(list)
    for migration in composed_migrations():
        by_stream[migration.stream].append(migration.version)

    for stream, versions in by_stream.items():
        if stream == "core":
            continue
        assert sorted(versions) == list(range(1, len(versions) + 1)), (
            f"stream {stream!r} has gaps or duplicates: {sorted(versions)}"
        )


def test_apply_migrations_drops_conflicting_unowned_index():
    class IndexConflictGraph(FakeGraph):
        def __init__(self):
            super().__init__()
            self.index_dropped = False

        def run(self, query, **params):
            if "CREATE CONSTRAINT revision_id" in query and not self.index_dropped:
                raise Exception(
                    "Neo.ClientError.Schema.IndexAlreadyExists: There already exists an index (:Revision {id}). "
                    "A constraint cannot be created until the index has been dropped."
                )
            if "SHOW INDEXES" in query:
                return [{
                    "name": "idx_revision_id",
                    "labelsOrTypes": ["Revision"],
                    "properties": ["id"],
                    "owningConstraint": None,
                }]
            if "DROP INDEX" in query:
                self.index_dropped = True
                return []
            return super().run(query, **params)

    graph = IndexConflictGraph()
    migration = Migration(
        version=999,
        name="test_conflict",
        statements=("CREATE CONSTRAINT revision_id IF NOT EXISTS FOR (n:Revision) REQUIRE n.id IS UNIQUE",),
        stream="core",
    )
    res = apply_migrations(graph, migrations=(migration,))
    assert res == [("core", 999)]
    assert graph.index_dropped

