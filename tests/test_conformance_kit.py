"""Tests of the conformance kit itself: does it actually reject a broken package?

The kit (`knowledge_bases/conformance/`) is the suite an ontology package runs
against itself to prove it is a legal package before a product installs it.
After the planned repo split it is the ONLY check a third-party package gets, so
these tests care about one thing above all: that it FIRES. A conformance kit
that passes everything is worse than no kit at all, because it certifies broken
packages, and the author it certified them to is in another repository and has no
way to notice.

So almost every test below builds a package that is deliberately wrong in one
specific way and asserts the kit names that rule. The fixtures are constructed
in memory rather than on disk, which is also the proof that the kit works on a
package it has never seen and cannot look up in this repository's registry.

The seed-backed fixtures resolve to module-level constants at the bottom of this
file, because seeds are `"module.path:ATTR"` strings by design -- a package
declares contributions it does not eagerly import.
"""
from __future__ import annotations

import pytest

from knowledge_bases.base import (
    CascadeStep,
    EntityType,
    KindRequirement,
    KnowledgeBase,
    OntologyPackage,
    PropertyType,
    RelationshipType,
    SearchType,
    VersionedKind,
)
from knowledge_bases.conformance import (
    ConformanceError,
    PackageContext,
    Severity,
    check_package,
    check_packages,
    conformance_suite,
    installed_packages,
)
from knowledge_bases.conformance.rules import ALL_RULES, RULES_BY_NAME

THIS_MODULE = "tests.test_conformance_kit"


def _kb(key: str = "widgets", **kwargs) -> KnowledgeBase:
    return KnowledgeBase(
        key=key,
        name=key.title(),
        description=kwargs.pop(
            "description", "A bounded knowledge base used by the kit's own tests."
        ),
        labels=kwargs.pop("labels", ("Widget",)),
        relationships=kwargs.pop("relationships", ("COMPOSES",)),
        **kwargs,
    )


def _package(key: str = "widgets", **kwargs) -> OntologyPackage:
    """A minimally legal package, so each test can break exactly one thing."""
    kwargs.setdefault("knowledge_bases", (_kb(key),))
    kwargs.setdefault("entity_types_seed", f"{THIS_MODULE}:LEGAL_ENTITIES")
    kwargs.setdefault("relationship_types_seed", f"{THIS_MODULE}:LEGAL_RELATIONSHIPS")
    return OntologyPackage(key=key, **kwargs)


def _rules_fired(report) -> set[str]:
    return {finding.rule for finding in report.findings}


def _items_of(report, rule: str) -> set[str]:
    return {finding.item for finding in report.of_rule(rule)}


# -- the kit refuses to certify without checking --------------------------------


def test_a_package_with_no_rules_run_is_reported_as_vacuous_rather_than_passing():
    """The failure mode that would make every other test here worthless.

    If a filtered or empty rule set produced an empty report, the kit would
    answer "conformant" for a package it never looked at -- and that answer would
    be given to an author in another repository who has no other check. A report
    that checked nothing must be distinguishable from a report that found
    nothing.
    """
    report = check_package(_package(), peers=(), rules=())

    assert report.is_vacuous, "a report with zero rules run claimed to be a check"
    assert report.findings == ()
    assert not check_package(_package(), peers=()).is_vacuous


def test_asking_for_a_rule_that_does_not_exist_fails_instead_of_checking_nothing():
    """A typo in a `rules=` filter must not silently narrow the suite to zero.
    Skipping a rule an author believed was running is how a package ships with
    the one defect they thought they were guarding against."""
    with pytest.raises(ValueError, match="unknown conformance rule"):
        check_package(_package(), peers=(), rules=("no_such_rule",))


def test_every_rule_documents_the_invariant_it_enforces_and_where_it_came_from():
    """A conformance failure is read by someone who cannot ask us why the rule
    exists. A rule whose docstring does not name its source -- a `raise` in the
    engine, a test, or a recorded defect -- cannot be evaluated or safely changed
    by them, so the provenance is part of the deliverable, not decoration.
    """
    for rule in ALL_RULES:
        doc = rule.__doc__ or ""
        assert len(doc.split()) >= 25, f"{rule.__name__} is not documented"
        assert "Source:" in doc or "source:" in doc, (
            f"{rule.__name__} does not name where its invariant comes from"
        )


def test_a_finding_names_the_package_the_rule_the_item_and_a_remedy():
    """The design point of the whole kit. The audience is an author in a
    different repository with no access to this codebase's history, so a finding
    that does not say which item is wrong and what to do about it is as useless
    as a bare `assert False`."""
    broken = _package(knowledge_bases=(_kb("widgets", labels=("Widget", "Gizmo")),))

    report = check_package(broken, peers=())
    findings = report.of_rule("every_declared_label_is_described_by_this_package")

    assert findings, "the kit did not report an undescribed label"
    finding = findings[0]
    assert finding.package == "widgets"
    assert finding.item == "Gizmo"
    assert len(finding.detail.split()) >= 8, "the detail does not explain anything"
    assert len(finding.remedy.split()) >= 5, "the remedy is not actionable"
    assert "EntityType" in finding.remedy, "the remedy does not name the fix"
    assert "Gizmo" in str(finding)


# -- describedness ---------------------------------------------------------------


def test_a_label_declared_without_a_description_is_rejected():
    """The gate behind ontology-driven ingestion: a model choosing among bare
    label names has nothing to choose on, so an undescribed label is one that
    will be MIS-extracted, not merely undocumented. Left to composition this
    surfaces as silently wrong data."""
    broken = _package(knowledge_bases=(_kb("widgets", labels=("Widget", "Gizmo")),))

    report = check_package(broken, peers=())

    assert "every_declared_label_is_described_by_this_package" in _rules_fired(report)
    assert _items_of(
        report, "every_declared_label_is_described_by_this_package"
    ) == {"Gizmo"}
    assert not report.is_conformant


def test_describing_a_type_the_package_does_not_declare_is_rejected():
    """The reverse direction, and the one that steals ownership. Descriptions are
    single-owner -- `registry.entity_types()` raises on a label described twice --
    so a package holding a description for someone else's label stops the real
    owner from describing it at all."""
    broken = _package(entity_types_seed=f"{THIS_MODULE}:FOREIGN_ENTITIES")

    report = check_package(broken, peers=())

    rule = "no_description_targets_a_type_this_package_does_not_declare"
    assert "SomebodyElsesLabel" in _items_of(report, rule)


def test_describing_a_relationship_the_package_does_not_declare_is_rejected():
    """The `DEPENDS_ON` defect's root cause, recorded in
    `docs/operations/split_state.md`: an edge described by a package that a
    narrower product does not install leaves the edge undescribed for everyone
    who does declare it."""
    broken = _package(relationship_types_seed=f"{THIS_MODULE}:FOREIGN_RELATIONSHIPS")

    report = check_package(broken, peers=())

    rule = "no_description_targets_a_type_this_package_does_not_declare"
    assert "STOLEN_EDGE" in _items_of(report, rule)


def test_a_shared_edge_described_only_by_a_peer_is_reported_as_fragile_not_illegal():
    """This is the real `DEPENDS_ON` shape and the kit must get its severity
    right. Sharing an edge under single ownership is deliberate and legal --
    demanding every declarer describe it would ban sharing -- but it makes this
    package incomplete without that peer, which is precisely how the documented
    defect reached a shipped profile. So: WARNING, named, with the peer named.
    """
    peer = OntologyPackage(
        key="tooling",
        knowledge_bases=(_kb("tooling", labels=("Tool",), relationships=("USES",)),),
        relationship_types_seed=f"{THIS_MODULE}:SHARED_RELATIONSHIPS",
        entity_types_seed=f"{THIS_MODULE}:PEER_ENTITIES",
    )
    sharer = _package(
        knowledge_bases=(_kb("widgets", relationships=("COMPOSES", "USES")),),
    )

    report = check_package(sharer, peers=(peer,))
    findings = report.of_rule(
        "every_declared_relationship_is_described_by_this_package"
    )

    assert [f.item for f in findings] == ["USES"]
    assert findings[0].severity is Severity.WARNING, (
        "a legally shared edge was reported as illegal, which would ban sharing"
    )
    assert "tooling" in findings[0].detail, "the finding does not name the describer"
    assert report.is_conformant, "a warning must not make the package illegal"


def test_an_undescribed_edge_with_no_peer_to_blame_is_an_error():
    """The other half of the same rule: without a peer that describes it, the
    edge is simply undescribed and the package is not legal. The kit must not
    downgrade a real error just because it cannot see the whole product."""
    broken = _package(knowledge_bases=(_kb("widgets", relationships=("COMPOSES", "ORBITS")),))

    report = check_package(broken, peers=())
    findings = report.of_rule(
        "every_declared_relationship_is_described_by_this_package"
    )

    assert [f.item for f in findings] == ["ORBITS"]
    assert findings[0].severity is Severity.ERROR


def test_a_placeholder_description_is_rejected():
    """A one-word description disambiguates nothing, and placeholders are what
    authors actually leave behind -- not empty strings."""
    broken = _package(entity_types_seed=f"{THIS_MODULE}:THIN_ENTITIES")

    report = check_package(broken, peers=())

    assert "Widget" in _items_of(report, "descriptions_are_substantive")


def test_a_contrast_with_a_type_that_does_not_exist_is_rejected():
    """`not_to_be_confused_with` is rendered into the extraction prompt, so a
    stale name there actively teaches the model about a type that is not there --
    worse than saying nothing."""
    broken = _package(entity_types_seed=f"{THIS_MODULE}:STALE_CONTRAST_ENTITIES")
    peer = OntologyPackage(key="tooling", knowledge_bases=(_kb("tooling", labels=("Tool",)),))

    report = check_package(broken, peers=(peer,))

    assert "Widget -> Vanished" in _items_of(
        report, "confusable_types_name_types_that_exist"
    )


# -- endpoints -------------------------------------------------------------------


def test_an_endpoint_naming_a_label_nothing_declares_is_rejected():
    """Endpoint typing is what rejects `(Metric)-[:REPORTS_TO]->(Table)`. An
    endpoint naming a label that does not exist can never match, so the
    relationship is silently unusable rather than loudly broken.

    Reported per ROLE rather than per label, because unusability is a property of the
    whole endpoint list: an edge is satisfied if ANY of its declared labels exists, so
    naming one absent label among three narrows the edge while naming the only label
    kills it. The finding fires here because `Nonexistent` is the object's only entry.
    """
    broken = _package(relationship_types_seed=f"{THIS_MODULE}:BAD_ENDPOINT_RELATIONSHIPS")
    peer = OntologyPackage(key="tooling", knowledge_bases=(_kb("tooling", labels=("Tool",)),))

    report = check_package(broken, peers=(peer,))

    rule = "relationship_endpoints_reference_labels_that_exist"
    assert "COMPOSES.object" in _items_of(report, rule)
    assert any(
        finding.severity is Severity.ERROR
        for finding in report.findings
        if finding.rule == rule
    ), "an endpoint with no usable label at all is an error, not advice"


def test_an_endpoint_that_keeps_one_usable_label_is_advice_not_an_error():
    """The distinction that lets one declaration serve several products.

    `decisions` declares `ADDRESSES.subject=(Idea, Requirement)`. A personal product
    installs `Idea` but not `Requirement`, and the edge works perfectly -- it matches
    everything it can match, and the prompt renders only `Idea`. Treating that as an
    error would force the declaration down to the narrowest product's vocabulary and
    degrade every wider one, which is the opposite of what the split is for.

    Nine such cases appear in `noesis-individual` across `decisions`, `architecture` and
    `claims`. They are warnings, so a genuinely dead endpoint still stands out.
    """
    partial = _package(
        relationship_types_seed=f"{THIS_MODULE}:PARTIAL_ENDPOINT_RELATIONSHIPS"
    )
    peer = OntologyPackage(key="tooling", knowledge_bases=(_kb("tooling", labels=("Tool",)),))

    report = check_package(partial, peers=(peer,))

    rule = "relationship_endpoints_reference_labels_that_exist"
    fired = [f for f in report.findings if f.rule == rule]
    assert fired, "the partially-absent endpoint should still be reported"
    assert all(finding.severity is Severity.WARNING for finding in fired), (
        "an edge that still has a usable endpoint is narrowed, not broken"
    )
    assert not report.errors, "a narrowed edge must not fail the package"


def test_an_endpoint_pointing_at_a_peers_label_is_not_a_finding():
    """Edges are how packages join: 9 of the 11 packages in this repo point at
    another package's label deliberately. A kit that flagged that would be
    unusable, and worse, would push authors to duplicate labels."""
    broken = _package(relationship_types_seed=f"{THIS_MODULE}:PEER_ENDPOINT_RELATIONSHIPS")
    peer = OntologyPackage(
        key="tooling",
        knowledge_bases=(_kb("tooling", labels=("Tool",)),),
        entity_types_seed=f"{THIS_MODULE}:PEER_ENTITIES",
    )

    report = check_package(broken, peers=(peer,))

    assert report.of_rule("relationship_endpoints_reference_labels_that_exist") == ()


def test_an_open_endpoint_must_say_it_is_deliberate():
    """An open endpoint switches off endpoint validation for that edge entirely,
    so an extractor may attach it to anything. Every open endpoint in this repo
    explains itself in its description; the risk is the author who left the
    fields empty because filling them in was work, and the two must not look
    alike."""
    silent = _package(relationship_types_seed=f"{THIS_MODULE}:SILENTLY_OPEN_RELATIONSHIPS")

    report = check_package(silent, peers=())
    findings = report.of_rule("an_open_endpoint_is_deliberate_and_explained")

    assert [f.item for f in findings] == ["COMPOSES"]
    assert findings[0].severity is Severity.WARNING


def test_an_open_endpoint_that_explains_itself_is_accepted():
    """The other half: the kit must not punish the correct form, or authors will
    learn to type endpoints falsely just to silence it."""
    explained = _package(relationship_types_seed=f"{THIS_MODULE}:EXPLAINED_OPEN_RELATIONSHIPS")

    report = check_package(explained, peers=())

    assert report.of_rule("an_open_endpoint_is_deliberate_and_explained") == ()


# -- properties and identity ------------------------------------------------------


def test_an_enum_with_a_duplicate_or_empty_value_is_rejected():
    """A drifted or degenerate enum is worse than no enum: it presents itself as
    the closed vocabulary while refusing or blurring values the code writes.
    `PropertyType.__post_init__` already blocks an enum on a non-string type;
    this covers what it cannot see."""
    broken = _package(entity_types_seed=f"{THIS_MODULE}:BAD_ENUM_ENTITIES")

    report = check_package(broken, peers=())
    items = _items_of(report, "declared_property_enums_are_usable")

    assert "Widget.status" in items, "a duplicated enum value was accepted"
    assert "Widget.phase" in items, "an empty enum value was accepted"


def test_a_value_type_the_engine_cannot_write_is_rejected():
    """`value_type` tells a writer how to serialise the property. A type outside
    the engine's set is not a documentation slip: nothing will know how to
    write it."""
    broken = _package(entity_types_seed=f"{THIS_MODULE}:BAD_VALUE_TYPE_ENTITIES")

    report = check_package(broken, peers=())

    assert "Widget.count" in _items_of(report, "declared_property_enums_are_usable")


def test_an_identity_property_that_the_type_does_not_declare_is_rejected():
    """Identity is what stops re-ingesting the same fact from creating a node per
    phrasing -- but ingestion also DROPS undeclared properties, so an identity
    naming a property the type does not declare can never be populated. Every
    mention then becomes a new node with a null key: convergence fails silently,
    in the one place the graph cannot recover from."""
    broken = _package(entity_types_seed=f"{THIS_MODULE}:UNDECLARED_IDENTITY_ENTITIES")

    report = check_package(broken, peers=())

    assert "Widget.serial" in _items_of(
        report, "an_identity_property_is_declared_as_a_property"
    )


def test_a_property_declared_both_bare_and_described_is_rejected():
    """`described_properties()` keys by name, so the duplicate silently keeps
    whichever description came last while `property_names()` still lists both.
    This is exactly what a half-finished bare-string-to-PropertyType migration
    leaves behind, and the two views of the type then disagree."""
    broken = _package(entity_types_seed=f"{THIS_MODULE}:DOUBLE_PROPERTY_ENTITIES")

    report = check_package(broken, peers=())

    assert "Widget.name" in _items_of(report, "a_property_is_declared_once_per_type")


# -- extractability ---------------------------------------------------------------


def test_an_extractable_edge_whose_every_endpoint_is_internal_is_reported():
    """The one combination the two `extractable` flags cannot both be right
    about: the edge is rendered into the extraction prompt while the model is
    forbidden to create either end of it. Offering it can only produce a
    confident wrong guess."""
    broken = _package(
        knowledge_bases=(_kb("widgets", labels=("Widget",), relationships=("COMPOSES",)),),
        entity_types_seed=f"{THIS_MODULE}:INTERNAL_ENTITIES",
    )

    report = check_package(broken, peers=())
    findings = report.of_rule("an_extractable_relationship_has_an_extractable_endpoint")

    assert [f.item for f in findings] == ["COMPOSES"]
    assert findings[0].severity is Severity.WARNING


# -- cascade steps ----------------------------------------------------------------


def test_a_cascade_step_producing_another_packages_label_is_rejected():
    """Uninstalling a package must remove the reasoning that fabricates its
    entities. A step producing a label its own package does not declare would
    survive that package's removal and keep writing a label that no longer
    exists -- and meanwhile fabricates another package's entities."""
    broken = _package(cascade_steps_seed=f"{THIS_MODULE}:FOREIGN_OUTPUT_STEPS")
    peer = OntologyPackage(key="tooling", knowledge_bases=(_kb("tooling", labels=("Tool",)),))

    report = check_package(broken, peers=(peer,))
    rule = "cascade_steps_are_owned_and_produce_only_extractable_owned_labels"

    assert "widgets.steal -> Tool" in _items_of(report, rule)
    assert not report.is_conformant


def test_a_cascade_step_producing_an_internal_label_is_rejected():
    """A step that invents telemetry or a versioning spine is a bug, not a
    feature: those labels exist because only repository code may write them, and
    a cascade step asserts something nobody said."""
    broken = _package(
        entity_types_seed=f"{THIS_MODULE}:INTERNAL_ENTITIES",
        cascade_steps_seed=f"{THIS_MODULE}:INTERNAL_OUTPUT_STEPS",
    )

    report = check_package(broken, peers=())
    rule = "cascade_steps_are_owned_and_produce_only_extractable_owned_labels"

    assert "widgets.spine -> Widget" in _items_of(report, rule)


def test_a_cascade_step_claiming_an_owner_it_is_not_declared_by_is_rejected():
    """Ownership is verified, not trusted from the declaration -- otherwise the
    "uninstalling a package removes its reasoning" guarantee is prose."""
    broken = _package(cascade_steps_seed=f"{THIS_MODULE}:MISOWNED_STEPS")

    report = check_package(broken, peers=())
    rule = "cascade_steps_are_owned_and_produce_only_extractable_owned_labels"

    assert "widgets.misowned" in _items_of(report, rule)


def test_a_cascade_step_writing_an_edge_nobody_describes_is_rejected():
    """A cascade write is validated like any other write, so a step naming an
    undescribed edge fails at EXECUTION -- after a user was told the inference
    would be made -- rather than at packaging time."""
    broken = _package(cascade_steps_seed=f"{THIS_MODULE}:UNDESCRIBED_EDGE_STEPS")

    report = check_package(broken, peers=())
    rule = "a_cascade_step_writes_a_relationship_that_is_described"

    assert "widgets.ghost -[:VANISHED]->" in _items_of(report, rule)


# -- dispatch surfaces -------------------------------------------------------------


def test_a_write_command_belonging_to_another_package_is_rejected():
    """The verified original defect: `add_business_term_alias` was hardcoded in
    the orchestrator, so a profile that dropped `business_dictionary` still
    advertised an allowlisted mutation whose handler no longer shipped."""
    broken = _package(write_commands_seed=f"{THIS_MODULE}:FOREIGN_WRITE_COMMANDS")

    report = check_package(broken, peers=())
    items = _items_of(report, "write_commands_are_namespaced_to_this_package")

    assert "tooling.do_something" in items


def test_an_unnamespaced_write_command_is_rejected():
    """Write commands share one flat allowlist keyed by `key`, so a bare verb can
    shadow another package's command and silently swap which handler a confirmed
    write runs."""
    broken = _package(write_commands_seed=f"{THIS_MODULE}:UNNAMESPACED_WRITE_COMMANDS")

    report = check_package(broken, peers=())

    assert "rename" in _items_of(
        report, "write_commands_are_namespaced_to_this_package"
    )


def test_a_tool_attributed_to_another_package_is_rejected():
    """The read-surface twin of the write-command defect: tool keys are split on
    '.' to attribute a tool to its package, and per-domain MCP servers filter on
    `knowledge_base`, so a mis-attributed tool is advertised under an ontology
    that may not be installed."""
    broken = _package(tools_seed=f"{THIS_MODULE}:FOREIGN_TOOLS")

    report = check_package(broken, peers=())
    items = _items_of(report, "contributed_tools_belong_to_this_package")

    assert "tooling.frobnicate" in items


def test_a_tool_key_that_collides_with_a_peers_tool_is_rejected():
    """The composed catalog is built in package order, so a duplicate key means
    the later package silently shadows the earlier one's handler."""
    peer = OntologyPackage(
        key="tooling",
        knowledge_bases=(_kb("tooling", labels=("Tool",)),),
        tools_seed=f"{THIS_MODULE}:PEER_TOOLS",
    )
    broken = _package(tools_seed=f"{THIS_MODULE}:COLLIDING_TOOLS")

    report = check_package(broken, peers=(peer,))

    assert "widgets.shared" in _items_of(
        report, "contributed_tools_belong_to_this_package"
    )


def test_an_expert_opening_on_a_tool_nothing_contributes_is_rejected():
    """The expert loop starts the delegated child at the entry tool, so an entry
    tool naming something uninstalled sends every delegation of that expert into
    a dead dispatch on its first move."""
    peer = OntologyPackage(key="tooling", knowledge_bases=(_kb("tooling", labels=("Tool",)),))
    broken = _package(experts_seed=f"{THIS_MODULE}:DANGLING_EXPERTS")

    report = check_package(broken, peers=(peer,))
    rule = "a_contributed_experts_entry_tools_are_reachable"

    assert "widgets -> widgets.vanished" in _items_of(report, rule)


def test_an_expert_opening_on_a_control_plane_tool_is_accepted():
    """`architecture` really does open on `orchestration.hybrid_search`. The
    control plane ships with the engine, so depending on it is not a
    cross-ontology dependency -- reported as INFO so the dependency is visible
    without being a failure."""
    peer = OntologyPackage(
        key="orchestration",
        knowledge_bases=(_kb("orchestration", labels=("RoutingTrace",)),),
        tools_seed=f"{THIS_MODULE}:CONTROL_PLANE_TOOLS",
    )
    package = _package(experts_seed=f"{THIS_MODULE}:CONTROL_PLANE_EXPERTS")

    report = check_package(package, peers=(peer,))
    findings = report.of_rule("a_contributed_experts_entry_tools_are_reachable")

    assert [f.severity for f in findings] == [Severity.INFO]
    assert report.is_conformant


def test_a_versioned_kind_naming_a_label_the_package_does_not_declare_is_rejected():
    """"Who owns a Requirement" must have exactly one answer. The composed kind
    registry raises on this, which means one bad kind makes the WHOLE product
    fail to validate any changeset -- so the package-local half has to be
    checkable before release."""
    broken = _package(kinds_seed=f"{THIS_MODULE}:FOREIGN_KINDS")
    peer = OntologyPackage(key="tooling", knowledge_bases=(_kb("tooling", labels=("Tool",)),))

    report = check_package(broken, peers=(peer,))

    assert "Tool" in _items_of(report, "versioned_kinds_are_owned_and_declared")


def test_a_versioned_kind_whose_requirement_names_an_unknown_edge_is_rejected():
    """A creation rule referencing an edge nothing declares can never be
    satisfied, so the kind can never be created through a changeset at all --
    the tool advertises it and every attempt is refused."""
    broken = _package(kinds_seed=f"{THIS_MODULE}:BAD_REQUIREMENT_KINDS")
    peer = OntologyPackage(key="tooling", knowledge_bases=(_kb("tooling", labels=("Tool",)),))

    report = check_package(broken, peers=(peer,))
    items = _items_of(report, "versioned_kinds_are_owned_and_declared")

    assert "Widget requires -[:NONEXISTENT]->" in items


def test_a_changeset_relationship_that_is_not_declared_ontology_is_rejected():
    """The nastiest failure shape available here: the changeset validates, the
    user is told the write was accepted, and then `graph.upsert_rel` refuses it
    because the graph ontology is enforced independently."""
    broken = _package(changeset_relationships=("UNDECLARED_EDGE",))

    report = check_package(broken, peers=())
    rule = "changeset_relationships_are_declared_by_this_package"

    assert "UNDECLARED_EDGE" in _items_of(report, rule)


def test_a_search_type_scoped_to_another_packages_family_is_rejected():
    """`family` is the caller-visible search scope, so a package that files its
    labels under another family makes them searchable only under a name it does
    not own -- and they vanish when that package is uninstalled."""
    broken = _package(
        search_types=(SearchType("tooling", "Widget", "widget_text"),),
    )

    report = check_package(broken, peers=())

    assert "tooling.Widget" in _items_of(
        report, "search_types_target_this_packages_own_labels"
    )


def test_a_search_type_with_no_index_at_all_is_rejected():
    """Registered but unsearchable: `hybrid_search` has nothing to retrieve it
    with, so the type appears in the scope allowlist and returns nothing."""
    broken = _package(search_types=(SearchType("widgets", "Widget", None),))

    report = check_package(broken, peers=())

    assert "widgets.Widget" in _items_of(
        report, "search_types_target_this_packages_own_labels"
    )


def test_a_search_type_reading_the_wrong_identity_property_is_rejected():
    """The quietest failure in the kit, and a live one in this repository.
    `hybrid_search` reads `properties[identity_property]` and DROPS any candidate
    whose value is null, so when the SearchType and the EntityType disagree the
    label is indexed, searchable, and silently returns nothing -- with no error
    logged anywhere."""
    broken = _package(
        entity_types_seed=f"{THIS_MODULE}:NAME_IDENTITY_ENTITIES",
        search_types=(SearchType("widgets", "Widget", "widget_text"),),
    )

    report = check_package(broken, peers=())

    assert "widgets.Widget" in _items_of(
        report, "a_search_type_reads_the_labels_identity_property"
    )


# -- declaration hygiene and independence -------------------------------------------


def test_a_package_whose_key_names_no_knowledge_base_it_declares_is_rejected():
    """Every ownership check in the engine compares against the PACKAGE key while
    the registry is keyed by the KNOWLEDGE BASE key. If they disagree, the
    ownership rules do not fail -- they pass vacuously against an empty label
    set, which is the worst possible outcome for a conformance kit."""
    broken = OntologyPackage(key="widgets", knowledge_bases=(_kb("gadgets"),))

    report = check_package(broken, peers=())

    assert "widgets" in _items_of(report, "package_declares_its_own_knowledge_base")


def test_a_package_declaring_no_knowledge_base_at_all_is_rejected():
    """An empty package composes an empty ontology, so installing it is a no-op
    that nonetheless looks like a successful install."""
    report = check_package(OntologyPackage(key="widgets"), peers=())

    assert "widgets" in _items_of(report, "package_declares_its_own_knowledge_base")
    assert "widgets" in _items_of(report, "the_package_composes_standalone")


def test_a_seed_reference_that_does_not_resolve_is_rejected():
    """Seeds resolve LAZILY, which is deliberate -- declaring a package must cost
    only one import. The price is that a typo stays invisible until the first
    caller touches that surface, which in a product is startup or a user's first
    search. This rule is what pays that price."""
    broken = _package(tools_seed=f"{THIS_MODULE}:NOT_A_REAL_ATTRIBUTE")

    report = check_package(broken, peers=())
    findings = report.of_rule("every_seed_reference_resolves")

    assert findings, "an unresolvable seed was accepted"
    assert "tools_seed" in findings[0].item
    assert "has no attribute" in findings[0].detail


def test_a_seed_pointing_at_a_module_that_does_not_exist_is_rejected():
    broken = _package(kinds_seed="no.such.module:KINDS")

    report = check_package(broken, peers=())

    assert report.of_rule("every_seed_reference_resolves"), (
        "a seed naming a nonexistent module was accepted"
    )


def test_a_malformed_seed_reference_is_rejected():
    """`"module.path"` without `:ATTR` is not a seed. `resolve_seed` would
    `getattr(module, "")` and fail with an unhelpful AttributeError at first use.
    """
    broken = _package(experts_seed="knowledge_bases.base")

    report = check_package(broken, peers=())

    assert report.of_rule("every_seed_reference_resolves")


def test_a_migration_in_another_packages_stream_is_rejected():
    """Streams version independently so packages can ship without renumbering
    each other. A migration in someone else's stream is counted against THEIR
    applied version, so it is silently skipped or re-run depending on what they
    had applied."""
    broken = _package(migration_stream=f"{THIS_MODULE}:FOREIGN_MIGRATIONS")

    report = check_package(broken, peers=())

    assert report.of_rule("declared_migrations_are_in_this_packages_stream")


def test_a_lowercase_relationship_name_is_rejected():
    """Type names are interpolated straight into Cypher by `graph.upsert_rel`, so
    a name in the wrong case does not fail loudly -- it creates or matches a
    different type than every writer intended."""
    broken = _package(
        knowledge_bases=(_kb("widgets", relationships=("composes",)),),
        relationship_types_seed=f"{THIS_MODULE}:LOWERCASE_RELATIONSHIPS",
    )

    report = check_package(broken, peers=())

    assert "composes" in _items_of(report, "declared_types_are_well_formed_names")


def test_declaring_a_revision_label_for_a_kind_this_package_versions_is_rejected():
    """`<Kind>Revision` is GENERATED by the methodology engine for every revision
    of a managed kind, so declaring it while also contributing `<Kind>` means the
    declared label and the generated nodes are the same thing under two owners.
    """
    broken = _package(
        knowledge_bases=(_kb("widgets", labels=("Widget", "WidgetRevision")),),
        entity_types_seed=f"{THIS_MODULE}:REVISION_ENTITIES",
        kinds_seed=f"{THIS_MODULE}:WIDGET_KINDS",
    )

    report = check_package(broken, peers=())

    assert "WidgetRevision" in _items_of(
        report, "declared_types_are_well_formed_names"
    )


def test_a_package_that_names_a_real_declared_revision_label_is_not_rejected():
    """The guard on the guard. `orchestration` declares `ExpertRevision` as a real
    label of its own and contributes no `Expert` kind, so a suffix-only rule would
    fire on shipped, correct ontology -- and a kit that cries wolf on the real
    tree is a kit nobody runs."""
    report = check_package("orchestration")

    assert _items_of(report, "declared_types_are_well_formed_names") == set()


def test_a_declaration_module_that_imports_more_than_the_contract_is_rejected(tmp_path):
    """`package.py` is executed STANDALONE at composition time, before any
    package's `__init__` has run, specifically so composition cannot pull half a
    codebase into its import path. A declaration that imports anything else
    reintroduces the load-order deadlock that design exists to prevent."""
    root = tmp_path / "knowledge_bases" / "widgets"
    root.mkdir(parents=True)
    (root / "package.py").write_text(
        "from knowledge_bases.base import OntologyPackage\n"
        "from knowledge_bases.registry import KNOWLEDGE_BASES\n"
        "PACKAGE = OntologyPackage(key='widgets')\n"
    )

    report = check_package(_package(), peers=(), source_root=root)
    findings = report.of_rule("the_declaration_module_imports_only_the_contract")

    assert findings, "a declaration importing the registry was accepted"
    assert "knowledge_bases.registry" in findings[0].item


def test_a_package_importing_a_sibling_ontology_is_rejected(tmp_path):
    """pip-not-installed is what a product distribution's missing package looks
    like. A package that imports a sibling ontology makes any product installing
    it without that sibling die at IMPORT, which no composition test can see."""
    base = tmp_path / "knowledge_bases"
    (base / "widgets").mkdir(parents=True)
    (base / "gadgets").mkdir(parents=True)
    (base / "gadgets" / "package.py").write_text("PACKAGE = None\n")
    (base / "widgets" / "queries.py").write_text(
        "from knowledge_bases.gadgets.models import Gadget\n"
    )

    report = check_package(_package(), peers=(), source_root=base / "widgets")
    rule = "the_package_does_not_import_a_sibling_ontology_package"
    findings = report.of_rule(rule)

    assert [f.item for f in findings] == ["widgets -> gadgets"]
    assert "queries.py:1" in findings[0].detail, (
        "the finding does not name the import site, so the author must grep"
    )


def test_importing_an_engine_module_is_not_mistaken_for_a_sibling_ontology(tmp_path):
    """Every package legitimately imports the engine. `knowledge_bases.registry`
    and `knowledge_bases.mcp` are flat engine MODULES in the same namespace as
    the package directories, so a rule that matched on the namespace alone would
    report all 11 real packages as coupled -- and the kit would be ignored."""
    base = tmp_path / "knowledge_bases"
    (base / "widgets").mkdir(parents=True)
    (base / "widgets" / "queries.py").write_text(
        "from knowledge_bases.registry import KNOWLEDGE_BASES\n"
        "from knowledge_bases.mcp import dispatch_tool\n"
        "from knowledge_bases.triplets import Triplet\n"
    )

    report = check_package(_package(), peers=(), source_root=base / "widgets")

    assert report.of_rule(
        "the_package_does_not_import_a_sibling_ontology_package"
    ) == ()


def test_importing_the_control_plane_is_not_a_finding(tmp_path):
    """All 8 domain packages import `tools`/`orchestration` for the shared
    catalog and expert builders, and those ship with the engine. Flagging that
    direction would make the kit unusable; the REVERSE direction is the real
    defect and is covered below."""
    base = tmp_path / "knowledge_bases"
    (base / "widgets").mkdir(parents=True)
    (base / "tools").mkdir(parents=True)
    (base / "tools" / "package.py").write_text("PACKAGE = None\n")
    (base / "widgets" / "catalog.py").write_text(
        "from knowledge_bases.tools.catalog import _skill\n"
    )

    report = check_package(_package(), peers=(), source_root=base / "widgets")

    assert report.of_rule(
        "the_package_does_not_import_a_sibling_ontology_package"
    ) == ()


def test_a_control_plane_package_importing_a_domain_ontology_is_rejected(tmp_path):
    """The direction that blocks extracting the engine at all: core must own zero
    ontologies, and `docs/operations/split_state.md` measured 26 of these as the
    hardest remaining coupling. Same rule, opposite direction, different remedy --
    invert the dependency rather than vendor the helper."""
    base = tmp_path / "knowledge_bases"
    (base / "orchestration").mkdir(parents=True)
    (base / "widgets").mkdir(parents=True)
    (base / "widgets" / "package.py").write_text("PACKAGE = None\n")
    (base / "orchestration" / "commands.py").write_text(
        "from knowledge_bases.widgets.models import Widget\n"
    )
    control_plane = OntologyPackage(
        key="orchestration", knowledge_bases=(_kb("orchestration", labels=("RoutingTrace",)),)
    )

    report = check_package(
        control_plane, peers=(), source_root=base / "orchestration"
    )
    findings = report.of_rule(
        "the_package_does_not_import_a_sibling_ontology_package"
    )

    assert [f.item for f in findings] == ["orchestration -> widgets"]
    assert "invert" in findings[0].remedy, (
        "a control-plane coupling got the ontology package's remedy, which would "
        "tell the engine to vendor a domain model"
    )


def test_a_package_with_a_duplicate_knowledge_base_key_fails_to_compose_standalone():
    """A standalone ontology distribution IS a one-package product, so the
    narrowest composition has to work. Calling the engine's real `compose()`
    rather than re-implementing it is deliberate: a re-implementation could drift
    from the code that actually runs at startup."""
    broken = OntologyPackage(
        key="widgets", knowledge_bases=(_kb("widgets"), _kb("widgets"))
    )

    report = check_package(broken, peers=())

    assert "widgets" in _items_of(report, "the_package_composes_standalone")


# -- severity, isolation, and the pytest integration ---------------------------------


def test_the_kit_distinguishes_illegal_from_merely_fragile():
    """If everything were an error, the shared-edge findings would make 8 of this
    repo's 11 packages illegal and the kit would be switched off. If everything
    were a warning, nothing would ever fail. The split is what makes it usable:
    errors mean not-a-legal-package, warnings mean legal-but-depends-on-a-peer.
    """
    peer = OntologyPackage(
        key="tooling",
        knowledge_bases=(_kb("tooling", labels=("Tool",), relationships=("USES",)),),
        relationship_types_seed=f"{THIS_MODULE}:SHARED_RELATIONSHIPS",
        entity_types_seed=f"{THIS_MODULE}:PEER_ENTITIES",
    )
    fragile = _package(knowledge_bases=(_kb("widgets", relationships=("COMPOSES", "USES")),))

    report = check_package(fragile, peers=(peer,))

    assert report.warnings and not report.errors
    assert report.is_conformant
    report.raise_for(Severity.ERROR)  # must not raise
    with pytest.raises(ConformanceError):
        report.raise_for(Severity.WARNING)


def test_strict_mode_raises_an_error_that_carries_the_findings():
    """The one-line integration an author writes:
    `check_package(PACKAGE, strict=True)`. It subclasses `AssertionError` so any
    runner reads it as a test failure, while still carrying structured findings
    for a caller that wants to inspect rather than print."""
    broken = _package(knowledge_bases=(_kb("widgets", labels=("Widget", "Gizmo")),))

    with pytest.raises(ConformanceError) as raised:
        check_package(broken, peers=(), strict=True)

    assert isinstance(raised.value, AssertionError)
    assert raised.value.findings, "the error carries no structured findings"
    assert "Gizmo" in str(raised.value)
    assert any("EntityType" in f.remedy for f in raised.value.findings)


def test_a_rule_that_cannot_be_decided_without_peers_reports_instead_of_passing():
    """A foreign package checked in isolation cannot know whether some other
    package declares the label its edge points at. Quietly passing would be the
    dangerous answer: the author would read a clean report as proof. INFO records
    that the question was asked and not settled."""
    isolated = _package(relationship_types_seed=f"{THIS_MODULE}:PEER_ENDPOINT_RELATIONSHIPS")

    report = check_package(isolated, peers=())
    findings = report.of_rule("relationship_endpoints_reference_labels_that_exist")

    assert findings, "an undecidable endpoint check reported nothing at all"
    assert all(f.severity is Severity.INFO for f in findings)
    assert report.is_conformant, "an undecidable check must not fail the package"
    assert "no peers were supplied" in findings[0].detail


def test_the_same_defect_is_an_error_once_peers_are_supplied():
    """The other side of the previous test: supplying the peers must convert
    "cannot tell" into a verdict, or the INFO would be a permanent excuse."""
    peer = OntologyPackage(key="tooling", knowledge_bases=(_kb("tooling", labels=("Gadget",)),))
    isolated = _package(relationship_types_seed=f"{THIS_MODULE}:PEER_ENDPOINT_RELATIONSHIPS")

    report = check_package(isolated, peers=(peer,))
    findings = report.of_rule("relationship_endpoints_reference_labels_that_exist")

    assert [f.severity for f in findings] == [Severity.ERROR]
    assert not report.is_conformant


def test_the_kit_checks_a_package_it_has_never_seen_without_touching_the_registry():
    """The whole purpose: after the split, the package under test lives in another
    repository and is absent from this product's profile. If any rule needed the
    composed registry to contain the package, the kit would only ever work on the
    11 packages that ship here."""
    from knowledge_bases.registry import KNOWLEDGE_BASES

    stranger = _package(key="widgets")
    assert "widgets" not in KNOWLEDGE_BASES, "fixture leaked into the real registry"

    report = check_package(stranger, peers=())

    assert report.is_conformant
    assert len(report.rules_run) == len(ALL_RULES)
    assert "widgets" not in KNOWLEDGE_BASES


def test_the_generated_pytest_suite_has_one_test_per_rule():
    """A package author's integration is `TestConformance = conformance_suite(
    PACKAGE)`. One test per rule rather than one assert over everything, so a
    broken package reports every rule it breaks in a single run, each under the
    rule name the author has to look up."""
    suite = conformance_suite(_package(), peers=())

    methods = {name for name in dir(suite) if name.startswith("test_")}

    assert len(methods) == len(ALL_RULES) + 1  # + the non-vacuity guard
    for rule in ALL_RULES:
        assert f"test_{rule.__name__}" in methods
    assert getattr(suite, "conformance_report").is_conformant


def test_the_generated_suite_fails_on_the_rule_that_was_broken():
    """The generated suite must be a real test, not a wrapper that always passes.
    The failing method has to be the one named after the broken rule, because that
    name is the author's only lead."""
    broken = _package(knowledge_bases=(_kb("widgets", labels=("Widget", "Gizmo")),))
    suite = conformance_suite(broken, peers=())
    instance = suite()

    with pytest.raises(AssertionError, match="Gizmo"):
        instance.test_every_declared_label_is_described_by_this_package()

    instance.test_the_suite_checked_something()
    instance.test_descriptions_are_substantive()  # an unrelated rule still passes


def test_the_generated_suite_carries_each_rules_rationale_as_its_docstring():
    """A failing test in a foreign repo shows its docstring. Carrying the rule's
    own documentation is what makes the failure self-explanatory to an author who
    cannot read this codebase."""
    suite = conformance_suite(_package(), peers=())

    method = getattr(suite, "test_cascade_steps_are_owned_and_produce_only_extractable_owned_labels")

    assert method.__doc__ and "Source:" in method.__doc__


# -- the 11 packages this repository actually ships ------------------------------------





# -- fixtures ------------------------------------------------------------------------
# Seeds are `"module.path:ATTR"` strings by design, so a package's contributions
# must be module-level constants somewhere. These are that somewhere. Each one is
# wrong in exactly one way, named after the way it is wrong, so a failing test
# points at the defect it was built from.


class _FakeTool:
    """Enough of a `Tool` for the ownership rules: they read `key` and
    `knowledge_base` only, and building a real one would drag the whole tool
    catalog (and its schemas) into a test about ownership."""

    def __init__(self, key: str, knowledge_base: str) -> None:
        self.key = key
        self.knowledge_base = knowledge_base


class _FakeWriteCommand:
    def __init__(self, key: str, knowledge_base: str) -> None:
        self.key = key
        self.knowledge_base = knowledge_base


class _FakeExpert:
    def __init__(self, key: str, entry_tools: tuple[str, ...]) -> None:
        self.key = key
        self.entry_tools = entry_tools


class _FakeMigration:
    def __init__(self, version: int, stream: str) -> None:
        self.version = version
        self.stream = stream


_WIDGET_DESCRIPTION = (
    "A discrete manufactured part tracked through assembly, as distinct from the "
    "assembly that contains it."
)
_COMPOSES_DESCRIPTION = (
    "Structural containment: the subject is built out of the object, which is a "
    "standing fact about the design rather than a delivery dependency."
)

LEGAL_ENTITIES = (
    EntityType(
        label="Widget",
        description=_WIDGET_DESCRIPTION,
        identity="id",
        properties=(PropertyType("id", "Stable surrogate key assigned at intake."),),
    ),
)
LEGAL_RELATIONSHIPS = (
    RelationshipType(
        relationship="COMPOSES",
        description=_COMPOSES_DESCRIPTION,
        subject=("Widget",),
        object=("Widget",),
    ),
)

PEER_ENTITIES = (
    EntityType(
        label="Tool",
        description="An instrument used to work a widget, owned by the peer package.",
        properties=(PropertyType("id", "Stable surrogate key for the tool."),),
    ),
)

FOREIGN_ENTITIES = LEGAL_ENTITIES + (
    EntityType(
        label="SomebodyElsesLabel",
        description="A label this package describes but never declares anywhere.",
    ),
)
FOREIGN_RELATIONSHIPS = LEGAL_RELATIONSHIPS + (
    RelationshipType(
        relationship="STOLEN_EDGE",
        description="An edge described here although this package never declares it.",
    ),
)
SHARED_RELATIONSHIPS = (
    RelationshipType(
        relationship="USES",
        description="Operational use of a tool by a widget, described by the peer.",
        subject=("Widget",),
        object=("Tool",),
    ),
)
THIN_ENTITIES = (EntityType(label="Widget", description="A widget."),)
STALE_CONTRAST_ENTITIES = (
    EntityType(
        label="Widget",
        description=_WIDGET_DESCRIPTION,
        not_to_be_confused_with=("Vanished",),
    ),
)
BAD_ENDPOINT_RELATIONSHIPS = (
    RelationshipType(
        relationship="COMPOSES",
        description=_COMPOSES_DESCRIPTION,
        subject=("Widget",),
        object=("Nonexistent",),
    ),
)
PEER_ENDPOINT_RELATIONSHIPS = (
    RelationshipType(
        relationship="COMPOSES",
        description=_COMPOSES_DESCRIPTION,
        subject=("Widget",),
        object=("Tool",),
    ),
)
# One present endpoint, one absent. The edge is narrowed, not broken -- which is the
# normal state of a shared edge on any product that omits a package.
PARTIAL_ENDPOINT_RELATIONSHIPS = (
    RelationshipType(
        relationship="COMPOSES",
        description=_COMPOSES_DESCRIPTION,
        subject=("Widget",),
        object=("Tool", "Nonexistent"),
    ),
)
SILENTLY_OPEN_RELATIONSHIPS = (
    RelationshipType(
        relationship="COMPOSES",
        description=(
            "Structural containment between two things in the assembly, recorded "
            "whenever one is built out of another."
        ),
    ),
)
EXPLAINED_OPEN_RELATIONSHIPS = (
    RelationshipType(
        relationship="COMPOSES",
        description=(
            "Structural containment, deliberately open at both ends because any "
            "declared label may participate in an assembly."
        ),
    ),
)
BAD_ENUM_ENTITIES = (
    EntityType(
        label="Widget",
        description=_WIDGET_DESCRIPTION,
        properties=(
            PropertyType(
                "status",
                "Where the widget is in assembly, from intake to shipment.",
                enum=("new", "done", "new"),
            ),
            PropertyType(
                "phase",
                "Which assembly phase currently holds the widget.",
                enum=("cut", "  "),
            ),
        ),
    ),
)
BAD_VALUE_TYPE_ENTITIES = (
    EntityType(
        label="Widget",
        description=_WIDGET_DESCRIPTION,
        properties=(
            PropertyType(
                "count",
                "How many of this widget the assembly requires.",
                value_type="integer",
            ),
        ),
    ),
)
UNDECLARED_IDENTITY_ENTITIES = (
    EntityType(
        label="Widget",
        description=_WIDGET_DESCRIPTION,
        identity="serial",
        properties=(PropertyType("id", "Stable surrogate key assigned at intake."),),
    ),
)
DOUBLE_PROPERTY_ENTITIES = (
    EntityType(
        label="Widget",
        description=_WIDGET_DESCRIPTION,
        identity="name",
        properties=(
            "name",
            PropertyType("name", "Human-readable name the widget converges on."),
        ),
    ),
)
NAME_IDENTITY_ENTITIES = (
    EntityType(
        label="Widget",
        description=_WIDGET_DESCRIPTION,
        identity="name",
        properties=(PropertyType("name", "Human-readable name the widget converges on."),),
    ),
)
INTERNAL_ENTITIES = (
    EntityType(
        label="Widget",
        description=_WIDGET_DESCRIPTION,
        extractable=False,
        properties=(PropertyType("id", "Stable surrogate key assigned at intake."),),
    ),
)
REVISION_ENTITIES = LEGAL_ENTITIES + (
    EntityType(
        label="WidgetRevision",
        description="A snapshot of one widget at one version of its definition.",
    ),
)
LOWERCASE_RELATIONSHIPS = (
    RelationshipType(
        relationship="composes",
        description=_COMPOSES_DESCRIPTION,
        subject=("Widget",),
        object=("Widget",),
    ),
)

FOREIGN_OUTPUT_STEPS = (
    CascadeStep(
        key="widgets.steal", owner="widgets", trigger_labels=("Widget",),
        produces_labels=("Tool",), relationship="COMPOSES",
        question="What tool does this widget imply?",
    ),
)
INTERNAL_OUTPUT_STEPS = (
    CascadeStep(
        key="widgets.spine", owner="widgets", trigger_labels=("Widget",),
        produces_labels=("Widget",), relationship="COMPOSES",
        question="What widget does this widget imply?",
    ),
)
MISOWNED_STEPS = (
    CascadeStep(
        key="widgets.misowned", owner="tooling", trigger_labels=("Widget",),
        produces_labels=("Widget",), relationship="COMPOSES",
        question="What widget does this widget imply?",
    ),
)
UNDESCRIBED_EDGE_STEPS = (
    CascadeStep(
        key="widgets.ghost", owner="widgets", trigger_labels=("Widget",),
        produces_labels=("Widget",), relationship="VANISHED",
        question="What widget does this widget imply?",
    ),
)

FOREIGN_WRITE_COMMANDS = (_FakeWriteCommand("tooling.do_something", "tooling"),)
UNNAMESPACED_WRITE_COMMANDS = (_FakeWriteCommand("rename", "widgets"),)
FOREIGN_TOOLS = (_FakeTool("tooling.frobnicate", "tooling"),)
PEER_TOOLS = (_FakeTool("widgets.shared", "tooling"),)
COLLIDING_TOOLS = (_FakeTool("widgets.shared", "widgets"),)
CONTROL_PLANE_TOOLS = (_FakeTool("orchestration.hybrid_search", "orchestration"),)
DANGLING_EXPERTS = (_FakeExpert("widgets", ("widgets.vanished",)),)
CONTROL_PLANE_EXPERTS = (_FakeExpert("widgets", ("orchestration.hybrid_search",)),)

FOREIGN_KINDS = (VersionedKind(kind="Tool", owner="widgets"),)
BAD_REQUIREMENT_KINDS = (
    VersionedKind(
        kind="Widget", owner="widgets",
        requirements=(
            KindRequirement(
                direction="outbound", relationship="NONEXISTENT",
                endpoint_kinds=("Widget",),
            ),
        ),
        message="A Widget must be linked to the widget it refines.",
    ),
)
WIDGET_KINDS = (VersionedKind(kind="Widget", owner="widgets"),)
FOREIGN_MIGRATIONS = (_FakeMigration(1, "core"),)



