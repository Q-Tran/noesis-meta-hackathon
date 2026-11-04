"""The conformance rules themselves, one function per rule.

Every rule here was HARVESTED, not invented: each carries the exact place this
repository already enforces it -- a `raise` in `knowledge_bases/registry.py`, a
`__post_init__` in `knowledge_bases/base.py`, an assertion in `tests/`, or a
defect recorded in `docs/operations/split_state.md`. The provenance is in each
docstring because a package author in another repository will want to know
whether a rule is a real invariant or somebody's preference, and because a rule
whose origin nobody can name is a rule nobody can safely change.

The repository's own tests check most of these at WHOLE-PRODUCT composition
time, over all installed packages at once. That is exactly what a third-party
package cannot do: it is not installed anywhere yet, and composing it needs a
product. So each rule here is re-expressed PER PACKAGE, using only the package's
own declarations plus whatever peers the caller supplied.

A rule is a function `(PackageContext) -> Iterable[Finding]`. It never raises
for a violation; raising is the caller's choice (`check_package(strict=True)`).
"""
from __future__ import annotations

from typing import Callable, Iterable

from knowledge_bases.base import PropertyType, property_names
from knowledge_bases.conformance.context import (
    LEGAL_VALUE_TYPES,
    PackageContext,
    declared_property_names,
    described_property_types,
)
from knowledge_bases.conformance.findings import Finding, Severity

Rule = Callable[[PackageContext], Iterable[Finding]]

# Minimum words in a type description. From
# `tests/test_ontology_descriptions.py::test_descriptions_are_substantive`:
# "A one-word description does not disambiguate anything."
MIN_DESCRIPTION_WORDS = 5


def _finding(
    context: PackageContext,
    rule: str,
    severity: Severity,
    item: str,
    detail: str,
    remedy: str,
) -> Finding:
    return Finding(
        package=context.key,
        rule=rule,
        severity=severity,
        item=item,
        detail=detail,
        remedy=remedy,
    )


# -- 1. the declaration itself -------------------------------------------------


def package_declares_its_own_knowledge_base(context: PackageContext):
    """A package's key must name a knowledge base it declares.

    Source: `registry.compose()` keys the live registry by `kb.key`, while every
    ownership rule in the engine compares against `package.key` --
    `registry.cascade_steps()` builds `owned[package.key]` from
    `package.knowledge_bases`, and `composed_write_commands()` compares
    `command.knowledge_base != package.key`. If the package key matches no
    knowledge base it declares, those two views of "who owns this" disagree and
    every ownership check silently passes on an empty set.
    """
    rule = "package_declares_its_own_knowledge_base"
    if not context.package.knowledge_bases:
        yield _finding(
            context, rule, Severity.ERROR, context.key,
            "the package declares no KnowledgeBase at all, so it contributes no "
            "labels or relationships and composes an empty ontology",
            "add at least one `KnowledgeBase(key=..., ...)` to "
            "`OntologyPackage.knowledge_bases`",
        )
        return
    keys = [kb.key for kb in context.package.knowledge_bases]
    if context.key not in keys:
        yield _finding(
            context, rule, Severity.ERROR, context.key,
            f"the package key {context.key!r} names none of the knowledge bases "
            f"it declares ({keys}); ownership checks in the engine compare "
            "against the PACKAGE key, so they would all pass vacuously",
            f"rename one KnowledgeBase's key to {context.key!r}, or rename the "
            "package key to match the knowledge base it owns",
        )


def declared_types_are_well_formed_names(context: PackageContext):
    """Labels are `PascalCase`, relationships are `SCREAMING_SNAKE_CASE`.

    Source: every one of the 61 labels and 85 relationships this repository
    declares already obeys it (verified), and the convention is load-bearing
    rather than cosmetic -- `knowledge_bases/ontology.py` treats any label
    ending in `Revision` as a derived managed-entity label, and
    `triplets.render_ontology_prompt` renders `-[:NAME]->` into the extraction
    prompt where a lowercase name reads as prose. Interpolated straight into
    Cypher by `graph.upsert_rel`, so a name with a backtick or a space is also a
    correctness problem, not a style one.
    """
    rule = "declared_types_are_well_formed_names"
    kind_names = {kind.kind for kind in context.kinds}
    for label in sorted(context.declared_labels):
        if not label[:1].isupper() or not label.replace("_", "").isalnum():
            yield _finding(
                context, rule, Severity.ERROR, label,
                "labels are written PascalCase and interpolated into Cypher, so "
                f"{label!r} either will not match what writers create or cannot "
                "be quoted safely",
                f"rename the label to PascalCase (e.g. {label.title().replace('_', '')!r})",
            )
        # `<Kind>Revision` is also the DERIVED label the methodology versioning
        # engine generates per managed kind (`ontology.is_known_label`), so a
        # declared one collides with generated nodes -- but ONLY if this package
        # also makes `<Kind>` a versionable kind. `ExpertRevision` is a real
        # declared label of its own (`orchestration`), which is why this checks
        # the collision rather than the suffix.
        if label.endswith("Revision") and label != "Revision":
            stem = label[: -len("Revision")]
            if stem and stem in kind_names:
                yield _finding(
                    context, rule, Severity.ERROR, label,
                    f"{label!r} is declared as a label while {stem!r} is "
                    "contributed as a versionable kind, and the methodology "
                    f"engine GENERATES `:{label}` for every revision of a "
                    f"{stem!r} -- the declared label and the generated one would "
                    "be the same nodes",
                    f"drop the {label!r} label (revisions are created for you), "
                    f"or stop contributing {stem!r} through `kinds_seed`",
                )
    for relationship in sorted(context.declared_relationships):
        if relationship != relationship.upper() or not relationship.replace(
            "_", ""
        ).isalnum():
            yield _finding(
                context, rule, Severity.ERROR, relationship,
                "relationship types are SCREAMING_SNAKE_CASE and interpolated "
                f"into Cypher, so {relationship!r} will not match writes made "
                "through `graph.upsert_rel`",
                f"rename it to {relationship.upper().replace(' ', '_')!r}",
            )


def every_seed_reference_resolves(context: PackageContext):
    """Every `"module:ATTR"` seed a package declares must import and resolve.

    Source: `registry.resolve_seed` does a bare `getattr(import_module(mod),
    attr)`, and the seeds are resolved LAZILY -- at `bootstrap_catalog()`,
    `entity_types()`, `versioned_kinds()` time, not at declaration time. So a
    typo in a seed reference is invisible until the first caller touches that
    surface, which in a product is startup or the first search. The lazy
    resolution is deliberate (`base.py`: "declaring a package costs only this
    module's import"); this rule is what pays for it.
    """
    rule = "every_seed_reference_resolves"
    seed_fields = (
        "tools_seed", "experts_seed", "kinds_seed", "write_commands_seed",
        "entity_types_seed", "relationship_types_seed", "cascade_steps_seed",
        "migration_stream",
    )
    # Touch every slice so `context.seed_errors` is populated by the same
    # resolution path the rest of the rules use.
    for field_name in seed_fields:
        context._seeded(field_name)  # noqa: SLF001 - same module family
    for field_name in seed_fields:
        error = context.seed_errors.get(field_name)
        if error is None:
            continue
        reference = getattr(context.package, field_name)
        yield _finding(
            context, rule, Severity.ERROR, f"{field_name}={reference!r}",
            f"the seed does not resolve: {error}. Seeds resolve lazily, so this "
            "surfaces at first use (startup, search, or a changeset) rather "
            "than at import",
            "fix the 'module.path:ATTR' string, or export that attribute from "
            "the named module; the attribute must be an iterable of the "
            "contribution type",
        )


def declared_migrations_are_in_this_packages_stream(context: PackageContext):
    """A package's migrations all carry `stream == package.key`.

    Source: `migrations.composed_migrations()` raises `ValueError` -- "a package
    owns exactly its own stream". Streams version independently so packages can
    ship without renumbering each other; a migration in someone else's stream
    would be applied under their version counter and skipped or re-run
    depending on what they had already applied.
    """
    rule = "declared_migrations_are_in_this_packages_stream"
    for migration in context.migrations:
        stream = getattr(migration, "stream", None)
        if stream != context.key:
            yield _finding(
                context, rule, Severity.ERROR,
                f"migration {getattr(migration, 'version', '?')} "
                f"(stream={stream!r})",
                "a migration declared by this package belongs to another "
                "package's stream; it would be counted against that package's "
                "applied version and silently skipped or re-run",
                f"set `stream={context.key!r}` on every Migration in "
                f"{context.package.migration_stream!r}",
            )


# -- 2. describedness ----------------------------------------------------------


def every_declared_label_is_described_by_this_package(context: PackageContext):
    """A label this package declares must be described by THIS package.

    Source: `tests/test_ontology_descriptions.py
    ::test_every_declared_label_is_described`, and `registry.entity_types()`'s
    docstring -- "a label with no description here cannot be targeted
    unambiguously". The reason it must be THIS package and not any package is
    single ownership: `registry.entity_types()` raises on a label described
    twice, so a description can live in exactly one package, and if that is not
    the declaring one then uninstalling an unrelated package makes this
    package's label unextractable.
    """
    rule = "every_declared_label_is_described_by_this_package"
    described = set(context.described_labels)
    for label in sorted(context.declared_labels - described):
        elsewhere = context.peer_described_labels.get(label, [])
        if elsewhere:
            yield _finding(
                context, rule, Severity.ERROR, label,
                f"this package declares {label!r} but {elsewhere} describes it. "
                "Descriptions are single-owner (`registry.entity_types()` raises "
                "on a duplicate), so a product that installs this package "
                f"without {elsewhere[0]!r} has a label no extractor can target",
                f"move the `EntityType({label!r}, ...)` into this package's "
                "`entity_types_seed`",
            )
        else:
            yield _finding(
                context, rule, Severity.ERROR, label,
                f"{label!r} is declared but has no `EntityType` description "
                "anywhere, so ingestion cannot extract it: a model choosing "
                "among bare names has nothing to choose on",
                f"add `EntityType(label={label!r}, description=..., "
                "identity=...)` to this package's `entity_types_seed`, stating "
                "what the label is AND what it is not",
            )


def every_declared_relationship_is_described_by_this_package(context: PackageContext):
    """A relationship this package declares must be described by THIS package.

    Source: the real, documented defect. `docs/operations/split_state.md`
    measured `DEPENDS_ON` "declared relationships NOT described: 1 --
    `DEPENDS_ON`" in a profile that drops `semantic_layer`, because
    `semantic_layer` is the only package that describes an edge that
    `architecture` and `project_management` also declare.
    `tests/test_ontology_descriptions.py
    ::test_a_shared_edge_is_described_by_a_package_that_declares_it` polices one
    direction of it; this is the other direction, per package.

    Severity is WARNING, not ERROR, when a peer that DECLARES the edge describes
    it: sharing an edge with single ownership is legal and deliberate, and
    requiring every declarer to describe it would ban sharing outright. What it
    costs is composability, so it is reported rather than accepted silently.
    """
    rule = "every_declared_relationship_is_described_by_this_package"
    described = set(context.described_relationships)
    for relationship in sorted(context.declared_relationships - described):
        describers = context.peer_described_relationships.get(relationship, [])
        if not describers and not context.has_peers:
            yield _finding(
                context, rule, Severity.ERROR, relationship,
                f"{relationship!r} is declared with no `RelationshipType` "
                "description in this package, and no peer packages were "
                "supplied to check whether anything else describes it",
                f"add `RelationshipType(relationship={relationship!r}, ...)` to "
                "this package's `relationship_types_seed`, or pass the peer "
                "packages your product installs so a shared edge can be "
                "resolved",
            )
        elif not describers:
            yield _finding(
                context, rule, Severity.ERROR, relationship,
                f"{relationship!r} is declared but described by no installed "
                "package, so an extractor is offered a bare edge name and the "
                "graph ontology cannot explain what it means",
                f"add `RelationshipType(relationship={relationship!r}, "
                "description=..., subject=..., object=...)` to this package's "
                "`relationship_types_seed`",
            )
        else:
            yield _finding(
                context, rule, Severity.WARNING, relationship,
                f"{relationship!r} is declared here but described by "
                f"{describers}. That is legal single ownership of a shared edge, "
                f"and it is exactly the recorded `DEPENDS_ON` defect: a product "
                f"that installs this package without {describers[0]!r} leaves "
                "the edge undescribed",
                f"either describe {relationship!r} here and remove it from "
                f"{describers[0]!r}, or record that this package depends on "
                f"{describers[0]!r} being installed",
            )


def no_description_targets_a_type_this_package_does_not_declare(
    context: PackageContext,
):
    """A package never describes a type it does not declare.

    Source: `tests/test_ontology_descriptions.py
    ::test_no_description_targets_an_undeclared_type` ("a description for a
    label no package declares is a type the graph would reject at write time")
    and `::test_a_shared_edge_is_described_by_a_package_that_declares_it`
    ("Describing someone else's edge is how an edge ends up owned by a package
    that a narrower product does not install, which is the `DEPENDS_ON`
    failure"). Per package this is the stronger statement: describing a type you
    do not declare steals ownership of it, and since descriptions are
    single-owner, the real owner then CANNOT describe it.
    """
    rule = "no_description_targets_a_type_this_package_does_not_declare"
    for label in sorted(set(context.described_labels) - context.declared_labels):
        owner = context.peer_declared_labels.get(label, [])
        detail = (
            f"this package describes {label!r} but does not declare it"
            + (f"; {owner} declares it" if owner else "; no package declares it")
        )
        yield _finding(
            context, rule, Severity.ERROR, label, detail,
            f"declare {label!r} on one of this package's KnowledgeBases, or "
            "remove the `EntityType` and let its declaring package describe it "
            "(descriptions are single-owner, so holding this one stops the "
            "owner from describing it)",
        )
    for relationship in sorted(
        set(context.described_relationships) - context.declared_relationships
    ):
        owner = context.peer_declared_relationships.get(relationship, [])
        detail = (
            f"this package describes {relationship!r} but does not declare it"
            + (f"; {owner} declares it" if owner else "; no package declares it")
        )
        yield _finding(
            context, rule, Severity.ERROR, relationship, detail,
            f"add {relationship!r} to this package's KnowledgeBase "
            "`relationships`, or remove the `RelationshipType` so the declaring "
            "package can own the description",
        )


def descriptions_are_substantive(context: PackageContext):
    """A description must say enough to choose the type by.

    Source: `tests/test_ontology_descriptions.py
    ::test_descriptions_are_substantive` (>= 5 words for labels and
    relationships) and `::test_a_property_reused_across_labels_with_different
    _meanings_is_described` (same bar for described properties). The threshold is
    crude on purpose: it catches the placeholder, which is the realistic failure,
    without pretending to judge prose.
    """
    rule = "descriptions_are_substantive"
    def _check(kind: str, item: str, description: str):
        if len(description.split()) < MIN_DESCRIPTION_WORDS:
            return _finding(
                context, rule, Severity.ERROR, item,
                f"the {kind} description is {len(description.split())} word(s) "
                f"({description!r}); an extractor deciding between near "
                "neighbours cannot use it",
                f"describe what {item!r} is AND what it is not, naming the types "
                f"it is confused with (at least {MIN_DESCRIPTION_WORDS} words)",
            )
        return None

    for kb in context.package.knowledge_bases:
        found = _check("knowledge base", kb.key, kb.description)
        if found:
            yield found
    for entity in context.entity_types:
        found = _check("label", entity.label, entity.description)
        if found:
            yield found
        for prop in described_property_types(entity.properties):
            found = _check(
                "property", f"{entity.label}.{prop.name}", prop.description
            )
            if found:
                yield found
    for rel in context.relationship_types:
        found = _check("relationship", rel.relationship, rel.description)
        if found:
            yield found
        for prop in described_property_types(rel.properties):
            found = _check(
                "property", f"{rel.relationship}.{prop.name}", prop.description
            )
            if found:
                yield found


def confusable_types_name_types_that_exist(context: PackageContext):
    """`not_to_be_confused_with` must name real types.

    Source: `tests/test_ontology_descriptions.py
    ::test_confusable_types_name_types_that_exist` -- the field is fed to the
    extraction prompt, "so a stale name there teaches the model about a type
    that is not there". Naming a type in ANOTHER package is fine and useful
    (`FOUND_IN` contrasts itself with seven edges from four packages); naming
    nothing at all is the defect.

    A WARNING, not an error, and the demotion is load-bearing. A contrast is written
    against the full vocabulary, so on any product that omits a package some contrasts
    necessarily name something absent -- `TaskList` contrasts itself with `Epic`, which
    is exactly the distinction a reader needs, and `project_management` is precisely
    what a personal product does not install. Raising an ERROR there would demand the
    declaration be weakened to suit the narrowest product, degrading every other one.

    The real harm -- a nonexistent type reaching the extraction prompt -- is fixed where
    it happens, in `triplets._present_contrasts`, which filters against what the product
    actually offers. What remains here is advisory: worth reading when a name is dead in
    EVERY product, which no automated check can tell from a single product's view.
    """
    rule = "confusable_types_name_types_that_exist"
    for entity in context.entity_types:
        for other in entity.not_to_be_confused_with:
            if context.known_label(other):
                continue
            yield _finding(
                context, rule, Severity.WARNING, f"{entity.label} -> {other}",
                f"{entity.label!r} contrasts itself with {other!r}, which no "
                "installed package declares"
                + ("" if context.has_peers else " and no peers were supplied"),
                f"drop {other!r} only if it names nothing in ANY product: a "
                "contrast against a package this product happens not to install "
                "is correct and is filtered when the prompt is rendered",
            )
    for rel in context.relationship_types:
        for other in rel.not_to_be_confused_with:
            if context.known_relationship(other):
                continue
            yield _finding(
                context, rule, Severity.WARNING, f"{rel.relationship} -> {other}",
                f"{rel.relationship!r} contrasts itself with {other!r}, which no "
                "installed package declares"
                + ("" if context.has_peers else " and no peers were supplied"),
                f"drop {other!r} only if it names nothing in ANY product",
            )


# -- 3. endpoints and properties -----------------------------------------------


def relationship_endpoints_reference_labels_that_exist(context: PackageContext):
    """Endpoint labels must exist; a genuinely open endpoint must say so.

    Source: `tests/test_ontology_descriptions.py
    ::test_relationship_endpoints_are_declared_labels` -- "Endpoint typing is
    what rejects `(Metric)-[:REPORTS_TO]->(Table)`. An endpoint naming an
    undeclared label could never match, silently making the relationship
    unusable." `base.RelationshipType` documents that empty subject/object means
    "any declared label", "kept for structural edges whose endpoints are
    genuinely open".

    An endpoint naming a PEER's label is normal and expected -- 9 of the 11
    packages here do it, because edges are how packages join. It is only a
    finding when no supplied package declares the label, and it degrades to INFO
    with no peers because a foreign package cannot be expected to vendor the
    labels it points at.
    """
    rule = "relationship_endpoints_reference_labels_that_exist"
    available = set(context.declared_labels) | set(context.peer_declared_labels)
    for rel in context.relationship_types:
        for role, endpoints in (("subject", rel.subject), ("object", rel.object)):
            if not endpoints:
                continue
            missing = [e for e in endpoints if e not in available]
            if not missing:
                continue
            # UNUSABLE vs merely narrowed, and the difference is the whole rule.
            # An endpoint list is satisfied if ANY of its labels is present: with
            # `subject=(Idea, Requirement)` and only `Idea` installed, the edge still
            # matches everything it can match, and the prompt renders just `Idea`.
            # Only when EVERY label for a role is absent can the edge never match.
            usable = len(missing) < len(endpoints)
            if usable:
                yield _finding(
                    context, rule, Severity.WARNING,
                    f"{rel.relationship}.{role}",
                    f"{rel.relationship!r} names {missing} as a legal {role}, which "
                    "no installed package declares; the edge still works through its "
                    f"remaining {role}s",
                    "nothing to do if those labels belong to a package this product "
                    "omits -- endpoints are declared against the whole vocabulary and "
                    "filtered when the prompt is rendered",
                )
                continue
            severity = Severity.ERROR if context.has_peers else Severity.INFO
            yield _finding(
                context, rule, severity,
                f"{rel.relationship}.{role}",
                f"{rel.relationship!r} declares {list(endpoints)} as its only legal "
                f"{role}s and no supplied package declares ANY of them, so the edge "
                "can never match and is silently unusable"
                + ("" if context.has_peers else " (no peers were supplied)"),
                f"declare one of them, correct the spelling, or leave `{role}=()` if "
                "the endpoint is genuinely open",
            )


def an_open_endpoint_is_deliberate_and_explained(context: PackageContext):
    """An endpoint left open must be open ON PURPOSE, and say why.

    Source: `base.RelationshipType` allows empty subject/object for edges "whose
    endpoints are genuinely open", and every one of the 7 open endpoints in this
    repository explains itself in its description (`DEPENDS_ON`: "Deliberately
    open at both ends"; `FOUND_IN`: "any entity in any other knowledge base may
    point outward"; `SUPERSEDES`: "endpoints are any versioned node of the same
    kind"). The risk the rule addresses is the other case: an author who left
    the fields empty because filling them in was work. An open endpoint disables
    endpoint validation for that edge entirely -- `ingestion` can then attach it
    to anything -- so a silent default and a deliberate choice must not look
    alike.

    WARNING, not ERROR: the kit cannot read intent, only whether intent was
    recorded. An open endpoint on a NON-extractable edge is not reported at all,
    since no extractor is ever offered it.
    """
    rule = "an_open_endpoint_is_deliberate_and_explained"
    markers = ("open", "any entity", "any declared", "any node", "any versioned",
               "either end", "both ends", "cross-cutting", "anything")
    for rel in context.relationship_types:
        if rel.subject and rel.object:
            continue
        if not rel.extractable:
            continue
        text = rel.description.lower()
        if any(marker in text for marker in markers):
            continue
        which = (
            "both endpoints" if not rel.subject and not rel.object
            else ("subject" if not rel.subject else "object")
        )
        yield _finding(
            context, rule, Severity.WARNING, rel.relationship,
            f"{which} of {rel.relationship!r} accept any label, and the "
            "description does not say that is deliberate. An open endpoint "
            "switches off endpoint validation for this edge, so an extractor may "
            "attach it to anything",
            f"either type the endpoint (`subject=(...)`, `object=(...)`), or say "
            "in the description that it is deliberately open and what it is open "
            "over -- the way `DEPENDS_ON` and `FOUND_IN` do",
        )


def declared_property_enums_are_usable(context: PackageContext):
    """An enum must be non-empty, unique, and on a `str`/`list` property.

    Source: `base.PropertyType.__post_init__` already raises when an enum is
    declared on a non-string `value_type`, so that half is enforced at import.
    What nothing checks is the degenerate enum: `enum=("",)`, a duplicated value,
    or a `value_type` the engine does not know how to write at all. The reason it
    matters is in `base.PropertyType`'s own docstring -- `status` is backed by
    five non-overlapping enums across 11 labels, and
    `tests/test_ontology_descriptions.py::test_a_described_enum_matches_the
    _values_the_code_actually_writes` calls a drifted enum "worse than no enum:
    it refuses values the repository legitimately writes".
    """
    rule = "declared_property_enums_are_usable"
    for owner, properties in _all_property_owners(context):
        for prop in described_property_types(properties):
            item = f"{owner}.{prop.name}"
            if prop.value_type not in LEGAL_VALUE_TYPES:
                yield _finding(
                    context, rule, Severity.ERROR, item,
                    f"`value_type={prop.value_type!r}` is not one the engine "
                    f"writes; legal values are {sorted(LEGAL_VALUE_TYPES)}",
                    "set `value_type` to one of str, int, float, bool, list, "
                    "datetime",
                )
            if not prop.enum:
                continue
            if len(set(prop.enum)) != len(prop.enum):
                duplicates = sorted(
                    {v for v in prop.enum if list(prop.enum).count(v) > 1}
                )
                yield _finding(
                    context, rule, Severity.ERROR, item,
                    f"the enum repeats {duplicates}, so the closed value set is "
                    "ambiguous about how many distinct states exist",
                    f"remove the duplicate value(s) {duplicates} from `enum`",
                )
            empty = [value for value in prop.enum if not str(value).strip()]
            if empty:
                yield _finding(
                    context, rule, Severity.ERROR, item,
                    "the enum contains an empty value, which an extractor cannot "
                    "distinguish from 'not populated'",
                    "drop the empty entry; if 'unset' is a real state, name it "
                    "(e.g. 'unknown') so it is queryable",
                )


def an_identity_property_is_declared_as_a_property(context: PackageContext):
    """Whatever decides identity must be a property the type actually declares.

    Source: `base.EntityType.identity` is "what makes append-only ingestion
    converge instead of creating a new node per phrasing", and
    `ingestion.py`/`tests/test_ingestion_service.py
    ::test_entities_converge_on_their_declared_identity_property` write the node
    keyed on it. But ingestion also DROPS any property not declared
    (`::test_undeclared_properties_never_reach_the_graph`), so an identity
    property missing from `properties` names a key that extraction is forbidden
    to populate: every mention becomes a new node with a null key.
    """
    rule = "an_identity_property_is_declared_as_a_property"
    for entity in context.entity_types:
        declared = declared_property_names(entity.properties)
        for identity in entity.identity_properties:
            if identity in declared:
                continue
            yield _finding(
                context, rule, Severity.ERROR, f"{entity.label}.{identity}",
                f"{entity.label!r} converges on {identity!r} but does not "
                f"declare it as a property, and ingestion drops undeclared "
                "properties -- so the identity key can never be populated and "
                "every mention creates a new node",
                f"add {identity!r} to {entity.label!r}'s `properties` (ideally as "
                "a described `PropertyType`), or point `identity` at a property "
                f"it does declare: {sorted(declared)[:6]}",
            )


def a_property_is_declared_once_per_type(context: PackageContext):
    """One property name per type, described or not.

    Source: `base.property_names` / `base.described_properties` -- the latter
    builds a DICT keyed by name, so a repeated property silently keeps only the
    last description while `property_names` still reports both. The two views
    then disagree about the type's shape, and which description an extractor
    sees depends on declaration order. Both forms are accepted so packages can
    migrate one property at a time (`base.EntityType.properties`), which is
    exactly how a name ends up declared twice: once bare, once described.
    """
    rule = "a_property_is_declared_once_per_type"
    for owner, properties in _all_property_owners(context):
        names = property_names(properties)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        for name in duplicates:
            forms = [
                "bare" if isinstance(prop, str) else "described"
                for prop in properties
                if (prop if isinstance(prop, str) else prop.name) == name
            ]
            yield _finding(
                context, rule, Severity.ERROR, f"{owner}.{name}",
                f"{name!r} is declared {len(forms)} times ({forms}); "
                "`described_properties()` keys by name, so only the last "
                "description survives while `property_names()` still lists both",
                f"declare {name!r} exactly once on {owner!r} -- keep the "
                "`PropertyType` form and delete the bare string",
            )


def _all_property_owners(context: PackageContext):
    for entity in context.entity_types:
        yield entity.label, entity.properties
    for rel in context.relationship_types:
        yield rel.relationship, rel.properties


# -- 4. extractability -----------------------------------------------------------


def an_extractable_relationship_has_an_extractable_endpoint(context: PackageContext):
    """An edge offered to an extractor must have an end it could actually create.

    Source: `base.EntityType.extractable` is False "for labels only repository
    code creates -- telemetry rows, routing traces, revision spines... nothing
    in a document should ever produce one", and
    `tests/test_ontology_descriptions.py::test_the_extraction_prompt_carries
    _every_extractable_type` puts every extractable type into the prompt. Put
    together: an extractable edge all of whose endpoints are non-extractable is
    offered to a model that is forbidden to create either end of it. It is the
    one combination the two flags cannot both be right about.

    WARNING rather than ERROR, because this repository has 9 real instances
    (`development`'s code-graph edges, `orchestration.DOCUMENTS`) where the edge
    is written by a repository indexer and only the label flags were set. The
    finding names the fix rather than the blame.
    """
    rule = "an_extractable_relationship_has_an_extractable_endpoint"
    described = context.described_labels
    for rel in context.relationship_types:
        if not rel.extractable:
            continue
        endpoints = [
            endpoint for endpoint in (*rel.subject, *rel.object)
            if endpoint in described
        ]
        if not endpoints:
            continue
        if any(described[endpoint].extractable for endpoint in endpoints):
            continue
        yield _finding(
            context, rule, Severity.WARNING, rel.relationship,
            f"{rel.relationship!r} is `extractable=True` so it is rendered into "
            "the extraction prompt, but every endpoint it declares "
            f"({sorted(set(endpoints))}) is `extractable=False` -- the model is "
            "offered an edge whose ends it may not create",
            f"set `extractable=False` on {rel.relationship!r} if repository code "
            "writes it, or make at least one endpoint label extractable",
        )


def cascade_steps_are_owned_and_produce_only_extractable_owned_labels(
    context: PackageContext,
):
    """A cascade step is owned by the package whose entities it produces.

    Source: three `raise ValueError`s in `registry.cascade_steps()` -- a step
    that "claims owner X but is declared by Y", one that "produces L, which X
    does not declare", and one that "produces L, which is written by repository
    code only" -- plus `tests/test_cascade.py
    ::test_every_step_is_owned_by_the_package_that_declares_its_output`
    ("Uninstalling a package must remove the reasoning that produces its
    entities, not leave a step whose output label no longer exists"). Registry
    raises these only when the whole product composes; this states them for one
    package, which is what a package author can actually run.
    """
    rule = "cascade_steps_are_owned_and_produce_only_extractable_owned_labels"
    described = context.described_labels
    seen: dict[str, int] = {}
    for step in context.cascade_steps:
        seen[step.key] = seen.get(step.key, 0) + 1
        if step.owner != context.key:
            yield _finding(
                context, rule, Severity.ERROR, step.key,
                f"the step claims `owner={step.owner!r}` but is declared by "
                f"{context.key!r}; `registry.cascade_steps()` raises on this",
                f"set `owner={context.key!r}`, or move the step into the "
                f"{step.owner!r} package if that is whose entities it produces",
            )
        if not step.key.startswith(f"{context.key}."):
            yield _finding(
                context, rule, Severity.WARNING, step.key,
                "step keys are composed into one flat namespace "
                "(`registry.cascade_steps()` raises on a duplicate key across "
                "packages) and are referenced by `depends_on` from other "
                "packages, so an unnamespaced key can collide",
                f"rename the step to {context.key}.{step.key!r}",
            )
        for label in step.produces_labels:
            if label not in context.declared_labels:
                owner = context.peer_declared_labels.get(label, [])
                yield _finding(
                    context, rule, Severity.ERROR, f"{step.key} -> {label}",
                    f"the step produces {label!r}, which this package does not "
                    "declare"
                    + (f" ({owner} does)" if owner else "")
                    + " -- it would fabricate another package's entities, and "
                    "uninstalling that package would leave this step writing a "
                    "label that no longer exists",
                    f"produce only labels this package declares, or move the "
                    "step into the package that owns "
                    f"{label!r}{' (' + owner[0] + ')' if owner else ''}",
                )
                continue
            entity = described.get(label)
            if entity is None:
                yield _finding(
                    context, rule, Severity.ERROR, f"{step.key} -> {label}",
                    f"the step produces {label!r}, which this package declares "
                    "but does not describe; the write could not be validated "
                    "against an ontology description",
                    f"describe {label!r} in `entity_types_seed`",
                )
            elif not entity.extractable:
                yield _finding(
                    context, rule, Severity.ERROR, f"{step.key} -> {label}",
                    f"the step produces {label!r}, which is `extractable=False` "
                    "-- a label only repository code may write. A step that "
                    "invents telemetry or a versioning spine is a bug, not a "
                    "feature",
                    f"produce an extractable label instead, or if {label!r} "
                    "really is extractable, correct its `EntityType`",
                )
        if not step.trigger_labels:
            yield _finding(
                context, rule, Severity.ERROR, step.key,
                "the step declares no `trigger_labels`, so `plan_cascade` can "
                "never schedule it: it is dead configuration",
                "name the labels whose appearance should run this step",
            )
        if not step.produces_labels:
            yield _finding(
                context, rule, Severity.ERROR, step.key,
                "the step produces nothing, so running it cannot have an effect",
                "name the labels the step may create in `produces_labels`",
            )
    for key, count in sorted(seen.items()):
        if count > 1:
            yield _finding(
                context, rule, Severity.ERROR, key,
                f"the step key is declared {count} times in this package's own "
                "seed; `registry.cascade_steps()` raises on a duplicate key",
                f"give each step a unique key (e.g. {key}_2)",
            )


def a_cascade_step_writes_a_relationship_that_is_described(context: PackageContext):
    """A step's relationship must be described, and ideally by this package.

    Source: `registry.cascade_steps()` raises "writes X, which no installed
    package describes", and `tests/test_cascade.py
    ::test_every_step_writes_a_described_relationship` ("a step naming an
    undeclared edge would fail at execution rather than at packaging time").
    The WARNING half is the composability consequence: 2 of the 4 real steps
    write an edge a DIFFERENT package describes (`claims` writes `FOUND_IN`,
    described by `provenance`), so those steps become unvalidatable in a product
    that drops the describer.
    """
    rule = "a_cascade_step_writes_a_relationship_that_is_described"
    for step in context.cascade_steps:
        if step.relationship in context.described_relationships:
            continue
        describers = context.peer_described_relationships.get(step.relationship, [])
        if describers:
            yield _finding(
                context, rule, Severity.WARNING,
                f"{step.key} -[:{step.relationship}]->",
                f"the step writes {step.relationship!r}, which {describers} "
                "describes rather than this package. The write is validated "
                "against the composed ontology, so a product without "
                f"{describers[0]!r} cannot run this step",
                f"record that this package needs {describers[0]!r} installed, or "
                "write an edge this package describes",
            )
        else:
            yield _finding(
                context, rule, Severity.ERROR,
                f"{step.key} -[:{step.relationship}]->",
                f"the step writes {step.relationship!r}, which no supplied "
                "package describes; the cascade write is validated like any "
                "other and would be refused at execution time",
                f"describe {step.relationship!r}, or point the step at an edge "
                "that is described",
            )


# -- 5. dispatch surfaces a package contributes ---------------------------------


def write_commands_are_namespaced_to_this_package(context: PackageContext):
    """A package owns exactly its own bounded writes.

    Source: `registry.composed_write_commands()` raises -- "a package owns only
    its own writes" -- and the verified defect behind it, recorded in
    `tests/test_write_command_ownership.py`'s module docstring:
    `add_business_term_alias` was hardcoded in `orchestration/commands.py`, "so
    a product profile that dropped `business_dictionary` still advertised the
    command -- an allowlisted mutation whose handler no longer shipped". The key
    prefix is checked too, because `WRITE_COMMANDS` is one flat allowlist keyed
    by `command.key` and an unnamespaced key can shadow another package's.
    """
    rule = "write_commands_are_namespaced_to_this_package"
    for command in context.write_commands:
        key = getattr(command, "key", "<unkeyed>")
        owner = getattr(command, "knowledge_base", None)
        if owner != context.key:
            yield _finding(
                context, rule, Severity.ERROR, key,
                f"the command declares `knowledge_base={owner!r}` but is "
                f"contributed by {context.key!r}; "
                "`registry.composed_write_commands()` raises on this, because "
                "uninstalling a package must remove its ability to mutate the "
                "graph",
                f"set `knowledge_base={context.key!r}` on the command, or move it "
                f"into the {owner!r} package",
            )
        if not str(key).startswith(f"{context.key}."):
            yield _finding(
                context, rule, Severity.ERROR, key,
                "write command keys share one flat allowlist, so an "
                f"unnamespaced key like {key!r} can shadow another package's "
                "command and silently swap which handler runs",
                f"rename the command key to {context.key}.<verb>",
            )


def contributed_tools_belong_to_this_package(context: PackageContext):
    """Every seeded `Tool` names this package and is namespaced to it.

    Source: `tools.Tool.knowledge_base` is what
    `tests/test_readme_architecture.py
    ::test_per_knowledge_base_tool_counts_in_the_diagram_are_right` counts by
    (`tool.key.split(".")[0]`), what the per-domain MCP servers filter by, and
    what `registry.composed_tools()` composes in package order. All 94 real
    tools obey both halves (verified). A tool that names another package is
    advertised under a knowledge base that may not be installed -- the same
    failure the write-command rule exists for, on the read surface.
    """
    rule = "contributed_tools_belong_to_this_package"
    peer_tools = context.peer_tool_keys
    seen: dict[str, int] = {}
    for tool in context.tools:
        key = getattr(tool, "key", None)
        owner = getattr(tool, "knowledge_base", None)
        if key is None:
            yield _finding(
                context, rule, Severity.ERROR, repr(tool)[:60],
                "a contributed tool has no `key`, so it cannot be dispatched or "
                "synced into the graph catalog",
                "give every Tool a `key` of the form <package>.<verb>",
            )
            continue
        seen[key] = seen.get(key, 0) + 1
        if owner != context.key:
            yield _finding(
                context, rule, Severity.ERROR, key,
                f"the tool declares `knowledge_base={owner!r}` while being "
                f"contributed by {context.key!r}; it would be advertised under a "
                "knowledge base this package does not own, and would survive "
                "that package being uninstalled",
                f"set `knowledge_base={context.key!r}`",
            )
        if not str(key).startswith(f"{context.key}."):
            yield _finding(
                context, rule, Severity.ERROR, key,
                "tool keys are one flat namespace and are split on '.' to "
                f"attribute a tool to its package, so {key!r} is attributed to "
                f"{str(key).split('.')[0]!r}",
                f"rename the tool key to {context.key}.<verb>",
            )
        if key in peer_tools:
            yield _finding(
                context, rule, Severity.ERROR, key,
                f"{peer_tools[key]!r} already contributes a tool with this key; "
                "the later one silently shadows the earlier in the composed "
                "catalog",
                "choose a key unique across installed packages",
            )
    for key, count in sorted(seen.items()):
        if count > 1:
            yield _finding(
                context, rule, Severity.ERROR, key,
                f"the tool key is contributed {count} times by this package",
                "contribute each tool key exactly once",
            )


def a_contributed_experts_entry_tools_are_reachable(context: PackageContext):
    """An expert's entry tools must be tools that are actually installed.

    Source: `registry.composed_experts()` composes expert definitions per
    package while `composed_tools()` composes the tools separately, and nothing
    ties the two together. `AGENTS.md`'s expert loop contract has the child
    "Start with {d['operation']}" -- the entry tool -- so an entry tool naming
    something uninstalled sends every delegation of that expert straight into a
    dead dispatch. Verified real case: `architecture`'s expert opens on
    `orchestration.hybrid_search`, a control-plane tool, which is legal (core is
    always installed) and reported only as INFO.
    """
    rule = "a_contributed_experts_entry_tools_are_reachable"
    own = {getattr(tool, "key", None) for tool in context.tools}
    peers = context.peer_tool_keys
    for expert in context.experts:
        entry_tools = tuple(getattr(expert, "entry_tools", ()) or ())
        expert_key = getattr(expert, "key", "<unkeyed>")
        for tool_key in entry_tools:
            if tool_key in own:
                continue
            owner = peers.get(tool_key)
            if owner is None:
                severity = Severity.ERROR if context.has_peers else Severity.INFO
                yield _finding(
                    context, rule, severity, f"{expert_key} -> {tool_key}",
                    f"expert {expert_key!r} opens on {tool_key!r}, which no "
                    "supplied package contributes, so every delegation to this "
                    "expert starts at a tool that cannot be dispatched"
                    + ("" if context.has_peers else " (no peers were supplied)"),
                    f"point `entry_tools` at a tool this package contributes, or "
                    "declare the dependency on whichever package owns "
                    f"{tool_key!r}",
                )
            elif owner in context.control_plane:
                yield _finding(
                    context, rule, Severity.INFO, f"{expert_key} -> {tool_key}",
                    f"expert {expert_key!r} opens on {tool_key!r} from the "
                    f"control-plane package {owner!r}; that ships with the "
                    "engine, so it is always installed",
                    "no action needed; recorded so the dependency is visible",
                )
            else:
                yield _finding(
                    context, rule, Severity.WARNING,
                    f"{expert_key} -> {tool_key}",
                    f"expert {expert_key!r} opens on {tool_key!r}, a tool owned "
                    f"by the {owner!r} ontology package; a product that installs "
                    f"this package without {owner!r} has an expert whose first "
                    "move is undispatchable",
                    f"open on a tool this package owns, or record that it "
                    f"requires {owner!r}",
                )


def versioned_kinds_are_owned_and_declared(context: PackageContext):
    """A contributed kind names this package as owner and a label it declares.

    Source: `methodology/kinds.py` raises when a kind's owner "does not own that
    label" (pinned by `tests/test_knowledge_bases.py
    ::test_a_kind_may_not_claim_an_owner_that_does_not_declare_its_label`:
    "Ownership is verified at composition, not trusted from the declaration"),
    and `base.VersionedKind` states it directly -- "Declared by the ontology
    package that owns the label (`owner`); the methodology package itself owns
    none -- it is the process, not the vocabulary."
    `docs/operations/split_state.md` calls this composed registry the hardest
    coupling for the ontologies repo precisely because it is a whole-set
    invariant, so checking the package-local half here is what lets a package be
    released on its own.
    """
    rule = "versioned_kinds_are_owned_and_declared"
    seen: dict[str, int] = {}
    for kind in context.kinds:
        seen[kind.kind] = seen.get(kind.kind, 0) + 1
        if kind.owner != context.key:
            yield _finding(
                context, rule, Severity.ERROR, kind.kind,
                f"the kind claims `owner={kind.owner!r}` but is contributed by "
                f"{context.key!r}; the composed kind registry raises on this, so "
                "the whole product fails to validate any changeset",
                f"set `owner={context.key!r}`, or contribute the kind from the "
                f"{kind.owner!r} package",
            )
        if kind.kind not in context.declared_labels:
            owner = context.peer_declared_labels.get(kind.kind, [])
            yield _finding(
                context, rule, Severity.ERROR, kind.kind,
                f"the kind names label {kind.kind!r}, which this package does "
                "not declare"
                + (f" ({owner} does)" if owner else "")
                + "; a versionable kind must be a label its own package owns, or "
                "'who owns a Requirement' has more than one answer",
                f"declare {kind.kind!r} on this package's KnowledgeBase, or move "
                "the kind contribution to the package that declares it",
            )
        for requirement in kind.requirements:
            if not context.known_relationship(requirement.relationship):
                severity = Severity.ERROR if context.has_peers else Severity.INFO
                yield _finding(
                    context, rule, severity,
                    f"{kind.kind} requires -[:{requirement.relationship}]->",
                    f"the creation rule for {kind.kind!r} requires a "
                    f"{requirement.relationship!r} edge, which no supplied "
                    "package declares, so the rule can never be satisfied and "
                    "the kind can never be created through a changeset"
                    + ("" if context.has_peers else " (no peers were supplied)"),
                    f"declare {requirement.relationship!r}, or point the "
                    "requirement at a declared edge",
                )
            for endpoint in requirement.endpoint_kinds:
                if context.known_label(endpoint):
                    continue
                severity = Severity.ERROR if context.has_peers else Severity.INFO
                yield _finding(
                    context, rule, severity, f"{kind.kind} -> {endpoint}",
                    f"the creation rule for {kind.kind!r} accepts endpoint kind "
                    f"{endpoint!r}, which no supplied package declares"
                    + ("" if context.has_peers else " (no peers were supplied)"),
                    f"correct {endpoint!r} to a declared label",
                )
        if kind.requirements and not kind.message:
            yield _finding(
                context, rule, Severity.WARNING, kind.kind,
                "the kind has creation requirements but no `message`, so a "
                "changeset that fails them is refused without saying what would "
                "satisfy them",
                "set `message=` to the sentence a user should read when the "
                "requirement is unmet",
            )
    for kind, count in sorted(seen.items()):
        if count > 1:
            yield _finding(
                context, rule, Severity.ERROR, kind,
                f"the kind is contributed {count} times by this package; the "
                "composed registry raises on a double declaration",
                "contribute each kind exactly once",
            )


def changeset_relationships_are_declared_by_this_package(context: PackageContext):
    """Every changeset relationship must also be declared as graph ontology.

    Source: `base.OntologyPackage.changeset_relationships` -- "each must also be
    declared on a `KnowledgeBase` above, since the graph ontology is enforced
    independently at write time" -- and `tests/test_knowledge_bases.py
    ::test_every_changeset_relationship_is_declared_somewhere` ("otherwise the
    write fails at `graph.upsert_rel` after validation passed"). That is the
    nastiest shape of failure available here: the changeset validates, the user
    is told it was accepted, and the write is then refused by the ontology guard.
    """
    rule = "changeset_relationships_are_declared_by_this_package"
    for relationship in context.package.changeset_relationships:
        if relationship in context.declared_relationships:
            continue
        owner = context.peer_declared_relationships.get(relationship, [])
        if owner:
            yield _finding(
                context, rule, Severity.WARNING, relationship,
                f"this package lets a changeset write {relationship!r} but does "
                f"not declare it ({owner} does); the changeset would validate "
                f"and then be refused by `ontology.require_known_relationship` "
                f"in any product without {owner[0]!r}",
                f"add {relationship!r} to this package's KnowledgeBase "
                f"`relationships`, or record the dependency on {owner[0]!r}",
            )
        else:
            yield _finding(
                context, rule, Severity.ERROR, relationship,
                f"this package lets a changeset write {relationship!r}, which no "
                "supplied package declares as graph ontology; the changeset "
                "validates and the write is then refused at `graph.upsert_rel`",
                f"add {relationship!r} to this package's KnowledgeBase "
                "`relationships`",
            )


def search_types_target_this_packages_own_labels(context: PackageContext):
    """A search type must scope to this package and target a label it declares.

    Source: `registry.search_types()` keys the registry by
    `f"{family}.{label}"`, and `base.SearchType` says "`family` scopes a
    hybrid_search call: every type in one call must share it". So the family is
    the caller-visible scope name, and a package whose family is not its own key
    puts its labels inside another package's scope -- where they disappear if
    that package is uninstalled, and collide if both declare the same label.
    """
    rule = "search_types_target_this_packages_own_labels"
    seen: dict[str, int] = {}
    for search_type in context.package.search_types:
        qualified = f"{search_type.family}.{search_type.label}"
        seen[qualified] = seen.get(qualified, 0) + 1
        if search_type.family != context.key:
            yield _finding(
                context, rule, Severity.ERROR, qualified,
                f"the search type scopes itself to family "
                f"{search_type.family!r}, not to this package {context.key!r}; "
                "the family is the caller-visible search scope, so this label "
                "would only be searchable under another package's name",
                f"set `family={context.key!r}`",
            )
        if search_type.label not in context.declared_labels:
            owner = context.peer_declared_labels.get(search_type.label, [])
            yield _finding(
                context, rule, Severity.ERROR, qualified,
                f"the search type targets {search_type.label!r}, which this "
                "package does not declare"
                + (f" ({owner} does)" if owner else ""),
                f"declare {search_type.label!r} here, or remove the SearchType",
            )
        if (
            not search_type.fulltext_index
            and not search_type.vector_collection
            and not search_type.include_managed
        ):
            yield _finding(
                context, rule, Severity.ERROR, qualified,
                "the search type has no fulltext index, no vector collection, "
                "and is not a managed kind, so `hybrid_search` has nothing to "
                "retrieve it with: it is registered but unsearchable",
                "give it a `fulltext_index` or a `vector_collection`, or set "
                "`include_managed=True` if its content lives on a methodology "
                "Revision",
            )
    for qualified, count in sorted(seen.items()):
        if count > 1:
            yield _finding(
                context, rule, Severity.ERROR, qualified,
                f"the (family, label) pair is declared {count} times; the "
                "composed registry is a dict keyed on it, so only one survives",
                "declare each (family, label) pair once",
            )


def a_search_type_reads_the_labels_identity_property(context: PackageContext):
    """A search type must read identity from the property the label converges on.

    Source: `base.EntityType.identity` is "what makes append-only ingestion
    converge", and `hybrid_search` reads
    `row["properties"].get(config.identity_property)` and then DROPS any
    candidate whose value is `None`, as well as matching on
    `` WHERE n.`{identity_property}` = selected.identity ``. So when the two
    disagree, the label is indexed and searchable and returns nothing --
    the quietest failure in the kit, because there is no error anywhere.

    This one is not harvested from an existing test; it is the rule that catches
    a live defect in this repository (see the kit's README).
    """
    rule = "a_search_type_reads_the_labels_identity_property"
    described = context.described_labels
    for search_type in context.package.search_types:
        entity = described.get(search_type.label)
        if entity is None:
            continue
        identity = entity.identity_properties
        if search_type.identity_property in identity:
            continue
        yield _finding(
            context, rule, Severity.ERROR,
            f"{search_type.family}.{search_type.label}",
            f"the search type reads identity from "
            f"{search_type.identity_property!r} while "
            f"`EntityType({search_type.label!r})` converges on {identity}. "
            "`hybrid_search` drops every candidate whose identity property is "
            "null, so this label is indexed, searchable, and returns nothing",
            f"set `identity_property={identity[0]!r}` on the SearchType, or "
            f"change {search_type.label!r}'s `identity` if "
            f"{search_type.identity_property!r} is really what writers key on",
        )


# -- 6. independence -------------------------------------------------------------


def the_package_does_not_import_a_sibling_ontology_package(context: PackageContext):
    """A package may depend on the engine, never on another ontology.

    Source: `tests/test_package_import_independence.py` -- "A profile that omits
    a package must survive that package being *absent*... pip-not-installed is
    what a real product distribution looks like" -- and
    `docs/operations/split_state.md`, which measured exactly one cross-package
    domain import in this tree (`business_dictionary/queries.py:5` importing
    `semantic_layer.resolution`) and called the ontologies monorepo "genuinely
    almost clean" because of it.

    Importing the CONTROL PLANE (`tools`, `orchestration`, `provenance`) is
    legal and universal: those ship with the engine, all 8 domain packages
    import them, and `registry.py` names them "cross-cutting infrastructure".
    Only sibling ONTOLOGY imports are findings. A control-plane package
    importing a domain package is the reverse direction and the worse one, since
    it makes the engine un-extractable, so it is reported as an ERROR.
    """
    rule = "the_package_does_not_import_a_sibling_ontology_package"
    imports = context.imported_packages()
    if not imports and context.source_root is None:
        yield _finding(
            context, rule, Severity.INFO, context.key,
            "no source directory was found for this package, so its imports "
            "could not be checked",
            "pass `source_root=` pointing at the package's source tree to enable "
            "this rule",
        )
        return
    is_control_plane = context.key in context.control_plane
    for other, sites in sorted(imports.items()):
        # The contract module every package declares against is engine code.
        if other in {"base", "conformance"}:
            continue
        other_is_control_plane = other in context.control_plane
        if is_control_plane and not other_is_control_plane:
            yield _finding(
                context, rule, Severity.ERROR, f"{context.key} -> {other}",
                f"this is a control-plane package and it imports the "
                f"{other!r} ontology at {sites[:3]}"
                + (f" (+{len(sites) - 3} more)" if len(sites) > 3 else "")
                + "; the engine must own zero ontologies, so every one of these "
                "blocks extracting it",
                f"invert the dependency: have {other!r} implement an interface "
                "the engine declares, rather than the engine importing the "
                "ontology",
            )
        elif other_is_control_plane:
            continue
        else:
            yield _finding(
                context, rule, Severity.ERROR, f"{context.key} -> {other}",
                f"this ontology package imports the sibling ontology {other!r} at "
                f"{sites[:3]}"
                + (f" (+{len(sites) - 3} more)" if len(sites) > 3 else "")
                + f"; a product that installs this package without {other!r} "
                "dies at import, which is what pip-not-installed looks like",
                "copy the small helper you need into this package, or move it "
                f"into the engine, so {other!r} stops being a hard dependency",
            )


def edge_contracts_describe_declared_relationships(context: PackageContext):
    """A package's edge contracts must describe edges IT declares, with
    endpoints and writers that exist among the supplied packages.

    Source: `base.EdgeContract` and `registry.compose` -- one describer per
    relationship, the describer must be a declarer, and endpoints no supplied
    package supplies make the contract inert. The kit reports what composition
    will see: naming a relationship this package does not declare is an ERROR
    (the describer rule raises), endpoints missing from the supplied set are a
    WARNING (inert in this product, legal in a wider one -- the personal
    product's `MEASURED_BY` situation), writers missing are INFO.
    """
    rule = "edge_contracts_describe_declared_relationships"
    if not context.package.edge_contracts:
        return
    all_labels = set(context.declared_labels) | set(context.peer_declared_labels)
    peer_keys = {p.key for p in context.peers} | {context.key}
    for contract in context.package.edge_contracts:
        if contract.relationship not in context.declared_relationships:
            yield _finding(
                context, rule, Severity.ERROR, contract.relationship,
                f"this package declares an edge contract for "
                f"{contract.relationship!r}, which it does not declare on its "
                "own knowledge base; the describer must be a declarer, and "
                "composition raises on this",
                f"add {contract.relationship!r} to this package's KnowledgeBase "
                "`relationships`, or move the contract to the package that "
                "declares it",
            )
        if not context.peers:
            continue
        endpoints = (set(contract.from_labels) | set(contract.to_labels)) - {"*"}
        missing = endpoints - all_labels
        if missing:
            yield _finding(
                context, rule, Severity.WARNING, contract.relationship,
                f"the contract's endpoints {sorted(missing)} are declared by no "
                "supplied package, so the contract is inert in this product -- "
                "the edge is unwritable here",
                "install the package that supplies the endpoint, or accept the "
                "dead edge (the personal product does, for `MEASURED_BY`)",
            )
        missing_writers = set(contract.writers) - {"*"} - peer_keys
        if missing_writers:
            yield _finding(
                context, rule, Severity.INFO, contract.relationship,
                f"the contract lists writers {sorted(missing_writers)} that are "
                "not supplied; their code is absent, so the write cannot "
                "happen -- inert, not an error",
                "no action needed in this product; the writers exist in the "
                "wider product that ships them",
            )


def the_declaration_module_imports_only_the_contract(context: PackageContext):
    """`package.py` must import nothing but the contract module.

    Source: `registry._load_declaration` executes `package.py` STANDALONE, via
    `spec_from_file_location`, precisely so "the declaration must compose even
    while this module (or the knowledge base's own `__init__`) is only partially
    initialized" -- importing it as a real module "would run the knowledge
    base's `__init__.py` and pull half the codebase into the composition path".
    `base.py` is documented as a leaf that "imports nothing from the rest of
    Noesis" for the same reason. All 11 real `package.py` files import only
    `knowledge_bases.base` (verified). A declaration that imports anything else
    reintroduces the load-order deadlock those two designs exist to avoid.
    """
    rule = "the_declaration_module_imports_only_the_contract"
    if context.source_root is None:
        return
    declaration = context.source_root / "package.py"
    if not declaration.exists():
        return
    import ast as _ast

    try:
        tree = _ast.parse(declaration.read_text())
    except (OSError, SyntaxError) as exc:
        yield _finding(
            context, rule, Severity.ERROR, "package.py",
            f"the declaration module could not be parsed: {exc}",
            "fix the syntax error; the file is executed standalone at "
            "composition time",
        )
        return
    for node in _ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, _ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, _ast.ImportFrom) and node.level == 0:
            modules = [node.module or ""]
        elif isinstance(node, _ast.ImportFrom):
            modules = [f".{node.module or ''}"]
        for module in modules:
            if module in {"knowledge_bases.base", "__future__"}:
                continue
            if module.endswith(".base"):
                continue
            yield _finding(
                context, rule, Severity.ERROR,
                f"package.py:{node.lineno} imports {module!r}",
                "the declaration is executed standalone at composition time, "
                "before any package's `__init__` has run, so importing anything "
                "beyond the contract module can deadlock or partially initialise "
                "the composition path",
                "import only `knowledge_bases.base` in `package.py`; put "
                "everything else behind a `\"module:ATTR\"` seed, which is "
                "resolved lazily",
            )


def the_package_composes_standalone(context: PackageContext):
    """The package must compose into a registry all by itself.

    Source: `registry.compose()` is the funnel every consumer reads and it
    raises on a duplicate knowledge-base key. `tests/test_product_profile.py
    ::test_a_narrower_product_is_declarable_without_touching_the_engine`
    composes a SUBSET to prove a narrower product needs no engine change; the
    narrowest interesting subset is one package, and that is the case a
    standalone ontology distribution actually ships as. This rule calls the real
    `compose()` when the engine is importable, so the package is validated by the
    engine's own code path rather than by a re-implementation that could drift.
    """
    rule = "the_package_composes_standalone"
    try:
        from knowledge_bases.registry import compose
    except Exception as exc:  # noqa: BLE001
        yield _finding(
            context, rule, Severity.INFO, context.key,
            f"the engine's `registry.compose` could not be imported ({exc}), so "
            "standalone composition was not exercised",
            "install the engine to enable this rule",
        )
        return
    try:
        composed = compose((context.package,))
    except Exception as exc:  # noqa: BLE001 - any raise is the finding
        yield _finding(
            context, rule, Severity.ERROR, context.key,
            f"composing this package on its own raised "
            f"{type(exc).__name__}: {exc}. A product that installs only this "
            "ontology therefore cannot start",
            "fix the declaration so `registry.compose((PACKAGE,))` succeeds; a "
            "duplicate knowledge-base key is the usual cause",
        )
        return
    if not composed:
        yield _finding(
            context, rule, Severity.ERROR, context.key,
            "the package composes an EMPTY registry, so installing it adds no "
            "ontology at all",
            "declare at least one KnowledgeBase with labels and relationships",
        )


# -- the suite ------------------------------------------------------------------

# Order is reporting order, grouped the way an author debugs: is the declaration
# well formed, is it described, are the endpoints real, is extraction sane, are
# the dispatch surfaces owned, does it stand alone.
ALL_RULES: tuple[Rule, ...] = (
    package_declares_its_own_knowledge_base,
    declared_types_are_well_formed_names,
    every_seed_reference_resolves,
    declared_migrations_are_in_this_packages_stream,
    every_declared_label_is_described_by_this_package,
    every_declared_relationship_is_described_by_this_package,
    no_description_targets_a_type_this_package_does_not_declare,
    descriptions_are_substantive,
    confusable_types_name_types_that_exist,
    relationship_endpoints_reference_labels_that_exist,
    an_open_endpoint_is_deliberate_and_explained,
    declared_property_enums_are_usable,
    an_identity_property_is_declared_as_a_property,
    a_property_is_declared_once_per_type,
    an_extractable_relationship_has_an_extractable_endpoint,
    cascade_steps_are_owned_and_produce_only_extractable_owned_labels,
    a_cascade_step_writes_a_relationship_that_is_described,
    write_commands_are_namespaced_to_this_package,
    contributed_tools_belong_to_this_package,
    a_contributed_experts_entry_tools_are_reachable,
    versioned_kinds_are_owned_and_declared,
    changeset_relationships_are_declared_by_this_package,
    search_types_target_this_packages_own_labels,
    a_search_type_reads_the_labels_identity_property,
    the_package_does_not_import_a_sibling_ontology_package,
    edge_contracts_describe_declared_relationships,
    the_declaration_module_imports_only_the_contract,
    the_package_composes_standalone,
)

RULES_BY_NAME: dict[str, Rule] = {rule.__name__: rule for rule in ALL_RULES}
