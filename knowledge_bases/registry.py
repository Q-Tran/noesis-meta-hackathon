"""Canonical ownership boundaries for Noesis graph knowledge.

`FOUND_IN` is a cross-cutting provenance relationship owned by the
`provenance` knowledge base: any entity in any other knowledge base may point
outward at a `SourceNode` via `(:Entity)-[:FOUND_IN {role}]->(:SourceNode)` to
record where it came from (see `graph.attach_source`/`graph.trace_sources` and
`docs/architecture/provenance.md`). Like `tools`, `provenance` is cross-cutting
infrastructure rather than a user-routable expert (see
`knowledge_bases/orchestration/experts_catalog.py`).

Composition: each knowledge base declares itself in
`knowledge_bases/<key>/package.py` as an `OntologyPackage`
(`knowledge_bases/base.py`). `DEFAULT_PACKAGES` lists the packages this
repository ships; a product is a subset (or superset, for separately
installed distributions) of that list -- `ontology_packages()`/`compose()`
are the single funnel every registry consumer reads, so installing or
removing a package changes the graph ontology, the search-type registry, the
seed catalogs, and the dispatch allowlists together. The package list itself
is code: composing it at runtime would be an architecture change without a
reviewed commit (`knowledge_bases/boundary.py`).
"""
from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache
from pathlib import Path

from typing import Any

import dataclasses

from knowledge_bases.base import (
    EntityType,
    ExtractionShape,
    ExtractionSource,
    KnowledgeBase,
    OntologyPackage,
    ProductProfile,
    PropertyContribution,
    PropertyType,
    RelationshipType,
    SearchType,
)
from knowledge_bases.product import PROFILE

__all__ = [
    "KNOWLEDGE_BASES", "KnowledgeBase", "OntologyPackage", "SearchType",
    "ProductProfile", "PROFILE", "product_profile",
    "knowledge_base", "ontology_packages", "compose", "resolve_seed",
    "search_types", "deprecatable_knowledge_bases",
    "extraction_shapes", "extraction_sources",
    "call_noesis_tool_knowledge_bases", "orchestrator_dispatch_knowledge_bases",
]


# The ontology packages this product ships, in registration order -- read from
# the product profile (`knowledge_bases/product.py`) rather than listed here,
# so which ontologies are installed is declared in exactly one place. Entries
# are knowledge-base keys; each resolves to `knowledge_bases/<key>/package.py`.
# Package order defines `KNOWLEDGE_BASES` iteration order; seed catalogs define
# their own composition order (see `tools/catalog.py`/
# `orchestration/experts_catalog.py`).
DEFAULT_PACKAGES: tuple[str, ...] = PROFILE.packages


def product_profile() -> ProductProfile:
    """The installed product's profile -- the one object a product distribution
    replaces (see `knowledge_bases/product.py`)."""
    return PROFILE


def resolve_seed(reference: str):
    """Resolve a `"module.path:ATTR"` seed reference, importing lazily."""
    module_path, _, attr = reference.partition(":")
    return getattr(importlib.import_module(module_path), attr)


def _declaration_path(key: str) -> Path | None:
    """Where `key`'s `package.py` lives, looking in this tree then in entry points.

    In-tree first, because that is the developing case and must not depend on the
    package being pip-installed. Then `noesis.ontologies` entry points, which is how a
    SEPARATELY DISTRIBUTED ontology is found: `noesis-core` ships no ontologies, so
    without this the registry can only ever see packages that happen to sit next to
    it, and an installed `noesis-ontology-tasks` would be invisible.

    Returns `None` rather than raising so the caller can report which key failed
    along with where it looked.
    """
    # Every directory `knowledge_bases` spans, not just the one holding this file.
    # `knowledge_bases/__init__.py` calls `pkgutil.extend_path`, so an ontology
    # distribution contributes `knowledge_bases.<key>` from its OWN installation
    # directory -- and `Path(__file__).parent` only ever sees the engine's. Checking one
    # directory made the registry blind to exactly the packages the split produces,
    # while still working perfectly in a single tree, which is the failure mode most
    # likely to survive to a release.
    import knowledge_bases

    for root in knowledge_bases.__path__:
        in_tree = Path(root) / key / "package.py"
        if in_tree.exists():
            return in_tree

    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib here
        return None

    for entry in entry_points(group="noesis.ontologies"):
        if entry.name != key:
            continue
        # The entry point names a MODULE, and its declaration sits beside it, so the
        # package's `__init__` is still never imported (see `_load_declaration`).
        module = importlib.import_module(entry.value)
        candidate = Path(module.__file__).parent / "package.py"
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=None)
def _load_declaration(key: str) -> OntologyPackage:
    """Load `key`'s `package.py` without importing the knowledge base.

    Cached: a declaration is an immutable frozen dataclass read on hot paths
    (every changeset validation composes the kind registry), so executing the
    file once per process is both safe and necessary.

    The declaration must compose even while this module (or the knowledge
    base's own `__init__`) is only partially initialized, so the file is
    executed standalone rather than as `knowledge_bases.<key>.package` --
    importing that name would run the knowledge base's `__init__.py` and pull
    half the codebase into the composition path.
    """
    path = _declaration_path(key)
    if path is None:
        raise ValueError(
            f"Ontology package {key!r} is declared by the product profile but not "
            f"installed: no {key}/package.py beside the registry, and no "
            f"{key!r} entry point in the 'noesis.ontologies' group. Install its "
            "distribution or remove it from knowledge_bases/product.py."
        )
    spec = importlib.util.spec_from_file_location(f"_noesis_pkg_{key}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Ontology package {key!r} has no loadable package.py at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PACKAGE


def ontology_packages(
    keys: tuple[str, ...] = DEFAULT_PACKAGES,
) -> tuple[OntologyPackage, ...]:
    """Load every declared package's `PACKAGE` object, in order."""
    return tuple(_load_declaration(key) for key in keys)


def compose(packages: tuple[OntologyPackage, ...]) -> dict[str, KnowledgeBase]:
    """The live knowledge-base registry for a set of ontology packages.

    Refuses two kinds of collision, both of which used to be silent in one of the
    two: a duplicate knowledge-base KEY was already caught, but a duplicate LABEL
    was not. Two packages could co-own `Task` and composition succeeded, leaving a
    label with two owners, two sets of descriptions competing for the extraction
    prompt, and two tool surfaces writing the same nodes with different rules.

    This is not hypothetical. A consumer-scale `tasks` ontology is a lighter
    alternative to `project_management`, and both naturally want the name `Task`.
    They are alternatives, so the collision is the mechanism that makes them
    mutually exclusive -- but only if it is raised. Silent co-ownership is the worst
    outcome: the product looks composable and writes contradictory data.

    Raised here rather than at write time because installing two conflicting
    ontologies is a packaging mistake, and the composition boundary is where
    packaging mistakes should surface.
    """
    composed: dict[str, KnowledgeBase] = {}
    label_owner: dict[str, str] = {}
    relationship_owner: dict[str, str] = {}
    for package in packages:
        for kb in package.knowledge_bases:
            if kb.key in composed:
                raise ValueError(
                    f"Knowledge base {kb.key!r} declared by more than one "
                    f"ontology package"
                )
            for label in kb.labels:
                if label in label_owner and label_owner[label] != kb.key:
                    raise ValueError(
                        f"Label {label!r} is declared by both "
                        f"{label_owner[label]!r} and {kb.key!r}. A label has one "
                        "owner; if these are alternative ontologies for the same "
                        "concept, they are mutually exclusive and a product must "
                        "install exactly one"
                    )
                label_owner[label] = kb.key
            # Relationships are deliberately NOT exclusive the same way: a shared
            # edge like DEPENDS_ON is declared by several packages by design (one
            # owns its DESCRIPTION, which `relationship_types()` enforces). Only
            # note the first declarer, for diagnostics.
            for relationship in kb.relationships:
                relationship_owner.setdefault(relationship, kb.key)
            composed[kb.key] = kb
    # Edge contracts: one describer per relationship, the describer must
    # declare the edge on its own knowledge base, and every named writer
    # must be installed. Endpoint labels are NOT validated here: a narrower
    # product keeps the declaration (the enterprise product needs it) and the
    # contract goes inert -- `edge_contracts()` filters, `inert_edge_contracts()`
    # reports. That is the `DEPENDS_ON` judgement: dead edges cost nothing,
    # competing describers are a packaging mistake.
    contract_describers: dict[str, str] = {}
    for package in packages:
        declared = {
            rel for kb in package.knowledge_bases for rel in kb.relationships
        }
        for contract in package.edge_contracts:
            if contract.relationship in contract_describers:
                raise ValueError(
                    f"Relationship {contract.relationship!r} has two edge "
                    f"contracts ({contract_describers[contract.relationship]!r} "
                    f"and {package.key!r}); exactly one package may describe "
                    "an edge"
                )
            if contract.relationship not in declared:
                raise ValueError(
                    f"{package.key!r} declares an edge contract for "
                    f"{contract.relationship!r}, which it does not declare on "
                    "its own knowledge base; the describer must be a declarer"
                )
            # Writers are NOT validated here. A writer that is not installed
            # is inert, not an error: its code is absent, so the write cannot
            # happen, and a narrowed product keeps the declaration because the
            # enterprise product needs it. Only competing describers and a
            # describer that is not a declarer are packaging mistakes.
            contract_describers[contract.relationship] = package.key
    return composed


_PACKAGES = ontology_packages()
KNOWLEDGE_BASES = compose(_PACKAGES)


def _installed_labels() -> set[str]:
    return {label for kb in KNOWLEDGE_BASES.values() for label in kb.labels}


def edge_contracts() -> tuple["EdgeContract", ...]:
    """Active edge contracts: every endpoint label is owned by an installed
    package. Inert contracts (an endpoint only an omitted package supplies)
    are excluded -- the edge is unwritable in this product, which is a
    supported configuration, not an error."""
    labels = _installed_labels()
    return tuple(
        contract
        for package in _PACKAGES
        for contract in package.edge_contracts
        if (set(contract.from_labels) | set(contract.to_labels)) - {"*"} <= labels
    )


def inert_edge_contracts() -> tuple["EdgeContract", ...]:
    """Contracts whose endpoints no installed package supplies, for
    diagnostics: these are the dead edges a narrowed product accepts."""
    labels = _installed_labels()
    return tuple(
        contract
        for package in _PACKAGES
        for contract in package.edge_contracts
        if (set(contract.from_labels) | set(contract.to_labels)) - {"*"} > labels
    )


def knowledge_base(key: str) -> KnowledgeBase:
    try:
        return KNOWLEDGE_BASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown knowledge base: {key}") from exc


def search_types() -> dict[str, SearchType]:
    """The composed search-type registry (`hybrid_search`'s scope allowlist)."""
    return {
        f"{search_type.family}.{search_type.label}": search_type
        for package in _PACKAGES
        for search_type in package.search_types
    }


def extraction_shapes() -> tuple[ExtractionShape, ...]:
    """Every installed package's extraction shapes, in package order.

    `extract.py` builds its schema from this, so the shapes the LLM is offered are
    exactly the labels this product can store. Three refusals, in the order they
    matter:

    - A shape for a label no installed package declares is a shape the product
      cannot write. Refused here, next to the composition that decided the label
      set, rather than offered to the model and skipped at write time.
    - A shape with `fuzzy_text` and no `identity` is a shape whose graph id is
      never generated on a strict path -- a half-declared contract.
    - Two shapes claiming one field, or one label. Two owners for one noun is the
      collision `compose()` already refuses for labels; two owners for one field
      is the same collision expressed in the extraction schema, where it shows up
      as one Pydantic field with two meanings.
    """
    shapes: list[ExtractionShape] = []
    field_owner: dict[str, str] = {}
    label_owner: dict[str, str] = {}
    labels = _installed_labels()

    for package in _PACKAGES:
        if not package.extraction_shapes_seed:
            continue
        for shape in resolve_seed(package.extraction_shapes_seed):
            if shape.label not in labels:
                raise ValueError(
                    f"{package.key} declares an extraction shape for {shape.label!r}, "
                    "which no installed package declares"
                )
            if shape.fuzzy_text is None and shape.identity is None:
                raise ValueError(
                    f"{package.key}'s shape for {shape.label!r} has no identity "
                    "function and no fuzzy text property"
                )
            if shape.field in field_owner:
                raise ValueError(
                    f"extraction field {shape.field!r} is claimed by both "
                    f"{field_owner[shape.field]!r} and {package.key!r}"
                )
            if shape.label in label_owner:
                raise ValueError(
                    f"extraction shape for {shape.label!r} is claimed by both "
                    f"{label_owner[shape.label]!r} and {package.key!r}"
                )
            field_owner[shape.field] = package.key
            label_owner[shape.label] = package.key
            shapes.append(shape)

    return tuple(shapes)


def extraction_sources() -> tuple[ExtractionSource, ...]:
    """Every installed package's ingestion sources, in package order.

    The worker subscribes to the topics in this list, and dispatches each message
    to the parser its source names. A topic claimed by two packages is refused:
    two parsers for one stream means the same message is normalized two ways, and
    whichever runs second writes the graph.

    A source is declared by the package that owns the labels it produces, so
    uninstalling that package removes both its topics and its parsers -- the
    product stops subscribing to streams it cannot write.
    """
    sources: list[ExtractionSource] = []
    topic_owner: dict[str, str] = {}
    key_owner: dict[str, str] = {}

    for package in _PACKAGES:
        for seed in package.extraction_sources_seed:
            for source in resolve_seed(seed):
                if source.parser is None and source.triplets is None:
                    raise ValueError(
                        f"{package.key} declares extraction source {source.key!r} "
                        "with neither a parser nor a triplet emitter -- the "
                        "product would subscribe to a stream it cannot read"
                    )
                if source.key in key_owner:
                    raise ValueError(
                        f"extraction source {source.key!r} is claimed by both "
                        f"{key_owner[source.key]!r} and {package.key!r}"
                    )
                if source.topic is not None and source.topic in topic_owner:
                    raise ValueError(
                        f"ingestion topic {source.topic!r} is claimed by both "
                        f"{key_owner[source.topic]!r} and {package.key!r}"
                    )
                key_owner[source.key] = package.key
                if source.topic is not None:
                    topic_owner[source.topic] = package.key
                sources.append(source)

    return tuple(sources)


def deprecatable_knowledge_bases() -> tuple[str, ...]:
    return tuple(
        key for key, kb in KNOWLEDGE_BASES.items() if kb.deprecatable
    )


def call_noesis_tool_knowledge_bases() -> tuple[str, ...]:
    return tuple(
        key for key, kb in KNOWLEDGE_BASES.items() if kb.call_noesis_tool
    )


def orchestrator_dispatch_knowledge_bases() -> tuple[str, ...]:
    return tuple(
        key for key, kb in KNOWLEDGE_BASES.items() if kb.orchestrator_dispatch
    )


def composed_tools() -> list:
    """Every package's seeded `Tool` definitions, in package order."""
    tools: list = []
    for package in _PACKAGES:
        if package.tools_seed:
            tools.extend(resolve_seed(package.tools_seed))
    return tools


def composed_experts() -> list:
    """Every package's seeded `ExpertDefinition`s, in package order."""
    experts: list = []
    for package in _PACKAGES:
        if package.experts_seed:
            experts.extend(resolve_seed(package.experts_seed))
    return experts


@lru_cache(maxsize=None)
def property_contributions() -> tuple[PropertyContribution, ...]:
    """Properties made available by packages that do not own the label.

    A connector knows the identifiers its payloads carry, and `Person` belongs to
    the delivery ontology, so the connector cannot describe it -- `entity_types()`
    is single-owner. It contributes. Composition is where the declaration becomes
    a product fact: installing the Graph connector makes `teams_id` a declared
    property of `Person`, so `describe_ontology` offers it, `filter_properties`
    keeps it, and the merge ladder gains `person_teams::`. Uninstalling the
    connector removes all four.

    Precedence, stated rather than left to chance: when the label owner already
    declares the property, the owner's description wins and the contribution
    supplies only the identity prefix and its place in the ladder. `tasks`
    declares `email` on `Person`; the Graph connector contributes `email` with
    the `person_email` prefix, and the merge key comes from the connector while
    the meaning stays with the ontology. `project_management` does not declare
    `email`, so the connector's descriptor is added.

    Refused, all at composition:
    - A contribution to a label no installed package declares. The property would
      describe a node the product cannot write.
    - Two packages contributing one property to one label: two descriptions, and
      the ladder would have to pick.
    - Two identity keys with one graph-id prefix: two sources writing the same
      merge key, so a Teams id and a Slack username could collide.
    """
    composed: list[PropertyContribution] = []
    seen_prop: dict[tuple[str, str], str] = {}
    seen_prefix: dict[str, str] = {}
    for package in _PACKAGES:
        for seed in package.property_contributions_seed:
            for contribution in resolve_seed(seed):
                key = (contribution.label, contribution.prop)
                if key in seen_prop:
                    raise ValueError(
                        f"Property {contribution.prop!r} on label "
                        f"{contribution.label!r} is contributed by both "
                        f"{seen_prop[key]!r} and {package.key!r}. A property has "
                        "one description; if two connectors supply the same key, "
                        "one of them declares it"
                    )
                seen_prop[key] = package.key
                if contribution.prefix:
                    if contribution.prefix in seen_prefix:
                        raise ValueError(
                            f"Identity prefix {contribution.prefix!r} is declared "
                            f"by both {seen_prefix[contribution.prefix]!r} and "
                            f"{package.key!r}. Two sources writing one merge key "
                            "would merge unrelated entities"
                        )
                    seen_prefix[contribution.prefix] = package.key
                composed.append(contribution)
    return tuple(composed)


def identity_keys() -> tuple[PropertyContribution, ...]:
    """The composed merge ladder for identity-bearing contributions, ordered.

    Lowest `priority` first, so the ladder is a declaration and not a constant
    baked into one package. A product with no Graph connector never merges on a
    Teams id it cannot receive.
    """
    return sorted(
        (c for c in property_contributions() if c.prefix),
        key=lambda c: (c.priority, c.prop),
    )


def contributions_for(label: str) -> tuple[PropertyContribution, ...]:
    """The identity and property contributions composed for one label."""
    return tuple(c for c in property_contributions() if c.label == label)


def identity_keys_for(label: str) -> tuple[PropertyContribution, ...]:
    """The merge ladder for one label, in the order installed connectors declare."""
    return sorted(
        (c for c in contributions_for(label) if c.prefix),
        key=lambda c: (c.priority, c.prop),
    )


@lru_cache(maxsize=None)
def entity_types() -> dict[str, "EntityType"]:
    """Every installed package's described labels, keyed by label.

    This is the vocabulary ingestion extracts against: a label with no
    description here cannot be targeted unambiguously, which is why
    `tests/test_ontology_descriptions.py` requires one for every declared label.

    Contributed properties are merged into the label owner's declaration, so a
    property a connector supplies is offered by `describe_ontology` and kept by
    `filter_properties`. The label stays single-owner: a contribution adds a
    property, it does not claim the noun.
    """
    described: dict[str, Any] = {}
    for package in _PACKAGES:
        if not package.entity_types_seed:
            continue
        for entity in resolve_seed(package.entity_types_seed):
            if entity.label in described:
                raise ValueError(
                    f"Label {entity.label!r} described by more than one package"
                )
            described[entity.label] = entity

    declared_props = {
        label: {
            name if isinstance(name, str) else name.name
            for name in entity.properties
        }
        for label, entity in described.items()
    }
    for contribution in property_contributions():
        entity = described.get(contribution.label)
        if entity is None:
            raise ValueError(
                f"Property {contribution.prop!r} is contributed to label "
                f"{contribution.label!r}, which no installed package declares. "
                "A property describes a node the product cannot write"
            )
        if contribution.prop in declared_props[contribution.label]:
            # The owner's descriptor stands; the contribution only adds the
            # identity prefix, which `identity_keys_for()` reads directly.
            continue
        declared_props[contribution.label].add(contribution.prop)
        added = (PropertyType(name=contribution.prop,
                              description=contribution.description)
                 if contribution.description
                 else contribution.prop)
        described[contribution.label] = dataclasses.replace(
            entity, properties=tuple(entity.properties) + (added,)
        )
    return described


@lru_cache(maxsize=None)
def relationship_types() -> dict[str, "RelationshipType"]:
    """Every installed package's described relationships, keyed by type."""
    described: dict[str, Any] = {}
    for package in _PACKAGES:
        if not package.relationship_types_seed:
            continue
        for rel in resolve_seed(package.relationship_types_seed):
            if rel.relationship in described:
                raise ValueError(
                    f"Relationship {rel.relationship!r} described by more than "
                    "one package"
                )
            described[rel.relationship] = rel
    return described


@lru_cache(maxsize=None)
def cascade_steps() -> dict[str, "CascadeStep"]:
    """Every installed package's automatic reasoning steps, keyed by step key.

    Validated here rather than at use time, because a malformed step is a
    packaging error and the composition boundary is where packaging errors should
    surface. A step is rejected when it produces a label its owner does not
    declare (the step would fabricate another package's entities), when it names
    a relationship or label no installed package describes (the write could not
    be validated), or when it produces a non-extractable type (only repository
    code may write telemetry and versioning spines).

    `depends_on` pointing at an uninstalled step is NOT an error: a product that
    installs a subset of packages must still get a runnable chain, so the missing
    hop is simply absent and the runner starts from what is present.
    """
    from knowledge_bases.registry import entity_types as _entities
    from knowledge_bases.registry import relationship_types as _rels

    steps: dict[str, Any] = {}
    owned: dict[str, set[str]] = {}
    for package in _PACKAGES:
        owned[package.key] = {
            label for kb in package.knowledge_bases for label in kb.labels
        }
    entities = _entities()
    relationships = _rels()

    for package in _PACKAGES:
        if not package.cascade_steps_seed:
            continue
        for step in resolve_seed(package.cascade_steps_seed):
            if step.key in steps:
                raise ValueError(
                    f"Cascade step {step.key!r} declared by more than one package"
                )
            if step.owner != package.key:
                raise ValueError(
                    f"Cascade step {step.key!r} claims owner {step.owner!r} but "
                    f"is declared by {package.key!r}"
                )
            if step.relationship not in relationships:
                raise ValueError(
                    f"Cascade step {step.key!r} writes {step.relationship!r}, "
                    "which no installed package describes"
                )
            for label in step.produces_labels:
                if label not in owned[package.key]:
                    raise ValueError(
                        f"Cascade step {step.key!r} produces {label!r}, which "
                        f"{package.key!r} does not declare"
                    )
                described = entities.get(label)
                if described is None:
                    raise ValueError(
                        f"Cascade step {step.key!r} produces undescribed {label!r}"
                    )
                if not described.extractable:
                    raise ValueError(
                        f"Cascade step {step.key!r} produces {label!r}, which is "
                        "written by repository code only"
                    )
            steps[step.key] = step
    return steps


def composed_write_commands() -> list:
    """Every installed package's bounded write commands, in package order.

    Uninstalling a package must remove its ability to mutate the graph too:
    an allowlisted command whose handler no longer ships would be a write the
    orchestrator offers and cannot perform.
    """
    commands: list = []
    for package in _PACKAGES:
        if package.write_commands_seed:
            for command in resolve_seed(package.write_commands_seed):
                if command.knowledge_base != package.key:
                    raise ValueError(
                        f"Package {package.key!r} contributed a write command "
                        f"for {command.knowledge_base!r}: a package owns only "
                        "its own writes."
                    )
                commands.append(command)
    return commands

def semantic_search_kinds() -> frozenset[str]:
    """Versioned kinds whose managed content is embedded for semantic search.

    Composed here, from each package's `kinds_seed`, so the engine never has to
    import an ontology package to learn which labels are worth embedding. This was
    a hardcoded 11-name set inside `methodology/search.py`, which forced
    `orchestration/hybrid_search.py` -- an engine module bound for `noesis-core` --
    to import an optional ontology package at module scope. A product installed
    without `methodology` therefore failed to import its own search layer.

    Reading it off `VersionedKind` keeps the judgement with the package that owns
    the label, which is also the only place that can answer it: whether a label's
    text is worth embedding depends on what the label means.
    """
    kinds: set[str] = set()
    for package in _PACKAGES:
        if not package.kinds_seed:
            continue
        for kind in resolve_seed(package.kinds_seed):
            if kind.semantic_search:
                kinds.add(kind.kind)
    return frozenset(kinds)

def capability(name: str):
    """The installed provider of an optional engine hook, or `None`.

    The engine has behaviours that only make sense when a particular ontology is
    installed: an orchestrated answer catalogs user-supplied claims, which requires
    a package that owns `Claim`. Expressing that as an import made
    `orchestration/service.py` -- engine code bound for `noesis-core` -- depend on
    the `claims` ontology, so a product without claims could not import its own
    orchestrator.

    Returning `None` rather than raising is the whole point. A missing capability is
    a product shipping fewer ontologies, which is a supported configuration, not an
    error. The caller degrades: it returns the answer without the catalog step
    instead of failing the request.

    Two packages claiming one capability IS an error, and is raised here rather than
    resolved by registration order, because silently picking one provider's
    behaviour over another's is not a decision composition should make.
    """
    providers: dict[str, str] = {}
    for package in _PACKAGES:
        for declared, reference in package.capabilities_seed:
            if declared != name:
                continue
            if declared in providers:
                raise ValueError(
                    f"capability {name!r} is provided by more than one installed "
                    f"package ({providers[declared]!r} and {package.key!r}); "
                    "exactly one package may own an engine hook"
                )
            providers[declared] = package.key
            resolved = resolve_seed(reference)
    if not providers:
        return None
    return resolved


def capability_providers() -> dict[str, str]:
    """Which package provides each installed capability, for diagnostics."""
    providers: dict[str, str] = {}
    for package in _PACKAGES:
        for declared, _ in package.capabilities_seed:
            providers[declared] = package.key
    return providers

def versioned_kinds() -> dict[str, "VersionedKind"]:
    """Every versionable kind, keyed by its domain label.

    Composed here rather than in `methodology/kinds.py` because the composition
    reads only `kinds_seed` off each installed package -- it needs no methodology
    state at all. Leaving it there forced `triplets.py`, which ships with the
    engine, to import the methodology ontology in order to learn which kinds carry
    traceability rules. A product without methodology could not extract triplets.

    Ownership is checked, not trusted: a kind must be declared by exactly one
    package, and that package's knowledge base must own the label. A kind declared
    twice is ambiguous about which package's rules apply, and a kind claiming a
    label its owner does not declare would version an entity nothing can write.
    """
    kinds: dict[str, "VersionedKind"] = {}
    for package in _PACKAGES:
        if not package.kinds_seed:
            continue
        for kind in resolve_seed(package.kinds_seed):
            if kind.kind in kinds:
                raise ValueError(
                    f"Versioned kind {kind.kind!r} contributed by more than "
                    "one ontology package"
                )
            owner = KNOWLEDGE_BASES.get(kind.owner)
            if owner is None or kind.kind not in owner.labels:
                raise ValueError(
                    f"Versioned kind {kind.kind!r} declares owner "
                    f"{kind.owner!r}, which does not own that label"
                )
            kinds[kind.kind] = kind
    return kinds

def index_specs() -> tuple[tuple[str, str], ...]:
    """(label, property) for every index the installed ontologies need.

    Derived from each described label's `identity`, not listed, because that is
    exactly the property `graph.upsert_node` MERGEs on. A hand-kept list can
    disagree with the identity writers actually use, and that disagreement is
    invisible: the MERGE still works, it just degrades to a full label scan.

    Replaces 17 hardcoded labels in `graph.ensure_indexes`, which was wrong in both
    directions. It named DOMAIN labels in the engine, so a product without
    `semantic_layer` still created Table and Metric indexes, and a new ontology
    package could not get an index for its own labels without editing the engine.
    And it covered only 17 of 61 described labels while the other 44 are MERGEd on
    identity just the same -- an accidental historical subset, not a judgement.

    Returned sorted so the emitted DDL is stable and reviewable in a diff.
    """
    return tuple(sorted(
        (entity.label, entity.identity)
        for entity in entity_types().values()
    ))

def installed_packages() -> tuple[OntologyPackage, ...]:
    """The packages this process actually has installed.

    Distinct from `ontology_packages()`, which LOADS a named list and defaults to
    every declared package. Calling that with no argument answers "what could be
    installed", not "what is", so any composition built on it silently ignores a
    narrower product profile -- migrations for labels the product lacks, experts for
    ontologies it does not ship.

    Every composition site should ask this instead. It is a function rather than the
    `_PACKAGES` tuple so tests and profile switching have one seam to patch.
    """
    return _PACKAGES
