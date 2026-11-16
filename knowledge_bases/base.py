"""Contracts an ontology package contributes to the engine.

This module is a leaf: it imports nothing from the rest of Noesis, so every
knowledge base's `package.py` (and, later, every separately-installed
ontology distribution) can declare its contribution without circularity.
`knowledge_bases/registry.py` composes the declared packages into the live
registry; `knowledge_bases/tools/catalog.py` and
`knowledge_bases/orchestration/experts_catalog.py` compose each package's
seed slices (`tools_seed`/`experts_seed`, resolved lazily by module:attr so
importing a package's seed never eagerly loads the whole catalog).

An `OntologyPackage` is the unit of product composition: a product (a
deployment profile) is a set of these packages plus routing/exposure policy.
Everything a package contributes is still code inside a reviewed commit --
the composition API is an immutable surface (`knowledge_bases/boundary.py`),
not a runtime extension point.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class PropertyType:
    """One property of an entity or relationship, with the meaning it is filled by.

    Descriptions matter for properties for the same reason they matter for labels,
    and the evidence is sharper. `status` is declared on 11 labels and the code
    backs it with five different enums whose values do not overlap: a Task is
    `todo`/`in_progress`/`done`, a Span is `ok`/`error`, a Claim is
    `active`/`refuted`/`superseded`. An extractor told only the name `status`
    cannot know which vocabulary applies, so it writes a plausible value from the
    wrong set and the property becomes unqueryable. `name`, declared on 34 labels,
    has the same problem in weaker form.

    `enum` is therefore the load-bearing field: when a property has a closed set of
    legal values, stating them is what makes extraction checkable rather than
    hopeful. `required` marks what a fact is not worth recording without, and
    `example` disambiguates format (an ISO date versus a year) far more cheaply
    than prose can.
    """

    name: str
    description: str
    # Closed value set, where one exists. Empty means free text.
    enum: tuple[str, ...] = ()
    # `str`, `int`, `float`, `bool`, `list`, or an ISO-8601 `datetime`.
    value_type: str = "str"
    required: bool = False
    example: str | None = None

    def __post_init__(self) -> None:
        if self.enum and self.value_type not in ("str", "list"):
            raise ValueError(
                f"property {self.name!r} declares an enum, so its value_type "
                f"must be 'str' or 'list', not {self.value_type!r}"
            )


@dataclass(frozen=True)
class EntityType:
    """One declared label, with the meaning an extractor needs to use it.

    A bare label name is not enough to extract against. Language is
    context-dependent, so `Claim` vs `Argument`, or `Capability` vs `Requirement`,
    are distinguishable only if each says what it is *and* what it is not. The
    `description` is written for a model deciding which label a sentence maps to,
    not for a human already fluent in the domain -- so it states the boundary
    against near neighbours explicitly.

    `identity` names the property (or properties) that decide whether two
    mentions are the same entity. It is what makes append-only ingestion
    converge instead of creating a new node per phrasing.

    Identity is sometimes composite: a `ReviewFinding` is matched on
    `(project_id, key)` everywhere in the code, never on `id` alone, so a
    single-property contract would make ingestion converge on the wrong node for
    such a type. A string is therefore accepted as shorthand for a one-property
    identity, and `identity_properties` is what callers should read.
    """

    label: str
    description: str
    identity: str | tuple[str, ...] = "id"
    # Properties an extractor may populate. Anything not listed is dropped
    # rather than guessed onto the node. A bare string declares a property whose
    # meaning is still undescribed; a `PropertyType` describes it, which is what
    # makes property extraction checkable rather than hopeful. Both forms are
    # accepted so the two can be migrated per package instead of all at once.
    properties: tuple[str | PropertyType, ...] = ()
    # Labels this one is most often confused with, named so the extraction
    # prompt can contrast them directly.
    not_to_be_confused_with: tuple[str, ...] = ()
    # False for labels only repository code creates -- telemetry rows, routing
    # traces, revision spines. They are real ontology, queryable and describable,
    # but nothing in a document should ever produce one.
    extractable: bool = True

    @property
    def identity_properties(self) -> tuple[str, ...]:
        """`identity` normalised to a tuple, whichever form was declared."""
        if isinstance(self.identity, str):
            return (self.identity,)
        return tuple(self.identity)


@dataclass(frozen=True)
class RelationshipType:
    """One declared relationship, with its meaning and its legal endpoints.

    Endpoint typing is half the disambiguation: `SUPPORTS` between a Claim and
    a Decision is evidence, while `SUPPORTS` between a Component and a Capability
    would be implementation. Declaring `subject`/`object` labels lets a write be
    rejected for connecting the wrong kinds of thing, which a bare relationship
    name cannot express -- today nothing stops `(Metric)-[:REPORTS_TO]->(Table)`.

    Empty `subject`/`object` means "any declared label", kept for structural
    edges whose endpoints are genuinely open.
    """

    relationship: str
    description: str
    subject: tuple[str, ...] = ()
    object: tuple[str, ...] = ()
    # True when the edge carries meaning in one direction only, so an extractor
    # must not emit the reverse.
    directed: bool = True
    not_to_be_confused_with: tuple[str, ...] = ()
    # False for edges written only by repository code -- versioning spines
    # (CURRENT, SUPERSEDES), containment written by a builder (INCLUDES),
    # telemetry (CALLED, PART_OF), and routing bookkeeping. Offering them to an
    # extractor reading prose only invites bad guesses, and the set was
    # previously a hardcoded constant in extract.py that could silently drift
    # from what the packages declare.
    extractable: bool = True
    # Edges carry properties too, and the repo already relies on it:
    # `(:CodeFunction)-[:CALLS {confidence, reason, line}]->()`,
    # `-[:FOUND_IN {role}]->`, `-[:INCLUDES {position}]->`,
    # `-[:CONFLICTS_WITH {status}]->`. Until now there was nowhere to say what
    # those values mean, so an extractor could not populate them at all.
    properties: tuple[str | PropertyType, ...] = ()


@dataclass(frozen=True)
class KnowledgeBase:
    key: str
    name: str
    description: str
    labels: tuple[str, ...]
    relationships: tuple[str, ...]
    # Dispatch-surface declarations (were hardcoded tuples in
    # orchestration/{mcp_tools,service,lifecycle}.py; see registry.py):
    # entities may be deprecated/reactivated through the bounded lifecycle.
    deprecatable: bool = False
    # reachable through `call_noesis_tool` from the orchestration server.
    call_noesis_tool: bool = True
    # the orchestrator dispatches delegated operations to it directly.
    orchestrator_dispatch: bool = False


@dataclass(frozen=True)
class SearchType:
    """One searchable `(family, label)` pair (was `hybrid_search.SEARCH_TYPES`).

    `family` scopes a hybrid_search call: every type in one call must share
    it. `include_managed` marks labels that may also exist as
    methodology-managed entities (content hydrated from the current
    `Revision` via the managed indexes).
    """

    family: str
    label: str
    fulltext_index: str | None
    vector_collection: str | None = None
    identity_property: str = "id"
    include_managed: bool = False


@dataclass(frozen=True)
class KindRequirement:
    """One way a changeset-created entity can satisfy its creation rule.

    Valid if the changeset creates a relationship of type `relationship` in
    `direction` whose other endpoint carries one of `endpoint_kinds`.
    `endpoint_must_be_new` restricts that endpoint to entities created in the
    same changeset (otherwise a created *or* already-existing endpoint counts).
    A kind is valid when ANY of its requirements holds.
    """

    direction: Literal["outbound", "inbound"]
    relationship: str
    endpoint_kinds: tuple[str, ...]
    endpoint_must_be_new: bool = False


@dataclass(frozen=True)
class VersionedKind:
    """A domain label the methodology versioning engine may manage.

    Declared by the ontology package that owns the label (`owner`); the
    methodology package itself owns none -- it is the process, not the
    vocabulary. `requirements` encodes the creation-traceability rules a
    changeset must satisfy for this kind (what was the hardcoded if/elif
    chain in `ChangeSetRequest.validate_creation_traceability`).
    """

    kind: str
    owner: str
    requirements: tuple[KindRequirement, ...] = ()
    message: str | None = None  # error text when requirements go unmet
    # Whether this kind's managed content is embedded for semantic search.
    # Declared here, by the owning package, because it is a judgement about the
    # label: `Requirement` text is worth searching by meaning, an `Iteration` is a
    # date range with a name and embedding it returns noise. It was a hardcoded
    # 11-label set inside the engine (`methodology/search.py`), which meant the
    # engine held an opinion about labels it does not own and a new package could
    # not opt its own kinds in.
    semantic_search: bool = False


@dataclass(frozen=True)
class PhaseGate:
    """One invariant a phase must satisfy before it can be baselined.

    `query` returns offending `entity_id`s; a non-empty result is a
    `ValidationIssue` carrying `code`/`message`. Gates are declared next to the
    phase they guard so a product with a different phase sequence gets
    different gates without editing the engine
    (`knowledge_bases/methodology/validation.py`).
    """

    code: str
    message: str
    query: str


@dataclass(frozen=True)
class WorkflowPhase:
    """One phase of a product's methodology workflow.

    The engine owns phase *mechanics* (order, gating, baselining); which
    phases exist, who facilitates them, and what they produce is product
    configuration. `entity_types` scopes the artifact context a phase reads
    and must name contributed `VersionedKind`s; `aliases` are the workflow
    names a caller may use for this phase.
    """

    key: str
    persona: str
    domain: str
    artifact: str
    entity_types: tuple[str, ...]
    questions: tuple[str, ...] = ()
    gates: tuple[PhaseGate, ...] = ()
    aliases: tuple[str, ...] = ()
    # Kinds whose presence means the phase's artifact exists at all (vs. the
    # phase still needing to be drafted with the user).
    artifact_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class EdgeContract:
    """One cross-ontology edge, declared by the package that DESCRIBES it.

    The bridges README table made cross-domain edges a documented convention;
    this makes them composition-time declarations, the same way labels, tools,
    and capabilities are owned. A contract names:

    - `relationship`: the edge type, which the declaring package must also
      declare on its own knowledge base (the describer rule).
    - `from_labels` / `to_labels`: the endpoint labels, each owned by some
      installed package. `"*"` means "any label" (the FOUND_IN spine).
    - `writers`: package keys allowed to write this edge; `"*"` means every
      installed package. A writer that is not installed is a composition
      error; an endpoint label that is not installed makes the contract
      INERT, not an error -- a narrower product keeps the declaration (the
      enterprise product needs it) and the edge is simply unwritable here,
      pruned from the extraction prompt by `drop_unsatisfiable`.
    - `description`: what the edge means, the row the bridges README table
      is generated from.

    One contract per relationship type, exactly as one package per label:
    two describers for one edge is the `DEPENDS_ON` defect, and raising is
    the mechanism that keeps it from happening.
    """

    relationship: str
    from_labels: tuple[str, ...]
    to_labels: tuple[str, ...]
    writers: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class OntologyPackage:
    """One installable ontology's contribution to a product.

    `tools_seed`/`experts_seed` are `"module.path:ATTR"` references resolved
    lazily at `bootstrap_catalog()`/`bootstrap_experts()` time, so declaring
    a package costs only this module's import.
    """

    key: str
    knowledge_bases: tuple[KnowledgeBase, ...] = ()
    search_types: tuple[SearchType, ...] = ()
    tools_seed: str | None = None
    experts_seed: str | None = None
    # `"module.path:ATTR"` -> tuple[VersionedKind, ...]: the domain nouns the
    # methodology engine may version, contributed by their owning package.
    kinds_seed: str | None = None
    # Relationship types a methodology changeset may write because this
    # package's entities produce them. The union over installed packages is
    # the whole changeset vocabulary (`methodology/kinds.py`); each must also
    # be declared on a `KnowledgeBase` above, since the graph ontology is
    # enforced independently at write time.
    changeset_relationships: tuple[str, ...] = ()
    # Migration stream this package owns (P1.3+). Packages with no stream
    # contribute no migrations; "core" is the pre-split shared history.
    migration_stream: str | None = None
    # `"module.path:ATTR"` -> tuple[WriteCommand, ...]: the bounded writes this
    # package exposes, composed into the orchestrator's allowlist
    # (`orchestration/commands.py`). A package owns its own writes for the same
    # reason it owns its own tools: uninstalling it must remove its ability to
    # mutate the graph, not leave an allowlisted command whose handler is gone.
    write_commands_seed: str | None = None
    # `"module.path:ATTR"` -> tuple[EntityType, ...] / tuple[RelationshipType, ...]:
    # what this package's labels and relationships MEAN. Ingestion extracts
    # against these descriptions, so a package that declares a label without
    # describing it contributes a type no extractor can target correctly.
    entity_types_seed: str | None = None
    relationship_types_seed: str | None = None
    # `"module.path:ATTR"` -> tuple[CascadeStep, ...]: the automatic reasoning
    # hops that PRODUCE this package's entities. Owned by the producer, not the
    # trigger: uninstalling a package must remove the steps that fabricate its
    # entities, rather than leaving a step whose output label is gone.
    cascade_steps_seed: str | None = None
    # `{"capability": "module.path:ATTR"}`: optional engine hooks this package
    # implements. The engine calls a capability by NAME and gets `None` when no
    # installed package provides it, instead of importing the provider -- which is
    # what lets an optional ontology supply an engine behaviour without the engine
    # depending on the ontology. `feedback.analyze` and `feedback.record` are
    # claims' contribution: an orchestrated answer catalogs user-supplied facts
    # only if a package that owns claims is installed.
    capabilities_seed: tuple[tuple[str, str], ...] = ()
    # `EdgeContract` tuples: cross-ontology edges this package describes. The
    # declaring package is the describer; `writers` lists the package keys
    # allowed to write the edge. Resolved by `registry.edge_contracts()`,
    # which returns only contracts whose endpoint labels are installed.
    edge_contracts: tuple[EdgeContract, ...] = ()
    # `ExtractionShape` tuples: the entity shapes this package's labels can be
    # extracted into. Owned by the package that declares the label, for the same
    # reason it owns its tools: uninstalling a package must remove the shapes the
    # extractor offers for its labels, not leave the model picking a type the
    # graph cannot store.
    extraction_shapes_seed: str | None = None
    # `"module.path:ATTR"` seeds of `ExtractionSource` tuples: the ingestion
    # streams and pull sources this package can parse. Lazy, like every other
    # seed, so importing a product profile does not import a Kafka connector.
    # Composed by `registry.extraction_sources()`, so the worker subscribes to
    # what the installed packages publish.
    extraction_sources_seed: tuple[str, ...] = ()
    # `"module.path:ATTR"` seeds of `PropertyContribution` tuples: properties this
    # package makes available on labels it does not own. A connector knows the
    # identifiers its payloads carry, and `Person` belongs to the delivery
    # ontology, so the connector contributes rather than describes. Composed by
    # `registry.property_contributions()` and merged into `entity_types()`, so
    # `describe_ontology` offers the property, `filter_properties` keeps it, and
    # the merge ladder gains its key -- all four from one declaration.
    property_contributions_seed: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractionShape:
    """One entity shape the LLM extractor is allowed to fill, owned by the package
    that declares its label.

    `extract.py` used to hardcode ten Pydantic classes in the engine, so the
    extraction schema was a constant: a package declaring `Metric` contributed a
    label the extractor could never produce, and a product that omitted
    `semantic_layer` still had `Table` offered to the model. Declaring the shape
    next to the label makes the schema composed -- installing a package extends
    extraction, uninstalling one narrows it, the same rule every other surface
    follows.

    Fields:
    - `field`: the JSON field the model fills (`"people"`), which is also the
      vector collection name for fuzzy dedup.
    - `label`: the graph label it writes. Must be declared by an installed
      package, checked at composition.
    - `model`: `"module.path:ATTR"`, the Pydantic class. Resolved lazily, so
      declaring a shape costs this module's import.
    - `props`: `"module.path:ATTR"` -> callable(item, url) -> dict. The graph
      properties, because the model's field names are not always the graph's:
      `confidence_score` is written as `confidence`, `table_name` as `table`.
    - `identity`: `"module.path:ATTR"` -> callable(props) -> str. The graph id,
      because identity is domain knowledge: a Person merges on email, then
      teams_id, then slugified name; a Column merges on `table.name`. Required
      when `fuzzy_text` is None.
    - `fuzzy_text`: the property whose text is embedded for dedup. `None` means
      strict identity -- no vector search, because nobody needs the revision
      history of a table name.
    """

    field: str
    label: str
    model: str
    props: str
    identity: str | None = None
    fuzzy_text: str | None = None


@dataclass(frozen=True)
class PropertyContribution:
    """A property a package makes available on a label it does not own.

    A connector knows the identifiers its payloads carry: the Microsoft Graph
    connector knows an email and a Teams id, a Slack connector knows a Slack
    username. The label `Person` is owned by the delivery ontology, so a
    connector cannot describe it -- `registry.entity_types()` is single-owner,
    and that exclusivity is what keeps one noun from having two competing
    definitions. A contribution is the narrower thing: it adds a property to a
    label the product has installed, without claiming the label.

    This is what makes a source declare its schema. Installing the Graph
    connector makes `teams_id` a declared property of `Person`, so
    `describe_ontology` offers it, `filter_properties` keeps it, and the merge
    ladder gains `person_teams::`. Uninstalling the connector removes all four.
    Today `filter_properties` drops `email` and `teams_id` on every write,
    because no installed package declares them.

    Fields:
    - `label`: the label this property is contributed to. Must be declared by an
      installed package, checked at composition.
    - `prop`: the property name. Refused if the label owner already declares it,
      so a connector cannot redefine a property the ontology describes.
    - `prefix`: the graph-id prefix this key produces (`person_teams` yields
      `person_teams::T-42`). `None` for a property that is not an identity key.
    - `priority`: order in the merge ladder, lowest first. Identity keys are
      tried in this order and the first present wins, so the ladder is a
      declaration and not a constant baked into one package.
    - `lowercase`: whether the merge key normalizes case. An email is one
      address whatever its casing; a Teams id is a server-assigned string whose
      casing the connector must not invent. So it is declared per key, not
      assumed.
    - `description`: what the key is, for the extraction prompt.
    """

    label: str
    prop: str
    prefix: str | None = None
    priority: int = 100
    lowercase: bool = False
    description: str = ""


def identity_ladder(props: dict, keys: tuple[PropertyContribution, ...],
                    fallback: str) -> str:
    """The graph id, from the composed identity keys, first present wins.

    The order is the point: a meeting note saying "Alice" and an email from
    `alice@corp` are one person, so the contact key runs before the name. The
    ladder is composed from installed connectors, so a product with no Graph
    connector never merges on a Teams id it cannot receive.
    """
    for key in keys:
        value = props.get(key.prop)
        if value:
            text = str(value).strip()
            return f"{key.prefix}::{text.lower() if key.lowercase else text}"
    return fallback


@dataclass(frozen=True)
class NormalizedDocument:
    """One source payload, normalized to the text the extractor reads.

    The contract between a package's parser and the engine's extraction path.
    Every source -- an email, a Teams message, a transcript, a repo file --
    arrives in its own shape and leaves in this one, so the extractor never
    branches on where a document came from.
    """

    text: str
    doc_title: str
    source_name: str
    url: str = ""
    # The identifiers the payload carries, keyed by the name the contributing
    # package declares (`{"email": ..., "teams_id": ...}`). It used to be two
    # named fields, `owner_email` and `owner_teams_id`, which put a Microsoft
    # Graph concept in the engine's normalization contract: a Slack product had
    # no field for a Slack username, so the engine enumerated identifiers it
    # cannot know. A dict keyed by declared names is the same contract with no
    # vocabulary.
    owners: dict[str, str] = field(default_factory=dict)
    source_type: str = "Document"
    event_id: str | None = None


@dataclass(frozen=True)
class ExtractionSource:
    """One ingestion source, owned by the package that can store what it yields.

    The worker used to subscribe to a hardcoded list of Kafka topics
    (`raw-emails`, `teams-messages`, `evidence-documents`) that belong to the
    `claims` package's Microsoft Graph connector. The engine knew about a
    connector it does not own, and a product without claims subscribed to streams
    it cannot write. Declaring sources per package makes the subscription
    composed, and pairs each topic with the parser that normalizes its payload --
    so a source arriving is parsed by the package that owns its vocabulary, then
    handed to the shared extraction path.

    Fields:
    - `key`: stable identifier, `"<package>.<source>"`.
    - `topic`: the stream this source publishes to. `None` for a pull source (a
      repo scan, a Power BI export), which has no topic and is collected on demand.
    - `source_type`: the `SourceNode.type` the normalized document carries.
    - `parser`: `"module.path:ATTR"` -> callable(raw: dict) -> NormalizedDocument.
      Turns one raw payload into text plus provenance. This is the whole point:
      it is where a source-specific format stops being the engine's problem.
    - `triplets`: `"module.path:ATTR"` -> callable(raw: dict) -> list[Triplet],
      for a source whose format is deterministic. A Power BI TMDL table needs no
      model to tell you which columns it has, so its parser emits typed triplets
      straight into `ingestion_service.ingest_information`, skipping the LLM.
      Exactly one of `parser` / `triplets` is declared; composition refuses a
      source with neither, because a source that produces nothing is a topic
      the product subscribes to for no reason.
    - `collect`: `"module.path:ATTR"` -> callable() -> iterable[dict], for pull
      sources. `None` for stream sources.
    """

    key: str
    topic: str | None
    source_type: str
    parser: str | None = None
    triplets: str | None = None
    collect: str | None = None
    # `True` for a source that IS repository code -- a code scan, a schema import.
    # The `extractable` flag on a label and a relationship means "may be written
    # from extracted information"; a scan is not extracted information, so it is
    # allowed past that filter. The label, relationship, and endpoint checks still
    # run, so relaxing who may write does not relax what may be written.
    code_authored: bool = False


@dataclass(frozen=True)
class CascadeStep:
    """One automatic reasoning hop: "given these entities, what follows?"

    The user-facing goal is a chain -- ingestion produces claims, claims produce
    decisions, decisions produce plans, plans produce architecture -- that runs
    without being driven hop by hop. The chain is not new vocabulary: every edge
    it needs (`SUPPORTS`, `ADDRESSES`, `IMPLEMENTS`, `REALIZES`, `CHANGES`) is
    already declared and described by some package. What was missing is the
    statement that one stage *implies work on* the next.

    A step is declared by the package that owns the entities it PRODUCES, for the
    same reason a package owns its own tools and write commands: uninstalling a
    package must remove the reasoning that fabricates its entities, not leave a
    step whose output label no longer exists.

    Fields:
    - `trigger_labels`: labels whose appearance makes this step worth running.
    - `produces_labels`: labels the step may create. Must be declared by the
      owning package, and every one must be `extractable`: a step that invents
      telemetry or a versioning spine is a bug, not a feature.
    - `relationship`: the declared edge tying what it produces back to what
      triggered it. Validated against the composed ontology like any other
      write, so a step cannot smuggle in an edge no package describes.
    - `question`: what the reasoner is being asked. Held here rather than in code
      so it is refinable without a release, like expert prompts.
    - `depends_on`: other step keys that must run first. This is what makes the
      chain a DAG rather than a fixed list, so a product that installs only some
      packages still gets a valid ordering.
    - `requires_confirmation`: whether a human must approve before the write
      lands. Default TRUE, which is the opposite of ingestion's default, and the
      difference is deliberate: ingestion records what somebody actually said,
      while a cascade step ASSERTS SOMETHING NOBODY SAID. Recording dictation
      immediately is safe; writing an inference immediately is how a graph fills
      with plausible fiction that later reasoning then treats as evidence.
    - `human_attribution`: an edge the confirming human must supply, naming
      themselves, before the write lands. See the field comment below.
    - `steerable`: whether a person may hand this step guidance for one run.

    PRODUCING AN ENTITY SOMEBODY MUST OWN. For a long time no step could create a
    `Decision`, because a Decision is definitionally a choice somebody is
    accountable for and fabricating one manufactures accountability nobody
    accepted. The resolution is not to relax that, and not to leave the hop
    manual: it is to separate the two things that were conflated.

    Proposing a strategy is reasoning over evidence -- read the goals, find the
    claims that bear on them, lay out what should be done. That is work an agent
    does well, and a person does the same work no differently.

    Being ANSWERABLE for the strategy is not reasoning at all, and no amount of
    evidence produces it. It arrives when a named person adopts the proposal.

    So a step may produce such an entity provided it declares `human_attribution`,
    which forces the confirming person into the graph on a described edge. The
    write cannot land attributed to "cascade". What the automation supplies is the
    proposal; what the human supplies is their name on it.
    """

    key: str
    owner: str
    trigger_labels: tuple[str, ...]
    produces_labels: tuple[str, ...]
    relationship: str
    question: str
    depends_on: tuple[str, ...] = ()
    requires_confirmation: bool = True
    # Cap on entities one run of this step may create, so a single large document
    # cannot expand into an unbounded write batch.
    max_produced: int = 25
    # A relationship the CONFIRMING HUMAN must supply before this step's output may
    # be written -- `"MADE_BY"` for a step that proposes Decisions.
    #
    # This is what lets a step produce an entity that is definitionally somebody's
    # responsibility. `requires_confirmation` asks a human to approve a write;
    # `human_attribution` additionally records WHO, as a declared edge in the graph
    # rather than an audit line. The distinction matters because approval and
    # authorship are different facts: a reviewer waving through twenty inferred
    # A person approving a claim is answerable for it, while a person choosing a strategy is answerable for it.
    #
    # Left None for steps producing entities nobody needs to own. A Claim is a
    # reading of a source, and attributing it to whoever ran the cascade would
    # invent a stance they never took.
    human_attribution: str | None = None
    # Guidance a person may supply per run to steer this step -- which goals to work
    # from, approaches to rule out, considerations to weigh. Declared here, as a
    # capability of the step rather than of the runner, because a step that CANNOT
    # be steered can only be accepted or discarded wholesale, and "discard" loses
    # the reasoning that was nearly right.
    steerable: bool = False

@dataclass(frozen=True)
class ProductProfile:
    """One shippable product: which ontology packages it installs, and how it
    presents them.

    A product is not a fork of the engine. `noesis-core` is ontology-free, an
    ontology package is a separately installable distribution, and a product is
    the declaration that binds a set of them together with its own naming,
    routing posture, and workflow profile. Two products (`noesis-enterprise`,
    `noesis-individual`) therefore differ only by this object, not by branched
    code.

    Fields:
    - `packages`: the ontology package keys this product installs, in
      registration order. This is the single lever: it decides the graph
      ontology, search-type registry, seeded tools and experts, versionable
      kinds, and migration streams together.
    - `routable_experts`: experts a user's question may be routed to. A product
      may install a package for its ontology while keeping its expert off the
      router (`tools` and `provenance` are cross-cutting infrastructure, not
      domains a user asks about). `None` means "whatever the installed packages
      declare routable".
    - `workflow_profile`: `"module.path:ATTR"` -> tuple[WorkflowPhase, ...],
      the phase/persona/gate sequence this product runs. A solo product runs a
      shorter ladder than an enterprise one without redefining the engine.
    - `generated_tool_target`: the knowledge-base key whose `catalog.py` and
      `mcp_tools.py` receive self-improvement-authored handlers. It must name a
      package this product owns in its own source tree: installed packages live
      in `site-packages`, so appending generated logic to one would be both
      unwritable and lost on upgrade. `None` disables tool generation for the
      product.
    - `mcp_server_name`: the name the product registers under with the harness,
      and the prefix of its exported Agent Skill directories
      (`skills/<mcp_server_name>-<kb>/`). Two products installed side by side
      must not collide on it.
    """

    key: str
    name: str
    packages: tuple[str, ...]
    routable_experts: tuple[str, ...] | None = None
    workflow_profile: str | None = None
    generated_tool_target: str | None = None
    mcp_server_name: str = "noesis"

def slugify(text: str) -> str:
    """The stable slug every name-keyed node merges on.

    Shared because identity is the one rule that must agree between the writer
    that created a node and the extractor that finds it: `(:BusinessTerm)` is
    merged on `term::{slug}`, and a slug computed two ways creates two nodes for
    one term. `extract.py` used to keep its own copy, far from the packages that
    own the labels.
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def property_names(
    properties: tuple[str | "PropertyType", ...],
) -> tuple[str, ...]:
    """Just the names, whichever form each property was declared in."""
    return tuple(
        prop if isinstance(prop, str) else prop.name for prop in properties
    )


def described_properties(
    properties: tuple[str | "PropertyType", ...],
) -> dict[str, "PropertyType"]:
    """Only the properties that actually carry a description, keyed by name."""
    return {
        prop.name: prop
        for prop in properties
        if isinstance(prop, PropertyType)
    }
