"""Ontology-driven triplet extraction: the checks a prompt cannot make.

Ingestion pipes any incoming information through triplet extraction against the
described ontology. Two properties have to hold in code rather than in the
prompt: an extractor must not be able to write a type nobody described, and it
must not be able to connect the wrong kinds of thing with a correctly-spelled
relationship. The second has no equivalent anywhere else in Noesis --
`graph.upsert_rel` checks that a label and a relationship each exist, never that
the combination means anything.
"""
from __future__ import annotations

import pytest

from knowledge_bases.base import EntityType, PropertyType, RelationshipType
from knowledge_bases.triplets import (
    Triplet,
    TripletError,
    filter_properties,
    partition,
    render_ontology_prompt,
    validate_triplet,
)


@pytest.fixture
def described(monkeypatch):
    """A tiny described ontology, so these tests pin behaviour rather than the
    real vocabulary (which each package owns and may legitimately reword)."""
    entities = {
        "Metric": EntityType(
            label="Metric", description="A quantified business measure with a definition.",
            properties=("name", "definition"),
        ),
        "Table": EntityType(
            label="Table", description="A physical table in the warehouse catalog.",
            properties=("name",),
        ),
        "Person": EntityType(
            label="Person", description="An individual human participant or owner.",
        ),
    }
    rels = {
        "USES": RelationshipType(
            relationship="USES", description="A metric reads from a physical table.",
            subject=("Metric",), object=("Table",),
        ),
        "OWNS": RelationshipType(
            relationship="OWNS", description="A person is accountable for a thing.",
            subject=("Person",),
        ),
    }
    monkeypatch.setattr("knowledge_bases.triplets.entity_types", lambda: entities)
    monkeypatch.setattr("knowledge_bases.triplets.relationship_types", lambda: rels)
    return entities, rels


def _triplet(**overrides):
    defaults = dict(
        subject_label="Metric", subject_identity="revenue",
        relationship="USES",
        object_label="Table", object_identity="fct_orders",
    )
    defaults.update(overrides)
    return Triplet(**defaults)


def test_a_well_typed_triplet_validates(described):
    assert validate_triplet(_triplet()) is not None


def test_an_undescribed_label_is_rejected(described):
    """An extractor must not invent an entity type: a label nobody described is
    one no reader knows how to interpret."""
    with pytest.raises(TripletError, match="not a described entity type"):
        validate_triplet(_triplet(subject_label="Sasquatch"))


def test_an_undescribed_relationship_is_rejected(described):
    with pytest.raises(TripletError, match="not a described relationship"):
        validate_triplet(_triplet(relationship="VIBES_WITH"))


def test_the_wrong_subject_type_is_rejected(described):
    """The case nothing else in Noesis catches: the relationship and both labels
    all exist, but the combination is meaningless."""
    with pytest.raises(TripletError, match="does not accept subject"):
        validate_triplet(_triplet(subject_label="Table", object_label="Table"))


def test_the_wrong_object_type_is_rejected(described):
    with pytest.raises(TripletError, match="does not accept object"):
        validate_triplet(_triplet(object_label="Person", object_identity="ada"))


def test_an_open_endpoint_accepts_any_described_label(described):
    """`OWNS` declares no object type, so it must accept any described label --
    open endpoints are deliberate, not an oversight to tighten here."""
    assert validate_triplet(_triplet(
        subject_label="Person", subject_identity="ada",
        relationship="OWNS", object_label="Metric", object_identity="revenue",
    ))


def test_partition_keeps_good_triplets_and_explains_the_rest(described):
    """Ingestion is best-effort: one bad triplet in a document must not discard
    the document, and each rejection must say why -- that is also the signal for
    a genuinely missing ontology type."""
    good = _triplet()
    bad = _triplet(relationship="VIBES_WITH")

    accepted, rejected = partition([good, bad])

    assert accepted == [good]
    assert len(rejected) == 1
    assert "VIBES_WITH" in rejected[0]["reason"]


def test_undeclared_properties_are_dropped(described):
    """An invented property is data no reader can query, so it is dropped rather
    than written onto the node."""
    kept = filter_properties("Metric", {"name": "Revenue", "vibe": "good"})

    assert kept == {"name": "Revenue"}


def test_the_prompt_states_meanings_and_legal_endpoints(described):
    """The prompt is the entire basis on which a model disambiguates, so it must
    carry the descriptions and the endpoint types, not just the names."""
    prompt = render_ontology_prompt()

    assert "A quantified business measure" in prompt
    assert "(Metric) -[:USES]-> (Table)" in prompt
    assert "any" in prompt  # OWNS has an open object


def test_the_prompt_can_be_scoped_to_one_knowledge_base(described):
    """A scoped extraction offers a narrower vocabulary, so a document about
    metrics is not tempted into unrelated types."""
    prompt = render_ontology_prompt(labels=("Metric",), relationships=("USES",))

    assert "Metric" in prompt
    assert "Person" not in prompt

# --- property extraction -------------------------------------------------
#
# Triplet extraction says *that* two things are related. Property extraction
# says what the things are like, and it is the half that fails silently: an
# extractor writing `status="ok"` on a Task produces a node that looks fine and
# matches no query anyone writes.


@pytest.fixture
def with_properties(monkeypatch):
    """`status` is the real ambiguity: 11 labels declare it, and the code backs it
    with five enums whose values do not overlap."""
    entities = {
        "Task": EntityType(
            label="Task", description="The smallest tracked unit of work.",
            properties=(
                "id",
                PropertyType(
                    name="status", description="Execution state of this task.",
                    enum=("todo", "in_progress", "blocked", "done"), required=True,
                ),
                PropertyType(
                    name="estimate", description="Estimated effort in points.",
                    value_type="int", example="3",
                ),
            ),
        ),
        "Span": EntityType(
            label="Span", description="One recorded dispatch of one tool.",
            properties=(
                PropertyType(
                    name="status", description="Whether the call succeeded.",
                    enum=("ok", "error"),
                ),
            ),
        ),
    }
    rels = {
        "PART_OF": RelationshipType(
            relationship="PART_OF", description="Groups a span into a trace.",
            properties=(
                PropertyType(
                    name="role", description="How the subject participates.",
                    enum=("primary", "secondary"),
                ),
            ),
        ),
    }
    monkeypatch.setattr("knowledge_bases.triplets.entity_types", lambda: entities)
    monkeypatch.setattr("knowledge_bases.triplets.relationship_types", lambda: rels)
    return entities, rels


def _span_triplet(**kw):
    defaults = dict(subject_label="Span", subject_identity="s1",
                    relationship="PART_OF",
                    object_label="Task", object_identity="t1",
                    object_props={"status": "todo"})
    defaults.update(kw)
    return Triplet(**defaults)


def test_a_value_from_another_labels_enum_is_rejected(with_properties):
    """The failure this exists to prevent. `ok` is a legal `status` -- on a Span.
    On a Task it is meaningless, but it is a plausible-looking string, so nothing
    downstream would flag it."""
    with pytest.raises(TripletError, match="not one of todo, in_progress"):
        validate_triplet(_span_triplet(object_props={"status": "ok"}))


def test_a_legal_enum_value_is_accepted(with_properties):
    assert validate_triplet(_span_triplet(object_props={"status": "done"}))


def test_a_missing_required_property_is_reported(with_properties):
    """Some facts are not worth recording incomplete: a Task with no status
    cannot answer the blocker question the label exists to serve."""
    with pytest.raises(TripletError, match="'status' is required"):
        validate_triplet(_span_triplet(object_props={}))


def test_an_undescribed_property_carries_no_constraint(with_properties):
    """A bare name declares nothing, so anything is allowed -- which is precisely
    the argument for describing it."""
    assert validate_triplet(_span_triplet(
        object_props={"status": "todo", "id": "anything at all"}))


def test_relationship_properties_are_validated_too(with_properties):
    """Edges carry properties in this repo already -- `CALLS {confidence, reason,
    line}`, `FOUND_IN {role}` -- so they need the same checking as nodes."""
    with pytest.raises(TripletError, match="role='driver' is not one of"):
        validate_triplet(_span_triplet(properties={"role": "driver"}))

    assert validate_triplet(_span_triplet(properties={"role": "primary"}))


def test_an_enum_list_is_checked_member_by_member(with_properties):
    """A list-valued property must not smuggle an illegal member past the check."""
    with pytest.raises(TripletError, match="not one of"):
        validate_triplet(_span_triplet(object_props={"status": ["todo", "ok"]}))


def test_the_prompt_states_each_described_propertys_legal_values(with_properties):
    """An enum the model cannot see is an enum it will violate."""
    prompt = render_ontology_prompt()

    assert "One of: todo, in_progress, blocked, done." in prompt
    assert "Required." in prompt
    assert "Example: 3" in prompt
    assert "How the subject participates." in prompt


def test_undescribed_and_described_properties_both_reach_the_prompt(with_properties):
    """Migration is per package, so a label may mix the two forms and must still
    offer its full property list."""
    prompt = render_ontology_prompt(labels=("Task",))

    assert "properties: id" in prompt
    assert "estimate" in prompt


def test_an_enum_on_a_numeric_property_is_a_declaration_error():
    """Catches the contradiction at import time rather than at extraction time."""
    with pytest.raises(ValueError, match="value_type"):
        PropertyType(name="count", description="How many.",
                     enum=("1", "2"), value_type="int")

# --- kind traceability --------------------------------------------------
#
# Some kinds are only meaningful with a companion edge, and methodology/kinds.py
# already says which. `submit_changeset` enforces those rules. Extraction must
# hold the same standard, or the same fact is accepted or refused depending only
# on which door it came through -- and the lenient door is how an unmeasurable
# Goal gets into the graph.


def test_a_goal_with_nothing_measuring_it_is_reported_as_incomplete():
    """`Goal requires an inbound MEASURES from a newly created SuccessMeasure`."""
    from knowledge_bases.triplets import Triplet, unmet_kind_requirements

    gaps = unmet_kind_requirements([Triplet(
        subject_label="Requirement", subject_identity="r1",
        relationship="ADVANCES", object_label="Goal", object_identity="g1",
    )])

    assert any(gap["label"] == "Goal" for gap in gaps)
    assert any("MEASURES" in gap["requirement"] for gap in gaps)


def test_a_goal_measured_in_the_same_batch_is_not_reported():
    """The companion edge normally comes from a different sentence of the same
    document, so the check spans the batch rather than one triplet."""
    from knowledge_bases.triplets import Triplet, unmet_kind_requirements

    gaps = unmet_kind_requirements([
        Triplet(subject_label="SuccessMeasure", subject_identity="sm1",
                relationship="MEASURES", object_label="Goal", object_identity="g1"),
    ])

    assert not [gap for gap in gaps if gap["label"] == "Goal"]


def test_a_kind_with_no_requirements_is_never_reported():
    from knowledge_bases.triplets import Triplet, unmet_kind_requirements

    gaps = unmet_kind_requirements([Triplet(
        subject_label="Metric", subject_identity="revenue",
        relationship="COMPUTED_FROM", object_label="Table",
        object_identity="fct_orders",
    )])

    assert gaps == []


def test_incomplete_knowledge_is_reported_but_still_written():
    """The judgement call, stated as a test: a changeset refuses an incomplete
    kind, ingestion records it and flags it. Refusing would discard the rest of a
    document over one half-stated fact, and an unmeasured Goal someone actually
    said is still true."""
    from knowledge_bases.ingestion_service import ingest_triplets
    from knowledge_bases.triplets import Triplet

    class FakeGraph:
        def __init__(self):
            self.nodes, self.rels, self.sources = [], [], []

        def upsert_node(self, label, key, props=None):
            self.nodes.append((label, key, props or {}))
            return [{}]

        def upsert_rel(self, sl, sk, rel, ol, ok, props=None):
            self.rels.append((sl, sk, rel, ol, ok, props))
            return [{}]

        def attach_source(self, label, key, source_id, source_props=None, role=None):
            return [{}]

    graph = FakeGraph()
    result = ingest_triplets(graph, [Triplet(
        subject_label="Requirement", subject_identity="r1",
        relationship="ADVANCES", object_label="Goal", object_identity="g1",
    )])

    assert result["relationships_written"] == 1, "the fact was discarded"
    assert result["incomplete"], "an unmeasurable Goal was written silently"
    assert any("MEASURES" in gap["requirement"] for gap in result["incomplete"])

# --- status is eleven different vocabularies ----------------------------------
#
# `status` is declared on eleven labels and backed by five non-overlapping sets of
# values. This is the property that most justifies describing properties at all:
# told only the name, an extractor writes a plausible value from the wrong set,
# and the result is worse than a crash because the node looks fine and matches no
# query anyone writes.
#
# Each value below is legal on some OTHER label, which is what makes it the
# realistic mistake rather than a strawman.


def _probe(label, value):
    """Validate a triplet whose edge is legal for `label`.

    The legal edge matters: endpoint checks run before property checks, so
    probing with a wrong edge would make these tests pass for the wrong reason.
    """
    from knowledge_bases.triplets import Triplet, TripletError, validate_triplet

    triplet = Triplet(
        subject_label=label, subject_identity="x1",
        relationship="FOUND_IN", object_label="SourceNode", object_identity="s1",
        subject_props={"status": value},
    )
    try:
        validate_triplet(triplet)
    except TripletError as exc:
        return str(exc)
    return None


@pytest.mark.parametrize("label,borrowed,owner", [
    ("Task", "ok", "Span"),
    ("Project", "done", "Task"),
    ("Claim", "completed", "WorkflowRun"),
    ("ReviewFinding", "cancelled", "Project"),
    ("Task", "planned", "Project"),
])
def test_a_status_value_from_another_labels_vocabulary_is_refused(label, borrowed, owner):
    error = _probe(label, borrowed)

    assert error is not None, (
        f"{label}.status={borrowed!r} belongs to {owner} and was accepted"
    )
    assert "status" in error, (
        f"refused for the wrong reason, which would make this test lie: {error}"
    )


@pytest.mark.parametrize("label,value", [
    ("Task", "blocked"),
    ("Project", "cancelled"),
    ("Claim", "refuted"),
    ("ReviewFinding", "rejected"),
])
def test_each_labels_own_status_values_are_accepted(label, value):
    """The other half of the guard. An enum that rejects everything would pass the
    test above while making ingestion useless."""
    assert _probe(label, value) is None


def test_decisions_capitalised_status_is_enforced_as_written():
    """`DecisionStatus` is `Active`/`ReviewNeeded` -- the only capitalised status
    in the graph. Lowercase `active` is legal on Claim and Project, so this is the
    single most likely status mistake in the codebase, and a `Decision` written
    with it would silently never match a health query.
    """
    assert _probe("Decision", "Active") is None
    assert _probe("Decision", "active") is not None


def test_every_status_declaration_is_described():
    """`status` is fully converted, so this is a total check rather than a ratchet:
    a new label declaring a bare `status` is a new silent-mismatch bug.
    """
    from knowledge_bases.base import PropertyType
    from knowledge_bases.registry import entity_types

    bare = [
        label
        for label, entity in entity_types().items()
        for prop in entity.properties
        if prop == "status" and not isinstance(prop, PropertyType)
    ]

    assert bare == [], (
        f"{bare} declare a bare `status`; say which values are legal, because "
        "eleven labels use the name for five different vocabularies"
    )

def test_status_on_an_internal_label_is_documentation_not_a_gate():
    """An honest limit, recorded so nobody reads more into the conversion than it
    gives. Five of the ten labels declaring `status` -- Span, WorkflowRun,
    RoutingTrace, WriteProposal, IngestionEvent -- are written only
    by repository code, so ingestion refuses them at the label before it ever
    reaches the enum. Their described values are still worth having: they are what
    `describe_ontology` reports and what a reader of the ontology learns from. They
    are simply not enforced on this path, because Python type hints and the
    repository's own enums already enforce them where those nodes are written.
    """
    from knowledge_bases.registry import entity_types

    internal = entity_types()["Span"]

    assert not internal.extractable
    status = next(p for p in internal.properties if getattr(p, "name", p) == "status")
    assert status.enum == ("ok", "error"), "still described for diagnostics"
    assert _probe("Span", "ok") is not None, "refused at the label, not the value"

# --- prompt scoping ----------------------------------------------------------
#
# Measured: relationship descriptions are ~42% of the prompt's bytes, so scoping
# labels alone leaves ~68% of the full cost (26,073 of 38,257 chars for a 5-label
# document). Dropping relationships no offered label can satisfy takes that to
# 8,533 -- a 67% cut on the scoped prompt.


def test_scoping_labels_alone_barely_reduces_the_prompt():
    """The measurement that motivates pruning. If this ever stops holding, the
    pruning below has lost its justification and should be re-argued."""
    from knowledge_bases.triplets import render_ontology_prompt

    full = len(render_ontology_prompt())
    labels_only = len(render_ontology_prompt(
        labels=("Person", "Task", "Project", "Decision", "Claim"),
        drop_unsatisfiable=False,
    ))

    assert labels_only > full * 0.6, (
        "label-only scoping now saves more than expected; re-measure"
    )


def test_pruning_unsatisfiable_relationships_cuts_the_scoped_prompt_hard():
    from knowledge_bases.triplets import render_ontology_prompt

    scope = ("Person", "Task", "Project", "Decision", "Claim")
    unpruned = len(render_ontology_prompt(labels=scope, drop_unsatisfiable=False))
    pruned = len(render_ontology_prompt(labels=scope))

    assert pruned < unpruned * 0.5, f"{pruned} vs {unpruned}: pruning is not working"


def test_pruning_never_removes_a_relationship_the_scope_could_use():
    """The correctness half, and the one that matters: a wrongly dropped edge is
    silent. The model is simply never told the move exists, so the fact is lost with
    no rejection and no error -- unlike an invalid triplet, which is loud.
    """
    from knowledge_bases.registry import relationship_types
    from knowledge_bases.triplets import render_ontology_prompt

    scope = ("Person", "Task", "Project", "Decision", "Claim")
    offered = set(scope)
    pruned = render_ontology_prompt(labels=scope)

    for name, rel in relationship_types().items():
        if not rel.extractable:
            continue
        satisfiable = (
            (not rel.subject or offered & set(rel.subject))
            and (not rel.object or offered & set(rel.object))
        )
        if satisfiable:
            assert f"-[:{name}]->" in pruned, (
                f"{name} is usable within {sorted(offered)} but was pruned away"
            )


def test_relationships_are_filtered_against_the_whole_label_set_not_one_package():
    """19 of the 61 extractable relationships cross package boundaries. Filtering
    per knowledge base would drop `MADE_BY` from a decisions-scoped prompt and lose
    the actor in "Alice decided X" -- the fact would vanish silently.
    """
    from knowledge_bases.registry import KNOWLEDGE_BASES
    from knowledge_bases.triplets import render_ontology_prompt

    decisions = KNOWLEDGE_BASES["strategy"].labels
    combined = tuple(set(decisions) | set(KNOWLEDGE_BASES["project_management"].labels))

    alone = render_ontology_prompt(labels=decisions)
    together = render_ontology_prompt(labels=combined)

    assert "-[:MADE_BY]->" not in alone, "Person is absent, so MADE_BY is unusable"
    assert "-[:MADE_BY]->" in together, (
        "adding the package that owns Person must restore the edge that needs it"
    )


def test_a_wider_scope_never_offers_less_than_a_narrower_one():
    """Monotonicity. Without it, adding a package could remove a capability, which
    would make scope selection unpredictable."""
    from knowledge_bases.registry import KNOWLEDGE_BASES
    from knowledge_bases.triplets import render_ontology_prompt

    narrow = set(KNOWLEDGE_BASES["strategy"].labels)
    wide = narrow | set(KNOWLEDGE_BASES["epistemology"].labels)

    narrow_edges = set(_re_edges(render_ontology_prompt(labels=tuple(narrow))))
    wide_edges = set(_re_edges(render_ontology_prompt(labels=tuple(wide))))

    assert narrow_edges <= wide_edges


def _re_edges(prompt):
    import re

    return re.findall(r"-\[:(\w+)\]->", prompt)


def test_the_unpruned_view_stays_available_for_diagnosis():
    """Needed to answer "was my scope too narrow?" -- which is unanswerable if the
    pruning cannot be turned off."""
    from knowledge_bases.triplets import render_ontology_prompt

    scope = ("Task",)
    assert len(render_ontology_prompt(labels=scope, drop_unsatisfiable=False)) > len(
        render_ontology_prompt(labels=scope)
    )


# -- the prompt must not describe what the installed product cannot create ----------
# Both of these became reachable only when products could install different ontology
# sets. Nothing is mis-declared: a declaration is written against the FULL vocabulary,
# which is correct, and the prompt is what must adapt.

def _knowledge_bases_roots() -> list[str]:
    """Every directory the `knowledge_bases` package currently spans."""
    import knowledge_bases

    return [str(root) for root in knowledge_bases.__path__]


def _personal_registry():
    """Compose a product that omits `project_management` and `semantic_layer`.

    A subprocess, because `registry` binds `_PACKAGES` at import and caches composed
    seeds, so rebinding the profile in-process leaks into every later test in the file.
    """
    import json
    import os
    import subprocess
    import sys

    program = """
import dataclasses, json, sys
import knowledge_bases

# The parent may have assembled `knowledge_bases` from several directories (the
# ontologies monorepo does this in `conftest.py`, mapping `packages/<key>` onto
# `knowledge_bases.<key>`). A subprocess runs no conftest, so that assembly is replayed
# here from the parent's own `__path__` -- otherwise this probe fails while importing
# the registry and reports nothing about the prompt.
for _root in %r:
    if _root not in knowledge_bases.__path__:
        knowledge_bases.__path__.append(_root)

import knowledge_bases.product as product""" % (list(_knowledge_bases_roots()),) + """
product.PROFILE = dataclasses.replace(product.PROFILE, packages=(
    "tools","orchestration","provenance","development","tasks","strategy",
    "methodology","architecture","epistemology"))
# Purging `knowledge_bases` itself discards the assembled `__path__` above, so it is
# re-applied on the fresh module object. Missing this produced an error about
# `development` "not installed" -- accurate, and nothing to do with the prompt.
_roots = list(knowledge_bases.__path__)
for name in [k for k in list(sys.modules)
             if k.startswith("knowledge_bases") and k != "knowledge_bases.product"]:
    del sys.modules[name]
import knowledge_bases
for _root in _roots:
    if _root not in knowledge_bases.__path__:
        knowledge_bases.__path__.append(_root)
from knowledge_bases import registry
from knowledge_bases.triplets import render_ontology_prompt
print(json.dumps({
    "labels": sorted(registry.entity_types()),
    "relationships": sorted(registry.relationship_types()),
    "prompt": render_ontology_prompt(),
}))
"""
    # `sys.path` is passed through PYTHONPATH because the child must resolve
    # `knowledge_bases` exactly as the parent does. In the ontologies monorepo that
    # package spans two directories and is assembled by `conftest.py`, which a bare
    # subprocess never runs -- so without this the probe fails on an import rather than
    # on the property under test.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [path for path in sys.path if path] + [environment.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    result = subprocess.run([sys.executable, "-c", program],
                            capture_output=True, text=True, cwd=".",
                            env=environment)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().split("\n")[-1])


def test_the_prompt_never_contrasts_against_a_type_the_product_does_not_install():
    """A contrast the reader cannot act on teaches a type instead of ruling one out.

    "Goal, NOT Metric" is the most load-bearing line in Goal's description on the
    enterprise product, where `Metric` exists and the confusion is real. Rendered to a
    personal product that installs no `semantic_layer`, it introduces `Metric`, asserts
    it exists, and offers no way to choose it -- which is how a scoped extraction starts
    inventing out-of-scope types.

    Measured before the fix: 21 distinct absent names from 53 stale contrasts.

    Filtered in the renderer rather than corrected at the declaration ON PURPOSE.
    Dropping `Metric` from `Goal` to satisfy the personal product would degrade the
    enterprise prompt, where those tokens earn their place.
    """
    data = _personal_registry()
    known = set(data["labels"]) | set(data["relationships"])

    offenders = []
    for line in data["prompt"].split("\n"):
        stripped = line.strip()
        if not stripped.startswith("NOT:"):
            continue
        for name in (n.strip() for n in stripped[4:].split(",")):
            if name and name not in known:
                offenders.append((stripped, name))

    assert offenders == [], (
        "the extraction prompt contrasts against types this product does not install, "
        f"so the model is taught labels it can never be offered: {offenders}"
    )


def test_the_prompt_never_offers_an_endpoint_the_product_cannot_create():
    """Worse than wasted tokens: the model is told a shape, then refused when it aims.

    `IMPLEMENTS` legitimately accepts `Epic`, `UserStory`, `Requirement`,
    `Specification` and `Capability` -- across the whole vocabulary. On a product
    without `project_management` none of those labels exists, so every one of them is a
    triplet `validate_triplet` rejects. Seven relationships advertised uncreatable
    endpoints before the fix.

    Parsed out of the rendered line rather than read off the declarations, because the
    declaration is RIGHT and the rendering was wrong; asserting on the declaration would
    test the wrong object.
    """
    import re

    data = _personal_registry()
    labels = set(data["labels"])

    offenders = []
    for line in data["prompt"].split("\n"):
        match = re.match(r"- \(([^)]*)\) -\[:(\w+)\]-> \(([^)]*)\):", line.strip())
        if not match:
            continue
        subject, relationship, obj = match.groups()
        for side in (subject, obj):
            if side == "any":
                continue
            for name in (n.strip() for n in side.split(",")):
                if name and name not in labels:
                    offenders.append((relationship, name))

    assert offenders == [], (
        "the prompt offers relationship endpoints whose label this product does not "
        f"install, so the extraction it invites would be rejected: {offenders}"
    )


def test_the_enterprise_prompt_keeps_every_contrast_it_declares():
    """The filter must be a function of what is INSTALLED, not a blanket narrowing.

    Guards against "fixing" the two tests above by dropping contrasts wholesale. On the
    full product every contrast names an installed type, so nothing may be pruned --
    including the cross-package ones, which are the most valuable (`FOUND_IN` contrasts
    itself with seven edges from four packages).
    """
    from knowledge_bases.registry import entity_types, relationship_types

    prompt = render_ontology_prompt()
    entities = entity_types()

    kept = set()
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.startswith("NOT:"):
            kept.update(n.strip() for n in stripped[4:].split(","))

    expected = {
        other
        for label, entity in entities.items()
        if entity.extractable
        for other in entity.not_to_be_confused_with
        if other in entities and entities[other].extractable
    }

    assert expected - kept == set(), (
        f"the enterprise prompt lost contrasts it should still make: "
        f"{sorted(expected - kept)}"
    )
