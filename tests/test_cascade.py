"""The automatic reasoning chain: ordering, gating, and the limits of automation.

The user's requirement is a chain -- ingestion produces claims, claims produce
decisions, decisions produce work -- where "user can start anywhere in this chain".
These tests pin the three properties that make that safe rather than merely
automatic.
"""
from __future__ import annotations

import time
from dataclasses import replace

import pytest

from knowledge_bases.base import CascadeStep
from knowledge_bases.cascade import (
    AttributionRequired,
    Confirmation,
    StrategyGuidance,
    plan_cascade,
    run_cascade,
)
from knowledge_bases.registry import cascade_steps, relationship_types
from knowledge_bases.triplets import Triplet


def test_a_source_alone_plans_the_whole_chain():
    """The headline behaviour: drop in a document, get the chain."""
    run = plan_cascade({"SourceNode"})

    flat = [step for wave in run.waves for step in wave]
    assert "epistemology.claims_from_sources" in flat
    assert "strategy.problems_from_claims" in flat
    assert "strategy.options_from_problems" in flat


@pytest.mark.parametrize("entry,expected", [
    ("SourceNode", "epistemology.claims_from_sources"),
    ("Claim", "strategy.problems_from_claims"),
    ("Problem", "strategy.options_from_problems"),
    ("Decision", "project_management.work_from_decisions"),
])
def test_the_user_can_start_anywhere_in_the_chain(entry, expected):
    """Verbatim from the TODO. An entity already in the graph must trigger its own
    hop directly -- a chain that only runs from the top is a pipeline, not a
    cascade, and would make every mid-chain entry a no-op.
    """
    run = plan_cascade({entry})

    flat = [step for wave in run.waves for step in wave]
    assert expected in flat, f"entering at {entry} planned {flat}"


def test_a_step_never_consumes_output_from_its_own_wave():
    """The property that separates automation from amplification. If a step could
    read what an earlier step in the same wave just inferred, one wrong Claim
    becomes a Problem, then an Idea, then committed work -- a single mistake
    converted into a structure that looks deliberate.
    """
    steps = cascade_steps()
    run = plan_cascade({"SourceNode"})

    position = {
        step: index for index, wave in enumerate(run.waves) for step in wave
    }
    for step_key, index in position.items():
        for dep in steps[step_key].depends_on:
            if dep in position:
                assert position[dep] < index, (
                    f"{step_key} shares a wave with its dependency {dep}"
                )


def test_a_step_with_no_trigger_present_is_skipped_with_a_reason():
    """Silence would be indistinguishable from a broken chain."""
    run = plan_cascade({"Metric"})

    assert run.waves == ()
    assert run.skipped
    assert all("needs one of" in entry["reason"] for entry in run.skipped)


def test_a_step_producing_an_entity_somebody_owns_must_name_that_person():
    """Replaces `test_the_chain_stops_before_fabricating_a_decision`, whose premise
    stopped being true when `decisions.strategy_from_goals` was added. It is REWRITTEN
    rather than deleted, because the thing it protected still needs protecting.

    What it used to assert: no step produces a `Decision`. That was a proxy. The real
    requirement was never "do not create Decisions" -- it was "do not manufacture
    accountability nobody accepted", and blocking the reasoning was one way to get
    that.

    What it asserts now is the requirement itself. A step may produce an owned entity
    provided it declares `human_attribution`, which forces a named Person onto a
    described edge before the write lands. Automation supplies the proposal; a human
    supplies their name.

    Stated as a rule over LABELS, not a whitelist of steps, so a future step producing
    a Decision -- or any other entity whose ownership is load-bearing -- is caught the
    day it is written.
    """
    # Labels that are definitionally somebody's: an unowned one is not an incomplete
    # record but a meaningless one.
    OWNED_LABELS = {"Decision"}

    for step in cascade_steps().values():
        owned = OWNED_LABELS.intersection(step.produces_labels)
        if not owned:
            continue
        assert step.human_attribution, (
            f"{step.key} produces {sorted(owned)}, which somebody must be "
            "accountable for, but declares no `human_attribution` -- it would write "
            "a choice traceable to nobody"
        )
        assert step.requires_confirmation, (
            f"{step.key} produces {sorted(owned)} without confirmation; attribution "
            "is meaningless if nobody was asked"
        )


def test_the_attribution_edge_points_at_a_person():
    """Guards the direction and the endpoint, not just the presence of a field.

    `human_attribution="MADE_BY"` is only worth anything if MADE_BY actually reaches a
    Person. A plausible-looking edge that pointed at a Team, or that ran the other way,
    would satisfy the test above while recording nothing anyone could be asked about.
    """
    relationships = relationship_types()

    for step in cascade_steps().values():
        if not step.human_attribution:
            continue
        rel = relationships.get(step.human_attribution)
        assert rel is not None, (
            f"{step.key} attributes via {step.human_attribution!r}, which no "
            "installed package describes"
        )
        assert "Person" in rel.object, (
            f"{step.key} attributes via {step.human_attribution!r}, whose object is "
            f"{rel.object} -- attribution must reach a Person, or there is nobody to "
            "hold answerable"
        )
        assert set(step.produces_labels).intersection(rel.subject), (
            f"{step.key} attributes via {step.human_attribution!r}, but none of "
            f"{step.produces_labels} is a legal subject ({rel.subject}); the edge "
            "would be written backwards"
        )


def test_every_step_is_owned_by_the_package_that_declares_its_output():
    """Uninstalling a package must remove the reasoning that produces its
    entities, not leave a step whose output label no longer exists."""
    from knowledge_bases.registry import KNOWLEDGE_BASES

    for step in cascade_steps().values():
        owner = KNOWLEDGE_BASES[step.owner]
        for label in step.produces_labels:
            assert label in owner.labels, (
                f"{step.key} is owned by {step.owner} but produces {label}"
            )


def test_every_step_writes_a_described_relationship():
    """A cascade write is validated like any other, so a step naming an undeclared
    edge would fail at execution rather than at packaging time."""
    from knowledge_bases.registry import relationship_types

    described = relationship_types()
    for step in cascade_steps().values():
        assert step.relationship in described, (
            f"{step.key} writes undescribed {step.relationship}"
        )


def test_a_dependency_cycle_is_reported_with_the_path():
    """A bare 'cycle exists' wastes the debugging time the path saves."""
    a = CascadeStep(key="a", owner="epistemology", trigger_labels=("SourceNode",),
                    produces_labels=("Claim",), relationship="FOUND_IN",
                    question="q", depends_on=("b",))
    b = CascadeStep(key="b", owner="epistemology", trigger_labels=("SourceNode",),
                    produces_labels=("Claim",), relationship="FOUND_IN",
                    question="q", depends_on=("a",))

    with pytest.raises(ValueError, match="cycle"):
        plan_cascade({"SourceNode"}, {"a": a, "b": b})


def test_a_dependency_on_an_uninstalled_step_is_not_an_error():
    """The composability requirement. `depends_on` names a step another package
    owns, so an absent dependency is the normal state of any product installing a
    subset -- raising would make the chain undeployable for exactly the narrower
    products the package split exists to serve.
    """
    orphan = CascadeStep(
        key="solo", owner="epistemology", trigger_labels=("SourceNode",),
        produces_labels=("Claim",), relationship="FOUND_IN", question="q",
        depends_on=("a.step.that.is.not.installed",),
    )

    run = plan_cascade({"SourceNode"}, {"solo": orphan})

    assert run.waves == (("solo",),)


class FakeGraph:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.nodes, self.rels = [], []

    def run(self, *args, **kwargs):
        return self.rows

    def upsert_node(self, label, key, props=None):
        self.nodes.append((label, key)); return [{}]

    def upsert_rel(self, sl, sk, rel, ol, ok, props=None):
        self.rels.append((sl, sk, rel, ol, ok)); return [{}]

    def attach_source(self, *args, **kwargs):
        return [{}]


def _proposal(step, triggers):
    return [Triplet(
        subject_label="Claim", subject_identity="i1",
        relationship="FOUND_IN",
        object_label="SourceNode", object_identity="s1",
    )]


def test_without_a_confirm_callback_inferences_are_planned_but_not_written():
    """The safe default. Treating absent approval as approval is how an unattended
    run fills the graph with unreviewed inference."""
    graph = FakeGraph(rows=[{"id": "s1", "labels": ["SourceNode"], "props": {}}])

    run = run_cascade(graph, {"SourceNode"}, _proposal)

    assert run.executed == ()
    assert graph.rels == [], "an inference was written without confirmation"
    assert run.pending_confirmation, "the proposal was dropped instead of queued"
    assert run.pending_confirmation[0]["question"]


def test_a_confirmed_step_writes_and_a_refused_step_does_not():
    graph = FakeGraph(rows=[{"id": "s1", "labels": ["SourceNode"], "props": {}}])

    approved = run_cascade(graph, {"SourceNode"}, _proposal,
                           confirm=lambda step, proposed: True)
    assert approved.executed, "confirmation did not lead to a write"
    assert graph.rels, "nothing reached the graph"

    refused_graph = FakeGraph(rows=[{"id": "s1", "labels": ["SourceNode"], "props": {}}])
    refused = run_cascade(refused_graph, {"SourceNode"}, _proposal,
                          confirm=lambda step, proposed: False)
    assert refused.executed == ()
    assert refused_graph.rels == []


def test_a_step_cannot_produce_more_than_its_cap():
    """One large document must not expand into an unbounded write batch."""
    graph = FakeGraph(rows=[{"id": "s1", "labels": ["SourceNode"], "props": {}}])

    def flood(step, triggers):
        return [Triplet(
            subject_label="Claim", subject_identity=f"i{n}",
            relationship="FOUND_IN",
            object_label="SourceNode", object_identity="s1",
        ) for n in range(500)]

    run = run_cascade(graph, {"SourceNode"}, flood,
                      confirm=lambda step, proposed: True)

    # Per step, not across the run: every triggered step gets its own budget, and
    # summing them would compare one step's cap against the whole wave's output.
    steps = cascade_steps()
    assert run.produced, "nothing ran, so the cap was not exercised"
    for entry in run.produced:
        cap = steps[entry["step"]].max_produced
        written = entry["result"]["relationships_written"]
        assert written <= cap, f"{entry['step']} wrote {written}, cap is {cap}"


def test_inferences_are_attributed_to_the_step_that_invented_them():
    """A graph that cannot tell observed from inferred will treat its own guesses
    as evidence on the next pass."""
    captured = {}

    class RecordingGraph(FakeGraph):
        def attach_source(self, label, key, source_id, source_props=None, role=None):
            captured["source_id"] = source_id
            return [{}]

    recording = RecordingGraph(rows=[{"id": "s1", "labels": ["SourceNode"], "props": {}}])
    run_cascade(recording, {"SourceNode"}, _proposal,
                confirm=lambda step, proposed: True)

    assert captured.get("source_id", "").startswith("cascade::"), (
        f"inference was not attributed to a cascade step: {captured}"
    )


# --- the strategy hop: proposing a Decision without fabricating accountability ----

_STRATEGY = CascadeStep(
    key="strategy", owner="strategy", trigger_labels=("Goal",),
    produces_labels=("Decision",), relationship="INTENDED_FOR",
    question="propose strategy", human_attribution="MADE_BY", steerable=True,
)


def _goal_graph():
    return FakeGraph(rows=[
        {"id": "g1", "labels": ["Goal"], "props": {"target": "cut churn"}},
        {"id": "g2", "labels": ["Goal"], "props": {"target": "raise ARPU"}},
    ])


def _strategy_proposal(step, triggers):
    return [
        Triplet(
            subject_label="Decision", subject_identity=f"d-{t['id']}",
            relationship="INTENDED_FOR",
            object_label="Goal", object_identity=t["id"],
        )
        for t in triggers
    ]


def test_approval_without_a_name_is_refused_for_a_step_that_needs_one():
    """The whole reason this step is allowed to exist.

    A bare `True` is approval: it records that a write was permitted. It does not
    record who stands behind the choice, and for a Decision that is the only part that
    matters -- the next hop turns Decisions into committed work, so an unattributed one
    puts work in front of people on the authority of nobody.

    It RAISES rather than skipping. Skipping would let an unattended run quietly drop
    exactly the step that needed a human, and the operator would read a clean report
    with the strategy hop silently missing.
    """
    graph = _goal_graph()

    with pytest.raises(AttributionRequired, match="accountable"):
        run_cascade(graph, {"Goal"}, _strategy_proposal,
                    steps={"strategy": _STRATEGY},
                    confirm=lambda step, proposed: True)

    assert graph.rels == [], "the refusal must happen before anything is written"


def test_a_named_confirmation_writes_the_decision_and_the_person_behind_it():
    """What the person supplies is their name on the proposal.

    The MADE_BY edge is asserted specifically: without it the Decision exists and
    looks fine, and the fact that nobody is answerable for it is invisible.
    """
    graph = _goal_graph()

    run = run_cascade(
        graph, {"Goal"}, _strategy_proposal, steps={"strategy": _STRATEGY},
        confirm=lambda step, proposed: Confirmation(
            approved=True, person_id="alice", person_name="Alice",
        ),
    )

    assert run.executed == ("strategy",)
    made_by = [r for r in graph.rels if r[2] == "MADE_BY"]
    assert len(made_by) == 2, "each proposed Decision needs its own attribution"
    assert all(r[0] == "Decision" and r[3] == "Person" for r in made_by)
    assert all(r[4] == {"id": "alice"} for r in made_by)


def test_attribution_names_only_what_the_step_produced():
    """Not everything the step touched.

    The step's output includes an edge back to the Goal that triggered it. Attributing
    the Goal as well would record that the confirming person authored the goal, which
    they did not -- and a graph that claims that will answer "who set this target"
    wrongly forever.
    """
    graph = _goal_graph()

    run_cascade(
        graph, {"Goal"}, _strategy_proposal, steps={"strategy": _STRATEGY},
        confirm=lambda step, proposed: Confirmation(approved=True, person_id="alice"),
    )

    attributed = {r[0] for r in graph.rels if r[2] == "MADE_BY"}
    assert attributed == {"Decision"}, f"also attributed: {attributed - {'Decision'}}"


def test_a_step_needing_no_attribution_still_accepts_a_bare_true():
    """Approving twenty inferred Claims is approval, not authorship.

    Demanding a name for every step would teach reviewers to supply one mechanically,
    and a name supplied mechanically is worth less than no name at all -- it looks like
    accountability in the graph.
    """
    graph = FakeGraph(rows=[{"id": "s1", "labels": ["SourceNode"], "props": {}}])

    run = run_cascade(graph, {"SourceNode"}, _proposal,
                      confirm=lambda step, proposed: True)

    assert run.executed
    assert not [r for r in graph.rels if r[2] == "MADE_BY"]


def test_an_unconfirmed_step_reports_that_a_name_will_be_needed():
    """So a review UI can ask for the name with the proposal, instead of rejecting
    the reviewer's submission after they have already decided."""
    graph = _goal_graph()

    run = run_cascade(graph, {"Goal"}, _strategy_proposal,
                      steps={"strategy": _STRATEGY})

    assert len(run.pending_confirmation) == 1
    assert run.pending_confirmation[0]["requires_attribution"] == "MADE_BY"


def test_the_confirming_person_is_the_actor_not_the_machine():
    """A Decision somebody put their name to must not be provenanced to `cascade`.

    Otherwise the graph holds two contradictory answers to "who did this": a MADE_BY
    edge naming a person, and provenance naming the automation. Anyone auditing later
    has to guess which one to believe.
    """
    graph = _goal_graph()
    written = {}

    def spy(label, key, props=None):
        written[label] = props or {}
        return [{}]

    graph.upsert_node = spy
    run_cascade(
        graph, {"Goal"}, _strategy_proposal, steps={"strategy": _STRATEGY},
        confirm=lambda step, proposed: Confirmation(approved=True, person_id="alice"),
        actor="cascade",
    )

    assert "Decision" in written, "no Decision was written"
    assert written["Decision"]["ingested_by"] == "alice", (
        "the Decision is provenanced to "
        f"{written['Decision']['ingested_by']!r}, contradicting the MADE_BY edge"
    )
    assert written["Person"]["ingested_by"] == "alice"


def test_a_proposed_decision_is_reported_incomplete_until_it_has_an_architecture():
    """The kind requirement is not bypassed by coming from the cascade.

    `Decision` requires an outgoing AFFECTS to an Architecture -- a choice with no
    structural consequence is not an architectural decision. An automatically proposed
    Decision satisfies MADE_BY and INTENDED_FOR but not that, so it lands flagged
    `incomplete` rather than silently looking finished.

    This is exactly the handoff to `architecture.structure_from_decisions`: the missing
    AFFECTS edge is what the next hop exists to supply, so `incomplete` here is the
    chain reporting its own next step rather than a defect.
    """
    graph = _goal_graph()

    run = run_cascade(
        graph, {"Goal"}, _strategy_proposal, steps={"strategy": _STRATEGY},
        confirm=lambda step, proposed: Confirmation(approved=True, person_id="alice"),
    )

    incomplete = run.produced[0]["result"]["incomplete"]
    decisions = [i for i in incomplete if i["label"] == "Decision"]
    assert decisions, "a Decision with no AFFECTS must be reported incomplete"
    assert "AFFECTS" in decisions[0]["requirement"]


# --- steering -------------------------------------------------------------------


def test_guidance_narrows_which_entities_the_step_reasons_over():
    """Steering corrects the INPUT. The usual failure is not a wild proposal but a
    nearly-right one reasoned from stale inputs -- every goal in the graph when only
    one is live this cycle."""
    graph = _goal_graph()
    saw = []

    def recording_reasoner(step, triggers):
        saw.append([t["id"] for t in triggers])
        return _strategy_proposal(step, triggers)

    run_cascade(
        graph, {"Goal"}, recording_reasoner, steps={"strategy": _STRATEGY},
        confirm=lambda step, proposed: Confirmation(approved=True, person_id="alice"),
        guidance={"strategy": StrategyGuidance(focus_ids=("g1",))},
    )

    assert saw == [["g1"]], "guidance did not narrow the triggers"


def test_guidance_that_selects_nothing_proposes_nothing():
    """The explicit no-op, asserted because the tempting implementation is worse.

    Falling back to "all triggers" when a focus list matches nothing would silently
    widen a scope somebody deliberately narrowed -- the one behaviour that would make
    steering untrustworthy, since the person would see a full run and believe they had
    constrained it.
    """
    graph = _goal_graph()

    run = run_cascade(
        graph, {"Goal"}, _strategy_proposal, steps={"strategy": _STRATEGY},
        confirm=lambda step, proposed: Confirmation(approved=True, person_id="alice"),
        guidance={"strategy": StrategyGuidance(focus_ids=("no-such-goal",))},
    )

    assert run.executed == ()
    assert graph.rels == []


def test_exclusions_and_emphasis_reach_the_reasoner_in_its_question():
    """Guidance has to change the prompt, not merely be accepted.

    An implementation that validated guidance and then ignored it would pass every
    other test here while a person watched the agent repeat the approach they had just
    ruled out.
    """
    graph = _goal_graph()
    asked = []

    def recording_reasoner(step, triggers):
        asked.append(step.question)
        return []

    run_cascade(
        graph, {"Goal"}, recording_reasoner, steps={"strategy": _STRATEGY},
        guidance={"strategy": StrategyGuidance(
            excluded=("rebuild billing from scratch",),
            emphasis=("migration risk",),
            note="Q3 only.",
        )},
    )

    assert len(asked) == 1
    assert "propose strategy" in asked[0], "the declared question must be preserved"
    assert "rebuild billing from scratch" in asked[0]
    assert "migration risk" in asked[0]
    assert "Q3 only." in asked[0]


def test_the_declared_step_is_not_mutated_by_steering():
    """Guidance is one person's framing for one run. A step whose `question` could be
    edited in place would carry that framing into everybody else's runs, and into the
    seed the next deployment reads."""
    graph = _goal_graph()
    before = _STRATEGY.question

    run_cascade(graph, {"Goal"}, lambda s, t: [], steps={"strategy": _STRATEGY},
                guidance={"strategy": StrategyGuidance(note="just this once")})

    assert _STRATEGY.question == before


def test_guiding_a_step_that_cannot_be_steered_is_refused():
    """Ignoring it silently is the harm: a person would believe they had directed
    reasoning that never saw their guidance."""
    unsteerable = replace(_STRATEGY, steerable=False)
    graph = _goal_graph()

    with pytest.raises(ValueError, match="steerable"):
        run_cascade(graph, {"Goal"}, _strategy_proposal,
                    steps={"strategy": unsteerable},
                    guidance={"strategy": StrategyGuidance(note="x")})


# --- concurrency ----------------------------------------------------------------


def test_steps_in_one_wave_can_reason_concurrently():
    """The thread limit the TODO asked for, applied where it is safe."""
    import threading

    active, peak, lock = 0, [0], threading.Lock()

    def slow(step, triggers):
        nonlocal active
        with lock:
            active += 1
            peak[0] = max(peak[0], active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return []

    steps = {
        f"s{n}": CascadeStep(
            key=f"s{n}", owner="epistemology", trigger_labels=("SourceNode",),
            produces_labels=("Claim",), relationship="FOUND_IN", question="q",
        )
        for n in range(4)
    }
    graph = FakeGraph(rows=[{"id": "s1", "labels": ["SourceNode"], "props": {}}])

    run_cascade(graph, {"SourceNode"}, slow, steps=steps, max_workers=4)

    assert peak[0] > 1, "max_workers=4 did not run any steps concurrently"


def test_waves_stay_sequential_however_many_workers_are_allowed():
    """The safety property `max_workers` must never be able to break.

    Wave separation is what stops a step consuming its own wave's unconfirmed output.
    Parallelising ACROSS waves would dismantle it, turning one wrong Claim into a
    Problem and then committed work inside a single run. So concurrency is bounded to
    within a wave, where steps are independent by construction.
    """
    order = []
    steps = {
        "first": CascadeStep(
            key="first", owner="epistemology", trigger_labels=("SourceNode",),
            produces_labels=("Claim",), relationship="FOUND_IN", question="q",
            requires_confirmation=False,
        ),
        "second": CascadeStep(
            key="second", owner="strategy", trigger_labels=("Claim",),
            produces_labels=("Problem",), relationship="ADDRESSES", question="q",
            depends_on=("first",), requires_confirmation=False,
        ),
    }

    def recorder(step, triggers):
        order.append(step.key)
        return []

    graph = FakeGraph(rows=[{"id": "s1", "labels": ["SourceNode"], "props": {}}])
    run = run_cascade(graph, {"SourceNode"}, recorder, steps=steps, max_workers=8)

    assert run.waves == (("first",), ("second",))
    assert order == ["first", "second"]


def test_concurrent_results_are_reported_in_wave_order():
    """Results arrive as threads finish. A report whose order depended on thread
    timing would be unreproducible for no benefit, and would make two runs of the
    same graph look like different runs."""
    steps = {
        f"s{n}": CascadeStep(
            key=f"s{n}", owner="epistemology", trigger_labels=("SourceNode",),
            produces_labels=("Claim",), relationship="FOUND_IN", question="q",
            requires_confirmation=False,
        )
        for n in range(4)
    }
    delays = {"s0": 0.04, "s1": 0.01, "s2": 0.03, "s3": 0.0}

    def variable(step, triggers):
        time.sleep(delays[step.key])
        return [Triplet(
            subject_label="Claim", subject_identity=f"i-{step.key}",
            relationship="FOUND_IN", object_label="SourceNode",
            object_identity="s1",
        )]

    graph = FakeGraph(rows=[{"id": "s1", "labels": ["SourceNode"], "props": {}}])
    run = run_cascade(graph, {"SourceNode"}, variable, steps=steps, max_workers=4)

    assert run.executed == ("s0", "s1", "s2", "s3")
