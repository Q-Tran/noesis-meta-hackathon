"""The automatic reasoning chain: ingestion -> claims -> decisions -> work.

A user can start anywhere in that chain and have the rest follow without being
driven hop by hop. This module decides WHAT should run and in WHICH ORDER; the
reasoning itself is a caller-supplied function, so the ordering logic is testable
without an LLM and the same plan can be executed by an agent, a worker, or a test.

Three properties are load-bearing.

ORDER COMES FROM DECLARED DEPENDENCIES, NOT A HARDCODED LIST. The chain in the
TODO reads as a straight line, but a product that installs only some packages must
still get a valid ordering, and a consumer product that swaps
`project_management` for a lighter `tasks` ontology changes the line's shape. So
the order is a topological sort over `CascadeStep.depends_on`, and a missing hop is
simply absent rather than an error.

A STEP NEVER CONSUMES ITS OWN UNCONFIRMED OUTPUT. Steps are grouped into waves by
dependency depth, and a wave's inferences are not visible to the next wave until
they are confirmed. Without this, one wrong Claim becomes a Problem, then an
Epic, then Tasks -- the chain converting a single mistake into a structure that
looks deliberate. This is the difference between automation and amplification.

INFERENCES DEFAULT TO NEEDING CONFIRMATION. Ingestion writes immediately because
it records what somebody actually said. A cascade step asserts something nobody
said, and a graph that cannot distinguish "observed" from "inferred" will treat
its own guesses as evidence on the next pass.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from knowledge_bases.base import CascadeStep


@dataclass(frozen=True)
class CascadeRun:
    """What the chain did, or would do. Returned rather than logged so a caller
    can show a plan for approval before anything executes."""

    waves: tuple[tuple[str, ...], ...]
    executed: tuple[str, ...] = ()
    produced: tuple[dict[str, Any], ...] = ()
    pending_confirmation: tuple[dict[str, Any], ...] = ()
    skipped: tuple[dict[str, str], ...] = ()


def _install_order(steps: dict[str, CascadeStep]) -> list[str]:
    """Step keys in dependency order, ignoring dependencies on absent steps.

    Ignoring rather than raising is deliberate. `depends_on` names a step that a
    DIFFERENT package owns, so an uninstalled dependency is the normal state of
    any product that installs a subset -- treating it as an error would make the
    chain undeployable for exactly the products the split exists to serve.
    """
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(key: str, stack: tuple[str, ...]) -> None:
        if state.get(key) == 2:
            return
        if state.get(key) == 1:
            cycle = " -> ".join(stack[stack.index(key):] + (key,))
            raise ValueError(f"Cascade step dependency cycle: {cycle}")
        state[key] = 1
        for dep in sorted(steps[key].depends_on):
            if dep in steps:
                visit(dep, stack + (key,))
        state[key] = 2
        order.append(key)

    for key in sorted(steps):
        visit(key, ())
    return order


def plan_cascade(
    present_labels: set[str],
    steps: dict[str, CascadeStep] | None = None,
) -> CascadeRun:
    """Which steps are worth running, grouped into dependency waves.

    A step is included when the graph already holds one of its `trigger_labels`,
    OR when an earlier wave in this same plan will produce one. The second half is
    what lets the user "start anywhere": handing this a bare `SourceNode` plans the
    whole chain, while handing it an existing `Decision` plans only the work hop.
    """
    if steps is None:
        from knowledge_bases.registry import cascade_steps

        steps = cascade_steps()

    ordered = _install_order(steps)
    reachable = set(present_labels)
    waves: list[tuple[str, ...]] = []
    remaining = list(ordered)
    skipped: list[dict[str, str]] = []

    # A step whose trigger is already in the graph is runnable now, whatever its
    # dependencies say. `depends_on` orders steps that feed each other; it must not
    # gate a step whose input already exists, or starting mid-chain (the user's
    # explicit requirement -- "user can start anywhere in this chain") would plan
    # nothing at all. So a dependency only delays a step while that dependency is
    # itself going to run in this plan.
    while remaining:
        runnable = [
            key for key in remaining
            if any(label in reachable for label in steps[key].trigger_labels)
        ]
        if not runnable:
            for key in remaining:
                skipped.append({
                    "step": key,
                    "reason": (
                        "no trigger label present: needs one of "
                        + ", ".join(steps[key].trigger_labels)
                    ),
                })
            break
        # Within the runnable set, hold back anything waiting on another runnable
        # step, so a step never consumes output produced in the same wave.
        wave = [
            key for key in runnable
            if not any(dep in runnable for dep in steps[key].depends_on if dep in steps)
        ]
        waves.append(tuple(wave))
        for key in wave:
            reachable.update(steps[key].produces_labels)
            remaining.remove(key)

    return CascadeRun(waves=tuple(waves), skipped=tuple(skipped))


# A reasoner turns one step plus its trigger entities into proposed triplets. Kept
# as a caller-supplied callable so ordering is testable without an LLM, and so the
# same plan can be run by an agent, a batch worker, or a test.
Reasoner = Callable[[CascadeStep, list[dict[str, Any]]], list[Any]]


class AttributionRequired(Exception):
    """A step producing an entity somebody must own was confirmed by nobody.

    Raised rather than skipped, and that choice is deliberate. Skipping would let an
    unattended run quietly drop exactly the steps that matter most, and the operator
    would see a clean report with the strategy hop silently absent. A refusal names
    the problem at the moment it can still be fixed.
    """


@dataclass(frozen=True)
class Confirmation:
    """A human's answer to one proposed step: approved, and by whom.

    `person_id` is not metadata. For a step declaring `human_attribution` it becomes
    a described edge from the produced entity to that Person, which is what makes an
    automatically-proposed Decision attributable to somebody. A bare `True` is
    accepted for steps that need no attribution -- approving twenty inferred Claims
    is approval, not authorship, and demanding a name there would teach reviewers to
    type one in without meaning it.
    """

    approved: bool
    person_id: str | None = None
    person_name: str | None = None
    note: str | None = None


def _resolve_confirmation(value: Any) -> Confirmation:
    """Accept `True`/`False` or a `Confirmation`, so existing callbacks keep working."""
    if isinstance(value, Confirmation):
        return value
    return Confirmation(approved=bool(value))


def run_cascade(
    graph: Any,
    present_labels: set[str],
    reasoner: Reasoner,
    *,
    steps: dict[str, CascadeStep] | None = None,
    confirm: Callable[[CascadeStep, list[Any]], Any] | None = None,
    max_waves: int = 6,
    actor: str = "cascade",
    guidance: dict[str, StrategyGuidance] | None = None,
    max_workers: int = 1,
) -> CascadeRun:
    """Execute the planned chain, wave by wave.

    `confirm` is consulted for any step with `requires_confirmation`, which is the
    default. Without a `confirm` callback those steps are PLANNED BUT NOT WRITTEN
    and come back in `pending_confirmation` -- the safe default, because silently
    treating absent approval as approval is how an unattended run fills the graph
    with unreviewed inference.

    `max_waves` bounds the walk. The plan is acyclic, so this is defence against a
    reasoner whose output re-triggers an earlier step forever rather than against
    the DAG itself.

    `guidance` maps a step key to `StrategyGuidance` for steerable steps, letting a
    person direct the reasoning rather than only judge its result.

    `max_workers` runs the steps WITHIN one wave concurrently. Waves themselves stay
    strictly sequential and always will: the separation between waves is what stops a
    step consuming its own wave's unconfirmed output, so parallelising across waves
    would dismantle the safety property the chain is built on. Within a wave the steps
    are independent by construction -- that is what put them in the same wave -- so
    concurrency there is free.

    Defaults to 1 because the reasoner is caller-supplied and may not be thread-safe,
    and because a serial run is reproducible. Raise it when the reasoner is an LLM
    call, where a wave of four steps is four round trips that need not be sequential.
    """
    from knowledge_bases.ingestion_service import ingest_triplets

    if steps is None:
        from knowledge_bases.registry import cascade_steps

        steps = cascade_steps()

    plan = plan_cascade(present_labels, steps)
    executed: list[str] = []
    produced: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    guidance = guidance or {}

    def propose(key: str) -> tuple[str, list[Any]] | None:
        """Reason about one step: load its triggers, apply guidance, call the
        reasoner. Read-only with respect to the graph, which is what makes it safe
        to run several of these at once -- every WRITE happens on the main thread
        below, so confirmation stays sequential and one reviewer is never asked two
        questions simultaneously.
        """
        step = steps[key]
        triggers = _load_triggers(graph, step)
        if not triggers:
            return None
        question = step.question
        steer = guidance.get(key)
        if steer is not None and not steer.is_empty():
            if not step.steerable:
                raise ValueError(
                    f"Guidance was supplied for {key}, which does not declare "
                    "`steerable`. Silently ignoring it would let a person believe "
                    "they had directed reasoning that ignored them."
                )
            question, triggers = steer.steer(step, triggers)
            if not triggers:
                return None
        proposed = reasoner(replace(step, question=question), triggers)
        if not proposed:
            return None
        # Cap before writing, not after: the limit exists to stop one large
        # document expanding into an unbounded batch.
        if len(proposed) > step.max_produced:
            proposed = proposed[: step.max_produced]
        return key, proposed

    for wave in plan.waves[:max_waves]:
        # Reasoning fans out across the wave; writing does not. Steps in one wave are
        # independent by construction, which is precisely what put them in the same
        # wave, so there is no ordering to preserve while proposing.
        if max_workers > 1 and len(wave) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(propose, key): key for key in wave}
                by_key = {}
                for future in as_completed(futures):
                    outcome = future.result()
                    if outcome:
                        by_key[outcome[0]] = outcome[1]
            # Re-imposed wave order. Results arrive as they finish, and a run whose
            # report order depended on thread timing would be unreproducible for no
            # benefit.
            batch = [(key, by_key[key]) for key in wave if key in by_key]
        else:
            batch = [outcome for outcome in (propose(key) for key in wave) if outcome]

        for key, proposed in batch:
            step = steps[key]
            if step.requires_confirmation and confirm is None:
                pending.append({
                    "step": key,
                    "question": step.question,
                    "proposed": len(proposed),
                    "triplets": proposed,
                    # Surfaced so a UI knows a name will be required before this one
                    # can be approved, rather than discovering it on submit.
                    "requires_attribution": step.human_attribution,
                })
                continue

            step_actor = actor
            if step.requires_confirmation:
                answer = _resolve_confirmation(confirm(step, proposed))
                if not answer.approved:
                    continue
                if step.human_attribution:
                    if not answer.person_id:
                        raise AttributionRequired(
                            f"{key} produces {list(step.produces_labels)}, which "
                            f"somebody must be accountable for, so its confirmation "
                            f"must name a Person for the {step.human_attribution} "
                            "edge. Approval alone records that a write was allowed, "
                            "not who stands behind it."
                        )
                    proposed = _attribute(
                        proposed, step, answer.person_id, answer.person_name,
                    )
                    # The confirming person, not "cascade". A Decision they are named
                    # on must not also be provenanced to the machine that drafted it.
                    step_actor = answer.person_id

            result = ingest_triplets(
                graph, proposed,
                source_id=f"cascade::{key}",
                source_props={"name": f"inferred by {key}", "type": "Inference"},
                actor=step_actor,
            )
            executed.append(key)
            produced.append({"step": key, "result": result})


    return CascadeRun(
        waves=plan.waves,
        executed=tuple(executed),
        produced=tuple(produced),
        pending_confirmation=tuple(pending),
        skipped=plan.skipped,
    )


def _load_triggers(graph: Any, step: CascadeStep, limit: int = 50) -> list[dict[str, Any]]:
    """The current entities that make this step worth running.

    Filtered by `current_filter()` so superseded and deprecated entities do not
    re-trigger reasoning that was already done and then corrected.
    """
    from knowledge_bases.ingestion import current_filter

    found: list[dict[str, Any]] = []
    for label in step.trigger_labels:
        rows = graph.run(
            f"MATCH (n:{label}) WHERE {current_filter('n')} "
            "RETURN n.id AS id, labels(n) AS labels, properties(n) AS props "
            "LIMIT $limit",
            limit=limit,
        )
        found.extend(rows or [])
    return found

def _attribute(
    proposed: list[Any],
    step: CascadeStep,
    person_id: str,
    person_name: str | None,
) -> list[Any]:
    """Add the attribution edge from each owned entity this step produced.

    Appended as ORDINARY TRIPLETS rather than written separately, so the edge is
    validated against the composed ontology exactly like everything else the step
    proposes. A step cannot attribute along an edge no package describes, and it
    cannot point the edge at a label that is not a legal endpoint -- the same
    validation that stops a reasoner inventing relationships stops a runner
    inventing an attribution.

    Only entities the step declares as `produces_labels` are attributed. A step's
    output typically includes edges back to its trigger (`INTENDED_FOR` to a Goal,
    `SUPPORTS` from a Claim), and attributing the Goal to the person who approved
    a strategy would claim they authored the goal too.
    """
    from knowledge_bases.triplets import Triplet

    owned: dict[str, str] = {}
    for triplet in proposed:
        for label, identity in (
            (triplet.subject_label, triplet.subject_identity),
            (triplet.object_label, triplet.object_identity),
        ):
            if label in step.produces_labels:
                owned[f"{label}\x00{identity}"] = label

    person_props = {"id": person_id}
    if person_name:
        person_props["name"] = person_name

    edges = [
        Triplet(
            subject_label=label,
            subject_identity=key.split("\x00", 1)[1],
            relationship=step.human_attribution or "",
            object_label="Person",
            object_identity=person_id,
            object_props=person_props,
        )
        for key, label in owned.items()
    ]
    return list(proposed) + edges

@dataclass(frozen=True)
class StrategyGuidance:
    """What a person hands a steerable step to direct its reasoning for one run.

    Steering and confirming are different controls, and a system with only the second
    is worse than it looks. Confirmation judges a finished proposal: accept it or
    throw it away. Throwing it away is lossy in the common case, which is not a wild
    proposal but a nearly-right one -- an agent that weighed a constraint the person
    knows was lifted last quarter, or that worked from every goal when only one is
    live this cycle. The reasoning was sound and the inputs were wrong, and
    confirmation cannot say so.

    Steering fixes the inputs. Confirmation still judges the output. Both are kept
    because they fail differently: steering cannot catch a bad conclusion from good
    inputs, and confirmation cannot catch a good conclusion from stale ones.

    Deliberately NOT persisted as graph entities. Guidance is one person's framing for
    one run, not a durable fact about the domain -- writing it into the ontology would
    turn "ignore the legacy billing path for now" into a standing constraint the next
    run treats as evidence. What IS persisted is the outcome: the Decision, and the
    person named on it.

    Fields are all optional; guidance is additive, and an empty instance is the
    unsteered case rather than an error.
    """

    # Narrow which trigger entities the step reasons over. Empty means all of them.
    focus_ids: tuple[str, ...] = ()
    # Approaches ruled out, with the reason. Stated rather than silently dropped so
    # the reasoner can avoid a whole family of proposals instead of being corrected
    # one at a time.
    excluded: tuple[str, ...] = ()
    # Considerations to weigh more heavily this run -- cost, migration risk, a
    # deadline the graph does not know about.
    emphasis: tuple[str, ...] = ()
    # Free-form framing appended to the step's question.
    note: str | None = None

    def is_empty(self) -> bool:
        return not (self.focus_ids or self.excluded or self.emphasis or self.note)

    def steer(self, step: CascadeStep, triggers: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """The step's question and triggers as this guidance shapes them.

        Returns rather than mutates, so the declared step is never edited: guidance
        is per-run, and a step whose `question` could be rewritten in place would
        carry one person's framing into everybody else's runs.
        """
        if self.focus_ids:
            wanted = set(self.focus_ids)
            focused = [t for t in triggers if t.get("id") in wanted]
            # Empty guidance selection is an explicit no-op, not "fall back to all".
            # Silently widening the scope after somebody narrowed it is the failure
            # mode that would make steering untrustworthy.
            triggers = focused
        parts = [step.question]
        if self.excluded:
            parts.append(
                "Rule out these approaches, and do not propose variants of them: "
                + "; ".join(self.excluded)
            )
        if self.emphasis:
            parts.append(
                "Weigh these considerations especially heavily: "
                + "; ".join(self.emphasis)
            )
        if self.note:
            parts.append(self.note)
        return "\n\n".join(parts), triggers

