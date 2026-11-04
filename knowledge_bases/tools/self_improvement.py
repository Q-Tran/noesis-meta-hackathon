"""Self-improvement: turn execution traces into reviewable suggestions.

This closes the loop the rest of `knowledge_bases.tools` only collects data
for: `SpanEvent`/`TraceOutcome` (DuckDB) and `RoutingTrace` (Neo4j) already
tell you *that* a skill keeps failing or a knowledge base keeps getting
routed to with low confidence. Nothing previously read that data back to ask
"should a skill's wording, schema, or a knowledge base's routing capability
description change?" This module does, then drafts a concrete suggestion
with an LLM (same `BaseLLM.extract()` pattern as
`knowledge_bases.epistemology.feedback.analyze_feedback`) and files it as a
`ReviewFinding` in the reflexive `noesis-self` methodology project -- mirroring
`scripts/self_recompute.py`'s architecture-drift loop, but for skill/routing quality
instead of code-vs-rationale drift.

Nothing here ever edits `knowledge_bases/tools/catalog.py`,
`knowledge_bases/orchestration/experts_catalog.py`, or any other source file
automatically. A suggestion is a graph `ReviewFinding` for a human (or a
architecture-phase workflow conversation) to accept, edit, or reject -- the same
propose-confirm-execute posture as every other Noesis write
(`docs/architecture/knowledge_write_orchestration.md`).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from knowledge_bases.tools.versioning import (
    changeset_request,
    create_entity,
    versioning_errors,
    versioning_repository,
)
from graph import Graph
from knowledge_bases.orchestration.repository import RoutingRepository
from knowledge_bases.tools.models import (
    SearchRetrySignal,
    GeneratedToolDraft,
    RoutingGapSignal,
    SelfImprovementAnalysis,
    SelfImprovementClassification,
    SelfImprovementDraft,
    SelfImprovementKind,
    SelfImprovementSuggestion,
    ToolHealthSignal,
)
from knowledge_bases.tools.models import ExpertCorrectionSignal
from knowledge_bases.tools.trace_store import TraceStore
from llm import BaseLLM, get_llm

SELF_PROJECT_ID = "noesis-self"

logger = logging.getLogger(__name__)
_SYSTEM_ACTOR = "self_improvement"

# A skill/knowledge base only becomes a candidate once it has enough traffic
# to distinguish a real pattern from noise (a single failed call proves
# nothing). Thresholds below (min error rate, min low-confidence share) are
# then applied on top of that minimum sample size.
DEFAULT_LOOKBACK = timedelta(days=7)
MIN_CALLS = 5
MIN_TRACES = 5
ERROR_RATE_THRESHOLD = 0.2
USER_CORRECTION_THRESHOLD = 2
LOW_CONFIDENCE_SHARE_THRESHOLD = 0.4
EXPERT_CORRECTION_SHARE_THRESHOLD = 0.3

# How many times one trace may call the same search tool before that counts as
# a retry loop. Two is a legitimate refinement ("broaden the entity types");
# three or more means the caller could not express what it wanted, which is a
# problem with the *tool's* wording or the expert's routing vocabulary rather
# than with the caller.
SEARCH_RETRY_THRESHOLD = 3
# How many separate traces must show the pattern before it is a signal rather
# than one user having an unusual day.
SEARCH_RETRY_MIN_TRACES = 2


def collect_skill_health_signals(
    store: TraceStore, since: datetime, min_calls: int = MIN_CALLS,
) -> list[ToolHealthSignal]:
    """Skills whose error rate or correction count crosses the threshold."""
    signals = []
    for row in store.skill_health(since=since, min_calls=min_calls):
        if row["error_rate"] < ERROR_RATE_THRESHOLD and (
            row["user_correction_count"] or 0
        ) < USER_CORRECTION_THRESHOLD:
            continue
        signals.append(ToolHealthSignal.model_validate(row))
    return signals


def collect_search_retry_signals(
    store: TraceStore, since: datetime,
    min_calls: int = SEARCH_RETRY_THRESHOLD,
    min_traces: int = SEARCH_RETRY_MIN_TRACES,
) -> list[SearchRetrySignal]:
    """Retrieval tools callers had to invoke repeatedly within one turn.

    The third kind of failure, alongside errors and user corrections: the call
    succeeded, nobody complained, and the caller still went round again. Two
    calls is legitimate refinement; three or more in one trace means they could
    not express what they wanted.
    """
    return [
        SearchRetrySignal.model_validate(row)
        for row in store.search_retry_loops(
            since=since, min_calls=min_calls, min_traces=min_traces,
        )
    ]


def collect_routing_gap_signals(
    repository: RoutingRepository, min_traces: int = MIN_TRACES,
) -> list[RoutingGapSignal]:
    """Knowledge bases whose routing traces are disproportionately
    low-confidence/clarification-needed -- a capability-wording gap."""
    signals = []
    for row in repository.routing_gaps(min_traces=min_traces):
        total = row["low_confidence_count"] + row["clarification_needed_count"]
        # routing_gaps() already filters to traces that were low-confidence
        # or needed clarification; a further share check against min_traces
        # avoids flagging a knowledge base with just one or two rough edges.
        if total < min_traces * LOW_CONFIDENCE_SHARE_THRESHOLD:
            continue
        signals.append(RoutingGapSignal.model_validate(row))
    return signals


def collect_expert_correction_signals(
    store: TraceStore, since: datetime, min_traces: int = MIN_TRACES,
) -> list[ExpertCorrectionSignal]:
    """Graph-defined experts whose traces the user keeps correcting even
    though no single skill stands out -- the expert's prompt guidance is the
    likely lever. Requires both enough traffic and a meaningful share."""
    signals = []
    for row in store.expert_corrections(since=since):
        if row["trace_count"] < min_traces:
            continue
        if (
            row["corrected_trace_count"] < USER_CORRECTION_THRESHOLD
            or row["correction_share"] < EXPERT_CORRECTION_SHARE_THRESHOLD
        ):
            continue
        signals.append(ExpertCorrectionSignal.model_validate(row))
    return signals


_URL_RE = re.compile(r"https?://\S+")
# Bare internal hostnames -- the part before the path, which `_URL_RE` misses
# when the error prints a host without a scheme.
_HOST_RE = re.compile(r"\b[\w-]+(?:\.[\w-]+){2,}\.(?:com|net|org|io|internal|local)\b")


def redact_for_prompt(text: str) -> str:
    """Strip URLs and internal hostnames before sending text to the LLM.

    Not cosmetic. The corporate LLM gateway rejects prompts containing URLs
    and internal hostnames outright -- 401 or 500, before the model ever sees
    them -- so a signal whose sample errors embed a Snowflake privatelink host
    (the single most common connectivity error here) could never be diagnosed
    at all. The host adds nothing to a diagnosis anyway: that a connection
    failed is the signal, not which box it failed against.
    """
    text = _URL_RE.sub("<url>", text)
    return _HOST_RE.sub("<host>", text)


def _draft_skill_diagnosis(signal: ToolHealthSignal, llm: BaseLLM) -> SelfImprovementDraft:
    """The first, always-run LLM call for a skill health signal: diagnose
    the likely cause and, if it's plausibly a wording/schema fix, draft the
    exact replacement text. This never decides on its own whether a
    non-wording cause needs a new skill or a structural change -- that's
    `classify_signal`'s job, run only when this returns no usable wording
    fix (see `draft_suggestions`)."""
    prompt = f"""A graph-registered MCP skill has a concerning execution record.

Tool: {signal.skill}
Knowledge base: {signal.knowledge_base}
Calls in window: {signal.call_count}
Error rate: {signal.error_rate:.0%}
Traces where the user later corrected/rejected the result: {signal.user_correction_count}
Sample errors: {redact_for_prompt(str(signal.sample_errors))}

Diagnose the likely cause (ambiguous description, missing/misdocumented
input_schema field, wrong handler logic, a missing capability entirely, or a
real upstream data problem). If the fix is plausibly a wording/schema change
to the Tool's `description` or `input_schema` (see
`knowledge_bases/tools/catalog.py`), draft the exact replacement text. If
the cause looks like anything else -- a logic bug, a missing capability, a
structural change -- say so in the diagnosis and leave suggested_change
empty; do not guess at code changes here.
"""
    return llm.extract(prompt, SelfImprovementDraft)


def classify_signal(
    skill: str, knowledge_base: str, diagnosis: str, llm: BaseLLM,
) -> SelfImprovementClassification:
    """Given a diagnosis that isn't a simple wording fix, classify the
    shape of code change actually needed:

    - `new_tool`: a self-contained new capability, expressible as one new
      handler function appended to this knowledge base's existing
      `mcp_tools.py` -- no new file, no new Neo4j labels/relationships, no
      change to more than one function.
    - `architecture`: anything wider than that -- a new knowledge base
      ("paradigm"), a new Neo4j label/relationship, a change spanning
      multiple files/packages, or a fix to existing handler logic (not
      just a missing capability). Also the safe default: if the diagnosis
      doesn't clearly describe a narrow, self-contained new capability,
      classify as `architecture` rather than guessing narrower.

    Defaults to `architecture` (see `SelfImprovementClassification`) on any
    unparseable LLM output -- never silently defaults to something that
    would auto-apply.
    """
    prompt = f"""A skill's execution problem has been diagnosed as NOT a
simple wording/schema fix. Classify what shape of code change would
actually address it.

Tool: {skill}
Knowledge base: {knowledge_base}
Diagnosis: {diagnosis}

Choose exactly one:
- "new_tool": the fix is a missing, self-contained capability that could
  be added as ONE new handler function in this knowledge base's existing
  mcp_tools.py file -- no new file, no new package, no new Neo4j label or
  relationship, no change to any other existing function.
- "architecture": anything else -- a new knowledge base, a new graph
  schema element, a change spanning multiple files, or a fix to this
  skill's own existing logic (not a missing capability).

If in doubt, choose "architecture" -- it is always safe to route something
to human review that could have been handled narrower, but never safe to
guess narrower than it actually is.
"""
    return llm.extract(prompt, SelfImprovementClassification)


def _draft_new_skill_suggestion(
    signal: ToolHealthSignal, diagnosis: str, llm: BaseLLM,
) -> SelfImprovementSuggestion:
    prompt = f"""Draft ONE new, self-contained MCP skill/handler to appended
to `knowledge_bases/{signal.knowledge_base}/mcp_tools.py` and registered in
`knowledge_bases/tools/catalog.py`, to address this diagnosed gap:

Tool this was diagnosed from: {signal.skill}
Knowledge base: {signal.knowledge_base}
Diagnosis: {diagnosis}

Requirements for `handler_source`:
- A single top-level Python function, signature exactly
  `def <name>(arguments: dict, graph: Graph) -> Any:`.
- Only use `graph.run(cypher, **params)` for any graph read -- no writes, no
  imports beyond what a typical read-only query handler needs (`Any` from
  typing, `Graph` from the `graph` module are already imported in the file).
- `key` must be `"{signal.knowledge_base}.<short_name>"`; `handler_function_name`
  must exactly match the function name used in `handler_source`.
- Keep the query narrowly scoped to what the diagnosis actually calls for.

If you cannot draft something this narrow and self-contained, leave
`handler_source` empty -- do not draft something wider than one function.
"""
    draft = llm.extract(prompt, GeneratedToolDraft)
    return SelfImprovementSuggestion(
        kind=SelfImprovementKind.new_tool,
        target=draft.key or signal.skill,
        diagnosis=diagnosis,
        confidence=draft.confidence,
        generated_skill=draft,
    )


_SEARCH_RETRY_PROMPTS = {
    # The level of failure decides which wording is wrong, so each level gets
    # its own framing rather than one generic "improve this description".
    "no_results": (
        "Callers invoked the retrieval tool {skill} {total_calls} times across "
        "{retry_traces} separate turns, and {empty_calls} of those calls returned "
        "NOTHING. Nothing errored; they simply could not find what they wanted. "
        "That means the tool's description misdescribes what it actually covers, "
        "so callers reach for it with the wrong vocabulary.\n\n"
        "Sample inputs: {sample_inputs}\n\n"
        "Current description: {description}\n\n"
        "Rewrite the description so a caller can tell what this tool does and "
        "does not cover. Reply with the description only."
    ),
    "wrong_results": (
        "Callers invoked the retrieval tool {skill} {total_calls} times across "
        "{retry_traces} separate turns. The calls returned results, and callers "
        "searched again anyway -- so they got the wrong KIND of answer, not no "
        "answer. That points at the tool's stated scope (what entities it "
        "searches, what question it answers) rather than its index.\n\n"
        "Sample inputs: {sample_inputs}\n\n"
        "Current description: {description}\n\n"
        "Rewrite the description to make its scope unambiguous. Reply with the "
        "description only."
    ),
}


def _draft_search_retry_suggestion(
    signal: SearchRetrySignal, llm: BaseLLM, description: str = "",
) -> SelfImprovementSuggestion | None:
    """Turn a retry loop into a wording fix aimed at the level that failed.

    Always `tool_wording`: a retry loop says callers misunderstood what the
    tool covers, which is a description problem. If it were a missing
    capability the calls would error, and `ToolHealthSignal` would have caught
    it instead.
    """
    prompt = _SEARCH_RETRY_PROMPTS[signal.failure_level].format(
        skill=signal.skill,
        total_calls=signal.total_calls,
        retry_traces=signal.retry_traces,
        empty_calls=signal.empty_calls,
        sample_inputs=str(signal.sample_inputs)[:500],
        description=description or "(unavailable)",
    )
    try:
        suggested = llm.complete(prompt).strip()
    except Exception:
        return None
    if not suggested:
        return None

    return SelfImprovementSuggestion(
        kind=SelfImprovementKind.tool_wording,
        target=signal.skill,
        diagnosis=(
            f"{signal.retry_traces} turns needed {signal.total_calls} calls to "
            f"{signal.skill} ({signal.empty_calls} returned nothing): "
            f"{signal.failure_level.replace('_', ' ')}."
        ),
        suggested_change=suggested,
        confidence=0.6,
    )


def _draft_skill_wording_suggestion(
    signal: ToolHealthSignal, llm: BaseLLM,
) -> SelfImprovementSuggestion | None:
    """Diagnose a skill health signal, then, if it's not a simple wording
    fix, classify and (if narrow enough) generate a new skill -- otherwise
    file it as `architecture` for human review. Returns `None` only when
    the LLM produced no usable diagnosis at all (see `draft_suggestions`).
    """
    draft = _draft_skill_diagnosis(signal, llm)
    if draft.suggested_change.strip():
        return SelfImprovementSuggestion(
            kind=SelfImprovementKind.tool_wording,
            target=signal.skill,
            diagnosis=draft.diagnosis,
            suggested_change=draft.suggested_change,
            confidence=draft.confidence,
        )
    if not draft.diagnosis.strip():
        return None

    classification = classify_signal(
        signal.skill, signal.knowledge_base, draft.diagnosis, llm,
    )
    if classification.kind == SelfImprovementKind.new_tool:
        suggestion = _draft_new_skill_suggestion(signal, classification.diagnosis or draft.diagnosis, llm)
        generated = suggestion.generated_skill
        # A `new_tool` finding whose draft never actually produced usable
        # handler source is worse than useless -- it would only ever fail
        # `apply_self_improvement` later. Fall back to `architecture` so a
        # human still sees the diagnosis, instead of filing a suggestion
        # nothing can ever act on.
        if generated is not None and generated.handler_source.strip() and generated.key.strip():
            return suggestion
        classification = SelfImprovementClassification(
            kind=SelfImprovementKind.architecture,
            diagnosis=classification.diagnosis or draft.diagnosis,
            confidence=classification.confidence,
        )

    return SelfImprovementSuggestion(
        kind=SelfImprovementKind.architecture,
        target=signal.skill,
        diagnosis=classification.diagnosis or draft.diagnosis,
        confidence=classification.confidence,
    )


def _draft_routing_wording_suggestion(
    signal: RoutingGapSignal, llm: BaseLLM,
) -> SelfImprovementSuggestion:
    prompt = f"""A graph-defined expert is frequently routed to with low confidence
or requires user clarification, suggesting its capability phrases (the
vocabulary routing matches a question against) don't match how questions
are actually phrased.

Expert: {signal.knowledge_base}
Low-confidence routing traces: {signal.low_confidence_count}
Clarification-needed traces: {signal.clarification_needed_count}
Sample question summaries actually asked:
{signal.sample_questions}

Diagnose what vocabulary or phrasing these questions use that the current
capability wording likely does not cover. Draft ONE additional capability
phrase: short, keyword-dense, and strictly about THIS expert's own domain --
never vocabulary that belongs to another expert (business terms, metrics,
claims, decisions, methodology workflow, project management), since the phrase is rejected
if it reads more like a different expert.
"""
    draft = llm.extract(prompt, SelfImprovementDraft)
    return SelfImprovementSuggestion(
        kind=SelfImprovementKind.routing_capability_wording,
        target=signal.knowledge_base,
        diagnosis=draft.diagnosis,
        suggested_change=draft.suggested_change,
        confidence=draft.confidence,
    )


def _draft_expert_prompt_suggestion(
    signal: ExpertCorrectionSignal, llm: BaseLLM,
) -> SelfImprovementSuggestion:
    prompt = f"""Users keep correcting answers produced by one graph-defined expert
even though its individual tools mostly succeed, which points at the
expert's own working guidance (its system prompt) rather than a tool.

Expert: {signal.expert_key}
Traces it acted in: {signal.trace_count}
Traces the user later corrected: {signal.corrected_trace_count} ({signal.correction_share:.0%})
Correction reasons the users gave:
{signal.sample_reasons}
Tools it used in those traces: {signal.skills}

Diagnose the recurring mistake. Then draft ONE short paragraph (2-4
sentences, imperative voice) to append to this expert's prompt that would
prevent it -- a concrete rule about what to check, call, or state, not a
generic "be careful". Leave suggested_change empty if the corrections don't
point at a prompt-level fix.
"""
    draft = llm.extract(prompt, SelfImprovementDraft)
    return SelfImprovementSuggestion(
        kind=SelfImprovementKind.expert_prompt_wording,
        target=signal.expert_key,
        diagnosis=draft.diagnosis,
        suggested_change=draft.suggested_change,
        confidence=draft.confidence,
    )


def _safe_draft(drafter, signal, llm: BaseLLM, label: str):
    """Draft one suggestion, returning `None` if the LLM call fails.

    Drafting is one LLM round-trip per signal, against a gateway that fails
    transiently. Letting that propagate means the whole pass files nothing,
    so the signals that drafted cleanly are lost along with the one that
    didn't.
    """
    try:
        return drafter(signal, llm)
    except Exception:
        logger.warning("Could not draft a suggestion for %s", label, exc_info=True)
        return None


def draft_suggestions(
    skill_signals: list[ToolHealthSignal],
    routing_signals: list[RoutingGapSignal],
    llm: BaseLLM | None = None,
    expert_signals: list[ExpertCorrectionSignal] | None = None,
    search_signals: list[SearchRetrySignal] | None = None,
) -> SelfImprovementAnalysis:
    llm = llm or get_llm()
    # One signal that cannot be drafted must not discard the others. A pass
    # analyzing six signals previously died on its second and filed nothing,
    # losing the four that were fine.
    skill_suggestions = [
        _safe_draft(_draft_skill_wording_suggestion, signal, llm, signal.skill)
        for signal in skill_signals
    ]
    routing_suggestions = [
        s for s in (
            _safe_draft(
                _draft_routing_wording_suggestion, signal, llm, signal.knowledge_base,
            )
            for signal in routing_signals
        )
        if s is not None
    ]
    expert_suggestions = [
        s for s in (
            _safe_draft(_draft_expert_prompt_suggestion, signal, llm, signal.expert_key)
            for signal in expert_signals or []
        )
        if s is not None and s.suggested_change.strip()
    ]
    search_suggestions = [
        s for s in (
            _safe_draft(_draft_search_retry_suggestion, signal, llm, signal.skill)
            for signal in search_signals or []
        )
        if s is not None
    ]
    suggestions = (
        [s for s in skill_suggestions if s is not None]
        + routing_suggestions + expert_suggestions + search_suggestions
    )
    return SelfImprovementAnalysis(suggestions=[s for s in suggestions if s.diagnosis.strip()])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_version(graph: Graph, project_id: str) -> int | None:
    rows = graph.run(
        "MATCH (p:Project {id: $project_id}) RETURN p.graph_version AS v",
        project_id=project_id,
    )
    return int(rows[0]["v"]) if rows else None


def _existing_finding_keys(graph: Graph) -> set[str]:
    rows = graph.run(
        "MATCH (f:BmadEntity:ReviewFinding {project_id: $project_id}) RETURN f.key AS key",
        project_id=SELF_PROJECT_ID,
    )
    return {row["key"] for row in rows}


def finding_key(suggestion: SelfImprovementSuggestion) -> str:
    """The `ReviewFinding.key` for one suggestion -- exported (not a
    private `_finding_key`) so `knowledge_bases.tools.self_improvement_apply`
    can compute the exact same key to look the finding this module filed
    back up, rather than duplicating the format string and risking drift."""
    return f"self_improvement::{suggestion.kind.value}::{suggestion.target}"


def file_suggestions(
    suggestions: list[SelfImprovementSuggestion], repo: Any | None = None,
) -> list[SelfImprovementSuggestion]:
    """File a `ReviewFinding` per new suggestion in the `noesis-self` methodology
    project. Idempotent by `(kind, target)`: a knowledge base or skill
    already flagged is skipped, not re-filed or overwritten, until a human
    resolves the existing finding (same posture as `scripts/self_recompute.py`).
    Never mutates the skill/capability catalog itself.
    """
    if not suggestions:
        return []
    repo = repo or versioning_repository()
    graph = repo.graph

    version = _current_version(graph, SELF_PROJECT_ID)
    if version is None:
        print(f"[self_improvement] project {SELF_PROJECT_ID!r} not seeded yet; run seed_self.py")
        return []

    already_flagged = _existing_finding_keys(graph)
    to_file = [s for s in suggestions if finding_key(s) not in already_flagged]
    if not to_file:
        print("[self_improvement] no new suggestions to file")
        return []

    run = repo.start_workflow(SELF_PROJECT_ID, "self_review", _SYSTEM_ACTOR)
    filed: list[SelfImprovementSuggestion] = []

    for suggestion in to_file:
        version = _current_version(graph, SELF_PROJECT_ID)
        properties: dict[str, Any] = {
            # `architecture`-kind findings are never auto-applied and always
            # need a human decision -- `blocking` makes that visible and
            # gated (see `knowledge_bases/methodology/validation.py`'s build-phase
            # check), unlike the routine `advisory` findings the other two
            # kinds file (those DO auto-apply and self-correct via
            # `self_improvement_apply.reevaluate_applied_findings`, so a
            # human is never required to act on them).
            "severity": (
                "blocking" if suggestion.kind == SelfImprovementKind.architecture
                else "advisory"
            ),
            "status": "open",
            "kind": suggestion.kind.value,
            "target": suggestion.target,
            "message": suggestion.diagnosis,
            "suggested_change": suggestion.suggested_change,
            "confidence": suggestion.confidence,
            "filed_at": _now(),
        }
        if suggestion.generated_skill is not None:
            properties["generated_skill_json"] = suggestion.generated_skill.model_dump_json()
        request = changeset_request()(
            project_id=SELF_PROJECT_ID,
            run_id=run.run_id,
            expected_graph_version=version,
            actor=_SYSTEM_ACTOR,
            operations=[
                create_entity()(
                    type="create_entity",
                    entity_id=str(uuid4()),
                    entity_type="ReviewFinding",
                    key=finding_key(suggestion),
                    properties=properties,
                )
            ],
        )
        try:
            repo.submit_changeset(request)
            filed.append(suggestion)
        except versioning_errors() as exc:
            print(f"[self_improvement] skipped {finding_key(suggestion)}: {exc}")

    return filed


def run_self_improvement(
    graph: Graph | None = None,
    llm: BaseLLM | None = None,
    lookback: timedelta = DEFAULT_LOOKBACK,
) -> list[SelfImprovementSuggestion]:
    """End-to-end pass: gather signals, draft suggestions, file findings.

    Returns the suggestions that were newly filed this run (empty if
    nothing crosses a threshold, or the project isn't seeded yet).
    """
    graph = graph or Graph()
    since = datetime.now(timezone.utc) - lookback

    store = TraceStore(graph)
    skill_signals = collect_skill_health_signals(store, since)
    routing_signals = collect_routing_gap_signals(RoutingRepository(graph))
    expert_signals = collect_expert_correction_signals(store, since)
    search_signals = collect_search_retry_signals(store, since)
    if not skill_signals and not routing_signals and not expert_signals \
            and not search_signals:
        print("[self_improvement] no signals crossed the threshold")
        return []

    analysis = draft_suggestions(
        skill_signals, routing_signals, llm,
        expert_signals=expert_signals, search_signals=search_signals,
    )
    return file_suggestions(analysis.suggestions, versioning_repository(graph))
