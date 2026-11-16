"""Apply a self-improvement suggestion, then measure whether it actually
helped -- the "did it help?" half of the loop
`knowledge_bases.tools.self_improvement` only ever drafted and filed
(read-only) suggestions for.

Per the user's explicit direction (see `docs/architecture/self_improvement.md`),
this is the one exception to Noesis's propose-confirm-execute posture for
every other write: the auto-appliable kinds need no human approval gate.
`architecture`-kind suggestions are the deliberate exception to the
exception: they never reach this module's apply path at all, always routed
to human review instead (see `self_improvement.classify_signal`).

Two application mechanisms, by kind:

- **Graph edits** (`tool_wording`, `routing_capability_wording`,
  `expert_prompt_wording` -- see `knowledge_bases.tools.graph_edits`): the
  skill descriptions and expert definitions being improved *are* graph data,
  so the change is a graph write -- a `Tool.description_override` or a new
  `ExpertRevision` carrying one more overlay item. Nothing in source
  changes, a catalog re-sync cannot clobber it, and every user of the shared
  graph sees it immediately. Routing wording additionally passes a
  vocabulary-pollution guard and a golden routing-eval gate before it is
  written. Revert is another graph write that removes exactly that item.
- **Source edit** (`new_tool`): LLM-authored *logic* has to live in code.
  `source_editor.generate_new_skill` appends one validated handler and its
  catalog registration -- both into the ontology package that owns the tool
  (`knowledge_bases/<kb>/mcp_tools.py` and that package's `TOOLS` seed list),
  so a generated tool ships and installs with its package rather than leaking
  into every product -- `run_test_gate()` runs the full test suite against
  the edited working tree *before* anything is committed (a failure
  discards the edit and marks the finding `"failed"`), and the change is an
  isolated git commit so revert is a plain `git revert`.

Every application snapshots the metric the suggestion was drafted from
*before* applying and re-measures it after a cooldown; if it got worse the
change is reverted automatically. The `ReviewFinding` node is the audit
trail: what was applied, when, the before/after snapshot, the graph-edit
record or commit sha, and the verdict (`kept`, `reverted`, or `failed`).
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from graph import Graph
from knowledge_bases.tools.versioning import (
    changeset_request,
    revise_entity,
    versioning_errors,
    versioning_repository,
)
from knowledge_bases.orchestration.repository import RoutingRepository
from knowledge_bases.orchestration.service import ExpertOrchestrator
from knowledge_bases.tools.git_ops import GitOpsError, commit_files, discard_paths, revert_commit
from knowledge_bases.tools.graph_edits import (
    GRAPH_EDIT_KINDS,
    GraphEditError,
    apply_graph_edit,
    revert_graph_edit,
)
from knowledge_bases.tools.models import SelfImprovementKind, SelfImprovementSuggestion
from knowledge_bases.tools.self_improvement import finding_key
from knowledge_bases.tools.source_editor import REPO_ROOT, SourceEditError, apply_suggestion
from knowledge_bases.tools.trace_store import TraceStore

logger = logging.getLogger(__name__)

SELF_PROJECT_ID = "noesis-self"
_SYSTEM_ACTOR = "self_improvement"

# How long an applied change runs before it's re-measured. Short enough to
# close the loop without waiting indefinitely; long enough to accumulate a
# handful of real calls for skills/knowledge bases with regular traffic.
DEFAULT_COOLDOWN = timedelta(days=3)

# Kinds that ever reach an edit/commit at all. `architecture` is
# deliberately excluded -- `apply_finding` rejects it before doing any
# work (see `apply_finding`'s own check), rather than relying on
# `apply_suggestion`'s downstream raise, so the failure message is clear
# about *why* (routed to human review) rather than "no strategy found".
AUTO_APPLIABLE_KINDS = frozenset({
    SelfImprovementKind.tool_wording,
    SelfImprovementKind.routing_capability_wording,
    SelfImprovementKind.expert_prompt_wording,
    SelfImprovementKind.expert_prompt_prefix,
    SelfImprovementKind.new_tool,
})

# Only `new_tool` carries real, LLM-authored logic (as opposed to a single
# string literal) -- the one kind held to the additional test-gate bar (see
# `run_test_gate`).
TEST_GATED_KINDS = frozenset({SelfImprovementKind.new_tool})

_TEST_GATE_TIMEOUT_SECONDS = 300


class ApplyError(RuntimeError):
    pass


class GateFailure(RuntimeError):
    """Raised by `run_test_gate` when the test suite fails against the
    edited working tree. Carries the captured output so the caller can
    file it on the `ReviewFinding` for a human to see exactly why the
    generated code was rejected."""

    def __init__(self, message: str, output: str):
        super().__init__(message)
        self.output = output


def run_test_gate(timeout: int = _TEST_GATE_TIMEOUT_SECONDS) -> None:
    """Run the full test suite against the current working tree.

    Raises `GateFailure` (never returns a bool the caller could
    forget to check) on a non-zero exit code, a timeout, or an inability
    to even launch pytest -- any of these means the just-generated code
    must not be committed.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GateFailure(
            "Test gate timed out", output=(exc.stdout or "") + (exc.stderr or ""),
        ) from exc
    except OSError as exc:
        raise GateFailure(f"Could not run test gate: {exc}", output="") from exc
    if result.returncode != 0:
        raise GateFailure(
            f"Test suite failed (exit {result.returncode})",
            output=(result.stdout or "") + (result.stderr or ""),
        )


@dataclass(frozen=True)
class MetricSnapshot:
    """A comparable scalar for one signal, plus the raw metrics it was
    computed from -- stored as-is on the `ReviewFinding` (before) and
    recomputed the same way at re-evaluation (after)."""

    kind: SelfImprovementKind
    target: str
    primary_metric: float  # lower is better for both kinds (error/correction rate; low-confidence share)
    raw: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({
            "kind": self.kind.value, "target": self.target,
            "primary_metric": self.primary_metric, "raw": self.raw,
        })


def _skill_snapshot(
    skill: str, since: datetime, store: TraceStore | None = None,
    kind: SelfImprovementKind = SelfImprovementKind.tool_wording,
) -> MetricSnapshot:
    store = store or TraceStore()
    raw = store.skill_metrics_for(skill, since=since.isoformat())
    # Correction rate (per call) folds user_correction_count and error_rate
    # into one comparable scalar without needing two separate thresholds.
    primary = raw["error_rate"] + (
        raw["user_correction_count"] / raw["call_count"] if raw["call_count"] else 0.0
    )
    return MetricSnapshot(kind=kind, target=skill, primary_metric=primary, raw=raw)


def _routing_snapshot(
    knowledge_base: str, since: datetime, repository: RoutingRepository,
) -> MetricSnapshot:
    """Live low-confidence share blended with golden-eval error rate, so a
    change that "helps" ambiguity but misroutes known questions is still
    measured as worse. The eval half is skipped when the repository can't
    supply expert definitions (e.g. a minimal test double)."""
    raw = repository.routing_metrics_for(knowledge_base, since=since.isoformat())
    primary = raw["low_confidence_share"]
    load = getattr(repository, "load_experts", None)
    experts = load() if callable(load) else []
    if experts:
        from knowledge_bases.orchestration.routing_eval import (
            answerable_cases,
            evaluate_routing,
        )

        # Only the cases this product could answer. Here the filter changes the
        # NUMBER, not just its resolution: accuracy is folded into `primary`, which
        # decides whether a wording change is kept or reverted after its cooldown.
        # An unfiltered score on a product that omits packages carries a permanent
        # penalty for experts it never installed, which would revert improvements
        # that actually helped.
        result = evaluate_routing(experts, answerable_cases(experts))
        raw = {**raw, "eval_accuracy": result["accuracy"], "eval_misses": len(result["misses"])}
        primary = 0.5 * primary + 0.5 * (1.0 - result["accuracy"])
    return MetricSnapshot(
        kind=SelfImprovementKind.routing_capability_wording,
        target=knowledge_base, primary_metric=primary, raw=raw,
    )


def _expert_snapshot(
    expert_key: str, since: datetime, store: TraceStore | None = None,
    kind: SelfImprovementKind = SelfImprovementKind.expert_prompt_wording,
) -> MetricSnapshot:
    store = store or TraceStore()
    raw = store.expert_metrics_for(expert_key, since=since.isoformat())
    return MetricSnapshot(
        kind=kind, target=expert_key, primary_metric=raw["correction_share"], raw=raw,
    )


def snapshot_metrics(
    suggestion: SelfImprovementSuggestion, since: datetime,
    store: TraceStore | None = None, repository: RoutingRepository | None = None,
) -> MetricSnapshot:
    if suggestion.kind in (SelfImprovementKind.tool_wording, SelfImprovementKind.new_tool):
        # `new_tool`'s target is the newly generated skill's own key
        # (see `GeneratedToolDraft.key`) -- the pre-apply snapshot is
        # necessarily all zeros (it doesn't exist to have called anything
        # yet), which is the correct baseline: any error/correction at all
        # after apply counts as a regression against "no calls, no
        # failures".
        return _skill_snapshot(suggestion.target, since, store, kind=suggestion.kind)
    if suggestion.kind == SelfImprovementKind.routing_capability_wording:
        if repository is None:
            raise ApplyError("routing_capability_wording snapshot requires a RoutingRepository")
        return _routing_snapshot(suggestion.target, since, repository)
    if suggestion.kind in (
        SelfImprovementKind.expert_prompt_wording,
        SelfImprovementKind.expert_prompt_prefix,
    ):
        # Both change one expert's prompt, so both are judged by that
        # expert's user-correction share.
        return _expert_snapshot(suggestion.target, since, store, kind=suggestion.kind)
    raise ApplyError(f"No metric snapshot strategy for suggestion kind {suggestion.kind!r}")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _current_version(graph: Graph, project_id: str) -> int | None:
    rows = graph.run(
        "MATCH (p:Project {id: $project_id}) RETURN p.graph_version AS v",
        project_id=project_id,
    )
    return int(rows[0]["v"]) if rows else None


def _find_finding(graph: Graph, key: str) -> dict | None:
    rows = graph.run(
        "MATCH (e:BmadEntity:ReviewFinding {project_id: $project_id, key: $key})"
        "-[:CURRENT]->(r:Revision) "
        "RETURN e.id AS entity_id, r.number AS revision_number, properties(r) AS properties",
        project_id=SELF_PROJECT_ID, key=key,
    )
    return rows[0] if rows else None


def _revise_finding(
    repo: Any, finding: dict, properties: dict[str, Any],
) -> None:
    graph = repo.graph
    version = _current_version(graph, SELF_PROJECT_ID)
    if version is None:
        raise ApplyError(f"Project {SELF_PROJECT_ID!r} not seeded yet; run seed_self.py")
    run = repo.start_workflow(SELF_PROJECT_ID, "self_review", _SYSTEM_ACTOR)
    request = changeset_request()(
        project_id=SELF_PROJECT_ID,
        run_id=run.run_id,
        expected_graph_version=version,
        actor=_SYSTEM_ACTOR,
        operations=[
            revise_entity()(
                type="revise_entity",
                entity_id=finding["entity_id"],
                expected_revision=finding["revision_number"],
                properties=properties,
            )
        ],
    )
    repo.submit_changeset(request)


def _suggestion_from_finding(key: str, properties: dict[str, Any]) -> SelfImprovementSuggestion:
    """Reconstruct the `SelfImprovementSuggestion` a `ReviewFinding` was
    filed from, from its own stored properties (see `self_improvement.py`'s
    `file_suggestions()`), so a caller only needs `finding_key` -- not the
    full diagnosis/suggested_change text -- to apply an already-filed
    finding."""
    from knowledge_bases.tools.models import GeneratedToolDraft

    generated_skill = None
    generated_skill_json = properties.get("generated_skill_json")
    if generated_skill_json:
        generated_skill = GeneratedToolDraft.model_validate_json(generated_skill_json)
    return SelfImprovementSuggestion(
        kind=SelfImprovementKind(properties["kind"]),
        target=properties["target"],
        diagnosis=properties.get("message", ""),
        suggested_change=properties.get("suggested_change", ""),
        confidence=properties.get("confidence", 0.5),
        generated_skill=generated_skill,
    )


def apply_finding_by_key(
    key: str, repo: Any | None = None, cooldown: timedelta = DEFAULT_COOLDOWN,
    store: TraceStore | None = None,
) -> dict[str, Any]:
    """Apply an already-filed `ReviewFinding` by its `key` alone (see
    `knowledge_bases.tools.self_improvement.finding_key`), reconstructing
    the suggestion from the finding's own stored properties."""
    repo = repo or versioning_repository()
    finding = _find_finding(repo.graph, key)
    if finding is None:
        raise ApplyError(f"No ReviewFinding for {key!r}; propose_self_improvements first")
    suggestion = _suggestion_from_finding(key, finding["properties"])
    return apply_finding(suggestion, repo=repo, cooldown=cooldown, store=store)


def apply_finding(
    suggestion: SelfImprovementSuggestion,
    repo: Any | None = None,
    cooldown: timedelta = DEFAULT_COOLDOWN,
    store: TraceStore | None = None,
) -> dict[str, Any]:
    """Apply one already-filed suggestion's code change, commit it, and
    record the pre-apply metric snapshot + cooldown deadline on its
    `ReviewFinding` (`status: "applied"`).

    Raises `ApplyError` for an `architecture`-kind suggestion immediately,
    before any work -- it is never a candidate for auto-apply, always
    routed to human review (see `AUTO_APPLIABLE_KINDS`).

    For `new_tool` specifically, the edit is written to disk and the full
    test suite is run against it (`run_test_gate`) *before* any git commit;
    a failure discards the working-tree edit and marks the finding
    `status: "failed"` (not `"applied"`) with the captured failure output,
    rather than leaving broken generated code committed or the finding
    stuck at `"open"` forever with no explanation.

    Otherwise raises `ApplyError` (leaving the finding untouched at
    `status: "open"`) if the finding doesn't exist, is not `"open"`, the
    source edit target can't be found, or there's nothing to commit --
    never partially applies: the source edit, the git commit, and the
    graph revision either all happen or none of them do (a failed commit
    after a successful source edit still leaves `status: "open"`, so a
    later reapply attempt is not blocked by a phantom "applied" state --
    see `test_self_improvement_apply.py` for the exact failure-ordering
    guarantee this relies on).
    """
    if suggestion.kind not in AUTO_APPLIABLE_KINDS:
        raise ApplyError(
            f"{suggestion.kind.value!r} suggestions are never auto-applied; "
            f"this one needs human review (see resolve_review_finding)"
        )

    repo = repo or versioning_repository()
    graph = repo.graph
    key = finding_key(suggestion)
    finding = _find_finding(graph, key)
    if finding is None:
        raise ApplyError(f"No ReviewFinding for {key!r}; propose_self_improvements first")
    if finding["properties"].get("status") != "open":
        raise ApplyError(
            f"ReviewFinding {key!r} is {finding['properties'].get('status')!r}, not 'open'"
        )

    since = _now()
    pre_snapshot = snapshot_metrics(
        suggestion, since=since - timedelta(days=7),
        store=store or TraceStore(graph), repository=RoutingRepository(graph),
    )

    if suggestion.kind in GRAPH_EDIT_KINDS:
        try:
            record = apply_graph_edit(suggestion, graph)
        except (GraphEditError, ValueError) as exc:
            raise ApplyError(f"Could not apply graph edit for {key!r}: {exc}") from exc
        applied_at = _now()
        _revise_finding(repo, finding, {
            "status": "applied",
            "applied_at": applied_at.isoformat(),
            "graph_edit": json.dumps(record),
            "pre_metric_snapshot": pre_snapshot.to_json(),
            "reevaluate_after": (applied_at + cooldown).isoformat(),
        })
        return {"finding_key": key, "graph_edit": record, "applied_at": applied_at.isoformat()}

    try:
        paths = apply_suggestion(suggestion)
    except (SourceEditError, ValueError) as exc:
        raise ApplyError(f"Could not apply source edit for {key!r}: {exc}") from exc

    if suggestion.kind in TEST_GATED_KINDS:
        try:
            run_test_gate()
        except GateFailure as exc:
            discard_paths(paths)
            _revise_finding(repo, finding, {
                "status": "failed",
                "failed_at": _now().isoformat(),
                "failure_reason": str(exc),
                "failure_output": exc.output[-4000:],  # bound what lands in the graph
            })
            raise ApplyError(
                f"Generated code for {key!r} failed the test gate and was discarded "
                f"(not committed): {exc}"
            ) from exc

    try:
        commit_sha = commit_files(
            paths,
            f"self-improve({suggestion.kind.value}): {suggestion.target}\n\n"
            f"{suggestion.diagnosis}\n\nFinding: {key}",
        )
    except GitOpsError as exc:
        discard_paths(paths)
        raise ApplyError(f"Could not commit source edit for {key!r}: {exc}") from exc

    _sync_after_edit()

    applied_at = _now()
    try:
        _revise_finding(repo, finding, {
            "status": "applied",
            "applied_at": applied_at.isoformat(),
            "commit_sha": commit_sha,
            "pre_metric_snapshot": pre_snapshot.to_json(),
            "reevaluate_after": (applied_at + cooldown).isoformat(),
        })
    except versioning_errors() as exc:
        # The code is already committed at this point; the graph record
        # failed to update, but re-running propose/apply is still safe
        # (the finding is still "open" from the graph's point of view) --
        # surface this loudly rather than silently losing the commit sha.
        raise ApplyError(
            f"Applied and committed {commit_sha} for {key!r}, but could not "
            f"record it on the ReviewFinding: {exc}. Commit is NOT reverted; "
            f"investigate and revise the finding manually."
        ) from exc

    return {"finding_key": key, "commit_sha": commit_sha, "applied_at": applied_at.isoformat()}


def _sync_after_edit() -> None:
    """Re-run the catalog/expert sync so the graph matches the just-edited
    source immediately, instead of waiting for the next manual
    `sync_tools_catalog.py`/`sync_noesis_experts` run."""
    from knowledge_bases.tools.catalog import sync_catalog

    sync_catalog()
    ExpertOrchestrator(Graph()).sync_experts()
    _warn_if_exports_stale()


def _warn_if_exports_stale() -> None:
    """Log when a generated harness file no longer matches the graph.

    Expert overlays, capabilities, skill wording, and the whole context plane
    are fetched live from the graph, so a wording self-improvement needs no
    re-export (see `knowledge_bases.orchestration.prime_export`). A change to
    the orchestrator's *base* prompt in `experts_catalog.py` still does, and
    this is the only thing that says so. Never raises: a stale export is a
    warning, not a reason to fail an apply.
    """
    try:
        from knowledge_bases.orchestration.prime_export import stale_exports

        with Graph() as graph:
            experts = RoutingRepository(graph).load_experts(include_unroutable=True)
        for warning in stale_exports(experts, REPO_ROOT):
            logger.warning("Prime context export is stale: %s", warning)
    except Exception:  # noqa: BLE001 -- advisory only
        logger.debug("Could not check harness export freshness", exc_info=True)


STALE_LOOKBACK = timedelta(days=7)
# A finding needs at least this many calls in the recent window before its
# recovery is believable. One lucky call is not evidence that a 40%% error rate
# is gone.
STALE_MIN_CALLS = 5
# The metric a target must fall back under to count as recovered. Deliberately
# below `ERROR_RATE_THRESHOLD` (0.2), so a target hovering exactly at the
# filing threshold is not retired and immediately re-filed.
STALE_RECOVERY_METRIC = 0.1


def _open_findings(graph: Graph) -> list[dict]:
    """Every `open` self-improvement finding, newest first."""
    return graph.run(
        "MATCH (e:BmadEntity:ReviewFinding {project_id: $project_id})-[:CURRENT]->(r:Revision) "
        "WHERE r.status = 'open' AND e.key STARTS WITH 'self_improvement::' "
        "RETURN e.id AS entity_id, e.key AS key, r.number AS revision_number, "
        "properties(r) AS properties "
        "ORDER BY r.filed_at DESC",
        project_id=SELF_PROJECT_ID,
    )


def _is_registered_tool(graph: Graph, target: str) -> bool:
    rows = graph.run(
        "MATCH (t:Tool {key: $key}) RETURN count(t) AS c", key=target,
    )
    return bool(rows and rows[0]["c"])


def retire_stale_findings(
    repo: Any | None = None,
    store: TraceStore | None = None,
    routing_repository: RoutingRepository | None = None,
    lookback: timedelta = STALE_LOOKBACK,
) -> list[dict[str, Any]]:
    """Close `open` findings whose underlying signal no longer holds.

    A finding is a claim about the present ("this tool is failing"), but
    nothing ever re-checked it, so findings accumulated as permanent
    accusations. Two ways that claim stops being true:

    - **The target is gone.** The failures sit under a skill key that is no
      longer registered -- `hybrid_search`'s 100%% error rate is five spans from
      a renamed key (it is `orchestration.hybrid_search` now), and no traffic
      will ever arrive under the old name to dilute it. That metric is frozen
      at 100%% forever; only retirement can clear it.
    - **The target recovered.** Enough recent calls, and the metric is back
      under `STALE_RECOVERY_METRIC`.

    Only `open` findings are considered. An `applied` finding is already owned
    by `reevaluate_applied_findings`, which can `git revert` it -- retiring one
    here would strand a real source edit with nothing tracking it.

    Never raises for one finding: a finding that cannot be evaluated is
    reported with `"error"` and left open, which is the safe direction.
    """
    repo = repo or versioning_repository()
    graph = repo.graph
    store = store or TraceStore(graph)
    routing_repository = routing_repository or RoutingRepository(graph)
    since = _now() - lookback

    results = []
    for finding in _open_findings(graph):
        properties = finding["properties"]
        target = properties.get("target")
        key = finding["key"]
        try:
            kind = SelfImprovementKind(properties["kind"])
        except (KeyError, ValueError):
            results.append({"finding_key": key, "verdict": "skipped",
                            "error": "unrecognized kind"})
            continue

        try:
            reason = _stale_reason(
                kind, target, since, graph, store, routing_repository,
            )
        except Exception as exc:  # noqa: BLE001 - one bad finding must not stop the rest
            results.append({"finding_key": key, "verdict": "skipped", "error": str(exc)})
            continue

        if reason is None:
            results.append({"finding_key": key, "verdict": "still_valid"})
            continue

        _revise_finding(repo, finding, {
            "status": "retired",
            "retired_at": _now().isoformat(),
            "retired_reason": reason,
        })
        results.append({"finding_key": key, "verdict": "retired", "reason": reason})

    return results


def _stale_reason(
    kind: SelfImprovementKind, target: str | None, since: datetime,
    graph: Graph, store: TraceStore, routing_repository: RoutingRepository,
) -> str | None:
    """Why this finding no longer holds, or `None` if it still does."""
    if not target:
        return None

    # Routing findings are about a knowledge base, which does not get renamed
    # out from under us the way a tool key does, so only recovery applies.
    if kind == SelfImprovementKind.routing_capability_wording:
        raw = routing_repository.routing_metrics_for(target, since=since.isoformat())
        if raw.get("trace_count", 0) < STALE_MIN_CALLS:
            return None
        if raw.get("low_confidence_share", 1.0) <= STALE_RECOVERY_METRIC:
            return (
                f"Routing for {target} recovered: low-confidence share "
                f"{raw['low_confidence_share']:.0%} over {raw['trace_count']} traces."
            )
        return None

    # Everything else is keyed by a tool key.
    if not _is_registered_tool(graph, target):
        return (
            f"{target} is no longer a registered tool, so its error rate cannot "
            "change and the finding can never resolve on its own."
        )

    raw = store.skill_metrics_for(target, since=since.isoformat())
    if raw.get("call_count", 0) < STALE_MIN_CALLS:
        return None
    metric = raw["error_rate"] + (
        raw.get("user_correction_count", 0) / raw["call_count"]
    )
    if metric <= STALE_RECOVERY_METRIC:
        return (
            f"{target} recovered: {metric:.0%} error/correction rate over "
            f"{raw['call_count']} recent calls."
        )
    return None


def _due_applied_findings(graph: Graph) -> list[dict]:
    now = _now().isoformat()
    return graph.run(
        "MATCH (e:BmadEntity:ReviewFinding {project_id: $project_id})-[:CURRENT]->(r:Revision) "
        "WHERE r.status = 'applied' AND r.reevaluate_after <= $now "
        "RETURN e.id AS entity_id, e.key AS key, r.number AS revision_number, "
        "properties(r) AS properties",
        project_id=SELF_PROJECT_ID, now=now,
    )


def reevaluate_applied_findings(
    repo: Any | None = None,
    store: TraceStore | None = None,
    routing_repository: RoutingRepository | None = None,
) -> list[dict[str, Any]]:
    """Re-measure every `"applied"` finding whose cooldown has elapsed,
    and decide `kept` (metric held or improved) vs. `reverted` (metric got
    worse -- `git revert` the applied commit, re-sync, and mark the
    finding `"reverted"`).

    Returns one result dict per finding processed this run (empty if none
    are due). Never raises for one finding's failure; a finding that can't
    be safely evaluated/reverted is skipped with `"error"` set, so one bad
    finding doesn't block re-evaluating the rest.
    """
    repo = repo or versioning_repository()
    graph = repo.graph
    store = store or TraceStore(graph)
    routing_repository = routing_repository or RoutingRepository(graph)

    results = []
    for finding in _due_applied_findings(graph):
        properties = finding["properties"]
        try:
            pre = json.loads(properties["pre_metric_snapshot"])
            kind = SelfImprovementKind(pre["kind"])
            target = pre["target"]
            applied_at = datetime.fromisoformat(properties["applied_at"])

            suggestion = SelfImprovementSuggestion(
                kind=kind, target=target, diagnosis="", suggested_change="",
            )
            post = snapshot_metrics(
                suggestion, since=applied_at, store=store, repository=routing_repository,
            )

            improved_or_held = post.primary_metric <= pre["primary_metric"]
            if improved_or_held:
                _revise_finding(repo, finding, {
                    "status": "kept",
                    "post_metric_snapshot": post.to_json(),
                    "verdict_at": _now().isoformat(),
                })
                results.append({
                    "finding_key": finding["key"], "verdict": "kept",
                    "pre": pre["primary_metric"], "post": post.primary_metric,
                })
            elif properties.get("graph_edit"):
                reverted = revert_graph_edit(json.loads(properties["graph_edit"]), graph)
                _revise_finding(repo, finding, {
                    "status": "reverted",
                    "post_metric_snapshot": post.to_json(),
                    "graph_revert": json.dumps(reverted),
                    "verdict_at": _now().isoformat(),
                })
                results.append({
                    "finding_key": finding["key"], "verdict": "reverted",
                    "pre": pre["primary_metric"], "post": post.primary_metric,
                    "graph_revert": reverted,
                })
            else:
                revert_sha = revert_commit(properties["commit_sha"])
                _sync_after_edit()
                _revise_finding(repo, finding, {
                    "status": "reverted",
                    "post_metric_snapshot": post.to_json(),
                    "revert_commit_sha": revert_sha,
                    "verdict_at": _now().isoformat(),
                })
                results.append({
                    "finding_key": finding["key"], "verdict": "reverted",
                    "pre": pre["primary_metric"], "post": post.primary_metric,
                    "revert_commit_sha": revert_sha,
                })
        except Exception as exc:  # noqa: BLE001 -- one bad finding must not block the rest
            results.append({"finding_key": finding.get("key", "?"), "verdict": "error", "error": str(exc)})
    return results
