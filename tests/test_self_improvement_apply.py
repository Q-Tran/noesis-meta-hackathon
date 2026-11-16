"""`knowledge_bases.tools.self_improvement_apply` is the "did it help?"
half of the self-improvement loop: apply a filed suggestion's code change
(committed to git), snapshot the metric it was drafted from, and later
auto-revert if a re-measurement shows it made things worse. Unlike every
other Noesis write, this is fully automatic by explicit design (see the
module's own docstring and `docs/architecture/self_improvement.md`)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from knowledge_bases.tools.models import SelfImprovementKind, SelfImprovementSuggestion
from knowledge_bases.tools.self_improvement import finding_key
from knowledge_bases.tools.self_improvement_apply import (
    ApplyError,
    apply_finding,
    apply_finding_by_key,
    reevaluate_applied_findings,
    snapshot_metrics,
)

SKILL_SUGGESTION = SelfImprovementSuggestion(
    kind=SelfImprovementKind.tool_wording,
    target="business_dictionary.lookup_business_term",
    diagnosis="Ambiguous description.",
    suggested_change="A clearer description.",
    confidence=0.7,
)
ROUTING_SUGGESTION = SelfImprovementSuggestion(
    kind=SelfImprovementKind.routing_capability_wording,
    target="methodology",
    diagnosis="Low confidence routing.",
    suggested_change="a new capability phrase",
    confidence=0.7,
)


# -- snapshot_metrics ----------------------------------------------------------

class FakeStore:
    def __init__(self, metrics):
        self.metrics = metrics
        self.calls = []

    def skill_metrics_for(self, skill, since):
        self.calls.append((skill, since))
        return self.metrics


class FakeRoutingRepository:
    def __init__(self, metrics):
        self.metrics = metrics
        self.calls = []

    def routing_metrics_for(self, knowledge_base, since):
        self.calls.append((knowledge_base, since))
        return self.metrics


def test_snapshot_metrics_combines_error_rate_and_correction_rate_for_a_skill():
    store = FakeStore({
        "call_count": 10, "error_count": 2, "error_rate": 0.2, "user_correction_count": 1,
    })

    snapshot = snapshot_metrics(SKILL_SUGGESTION, since=datetime.now(timezone.utc), store=store)

    assert snapshot.kind == SelfImprovementKind.tool_wording
    assert snapshot.target == "business_dictionary.lookup_business_term"
    assert snapshot.primary_metric == pytest.approx(0.2 + 1 / 10)
    assert snapshot.raw["call_count"] == 10


def test_snapshot_metrics_uses_low_confidence_share_for_routing():
    repository = FakeRoutingRepository({
        "trace_count": 10, "low_confidence_count": 3, "clarification_needed_count": 1,
        "low_confidence_share": 0.4, "avg_confidence": 0.3,
    })

    snapshot = snapshot_metrics(
        ROUTING_SUGGESTION, since=datetime.now(timezone.utc), repository=repository,
    )

    assert snapshot.kind == SelfImprovementKind.routing_capability_wording
    assert snapshot.target == "methodology"
    assert snapshot.primary_metric == 0.4


def test_snapshot_metrics_raises_for_routing_without_a_repository():
    with pytest.raises(ApplyError, match="requires a RoutingRepository"):
        snapshot_metrics(ROUTING_SUGGESTION, since=datetime.now(timezone.utc))


def test_snapshot_metrics_round_trips_through_json():
    store = FakeStore({
        "call_count": 5, "error_count": 0, "error_rate": 0.0, "user_correction_count": 0,
    })
    snapshot = snapshot_metrics(SKILL_SUGGESTION, since=datetime.now(timezone.utc), store=store)

    import json
    restored = json.loads(snapshot.to_json())

    assert restored["kind"] == "tool_wording"
    assert restored["target"] == "business_dictionary.lookup_business_term"
    assert restored["primary_metric"] == 0.0


NEW_SKILL_SUGGESTION = SelfImprovementSuggestion(
    kind=SelfImprovementKind.new_tool,
    target="project_management.summarize_blocked_epics",
    diagnosis="Missing capability.",
    confidence=0.7,
)


# -- apply_finding --------------------------------------------------------------

class FakeMethodologyRepository:
    """Minimal double for `MethodologyRepository`: enough of `graph.run` (finding
    lookup, project version) and `submit_changeset`/`start_workflow` to
    exercise `apply_finding`/`reevaluate_applied_findings` without Neo4j."""

    def __init__(self, findings, project_version=5):
        self.findings = findings  # key -> {"entity_id", "revision_number", "properties"}
        self.project_version = project_version
        self.changesets = []
        self.graph = SimpleNamespace(run=self._run)

    def _run(self, query, **params):
        if "RETURN p.graph_version" in query:
            return [] if self.project_version is None else [{"v": self.project_version}]
        if "ReviewFinding" in query and "reevaluate_after" in query:
            now = params["now"]
            return [
                {
                    "entity_id": f["entity_id"], "key": key,
                    "revision_number": f["revision_number"], "properties": f["properties"],
                }
                for key, f in self.findings.items()
                if f["properties"].get("status") == "applied"
                and f["properties"].get("reevaluate_after", "") <= now
            ]
        if "ReviewFinding" in query:
            key = params.get("key")
            finding = self.findings.get(key)
            if finding is None:
                return []
            return [{
                "entity_id": finding["entity_id"], "revision_number": finding["revision_number"],
                "properties": finding["properties"],
            }]
        return []

    def start_workflow(self, project_id, workflow, actor):
        return SimpleNamespace(run_id="run-1")

    def submit_changeset(self, request):
        self.changesets.append(request)
        operation = request.operations[0]
        key = next(k for k, f in self.findings.items() if f["entity_id"] == operation.entity_id)
        self.findings[key]["properties"] = {
            **self.findings[key]["properties"], **operation.properties,
        }
        self.findings[key]["revision_number"] += 1
        return {"graph_version": self.project_version}


def _open_finding(entity_id="finding-1"):
    return {entity_id: {
        "entity_id": entity_id, "revision_number": 1,
        "properties": {"status": "open"},
    }}


def test_apply_finding_rejects_architecture_kind_before_any_work():
    """`architecture`-kind suggestions never reach an edit/commit at all --
    always routed to human review, never a candidate for auto-apply."""
    architecture_suggestion = SelfImprovementSuggestion(
        kind=SelfImprovementKind.architecture, target="project_management.list_project_blockers",
        diagnosis="Needs a new domain package.", confidence=0.9,
    )
    repo = FakeMethodologyRepository(findings=_open_finding())

    with pytest.raises(ApplyError, match="never auto-applied"):
        apply_finding(architecture_suggestion, repo=repo)

    # Never even looked the finding up -- rejected before touching the graph.
    assert repo.changesets == []


def test_apply_finding_raises_when_no_finding_exists():
    repo = FakeMethodologyRepository(findings={})

    with pytest.raises(ApplyError, match="No ReviewFinding"):
        apply_finding(SKILL_SUGGESTION, repo=repo)


def test_apply_finding_raises_when_finding_is_not_open():
    key = finding_key(SKILL_SUGGESTION)
    findings = {key: {
        "entity_id": "finding-1", "revision_number": 2, "properties": {"status": "applied"},
    }}
    repo = FakeMethodologyRepository(findings=findings)

    with pytest.raises(ApplyError, match="not 'open'"):
        apply_finding(SKILL_SUGGESTION, repo=repo)


def _no_snapshot(monkeypatch):
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.snapshot_metrics",
        lambda suggestion, since, store=None, repository=None: SimpleNamespace(
            to_json=lambda: '{"kind": "tool_wording", "target": "x", "primary_metric": 0.1, "raw": {}}',
        ),
    )


def test_apply_finding_applies_a_wording_kind_as_a_graph_edit_and_marks_the_finding_applied(monkeypatch):
    """Wording kinds never touch source or git: the change is a graph write
    whose revert record lands on the finding."""
    key = finding_key(SKILL_SUGGESTION)
    repo = FakeMethodologyRepository(findings={key: {
        "entity_id": "finding-1", "revision_number": 1, "properties": {"status": "open"},
    }})
    captured = {}

    def fake_graph_edit(suggestion, graph, llm=None):
        captured["suggestion"] = suggestion
        return {"kind": "tool_wording", "target": suggestion.target,
                "value": suggestion.suggested_change, "previous_override": None}

    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.apply_graph_edit", fake_graph_edit,
    )
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.commit_files",
        lambda paths, message: pytest.fail("wording kinds must not commit"),
    )
    _no_snapshot(monkeypatch)

    result = apply_finding(SKILL_SUGGESTION, repo=repo)

    assert result["finding_key"] == key
    assert result["graph_edit"]["value"] == SKILL_SUGGESTION.suggested_change
    assert captured["suggestion"] is SKILL_SUGGESTION
    properties = repo.findings[key]["properties"]
    assert properties["status"] == "applied"
    assert "commit_sha" not in properties
    import json
    assert json.loads(properties["graph_edit"])["target"] == SKILL_SUGGESTION.target
    assert "pre_metric_snapshot" in properties and "reevaluate_after" in properties


def test_apply_finding_raises_and_does_not_touch_the_finding_when_graph_edit_is_rejected(monkeypatch):
    key = finding_key(SKILL_SUGGESTION)
    repo = FakeMethodologyRepository(findings={key: {
        "entity_id": "finding-1", "revision_number": 1, "properties": {"status": "open"},
    }})

    def boom(suggestion, graph, llm=None):
        from knowledge_bases.tools.graph_edits import GraphEditError
        raise GraphEditError("Rejected capability phrase")

    monkeypatch.setattr("knowledge_bases.tools.self_improvement_apply.apply_graph_edit", boom)
    _no_snapshot(monkeypatch)

    with pytest.raises(ApplyError, match="Could not apply graph edit"):
        apply_finding(SKILL_SUGGESTION, repo=repo)

    assert repo.findings[key]["properties"]["status"] == "open"
    assert repo.changesets == []


def test_apply_finding_raises_and_does_not_touch_the_finding_when_new_skill_source_edit_fails(monkeypatch):
    key = finding_key(NEW_SKILL_SUGGESTION)
    repo = FakeMethodologyRepository(findings={key: {
        "entity_id": "finding-1", "revision_number": 1, "properties": {"status": "open"},
    }})

    def boom(suggestion):
        from knowledge_bases.tools.source_editor import SourceEditError
        raise SourceEditError("no such module")

    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.apply_suggestion", boom,
    )
    _no_snapshot(monkeypatch)

    with pytest.raises(ApplyError, match="Could not apply source edit"):
        apply_finding(NEW_SKILL_SUGGESTION, repo=repo)

    assert repo.findings[key]["properties"]["status"] == "open"
    assert repo.changesets == []


def test_apply_finding_raises_and_does_not_touch_the_finding_when_new_skill_commit_fails(monkeypatch, tmp_path):
    key = finding_key(NEW_SKILL_SUGGESTION)
    repo = FakeMethodologyRepository(findings={key: {
        "entity_id": "finding-1", "revision_number": 1, "properties": {"status": "open"},
    }})
    edited_path = tmp_path / "catalog.py"
    edited_path.write_text("# edited")

    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.apply_suggestion",
        lambda suggestion: [edited_path],
    )
    monkeypatch.setattr("knowledge_bases.tools.self_improvement_apply.run_test_gate", lambda: None)

    def boom(paths, message):
        from knowledge_bases.tools.git_ops import GitOpsError
        raise GitOpsError("nothing to commit")

    monkeypatch.setattr("knowledge_bases.tools.self_improvement_apply.commit_files", boom)
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.discard_paths", lambda paths: None,
    )
    _no_snapshot(monkeypatch)

    with pytest.raises(ApplyError, match="Could not commit"):
        apply_finding(NEW_SKILL_SUGGESTION, repo=repo)

    assert repo.findings[key]["properties"]["status"] == "open"


def test_apply_finding_runs_the_test_gate_for_new_skill_before_committing(monkeypatch, tmp_path):
    key = finding_key(NEW_SKILL_SUGGESTION)
    repo = FakeMethodologyRepository(findings={key: {
        "entity_id": "finding-1", "revision_number": 1, "properties": {"status": "open"},
    }})
    edited_paths = [tmp_path / "mcp_tools.py", tmp_path / "catalog.py"]
    for path in edited_paths:
        path.write_text("# edited")

    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.apply_suggestion",
        lambda suggestion: edited_paths,
    )
    gate_called = {"count": 0}
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.run_test_gate",
        lambda: gate_called.__setitem__("count", gate_called["count"] + 1),
    )
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.commit_files",
        lambda paths, message: "abc123",
    )
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.snapshot_metrics",
        lambda suggestion, since, store=None, repository=None: SimpleNamespace(to_json=lambda: "{}"),
    )
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply._sync_after_edit", lambda: None,
    )

    result = apply_finding(NEW_SKILL_SUGGESTION, repo=repo)

    assert gate_called["count"] == 1
    assert result["commit_sha"] == "abc123"
    assert repo.findings[key]["properties"]["status"] == "applied"


def test_apply_finding_discards_and_marks_failed_when_the_test_gate_fails(monkeypatch, tmp_path):
    key = finding_key(NEW_SKILL_SUGGESTION)
    repo = FakeMethodologyRepository(findings={key: {
        "entity_id": "finding-1", "revision_number": 1, "properties": {"status": "open"},
    }})
    edited_paths = [tmp_path / "mcp_tools.py", tmp_path / "catalog.py"]
    for path in edited_paths:
        path.write_text("# edited")

    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.apply_suggestion",
        lambda suggestion: edited_paths,
    )

    def failing_gate():
        from knowledge_bases.tools.self_improvement_apply import GateFailure
        raise GateFailure("Test suite failed (exit 1)", output="FAILED test_x.py::test_x")

    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.run_test_gate", failing_gate,
    )
    discarded = {"paths": None}
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.discard_paths",
        lambda paths: discarded.__setitem__("paths", paths),
    )
    commit_called = {"count": 0}
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.commit_files",
        lambda paths, message: commit_called.__setitem__("count", commit_called["count"] + 1),
    )
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.snapshot_metrics",
        lambda suggestion, since, store=None, repository=None: SimpleNamespace(to_json=lambda: "{}"),
    )

    with pytest.raises(ApplyError, match="failed the test gate"):
        apply_finding(NEW_SKILL_SUGGESTION, repo=repo)

    assert discarded["paths"] == edited_paths
    assert commit_called["count"] == 0  # never committed
    assert repo.findings[key]["properties"]["status"] == "failed"
    assert "Test suite failed" in repo.findings[key]["properties"]["failure_reason"]
    assert "FAILED test_x.py::test_x" in repo.findings[key]["properties"]["failure_output"]


# -- apply_finding_by_key --------------------------------------------------------

def test_apply_finding_by_key_reconstructs_the_suggestion_from_stored_properties(monkeypatch, tmp_path):
    key = finding_key(SKILL_SUGGESTION)
    repo = FakeMethodologyRepository(findings={key: {
        "entity_id": "finding-1", "revision_number": 1,
        "properties": {
            "status": "open", "kind": "tool_wording",
            "target": "business_dictionary.lookup_business_term",
            "message": "Ambiguous description.", "suggested_change": "A clearer description.",
            "confidence": 0.7,
        },
    }})

    captured = {}

    def fake_graph_edit(suggestion, graph, llm=None):
        captured["suggestion"] = suggestion
        return {"kind": "tool_wording", "target": suggestion.target, "value": "x"}

    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.apply_graph_edit", fake_graph_edit,
    )
    _no_snapshot(monkeypatch)

    result = apply_finding_by_key(key, repo=repo)

    assert result["graph_edit"]["target"] == "business_dictionary.lookup_business_term"
    assert captured["suggestion"].target == "business_dictionary.lookup_business_term"
    assert captured["suggestion"].suggested_change == "A clearer description."
    assert repo.findings[key]["properties"]["status"] == "applied"


def test_apply_finding_by_key_raises_for_an_unknown_key():
    repo = FakeMethodologyRepository(findings={})

    with pytest.raises(ApplyError, match="No ReviewFinding"):
        apply_finding_by_key("does-not-exist", repo=repo)


# -- reevaluate_applied_findings -------------------------------------------------

def _applied_finding(pre_metric=0.5, commit_sha="abc123"):
    import json
    return {
        "entity_id": "finding-1",
        "revision_number": 2,
        "properties": {
            "status": "applied",
            "applied_at": (datetime.now(timezone.utc) - timedelta(days=4)).isoformat(),
            "commit_sha": commit_sha,
            "pre_metric_snapshot": json.dumps({
                "kind": "tool_wording",
                "target": "business_dictionary.lookup_business_term",
                "primary_metric": pre_metric,
                "raw": {},
            }),
            "reevaluate_after": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        },
    }


def test_reevaluate_keeps_a_finding_whose_metric_improved(monkeypatch):
    key = "self_improvement::tool_wording::business_dictionary.lookup_business_term"
    repo = FakeMethodologyRepository(findings={key: _applied_finding(pre_metric=0.5)})
    store = FakeStore({
        "call_count": 10, "error_count": 0, "error_rate": 0.0, "user_correction_count": 0,
    })

    results = reevaluate_applied_findings(repo=repo, store=store)

    assert len(results) == 1
    assert results[0]["verdict"] == "kept"
    assert repo.findings[key]["properties"]["status"] == "kept"


def test_reevaluate_reverts_a_finding_whose_metric_got_worse(monkeypatch):
    key = "self_improvement::tool_wording::business_dictionary.lookup_business_term"
    repo = FakeMethodologyRepository(findings={key: _applied_finding(pre_metric=0.0)})
    store = FakeStore({
        "call_count": 10, "error_count": 5, "error_rate": 0.5, "user_correction_count": 3,
    })
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.revert_commit",
        lambda sha: "revert-sha-1",
    )
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply._sync_after_edit", lambda: None,
    )

    results = reevaluate_applied_findings(repo=repo, store=store)

    assert len(results) == 1
    assert results[0]["verdict"] == "reverted"
    assert results[0]["revert_commit_sha"] == "revert-sha-1"
    assert repo.findings[key]["properties"]["status"] == "reverted"
    assert repo.findings[key]["properties"]["revert_commit_sha"] == "revert-sha-1"


def test_reevaluate_reverts_a_graph_edit_in_the_graph_not_git(monkeypatch):
    import json
    key = "self_improvement::tool_wording::business_dictionary.lookup_business_term"
    finding = _applied_finding(pre_metric=0.0, commit_sha=None)
    del finding["properties"]["commit_sha"]
    record = {"kind": "tool_wording", "target": "business_dictionary.lookup_business_term",
              "value": "new", "previous_override": None}
    finding["properties"]["graph_edit"] = json.dumps(record)
    repo = FakeMethodologyRepository(findings={key: finding})
    store = FakeStore({
        "call_count": 10, "error_count": 5, "error_rate": 0.5, "user_correction_count": 3,
    })
    reverted = {}
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.revert_graph_edit",
        lambda rec, graph: reverted.setdefault("record", rec) and {"reverted": rec["kind"]},
    )
    monkeypatch.setattr(
        "knowledge_bases.tools.self_improvement_apply.revert_commit",
        lambda sha: pytest.fail("graph edits must not be git-reverted"),
    )

    results = reevaluate_applied_findings(repo=repo, store=store)

    assert results[0]["verdict"] == "reverted"
    assert reverted["record"] == record
    assert repo.findings[key]["properties"]["status"] == "reverted"
    assert "graph_revert" in repo.findings[key]["properties"]


def test_reevaluate_skips_findings_not_yet_due(monkeypatch):
    key = "self_improvement::tool_wording::business_dictionary.lookup_business_term"
    finding = _applied_finding(pre_metric=0.5)
    finding["properties"]["reevaluate_after"] = (
        datetime.now(timezone.utc) + timedelta(days=2)
    ).isoformat()
    repo = FakeMethodologyRepository(findings={key: finding})
    store = FakeStore({
        "call_count": 10, "error_count": 0, "error_rate": 0.0, "user_correction_count": 0,
    })

    results = reevaluate_applied_findings(repo=repo, store=store)

    assert results == []


def test_reevaluate_reports_an_error_for_one_bad_finding_without_blocking_others(monkeypatch):
    good_key = "self_improvement::tool_wording::business_dictionary.lookup_business_term"
    bad_key = "self_improvement::tool_wording::broken.target"
    good_finding = _applied_finding(pre_metric=0.5)
    bad_finding = _applied_finding(pre_metric=0.5)
    bad_finding["properties"]["pre_metric_snapshot"] = "not valid json"
    repo = FakeMethodologyRepository(findings={good_key: good_finding, bad_key: bad_finding})
    store = FakeStore({
        "call_count": 10, "error_count": 0, "error_rate": 0.0, "user_correction_count": 0,
    })

    results = reevaluate_applied_findings(repo=repo, store=store)

    verdicts = {r["finding_key"]: r["verdict"] for r in results}
    assert verdicts[good_key] == "kept"
    assert verdicts[bad_key] == "error"
