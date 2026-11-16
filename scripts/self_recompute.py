"""Self decision-health-style re-evaluation for Noesis's reflexive methodology project.

Mirrors `recompute.py`'s shape: instead of recomputing SQL-backed `Claim
nodes and evaluating `Decision` health via `SUPPORTS`/`CHALLENGES` edges, this
walks the `noesis-self` methodology project's `Architecture` entities, compares
what the components of their `CATALOGS`-bridged `CodeRepository` actually depend
on (derived from the live `CodeFile -[:IMPORTS]->` graph, refreshed by
`scripts/dev_catalog.py`) against what any rationale `Claim` for a `Decision`
that `AFFECTS` that `Architecture` already accounts for, and files a
`ReviewFinding` `BmadEntity` when the two diverge -- i.e. "the architecture
changed without a recorded decision".

This never mutates or supersedes an existing `Decision` automatically. It only
raises findings for human/LLM review, per the propose-confirm-execute pattern
described in `docs/architecture/user_feedback_loop.md`. Findings are capped per
run (`_MAX_FINDINGS_PER_RUN`): they are only worth filing if a human reads them.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph import Graph
from knowledge_bases.methodology.models import ChangeSetRequest, CreateEntity
from knowledge_bases.methodology.repository import (
    MethodologyRepository,
    ConcurrentUpdateError,
    EntityNotFound,
)

SELF_PROJECT_ID = "noesis-self"
_SYSTEM_ACTOR = "self_recompute"

# Findings are advisory and human-reviewed, so throughput is bounded by
# attention, not by how many the query can find.
_MAX_FINDINGS_PER_RUN = 5


def _component_dependency_weights(graph: Graph) -> list[dict]:
    """Inter-component import counts, used to rank which drift matters most."""
    # Aliased import: a bare `from ... import self` would shadow `self`.
    from knowledge_bases.development.self import components as self_components
    from knowledge_bases.development.self import SELF_REPO_ID

    return self_components.component_dependencies(graph, SELF_REPO_ID)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_version(graph: Graph, project_id: str) -> int | None:
    rows = graph.run(
        "MATCH (p:Project {id: $project_id}) RETURN p.graph_version AS v",
        project_id=project_id,
    )
    return int(rows[0]["v"]) if rows else None


def _architecture_drift(graph: Graph) -> list[dict]:
    """Find (architecture, component, undocumented dependency) triples.

    A component under the CATALOGS-bridged self repo depends on another
    component that no rationale Claim for a Decision affecting that
    Architecture mentions.

    Components are *derived* from the live import graph
    (`knowledge_bases/development/self/components.py`) rather than read from
    stored nodes. That is deliberate: the previous version walked
    `SelfComponent -[:DEPENDS_ON]-> SelfComponent` nodes written by a second
    sync script, and when those went stale this check spent months filing
    drift findings against knowledge bases that no longer existed (4 of its 7
    findings, deprecated by migration 23).
    """
    # Aliased import: a bare `from ... import self` would shadow `self`.
    from knowledge_bases.development.self import components as self_components

    bridged = graph.run(
        "MATCH (arch:BmadEntity:Architecture {project_id: $project_id})"
        "-[:CATALOGS]->(r:CodeRepository) "
        "RETURN arch.id AS architecture_id, arch.key AS architecture_key, "
        "r.id AS repo",
        project_id=SELF_PROJECT_ID,
    )
    if not bridged:
        return []

    rationales = [
        row["statement"] or ""
        for row in graph.run(
            "MATCH (arch:BmadEntity:Architecture {project_id: $project_id})"
            "<-[:AFFECTS]-(:BmadEntity:Decision)"
            "<-[:SUPPORTS]-(claim:Claim {claim_type: 'rationale', "
            "project_id: $project_id}) "
            "RETURN claim.statement AS statement",
            project_id=SELF_PROJECT_ID,
        )
    ]

    drift: list[dict] = []
    for arch in bridged:
        for edge in self_components.component_dependencies(graph, arch["repo"]):
            src, dep = edge["src"], edge["dst"]
            if any(dep in statement for statement in rationales):
                continue
            drift.append({
                "architecture_id": arch["architecture_id"],
                "architecture_key": arch["architecture_key"],
                "component_id": src,
                "undocumented_dependency": dep,
            })
    return drift


def _existing_finding_keys(graph: Graph) -> set[str]:
    rows = graph.run(
        "MATCH (f:BmadEntity:ReviewFinding {project_id: $project_id}) RETURN f.key AS key",
        project_id=SELF_PROJECT_ID,
    )
    return {row["key"] for row in rows}


def evaluate_self_architecture(repo: MethodologyRepository | None = None) -> list[dict]:
    """Re-evaluate the noesis-self methodology project against the current self-catalog.

    Files a `ReviewFinding` for each new structural drift. Returns the drift
    rows that were newly flagged this run (empty if healthy, or if the
    `noesis-self` project hasn't been seeded yet -- see `seed_self.py`).
    """
    repo = repo or MethodologyRepository(Graph())
    graph = repo.graph

    version = _current_version(graph, SELF_PROJECT_ID)
    if version is None:
        print(f"[self_recompute] project {SELF_PROJECT_ID!r} not seeded yet; run seed_self.py")
        return []

    drift = _architecture_drift(graph)
    already_flagged = _existing_finding_keys(graph)
    to_file = [
        row for row in drift
        if f"drift::{row['component_id']}::{row['undocumented_dependency']}" not in already_flagged
    ]

    # A finding a human will not read is not a finding. The previous version
    # walked stored component nodes and saw 7 dependencies; deriving them from
    # the live import graph sees every real edge, which would file ~15 at once
    # and bury the signal. Cap per run, worst first, so the queue stays
    # reviewable and the rest surface on later runs as earlier ones are closed.
    if len(to_file) > _MAX_FINDINGS_PER_RUN:
        weights = {
            (edge["src"], edge["dst"]): edge["weight"]
            for edge in _component_dependency_weights(graph)
        }
        to_file.sort(
            key=lambda row: weights.get(
                (row["component_id"], row["undocumented_dependency"]), 0),
            reverse=True,
        )
        print(
            f"[self_recompute] {len(to_file)} undocumented dependencies; "
            f"filing the {_MAX_FINDINGS_PER_RUN} heaviest this run"
        )
        to_file = to_file[:_MAX_FINDINGS_PER_RUN]

    if not to_file:
        print("[self_recompute] no new architecture drift found")
        return []

    run = repo.start_workflow(SELF_PROJECT_ID, "self_review", _SYSTEM_ACTOR)
    filed: list[dict] = []

    for row in to_file:
        key = f"drift::{row['component_id']}::{row['undocumented_dependency']}"
        version = _current_version(graph, SELF_PROJECT_ID)
        request = ChangeSetRequest(
            project_id=SELF_PROJECT_ID,
            run_id=run.run_id,
            expected_graph_version=version,
            actor=_SYSTEM_ACTOR,
            operations=[
                CreateEntity(
                    type="create_entity",
                    entity_id=str(uuid4()),
                    entity_type="ReviewFinding",
                    key=key,
                    properties={
                        "severity": "advisory",
                        "status": "open",
                        "message": (
                            f"Component '{row['component_id']}' now depends on "
                            f"'{row['undocumented_dependency']}', which is not "
                            f"covered by any recorded rationale for "
                            f"architecture '{row['architecture_key']}'."
                        ),
                        "filed_at": _now(),
                    },
                )
            ],
        )
        try:
            repo.submit_changeset(request)
            filed.append(row)
        except (EntityNotFound, ConcurrentUpdateError) as exc:
            print(f"[self_recompute] skipped {key}: {exc}")

    return filed


if __name__ == "__main__":
    findings = evaluate_self_architecture()
    print(f"[self_recompute] filed {len(findings)} new review finding(s)")
