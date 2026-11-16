"""One-time seed: the reflexive `noesis-self` methodology project.

Wires a worked example of the self-model design end to end:

1. Noesis's own source must already be indexed into the code graph by
   `scripts/dev_catalog.py sync --repo noesis` (see
   `knowledge_bases/development/self`).
2. A full methodology traceability chain (Problem -> Requirement -> Capability ->
   Architecture -> Decision) is created in one changeset in the `noesis-self`
   project, using a "Neo4j cold tier" graph-tiering decision as the
   worked example.
3. A rationale `Claim` is recorded against the `Decision`, satisfying the methodology
   `DECISION_WITHOUT_RATIONALE` architecture-phase validation gate.
4. The `Architecture` entity is bridged to the indexed self repo via
   `(:BmadEntity:Architecture)-[:CATALOGS]->(:CodeRepository {id: "noesis"})`.
   This bridge edge is written with the plain `Graph.upsert_rel` helper, not
   through the methodology changeset API, because `CodeRepository` is not a
   `BmadEntity` -- mirroring how `graph.attach_source`/`FOUND_IN` bridges are
   written outside any single knowledge base's own transactional API.

   The reasoning layer deliberately stays in a *different* ontology from the
   code graph: design intent is human-authored and revisioned
   (`decisions`/`architecture`/`project_management` + `methodology`), while
   the code graph is regenerated wholesale from source. Only this one edge
   joins them.

Run `python scripts/self_recompute.py` afterward to see the health-check pass
over this project (it should report no drift immediately after seeding).
"""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph import Graph
from knowledge_bases.methodology.models import (
    ChangeSetRequest,
    ClaimType,
    CreateEntity,
    CreateRelationship,
    RecordClaimRequest,
)
from knowledge_bases.methodology.repository import MethodologyRepository
from knowledge_bases.development.self import SELF_REPO_ID

PROJECT_ID = "noesis-self"
ACTOR = "seed_self"


def seed_self_model(graph: Graph | None = None) -> dict:
    graph = graph or Graph()
    repo = MethodologyRepository(graph)

    # The code graph is a prerequisite, not something this script builds: it
    # is produced by `scripts/dev_catalog.py`, which indexes every repo the
    # same way. Fail loudly rather than seeding a bridge to a missing node.
    indexed = graph.run(
        "MATCH (r:CodeRepository {id: $repo}) RETURN count(r) AS c",
        repo=SELF_REPO_ID,
    )[0]["c"]
    if not indexed:
        raise SystemExit(
            f"[seed_self] repo {SELF_REPO_ID!r} is not indexed. Run "
            f"`python scripts/dev_catalog.py sync --repo {SELF_REPO_ID}` first."
        )

    graph_version = repo.create_project(PROJECT_ID, "Noesis Self-Model")
    run = repo.start_workflow(PROJECT_ID, "self_architecture_seed", ACTOR)

    problem_id = str(uuid4())
    requirement_id = str(uuid4())
    capability_id = str(uuid4())
    architecture_id = str(uuid4())
    decision_id = str(uuid4())

    request = ChangeSetRequest(
        project_id=PROJECT_ID,
        run_id=run.run_id,
        expected_graph_version=graph_version,
        actor=ACTOR,
        operations=[
            CreateEntity(
                type="create_entity",
                entity_id=problem_id,
                entity_type="Problem",
                key="archived-decisions-still-relationship-shaped",
                properties={
                    "statement": (
                        "Archived Decisions and superseded evidence edges are "
                        "rarely queried but still relationship-shaped; keeping "
                        "them live in the hot graph forever wastes RAM, and "
                        "flattening them into plain SQL rows loses traversal."
                    ),
                },
            ),
            CreateEntity(
                type="create_entity",
                entity_id=requirement_id,
                entity_type="Requirement",
                key="disk-backed-cold-graph-tier",
                properties={
                    "statement": "Cold, rarely-accessed graph data must remain queryable as a graph, at lower cost than RAM.",
                },
            ),
            CreateRelationship(
                type="create_relationship",
                from_id=requirement_id,
                relationship="ADDRESSES",
                to_id=problem_id,
            ),
            CreateEntity(
                type="create_entity",
                entity_id=capability_id,
                entity_type="Capability",
                key="second-neo4j-instance-small-page-cache",
                properties={
                    "statement": "Run a second Neo4j Community instance with a small page cache as the cold/archive tier.",
                },
            ),
            CreateRelationship(
                type="create_relationship",
                from_id=capability_id,
                relationship="SATISFIES",
                to_id=requirement_id,
            ),
            CreateEntity(
                type="create_entity",
                entity_id=architecture_id,
                entity_type="Architecture",
                key="graph-storage-hot-cold-tiering",
                properties={
                    "statement": "Hot graph tier (Memgraph or existing Neo4j) plus a disk-bound cold Neo4j tier for archived/superseded evidence.",
                },
            ),
            CreateRelationship(
                type="create_relationship",
                from_id=architecture_id,
                relationship="IMPLEMENTS",
                to_id=capability_id,
            ),
            CreateEntity(
                type="create_entity",
                entity_id=decision_id,
                entity_type="Decision",
                key="use-neo4j-for-cold-tier-not-kuzu",
                properties={
                    "statement": "Use Neo4j Community (small page cache) for the cold tier instead of an embedded engine like Kuzu.",
                },
            ),
            CreateRelationship(
                type="create_relationship",
                from_id=decision_id,
                relationship="AFFECTS",
                to_id=architecture_id,
            ),
        ],
    )
    result = repo.submit_changeset(request)

    claim_result = repo.record_claim(RecordClaimRequest(
        project_id=PROJECT_ID,
        run_id=run.run_id,
        expected_graph_version=result.graph_version,
        actor=ACTOR,
        claim_type=ClaimType.rationale,
        statement=(
            "Neo4j shares Cypher-dialect lineage and the existing `neo4j` "
            "Bolt driver already used by graph.py (component: core), so "
            "porting queries and reusing the driver carries near-zero risk. "
            "Its disk-first storage engine degrades gracefully (slower, not "
            "OOM) when the page cache is undersized, unlike Memgraph's "
            "unsupported on-disk mode. Depends on component: core."
        ),
        about_entity_ids=[decision_id],
    ))

    # Bridge the Architecture entity to the indexed self repo -- written with
    # the raw Graph helper, not the methodology changeset API, since
    # CodeRepository is not a BmadEntity. This is the single edge that joins
    # recorded design intent to the code graph; see
    # `knowledge_bases/development/self/queries.py:architectural_rationale`.
    graph.upsert_rel("Architecture", {"id": architecture_id}, "CATALOGS",
                     "CodeRepository", {"id": SELF_REPO_ID})

    print(f"[seed_self] seeded {PROJECT_ID!r}: "
          f"problem={problem_id[:8]} requirement={requirement_id[:8]} "
          f"capability={capability_id[:8]} architecture={architecture_id[:8]} "
          f"decision={decision_id[:8]} claim={claim_result.claim_id[:8]}")

    return {
        "graph_version": claim_result.graph_version,
        "problem_id": problem_id,
        "requirement_id": requirement_id,
        "capability_id": capability_id,
        "architecture_id": architecture_id,
        "decision_id": decision_id,
        "claim_id": claim_result.claim_id,
    }


if __name__ == "__main__":
    seed_self_model()
