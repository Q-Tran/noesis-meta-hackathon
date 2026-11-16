"""Creation time under the full composition, against a real Neo4j.

`noesis-core/tests/test_live_graph_writes.py` pins the boundary stamp in the
core-only profile, where the domain labels are not installed. This is the other
half: the labels that only exist once `noesis-ontologies` composes in, and the
conversation grounding edge, which had never been exercised end-to-end against
a live graph.

The rule being pinned (see `docs/architecture/provenance.md`): a creation
timestamp is a property of the *write*, so `graph.upsert_node`/`upsert_rel`
stamp it, and packages do not declare a timestamp per label. `Revision` is the
case that used to lie -- it declares `created_at` and no writer filled it.
"""
from __future__ import annotations

from uuid import uuid4

import pytest


def _live_graph():
    from graph import Graph

    try:
        graph = Graph()
        graph.run("RETURN 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"graph unavailable: {exc}")
    return graph


@pytest.fixture
def graph():
    g = _live_graph()
    yield g
    g.close()


# `Person`/`DocumentNode` declare no timestamp property at all; `Revision`
# declares `created_at`, which the registry has been promising since it was
# declared and no repository ever wrote.
_DOMAIN_LABELS_WITHOUT_A_DECLARED_CREATION_TIME = ("Person", "DocumentNode", "Revision")


@pytest.mark.parametrize("label", _DOMAIN_LABELS_WITHOUT_A_DECLARED_CREATION_TIME)
def test_domain_labels_are_datable_through_the_boundary(graph, label):
    """Every node written through the boundary is datable, including the domain
    labels the core profile cannot see."""
    probe = f"stamp-probe-{label.lower()}-{uuid4()}"
    try:
        graph.upsert_node(label, {"id": probe}, {"name": "stamp probe"})
        rows = graph.run(
            f"MATCH (n:`{label}` {{id: $id}}) RETURN n.created_at AS created_at",
            id=probe,
        )
        assert rows and rows[0]["created_at"] is not None, (
            f"{label} written by the boundary carries no creation time"
        )
    finally:
        graph.run(f"MATCH (n:`{label}` {{id: $id}}) DETACH DELETE n", id=probe)


def test_the_conversation_grounding_edge_is_live_dated_and_declares_nothing_extra(graph):
    """The grounding edge, end to end.

    It used to be written with a hand-rolled `MERGE ... {role, recorded_at:
    datetime()}`, a `recorded_at` the `FOUND_IN` ontology does not declare --
    and nothing in the suite touched a real database, so it had never been
    observed landing. Now it is an `upsert_rel` write: `{role}` as declared,
    and `created_at` from the boundary.
    """
    from knowledge_bases.provenance import conversation

    claim = f"stamp-probe-claim-{uuid4()}"
    trace = f"stamp-probe-trace-{uuid4()}"
    try:
        graph.upsert_node("Claim", {"id": claim}, {"statement": "probe", "claim_type": "fact"})
        graph.upsert_node("RoutingTrace", {"id": trace}, {"question": "the verbatim words"})

        result = conversation.ground_claim_in_trace(graph, trace, claim)
        assert result["grounded"] is True

        rows = graph.run(
            "MATCH (cl:Claim {id: $claim})-[f:FOUND_IN]->(t:RoutingTrace {id: $trace}) "
            "RETURN properties(f) AS props",
            claim=claim, trace=trace,
        )
        assert rows, "the grounding edge was not written"
        props = rows[0]["props"]
        assert props.get("role") == "origin", "the spine's verbatim role, unchanged"
        assert props.get("created_at") is not None, "the boundary stamp must land on the spine"
        assert "recorded_at" not in props, (
            "FOUND_IN declares only role: a bespoke per-edge timestamp is "
            "undeclared property drift"
        )
        assert set(props) <= {"role", "weight", "created_at"}, (
            f"unexpected FOUND_IN properties: {sorted(props)}"
        )
    finally:
        graph.run("MATCH (c:Claim {id: $id}) DETACH DELETE c", id=claim)
        graph.run("MATCH (t:RoutingTrace {id: $id}) DETACH DELETE t", id=trace)


def test_FOUND_IN_declares_no_timestamp_so_the_boundary_is_the_only_source(graph):
    """The other side of the same contract: if a package ever declares a
    timestamp property on `FOUND_IN`, the write boundary stops being the single
    source of the spine's date, and the drift comes back one package at a time."""
    from knowledge_bases.registry import relationship_types

    found_in = relationship_types()["FOUND_IN"]
    names = {(p.name if hasattr(p, "name") else p) for p in found_in.properties}
    assert not [n for n in names if n.endswith("_at")], (
        f"FOUND_IN declares a timestamp ({sorted(names)}): creation time belongs "
        "to the write boundary, not to the edge declaration"
    )
