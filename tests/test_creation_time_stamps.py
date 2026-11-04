"""Creation time is stamped by the code that writes, at every write.

`graph.upsert_node`/`upsert_rel` add `created_at` on `ON CREATE SET` (see
`docs/architecture/provenance.md`), and Neo4j runs in community edition here,
so there is no database trigger to fall back on. The repositories that build
their Cypher by hand -- the tool registry, the expert catalog, the telemetry
store -- are not covered by those two functions, so each of those sites has to
spell the stamp out itself. This file is the guard against that leaking: one
static check over every relationship-creating clause in this repo, plus a live
write through the raw-Cypher telemetry path.

Mirrored byte-for-byte in the other repo, with the same root-relative scan, so
each copy checks the code it ships.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WRITERS = REPO_ROOT / "knowledge_bases"

CLAUSE = re.compile(r"(?:MERGE|CREATE)\s+\([^()]*\)-\[[^\]]*\]->\([^()]*\)")
STAMP = re.compile(
    r"created_at|started_at|recorded_at|computed_at|published_at|observed_at|made_at|confirmed_at"
)

# Prose, not a write: `ontology.py`'s module docstring quotes the `MERGE` that
# `graph.upsert_rel` builds, to explain what its registry check defends.
PROSE_EXEMPTIONS = {
    ("knowledge_bases/ontology.py", "MERGE (a)-[r:`{rel_type}`]->(b)"),
}


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


def _joined_cypher(path: Path) -> str:
    """The Cypher in these files is split across adjacent string literals, so
    join them: one clause has to be readable as one clause."""
    return " ".join(path.read_text().replace('"', "").split())


def test_no_writer_creates_a_relationship_without_a_creation_time():
    violations, scanned = [], 0
    for path in sorted(WRITERS.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        joined = _joined_cypher(path)
        for match in CLAUSE.finditer(joined):
            scanned += 1
            clause = match.group(0)
            if any(rel == ex_rel and clause.startswith(ex_clause)
                   for ex_rel, ex_clause in PROSE_EXEMPTIONS):
                continue
            if not STAMP.search(joined[match.start(): match.end() + 90]):
                violations.append(f"{rel}: {clause}")
    assert scanned > 15, (
        f"the scan found only {scanned} clauses: it has stopped checking anything"
    )
    assert not violations, (
        "relationships written with no creation time: " + "; ".join(violations)
    )


def test_the_raw_cypher_registry_and_telemetry_writes_are_dated(graph):
    """The path the primitives do not cover, end to end on a live graph.

    `sync_tool` writes the `Tool` node and `insert_span` writes the span, its
    `CALLED` and `PART_OF` edges, and the session's `HAS_TRACE` -- all in
    hand-built Cypher. Before this rule the tool registry's nodes carried
    `updated_at` and no birth time, and none of these edges carried a time of
    any kind, so "when did this tool enter the registry" and "when was this
    call linked to its trace" were unanswerable."""
    from knowledge_bases.tools.models import SpanEvent, SpanKind, Tool
    from knowledge_bases.tools.repository import ToolRepository
    from knowledge_bases.tools.trace_store import TraceStore

    token = uuid4()
    tool_key = f"stamp-probe-tool-{token}"
    span_id = f"stamp-probe-span-{token}"
    trace_id = f"stamp-probe-trace-{token}"
    session_id = f"stamp-probe-session-{token}"
    try:
        ToolRepository(graph).sync_tool(Tool(
            key=tool_key, name="stamp probe", version="1", knowledge_base="tools",
            description="probe", handler="tests.probe:noop",
        ))
        TraceStore(graph).insert_span(SpanEvent(
            span_id=span_id, trace_id=trace_id, session_id=session_id,
            name="tool.stamp_probe", kind=SpanKind.tool_call, knowledge_base="tools",
            skill=tool_key, started_at=datetime.now(timezone.utc),
        ))

        tool = graph.run(
            "MATCH (t:Tool {key: $key}) "
            "RETURN t.created_at AS created_at, t.updated_at AS updated_at",
            key=tool_key,
        )
        assert tool, "the registry write produced no Tool node"
        assert tool[0]["created_at"] is not None, "the tool registry carries no creation time"

        for rel_type in ("CALLED", "PART_OF"):
            rows = graph.run(
                f"MATCH (s:Span {{span_id: $span}})-[r:`{rel_type}`]->() "
                "RETURN r.created_at AS created_at",
                span=span_id,
            )
            assert rows, f"the telemetry write produced no {rel_type} edge"
            assert rows[0]["created_at"] is not None, f"{rel_type} carries no creation time"

        trace_edge = graph.run(
            "MATCH (:Session {id: $session})-[ht:HAS_TRACE]->(:RoutingTrace {id: $trace}) "
            "RETURN ht.created_at AS created_at",
            session=session_id, trace=trace_id,
        )
        assert trace_edge, "insert_span did not link the session to its trace"
        assert trace_edge[0]["created_at"] is not None, "HAS_TRACE carries no creation time"

        # The nodes, not just the edges: `insert_span` and the session writers
        # are hand-built Cypher too, and they created the trace, its span and
        # the session with `started_at` but no universal birth time.
        nodes = graph.run(
            "MATCH (s:Span {span_id: $span}) "
            "MATCH (t:RoutingTrace {id: $trace}) "
            "MATCH (sess:Session {id: $session}) "
            "RETURN s.created_at AS span, t.created_at AS trace, sess.created_at AS session",
            span=span_id, trace=trace_id, session=session_id,
        )
        assert nodes and all(nodes[0].values()), (
            f"telemetry nodes carry no creation time: {nodes}"
        )
    finally:
        graph.run("MATCH (t:Tool {key: $key}) DETACH DELETE t", key=tool_key)
        graph.run("MATCH (s:Span {span_id: $id}) DETACH DELETE s", id=span_id)
        graph.run("MATCH (t:RoutingTrace {id: $id}) DETACH DELETE t", id=trace_id)
        graph.run("MATCH (s:Session {id: $id}) DETACH DELETE s", id=session_id)


# Node clauses: the same rule one level down. The relationship scan above only
# matched `(a)-[r]->(b)`, so `MERGE (n:Label)` statements -- the telemetry
# store, the session writers, delegation results -- created their nodes
# unstamped until this check existed. Each node must stamp its *own* alias,
# which is what rejects a neighbouring edge's `ht.created_at` in the same
# statement.
NODE_CLAUSE = re.compile(
    r"(?:MERGE|CREATE)\s+\(\s*(?P<alias>\w+)\s*:\s*\w+[^()]*\)"
)
NODE_PROSE_EXEMPTIONS = {
    ("knowledge_bases/tools/trace_store.py", "MERGE (t:RoutingTrace {id})"),
}


def test_no_writer_creates_a_node_without_its_own_creation_time():
    violations, scanned = [], 0
    for path in sorted(WRITERS.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        joined = _joined_cypher(path)
        for match in NODE_CLAUSE.finditer(joined):
            scanned += 1
            clause = joined[match.start(): match.end()]
            if any(rel == ex_rel and clause.startswith(ex_clause)
                   for ex_rel, ex_clause in NODE_PROSE_EXEMPTIONS):
                continue
            alias = match.group("alias")
            window = joined[match.end(): match.end() + 260]
            stamped = (
                re.search(rf"\b{re.escape(alias)}\.created_at\s*=", window)
                or "created_at: datetime()" in clause
            )
            if not stamped:
                violations.append(f"{rel}: {clause}")
    assert scanned > 15, (
        f"the node scan found only {scanned} clauses: it has stopped checking anything"
    )
    assert not violations, (
        "nodes written with no creation time: " + "; ".join(violations)
    )
