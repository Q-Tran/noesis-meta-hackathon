"""Graph and disk must agree.

The recurring failure in this codebase is not a crash -- it is an enumerated
fact going quietly stale. Four instances so far: `call_noesis_tool`'s hardcoded
knowledge-base enum, a routing eval question that stopped testing what it
named, `AgentSkill` nodes the sync script never wrote, and a README expert
table missing an expert.

None of them broke a test, because nothing compared the two sides. These do.
They skip without a graph so a laptop without Docker still runs the suite.
"""
from __future__ import annotations

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


@pytest.fixture
def own_catalog_graph(graph):
    """Skip when the graph holds a DIFFERENT product's catalog.

    What these tests pin is "this product's disk vs this product's graph". The
    dev database is shared across the four distributions, so a graph synced
    from `noesis` (enterprise) failing `noesis-core`'s suite would be
    reporting the product boundary, not staleness. The catalog owner is
    inferred from the synced tools' knowledge bases; to exercise the drift
    checks for this product, run scripts/sync_tools_catalog.py against a
    database dedicated to it.
    """
    from knowledge_bases.product import PROFILE

    rows = graph.run("MATCH (t:Tool) RETURN DISTINCT t.knowledge_base AS kb")
    synced = {row["kb"] for row in rows if row["kb"]}
    foreign = synced - set(PROFILE.packages)
    if foreign:
        pytest.skip(
            f"graph holds another product's catalog ({sorted(foreign)} not in "
            f"{PROFILE.key}); run scripts/sync_tools_catalog.py against a "
            "dedicated database to test this product's drift"
        )
    return graph


def test_every_tool_on_disk_is_in_the_graph(own_catalog_graph):
    """A tool defined in `catalog.py` but never synced is invisible to every
    caller, because dispatch resolves handlers from the graph."""
    from knowledge_bases.tools.catalog import bootstrap_catalog
    from knowledge_bases.tools.repository import ToolRepository

    on_disk = {tool.key for tool in bootstrap_catalog()}
    in_graph = {tool.key for tool in ToolRepository(own_catalog_graph).load_tools()}

    assert not on_disk - in_graph, (
        "tools defined on disk but missing from the graph; "
        "run scripts/sync_tools_catalog.py"
    )
    assert not in_graph - on_disk, (
        "tools in the graph with no definition on disk; they will fail to "
        "resolve a handler"
    )


def test_every_expert_on_disk_is_in_the_graph(own_catalog_graph):
    from knowledge_bases.orchestration.experts_catalog import bootstrap_experts
    from knowledge_bases.orchestration.repository import RoutingRepository

    on_disk = {expert.key for expert in bootstrap_experts()}
    in_graph = {
        expert.key
        for expert in RoutingRepository(own_catalog_graph).load_experts(include_unroutable=True)
    }

    assert on_disk == in_graph, "run scripts/sync_tools_catalog.py"


def test_every_agent_skill_stub_has_a_graph_node(graph):
    """`SKILL.md` files are stubs that fetch their body live. A stub whose
    `AgentSkill` node is missing renders as an empty skill at runtime -- which
    is exactly the gap that shipped once already (10 stubs, 9 nodes)."""
    import pathlib

    stub_dirs = {
        path.parent.name
        for path in pathlib.Path(".prime/agent/skills").glob("*/SKILL.md")
    }
    nodes = {row["key"] for row in graph.run("MATCH (s:AgentSkill) RETURN s.key AS key")}

    assert stub_dirs, "no SKILL.md stubs found; this test would be vacuous"
    assert not stub_dirs - nodes, (
        "SKILL.md stubs with no AgentSkill node; they would render empty"
    )


def test_every_registered_tool_handler_can_be_imported(graph):
    """A handler string that does not resolve fails only when someone calls
    that tool -- possibly months later."""
    import importlib

    from knowledge_bases.tools.repository import ToolRepository

    broken = []
    for tool in ToolRepository(graph).load_tools():
        if not tool.handler:
            continue
        module_name, _, function_name = tool.handler.partition(":")
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            broken.append(f"{tool.key}: {exc}")
            continue
        if not hasattr(module, function_name):
            broken.append(f"{tool.key}: {module_name} has no {function_name}")

    assert not broken, f"tools with unresolvable handlers: {broken}"


def test_the_call_noesis_tool_enum_matches_its_handler(own_catalog_graph):
    """The enum and the handler's own validation used to be two hardcoded
    lists. They are one import now; this keeps it that way."""
    from knowledge_bases.orchestration.mcp_tools import CALL_NOESIS_TOOL_KNOWLEDGE_BASES
    from knowledge_bases.tools.repository import ToolRepository

    tool = next(
        t for t in ToolRepository(own_catalog_graph).load_tools()
        if t.key == "orchestration.call_noesis_tool"
    )
    enum = tool.input_schema["properties"]["knowledge_base"]["enum"]

    assert set(enum) == set(CALL_NOESIS_TOOL_KNOWLEDGE_BASES)
