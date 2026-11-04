"""A live tool surface is only live if the harness is told it changed.

Every Noesis tool is read from the graph at `tools/list` time, so a retired
skill or a corrected schema takes effect server-side immediately. A harness
caches that list, though, so without `notifications/tools/list_changed` the
change is invisible to the running client until it reconnects. These tests
cover that edge, and -- more importantly -- that it can never break the write
that triggered it.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from knowledge_bases.tools import catalog_notify


def setup_function():
    catalog_notify.reset_for_testing()


def teardown_function():
    catalog_notify.reset_for_testing()


class FakeSession:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.notifications = 0

    async def send_tool_list_changed(self):
        if self.fail:
            raise RuntimeError("connection gone")
        self.notifications += 1


# -- capability declaration ------------------------------------------------------

def test_the_server_advertises_that_its_lists_can_change():
    """A client only subscribes if the server says so; OpenCode gates its
    re-fetch on this capability."""
    from knowledge_bases.mcp import NOESIS_NOTIFICATION_OPTIONS, create_domain_server
    from knowledge_bases.registry import knowledge_base

    server, _ = create_domain_server(knowledge_base("orchestration"))
    options = server.create_initialization_options(
        notification_options=NOESIS_NOTIFICATION_OPTIONS,
    )

    assert options.capabilities.tools.listChanged is True
    assert options.capabilities.prompts.listChanged is True


def test_orchestration_server_exposes_pruned_tool_surface(monkeypatch):
    """By default, the orchestration server exposes only the minimal 3-tool gateway:
    search_noesis_tools, route_noesis_question, and call_noesis_tool."""
    import mcp.types
    from knowledge_bases.mcp import create_domain_server
    from knowledge_bases.registry import knowledge_base

    mock_tools = [
        mcp.types.Tool(name="search_noesis_tools", description="", inputSchema={}),
        mcp.types.Tool(name="route_noesis_question", description="", inputSchema={}),
        mcp.types.Tool(name="call_noesis_tool", description="", inputSchema={}),
        mcp.types.Tool(name="register_tool", description="", inputSchema={}),
        mcp.types.Tool(name="get_noesis_context", description="", inputSchema={}),
    ]
    class FakeGraphContext:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr("knowledge_bases.mcp.load_tool_definitions", lambda graph, kb: mock_tools)
    monkeypatch.setattr("knowledge_bases.mcp.Graph", FakeGraphContext)

    server, _ = create_domain_server(knowledge_base("orchestration"))
    handler = server.request_handlers[mcp.types.ListToolsRequest]
    res = asyncio.run(handler(mcp.types.ListToolsRequest()))
    tool_names = [t.name for t in res.root.tools]
    assert tool_names == ["search_noesis_tools", "route_noesis_question", "call_noesis_tool"]


# -- notification mechanics ------------------------------------------------------

def test_no_connected_session_is_a_silent_no_op():
    """The normal case for a CLI script: nothing to notify, nothing to fail."""
    assert catalog_notify.notify_tool_list_changed() == 0


def test_registered_sessions_are_notified():
    async def scenario():
        session = FakeSession()
        catalog_notify.register_session(session)
        assert catalog_notify.notify_tool_list_changed() == 1
        await asyncio.sleep(0)  # let the scheduled task run
        return session

    session = asyncio.run(scenario())
    assert session.notifications == 1


def test_every_live_session_is_notified():
    async def scenario():
        sessions = [FakeSession(), FakeSession()]
        for session in sessions:
            catalog_notify.register_session(session)
        count = catalog_notify.notify_tool_list_changed()
        await asyncio.sleep(0)
        return count, sessions

    count, sessions = asyncio.run(scenario())
    assert count == 2
    assert all(s.notifications == 1 for s in sessions)


def test_registering_the_same_session_twice_does_not_double_notify():
    async def scenario():
        session = FakeSession()
        catalog_notify.register_session(session)
        catalog_notify.register_session(session)
        catalog_notify.notify_tool_list_changed()
        await asyncio.sleep(0)
        return session

    assert asyncio.run(scenario()).notifications == 1


def test_a_dead_session_is_dropped_and_never_raises():
    """A notification failure must not surface to the caller: the graph write
    already succeeded and is authoritative."""
    async def scenario():
        catalog_notify.register_session(FakeSession(fail=True))
        catalog_notify.notify_tool_list_changed()
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert catalog_notify.notify_tool_list_changed() == 0  # pruned


def test_notifying_with_no_event_loop_is_safe():
    """Tool handlers are synchronous and also run from plain scripts."""
    catalog_notify.register_session(FakeSession())
    assert catalog_notify.notify_tool_list_changed() == 0  # nothing scheduled, no raise


def test_unregister_removes_a_session():
    session = FakeSession()
    catalog_notify.register_session(session)
    catalog_notify.unregister_session(session)

    assert catalog_notify.notify_tool_list_changed() == 0


# -- the writes actually notify --------------------------------------------------

class AdminGraph:
    def __init__(self, skills):
        self.skills = skills

    def run(self, query, **params):
        if "MATCH (s:Tool {active: true})" in query:
            return self.skills
        row = next((s for s in self.skills if s["key"] == params.get("key")), None)
        if row is None:
            return []
        if "RETURN s.input_schema AS base" in query:
            return [{"base": row["input_schema"], "handler": row["handler"]}]
        if "s.input_schema_override" in query:
            return [{"previous_override": None}]
        if "s.active = $active" in query:
            return [{"previous": row["active"]}]
        return [row]


def _skill_row():
    return {"key": "project_management.list_project_blockers", "name": "n", "version": "1",
            "knowledge_base": "project_management", "active": True, "description": "d",
            "handler": "m:f",
            "input_schema": json.dumps({"type": "object", "properties": {}})}


@pytest.mark.parametrize("operation,payload", [
    ("retire_skill", {"key": "project_management.list_project_blockers"}),
    ("reactivate_skill", {"key": "project_management.list_project_blockers"}),
    ("override_skill_input_schema",
     {"key": "project_management.list_project_blockers",
      "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}}),
])
def test_skill_lifecycle_writes_ask_the_harness_to_refetch(operation, payload, monkeypatch):
    from knowledge_bases.tools import tool_admin

    calls = []
    monkeypatch.setattr(tool_admin, "notify_tool_list_changed", lambda: calls.append(1))
    getattr(tool_admin, operation)(payload, AdminGraph([_skill_row()]))

    assert calls, f"{operation} changed the tool surface without notifying"


def test_expert_roster_writes_ask_the_harness_to_refetch(monkeypatch):
    from knowledge_bases.orchestration import expert_admin

    calls = []
    monkeypatch.setattr(expert_admin, "notify_tool_list_changed", lambda: calls.append(1))
    expert_admin._reindex_experts(AdminGraph([]))

    assert calls


def test_skill_wording_self_improvement_notifies_but_prompt_overlays_do_not(monkeypatch):
    """Overlays are fetched per delegation, so they need no notification; a
    reworded description changes what the client is showing the model."""
    from knowledge_bases.tools import graph_edits

    calls = []
    monkeypatch.setattr(graph_edits, "notify_tool_list_changed", lambda: calls.append(1))

    graph_edits._reindex(AdminGraph([]), skills=True)
    assert len(calls) == 1

    graph_edits._reindex(AdminGraph([]), experts=True)
    assert len(calls) == 1  # unchanged
