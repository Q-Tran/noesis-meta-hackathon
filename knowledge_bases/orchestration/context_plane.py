"""The context plane: everything an agent's context should contain, from the graph.

Noesis and Prime Agent divide cleanly, and this module is the seam:

- **Prime owns execution.** The agent loop, the Python kernel, sub-agent
  lifecycle, daemon continuity, compaction, heartbeats, schedules, autonomous
  budgets. All of it stays Prime's, unmodified.
- **Noesis owns context.** Which experts exist and what they know, which tools
  are callable and how, what was established in prior sessions, what is
  unresolved, and which durable instructions apply. All of it is graph data.

Everything a harness previously discovered from local files -- agent
definitions, skill descriptions, project instructions -- is served from here
instead. That inverts the old arrangement, where the graph was one *source* the
agent could consult and the files were its identity. Now the graph *is* the
identity, and files are a cache of it.

## Why one tool rather than several

`get_noesis_context` returns the whole plane in a single call because a session
starts by needing all of it at once, and a round trip per concern (experts, then
tools, then session, then instructions) burns turns on bookkeeping the model
should never do. The payload is deliberately bounded and summary-shaped: full
prompts and schemas are fetched per delegation via `get_noesis_expert`, so this
is a manifest, not a dump.

## What "source of all context" does not mean

It does not mean Noesis injects arbitrary text into the model. The plane is
structured, versioned, and every element is traceable to a graph node with a
revision history. It also does not mean Noesis supersedes Prime's own prompt:
the orchestrator instructions are additive, and the base prompt's prohibitions
declare themselves non-overridable (see `experts_catalog.py`).
"""
from __future__ import annotations

from typing import Any

from graph import Graph

# Bounds on the manifest. A context plane is read at session start, so it has
# to be cheap and skimmable; anything unbounded here shows up as latency and
# tokens on every single session.
MAX_CONTEXT_TOOLS = 200
MAX_CONTEXT_EXPERTS = 50
MAX_RESUMABLE_SESSIONS = 5


def _experts(graph: Graph) -> list[dict[str, Any]]:
    """Every routable expert as a routing-time summary.

    Capabilities are included because they are what makes an expert
    *selectable*; the full prompt is not, because a worker fetches that itself
    when it adopts the expert.
    """
    from knowledge_bases.orchestration.repository import RoutingRepository

    experts = RoutingRepository(graph).load_experts()
    return [
        {
            "key": expert.key,
            "name": expert.name,
            "description": expert.description,
            "capabilities": expert.effective_capabilities,
            "entry_tools": expert.entry_tools,
            "version": expert.version,
        }
        for expert in experts[:MAX_CONTEXT_EXPERTS]
    ]


def _tools(graph: Graph) -> list[dict[str, Any]]:
    """The callable tool surface, by knowledge base.

    Descriptions come from the graph (including any `description_override` a
    self-improvement applied), so what the model reads is always the current
    wording rather than whatever shipped in a file.
    """
    from knowledge_bases.tools.repository import ToolRepository

    tools = ToolRepository(graph).load_tools()
    return [
        {
            "key": tool.key,
            "knowledge_base": tool.knowledge_base,
            "operation": tool.key.split(".", 1)[1] if "." in tool.key else tool.key,
            "description": tool.description,
        }
        for tool in tools[:MAX_CONTEXT_TOOLS]
    ]


def _instructions(graph: Graph) -> dict[str, Any]:
    """The orchestrator's own durable instructions.

    `base` is the immutable half that lives in `experts_catalog.py`;
    `supplemental` is what the graph has learned or been told since (prompt
    prefix + overlays). Split so a consumer can render them with the right
    precedence rather than concatenating blindly.
    """
    from knowledge_bases.orchestration.experts_catalog import ORCHESTRATOR_KEY
    from knowledge_bases.orchestration.repository import RoutingRepository

    orchestrator = RoutingRepository(graph).get_expert(ORCHESTRATOR_KEY)
    if orchestrator is None:
        return {"base": "", "supplemental": [], "version": 0}
    return {
        "base": orchestrator.system_prompt,
        "supplemental": (
            list(orchestrator.prompt_prefix_overlays) + list(orchestrator.prompt_overlays)
        ),
        "version": orchestrator.version,
    }


def _memory(graph: Graph, session_id: str | None) -> dict[str, Any]:
    """What prior work this session should already know about.

    This is the half Prime cannot supply on its own: its session state is a
    local file per checkout, whereas these traces are shared, queryable, and
    survive the process that produced them.
    """
    from knowledge_bases.orchestration.sessions import SessionRepository

    repository = SessionRepository(graph)
    recent = repository.list_sessions(limit=MAX_RESUMABLE_SESSIONS)
    memory: dict[str, Any] = {
        "resumable_sessions": [
            {
                "id": summary.id,
                "last_active_at": str(summary.last_active_at),
                "trace_count": summary.trace_count,
                "first_question": summary.first_question,
            }
            for summary in recent
        ],
    }
    if session_id:
        memory["digest"] = repository.digest(session_id)
    return memory


def get_noesis_context(arguments: dict[str, Any], graph: Graph) -> dict[str, Any]:
    """The complete context plane for one session.

    Called once at session start (and again after `/reload`). Pass
    `session_id` to include that session's resume digest; omit it to get the
    roster plus a list of what is resumable.
    """
    include = set(
        arguments.get("include") or ("experts", "tools", "instructions", "memory")
    )
    # `skills` was this section's name when a callable operation was called a
    # skill. Still accepted so an older caller keeps working, but the section is
    # emitted as `tools`, which is what it always contained.
    if "skills" in include:
        include.add("tools")
    plane: dict[str, Any] = {
        "source": "noesis-graph",
        "contract": (
            "Noesis is the source of context: experts, tools, durable instructions, "
            "and cross-session memory all come from the graph and change without a "
            "restart. Prime Agent owns execution: the agent loop, the kernel, "
            "sub-agents, and continuity."
        ),
    }
    if "instructions" in include:
        plane["instructions"] = _instructions(graph)
    if "experts" in include:
        plane["experts"] = _experts(graph)
    if "tools" in include:
        plane["tools"] = _tools(graph)
    if "memory" in include:
        plane["memory"] = _memory(graph, arguments.get("session_id"))
    return plane
