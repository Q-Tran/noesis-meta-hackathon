"""Isolated, ephemeral subagent execution for cross-domain investigations.

Replaces the legacy polling-based AgentMessage mailbox. Instead of passing
messages back and forth, an expert (or the orchestrator) can spawn an isolated
subagent for a specific expert domain, give it a focused task, and receive
a clean, structured conclusion without polluting its own working context.
"""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from graph import Graph
from knowledge_bases.mcp import _short_name, dispatch_tool
from knowledge_bases.orchestration.delegation_results import (
    _entities_from,
    record_delegation_result,
)
from knowledge_bases.orchestration.repository import RoutingRepository, _skill_allowed
from knowledge_bases.orchestration.routing import (
    _family_entity_types,
    _fill_arguments,
)
from knowledge_bases.tools.active_trace import (
    active_trace_id,
    current_session_id,
    resolve_identity,
    set_active_identity,
)
from knowledge_bases.tools.repository import ToolRepository


class SubagentError(ValueError):
    """Raised when subagent invocation or delegation fails validation."""
    pass


class SpawnSubagentRequest(BaseModel):
    expert_key: str = ""
    task: str = ""
    starting_arguments: dict[str, Any] = Field(default_factory=dict)
    operation: str | None = None
    trace_id: str | None = None
    session_id: str | None = None
    calling_expert: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if not data.get("expert_key"):
                for alias in ("agent", "subagent_type", "subagent", "knowledge_base"):
                    if data.get(alias):
                        data["expert_key"] = data[alias]
                        break
            if not data.get("task"):
                for alias in ("prompt", "instruction", "task_instruction", "query", "question"):
                    if data.get(alias):
                        data["task"] = data[alias]
                        break
        return data


def spawn_ephemeral_subagent(
    arguments: dict[str, Any],
    graph: Graph,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Spawn an ephemeral subagent to resolve a focused task.

    Runs the delegated expert with its own system prompt and bounded tools,
    captures the tool output and resolved entities, synthesizes findings concisely,
    records a DelegationResult in the graph, and returns the result to the caller
    without context pollution.
    """
    request = SpawnSubagentRequest.model_validate(arguments)
    expert_key = request.expert_key.strip()
    if not expert_key:
        raise SubagentError("An expert_key (or agent) is required to spawn a subagent.")
    task = request.task.strip()
    if not task:
        raise SubagentError("A subagent task cannot be empty.")

    caller = request.calling_expert or resolve_identity({}).get("expert_key")
    if caller and caller == expert_key:
        raise SubagentError("An expert cannot spawn a subagent of itself.")

    repository = RoutingRepository(graph)
    expert = repository.get_expert(expert_key)
    if expert is None:
        raise SubagentError(f"Unknown expert {expert_key!r}")
    if not expert.active:
        raise SubagentError(f"Expert {expert_key!r} is not active.")

    trace_id = request.trace_id or active_trace_id() or str(uuid4())
    session_id = request.session_id or current_session_id()
    subagent_id = f"subagent-{uuid4().hex[:12]}"

    if request.operation:
        if "." in request.operation:
            kb, op = request.operation.split(".", 1)
        else:
            kb, op = expert.key, request.operation
    else:
        if expert.entry_tools:
            entry = expert.entry_tools[0]
            if "." in entry:
                kb, op = entry.split(".", 1)
            else:
                kb, op = expert.key, entry
        else:
            kb, op = expert.key, "hybrid_search"

    skill_key = f"{kb}.{op}"
    if not _skill_allowed(skill_key, expert.allowed_tools):
        raise SubagentError(
            f"Expert {expert.key!r} is not allowed to run {skill_key!r}: "
            f"allowed skills are {expert.allowed_tools}"
        )

    tool_args = dict(request.starting_arguments)
    if op == "hybrid_search":
        tool_args.setdefault("query", task)
        entity_types = _family_entity_types(expert.key)
        if entity_types:
            tool_args.setdefault("entity_types", entity_types)
        tool_args.setdefault("limit", 5)
    else:
        tools = ToolRepository(graph).load_tools(kb)
        matched_tool = next((t for t in tools if _short_name(t.key, kb) == op), None)
        if matched_tool and matched_tool.input_schema:
            filled = _fill_arguments(matched_tool.input_schema, task)
            if filled:
                for k, v in filled.items():
                    tool_args.setdefault(k, v)
            elif not tool_args and not request.operation:
                # If the entry tool cannot be filled from the question and no arguments were given,
                # fall back to hybrid_search over the expert's entity family
                kb, op = expert.key, "hybrid_search"
                skill_key = f"{kb}.{op}"
                tool_args = {
                    "query": task,
                    "entity_types": _family_entity_types(expert.key),
                    "limit": 5,
                }

    old_identity = resolve_identity({})
    status = "completed"
    error_msg = None
    raw_result: Any = None

    try:
        set_active_identity(
            expert_key=expert.key,
            user_id=old_identity.get("user_id"),
            client=old_identity.get("client"),
        )
        if kb == "orchestration" and op == "hybrid_search":
            from knowledge_bases.orchestration.mcp_tools import hybrid_search_tool
            raw_result = hybrid_search_tool(tool_args, graph)
        else:
            raw_result = dispatch_tool(
                graph,
                kb,
                op,
                tool_args,
                trace_id=trace_id,
                session_id=session_id,
            )
    except Exception as exc:
        status = "error"
        error_msg = str(exc)
        raw_result = {"error": error_msg}
    finally:
        set_active_identity(
            expert_key=old_identity.get("expert_key"),
            user_id=old_identity.get("user_id"),
            client=old_identity.get("client"),
        )

    tools_called = [skill_key]
    entities = _entities_from(raw_result)
    resolved_entities = [{"label": l, "key": k} for l, k in entities]

    if llm is None:
        from llm import get_llm
        llm = get_llm()

    if status == "error":
        summary = f"Subagent encountered an error executing {skill_key}: {error_msg}"
        confidence = 0.0
        stopped_because = "error"
        unresolved = [error_msg]
    else:
        prompt = (
            f"Task: {task}\n"
            f"Tool called: {skill_key}\n"
            f"Arguments: {json.dumps(tool_args, default=str)}\n"
            f"Result: {json.dumps(raw_result, default=str)}\n\n"
            "Synthesize these findings concisely for the calling expert. Focus on facts, resolved entities, and answers to the task."
        )
        try:
            completion = llm.complete(prompt, system=expert.effective_prompt).strip()
        except Exception:
            completion = ""
        if completion:
            summary = completion
        else:
            summary = f"Subagent completed task {task!r} via {skill_key}."
        confidence = 0.9
        stopped_because = "completed"
        unresolved = []

    try:
        record_delegation_result(
            {
                "trace_id": trace_id,
                "expert_key": expert.key,
                "summary": summary,
                "confidence": confidence,
                "unresolved": unresolved,
                "stopped_because": stopped_because,
            },
            graph,
        )
    except Exception:
        pass

    return {
        "status": status,
        "subagent_id": subagent_id,
        "expert_key": expert.key,
        "task": task,
        "summary": summary,
        "tools_called": tools_called,
        "entities_resolved": resolved_entities,
        "data": raw_result,
    }
