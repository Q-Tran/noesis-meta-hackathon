"""Bootstrap catalog of every graph-registered Tool.

This is the one place that still declares tool surface in Python -- as seed
data, analogous to `orchestration/experts_catalog.py`'s `CAPABILITIES`. Once
`sync_catalog()` MERGEs these into Neo4j, every MCP server builds its tool
list and dispatch purely from graph `Tool` nodes (see
`knowledge_bases.mcp.create_domain_server`), not from this module. Editing a
skill's description/schema/handler here and re-running `sync_catalog()` is
how the catalog changes; the graph, not this file, is what harnesses read
from at runtime.
"""
from __future__ import annotations

from typing import Any

from knowledge_bases.tools.models import Tool
from knowledge_bases.tools.repository import ToolRepository

VERSION = "1.0.0"

TRACE_ID_INPUT = {
    "type": "object",
    "properties": {"trace_id": {"type": "string"}},
    "required": ["trace_id"],
}

TERM_INPUT = {
    "type": "object",
    "properties": {"term": {"type": "string"}},
    "required": ["term"],
}

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
}

# Merged into every skill's `input_schema` (see `_skill`) so a client-side
# agent driving its own tool-calling loop -- not just `delegate_noesis_question`
# -- can correlate every call it makes into one trace/session. Both are
# optional and additive: existing callers that never set them are unaffected,
# and a handler that already treats `trace_id` as functional input (e.g.
# `get_trace_tool`) keeps working unchanged (see `knowledge_bases.mcp.dispatch_tool`
# and `knowledge_bases.mcp.trace_context`).
TRACE_CONTEXT_PROPERTIES = {
    "trace_id": {
        "type": "string",
        "description": (
            "Leave unset. Every call is already correlated automatically "
            "with the most recent route_noesis_question/delegate_noesis_question "
            "in this session -- there is no need to read, copy, or pass this "
            "value yourself. Only set it to deliberately override that "
            "automatic correlation (e.g. re-tagging a call against an "
            "earlier, different trace for debugging)."
        ),
    },
    "session_id": {
        "type": "string",
        "description": (
            "Leave unset. A session id is already assigned automatically "
            "for this connection -- there is no need to read, copy, or pass "
            "this value yourself. Only set it to deliberately override that "
            "automatic assignment."
        ),
    },
}


def _skill(
    key: str,
    knowledge_base: str,
    name: str,
    description: str,
    input_schema: dict,
    handler: str,
) -> Tool:
    schema = dict(input_schema)
    properties = dict(schema.get("properties", {}))
    for prop_name, prop_schema in TRACE_CONTEXT_PROPERTIES.items():
        properties.setdefault(prop_name, prop_schema)
    schema["properties"] = properties
    return Tool(
        key=key,
        name=name,
        version=VERSION,
        knowledge_base=knowledge_base,
        description=description,
        input_schema=schema,
        handler=handler,
    )


def bootstrap_catalog() -> list[Tool]:
    """Compose every installed ontology package's seeded `Tool`s.

    Each knowledge base owns its seed slice in `knowledge_bases/<key>/catalog.py`
    (this knowledge base's own entries are the `TOOLS` list at the bottom of
    this module, since the `_skill` helper and schema builders they call are
    private to it); `composed_tools()` resolves each package's `tools_seed`
    lazily in `_CATALOG_PACKAGE_ORDER`. Anything not in the composed order
    still syncs -- it is appended at the end.
    """
    from knowledge_bases.registry import installed_packages, resolve_seed

    packages = {package.key: package for package in installed_packages()}
    tools: list[Tool] = []
    seen: set[str] = set()
    for key in _CATALOG_PACKAGE_ORDER:
        package = packages.pop(key, None)
        if package and package.tools_seed:
            tools.extend(resolve_seed(package.tools_seed))
            seen.add(key)
    for package in packages.values():
        if package.tools_seed:
            tools.extend(resolve_seed(package.tools_seed))
    return tools

def _node_reference_schema() -> dict:
    """One pointer to an existing graph node.

    Spelled out rather than left as a bare `{"type": "object"}`: callers were
    guessing at the shape and getting raw pydantic validation errors back.
    """
    return {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": (
                    "Entity type, qualified or bare "
                    "(e.g. 'semantic_layer.Metric' or 'Metric')."
                ),
            },
            "id": {"type": "string", "description": "The node's id, if it has one."},
            "name": {
                "type": "string",
                "description": (
                    "The node's name. Use this when the label has no id -- "
                    "BusinessTerm nodes are identified by name."
                ),
            },
            "key": {"type": "string", "description": "The node's key, if it has one."},
        },
        "required": ["label"],
    }


def _hybrid_search_schema() -> dict:
    from knowledge_bases.orchestration.hybrid_search import SEARCH_TYPES

    return {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "entity_types": {
                "type": "array", "minItems": 1, "uniqueItems": True,
                "items": {"type": "string", "enum": sorted(SEARCH_TYPES)},
            },
            "filters": {"type": "object"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "required": ["query", "entity_types"],
    }


def _propose_write_schema() -> dict:
    from knowledge_bases.orchestration.commands import WRITE_COMMANDS

    return {
        "type": "object",
        "properties": {
            "command": {"type": "string", "enum": sorted(WRITE_COMMANDS)},
            "arguments": {"type": "object"},
            "summary": {"type": "string"},
        },
        "required": ["command", "arguments", "summary"],
    }


def _deprecate_entity_schema() -> dict:
    from knowledge_bases.orchestration.lifecycle import DeprecateEntityRequest

    return DeprecateEntityRequest.model_json_schema()


def _reactivate_entity_schema() -> dict:
    from knowledge_bases.orchestration.lifecycle import ReactivateEntityRequest

    return ReactivateEntityRequest.model_json_schema()


def _record_delegation_result_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "trace_id": {"type": "string"},
            "expert_key": {"type": "string"},
            "summary": {
                "type": "string",
                "description": "What you established, in a few sentences.",
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "unresolved": {
                "type": "array", "items": {"type": "string"},
                "description": "Anything you could not settle, one item each.",
            },
            "stopped_because": {
                "type": "string",
                "description": "Why you stopped: resolved, dead end, or missing data.",
            },
        },
        "required": ["trace_id", "expert_key"],
    }


def _override_input_schema_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Full skill key, '<kb>.<name>'."},
            "input_schema": {
                "type": "object",
                "description": (
                    "The complete replacement JSON Schema for the skill's arguments. "
                    "Must still declare every argument the handler requires. Empty "
                    "object clears the override."
                ),
            },
        },
        "required": ["key", "input_schema"],
    }


def _create_expert_schema() -> dict:
    from knowledge_bases.orchestration.expert_admin import CreateExpertRequest

    return CreateExpertRequest.model_json_schema()


def _revise_expert_schema() -> dict:
    from knowledge_bases.orchestration.expert_admin import ReviseExpertRequest

    return ReviseExpertRequest.model_json_schema()


def _set_expert_prompt_prefix_schema() -> dict:
    from knowledge_bases.orchestration.models import (
        MAX_PROMPT_PREFIX_CHARS,
        MAX_PROMPT_PREFIX_ENTRIES,
    )

    return {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Expert key, or 'orchestrator'."},
            "entries": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_PROMPT_PREFIX_ENTRIES,
                "description": (
                    f"The complete prefix: at most {MAX_PROMPT_PREFIX_ENTRIES} "
                    f"entries totalling {MAX_PROMPT_PREFIX_CHARS} characters. "
                    "Replaces any existing prefix; empty list clears it."
                ),
            },
        },
        "required": ["key", "entries"],
    }


def _call_noesis_tool_schema() -> dict:
    # See `knowledge_bases.orchestration.mcp_tools.CALL_NOESIS_TOOL_KNOWLEDGE_BASES`
    # for what's in/excluded and why; imported (not duplicated) so the tool's
    # declared `knowledge_base` enum and its handler's own validation can
    # never drift apart.
    from knowledge_bases.orchestration.mcp_tools import CALL_NOESIS_TOOL_KNOWLEDGE_BASES

    return {
        "type": "object",
        "properties": {
            "knowledge_base": {
                "type": "string", "enum": sorted(CALL_NOESIS_TOOL_KNOWLEDGE_BASES),
            },
            "operation": {"type": "string"},
            "arguments": {"type": "object"},
            "expert_key": {
                "type": "string",
                "description": (
                    "Key of the graph-defined expert making this call (from "
                    "get_noesis_expert), so the resulting span is attributed to it."
                ),
            },
        },
        "required": ["knowledge_base", "operation"],
    }


def _search_noesis_tools_schema() -> dict:
    # Wider than `call_noesis_tool`'s own filter enum: this can also search
    # (though not itself directly call through this proxy) `orchestration`'s
    # own tools, since `search_noesis_tools` is a discovery surface over the
    # *entire* catalog, not a dispatch boundary.
    from knowledge_bases.orchestration.mcp_tools import CALL_NOESIS_TOOL_KNOWLEDGE_BASES

    return {
        "type": "object",
        "properties": {
            "need": {"type": "string"},
            "knowledge_base": {
                "type": "string",
                "enum": sorted({*CALL_NOESIS_TOOL_KNOWLEDGE_BASES, "orchestration"}),
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "required": ["need"],
    }


def _record_outcome_schema() -> dict:
    from knowledge_bases.tools.models import OutcomeSource

    return {
        "type": "object",
        "properties": {
            "trace_id": {"type": "string"},
            "success": {"type": "boolean"},
            "reason": {"type": "string"},
            "source": {
                "type": "string",
                "enum": [source.value for source in OutcomeSource],
                "default": OutcomeSource.user_correction.value,
            },
        },
        "required": ["trace_id", "success"],
    }


def sync_catalog(
    repository: ToolRepository | None = None, index: Any | None = None,
) -> list[Tool]:
    """MERGE every bootstrap skill into the graph and refresh the semantic
    catalog index (see `knowledge_bases.tools.catalog_index`; skipped
    gracefully without Milvus). Idempotent.

    Also asks any connected harness to re-fetch its tool list: a `new_tool`
    self-improvement re-syncs mid-session (see
    `knowledge_bases.tools.self_improvement_apply._sync_after_edit`), and the
    new tool is useless if the client's cached list predates it.
    """
    from graph import Graph
    from knowledge_bases.tools.catalog_index import CatalogIndex
    from knowledge_bases.tools.catalog_notify import notify_tool_list_changed

    skills = bootstrap_catalog()
    repository = repository or ToolRepository(Graph())
    for skill in skills:
        repository.sync_tool(skill)
    (index or CatalogIndex()).index_skills(skills)
    notify_tool_list_changed()
    return skills


# The `tools` knowledge base's own seed entries, kept in this module because
# the `_skill()` helper and schema builders they call are private to it.
TOOLS = [
        _skill(
        "tools.override_skill_input_schema",
        "tools",
        "Override Tool Input Schema",
        "Correct one skill's argument schema in the graph when the seeded schema "
        "is wrong or incomplete -- a missing property or wrong enum makes a "
        "working tool uncallable. The seeded base is left intact so a catalog "
        "re-sync cannot clobber the fix, and the override is validated against "
        "the handler's real required arguments. Pass an empty schema to clear it. "
        "Cannot change what the skill does: the handler is untouched.",
        _override_input_schema_schema(),
        "knowledge_bases.tools.mcp_tools:override_skill_input_schema_tool",
    ),

    _skill(
        "tools.retire_skill",
        "tools",
        "Retire Tool",
        "Stop listing and dispatching one skill across every knowledge base, "
        "without deleting it -- for a tool that misfires or has been superseded. "
        "Its node, description history, and every Span that called it stay "
        "intact, and it can be reactivated.",
        {
            "type": "object",
            "properties": {"key": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["key"],
        },
        "knowledge_bases.tools.mcp_tools:retire_skill_tool",
    ),

    _skill(
        "tools.reactivate_skill",
        "tools",
        "Reactivate Tool",
        "Return a previously retired skill to tool discovery and dispatch.",
        {
            "type": "object",
            "properties": {"key": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["key"],
        },
        "knowledge_bases.tools.mcp_tools:reactivate_skill_tool",
    ),

    _skill(
        "tools.get_span",
        "tools",
        "Get Span",
        "Fetch one execution span by id, including raw input/output.",
        {
            "type": "object",
            "properties": {"span_id": {"type": "string"}},
            "required": ["span_id"],
        },
        "knowledge_bases.tools.mcp_tools:get_span_tool",
    ),

    _skill(
        "tools.get_trace",
        "tools",
        "Get Trace",
        "Fetch every span belonging to one trace, ordered by start time.",
        TRACE_ID_INPUT,
        "knowledge_bases.tools.mcp_tools:get_trace_tool",
    ),

    _skill(
        "tools.get_trace_metrics",
        "tools",
        "Get Trace Metrics",
        "Get the current success verdict, resource usage, and tool-call "
        "count for one trace.",
        TRACE_ID_INPUT,
        "knowledge_bases.tools.mcp_tools:get_trace_metrics_tool",
    ),

    _skill(
        "tools.propose_self_improvements",
        "tools",
        "Propose Self Improvements",
        "Analyze recent skill execution spans and routing traces for "
        "recurring failures, corrections, or low-confidence/ambiguous "
        "routing, and file reviewable ReviewFinding suggestions (with an "
        "LLM-drafted diagnosis and suggested wording/schema change) in "
        "the noesis-self BMAD project. Never edits the skill or "
        "capability catalog directly; use dry_run to preview without "
        "filing.",
        {
            "type": "object",
            "properties": {
                "lookback_days": {"type": "integer", "minimum": 1, "maximum": 90},
                "min_calls": {"type": "integer", "minimum": 1},
                "min_traces": {"type": "integer", "minimum": 1},
                "dry_run": {"type": "boolean"},
            },
        },
        "knowledge_bases.tools.mcp_tools:propose_self_improvements_tool",
    ),

    _skill(
        "tools.apply_self_improvement",
        "tools",
        "Apply Self Improvement",
        "Apply one already-filed ReviewFinding's suggested wording/"
        "schema change as a real, git-committed source edit to "
        "catalog.py/profiles.py, and snapshot the metric it was drafted "
        "from so a later reevaluate_self_improvements call can measure "
        "whether it helped. Fully automatic -- no separate proposal/"
        "confirmation step -- unlike every other Noesis write.",
        {
            "type": "object",
            "properties": {"finding_key": {"type": "string"}},
            "required": ["finding_key"],
        },
        "knowledge_bases.tools.mcp_tools:apply_self_improvement_tool",
    ),

    _skill(
        "tools.retire_stale_self_improvements",
        "tools",
        "Retire Stale Self-Improvements",
        "Close open self-improvement findings whose signal no longer holds -- "
        "the target recovered, or it is no longer a registered tool.",
        {"type": "object", "properties": {}},
        "knowledge_bases.tools.mcp_tools:retire_stale_self_improvements_tool",
    ),

    _skill(
        "tools.reevaluate_self_improvements",
        "tools",
        "Reevaluate Self Improvements",
        "Re-measure every applied self-improvement finding whose "
        "cooldown has elapsed, and automatically keep it or git-revert "
        "it based on whether the metric it was drafted from actually "
        "improved. The 'did it help?' step that closes the "
        "self-improvement loop.",
        {"type": "object", "properties": {}},
        "knowledge_bases.tools.mcp_tools:reevaluate_self_improvements_tool",
    ),
]


# Composition order for seeded tool slices: the historical bootstrap order,
# kept stable so the catalog (and anything ordered off it) does not drift
# when packages are added or reordered in `registry.DEFAULT_PACKAGES`.
# Packages not listed here are appended in registration order.
_CATALOG_PACKAGE_ORDER = (
    "semantic_layer",
    "business_dictionary",
    "epistemology",
    "strategy",
    "project_management",
    "tools",
    "methodology",
    "orchestration",
    "development",
)
