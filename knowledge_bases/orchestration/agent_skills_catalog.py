"""Bootstrap catalog of the harness `AgentSkill` bundles, built from the graph.

Seed data, like `experts_catalog.py` and `knowledge_bases/tools/catalog.py`:
`sync_agent_skills()` MERGEs these into Neo4j, and from then on the graph is
what a harness reads.

The bodies are *derived*, not hand-written. One skill is generated per
knowledge base that owns callable tools, and its body lists that knowledge
base's current tools straight from `Tool` nodes. So retiring a tool or
rewording its description changes what the skill says on the next lookup,
with no file to regenerate.
"""
from __future__ import annotations

from typing import Any

from graph import Graph
from knowledge_bases.orchestration.agent_skills import AgentSkillRepository
from knowledge_bases.orchestration.models import (
    MAX_SKILL_DESCRIPTION_CHARS,
    AgentSkill,
)
from knowledge_bases.registry import KNOWLEDGE_BASES

SKILL_KEY_PREFIX = "noesis"


def skill_name(knowledge_base: str) -> str:
    """The harness-visible skill name. Underscores are illegal in a skill
    name, and the name must equal its directory, so both are derived here."""
    return f"{SKILL_KEY_PREFIX}-{knowledge_base.replace('_', '-')}"


def render_description(name: str, description: str) -> str:
    """The one line a harness puts in the startup prompt.

    Truncated on a word boundary rather than refused: a knowledge base
    description is human prose that may legitimately grow past the harness
    limit, and losing the tail of a sentence is better than losing the whole
    skill (an over-long description makes a harness warn and drop it).
    """
    text = " ".join(
        f"{description} Use for questions about {name.lower()}. "
        "Tools are graph-registered and called through call_noesis_tool.".split()
    )
    if len(text) <= MAX_SKILL_DESCRIPTION_CHARS:
        return text
    clipped = text[: MAX_SKILL_DESCRIPTION_CHARS - 1]
    return clipped[: clipped.rfind(" ")].rstrip(",;:") + "\u2026"


def render_body(knowledge_base: str, name: str, tools: list[dict[str, Any]]) -> str:
    """The instructions a harness loads when the model opens the skill.

    Rendered from the tools passed in, which the caller read from the graph a
    moment earlier -- this is the content that used to be frozen into a file at
    export time.
    """
    lines = [
        f"# {name}",
        "",
        f"Bounded tools owned by the `{knowledge_base}` knowledge base. Call them "
        "from the Python kernel through the `noesis` MCP server, passing your "
        "`expert_key` and `trace_id`:",
        "",
        "```python",
        'json.loads(await mcp.call_tool("noesis", "call_noesis_tool", {',
        f'    "knowledge_base": "{knowledge_base}",',
        '    "operation": "<one of the tools below>",',
        '    "arguments": {...},',
        '    "expert_key": expert_key, "trace_id": trace_id,',
        "}))",
        "```",
        "",
        "## Tools",
        "",
    ]
    for tool in tools:
        lines.append(f"- **`{tool['operation']}`** -- {tool['description']}")
    lines += [
        "",
        "This list was read from the graph when you opened this skill, so it is "
        "current. Tools can still be added, retired, or reworded at any time: "
        "prefer `search_noesis_tools` when the right operation is not obvious, "
        "and never conclude a capability is missing without asking.",
    ]
    return "\n".join(lines)


def bootstrap_agent_skills(tools: list[dict[str, Any]]) -> list[AgentSkill]:
    """One instruction bundle per knowledge base that owns callable tools."""
    by_kb: dict[str, list[dict[str, Any]]] = {}
    for tool in tools:
        by_kb.setdefault(tool["knowledge_base"], []).append(tool)

    skills = []
    for kb_key, kb_tools in sorted(by_kb.items()):
        kb = KNOWLEDGE_BASES.get(kb_key)
        if kb is None:
            continue
        skills.append(AgentSkill(
            key=skill_name(kb_key),
            name=skill_name(kb_key),
            description=render_description(kb.name, kb.description),
            body=render_body(kb_key, kb.name, kb_tools),
            knowledge_base=kb_key,
        ))
    return skills


def _tool_rows(graph: Graph) -> list[dict[str, Any]]:
    from knowledge_bases.tools.repository import ToolRepository

    return [
        {
            "key": tool.key,
            "knowledge_base": tool.knowledge_base,
            "operation": tool.key.split(".", 1)[1] if "." in tool.key else tool.key,
            "description": tool.description,
        }
        for tool in ToolRepository(graph).load_tools()
    ]


def sync_agent_skills(graph: Graph) -> list[AgentSkill]:
    """Regenerate every instruction bundle from the current tool catalog.

    Also writes `(AgentSkill)-[:DOCUMENTS]->(Tool)` edges for the tools in the
    skill's knowledge base, so "a skill contains its tools" is graph-true, not
    just a convention of the generated body.
    """
    tools = _tool_rows(graph)
    skills = bootstrap_agent_skills(tools)
    repository = AgentSkillRepository(graph)
    for skill in skills:
        repository.sync_skill(skill)
        for tool in tools:
            if tool["knowledge_base"] == skill.knowledge_base:
                graph.upsert_rel(
                    "AgentSkill", {"key": skill.key},
                    "DOCUMENTS", "Tool", {"key": tool["key"]},
                )
    return skills
