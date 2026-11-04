"""MERGE the bootstrap Tool and Expert catalogs into Neo4j.

Every MCP connector (`knowledge_bases/*/mcp_server.py`) reads its tool
surface live from graph `Tool` nodes, and the orchestrator serves expert
subagent definitions live from `KnowledgeBaseExpert` nodes -- see
`knowledge_bases/tools/catalog.py`, `knowledge_bases/orchestration/
experts_catalog.py`, and `knowledge_bases/mcp.py`. Run this after editing
either catalog (or a knowledge base's registry entry) to make the graph match
the code.

Afterwards it checks whether the generated Prime context export
(`AGENTS.md`, `.prime/agent/skills/noesis-*/SKILL.md`) still matches the code
and graph, and if not regenerates it and asks any running Prime agent to
`/reload` -- Prime reads skill frontmatter only at startup or on that command,
so this is what makes a renamed or redescribed knowledge base reach a session
already in progress. Pass `--no-auto-export` to keep the old two-step flow and
run `scripts/export_prime_context.py` yourself.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from graph import Graph
from knowledge_bases.orchestration.agent_skills_catalog import sync_agent_skills
from knowledge_bases.orchestration.experts_catalog import bootstrap_experts
from knowledge_bases.orchestration.prime_export import (
    describe_reload,
    export_prime_context,
    stale_exports,
    trigger_prime_reload,
)
from knowledge_bases.orchestration.repository import RoutingRepository
from knowledge_bases.tools.catalog import sync_catalog
from knowledge_bases.tools.repository import ToolRepository


def refresh_export_if_stale(graph: Graph, root: Path) -> list[str]:
    """Re-export and reload only when something in a file has drifted."""
    experts = RoutingRepository(graph).load_experts(include_unroutable=True)
    skills = [
        {
            "key": tool.key,
            "knowledge_base": tool.knowledge_base,
            "operation": tool.key.split(".", 1)[1] if "." in tool.key else tool.key,
            "description": tool.description,
        }
        for tool in ToolRepository(graph).load_tools()
    ]
    warnings = stale_exports(experts, root, skills)
    if not warnings:
        return ["Prime context export is current; nothing to regenerate."]
    lines = [f"stale: {w}" for w in warnings]
    for path in export_prime_context(experts, skills, root):
        lines.append(f"wrote {path.relative_to(root)}")
    lines.append(describe_reload(trigger_prime_reload(root)))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--no-auto-export", action="store_true",
        help="Do not regenerate the Prime context export or reload running "
             "agents when the export has drifted.",
    )
    args = parser.parse_args()

    synced = sync_catalog()
    print(f"Synced {len(synced)} tools into the graph Tool catalog.")
    with Graph() as graph:
        experts = bootstrap_experts()
        RoutingRepository(graph).sync_experts(experts)
        print(f"Synced {len(experts)} experts into the graph.")
        # AgentSkill bodies are derived from the tool catalog, so a knowledge
        # base that gains its first tool gains an instruction bundle here.
        # Without this, `export_prime_context.py` would write a SKILL.md stub
        # pointing at an `AgentSkill` node that does not exist, and the
        # harness's first `get_noesis_agent_skill` would fail -- which is
        # exactly what happened when `self_catalog` gained tools.
        skills = sync_agent_skills(graph)
        print(f"Synced {len(skills)} agent skills into the graph.")
        if not args.no_auto_export:
            for line in refresh_export_if_stale(graph, args.root):
                print(line)


if __name__ == "__main__":
    main()
