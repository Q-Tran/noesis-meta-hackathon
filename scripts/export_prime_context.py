"""Export Noesis's context plane into a Prime Agent workspace.

Noesis targets Prime Agent exclusively: Prime owns execution (agent loop,
kernel, sub-agents, daemon continuity, heartbeats, schedules, autonomous
budgets), Noesis owns context (experts, tools, durable instructions,
cross-session memory). See
`knowledge_bases/orchestration/prime_export.py` for the reasoning.

Writes, all generated:

- `.prime/agent/AGENTS.md` -- the orchestrator's *base* instructions
- `.prime/agent/skills/noesis-<kb>/SKILL.md` -- one Agent Skills *stub* per
  knowledge base, for Prime's progressive disclosure. The instructions
  themselves live in the graph as `AgentSkill` nodes and are fetched on
  demand, so retiring a tool or rewording a skill needs no re-export.
- `.prime/agent/settings.json` -- points Prime at those skills (other keys kept)

Then prints the one-time `prime-agent mcp add` command that registers Noesis.
That is user-level and deliberately not written here: Prime ignores
project-level `mcpServers` entries so a repository cannot start a process on a
user's behalf.

Also removes generated artifacts from the retired OpenCode/Claude Code/Pi
adapters, so a workspace cannot keep running off a stale roster.

You do NOT need to re-run this after creating an expert, retiring a skill,
correcting a schema, or a self-improvement refining a prompt -- all of that is
fetched live. Re-run it when the orchestrator's *base* prompt changes in
`experts_catalog.py` or a knowledge base's name/description changes in
`knowledge_bases/registry.py` (`scripts/sync_tools_catalog.py` does this for
you when it detects either). Idempotent.

After writing, it asks any running Prime agent to `/reload`: Prime reads skill
frontmatter only at startup or on that command, so without it the new
descriptions would wait for the next session. Best-effort; `--no-reload`
skips it.

    python scripts/export_prime_context.py [--root PATH] [--python PATH] [--no-reload]
"""
import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from graph import Graph
from knowledge_bases.orchestration.prime_export import (
    describe_reload,
    export_prime_context,
    mcp_setup_command,
    trigger_prime_reload,
)
from knowledge_bases.orchestration.repository import RoutingRepository
from knowledge_bases.tools.repository import ToolRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--python",
        help="Interpreter Prime should launch the MCP server with "
             "(default: this checkout's .venv).",
    )
    parser.add_argument(
        "--keep-retired", action="store_true",
        help="Do not remove generated OpenCode/Claude Code/Pi artifacts.",
    )
    parser.add_argument(
        "--no-reload", action="store_true",
        help="Do not ask running Prime agents to /reload after writing.",
    )
    args = parser.parse_args()

    with Graph() as graph:
        experts = RoutingRepository(graph).load_experts(include_unroutable=True)
        skills = [
            {
                "key": skill.key,
                "knowledge_base": skill.knowledge_base,
                "operation": skill.key.split(".", 1)[1] if "." in skill.key else skill.key,
                "description": skill.description,
            }
            for skill in ToolRepository(graph).load_tools()
        ]
    if not experts:
        raise SystemExit("No experts in the graph; run scripts/sync_tools_catalog.py first.")

    written = export_prime_context(
        experts, skills, args.root, prune_retired=not args.keep_retired,
    )
    for path in written:
        print(f"wrote {path.relative_to(args.root)}")
    if not args.no_reload:
        print(describe_reload(trigger_prime_reload(args.root)))

    command = mcp_setup_command(args.root.resolve(), python=args.python)
    print(
        "\nRegister the MCP server once (user-level; Prime ignores project-level\n"
        "mcpServers so a repo cannot start a process on your behalf):\n\n"
        f"  {shlex.join(command)}\n"
    )


if __name__ == "__main__":
    main()
