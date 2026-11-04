"""Export Noesis's context plane into a Prime Agent workspace.

Noesis targets one harness: **Prime Agent**. The division is deliberate and is
the reason this file is small.

    Prime owns execution        Noesis owns context
    -------------------        -------------------
    agent loop                 which experts exist, and what each knows
    Python kernel              which tools are callable, and how
    sub-agent lifecycle        durable instructions
    daemon continuity          cross-session memory
    compaction                 what is unresolved
    heartbeats / schedules     what improved and why
    autonomous budgets

Every previous adapter tried to teach a harness to *be* Prime -- OpenCode got a
worker pool because it lacks spawn-by-prompt, Pi got a whole MCP client and an
in-process delegation loop because it has neither. That work is retired: Prime
already has sub-agents, a daemon, heartbeats, schedules, and bounded autonomous
mode, so reimplementing them per harness was effort spent reaching a baseline
Prime ships with.

What Prime does *not* have is memory that outlives a checkout. Its harness state
is `harness_state.json` and `kernel-state.dill` on one machine, per session. So
Noesis supplies the half Prime is missing, and nothing else.

## What gets written

`.prime/agent/` in the workspace, all of it generated:

- `AGENTS.md` (repo root, where Prime actually looks) -- project context. Carries the
  orchestrator's *base* instructions only; everything refinable is fetched live.
- `skills/<product>-<kb>/SKILL.md` -- one Agent Skills entry per knowledge base
  (`<product>` is the profile's `mcp_server_name`, `noesis` here),
  so Prime's own progressive disclosure surfaces Noesis capabilities the way it
  surfaces any other skill. Each is a *stub*: the frontmatter is a cache (Prime
  needs a name and description before the session starts and cannot resolve
  either at runtime), while the body tells the model to fetch the real
  instructions from the graph with `get_noesis_agent_skill`. Prime reads a body
  on demand, which is what makes that indirection possible -- see
  `docs/architecture/prime_skill_loading.md`.
- `settings.json` -- points Prime at those skills.

Notably absent: a bridge extension. Prime already speaks MCP, so Noesis is
registered once as an ordinary MCP server:

    prime-agent mcp add noesis --cwd <repo> -- <python> -m knowledge_bases.orchestration.mcp_server

That has to be a *user-level* registration, not something this export writes,
because Prime deliberately ignores project-level `mcpServers` entries -- "a
repository cannot start a local process or shadow a user server". Respecting
that boundary rather than routing around it with an extension is the right call:
it is the same posture Noesis takes about bounded tools.

An earlier version of this module generated a TypeScript bridge that
reimplemented an MCP client and registered each tool with `pi.registerTool`.
It was deleted for the same reason the OpenCode and Pi adapters were: it
rebuilt something Prime already had.

The generated artifacts are a *cache* of the graph, not a second source of
truth. Anything that can change at runtime -- experts, skill descriptions,
schemas, prompt overlays, session memory -- is fetched through
`get_noesis_context` and the bounded tools, so re-export is only needed when the
immutable base prompt or a knowledge base's registry name/description changes.
`stale_exports` detects both, `scripts/sync_tools_catalog.py` re-exports when
it does, and `trigger_prime_reload` asks any running Prime agent to `/reload`
so the change reaches a session already in progress -- Prime reads frontmatter
only at startup or on that command.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_bases.orchestration.experts_catalog import (
    EXPERT_LOOP_CONTRACT,
    ORCHESTRATOR_KEY,
)
from knowledge_bases.orchestration.models import ExpertDefinition
from knowledge_bases.registry import product_profile

PRIME_DIR = ".prime/agent"
# Prime discovers project context as `AGENTS.md` in the working directory and
# its parents (plus `~/.prime/agent/AGENTS.md` globally) -- NOT inside
# `.prime/agent/`. Writing it there produced a file Prime never read, so the
# orchestrator never learned the context plane existed.
AGENTS_PATH = "AGENTS.md"
SKILLS_DIR = f"{PRIME_DIR}/skills"
SETTINGS_PATH = f"{PRIME_DIR}/settings.json"

# The `settings.json` entry this export owns. Paths in a project settings file
# resolve relative to `.prime/agent`, so it is written relative.
SKILLS_ENTRY = "./skills"

# An entry a previous version of this export wrote, before Prime's native MCP
# support made the bridge extension redundant. Removed on upgrade so settings
# do not keep pointing at a file that is no longer generated.
RETIRED_EXTENSION_ENTRY = "./extensions/noesis.ts"

# The harness-facing name of the installed product, read from its profile
# (`knowledge_bases/product.py`) so a differently-named product exports under
# its own name instead of overwriting this one's registration and skills.
MCP_SERVER_NAME = product_profile().mcp_server_name


def _skill_dir_prefix() -> str:
    """Prefix for exported Agent Skill directories: `<server>-<kb>`."""
    return f"{MCP_SERVER_NAME}-"

GENERATED_MARKER = "GENERATED by scripts/export_prime_context.py"

# Retired harness artifacts, removed on export so a workspace cannot end up
# driven by a stale roster from a harness Noesis no longer targets. The
# `.prime/agent/extensions/noesis.ts` entry covers the short-lived bridge
# extension this module used to generate before Prime's native MCP support
# made it redundant.
RETIRED_ARTIFACTS = (
    ".opencode/agents", ".claude/agents",
    ".pi/extensions/noesis.ts", ".prime/agent/extensions/noesis.ts",
)


def mcp_setup_command(repo_root: Path, python: str | None = None) -> list[str]:
    """The one-time, user-level command that registers Noesis with Prime.

    Deliberately returned rather than executed: Prime ignores project-level
    `mcpServers` entries so a repository cannot start a process on a user's
    behalf, and writing `~/.prime/agent/settings.json` from here would route
    around exactly that protection.
    """
    interpreter = python or str(repo_root / ".venv" / "bin" / "python")
    return [
        "prime-agent", "mcp", "add", MCP_SERVER_NAME,
        "--cwd", str(repo_root),
        "--", interpreter, "-m", "knowledge_bases.orchestration.mcp_server",
    ]

_MD_HEADER = (
    f"<!-- {GENERATED_MARKER} from graph KnowledgeBaseExpert `{{key}}` v{{version}} -- "
    "edit knowledge_bases/orchestration/experts_catalog.py (or revise the expert "
    "in the graph) and re-export; do not hand-edit. Everything refinable -- "
    "experts, skills, overlays, memory -- is fetched live and needs no "
    "re-export. -->"
)

# What the orchestrator is told before its own graph-defined prompt. Short on
# purpose: it establishes only that context comes from the graph, then gets out
# of the way.
CONTEXT_CONTRACT = """
## Noesis is your context

**Before answering any question about this project -- its experts, tools,
metrics, decisions, delivery, documents, or prior sessions -- query the graph.
Never answer from inspection of the filesystem, `pip`, or your own assumptions;
those describe the code, not the knowledge it serves.**

Your experts, tools, durable instructions, and memory of prior work all come
from the Noesis graph, not from this file.

Noesis is registered as the `noesis` MCP server. Key control-plane tools:
- `get_noesis_context`: Call once at session start. It returns the routable experts and what each handles, the callable tool surface, supplemental instructions, and resumable sessions.
- `route_noesis_question`: Retrieval-based routing to rank experts against a question.
- `search_noesis_tools`: Plain-language semantic search across all callable tools in the graph.
- `get_noesis_expert`: Fetch an expert's live system prompt, allowed tools, and entry skills.
- `spawn_ephemeral_subagent`: Spawn an isolated subagent under an expert's persona to solve a focused task.
- `register_tool`: Register a new tool or capability directly into the graph so any harness or agent can discover and call it.

Because that is graph data, it is current: an expert created an hour ago, a tool
retired this morning, and a prompt refinement measured last week are all already
in effect. Never assume a capability is missing because it is absent from this
file -- ask the graph.

## Delegating

Route first, then delegate to one subagent per routed delegation without context pollution:

1. Call `route_noesis_question` (or `search_noesis_tools`) to discover the relevant expert and starting tool.
2. Delegate using either:
   - **Noesis MCP Tool**: Call `spawn_ephemeral_subagent(expert_key=d["knowledge_base"], task=question)` (or pass `operation` and `starting_arguments`). Noesis runs the expert in an isolated scope under its graph prompt, records a `DelegationResult`, and returns the synthesized answer directly.
   - **Harness-Native Subagent**: If your harness provides a native subagent tool (e.g. OpenCode's `task` tool), spawn a child session instructing it:
     `"Act as Noesis expert '{knowledge_base}'. Load your instructions and tools with get_noesis_expert('{knowledge_base}'). Question: {question}. Trace: {trace_id}"`
3. Synthesize your final response from the structured results (`get_delegation_results` with `trace_id`).
""".strip()

WORKER_CONTRACT = """
## Acting as a Noesis expert

You have been spawned to act as one graph-defined expert. You have no domain
knowledge of your own until you load it:

1. Call `get_noesis_expert(key=expert_key)` to retrieve that expert's instructions, its entry skills in order, and the tools it may call with their schemas. That call also registers which expert you are acting as, so your work is attributed.
2. Adopt those instructions for this delegation. They supersede anything you assumed about the domain.
3. Pass the same `expert_key` and `trace_id` on every Noesis tool call so your work stays correlated with the routing decision.
4. Findings, resolved entities, and tool execution spans are automatically recorded to the knowledge graph by the harness — manual result recording is not required.
""".strip()


def render_agents_md(orchestrator: ExpertDefinition) -> str:
    """`AGENTS.md`, which Prime loads natively as project context.

    Only the *base* prompt is written; overlays and prefixes are fetched via
    `get_noesis_context`, so a refinement never requires touching this file.
    """
    return "\n\n".join([
        _MD_HEADER.format(key=orchestrator.key, version=orchestrator.version),
        "# Noesis",
        CONTEXT_CONTRACT,
        WORKER_CONTRACT,
        "## Orchestrator instructions",
        orchestrator.system_prompt.rstrip(),
        "## Expert loop contract",
        EXPERT_LOOP_CONTRACT,
    ]) + "\n"


def _yaml_scalar(value: str) -> str:
    """Quote a one-line string so YAML frontmatter parses as a plain scalar.

    Prime parses SKILL.md frontmatter with a real YAML parser, so a description
    is not free text: an unquoted `key: value` breaks on the first ": " it
    contains ("Mixture-of-experts control plane: routing, ..." raised "Nested
    mappings are not allowed in compact mappings" and the skill was dropped
    from the roster entirely). Knowledge base descriptions are prose written for
    humans, so assume they contain YAML metacharacters and always quote.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{" ".join(escaped.split())}"'


def render_skill(knowledge_base: str, name: str, description: str,
                 skills: list[dict[str, Any]] | None = None) -> str:
    """One Agent Skills entry per knowledge base -- a *stub*, not a snapshot.

    Prime loads only the frontmatter into the startup prompt and reads the body
    on demand, when the model decides the skill is relevant (verified against
    Prime's loader: `loadSkillFromFile` reads `name`/`description` only, and the
    prompt tells the model to inspect the skill file itself -- see
    `docs/architecture/prime_skill_loading.md`).

    That on-demand read is the seam this export exploits. The body no longer
    lists the tools; it tells the model to fetch the current instructions from
    the graph. So a retired tool, a reworded description, or a refined
    instruction takes effect on the next lookup, with nothing to regenerate.

    The frontmatter stays a cache because it *must*: a harness needs a name and
    a description before a session starts, and cannot resolve either at runtime.
    `skills` is accepted and ignored for callers that still pass a tool list.
    """
    skill_name = f"{_skill_dir_prefix()}{knowledge_base.replace('_', '-')}"
    lines = [
        "---",
        f"name: {skill_name}",
        "description: " + _yaml_scalar(
            f"{description} Use for questions about {name.lower()}. "
            "Tools are graph-registered and called through call_noesis_tool."
        ),
        "---",
        f"<!-- {GENERATED_MARKER} -- deliberately a stub. The instructions live "
        "in the graph and are fetched on demand; only the frontmatter is a "
        "cache, because a harness needs it before a session starts. -->",
        "",
        f"# {name}",
        "",
        "**Fetch these instructions from the graph before acting on this "
        "skill.** This file is a pointer, not the content: the authoritative "
        f"body for `{skill_name}` is an `AgentSkill` node, so it can be "
        "reworded or corrected without regenerating anything on disk.",
        "",
        "```python",
        "import json",
        "# A direct tool call, not call_noesis_tool: orchestration's own tools",
        "# are served by this same server, so they are callable directly.",
        'skill = json.loads(await mcp.call_tool(',
        '    "noesis", "get_noesis_agent_skill",',
        f'    {{"key": "{skill_name}"}},',
        "))",
        'print(skill["body"])',
        "```",
        "",
        f"The body lists every bounded tool the `{knowledge_base}` knowledge "
        "base currently owns, read from the live catalog at the moment you "
        "call -- including any tool added or retired since this file was "
        "written. Call those tools through `call_noesis_tool`, passing your "
        "`expert_key` and `trace_id`.",
        "",
        "If that call fails, fall back to `search_noesis_tools` with a short "
        "description of what you need. Never conclude a capability is missing "
        "because it is absent from this file -- this file lists nothing.",
    ]
    return "\n".join(lines) + "\n"


def render_settings(existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """`.prime/agent/settings.json`, pointing Prime at the generated resources.

    Paths in a project settings file resolve relative to `.prime/agent`, so they
    are written relative. Every other key of `existing` is preserved -- a user's
    model, theme, and compaction choices are theirs.
    """
    settings = dict(existing or {})

    def _without(entries: list[Any], managed: str) -> list[Any]:
        # Match the exact path this function writes, not a substring: the skills
        # entry is "./skills", which carries no "noesis" marker, so a substring
        # filter silently appended a duplicate on every re-export.
        return [e for e in entries if str(e).rstrip("/") != managed.rstrip("/")]

    settings["skills"] = _without(settings.get("skills", []), SKILLS_ENTRY) + [SKILLS_ENTRY]

    # Upgrade path: drop the entry an older version wrote for the bridge
    # extension, without disturbing a user's own extensions.
    extensions = _without(settings.get("extensions", []), RETIRED_EXTENSION_ENTRY)
    if extensions:
        settings["extensions"] = extensions
    else:
        settings.pop("extensions", None)
    return settings


def export_prime_context(
    experts: list[ExpertDefinition],
    skills: list[dict[str, Any]],
    root: Path,
    prune_retired: bool = True,
) -> list[Path]:
    """Write the whole `.prime/agent/` context export. Returns paths written."""
    from knowledge_bases.registry import KNOWLEDGE_BASES

    orchestrator = next((e for e in experts if e.key == ORCHESTRATOR_KEY), None)
    if orchestrator is None:
        raise ValueError(
            f"No {ORCHESTRATOR_KEY!r} expert in the graph; run "
            "scripts/sync_tools_catalog.py first."
        )

    written: list[Path] = []
    (root / PRIME_DIR).mkdir(parents=True, exist_ok=True)

    agents = root / AGENTS_PATH
    agents.write_text(render_agents_md(orchestrator))
    written.append(agents)

    # One skill per knowledge base that actually owns callable tools.
    by_kb: dict[str, list[dict[str, Any]]] = {}
    for skill in skills:
        by_kb.setdefault(skill["knowledge_base"], []).append(skill)
    skills_root = root / SKILLS_DIR
    expected_dirs = set()
    for kb_key, kb_skills in sorted(by_kb.items()):
        kb = KNOWLEDGE_BASES.get(kb_key)
        if kb is None:
            continue
        directory = skills_root / f"{_skill_dir_prefix()}{kb_key.replace('_', '-')}"
        directory.mkdir(parents=True, exist_ok=True)
        expected_dirs.add(directory.name)
        path = directory / "SKILL.md"
        path.write_text(render_skill(kb_key, kb.name, kb.description, kb_skills))
        written.append(path)

    # Drop skills for knowledge bases that no longer expose tools.
    if skills_root.is_dir():
        for stale in skills_root.iterdir():
            if stale.is_dir() and stale.name.startswith(_skill_dir_prefix()) \
                    and stale.name not in expected_dirs:
                for child in stale.rglob("*"):
                    if child.is_file():
                        child.unlink()
                stale.rmdir()

    settings_path = root / SETTINGS_PATH
    existing = json.loads(settings_path.read_text()) if settings_path.exists() else None
    settings_path.write_text(json.dumps(render_settings(existing), indent=2) + "\n")
    written.append(settings_path)

    if prune_retired:
        _prune_retired(root)
    return written


def _prune_retired(root: Path) -> list[Path]:
    """Remove generated artifacts of harnesses Noesis no longer targets.

    Leaving them would let a workspace be driven by a stale roster -- an
    `.opencode/agents/` pool that predates the pivot would still work, and would
    silently bypass the context plane.
    """
    removed = []
    for relative in RETIRED_ARTIFACTS:
        target = root / relative
        if target.is_file():
            if GENERATED_MARKER in target.read_text() or "GENERATED" in target.read_text():
                target.unlink()
                removed.append(target)
        elif target.is_dir():
            for path in sorted(target.glob("*.md")):
                if "GENERATED" in path.read_text():
                    path.unlink()
                    removed.append(path)
            if not any(target.iterdir()):
                target.rmdir()
        # A pruned file can leave its directory behind (e.g. the bridge
        # extension's `.prime/agent/extensions/`); drop it when it is empty so
        # the workspace does not keep a directory nothing writes to.
        parent = target.parent
        if parent != root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    return removed


def stale_exports(
    experts: list[ExpertDefinition],
    root: Path,
    skills: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Whether the generated files still match the code and the graph.

    Two things live in files and can therefore go stale: the orchestrator's
    base prompt in `AGENTS.md` (versioned against the graph), and each
    knowledge base's `SKILL.md` stub, whose frontmatter is rendered from
    `KNOWLEDGE_BASES` in `knowledge_bases/registry.py`. Experts, skill bodies,
    overlays, and memory are all fetched live and never appear here.

    `skills` (the tool rows `export_prime_context` takes) says which knowledge
    bases should have a stub; without it only the stubs already on disk are
    checked, so a caller with no graph access can still detect drift. Used by
    `_sync_after_edit` and `scripts/sync_tools_catalog.py` so a change that
    needs a re-export says so instead of silently having no effect.
    """
    import re

    from knowledge_bases.registry import KNOWLEDGE_BASES

    warnings = []
    orchestrator = next((e for e in experts if e.key == ORCHESTRATOR_KEY), None)
    if orchestrator is None:
        return warnings
    agents = root / AGENTS_PATH
    if not agents.exists():
        warnings.append(f"{AGENTS_PATH} is missing; run scripts/export_prime_context.py")
        return warnings
    match = re.search(r"`[a-z_]+` v(\d+)", agents.read_text())
    exported = int(match.group(1)) if match else None
    if exported != orchestrator.version:
        warnings.append(
            f"{AGENTS_PATH} was exported from {ORCHESTRATOR_KEY} v{exported}, graph "
            f"is at v{orchestrator.version}; re-run scripts/export_prime_context.py "
            "to pick up the new base instructions (everything else is already live)"
        )

    skills_root = root / SKILLS_DIR
    if skills is not None:
        expected = {s["knowledge_base"] for s in skills} & set(KNOWLEDGE_BASES)
    elif skills_root.is_dir():
        expected = {
            d.name[len(_skill_dir_prefix()):].replace("-", "_")
            for d in skills_root.iterdir()
            if d.is_dir() and d.name.startswith(_skill_dir_prefix())
        } & set(KNOWLEDGE_BASES)
    else:
        expected = set()
    for kb_key in sorted(expected):
        kb = KNOWLEDGE_BASES[kb_key]
        path = skills_root / f"{_skill_dir_prefix()}{kb_key.replace('_', '-')}" / "SKILL.md"
        relative = path.relative_to(root).as_posix()
        if not path.exists():
            warnings.append(f"{relative} is missing; run scripts/export_prime_context.py")
        elif path.read_text() != render_skill(kb_key, kb.name, kb.description):
            warnings.append(
                f"{relative} no longer matches the {kb_key!r} registry entry; re-run "
                "scripts/export_prime_context.py so Prime sees the current name and "
                "description (it reads the frontmatter only at startup or /reload)"
            )
    return warnings


# The Prime CLI, resolved from PATH. Only the one-time `mcp add` and the
# post-export reload below ever invoke it.
PRIME_CLI = "prime-agent"
RELOAD_TIMEOUT_SECONDS = 10.0


def trigger_prime_reload(
    root: Path | None = None, *, run: Any = None,
) -> dict[str, Any]:
    """Ask every running Prime agent to `/reload` after the export was rewritten.

    Prime reads skill frontmatter from disk only at session start or on the
    `/reload` slash command -- there is no file watcher -- so without this a
    re-export takes effect on the *next* session while the current one keeps
    the old descriptions. `prime-agent send` delivers a message into a running
    agent's turn queue, and text starting with `/` is dispatched as a slash
    command whichever way it arrived, so sending "/reload" is the same as the
    user typing it.

    When `list --json` reports each agent's working directory, only agents
    running on `root` are targeted; an unrelated session in another repository
    should not be reloaded because this one changed.

    Best-effort, like `expert_admin._reindex_experts`: the CLI being absent, no
    daemon running, or a malformed reply is reported in the returned dict and
    never raised, because the files are already correct and the reload is only
    an accelerator for whoever is mid-session.
    """
    import subprocess

    run = run or subprocess.run
    report: dict[str, Any] = {"attempted": [], "reloaded": [], "errors": {}}
    try:
        listed = run(
            [PRIME_CLI, "list", "--json"], capture_output=True, text=True,
            timeout=RELOAD_TIMEOUT_SECONDS, check=False,
        )
        if listed.returncode != 0:
            report["errors"]["list"] = (listed.stderr or listed.stdout or "").strip()[:200]
            return report
        agents = _running_agents(json.loads(listed.stdout), root)
    except FileNotFoundError:
        report["errors"]["list"] = f"{PRIME_CLI} is not on PATH"
        return report
    except Exception as exc:  # noqa: BLE001 -- advisory only
        report["errors"]["list"] = f"{type(exc).__name__}: {exc}"[:200]
        return report

    for agent_id in agents:
        report["attempted"].append(agent_id)
        try:
            sent = run(
                [PRIME_CLI, "send", agent_id, "/reload"], capture_output=True,
                text=True, timeout=RELOAD_TIMEOUT_SECONDS, check=False,
            )
            if sent.returncode == 0:
                report["reloaded"].append(agent_id)
            else:
                report["errors"][agent_id] = (sent.stderr or sent.stdout or "").strip()[:200]
        except Exception as exc:  # noqa: BLE001 -- advisory only
            report["errors"][agent_id] = f"{type(exc).__name__}: {exc}"[:200]
    return report


def describe_reload(report: dict[str, Any]) -> str:
    """One line for the operator about what `trigger_prime_reload` did."""
    if report["reloaded"]:
        return f"asked {len(report['reloaded'])} running Prime agent(s) to /reload"
    if "list" in report["errors"]:
        return f"skipped Prime reload ({report['errors']['list']})"
    if report["attempted"]:
        return f"Prime reload failed for {', '.join(report['attempted'])}"
    return "no running Prime agents to reload"


def _running_agents(payload: Any, root: Path | None) -> list[str]:
    """Agent ids from `prime-agent list --json`.

    The reply is `{"sessions": [SessionSummary, ...]}` (Prime's
    `daemon-session-list.d.ts`): `id`, `cwd`, `lifecycle` (draft / live /
    archived), `isSessionActive`, `runtimeKind` (top-level / subagent). Only
    live, active, top-level sessions running under `root` are returned --
    RLM children reload with their parent, and a session in another repository
    should not be interrupted because this one changed. A bare list is
    accepted too, in case the envelope ever goes away.
    """
    entries = payload.get("sessions", []) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return []
    resolved_root = root.resolve() if root is not None else None
    ids = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        if entry.get("lifecycle") == "archived" or entry.get("isSessionActive") is False:
            continue
        if entry.get("runtimeKind") == "subagent":
            continue
        cwd = entry.get("cwd")
        if resolved_root is not None and isinstance(cwd, str):
            try:
                Path(cwd).resolve().relative_to(resolved_root)
            except ValueError:
                continue
        ids.append(entry["id"])
    return ids
