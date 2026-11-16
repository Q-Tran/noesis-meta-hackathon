"""Noesis as Prime Agent's source of context.

The division under test: Prime owns execution, Noesis owns context. So the
export has to (a) carry only what cannot change at runtime, (b) point Prime at
the graph for everything that can, and (c) add no capability of its own.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest

from knowledge_bases.orchestration.context_plane import (
    MAX_CONTEXT_EXPERTS,
    MAX_CONTEXT_TOOLS,
    get_noesis_context,
)
from knowledge_bases.orchestration.experts_catalog import (
    EXPERT_LOOP_CONTRACT,
    ORCHESTRATOR_KEY,
    bootstrap_experts,
    bootstrap_profiles,
)
from knowledge_bases.orchestration.models import ExpertDefinition
from knowledge_bases.orchestration.prime_export import (
    AGENTS_PATH,
    SETTINGS_PATH,
    describe_reload,
    export_prime_context,
    mcp_setup_command,
    render_agents_md,
    render_settings,
    render_skill,
    stale_exports,
    trigger_prime_reload,
)


def _orchestrator(**overrides) -> ExpertDefinition:
    base = dict(
        key=ORCHESTRATOR_KEY, name="Noesis Orchestrator", description="Primary agent.",
        system_prompt="Route and synthesize.", routable=False, version=3,
        entry_tools=["orchestration.route_noesis_question"],
        allowed_tools=["orchestration.*"],
    )
    base.update(overrides)
    return ExpertDefinition(**base)


def _expert(key="project_management", **overrides) -> ExpertDefinition:
    base = dict(
        key=key, name=key.title(), description="d", capabilities=["c"], priority=60,
        system_prompt="p", entry_tools=[f"{key}.thing"], allowed_tools=[f"{key}.*"],
    )
    base.update(overrides)
    return ExpertDefinition(**base)


def _skills():
    return [
        {"key": "project_management.list_project_blockers", "knowledge_base": "project_management",
         "operation": "list_project_blockers", "description": "Blocked work."},
        {"key": "epistemology.get_claim_provenance", "knowledge_base": "epistemology",
         "operation": "get_claim_provenance", "description": "Where it came from."},
    ]


# -- the context plane -------------------------------------------------------------

class PlaneGraph:
    """Serves experts, skills, and session rows for the plane's three readers."""

    def __init__(self, experts, skills, sessions=None):
        self.experts, self.skills, self.sessions = experts, skills, sessions or []

    def run(self, query, **params):
        if "MATCH (s:Tool {active: true})" in query:
            return self.skills
        if "MATCH (s:Session)" in query:
            return self.sessions
        if "MATCH (e:KnowledgeBaseExpert" in query:
            rows = [e.model_dump() for e in self.experts]
            if "key: $key" in query:
                rows = [r for r in rows if r["key"] == params.get("key")]
            if "$all OR coalesce(e.routable, true) = true" in query and not params.get("all"):
                rows = [r for r in rows if r.get("routable", True)]
            return rows
        return []


def _skill_row(key, kb):
    return {"key": key, "name": key, "version": "1", "knowledge_base": kb,
            "active": True, "description": "does a thing", "handler": "m:f",
            "input_schema": json.dumps({"type": "object", "properties": {}})}


def test_the_plane_returns_everything_a_session_needs_in_one_call():
    graph = PlaneGraph(
        [_orchestrator(), _expert()],
        [_skill_row("project_management.list_project_blockers", "project_management")],
    )
    plane = get_noesis_context({}, graph)

    assert set(plane) == {"source", "contract", "instructions", "experts", "tools", "memory"}
    assert plane["experts"][0]["key"] == "project_management"
    assert plane["experts"][0]["capabilities"] == ["c"]
    assert plane["tools"][0]["operation"] == "list_project_blockers"
    assert plane["instructions"]["base"] == "Route and synthesize."


def test_the_plane_excludes_the_orchestrator_from_the_routable_roster():
    """It is the primary agent, not something to delegate to."""
    graph = PlaneGraph([_orchestrator(), _expert()], [])
    plane = get_noesis_context({}, graph)

    assert [e["key"] for e in plane["experts"]] == ["project_management"]


def test_the_plane_separates_immutable_instructions_from_learned_ones():
    """A consumer needs the precedence, so concatenating blindly is wrong."""
    graph = PlaneGraph(
        [_orchestrator(prompt_overlays=["Learned guidance."],
                       prompt_prefix_overlays=["Deployment framing."])],
        [],
    )
    plane = get_noesis_context({}, graph)

    assert plane["instructions"]["base"] == "Route and synthesize."
    assert plane["instructions"]["supplemental"] == [
        "Deployment framing.", "Learned guidance.",
    ]


def test_the_plane_carries_expert_descriptions_but_not_full_prompts():
    """Prompts are fetched per delegation; shipping them all would make the
    plane enormous and instantly stale."""
    graph = PlaneGraph([_orchestrator(), _expert(system_prompt="SECRET DOMAIN RULES")], [])
    plane = get_noesis_context({}, graph)

    assert "SECRET DOMAIN RULES" not in json.dumps(plane["experts"])


def test_the_plane_can_be_requested_in_parts():
    graph = PlaneGraph([_orchestrator(), _expert()], [])
    plane = get_noesis_context({"include": ["experts"]}, graph)

    assert "experts" in plane
    assert "tools" not in plane and "memory" not in plane


def test_the_plane_still_answers_to_the_old_skills_section_name():
    """`skills` was this section's name when a callable operation was called a
    skill. An older caller asking for it must still get the tool surface rather
    than silently nothing."""
    graph = PlaneGraph([_orchestrator()], [_skill_row("project_management.x", "project_management")])

    plane = get_noesis_context({"include": ["skills"]}, graph)

    assert plane["tools"][0]["operation"] == "x"


def test_the_plane_is_bounded():
    """It is read at every session start, so it cannot grow without limit."""
    graph = PlaneGraph(
        [_orchestrator()] + [_expert(key=f"e{i}") for i in range(MAX_CONTEXT_EXPERTS + 20)],
        [_skill_row(f"project_management.s{i}", "project_management") for i in range(MAX_CONTEXT_TOOLS + 50)],
    )
    plane = get_noesis_context({}, graph)

    assert len(plane["experts"]) == MAX_CONTEXT_EXPERTS
    assert len(plane["tools"]) == MAX_CONTEXT_TOOLS


def test_the_plane_states_the_division_of_responsibility():
    plane = get_noesis_context({}, PlaneGraph([_orchestrator()], []))

    assert "source of context" in plane["contract"]
    assert "Prime Agent owns execution" in plane["contract"]


def test_a_missing_orchestrator_degrades_rather_than_raising():
    """A fresh graph, mid-sync: an empty plane beats a crashed session start."""
    plane = get_noesis_context({}, PlaneGraph([], []))

    assert plane["instructions"] == {"base": "", "supplemental": [], "version": 0}


# -- AGENTS.md ---------------------------------------------------------------------

def test_agents_md_carries_the_base_prompt_and_both_contracts():
    text = render_agents_md(_orchestrator())

    assert "Route and synthesize." in text
    assert "get_noesis_context" in text          # how to load context
    assert "get_noesis_expert" in text           # how a child adopts an expert
    assert "spawn" in text                       # delegation subagent spawning
    assert EXPERT_LOOP_CONTRACT in text
    assert f"`{ORCHESTRATOR_KEY}` v3" in text


def test_agents_md_omits_everything_refinable():
    """The whole point: a refinement must not require re-exporting a file."""
    text = render_agents_md(_orchestrator(
        prompt_overlays=["Learned guidance."],
        prompt_prefix_overlays=["Deployment framing."],
    ))

    assert "Learned guidance." not in text
    assert "Deployment framing." not in text


def test_agents_md_tells_the_agent_not_to_trust_the_file_over_the_graph():
    text = render_agents_md(_orchestrator())

    assert "not from this file" in text
    assert "ask the graph" in text


# -- skills ------------------------------------------------------------------------

def test_a_skill_is_generated_per_knowledge_base_with_progressive_disclosure():
    text = render_skill("project_management", "Execution", "People and projects.", _skills()[:1])

    assert text.startswith("---\n")
    assert "name: noesis-project-management" in text
    assert "description:" in text
    # Reached through the kernel's MCP client, which is how Prime exposes MCP.
    assert "mcp.call_tool(" in text
    assert '"noesis"' in text


def test_a_skill_is_a_stub_that_fetches_its_body_from_the_graph():
    """The point of the exercise: the instructions are looked up dynamically.

    Prime reads a skill body on demand rather than at startup, so the body can
    be a pointer to the graph instead of a frozen copy of it. A tool list baked
    in at export time is exactly what went stale before."""
    text = render_skill("project_management", "Execution", "People and projects.", _skills()[:1])

    assert "get_noesis_agent_skill" in text
    assert '"key": "noesis-project-management"' in text
    # The tool list must NOT be inlined -- that would reintroduce the snapshot.
    assert "`list_project_blockers`" not in text


def test_a_skill_fetches_its_body_with_a_direct_call_not_call_noesis_tool():
    """`call_noesis_tool`'s `knowledge_base` enum deliberately excludes
    `orchestration`, whose tools are already served directly by the same
    server. A stub that told the model to proxy through `call_noesis_tool`
    produced a validation error instead of the instructions -- so the stub
    must name the tool directly."""
    text = render_skill("project_management", "Execution", "d")

    block = text.split("```python")[1].split("```")[0]
    code = "\n".join(l for l in block.splitlines() if not l.strip().startswith("#"))

    assert "get_noesis_agent_skill" in code
    # The proxy would reject `orchestration`; the direct call is the one that works.
    assert "call_noesis_tool" not in code


def test_a_skill_never_reads_as_a_complete_capability_list():
    """A stub that looks authoritative is worse than a stale list: the model
    would conclude a missing tool does not exist."""
    text = render_skill("project_management", "Execution", "d", _skills()[:1])

    assert "search_noesis_tools" in text
    assert "this file lists nothing" in text


def test_a_skill_renders_without_being_handed_any_tools():
    """The renderer no longer needs the catalog, because the body does not
    contain it. Callers may still pass one; it is ignored."""
    assert render_skill("project_management", "Execution", "d") == render_skill(
        "project_management", "Execution", "d", _skills()
    )


@pytest.mark.parametrize("description", [
    # The two that actually broke: a colon in prose made Prime's YAML parser
    # read the description as a nested mapping, and it dropped the skill.
    "Mixture-of-experts control plane: routing, feedback cataloging, and writes.",
    'Telemetry recorded against them: every dispatch is a "Span" node.',
    "Bounded SQL against PROD_BDI -- owns no domain entities; #not a comment.",
    "Trailing colon:",
    "A description\nsplit over lines.",
])
def test_a_skills_frontmatter_parses_as_yaml_whatever_the_description_says(description):
    """Prime parses the frontmatter with a real YAML parser, so the description
    is a scalar and not free text. A knowledge base description is human prose
    that routinely contains ": ", quotes, and dashes."""
    yaml = pytest.importorskip("yaml")

    text = render_skill("orchestration", "Orchestration", description, _skills()[:1])
    _, frontmatter, _ = text.split("---\n", 2)
    parsed = yaml.safe_load(frontmatter)

    assert parsed["name"] == "noesis-orchestration"
    # Parsed back as one scalar, with the prose intact.
    assert isinstance(parsed["description"], str)
    # Collapsed to one line -- frontmatter is a single scalar -- but not truncated.
    assert " ".join(description.split()) in parsed["description"]
    assert "call_noesis_tool" in parsed["description"]


def test_a_skills_name_matches_its_directory():
    """Prime warns when frontmatter `name` differs from the parent directory,
    and `export_prime_context` derives that directory the same way."""
    text = render_skill("business_dictionary", "Business Dictionary", "d", _skills()[:1])

    assert "name: noesis-business-dictionary" in text


# -- settings --------------------------------------------------------------------

def test_settings_point_prime_at_the_generated_skills():
    settings = render_settings()

    assert settings["skills"] == ["./skills"]
    # Noesis is an MCP server, not an extension; nothing to register here.
    assert "extensions" not in settings


def test_settings_preserve_the_users_own_choices():
    settings = render_settings({
        "defaultModel": "claude-sonnet-4", "theme": "light",
        "compaction": {"enabled": False},
        "extensions": ["./extensions/mine.ts"],
    })

    assert settings["defaultModel"] == "claude-sonnet-4"
    assert settings["theme"] == "light"
    assert settings["compaction"] == {"enabled": False}
    # A user's own extension is untouched -- this export does not manage them.
    assert settings["extensions"] == ["./extensions/mine.ts"]


def test_re_exporting_does_not_duplicate_the_noesis_entries():
    once = render_settings()
    twice = render_settings(once)

    assert twice["skills"] == ["./skills"]


# -- MCP registration ------------------------------------------------------------

def test_the_setup_command_registers_noesis_as_an_ordinary_mcp_server():
    """Prime already speaks MCP, so there is nothing to build -- and nothing to
    maintain. An earlier version generated a TypeScript bridge that
    reimplemented an MCP client; it was deleted for the same reason the
    OpenCode and Pi adapters were."""
    command = mcp_setup_command(pathlib.Path("/repo"))

    assert command[:4] == ["prime-agent", "mcp", "add", "noesis"]
    assert "--cwd" in command and "/repo" in command
    assert command[-3:] == ["-m", "knowledge_bases.orchestration.mcp_server", ] or \
        command[-2:] == ["-m", "knowledge_bases.orchestration.mcp_server"]


def test_the_setup_command_accepts_an_explicit_interpreter():
    command = mcp_setup_command(pathlib.Path("/repo"), python="/usr/bin/python3")

    assert "/usr/bin/python3" in command


def test_the_export_never_writes_a_user_level_mcp_registration(tmp_path):
    """Prime ignores project-level mcpServers entries so a repository cannot
    start a process on a user's behalf. Writing one from here would route
    around exactly that protection, so the command is returned, not run."""
    export_prime_context([_orchestrator(), _expert()], _skills(), tmp_path)
    settings = json.loads((tmp_path / SETTINGS_PATH).read_text())

    assert "mcpServers" not in settings


# -- the whole export ------------------------------------------------------------

def test_export_writes_the_full_prime_workspace(tmp_path):
    written = export_prime_context(
        [_orchestrator(), _expert()], _skills(), tmp_path,
    )

    names = {path.relative_to(tmp_path).as_posix() for path in written}
    assert AGENTS_PATH in names
    assert SETTINGS_PATH in names
    assert ".prime/agent/skills/noesis-project-management/SKILL.md" in names
    assert ".prime/agent/skills/noesis-epistemology/SKILL.md" in names


def test_export_is_idempotent(tmp_path):
    args = ([_orchestrator(), _expert()], _skills(), tmp_path)
    first = export_prime_context(*args)
    snapshot = {path: path.read_text() for path in first}
    export_prime_context(*args)

    assert {path: path.read_text() for path in first} == snapshot


def test_export_requires_an_orchestrator(tmp_path):
    with pytest.raises(ValueError, match=ORCHESTRATOR_KEY):
        export_prime_context([_expert()], _skills(), tmp_path)


def test_export_removes_retired_harness_artifacts(tmp_path):
    """A leftover OpenCode pool would still work and would bypass the context
    plane entirely, so the pivot has to clean up after itself."""
    opencode = tmp_path / ".opencode/agents"
    opencode.mkdir(parents=True)
    (opencode / "noesis-expert-1.md").write_text("<!-- GENERATED by something -->")
    (opencode / "handwritten.md").write_text("keep me")
    pi = tmp_path / ".pi/extensions"
    pi.mkdir(parents=True)
    (pi / "noesis.ts").write_text("// GENERATED by scripts/export_harness_agents.py")

    export_prime_context([_orchestrator(), _expert()], _skills(), tmp_path)

    assert not (opencode / "noesis-expert-1.md").exists()
    assert (opencode / "handwritten.md").read_text() == "keep me"
    assert not (pi / "noesis.ts").exists()


def test_export_drops_skills_for_knowledge_bases_that_lost_their_tools(tmp_path):
    export_prime_context([_orchestrator()], _skills(), tmp_path)
    assert (tmp_path / ".prime/agent/skills/noesis-epistemology/SKILL.md").exists()

    export_prime_context([_orchestrator()], _skills()[:1], tmp_path)
    assert not (tmp_path / ".prime/agent/skills/noesis-epistemology").exists()
    assert (tmp_path / ".prime/agent/skills/noesis-project-management/SKILL.md").exists()


def test_export_uses_real_knowledge_base_names(tmp_path):
    """The registry owns names and descriptions; the export must not invent them."""
    from knowledge_bases.registry import KNOWLEDGE_BASES

    export_prime_context(bootstrap_experts(), _skills(), tmp_path)
    text = (tmp_path / ".prime/agent/skills/noesis-project-management/SKILL.md").read_text()

    assert KNOWLEDGE_BASES["project_management"].name in text


# -- staleness -------------------------------------------------------------------

def test_only_base_prompt_drift_is_reported_stale(tmp_path):
    export_prime_context([_orchestrator(version=3)], _skills(), tmp_path)

    # An overlay refinement bumps nothing that lives in a file.
    revised = _orchestrator(version=3, prompt_overlays=["learned"])
    assert stale_exports([revised], tmp_path) == []

    bumped = _orchestrator(version=4)
    warnings = stale_exports([bumped], tmp_path)
    assert any("v3" in w and "v4" in w for w in warnings)


def test_adding_experts_never_makes_the_export_stale(tmp_path):
    """Experts come from the plane, so the roster and the files are unrelated."""
    export_prime_context([_orchestrator(version=3)], _skills(), tmp_path)

    many = [_orchestrator(version=3)] + [_expert(key=f"e{i}") for i in range(30)]
    assert stale_exports(many, tmp_path) == []


def test_a_missing_export_is_reported(tmp_path):
    warnings = stale_exports([_orchestrator()], tmp_path)

    assert any("missing" in w for w in warnings)


def test_a_redescribed_knowledge_base_makes_its_stub_stale(tmp_path, monkeypatch):
    """Prime reads the frontmatter only at startup, so a registry change that
    is not re-exported is invisible to it. The check has to say so."""
    from knowledge_bases.registry import KNOWLEDGE_BASES

    export_prime_context([_orchestrator()], _skills(), tmp_path)
    assert stale_exports([_orchestrator()], tmp_path, _skills()) == []

    renamed = dataclasses.replace(KNOWLEDGE_BASES["project_management"], description="Now about ships.")
    monkeypatch.setitem(KNOWLEDGE_BASES, "project_management", renamed)
    warnings = stale_exports([_orchestrator()], tmp_path, _skills())

    assert len(warnings) == 1
    assert "noesis-project-management/SKILL.md" in warnings[0] and "frontmatter" in warnings[0]


def test_a_knowledge_base_that_gained_tools_needs_a_stub(tmp_path):
    export_prime_context([_orchestrator()], _skills()[:1], tmp_path)

    warnings = stale_exports([_orchestrator()], tmp_path, _skills())

    assert any("noesis-epistemology/SKILL.md is missing" in w for w in warnings)


def test_without_the_tool_list_only_stubs_on_disk_are_checked(tmp_path, monkeypatch):
    """`_sync_after_edit` has no tool rows; it must still catch drift in what
    was exported, without inventing stubs for knowledge bases it cannot see."""
    from knowledge_bases.registry import KNOWLEDGE_BASES

    export_prime_context([_orchestrator()], _skills()[:1], tmp_path)
    assert stale_exports([_orchestrator()], tmp_path) == []

    renamed = dataclasses.replace(KNOWLEDGE_BASES["project_management"], name="Shipping")
    monkeypatch.setitem(KNOWLEDGE_BASES, "project_management", renamed)

    assert len(stale_exports([_orchestrator()], tmp_path)) == 1


# -- reloading a running session -----------------------------------------------------

class FakeRun:
    """Stands in for `subprocess.run`, scripted per command prefix."""

    def __init__(self, listing=None, list_rc=0, send_rc=0, raise_=None):
        self.listing, self.list_rc, self.send_rc, self.raise_ = listing, list_rc, send_rc, raise_
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        if self.raise_ is not None:
            raise self.raise_
        if argv[1] == "list":
            return _Completed(self.list_rc, json.dumps(self.listing) if self.listing is not None else "")
        return _Completed(self.send_rc, "", "daemon refused" if self.send_rc else "")


class _Completed:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _session(id, cwd, **overrides):
    """The shape `prime-agent list --json` emits (Prime's SessionSummary)."""
    base = {"id": id, "cwd": str(cwd), "lifecycle": "live", "isSessionActive": True,
            "runtimeKind": "top-level"}
    base.update(overrides)
    return base


def test_every_running_agent_on_this_repo_is_asked_to_reload(tmp_path):
    run = FakeRun(listing={"sessions": [
        _session("a1", tmp_path, activity="working"),
        _session("a2", tmp_path / "sub", activity="idle"),
    ]})

    report = trigger_prime_reload(tmp_path, run=run)

    assert report["reloaded"] == ["a1", "a2"] and report["errors"] == {}
    assert run.calls[0] == ["prime-agent", "list", "--json"]
    assert run.calls[1:] == [["prime-agent", "send", "a1", "/reload"],
                             ["prime-agent", "send", "a2", "/reload"]]


def test_other_repositories_archived_sessions_and_subagents_are_left_alone(tmp_path):
    """Reloading is an interruption, so only sessions that actually read this
    repo's skills get one; RLM children inherit their parent's reload."""
    run = FakeRun(listing={"sessions": [
        _session("elsewhere", "/somewhere/else"),
        _session("gone", tmp_path, lifecycle="archived", isSessionActive=False),
        _session("child", tmp_path, runtimeKind="subagent"),
        _session("here", tmp_path),
    ]})

    report = trigger_prime_reload(tmp_path, run=run)

    assert report["reloaded"] == ["here"]


def test_no_running_agents_means_nothing_is_sent(tmp_path):
    run = FakeRun(listing={"sessions": []})

    report = trigger_prime_reload(tmp_path, run=run)

    assert report == {"attempted": [], "reloaded": [], "errors": {}}
    assert len(run.calls) == 1
    assert describe_reload(report) == "no running Prime agents to reload"


def test_a_missing_cli_is_reported_not_raised(tmp_path):
    report = trigger_prime_reload(tmp_path, run=FakeRun(raise_=FileNotFoundError("prime-agent")))

    assert "not on PATH" in report["errors"]["list"]
    assert "skipped" in describe_reload(report)


def test_a_dead_daemon_is_reported_not_raised(tmp_path):
    """`prime-agent list` exits non-zero with a connect error when no daemon
    is up -- the normal state on a machine that only ever ran the export."""
    run = FakeRun(listing=None, list_rc=1)

    report = trigger_prime_reload(tmp_path, run=run)

    assert report["reloaded"] == [] and "list" in report["errors"]


def test_unparsable_listing_is_reported_not_raised(tmp_path):
    class Garbage(FakeRun):
        def __call__(self, argv, **kwargs):
            return _Completed(0, "not json")

    report = trigger_prime_reload(tmp_path, run=Garbage())

    assert "JSONDecodeError" in report["errors"]["list"]


def test_a_failed_send_is_attributed_to_that_agent(tmp_path):
    run = FakeRun(listing={"sessions": [_session("a1", tmp_path)]}, send_rc=1)

    report = trigger_prime_reload(tmp_path, run=run)

    assert report["attempted"] == ["a1"] and report["reloaded"] == []
    assert report["errors"] == {"a1": "daemon refused"}
    assert describe_reload(report) == "Prime reload failed for a1"


# -- sync_tools_catalog auto re-export ----------------------------------------------

def _load_sync_script():
    import importlib.util

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "sync_tools_catalog.py"
    spec = importlib.util.spec_from_file_location("sync_tools_catalog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sync_regenerates_and_reloads_only_when_something_drifted(tmp_path, monkeypatch):
    script = _load_sync_script()
    graph = PlaneGraph([_orchestrator()], [_skill_row("project_management.list_project_blockers", "project_management")])
    reloads = []
    monkeypatch.setattr(script, "trigger_prime_reload", lambda root: reloads.append(root) or
                        {"attempted": [], "reloaded": [], "errors": {}})

    first = script.refresh_export_if_stale(graph, tmp_path)
    assert any("AGENTS.md is missing" in line for line in first)
    assert (tmp_path / AGENTS_PATH).exists()
    assert (tmp_path / ".prime/agent/skills/noesis-project-management/SKILL.md").exists()
    assert reloads == [tmp_path]

    second = script.refresh_export_if_stale(graph, tmp_path)
    assert second == ["Prime context export is current; nothing to regenerate."]
    assert reloads == [tmp_path]


# -- upgrade path ------------------------------------------------------------------

def test_upgrading_removes_the_old_bridge_extension_and_its_settings_entry(tmp_path):
    """An older export generated a TypeScript bridge before Prime's native MCP
    support made it redundant. Upgrading must not leave settings pointing at a
    file that is no longer written."""
    extensions = tmp_path / ".prime/agent/extensions"
    extensions.mkdir(parents=True)
    (extensions / "noesis.ts").write_text("// GENERATED by scripts/export_prime_context.py")
    settings = tmp_path / SETTINGS_PATH
    settings.write_text(json.dumps({
        "extensions": ["./extensions/noesis.ts", "./extensions/mine.ts"],
        "skills": ["./skills"],
        "theme": "light",
    }))

    export_prime_context([_orchestrator(), _expert()], _skills(), tmp_path)
    result = json.loads(settings.read_text())

    assert not (extensions / "noesis.ts").exists()
    assert not extensions.exists()          # empty directory cleaned up too
    assert result["extensions"] == ["./extensions/mine.ts"]   # user's own kept
    assert result["skills"] == ["./skills"]
    assert result["theme"] == "light"


def test_upgrading_drops_the_extensions_key_when_only_ours_was_there(tmp_path):
    settings = tmp_path / SETTINGS_PATH
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"extensions": ["./extensions/noesis.ts"]}))

    export_prime_context([_orchestrator()], _skills(), tmp_path)

    assert "extensions" not in json.loads(settings.read_text())
