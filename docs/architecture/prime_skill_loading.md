# How Prime Agent loads Agent Skills (evidence)

Source: /home/k1ng/.local/lib/node_modules/prime-agent/dist

1. Discovery is from DISK, at session start and on `/reload`.
   - core/resource-loader.js:~220 `async reload()` -> `loadSkills({...})`
   - core/skills.js:381 `loadSkills`, :431-432 loads `<agentDir>/skills` (user)
     and `<cwd>/.prime/agent/skills` (project), :463 loads `settings.json` paths.
   - core/skills.js:196 `loadSkillsFromDirInternal`: a dir containing SKILL.md is a
     skill root and is NOT recursed further.

2. Only FRONTMATTER reaches the system prompt. The BODY is never loaded by Prime.
   - core/skills.js `loadSkillFromFile`: reads only `frontmatter.name`,
     `frontmatter.description`, `frontmatter["disable-model-invocation"]`.
   - `formatSkillsForPrompt` emits <name>, <type>, <description>, <location>
     and instructs: "Use ipython to inspect a skill's file when the task matches
     its description."
   => The body is fetched ON DEMAND, by the agent itself, as an ordinary read.
   => THIS IS THE HOOK: a body can be a pointer to a live source.

3. Validation (core/skills.js):
   - `name` must equal the parent directory name; lowercase [a-z0-9-] only;
     no leading/trailing/double hyphen; MAX_NAME_LENGTH = 64.
   - `description` is REQUIRED (empty -> skill dropped entirely);
     MAX_DESCRIPTION_LENGTH = 1024.
   - frontmatter is parsed by a real YAML parser (hence the quoting fix).
   - a python skill needs pyproject.toml + src/<import_name>/__init__.py.

4. `settings.json` "skills" accepts only local file/directory paths
   (config.js: `skills?: string[]` - "Array of local skill file paths or
   directories"). No command, URL, or MCP source.

5. No MCP-side skill registration. Extensions can contribute skills
   (`extensionSkillSourceInfos`), but an MCP server cannot.

## Consequence for the design

A skill's *discoverability* (name + description) must exist on disk before the
session starts. A skill's *content* need not: the body is read on demand.

So the maximum achievable dynamic design is:
  - keep one thin generated SKILL.md per skill: frontmatter = live description
    snapshot (needed for startup discovery), body = an instruction to fetch the
    authoritative instructions from the graph via an MCP tool.
  - the graph holds the real body, versioned and refinable at runtime.
Hard blocker: the frontmatter description itself cannot be resolved at runtime,
so it stays a cache refreshed by re-export (plus `/reload`).

## Closing the gap without a watcher

The frontmatter is rendered from `KNOWLEDGE_BASES` in
`knowledge_bases/registry.py`, so it can only drift because of a code change --
which already means running `scripts/sync_tools_catalog.py`. That script now
does the rest:

- `stale_exports()` (`prime_export.py`) compares every on-disk
  `noesis-<kb>/SKILL.md` against a fresh `render_skill()` of the registry entry,
  in addition to the `AGENTS.md` base-prompt version it already checked.
- If anything drifted, `sync_tools_catalog.py` re-runs `export_prime_context()`.
- `trigger_prime_reload()` then runs `prime-agent list --json` and
  `prime-agent send <id> "/reload"` for each agent running on this repo.
  Prime classifies incoming text starting with `/` as a slash command whichever
  way it arrived (`core/agent-session.js`), so this is the same as the user
  typing `/reload`. Best-effort: no CLI, no daemon, or a refused send is
  reported, never raised.

`scripts/export_prime_context.py` does the same reload after a manual export.
Both have opt-outs (`--no-auto-export`, `--no-reload`) for CI and machines
without Prime.
