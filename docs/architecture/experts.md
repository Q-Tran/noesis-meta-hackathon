# Graph-Defined Experts

Noesis is a graph of knowledge bases *and* a graph of the agents that work
over them. An **expert** is one subagent per knowledge base; its complete
definition is a `KnowledgeBaseExpert` node, so a prompt tweak, a new allowed
skill, or a routing-vocabulary fix is a graph write that every user and every
harness picks up on the next call -- not a per-checkout file edit.

## Graph model

```
(:KnowledgeBaseExpert {key, name, description, priority, routable,
                       system_prompt, prompt_overlays, prompt_prefix_overlays,
                       capabilities, capability_overlays,
                       entry_tools, allowed_tools, model_hint, version})
   -[:CURRENT]->(:ExpertRevision {id: key:version, version, reason, ...snapshot})
                    -[:SUPERSEDES]->(:ExpertRevision)          # full history
   -[:HAS_CAPABILITY]->(:ExpertCapability {statement, position}) # effective phrases
   -[:CAN_USE]->(:Tool)                                         # allowed_tools resolved
(:RoutingTrace)-[:CONSIDERED {score, reasons}]->(:KnowledgeBaseExpert)
(:RoutingTrace)-[:DELEGATED_TO {sequence, operation}]->(:KnowledgeBaseExpert)
(:Span {expert_key})-[:CALLED]->(:Tool)                        # who used what
```

- **Seed**: `knowledge_bases/orchestration/experts_catalog.py` --
  `bootstrap_experts()` defines the six routable experts and the
  non-routable `orchestrator` (the primary agent's own prompt). Each routable
  prompt = domain rules + the shared `EXPERT_LOOP_CONTRACT` (how to loop over
  `search_noesis_tools`/`call_noesis_tool`/`hybrid_search`).
- **Sync**: `RoutingRepository.sync_experts()` (via
  `scripts/sync_tools_catalog.py` or the `sync_noesis_experts` tool) MERGEs
  the live node, bumps `version` only when something changed, snapshots an
  `ExpertRevision`, and rebuilds `HAS_CAPABILITY`/`CAN_USE`. Graph-side
  overlays are preserved across re-seeds.
- **Overlays**: `capability_overlays`/`prompt_overlays` are what
  self-improvement adds. `effective_capabilities` = base + overlays;
  `effective_prompt` = supplemental prefix + base prompt + overlay paragraphs.
  Reverting one improvement removes one list item as a new revision.
- **Prompt prefix**: `prompt_prefix_overlays` is the one channel that puts text
  *ahead* of the base prompt, for deployment or session framing an expert
  should read first (`orchestration.set_expert_prompt_prefix`, or the
  `expert_prompt_prefix` self-improvement kind). Bounded at 4 entries / 2000
  characters, enforced at write time -- an over-budget prefix is refused, never
  truncated -- and rendered inside an explicitly-labelled advisory block. The
  base prompt's prohibitions declare themselves non-overridable by it.

## Changing the roster at runtime

An expert composed of *existing* knowledge bases' registered skills adds no
label, relationship, or migration, so it is data rather than architecture (see
`knowledge_bases/boundary.py`). `orchestration.create_expert` /
`revise_expert` / `retire_expert` are ordinary bounded write commands under
propose/confirm/execute (`knowledge_bases/orchestration/expert_admin.py`),
gated by the same two checks self-improvement uses: the capability-pollution
guard and the golden routing eval. `allowed_tools` patterns must resolve
against the live catalog, so an expert can never be created that claims a
capability nothing implements. Retirement sets `active = false`; nothing is
deleted. The `orchestrator` itself is protected -- it can be framed with a
prefix, but not created, revised, or retired at runtime.

## Serving experts

| Surface | What it returns |
|---|---|
| `list_noesis_experts` | every active expert (key, description, capabilities, entry skills, version, routable); `include_skills` adds callable skills |
| `get_noesis_expert {key}` | one expert's `system_prompt` (effective, including overlays and prefix), `entry_tools`, live `skills` with `input_schema`, optional `revisions`; also registers `expert_key` as the connection's active identity |
| MCP `prompts/list`, `prompts/get <key>` | the same prompts through the protocol's own primitive |
| `get_noesis_context` | the whole plane at once: experts, tools, instructions, memory (see below) |
| `scripts/export_prime_context.py` | the generated `.prime/agent/` cache: bridge extension, `AGENTS.md`, per-KB skills, settings |

### One harness: Prime Agent

Noesis targets Prime Agent exclusively, and the division is the reason the
integration is small:

| Prime owns execution | Noesis owns context |
|---|---|
| agent loop, Python kernel | which experts exist and what each knows |
| `rlm.spawn` sub-agents | which tools are callable, and how |
| daemon continuity, detach/reattach | durable instructions |
| compaction | cross-session memory |
| heartbeats, schedules | what is unresolved |
| autonomous budgets | what improved, and whether it helped |

Earlier versions shipped adapters for OpenCode, Claude Code, and Pi. Each spent
effort teaching a harness to *be* Prime -- OpenCode got an interchangeable worker
pool because it lacks spawn-by-prompt; Pi got an entire MCP client and an
in-process delegation loop because it has neither. That is retired. Prime already
has sub-agents, a daemon, heartbeats, schedules, and bounded autonomous mode, so
reimplementing them per harness was work spent reaching a baseline Prime ships
with.

What Prime lacks is memory that outlives a checkout: its harness state is
`harness_state.json` and `kernel-state.dill`, on one machine, per session. Noesis
supplies that half and nothing else.

### The context plane

`get_noesis_context` returns the whole thing in one call at session start:

| Element | Contents |
|---|---|
| `instructions` | the orchestrator's `base` prompt plus `supplemental` (prefix + overlays), kept separate so precedence is explicit |
| `experts` | every routable expert with its capabilities and entry skills -- enough to route, not the full prompt |
| `skills` | the callable surface with live descriptions, including any `description_override` |
| `memory` | resumable sessions, and one session's digest when asked |

One call rather than four because a session needs all of it at once, and a round
trip per concern burns turns on bookkeeping the model should never do. It is a
manifest, not a dump: full prompts and schemas are fetched per delegation via
`get_noesis_expert`, which is also what registers the child's `expert_key` for
tracing.

### What is written into a workspace

`scripts/export_prime_context.py` generates `.prime/agent/`:

| Artifact | Why |
|---|---|
| `AGENTS.md` | Prime reads this natively; carries the orchestrator's *base* instructions only |
| `skills/noesis-<kb>/SKILL.md` | one Agent Skills entry per knowledge base, so Prime's progressive disclosure surfaces Noesis capabilities natively |
| `settings.json` | points Prime at those skills; every other key is preserved |

Plus one command, printed rather than executed:

```bash
prime-agent mcp add noesis --cwd <repo> -- <python> -m knowledge_bases.orchestration.mcp_server
```

Noesis is an ordinary MCP server, so there is no bridge to build. Registration is
deliberately *user-level*: Prime ignores project-level `mcpServers` entries so
that "a repository cannot start a local process or shadow a user server", and
writing one from the export would route around exactly that protection. An
earlier version generated a TypeScript bridge that reimplemented an MCP client;
it was deleted for the same reason the OpenCode and Pi adapters were -- it
rebuilt something Prime already had.

Tools are reached from the kernel, which is Prime's design ("MCP integrations
are *not* exposed as new agent tools"):

```python
plane = await mcp.call_tool("noesis", "get_noesis_context", {})
```

These are a **cache of the graph, not a second source of truth**. Anything that
can change at runtime is fetched, so re-export is needed only when the immutable
base prompt or a knowledge base's registry name/description changes --
`stale_exports()` detects both, `scripts/sync_tools_catalog.py` re-exports when
it does, and `trigger_prime_reload()` asks running Prime agents to `/reload`
(see `prime_skill_loading.md`).

## What survives a long session without a restart

Everything the graph owns. Prime reads `AGENTS.md` and its settings once at
startup, but those carry only the immutable base; the plane and the tool catalog
are fetched, and Prime's `registerTool` takes effect immediately.

| Change | Live mid-session? | How |
|---|---|---|
| **Add an expert** | **yes** | appears in the next `get_noesis_context`; routing selects it; a child adopts it with `get_noesis_expert` |
| Revise or retire an expert | yes | graph write |
| Expert prompt overlay / prefix | yes | fetched per delegation |
| Expert capabilities (routing vocabulary) | yes | routing reads the graph each `route_noesis_question` |
| **Add a skill/tool** (incl. LLM-authored) | **yes** | `tools/list_changed` -> bridge re-registers |
| Skill description or `input_schema` corrected | yes | overrides, read live |
| Skill retired / reactivated | yes | `active` filters discovery *and* dispatch |
| Handler *source* edited on disk | yes | mtime-based module reload in `skills/resolver.py` |
| Session memory | yes | `:Session` nodes; `resume_noesis_session` |
| Orchestrator *base* prompt in `experts_catalog.py` | no | baked into `AGENTS.md`; `sync_tools_catalog.py` re-exports it and sends running agents `/reload` |
| Knowledge base name/description in `registry.py` | no | baked into `SKILL.md` frontmatter, which Prime reads only at startup; same re-export + `/reload` |
| New knowledge base, label, or migration | no | architecture, by design (`knowledge_bases/boundary.py`) |

Verified live: activating an expert mid-session made it appear in the plane,
routing selected it for a matching question, a child adopted its prompt, and
`stale_exports()` reported nothing to regenerate at any point.

## Routing

`route_noesis_question` ranks experts by fusing two signals
(`knowledge_bases/orchestration/routing.py`):

- **keyword** overlap between the question and the expert's effective
  capability phrases/description (plural-folded, function words removed);
- **semantic** cosine similarity from `knowledge_bases/tools/catalog_index.py`:
  the best-matching capability phrase (`noesis_experts` collection) or the
  best-matching skill the expert may call (`noesis_skills`), excluding generic
  control-plane skills.

`score = 0.7 * max(kw, sem) + 0.3 * min(kw, sem)`. Experts at/above
`SELECT_THRESHOLD` are selected (metric questions always add Business
Dictionary → Semantic Layer); each gets a starting delegation built from its
`entry_tools` and their input schemas (`starting_delegation`), falling back
to `hybrid_search` over the expert's entity family when the entry skill needs
an identifier. Confidence is the top score damped by how close the runner-up
is; the harness LLM makes the final call among close candidates, or
`decide: true` asks Noesis's own LLM. Without Milvus the whole thing degrades
to keyword-only.

`knowledge_bases/orchestration/routing_eval.jsonl` is a golden set of
questions → expected expert. `evaluate_routing()` scores it keyword-only
(deterministic); self-improvement uses it as an apply-time gate and as half
of the routing metric.

## Identity

Spans record `user_id`, `client`, and `expert_key`. A harness sets
`NOESIS_CLIENT`/`NOESIS_USER_ID` on the server process
or passes them on `call_noesis_tool`; a spawned child registers its
`expert_key` by calling `get_noesis_expert`, and `route_noesis_question`
registers `orchestrator` for the routing turn -- so the orchestrator's own
prompt is a self-improvement target like any expert's. `skill_health()` breaks
results down `by_client`/`by_user`, and `expert_corrections()` attributes user
corrections to the expert that acted.

Identity and trace correlation live in a per-connection scope
(`knowledge_bases/tools/active_trace.py`). It was process-global, which was
only safe while one server process served one conversation; durable sessions
break that assumption, so state is now keyed by connection with the default
scope preserving single-conversation behaviour exactly.

## Execution is scoped to the same boundary as discovery

`search_noesis_tools` has always narrowed results to an expert's
`allowed_tools`, but `call_noesis_tool` validated only the knowledge base --
so "never bypass a knowledge base's bounded tools" was prompt text rather than
an enforced rule. Because every child dispatches through that one proxy, it is
the natural chokepoint: a call whose `expert_key` is not permitted the target
skill is refused. Calls with no `expert_key` (the orchestrator's own, and any
harness not acting as an expert) are unaffected, and the knowledge-base
allowlist still applies to them.

Every expert's `allowed_tools` therefore also includes the control-plane
skills its loop contract mandates (`SHARED_EXPERT_TOOLS`): tool discovery,
cross-domain search, recording its delegation result, and the agent mailbox.

Worth stating plainly: under Prime this is the *only* enforced tool boundary.
Prime's kernel is a control environment, not a sandbox, so there is no per-agent
permission frontmatter to fall back on -- the bounded-tool surface, the write
allowlist, and the absence of any raw-Cypher path are what hold.
