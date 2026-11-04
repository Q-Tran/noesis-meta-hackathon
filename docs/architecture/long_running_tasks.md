# Long-running work: Prime executes, Noesis remembers

The question this answers: *can this withstand long research tasks the way Prime
Agent does, with long-term memory in the graph?*

**Yes -- because Noesis stopped trying to be a harness.** Long-running capability
was never the scarce part; Prime Agent already ships it. Durable, shared,
queryable memory was. So the two are composed rather than reimplemented.

## The split

| Prime Agent supplies | Noesis supplies |
|---|---|
| the agent loop and Python kernel | which experts exist, and what each knows |
| `rlm.spawn` sub-agents with independent contexts | which tools are callable, and how |
| daemon-backed sessions that survive terminal close | durable instructions, refinable |
| detach / reattach, `prime-agent attach` | memory across sessions **and users** |
| automatic compaction | what was established, and what is unresolved |
| heartbeats, cron schedules, persistent goals | what improved, and whether it helped |
| bounded autonomous mode with quality gates | provenance for every claim |

Earlier versions of this repo shipped adapters for OpenCode, Claude Code, and Pi,
each reimplementing some slice of the left-hand column: OpenCode got an
interchangeable worker pool because it lacks spawn-by-prompt; Pi got an entire
MCP client and an in-process delegation loop because it has neither. All of that
is retired. It was effort spent reaching a baseline Prime already had.

## What a long research task actually gets

**From Prime, unmodified:** the process outlives the terminal; a session can be
reattached tomorrow; heartbeats and cron re-enter it on a schedule; autonomous
mode continues within token, turn, and wall-clock budgets against quality gates;
compaction keeps context bounded; and `rlm.spawn` gives each expert its own
context window rather than crowding the parent's.

**From Noesis, and this is the half Prime cannot do on its own:**

1. **Findings are durable as they happen.** Every dispatch is a `Span`; every
   routed question a `RoutingTrace`; every expert's conclusion a
   `DelegationResult` recording what it resolved, what it could not, and why it
   stopped. A crash mid-task loses the conversation, not the knowledge.
2. **Memory outlives the checkout.** Prime's own state is `harness_state.json`
   and `kernel-state.dill` -- one machine, one session. Noesis's is a graph:
   queryable, shared across users, and readable by a session that starts on a
   different machine a week later.
3. **Compaction is not a loss event.** The record lives outside the context
   window, so after compaction the orchestrator re-reads
   `get_delegation_results` and continues from facts rather than from a summary
   of a summary.
4. **A resumed session knows what was established.** `resume_noesis_session`
   returns prior questions with verdicts, resolved entities, unresolved items,
   and open questions -- *derived from traces on demand*, not a serialized blob.
   Verified across two connections sharing nothing but the graph.
5. **Experts build on each other instead of re-deriving.** A dependent
   delegation reads the earlier expert's structured result rather than having it
   retyped into a brief.
6. **The system improves at the task while running it.** Skill wording, routing
   vocabulary, and expert prompts are graph data that self-improvement revises,
   measures against live telemetry, and reverts if the metric worsens.
7. **Capability grows mid-task.** New experts, new skills, corrected schemas all
   land live -- see the restart matrix in `experts.md`.

## Honest limits

- **The resume digest is bounded**: 10 turns, 25 entities, 10 open items
  (`sessions.py`). A 200-turn session resumes with a briefing plus everything
  the graph knows, not a replay. Full history stays queryable.
- **The orchestrator's base prompt is the one thing baked into a file.**
  `AGENTS.md` carries it; changing it needs a re-export and a restart.
  `stale_exports()` warns rather than letting it drift silently.
- **Prime's kernel is not a sandbox**, by its own documentation. Under Prime the
  enforced boundary is the bounded-tool surface -- `WRITE_COMMANDS`, the
  `call_noesis_tool` expert-scope check, no raw-Cypher path -- not per-agent
  permission frontmatter, which no longer exists. This is a real and deliberate
  narrowing versus the OpenCode-era `edit: deny` / `bash: deny`, accepted because
  it is the price of Prime's execution model.
- **Subagent spawning is ephemeral and isolated.** Instead of a polling mailbox
  or shared context loops, Noesis uses `spawn_ephemeral_subagent` to delegate
  cross-domain tasks to isolated expert subagents and return clean structured results.

## What this means in practice

Install Prime Agent, run `scripts/export_prime_context.py` once, and a session
gets both halves: Prime's continuity and Noesis's memory. Leave a research task
running overnight under `--autonomous` with a heartbeat; come back, reattach, and
the agent still knows what it established -- because that part was never in the
process to begin with.
