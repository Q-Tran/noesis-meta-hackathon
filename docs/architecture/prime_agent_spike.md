# Spike: can a Prime Agent extension spawn RLM children without the kernel?

**Verdict: yes via the SDK, no via an extension.** Variant B is viable, but not
in the form originally assumed -- it requires embedding Prime Agent as a library
rather than writing an extension for the shipped CLI.

Read against `PrimeIntellect-ai/prime-agent` at commit `55c611f`
(coding-agent `0.9.4`, 2026-09-10). Line numbers are from that commit.

## The question

Prime Agent's continuity features (daemon-backed sessions, push/steer agent
messaging, heartbeats, schedules) are real and are things Noesis cannot do as an
MCP server. But its MCP integration model is the blocker: MCP tools are reached
**only from inside the Python kernel** (`docs/mcp-integrations.md`: "MCP
integrations are *not* exposed as new agent tools... The MCP connection runs
inside the kernel"), and the kernel is not a security sandbox by its own repeated
admission. Adopting it wholesale would make Noesis's bounded-tool guarantee,
no-raw-Cypher rule, and propose/confirm/execute write gate advisory rather than
enforced -- on a graph that multiple users share.

Variant B was the hypothesis that we could keep the bounded surface *and* gain
continuity: disable the kernel, register Noesis's graph-registered skills as
native Prime tools, and still get recursive subagents. That last clause is what
this spike tested, because `rlm.spawn` is documented only as a Python API.

## What the source says

### 1. Both spawn paths are reachable only from the kernel's host handlers

`runRlmChild` and `createRlmSession` each have exactly one caller, and it is the
kernel host-handler map:

```
agent-session.ts:9679  private _createKernelHostHandlers(): HostRequestHandlers {
agent-session.ts:9681    "rlm.run":            ... this.runRlmChild(...)
agent-session.ts:9684    "rlm.create_session": ... this.createRlmSession(...)
```

`grep -rn "runRlmChild"` returns exactly two hits: the definition
(`agent-session.ts:11454`) and that handler. The handler map is constructed
inside the kernel provisioner's options:

```
agent-session.ts:9567    hostHandlers: this._createKernelHostHandlers(),
```

So on the *shipped code paths*, "spawn a child" means "the model executed Python
that issued an `rlm.run` host request". There is no CLI flag, no slash command,
and no tool that reaches it.

### 2. The extension API has no spawn surface, by construction

`ExtensionAPI` (`core/extensions/types.ts:992`) and `ExtensionContext` were
grepped for `rlm|spawn|child|subagent|agentSession`: **no matches in either.**
The extension host is a hand-wired closure allowlist (`agent-session.ts`
~9200-9440) -- `appendEntry`, `setActiveTools`, `setModel`, `compact`, and so on,
each individually bound. `this` (the `AgentSession`) is never exposed, and
extensions receive a `SessionManager`, not the session.

This is a deliberate boundary, not an oversight: extensions get a curated
capability set, and child-agent lifecycle is not in it.

### 3. But `AgentSession.runRlmChild` is public and the class is exported

```
agent-session.ts:11454   async runRlmChild(prompt, kwargs = {}, spawnCode?): Promise<RlmSpawnHandle>
index.ts:6               export { AgentSession, ... }
sdk.ts:80                session: AgentSession;   // CreateAgentSessionResult
```

`runRlmChild` carries no `private` modifier, and `createAgentSession()` hands
back the live `AgentSession`. So an **SDK embedder** -- a Node process that
imports Prime Agent as a library rather than shelling out to the CLI -- can call
it directly. That is the documented "consider using `AgentSession` directly"
path from `docs/rpc.md`.

### 4. The spawn path itself has no kernel dependency

`_startRlmChildRun` (`agent-session.ts:10920-11395`) was grepped for
`ipython|prewarm|baseToolsOverride|noTools`: the only hits are its own
`_createRlmSubagentRuntimeOptions` call. Everything it does -- depth check,
model resolution, child session dir, registry admission, usage attribution --
is pure TypeScript session machinery. It never touches the kernel.

### 5. The kernel stays dormant if `ipython` is never active

`IpythonKernelProvisioner`'s constructor is empty (`tools/ipython.ts:318-321`);
the kernel process spawns only on `ensure()`, reached from `prewarm()` or the
tool's `execute`. And `prewarm()` is gated:

```
agent-session.ts:9644   if ((this._prewarmIpythonKernel || hasSnapshot) &&
                            this.getActiveToolNames().includes("ipython")) {
                          this._ipythonKernelProvisioner?.prewarm();
                        }
```

So a session whose active tools exclude `ipython` never starts a Python process.
The provisioner object is still *constructed* (and the host handlers with it),
but nothing is listening: host requests arrive over the kernel's stdio
transport, and there is no kernel to send them.

### 6. Children inherit the parent's tool configuration

```
agent-session.ts:9980   activeToolNames: this.getActiveToolNames(),
agent-session.ts:9981   allowedToolNames: this._allowedToolNames ? [...] : undefined,
agent-session.ts:9982   customTools: [...this._customTools],
```

A child of a kernel-less, custom-tool-only parent is itself kernel-less and
custom-tool-only. The property we need -- recursion without code execution --
holds transitively.

### 7. One caveat: `--no-builtin-tools` does not *forbid* the kernel

```
sdk.ts:235   allowedToolNames = options.allowedToolNames ?? options.tools ??
                                (options.noTools === "all" ? [] : undefined)
sdk.ts:238   initialActiveToolNames = ... (options.tools ? [...] : options.noTools ? [] : ["ipython"])
```

For `noTools: "builtin"`, `allowedToolNames` is `undefined` -- *no* restriction --
and only the initially-active list is emptied. `ipython` remains in the registry,
so `setActiveToolsByName` (`agent-session.ts:4460`) could re-activate it, and any
extension calling `pi.setActiveTools([...])` would restore code execution.

The correct configuration is therefore **`tools: [...]` / `allowedToolNames`**,
not `--no-builtin-tools`: an explicit allowlist omitting `ipython` makes
re-activation impossible, because `_refreshToolRegistry` filters the registry
itself through `isAllowedTool` (`agent-session.ts:9454-9463`).

## What this means for Noesis

Variant B is viable with a specific shape:

- **Embed via SDK, don't extend the CLI.** A Node host calls
  `createAgentSession({ allowedToolNames: [...noesis tools] })`, registers the
  graph-registered `Tool` catalog as custom tools, and calls
  `session.runRlmChild(...)` for delegation. Extensions alone cannot do this.
- **Constrain tools with an allowlist, not `--no-builtin-tools`**, or the kernel
  remains one `setActiveTools` call away.
- What we would gain: daemon continuity, push/steer messaging (the one real
  ceiling on `B3`'s mailbox), heartbeats, schedules, and spawn-by-prompt.
- What it would cost: a Node/TypeScript service in a pure-Python repo, tracking
  a fast-moving fork of a fast-moving base (`0.9.4`, 4,670 commits, with
  documented breaking changes already), reimplementing the Noesis tool bridge in
  TS, and owning the risk that `runRlmChild` is public-by-accident -- it is
  reachable but *undocumented* as an embedder API, so it has no stability
  guarantee.

## Recommendation

Do not migrate now. The finding removes the blocking technical objection but not
the cost/benefit one:

- The problem that originally motivated the harness question -- new experts
  needing a restart -- is already solved by the generic worker pool, in Python,
  with no new runtime.
- The remaining gap is push delivery and background sessions. That is worth a
  Node sidecar only if long-running unattended Noesis sessions become a real
  requirement, not a hypothetical one.
- If they do, this is the shape to build, and the two configuration traps above
  (`allowedToolNames` vs `--no-builtin-tools`; SDK vs extension) are the things
  that would otherwise have been discovered late.

Re-verify against a newer commit before acting: this rests on `runRlmChild`
being public and `AgentSession` being exported, neither of which Prime documents
as a contract.
