# Trace-Driven Self-Improvement

Every skill/tool dispatch through `noesis-orchestrator` (or any domain MCP
server called directly) produces a `SpanEvent`, and every routed question
produces a `RoutingTrace` (see `docs/architecture/graph_driven_mcp_connectors.md`'s
"Every dispatch is a recorded span" section). Until now nothing read that
data back to ask "should a skill's wording, schema, or a knowledge base's
routing capability actually change?" This closes that loop, as a skill of
its own (`skills.propose_self_improvements`) rather than a one-off script.

It reuses the existing reflexive self-model instead of inventing a new one:
`scripts/self_recompute.py` already re-evaluates the `noesis-self` methodology project and
files advisory `ReviewFinding`s when the codebase's structure drifts from
recorded architecture rationale. Self-improvement files into the same
`ReviewFinding`/project shape -- but the signal is execution quality, not
code structure, and, unlike `self_recompute.py`'s propose-only findings,
each self-improvement finding is also *applied* and later *re-evaluated*
automatically (see "Applying and re-evaluating" below) -- the one place
this loop departs from `self_recompute.py`'s posture and every other
Noesis write's propose-confirm-execute pattern, by explicit design.

## Three independent signals

A skill can fail outright (errors, or results the user later corrects)
without any routing ambiguity, or routing can be ambiguous while every
delegated skill call succeeds perfectly. Both are tracked separately:

- **Skill health** (`knowledge_bases.tools.trace_store.TraceStore.skill_health`,
  graph `Span`/`TraceOutcome` nodes) -- per skill, over a lookback window:
  call count, error count/rate, sample error messages, average duration, and
  `user_correction_count` (the number of *distinct traces* containing a call
  to this skill whose latest recorded `TraceOutcome` was
  `success=false, source="user_correction"` -- the user said the final
  answer was wrong, a stronger signal than one span merely erroring).
  A skill only qualifies once it has at least `MIN_CALLS` (default 5) in
  the window, and only if `error_rate >= 0.2` or
  `user_correction_count >= 2` (`knowledge_bases/tools/self_improvement.py`).
- **Routing gaps** (`knowledge_bases.orchestration.repository.RoutingRepository.routing_gaps`,
  Neo4j `RoutingTrace`) -- per knowledge base, the count of routing traces
  where it was `primary_expert` and either `confidence < 0.5` or
  `clarification_needed = true`, plus up to 5 sample `question_summary`
  excerpts (the 240-char list-view truncation, which is what keeps the
  self-improvement prompt bounded -- the full verbatim prompt lives beside it
  on the same trace as `question`, the graph's single store of what the user
  said; see `docs/architecture/provenance.md`). A knowledge base only
  qualifies once at least `MIN_TRACES` (default 5) such traces exist *and*
  they're at least 40% of its traffic in-window
  (`LOW_CONFIDENCE_SHARE_THRESHOLD`), so a knowledge base with just one or
  two rough edges isn't flagged.

- **Expert corrections** (`TraceStore.expert_corrections`) -- per
  graph-defined expert (`Span.expert_key`), the share of traces it acted in
  whose latest outcome was a user correction, with the correction reasons.
  Qualifies at `MIN_TRACES` traces, at least 2 corrections, and a 30%
  share. This points at the expert's *prompt* rather than one skill.

All thresholds exist to distinguish a real pattern from noise -- a single
failed call or one ambiguous question proves nothing.

## LLM-drafted suggestions, classified before anything is filed

Each qualifying signal is first handed to an LLM (`llm.get_llm()`, the same
`BaseLLM.extract()` pattern as `knowledge_bases.epistemology.feedback.analyze_feedback`)
to diagnose the likely cause and, if plausible, draft concrete replacement
wording:

- Skill signals ask whether the fix is a `description`/`input_schema`
  wording change (see `knowledge_bases/tools/catalog.py`) versus something
  else entirely -- the model is told to leave `suggested_change` empty and
  explain in `diagnosis` if it looks like anything wider, not guess at code
  in this first pass.
- Routing-gap signals ask for a concrete additional/replacement
  `ExpertCapability` phrase (see `knowledge_bases/orchestration/experts_catalog.py`'s
  `CAPABILITIES`) that would better match the sample questions' vocabulary.
  This path only ever produces `routing_capability_wording` -- there is no
  further classification step for routing gaps, only for skill signals.

If a skill signal's diagnosis is *not* a wording fix, it goes through a
second LLM call, `classify_signal()`
(`knowledge_bases/tools/self_improvement.py`), to decide the shape of code
change actually needed:

- **`new_tool`**: the diagnosis describes a missing, self-contained
  capability expressible as ONE new handler function appended to that
  knowledge base's existing `mcp_tools.py` -- no new file, no new Neo4j
  label/relationship, no change to any existing function. A third LLM call
  (`_draft_new_skill_suggestion`) then drafts the actual handler source
  (`GeneratedToolDraft`) for that capability.
- **`architecture`**: anything wider -- a new knowledge base ("paradigm": a
  new `knowledge_bases/<name>` package, new Neo4j labels/relationships, a
  new `KnowledgeBase` registry entry), a change spanning multiple
  files/packages, or a fix to a skill's *existing* logic rather than a
  missing capability. This is also the **safe default**: `classify_signal`
  falls back to `architecture` on any unparseable/empty LLM output, and
  `_draft_skill_wording_suggestion` falls back to `architecture` if a
  `new_tool` classification's own generation step never produced usable
  handler source -- a suggestion nothing downstream could ever act on is
  worse than one correctly routed to a human.

The LLM's diagnosis output is parsed into `SelfImprovementDraft`/
`SelfImprovementClassification`/`GeneratedToolDraft` (all fields
optional/defaulted) rather than the final `SelfImprovementSuggestion`
directly, so a provider that returns nothing parseable (including the
offline `NullLLM`) degrades safely -- an empty diagnosis is dropped, never
filed as a hollow finding; an unparseable classification defaults to
`architecture`, never to something that would auto-apply. `kind`/`target`
on the final `SelfImprovementSuggestion` are always set by code from the
signal/classification, never trusted from the model's output.

`file_suggestions()` (`knowledge_bases/tools/self_improvement.py`) files
each surviving suggestion as a `ReviewFinding` `BmadEntity` in the
`noesis-self` project via the normal `BmadRepository.submit_changeset`
path, `status: "open"`. Filing itself never edits any source file, and is
idempotent by `(kind, target)` (`finding_key()`): a skill or knowledge base
already flagged is skipped on the next run, not re-filed or overwritten,
until the finding moves past `"open"` -- identical to
`scripts/self_recompute.py`'s drift-finding dedup.

Severity at filing time depends on the kind: `tool_wording`/
`routing_capability_wording`/`expert_prompt_wording`/`expert_prompt_prefix`/
`new_tool` are filed `severity: "advisory"`
(they auto-apply and self-correct -- see below -- so a human is never
*required* to act). `architecture` is filed `severity: "blocking"` -- it
is never auto-applied, so this makes it visible and gated (see
`knowledge_bases/methodology/validation.py`'s `build`-phase unresolved-blocking-
finding check). A `new_tool` finding's `GeneratedToolDraft` (the handler
source itself) is persisted as `generated_skill_json` on the finding, so
`apply_self_improvement` can reconstruct exactly what to apply from the
finding alone, with no need to re-run drafting.

## Applying and re-evaluating -- "did it help?"

This is the part that closes the loop, in
`knowledge_bases/tools/self_improvement_apply.py`. For `tool_wording`,
`routing_capability_wording`, `expert_prompt_wording`, `expert_prompt_prefix`,
and `new_tool` (`AUTO_APPLIABLE_KINDS`), **applying a finding is fully
automatic, with no human approval gate** -- the one place this feature departs
from propose-confirm-execute. `architecture` is the deliberate exception to the
exception: `apply_finding()` rejects it immediately, before doing any work.

The boundary between "may auto-apply" and "needs a human" is no longer only
described here: `knowledge_bases/boundary.py` declares it, and
`tests/test_architecture_boundary.py` asserts that no `SelfImprovementKind` and
no `WriteCommand` can reach a structural surface -- including that
`apply_finding` performs its `AUTO_APPLIABLE_KINDS` check before any edit,
commit, or graph write.

Two mechanisms, by kind:

1. **Wording kinds are graph edits** (`knowledge_bases/tools/graph_edits.py`).
   The skill descriptions and expert definitions being improved *are* graph
   data, so the change is a graph write and nothing in source changes:
   - `tool_wording` sets `Tool.description_override`; `load_tools()`
     returns `coalesce(description_override, description)`, and the seeded
     base `description` is untouched, so `sync_catalog()` cannot clobber it.
   - `routing_capability_wording` appends one phrase to the expert's
     `capability_overlays`; `expert_prompt_wording` appends one paragraph to
     its `prompt_overlays`; `expert_prompt_prefix` adds one bounded entry to
     `prompt_prefix_overlays`, which renders *ahead* of the base prompt (4
     entries / 2000 characters, refused rather than truncated when over
     budget). Each is a new `ExpertRevision` (reason
     `self_improvement:<kind>`), and `sync_experts()` carries overlays
     forward when re-seeding.

   **These take effect immediately, with no re-export.** They did not used to:
   subagent prompts came from generated per-expert files, and `_sync_after_edit`
   never regenerated them -- so an `expert_prompt_wording` improvement passed
   its gates, was measured, and was kept or reverted while the running subagent
   had never seen it. Harness workers are now generic and fetch their expert's
   live definition via `get_noesis_expert` at delegation time (see
   `docs/architecture/experts.md`), so a graph revision reaches the next
   delegation. `_sync_after_edit` additionally warns when a *base* prompt has
   drifted from the exported file, which is the one case a re-export is still
   needed for.
   - Before a routing phrase is written it passes two gates: the
     **pollution guard** (`check_capability_pollution`: the phrase must
     resemble the target expert's own vocabulary more than any other
     expert's -- exactly the failure that once appended methodology/startup wording
     to `business_dictionary`) and the **eval gate** (`check_routing_eval`:
     top-1 accuracy on `knowledge_bases/orchestration/routing_eval.jsonl`
     must not drop).
   - The semantic catalog index is refreshed so routing/discovery see the
     change immediately, and every user sharing the graph gets it at once.
2. **`new_tool` is a source edit** (`knowledge_bases/tools/source_editor.py`).
   LLM-authored logic has to live in code: `generate_new_skill()` appends one
   statically-validated handler to the knowledge base's existing
   `mcp_tools.py` and registers it in `catalog.py`; `_validate_handler_source`
   rejects anything that isn't exactly one `(arguments, graph)` function
   with no unsafe names; `run_test_gate()` runs the **full test suite** in a
   subprocess before any commit (a failure discards the edit and marks the
   finding `failed`); and the change is one isolated git commit
   (`git_ops.commit_files`) so revert is a plain `git revert`.

**Every application is measured, and reverted automatically if it made
things worse.** `apply_finding()` snapshots the metric the suggestion was
drafted from immediately before changing anything -- error + correction
rate per call for a skill; for routing, half the live low-confidence share
plus half the golden-eval error rate; correction share for an expert -- and
records it with a cooldown deadline (`DEFAULT_COOLDOWN`, 3 days) on the
finding (`status: "applied"`, plus `graph_edit` or `commit_sha`).
`reevaluate_applied_findings()` re-measures any finding whose cooldown has
elapsed: held or improved → `kept`; worse → the graph edit is undone
(`revert_graph_edit`: clear the override / remove the one overlay item as
another revision) or the commit is `git revert`ed, and the finding is
marked `reverted`. One finding's failure never blocks the rest.

The `ReviewFinding` node is therefore a full audit trail of the whole
lifecycle: `open` -> `applied` (with `graph_edit` or `commit_sha`,
`pre_metric_snapshot`, `reevaluate_after`) -> `kept` or `reverted` (with
`post_metric_snapshot`, and `graph_revert` / `revert_commit_sha`) -- or,
for a `new_tool` that failed its test gate, `open` -> `failed`.

## Human review for structural changes

`architecture`-kind findings never reach an edit/commit at all --
`apply_finding()` raises `ApplyError` immediately if asked to apply one.
They are filed `severity: "blocking"` specifically so they're visible
(rather than silently sitting at `"open"` forever like every other
advisory finding). A human makes the actual code/schema change themselves,
outside Noesis, then resolves the finding with the bounded
`methodology.resolve_review_finding` `WriteCommand`
(`knowledge_bases/methodology/review_findings.py`) -- `{project_id, finding_key,
status: "resolved"|"rejected", note}` -- through the normal
`propose_noesis_write`/`execute_noesis_write` confirm flow like any other
Noesis write. This command only ever touches `status`/`resolution_note`/
`resolved_at` on the finding; it cannot be used to alter a finding's own
diagnosis or any self-improvement bookkeeping field, and it works for any
`ReviewFinding` in any project, including `scripts/self_recompute.py`'s
architecture-drift findings, which had no dedicated resolution action
before this either.

## Running it

As graph-registered skills, callable like anything else (see
`docs/architecture/graph_driven_mcp_connectors.md`):

```
propose_self_improvements {"dry_run": true}
apply_self_improvement {"finding_key": "self_improvement::tool_wording::business_dictionary.resolve_business_language"}
reevaluate_self_improvements {}
resolve_review_finding {"project_id": "noesis-self", "finding_key": "self_improvement::architecture::project_management.some_skill", "status": "resolved", "note": "Added the new knowledge base manually."}
```

`propose_self_improvements`'s `dry_run: true` returns the raw signals and
drafted suggestions without filing anything -- useful for reviewing what
would be flagged before it's filed (and, for auto-appliable kinds, once
filed, immediately applied). Omit it (or set `false`) to actually file new
`ReviewFinding`s. `lookback_days`/`min_calls`/`min_traces` override the
thresholds above per call. Each entry in the returned `filed` list carries
its own `kind` and `finding_key`; `apply_self_improvement` looks a finding
up by that same `finding_key()` format, and reconstructs the suggestion
(including a `new_tool`'s generated handler source) from the finding's
own stored properties -- it returns an `{"error": ...}` payload rather than
raising through MCP for an `architecture`-kind finding, or any other
apply failure.

As a standalone script, for a scheduled job that runs all three phases in
sequence (propose -> apply each new finding -> re-evaluate anything whose
cooldown has elapsed):

```bash
python scripts/self_improve.py
```

The `noesis-orchestrator` agent also triggers `propose_self_improvements`
+ `apply_self_improvement` + `reevaluate_self_improvements` itself,
mid-conversation, on two triggers: an explicit user request, or immediately
after a negative outcome or execution error is recorded for a trace (see
`AGENTS.md`).

## What this does not do

- It does not retrain, fine-tune, or change any model.
- It never edits source for a wording change; those are graph revisions.
- It never generates a brand-new knowledge base/graph schema
  ("paradigm") or a change spanning multiple files -- that's exactly what
  the `architecture` kind exists to route to a human instead of guessing
  at. `new_tool` generation is deliberately scoped to one new function in
  one existing file; nothing here can create a new `knowledge_bases/<name>`
  package, a new Neo4j label/relationship, or a new `KnowledgeBase`
  registry entry.
- It does not infer *why* a skill is failing beyond what the LLM can
  diagnose from error messages/sample questions; a signal that isn't a
  wording fix and isn't narrow enough for `new_tool` is classified
  `architecture` and filed for a human to investigate, never guessed at.
- It is not a replacement for `scripts/self_recompute.py`'s architecture-drift
  check -- the two loops are complementary (execution quality vs. code
  structure) and file into the same project (and, since
  `methodology.resolve_review_finding` works on any `ReviewFinding`, now share the
  same resolution path too), but analyze different data, and only
  self-improvement's auto-appliable findings are ever auto-applied/reverted.
