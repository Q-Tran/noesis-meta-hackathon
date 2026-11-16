# Business Knowledge Write Orchestration

`noesis-orchestrator` is the user-facing mixture-of-experts control plane for both
reading and changing business knowledge. Each domain expert owns its schemas, queries,
validation, and mutations. The orchestrator may select and invoke those bounded
capabilities, but it cannot issue Cypher or generic graph mutations.

## Read Flow

1. Score graph-defined expert capabilities against the question.
2. Delegate to the selected bounded experts in priority order.
3. Combine supported results and report partial failures.
4. Catalog relevant user assertions, conflicts, and missing-information questions as
   sourced claims (the `claims` KB).

## Write Flow

1. List the allowlisted commands owned by domain experts.
2. Select one command and validate its structured arguments.
3. If the command declares a preview (currently only `deprecate_entity`, whose blast
   radius isn't obvious from its arguments alone), run it read-only and store the result
   alongside the proposal.
4. Store an immutable `WriteProposal` with a 30-minute expiration.
5. Present the exact command, arguments, and any preview to the user.
6. Require explicit confirmation in a later turn -- or an explicit rejection
   (`reject_noesis_write`), which records `status: 'rejected'` instead of a silent expiry.
7. Reserve the proposal once, revalidate its stored arguments, and dispatch only to the
   registered domain handler.
8. Record completion or failure on the proposal.

The execution tool accepts only a proposal ID and confirmation flag. It cannot replace
the command or arguments. Expired, completed, failed, rejected, or already-reserved
proposals cannot be reused.

## Current Commands

- Business Dictionary: add an alias to an existing canonical term.
- Decisions: recompute decision health from graph evidence.
- Methodology: create projects, start workflows, submit typed changesets, record claims, and
  approve validated baselines.
- Orchestration: deprecate or reactivate a node (or a cascaded block of nodes) per
  `docs/architecture/lifecycle.md`'s "never delete, tag instead" policy. This is the
  one command pair that takes a `label`/relationship-type argument (`deprecate_entity`'s
  `label`/`cascade`), but it stays within the same "no arbitrary graph target" invariant
  below: every `label`/`cascade` value is validated against that knowledge base's
  declared `labels`/`relationships` allowlist (`knowledge_bases/registry.py`) before
  anything is staged, and dispatch never accepts raw Cypher.

Semantic Layer snapshot ingestion is excluded because it accepts a filesystem path and
can reconcile deletions. It requires a separate administrative trust boundary. Delivery
currently has no domain-specific mutation commands beyond the generic deprecate/reactivate
above; status transitions, assignment, and blocker changes must be implemented in its own
repository before orchestration can expose them.

## Identity

Until OIDC is available, the trusted runtime actor is `unknown`. The command validator
overwrites model-supplied actors with this value. Future OIDC middleware must inject
the verified employee ID into the write service as trusted runtime context. Prompt text
and model-generated tool arguments must never establish identity.

## Safety Invariants

- Every executable command is registered server-side to exactly one expert.
- No command accepts raw Cypher, labels, relationship types, or arbitrary graph targets.
- Proposal does not mutate domain knowledge.
- Execution requires explicit confirmation and uses stored arguments only.
- Domain validation still runs immediately before execution.
- Unsupported writes remain explicit capability gaps; they are not converted into
  generic mutations.
