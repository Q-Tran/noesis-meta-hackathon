# User Feedback Loop

Noesis catalogs relevant user-provided data and unanswered information needs as
`Claim`s in the Evidence knowledge base, sourced to a `SourceNode` (owned by the
Provenance knowledge base — see `docs/architecture/provenance.md`). This preserves
subjectivity: user assertions are sourced graph data, not a separate class of
canonical truth.

## Graph Model

- `SourceNode {id: "user::<user_id>", type: "user"}` (`knowledge_bases.provenance`)
  identifies the person supplying data.
- `Claim {claim_type: "fact"}` stores a relevant excerpt explicitly supplied by that user.
- `Claim {claim_type: "question", status: "open"}` stores information Noesis needs but
  could not find.
- `(SourceNode)-[:ASSERTS]->(Claim)` records provenance.
- `(Claim)-[:SUPPORTS|CHALLENGES]->(entity)` files a claim with a stance
  toward an exact graph entity.
- `(Claim)-[:CONFLICTS_WITH]->(claim_or_entity)` preserves contradictory sourced data.
- `(fact:Claim)-[:ANSWERS]->(question:Claim)` resolves a missing-information question and
  changes the question status to `answered`.

Only relevant excerpts are retained as claims: this feedback mechanism stores no
transcript of its own. The complete user prompt is stored once, by the tracer and
the routing repository, as `RoutingTrace.question` on the turn it was said on --
which is exactly what the grounding edge below points at, so nothing needs a copy.
The agent gives any supported partial answer before asking up to
three focused questions.

## Conversation Grounding

Claims cataloged from an interaction are substantiated directly by the turn
they were made on, with no transcript copying (see
`docs/architecture/provenance.md`, "User conversations as provenance"): the
tracer and the routing repository record every turn verbatim as
`(:RoutingTrace {question})`, and `catalog_noesis_feedback` grounds each
filed claim with `(Claim)-[:FOUND_IN {role: "origin"}]->(RoutingTrace)`. The trace resolves
from trusted runtime context (the `trace_id` argument, else the connection's
active trace); when there is none, the filing succeeds and the result reports
the claim ungrounded, per claim. Nothing is duplicated as `Chunk` nodes and
no conversation `SourceNode` exists.

## Current User Source

Authentication is not currently available to the web, CLI, or OpenCode entry points.
They therefore pass `user_id="unknown"`, and feedback is sourced to:

```text
SourceNode {id: "user::unknown", identity_status: "placeholder"}
```

This value is deliberately a temporary identity placeholder, not an anonymous-user
design decision.

## Future OIDC Migration

When OIDC authentication is introduced, the authenticated employee ID must replace
`unknown` at the application boundary. The Python `answer()` and orchestration service
already accept a trusted `user_id`. The MCP tool schemas intentionally do not accept
identity today, because model-supplied tool arguments are not authenticated. Extend the
MCP runtime with trusted identity context rather than exposing a free-form identity
argument. The repository will then source claims to
`SourceNode {id: "user::<employee_id>", identity_status: "verified"}`.

Do not accept an employee ID from prompt text or an unverified request parameter. The
OIDC middleware must derive it from verified token claims and provide it to Noesis as
trusted runtime context. Before rollout, decide whether existing `user::unknown` claims
remain attributed to the placeholder or are migrated using separately audited session
records; they must not be assigned to employees heuristically.

## Domain Knowledge Writes

Feedback claims and canonical domain changes are separate operations. The orchestrator
routes supported changes through bounded commands owned by the relevant knowledge-base
expert. It first creates a one-time proposal, presents the exact command and arguments
for user confirmation, and only then executes the stored proposal. Domain writes never
accept raw Cypher from the agent, and the one command pair that does take a
label/relationship-type argument (deprecating or reactivating a node, see
`docs/architecture/lifecycle.md`) validates every such value against that knowledge
base's declared allowlist before staging the proposal, rather than trusting an
arbitrary graph target.

A proposal is not limited to a silent 30-minute expiry: the user can also explicitly
reject a not-yet-confirmed proposal, recording an audit trail instead. A proposal whose
effect isn't obvious from its arguments alone (currently, deprecating a cascaded block
of nodes) additionally computes and stores a read-only preview -- an affected-node
count and sample -- so the user sees the blast radius before confirming. The
confirmation itself needs no separate provenance record: the user's explicit
`confirmed: true` is the `execute_noesis_write` tool-call span, already kept on the
turn's trace.

Current orchestrated write capabilities are Business Dictionary alias management,
Decision Intelligence health evaluation, typed methodology workflow operations, and generic
node deprecation/reactivation. Semantic Layer snapshot ingestion remains an
administrative operation, and Delivery writes must wait for explicit status-transition,
assignment, and blocker command contracts beyond the generic deprecate/reactivate above.

Orchestrated commands currently overwrite any model-supplied `actor` with
`unknown`. When OIDC arrives, the write service must inject the verified employee ID
from runtime identity context at this same boundary. It must never trust an `actor` or
employee ID generated from prompt text.
