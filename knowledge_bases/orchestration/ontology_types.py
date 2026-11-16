"""What the Orchestration knowledge base's labels and relationships mean.

Written for an extractor choosing between near-synonyms: this package owns the
*control plane*, not any business domain. It records who the graph-defined
experts are, how a question was routed to them, what each delegation concluded,
what text an agent is given as context, and which writes are awaiting
confirmation. The bounded operations an expert may call are `Tool` nodes owned by
`tools`; the execution telemetry of calling them is `Span`/`TraceOutcome`, also
owned by `tools`.

Grounded in the real writers -- `repository.RoutingRepository` (experts,
revisions, capabilities, plans), `sessions.py`, `delegation_results.py`,
`agent_messages.py`, `agent_skills.py`, `memory.py`, and
`write_repository.py` -- so each endpoint pair is one the code produces.

`CURRENT` and `SUPERSEDES` are declared here but described by `methodology`,
which owns the versioning edges (`KnowledgeBaseExpert`/`ExpertRevision` and
`SessionMemory` appear as endpoints in that package's descriptions).
"""
from knowledge_bases.base import EntityType, PropertyType, RelationshipType

ENTITY_TYPES = (
    EntityType(
        label="KnowledgeBaseExpert",
        extractable=False,
        description=(
            "One graph-defined expert: the complete, versioned definition a "
            "harness needs to instantiate a subagent for one knowledge base -- its "
            "prompt, capability phrases, entry and allowed tools, and whether "
            "routing may pick it. It is a role definition, not a human (Person) "
            "and not a callable operation (Tool)."
        ),
        identity="key",
        properties=(
            "key", "name", "description", "capabilities", "priority",
            "system_prompt", "entry_tools", "allowed_tools", "model_hint",
            "version", "active", "routable", "capability_overlays",
            "prompt_overlays", "prompt_prefix_overlays", "updated_at",
        ),
        not_to_be_confused_with=("Person", "Tool", "AgentSkill", "ExpertRevision"),
    ),
    EntityType(
        label="ExpertRevision",
        extractable=False,
        description=(
            "An immutable snapshot of one expert definition at one version, "
            "created whenever that expert's properties actually change. It is the "
            "history of a KnowledgeBaseExpert, the same role a Revision plays for "
            "a methodology-managed artifact -- reading the live definition means "
            "following CURRENT to the newest one."
        ),
        identity="id",
        properties=(
            "id", "key", "version", "reason", "created_at", "system_prompt",
            "capabilities", "allowed_tools", "entry_tools",
        ),
        not_to_be_confused_with=("KnowledgeBaseExpert", "Revision"),
    ),
    EntityType(
        label="ExpertCapability",
        extractable=False,
        description=(
            "One short phrase stating a kind of question an expert handles, held "
            "at a fixed position in that expert's capability list -- the text "
            "routing retrieval matches an incoming question against. It is "
            "routing vocabulary, not a product capability being built "
            "(`Capability`) and not an operation that can be invoked (Tool)."
        ),
        identity="id",
        properties=("id", "statement", "position"),
        not_to_be_confused_with=("Capability", "Tool", "KnowledgeBaseExpert"),
    ),
    EntityType(
        label="RoutingTrace",
        extractable=False,
        description=(
            "The record of one routed question: its verbatim text (the single "
            "graph store of what the user said -- claims are substantiated by "
            "this node, not by copies of it), its summary and request hash, the "
            "primary expert chosen, the confidence, and whether clarification was "
            "needed. It is the unit everything else correlates on via `trace_id`; "
            "a single tool dispatch inside it is a Span, and the verdict on its "
            "answer is a TraceOutcome."
        ),
        identity="id",
        properties=(
            "id", "request_hash", "question", "question_summary", "started_at", PropertyType(
                "status",
                "How the delegations for this trace turned out. `partial` means some experts succeeded and others did not -- the common case with parallel delegation, and the reason this is not a boolean.",
                enum=("planned", "completed", "partial", "failed",),
            ),
            "strategy", "primary_expert", "confidence", "clarification_needed",
            "session_id",
        ),
        not_to_be_confused_with=("Span", "TraceOutcome", "Session", "WorkflowRun"),
    ),
    EntityType(
        label="Session",
        extractable=False,
        description=(
            "One conversation between a user through a client and the graph, "
            "carrying its activity clock and identity, and owning every trace "
            "asked in it. A session is the whole conversation; one question inside "
            "it is a RoutingTrace."
        ),
        identity="id",
        properties=(
            "id", "started_at", "last_active_at", "user_id", "client",
        ),
        not_to_be_confused_with=("RoutingTrace", "SessionMemory", "WorkflowRun"),
    ),
    EntityType(
        label="DelegationResult",
        extractable=False,
        description=(
            "What one expert concluded for one delegation of one trace: its "
            "summary, confidence, what it could not settle, why it stopped, plus "
            "the tools it called and entities it resolved. It is a declared "
            "judgement the orchestrator synthesizes from, not the mechanical "
            "telemetry of the calls (Span) and not a verdict on the final answer "
            "(TraceOutcome)."
        ),
        identity="id",
        properties=(
            "id", "trace_id", "expert_key", "summary", "confidence",
            "unresolved", "stopped_because", "tools_called", "entity_labels",
            "entity_keys", "recorded_at",
        ),
        not_to_be_confused_with=("Span", "TraceOutcome", "Claim"),
    ),
    EntityType(
        label="AgentSkill",
        extractable=False,
        description=(
            "One versioned instruction bundle a harness discloses to a model: a "
            "name and description shown at startup, with a body fetched live when "
            "a task matches. It has no handler and is never dispatched -- a "
            "callable operation is a Tool, which this skill may merely document."
        ),
        identity="key",
        properties=(
            "key", "name", "description", "body", "knowledge_base", "version",
            "active", "created_at", "updated_at",
        ),
        not_to_be_confused_with=("Tool", "ExpertCapability", "KnowledgeBaseExpert"),
    ),
    EntityType(
        label="SessionMemory",
        extractable=False,
        description=(
            "One durable lesson learned in a session -- a fact, procedure, "
            "pitfall, or preference -- scoped to a repo, an expert, both, or "
            "globally, whose prose body is read back as context. Memory never "
            "expires and is corrected append-only by a newer memory that "
            "SUPERSEDES it; it is a retained lesson, not a conversation record "
            "(Session) and not an evidenced assertion (Claim)."
        ),
        identity="key",
        properties=(
            "key", "title", "body", "kind", "scope", "repo", "expert_key",
            "session_id", "trace_id", "version", "created_at", "updated_at",
            "unmapped_claims",
        ),
        not_to_be_confused_with=("Session", "Claim"),
    ),
    EntityType(
        label="WriteProposal",
        extractable=False,
        description=(
            "One bounded knowledge write staged for confirmation, holding the "
            "command, its serialized arguments, a summary and preview, a status, "
            "and a 30-minute expiry. It is a write that has not happened yet and "
            "may never happen -- a committed batch of graph changes is a ChangeSet."
        ),
        identity="id",
        properties=(
            "id", "command", "knowledge_base", "operation", "arguments_json",
            "summary", "preview_json", PropertyType(
                "status",
                "Where the proposal is in propose/confirm/execute. Only a `proposed` one can be confirmed or rejected, which is what stops a double execution. `failed` means the command raised after confirmation.",
                enum=("proposed", "executing", "completed", "failed", "rejected",),
            ), "created_at", "confirmed_at",
            "expires_at",
        ),
        not_to_be_confused_with=("ChangeSet", "Revision", "Alternative"),
    ),
    EntityType(
        label="Worker",
        extractable=False,
        description=(
            "A fine-grained agent employee. Workers replace general-purpose "
            "experts: each worker owns a narrow task, a prompt, skills, tools, "
            "and a routing vocabulary. It is the stable identity; a "
            "WorkerVersion is the mutable definition that reflection may change."
        ),
        identity="key",
        properties=(
            "key", "name", "description", "mission", "active", "kind",
            "version", "autonomy", "routable", "priority", "model_hint",
            "created_at", "updated_at",
        ),
        not_to_be_confused_with=("KnowledgeBaseExpert", "AgentSkill", "Tool"),
    ),
    EntityType(
        label="WorkerVersion",
        extractable=False,
        description=(
            "An immutable, task-scoped configuration of one Worker at one "
            "version. This is what reflection tunes -- never the global "
            "system prompt. A WorkerVersion is proposed, evaluated, canaried, "
            "activated, superseded, or reverted; it is never edited in place."
        ),
        identity="id",
        properties=(
            "id", "key", "version",
            PropertyType("status", "The lifecycle state of this worker version.", enum=("draft", "evaluating", "canary", "active", "superseded", "rejected", "reverted")),
            "system_prompt", "capabilities",
            "entry_tools", "allowed_tools", "skills", "model_hint", "reason",
            "created_at", "activated_at",
        ),
        not_to_be_confused_with=("ExpertRevision", "Worker", "AgentSkill"),
    ),
    EntityType(
        label="WorkerCapability",
        extractable=False,
        description=(
            "One routing phrase on a Worker, held at a position. Same role as "
            "ExpertCapability, but for fine-grained workers rather than "
            "knowledge-base experts."
        ),
        identity="id",
        properties=("id", "statement", "position"),
        not_to_be_confused_with=("ExpertCapability", "Capability"),
    ),
    EntityType(
        label="WorkerAssignment",
        extractable=False,
        description=(
            "A binding from a Worker to a task, queue, channel, or connector "
            "scope. It states what the worker is responsible for; it does not "
            "itself run anything."
        ),
        identity="id",
        properties=("id", "task", "queue", "channel", "scope", "active"),
        not_to_be_confused_with=("Worker", "DelegationResult"),
    ),
    EntityType(
        label="ChangeProposal",
        extractable=False,
        description=(
            "The record of one proposed change, kept as an argumentative "
            "claim: what should change, why (its evidence), what the gate "
            "decided, and what happened afterward. A proposal is created by "
            "reflection, a user, or an ontology reflector. It is never edited "
            "in place; a refinement is a newer proposal."
        ),
        identity="id",
        properties=(
            "id", "kind", "target_key",
            PropertyType("status", "The lifecycle state of this change proposal.", enum=("proposed", "evaluating", "eval_passed", "eval_failed", "approved", "applied", "rejected", "superseded", "reverted")),
            "title", "summary",
            "motivation", "changes_json", "actor", "created_at", "updated_at",
        ),
        not_to_be_confused_with=("WriteProposal", "ReviewFinding", "ChangeSet"),
    ),
    EntityType(
        label="ProposalEvidence",
        extractable=False,
        description=(
            "One premise of a ChangeProposal: a sampled trace, a score, a "
            "failure cluster, or a correction event. Evidence is the reason a "
            "proposal exists; without it, a change is an assertion."
        ),
        identity="id",
        properties=("id", "kind", "summary", "trace_ids", "score", "created_at"),
        not_to_be_confused_with=("Claim", "Span"),
    ),
    EntityType(
        label="EvalReport",
        extractable=False,
        description=(
            "The result of one stage of one change's evaluation gate: "
            "replay, shadow, canary, post-apply, or scorer-calibration. It is "
            "the argument that the change is safe to activate."
        ),
        identity="id",
        properties=(
            "id", "stage", "passed", "baseline_score", "candidate_score",
            "detail", "samples_json", "created_at",
        ),
        not_to_be_confused_with=("TraceOutcome", "ChangeProposal"),
    ),
    EntityType(
        label="ReflectionRun",
        extractable=False,
        description=(
            "One reflection pass over a worker, ontology, or connector: the "
            "sampled traces inspected, the scorer version used, and the "
            "proposal(s) it produced. Reflection is itself traced."
        ),
        identity="id",
        properties=("id", "target_kind", "target_key", "sampled_trace_ids", "scorer_version", "created_at"),
        not_to_be_confused_with=("RoutingTrace", "ChangeProposal"),
    ),
    EntityType(
        label="OntologyProposal",
        extractable=False,
        description=(
            "A proposed change to an ontology package: a label, property, "
            "relationship, merge, or deprecation. By default reviewed by a "
            "human; auto-apply is an explicit per-ontology setting and never "
            "skips the gate. Approved proposals graduate to a reviewed code "
            "change; the graph keeps the rationale forever."
        ),
        identity="id",
        properties=(
            "id", "package_key", "kind", "target",
            PropertyType("status", "The lifecycle state of this ontology proposal.", enum=("proposed", "approved", "rejected", "applied")),
            "rationale",
            "proposed_declaration_json", "created_at", "updated_at",
        ),
        not_to_be_confused_with=("ChangeProposal", "ChangeSet", "Revision"),
    ),
    EntityType(
        label="Connector",
        extractable=False,
        description=(
            "A catalog entry for one data source connector: what system it "
            "reads, which ontology labels it produces, and the handler that "
            "pulls it. The catalog is data; the connector code is bounded and "
            "reviewed."
        ),
        identity="key",
        properties=("key", "name", "description", "source_system", "handler", "active", "config_schema_json"),
        not_to_be_confused_with=("Tool", "SourceNode"),
    ),
    EntityType(
        label="ConnectorInstance",
        extractable=False,
        description=(
            "One configured instance of a Connector: its scope, schedule, "
            "health, and last sync. Instances are user-owned data, versioned "
            "like other runtime data."
        ),
        identity="id",
        properties=(
            "id", "name", "connector_key", "config_json", "schedule", "health",
            "last_sync_at", "records_added", "last_error", "created_at", "updated_at",
        ),
        not_to_be_confused_with=("Connector", "SourceNode", "IngestionEvent"),
    ),
)

RELATIONSHIP_TYPES = (
    RelationshipType(
        relationship="HAS_CAPABILITY",
        extractable=False,
        description=(
            "Attaches one routing capability phrase to the expert that claims it. "
            "The edge set is rewritten wholesale on every expert sync, so it "
            "always reflects the expert's effective capability list -- it grants "
            "no permission to call anything, which is CAN_USE."
        ),
        subject=("KnowledgeBaseExpert",),
        object=("ExpertCapability",),
        not_to_be_confused_with=("CAN_USE", "DELIVERS", "REALIZES"),
    ),
    RelationshipType(
        relationship="CAN_USE",
        extractable=False,
        description=(
            "A standing permission: this expert is allowed to dispatch that "
            "active Tool, resolved from its `allowed_tools` key patterns at sync "
            "time. It says what may be called in future, whereas CALLED records "
            "that a call already happened."
        ),
        subject=("KnowledgeBaseExpert", "Worker", "WorkerVersion"),
        object=("Tool",),
        not_to_be_confused_with=("CALLED", "HAS_CAPABILITY", "USES", "CALLS"),
    ),
    RelationshipType(
        relationship="ACTIVE_VERSION",
        extractable=False,
        description=(
            "Points a Worker at its one active WorkerVersion. Reverting a "
            "worker is re-pointing this edge to an older version; the version "
            "history itself is untouched."
        ),
        subject=("Worker",),
        object=("WorkerVersion",),
        not_to_be_confused_with=("CURRENT", "SUPERSEDES", "HAS_VERSION"),
    ),
    RelationshipType(
        relationship="HAS_VERSION",
        extractable=False,
        description=(
            "Links a WorkerVersion to the Worker it belongs to. It is the "
            "membership edge; ACTIVE_VERSION is the one live configuration."
        ),
        subject=("WorkerVersion",),
        object=("Worker",),
        not_to_be_confused_with=("CURRENT", "ACTIVE_VERSION", "HAS_VERSION"),
    ),
    RelationshipType(
        relationship="WORKER_HAS_CAPABILITY",
        extractable=False,
        description=(
            "Attaches one routing capability phrase to a Worker at a fixed "
            "position. The same role as HAS_CAPABILITY, for fine-grained "
            "workers rather than knowledge-base experts."
        ),
        subject=("Worker",),
        object=("WorkerCapability",),
        not_to_be_confused_with=("HAS_CAPABILITY", "CAN_USE", "REALIZES"),
    ),
    RelationshipType(
        relationship="WORKER_CAN_USE",
        extractable=False,
        description=(
            "A standing permission for a WorkerVersion to dispatch a Tool. "
            "Resolved from the version's allowed_tools patterns when the "
            "version is created."
        ),
        subject=("WorkerVersion",),
        object=("Tool",),
        not_to_be_confused_with=("CAN_USE", "CALLED", "USES"),
    ),
    RelationshipType(
        relationship="HAS_ASSIGNMENT",
        extractable=False,
        description=(
            "Binds a Worker to a task, queue, channel, or connector scope it "
            "is responsible for."
        ),
        subject=("Worker",),
        object=("WorkerAssignment",),
        not_to_be_confused_with=("DELEGATED_TO", "ASSIGNED_TO", "HAS_VERSION"),
    ),
    RelationshipType(
        relationship="TARGETS",
        extractable=False,
        description=(
            "A ChangeProposal targets the Worker, WorkerVersion, Connector "
            "instance, ontology, or decomposition it proposes to change."
        ),
        subject=("ChangeProposal", "OntologyProposal"),
        object=("Worker", "WorkerVersion", "ConnectorInstance", "OntologyProposal"),
        not_to_be_confused_with=("DELEGATED_TO", "RESULTED_IN"),
    ),
    RelationshipType(
        relationship="EVIDENCED_BY",
        extractable=False,
        description=(
            "A ChangeProposal is supported by one piece of ProposalEvidence: "
            "a sampled trace, an outcome score, or a failure cluster."
        ),
        subject=("ChangeProposal", "OntologyProposal"),
        object=("ProposalEvidence",),
        not_to_be_confused_with=("USES_PREMISE", "EVALUATED_BY"),
    ),
    RelationshipType(
        relationship="EVALUATED_BY",
        extractable=False,
        description=(
            "A ChangeProposal is judged by one EvalReport from a gate stage "
            "(replay, shadow, canary, post-apply, calibration)."
        ),
        subject=("ChangeProposal", "OntologyProposal"),
        object=("EvalReport",),
        not_to_be_confused_with=("EVIDENCED_BY", "JUDGES"),
    ),
    RelationshipType(
        relationship="CREATED_BY_REFLECTION",
        extractable=False,
        description=(
            "A ReflectionRun created one ChangeProposal or OntologyProposal. "
            "Reflection is itself a traced act with provenance."
        ),
        subject=("ReflectionRun",),
        object=("ChangeProposal", "OntologyProposal"),
        not_to_be_confused_with=("RESULTED_IN", "DELEGATED_TO"),
    ),
    RelationshipType(
        relationship="CONFIGURES",
        extractable=False,
        description=(
            "A ConnectorInstance configures one catalog Connector: the "
            "instance is the user's concrete setup of the catalog entry."
        ),
        subject=("ConnectorInstance",),
        object=("Connector",),
        not_to_be_confused_with=("USES", "HAS_INSTANCE"),
    ),
    RelationshipType(
        relationship="DECOMPOSES_INTO",
        extractable=False,
        description=(
            "A broad (migrated) Worker decomposed into finer Workers, "
            "proposed from trace clusters. The parent is superseded as work "
            "moves to the children; it is not deleted."
        ),
        subject=("Worker",),
        object=("Worker",),
        not_to_be_confused_with=("SUPERSEDES", "DELEGATED_TO"),
    ),
    RelationshipType(
        relationship="CONSIDERED",
        extractable=False,
        description=(
            "Records that routing scored this expert as a candidate for that "
            "question, carrying its `rank`, `score` and `reasons` -- including "
            "candidates that were not chosen. Selection is DELEGATED_TO; this "
            "edge is the ranking evidence behind it."
        ),
        subject=("RoutingTrace",),
        object=("KnowledgeBaseExpert",),
        not_to_be_confused_with=("DELEGATED_TO", "CHOSEN_OVER", "ANSWERS"),
    ),
    RelationshipType(
        relationship="DELEGATED_TO",
        extractable=False,
        description=(
            "Records that routing actually handed a delegation of this question to "
            "that expert, in `sequence` order with the starting operation. It is "
            "the chosen subset of the CONSIDERED candidates, and it assigns work "
            "to an expert role, never to a Person (ASSIGNED_TO)."
        ),
        subject=("RoutingTrace",),
        object=("KnowledgeBaseExpert",),
        not_to_be_confused_with=("CONSIDERED", "ASSIGNED_TO", "RESULTED_IN"),
    ),
    RelationshipType(
        relationship="HAS_TRACE",
        extractable=False,
        description=(
            "Places one routed question inside the conversation it was asked in, "
            "so a session's history is a traversal rather than a property scan. It "
            "groups questions under a conversation, not spans under a question "
            "(PART_OF)."
        ),
        subject=("Session",),
        object=("RoutingTrace",),
        not_to_be_confused_with=("PART_OF", "CONTAINS", "HAS_RUN"),
    ),
    RelationshipType(
        relationship="RESULTED_IN",
        extractable=False,
        description=(
            "Links a routed question to a declared conclusion recorded against it "
            "by one expert. It points at what a delegation concluded, not at which "
            "expert was asked (DELEGATED_TO) and not at whether the final answer "
            "held up (JUDGES)."
        ),
        subject=("RoutingTrace",),
        object=("DelegationResult",),
        not_to_be_confused_with=("DELEGATED_TO", "JUDGES", "PRODUCED", "ANSWERS"),
    ),
    RelationshipType(
        relationship="DOCUMENTS",
        description=(
            "An instruction bundle contains the operation it teaches: tools live "
            "inside skills, made graph-true when the agent-skill catalog syncs. It "
            "asserts that the skill's text explains how to use that Tool, not that "
            "the tool implements the skill."
        ),
        subject=("AgentSkill",),
        object=("Tool",),
        not_to_be_confused_with=("CAN_USE", "CATALOGS", "SUPPORTS", "IMPLEMENTS"),
    ),
)
