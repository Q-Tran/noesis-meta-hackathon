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
        subject=("KnowledgeBaseExpert",),
        object=("Tool",),
        not_to_be_confused_with=("CALLED", "HAS_CAPABILITY", "USES", "CALLS"),
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
