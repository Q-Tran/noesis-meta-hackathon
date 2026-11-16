"""Seed `Tool` definitions for the Orchestration knowledge base.

Composed into `bootstrap_catalog()` via this package's `tools_seed` (see `knowledge_bases/registry.py`); nothing reads this
module at runtime -- `sync_catalog()` MERGEs these into graph `Tool` nodes and the MCP servers serve from there.
"""
from knowledge_bases.tools.catalog import QUESTION_SCHEMA, TRACE_ID_INPUT, _call_noesis_tool_schema, _create_expert_schema, _deprecate_entity_schema, _hybrid_search_schema, _node_reference_schema, _propose_write_schema, _reactivate_entity_schema, _record_delegation_result_schema, _record_outcome_schema, _revise_expert_schema, _search_noesis_tools_schema, _set_expert_prompt_prefix_schema, _skill


TOOLS = [
        _skill(
        "orchestration.search_noesis_tools",
        "orchestration",
        "Search Noesis Tools",
        "Search the full graph-registered tool/skill catalog by plain-"
        "language need (e.g. 'check project blockers', 'resolve a "
        "metric'), across every knowledge base or one in particular. "
        "Every result is already callable with no separate install "
        "step -- call it via call_noesis_tool, or directly if already "
        "connected to that knowledge base's own MCP server.",
        _search_noesis_tools_schema(),
        "knowledge_bases.orchestration.mcp_tools:search_noesis_tools_tool",
    ),

    _skill(
        "orchestration.hybrid_search",
        "orchestration",
        "Hybrid Search",
        "Hybrid semantic and keyword search within one explicit knowledge "
        "family and entity-type scope.",
        _hybrid_search_schema(),
        "knowledge_bases.orchestration.mcp_tools:hybrid_search_tool",
    ),

    _skill(
        "orchestration.catalog_noesis_feedback",
        "orchestration",
        "Catalog Noesis Feedback",
        "Catalog relevant user-provided facts and missing-information "
        "questions. Link contradictions to exact returned graph nodes. "
        "Filed claims are automatically substantiated by the conversation "
        "turn they came from (the trace's recorded question).",
        {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "statement": {"type": "string"},
                            "claim_type": {"type": "string", "enum": ["fact", "question"]},
                            "supports": {
                                "type": "array",
                                "description": (
                                    "Existing graph nodes this claim backs. "
                                    "Each needs a label and ONE identifier."
                                ),
                                "items": _node_reference_schema(),
                            },
                            "challenges": {
                                "type": "array",
                                "description": (
                                    "Existing graph nodes this claim disputes. "
                                    "Each needs a label and ONE identifier."
                                ),
                                "items": _node_reference_schema(),
                            },
                            "conflicts_with": {
                                "type": "array",
                                "description": (
                                    "Existing graph nodes this claim contradicts."
                                ),
                                "items": _node_reference_schema(),
                            },
                            "answers_question_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["statement"],
                    },
                },
                "trace_id": {
                    "type": "string",
                    "description": "The trace of the turn the claims were made "
                                   "on (from route_noesis_question). Leave "
                                   "unset to use this connection's active trace.",
                },
            },
            "required": ["claims"],
        },
        "knowledge_bases.orchestration.mcp_tools:catalog_noesis_feedback_tool",
    ),

    _skill(
        "orchestration.list_noesis_write_commands",
        "orchestration",
        "List Noesis Write Commands",
        "List bounded business-knowledge write commands owned by Noesis experts.",
        {"type": "object", "properties": {}},
        "knowledge_bases.orchestration.mcp_tools:list_noesis_write_commands_tool",
    ),

    _skill(
        "orchestration.propose_noesis_write",
        "orchestration",
        "Propose Noesis Write",
        "Validate and stage one bounded expert-owned business-knowledge write. "
        "This does not mutate domain knowledge; show the proposal to the user.",
        _propose_write_schema(),
        "knowledge_bases.orchestration.mcp_tools:propose_noesis_write_tool",
    ),

    _skill(
        "orchestration.execute_noesis_write",
        "orchestration",
        "Execute Noesis Write",
        "Execute a previously staged business-knowledge write after the user "
        "explicitly confirms that exact proposal. Proposals are one-time use.",
        {
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string"},
                "confirmed": {"type": "boolean", "const": True},
            },
            "required": ["proposal_id", "confirmed"],
        },
        "knowledge_bases.orchestration.mcp_tools:execute_noesis_write_tool",
    ),

    _skill(
        "orchestration.deprecate_entity",
        "orchestration",
        "Deprecate Entity",
        "Tag one node, or a cascaded block of nodes reached by declared "
        "containment relationships, with :Deprecated. Nodes are never "
        "deleted (see docs/architecture/lifecycle.md); default read/search "
        "paths skip :Deprecated nodes unless explicitly asked to include them.",
        _deprecate_entity_schema(),
        "knowledge_bases.orchestration.mcp_tools:deprecate_entity_tool",
    ),

    _skill(
        "orchestration.reactivate_entity",
        "orchestration",
        "Reactivate Entity",
        "Remove the :Deprecated tag from one node or a cascaded block of nodes.",
        _reactivate_entity_schema(),
        "knowledge_bases.orchestration.mcp_tools:reactivate_entity_tool",
    ),

    _skill(
        "orchestration.spawn_ephemeral_subagent",
        "orchestration",
        "Spawn Ephemeral Subagent",
        "Spawn an isolated, ephemeral subagent acting as another graph-defined "
        "expert to resolve a focused question or operation. Prevents context "
        "pollution by executing the inquiry in an isolated trace scope, returning "
        "a clean structured result and recording a DelegationResult in the graph.",
        {
            "type": "object",
            "properties": {
                "expert_key": {
                    "type": "string",
                    "description": "The expert persona to spawn (e.g. 'semantic_layer', 'business_dictionary').",
                },
                "task": {
                    "type": "string",
                    "description": "The focused question or task for the subagent to solve.",
                },
                "operation": {
                    "type": "string",
                    "description": "Optional specific tool operation to execute directly.",
                },
                "starting_arguments": {
                    "type": "object",
                    "description": "Optional arguments payload for the tool operation.",
                },
            },
            "required": ["expert_key", "task"],
        },
        "knowledge_bases.orchestration.mcp_tools:spawn_ephemeral_subagent_tool",
    ),

    _skill(
        "orchestration.register_tool",
        "orchestration",
        "Register Tool",
        "Register a new tool or capability directly into the Noesis graph. "
        "The tool is saved as a :Tool node, indexed for semantic search "
        "(search_noesis_tools), and immediately discoverable by any agent "
        "or harness across all sessions.",
        {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Human-readable name of the tool (e.g. 'Query Postgres', 'Search Linear Issues').",
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of what the tool does, when to use it, and what parameters it accepts. Used for semantic search.",
                },
                "input_schema": {
                    "type": "object",
                    "description": "Optional JSON schema describing expected arguments. Defaults to empty object schema.",
                },
                "key": {
                    "type": "string",
                    "description": "Optional unique dotted key (e.g. 'custom.query_postgres'). Generated automatically from name if omitted.",
                },
                "knowledge_base": {
                    "type": "string",
                    "description": "Optional knowledge base / namespace. Defaults to 'custom'.",
                },
                "handler": {
                    "type": "string",
                    "description": "Optional execution handler (e.g. 'module:function' or 'harness:<name>'). Defaults to 'harness:custom'.",
                },
                "version": {
                    "type": "string",
                    "description": "Optional semantic version string. Defaults to '1.0.0'.",
                },
            },
            "required": ["name", "description"],
        },
        "knowledge_bases.orchestration.mcp_tools:register_tool_tool",
    ),


    _skill(
        "orchestration.get_noesis_context",
        "orchestration",
        "Get Noesis Context",
        "The complete context plane in one call: which graph-defined experts "
        "exist and what each handles, the callable tool surface with live "
        "descriptions, the orchestrator's durable instructions (base plus "
        "anything learned since), and what prior sessions established. Call "
        "this once at session start -- it is all graph data, so it reflects "
        "experts created, skills retired, and prompts refined since the "
        "harness launched, with no restart.",
        {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Include this session's resume digest.",
                },
                "include": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["experts", "tools", "instructions", "memory"],
                    },
                    "description": "Subset to return; defaults to all four.",
                },
            },
        },
        "knowledge_bases.orchestration.mcp_tools:get_noesis_context_tool",
    ),

    _skill(
        "orchestration.resume_noesis_session",
        "orchestration",
        "Resume Noesis Session",
        "Recover what an earlier conversation established so it can be "
        "continued: prior questions with their experts and verdicts, the "
        "entities that were resolved, what was left unresolved, and any open "
        "questions. Derived live from the session's recorded traces -- not a "
        "saved transcript -- so it summarizes established knowledge rather than "
        "replaying the conversation. Omit session_id for the most recent one.",
        {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Omit to resume this user's most recent session.",
                },
            },
        },
        "knowledge_bases.orchestration.mcp_tools:resume_noesis_session_tool",
    ),

    _skill(
        "orchestration.list_noesis_sessions",
        "orchestration",
        "List Noesis Sessions",
        "Recent conversations, most recently active first, with their first "
        "question and trace count -- for choosing which one to resume.",
        {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
        },
        "knowledge_bases.orchestration.mcp_tools:list_noesis_sessions_tool",
    ),

    _skill(
        "orchestration.get_delegation_results",
        "orchestration",
        "Get Delegation Results",
        "Every expert's structured result for one trace: the skills each "
        "called, the entities each resolved, errors, and (when the expert "
        "recorded it) its confidence and open items. Read this before "
        "synthesizing instead of re-reading each worker's prose, and in a "
        "dependent delegation to build on what an earlier expert resolved.",
        {
            "type": "object",
            "properties": {"trace_id": {"type": "string"}},
            "required": ["trace_id"],
        },
        "knowledge_bases.orchestration.mcp_tools:get_delegation_results_tool",
    ),

    _skill(
        "orchestration.create_expert",
        "orchestration",
        "Create Expert",
        "Define a new graph-defined expert (a routable subagent persona) from "
        "knowledge bases' already-registered skills. Refused if any "
        "allowed_tools pattern matches no active skill, if the capability "
        "wording reads more like an existing expert's, or if it would lower "
        "golden routing accuracy. Delegable immediately -- no harness restart.",
        _create_expert_schema(),
        "knowledge_bases.orchestration.mcp_tools:create_expert_tool",
    ),

    _skill(
        "orchestration.revise_expert",
        "orchestration",
        "Revise Expert",
        "Change an existing expert's name, description, prompt, capabilities, "
        "priority, entry skills, or allowed skills as a new versioned "
        "ExpertRevision. Same routing gates as create_expert.",
        _revise_expert_schema(),
        "knowledge_bases.orchestration.mcp_tools:revise_expert_tool",
    ),

    _skill(
        "orchestration.retire_expert",
        "orchestration",
        "Retire Expert",
        "Deactivate an expert so routing no longer selects it. Never deletes: "
        "revisions and telemetry stay intact and it can be brought back.",
        {
            "type": "object",
            "properties": {"key": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["key"],
        },
        "knowledge_bases.orchestration.mcp_tools:retire_expert_tool",
    ),

    _skill(
        "orchestration.set_expert_prompt_prefix",
        "orchestration",
        "Set Expert Prompt Prefix",
        "Set the bounded supplemental context rendered ahead of an expert's "
        "immutable base prompt -- deployment or session framing it should read "
        "first. Replaces the whole prefix; pass an empty list to clear. Max 4 "
        "entries / 2000 characters total, refused rather than truncated. Cannot "
        "override the base prompt's stated prohibitions.",
        _set_expert_prompt_prefix_schema(),
        "knowledge_bases.orchestration.mcp_tools:set_expert_prompt_prefix_tool",
    ),

    _skill(
        "orchestration.reject_noesis_write",
        "orchestration",
        "Reject Noesis Write",
        "Explicitly cancel a previously staged, not-yet-confirmed write "
        "proposal, recording an audit trail instead of a silent expiry.",
        {
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["proposal_id"],
        },
        "knowledge_bases.orchestration.mcp_tools:reject_noesis_write_tool",
    ),

    _skill(
        "orchestration.route_noesis_question",
        "orchestration",
        "Route Noesis Question",
        "Rank graph-defined experts for a question by keyword + semantic "
        "evidence and return a trace_id, ranked candidates (reasons, "
        "suggested_tools), and a starting delegation per selected expert. "
        "You decide among close candidates; pass decide=true for a "
        "server-side LLM decision instead.",
        {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "decide": {"type": "boolean"},
            },
            "required": ["question"],
        },
        "knowledge_bases.orchestration.mcp_tools:route_noesis_question_tool",
    ),

    _skill(
        "orchestration.delegate_noesis_question",
        "orchestration",
        "Delegate Noesis Question",
        "Route a question, invoke the selected bounded experts, and record the trace.",
        QUESTION_SCHEMA,
        "knowledge_bases.orchestration.mcp_tools:delegate_noesis_question_tool",
    ),

    _skill(
        "orchestration.get_noesis_routing_trace",
        "orchestration",
        "Get Noesis Routing Trace",
        "Get expert candidates and delegations for a routing trace.",
        TRACE_ID_INPUT,
        "knowledge_bases.orchestration.mcp_tools:get_noesis_routing_trace_tool",
    ),

    _skill(
        "orchestration.list_noesis_agent_skills",
        "orchestration",
        "List Noesis Agent Skills",
        "List the harness instruction bundles (Agent Skills) the graph "
        "offers, each with the one-line description a harness shows in its "
        "startup prompt. An Agent Skill is instructions a model is given, "
        "not a callable operation -- for the callable surface use "
        "search_noesis_tools. Bodies are omitted; fetch one with "
        "get_noesis_agent_skill.",
        {
            "type": "object",
            "properties": {"include_inactive": {"type": "boolean"}},
        },
        "knowledge_bases.orchestration.mcp_tools:list_noesis_agent_skills_tool",
    ),

    _skill(
        "orchestration.get_noesis_agent_skill",
        "orchestration",
        "Get Noesis Agent Skill",
        "Fetch one Agent Skill's full instruction body, live from the graph "
        "-- the call a generated SKILL.md stub tells you to make when you "
        "open a Noesis skill. The tool list inside the body is re-read from "
        "the current catalog on every call, so a retired or reworded tool is "
        "reflected immediately without any re-export or restart.",
        {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The skill key, e.g. 'noesis-semantic-layer'.",
                },
                "refresh": {
                    "type": "boolean",
                    "description": "Re-render the tool list from the live "
                                   "catalog (default true).",
                },
            },
            "required": ["key"],
        },
        "knowledge_bases.orchestration.mcp_tools:get_noesis_agent_skill_tool",
    ),

    _skill(
        "orchestration.sync_noesis_agent_skills",
        "orchestration",
        "Sync Noesis Agent Skills",
        "Regenerate every Agent Skill instruction bundle from the current "
        "tool catalog.",
        {"type": "object", "properties": {}},
        "knowledge_bases.orchestration.mcp_tools:sync_noesis_agent_skills_tool",
    ),


    _skill(
        "orchestration.entity_neighborhood",
        "orchestration",
        "Entity Neighborhood",
        "Walk out from one entity to what is related to it, grouped by "
        "relationship type. This is the step after hybrid_search: search "
        "finds an entity by meaning, this says what it connects to and "
        "how. Traversal is bounded by the declared ontology -- an "
        "undeclared relationship type is refused, not silently ignored.",
        {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Node label, e.g. 'Metric', 'Project'.",
                },
                "identity": {"type": "string"},
                "identity_property": {
                    "type": "string",
                    "description": "Property holding the identity (default 'id').",
                },
                "relationship_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict to these declared relationship "
                                   "types; omit for all.",
                },
                "depth": {
                    "type": "integer",
                    "description": "1-3. Capped because an unbounded walk "
                                   "returns the whole connected component.",
                },
                "limit": {"type": "integer"},
                "include_deprecated": {"type": "boolean"},
            },
            "required": ["label", "identity"],
        },
        "knowledge_bases.orchestration.mcp_tools:entity_neighborhood_tool",
    ),


    _skill(
        "orchestration.record_noesis_session_memory",
        "orchestration",
        "Record Noesis Session Memory",
        "Store a durable lesson from this session in the graph. Memory never "
        "expires: to correct one, pass `supersedes_key` and a new body -- the "
        "old memory is kept as history, not overwritten. `entity_links` may "
        "only point at entities that already exist, using relationship types "
        "the ontology already declares; anything else comes back in "
        "`rejected` for review instead of being written. An entity the "
        "graph does not have yet is created from `target_name` (matched "
        "against existing names first, then tagged provisional with a "
        "source), so a lesson about something new brings it into the "
        "graph; creating a *label* remains a reviewed schema change.",
        {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short imperative summary; part of the "
                                   "memory's identity, so keep it stable.",
                },
                "body": {
                    "type": "string",
                    "description": "The lesson in prose. This is what gets "
                                   "read back, so keep conditions and "
                                   "ordering intact.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["fact", "procedure", "pitfall", "preference"],
                },
                "scope": {
                    "type": "string",
                    "enum": ["session", "repo", "expert", "global"],
                    "description": "How widely this applies. `repo` needs "
                                   "`repo`; `expert` needs `expert_key`.",
                },
                "repo": {"type": "string", "description": "Repository id, or 'self' for Noesis's own source."},
                "expert_key": {"type": "string"},
                "session_id": {"type": "string"},
                "supersedes_key": {
                    "type": "string",
                    "description": "Key of the memory this corrects. The old "
                                   "one is marked superseded, never deleted.",
                },
                "create_missing_entities": {
                    "type": "boolean",
                    "description": "Create entities that do not exist yet "
                                   "(default true).",
                },
                "entity_links": {
                    "type": "array",
                    "description": "Triplets from this memory to entities, "
                                   "for retrieval by traversal. Entities "
                                   "that do not exist yet are created.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "predicate": {
                                "type": "string",
                                "description": "Must be a relationship type "
                                               "the ontology already declares.",
                            },
                            "target_id": {"type": "string"},
                            "target_label": {
                                "type": "string",
                                "description": "Must be a label the ontology "
                                               "already declares.",
                            },
                            "target_name": {
                                "type": "string",
                                "description": "Human-readable name. Used to "
                                               "match an existing entity "
                                               "before creating one; required "
                                               "to create.",
                            },
                            "source_text": {"type": "string"},
                        },
                        "required": ["predicate", "target_id", "target_label"],
                    },
                },
            },
            "required": ["title", "body"],
        },
        "knowledge_bases.orchestration.mcp_tools:record_noesis_session_memory_tool",
    ),


    _skill(
        "orchestration.search_noesis_session_memories",
        "orchestration",
        "Search Noesis Session Memories",
        "Retrieve durable lessons that apply here. Repo, expert, and global "
        "scopes are unioned, most specific first; superseded memories are "
        "hidden unless you ask for them.",
        {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository id, or 'self' for Noesis's own source."},
                "expert_key": {"type": "string"},
                "query": {"type": "string", "description": "Text filter."},
                "limit": {"type": "integer"},
                "include_superseded": {"type": "boolean"},
            },
        },
        "knowledge_bases.orchestration.mcp_tools:search_noesis_session_memories_tool",
    ),


    _skill(
        "orchestration.get_noesis_session_memory",
        "orchestration",
        "Get Noesis Session Memory",
        "Fetch one memory with its entity links, its supersession history, "
        "and any claims the ontology could not express.",
        {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
        "knowledge_bases.orchestration.mcp_tools:get_noesis_session_memory_tool",
    ),


    _skill(
        "orchestration.sync_noesis_experts",
        "orchestration",
        "Sync Noesis Experts",
        "Materialize the bounded knowledge-base experts and capabilities in Neo4j.",
        {"type": "object", "properties": {}},
        "knowledge_bases.orchestration.mcp_tools:sync_noesis_experts_tool",
    ),


    _skill(
        "orchestration.list_noesis_experts",
        "orchestration",
        "List Noesis Experts",
        "List every active graph-defined expert (key, name, description, "
        "capabilities, entry skills, version, routable). These are the "
        "subagents a harness can delegate to; pass include_skills to also "
        "get each expert's callable skills.",
        {
            "type": "object",
            "properties": {"include_skills": {"type": "boolean"}},
        },
        "knowledge_bases.orchestration.mcp_tools:list_noesis_experts_tool",
    ),


    _skill(
        "orchestration.get_noesis_expert",
        "orchestration",
        "Get Noesis Expert",
        "Fetch one graph-defined expert's full definition: system_prompt, "
        "entry_tools, and the live skills (with input schemas) it may call "
        "via call_noesis_tool. Instantiate the expert's subagent from this, "
        "and pass its key as expert_key on subsequent calls.",
        {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "include_revisions": {"type": "boolean"},
            },
            "required": ["key"],
        },
        "knowledge_bases.orchestration.mcp_tools:get_noesis_expert_tool",
    ),


    _skill(
        "orchestration.call_noesis_tool",
        "orchestration",
        "Call Noesis Tool",
        "Call one bounded tool owned by any knowledge base (e.g. "
        "resolve_business_language, resolve_metric_question, consult_workflow, "
        "list_project_blockers, list_compromised_decisions) through this "
        "single MCP connection, without needing a direct connection to "
        "that knowledge base's own MCP server. Use the knowledge_base/"
        "operation/arguments named in route_noesis_question's delegations.",
        _call_noesis_tool_schema(),
        "knowledge_bases.orchestration.mcp_tools:call_noesis_tool_tool",
    ),


]
