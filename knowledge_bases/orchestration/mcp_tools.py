"""MCP-callable handlers for the orchestrator's own skill catalog.

These are the mixture-of-experts control plane's own capabilities (routing,
feedback cataloging, bounded write propose/execute) -- not a delegated
domain skill -- but they still register in the graph like every other
skill (see `knowledge_bases/tools/catalog.py`) so the orchestrator's MCP
surface has the same single graph-driven origin as every domain connector.
"""
from __future__ import annotations

from typing import Any

from graph import Graph
from knowledge_bases.mcp import dispatch_tool
from knowledge_bases.orchestration import expert_admin
from knowledge_bases.orchestration.subagents import spawn_ephemeral_subagent
from knowledge_bases.orchestration.context_plane import get_noesis_context
from knowledge_bases.orchestration.delegation_results import (
    get_delegation_results,
    record_delegation_result,
)
from knowledge_bases.orchestration.experts_catalog import ORCHESTRATOR_KEY
from knowledge_bases.orchestration.hybrid_search import hybrid_search as _hybrid_search
from knowledge_bases.orchestration.lifecycle import deprecate_entity, reactivate_entity
from knowledge_bases.orchestration.repository import RoutingRepository, _skill_allowed
from knowledge_bases.orchestration.service import ExpertOrchestrator
from knowledge_bases.orchestration.sessions import SessionRepository
from knowledge_bases.orchestration.write_service import KnowledgeWriteService
from knowledge_bases.provenance import conversation
from knowledge_bases.registry import call_noesis_tool_knowledge_bases
from knowledge_bases.tools.active_trace import (
    adopt_session_id,
    current_session_id,
    resolve_identity,
    resolve_trace_context,
    set_active_identity,
    set_active_trace,
)
from knowledge_bases.tools.models import OutcomeSource, TraceOutcome
from knowledge_bases.tools.queries import get_trace, get_trace_metrics
from knowledge_bases.tools.search import search_noesis_tools as _search_noesis_tools
from knowledge_bases.tools.tracing import record_trace_outcome

# Every knowledge base with its own dispatchable `Tool`s and its own
# `mcp_server.py` (see `knowledge_bases/*/mcp_server.py`), reachable through
# `call_noesis_tool_tool` instead of a direct connection to each one.
# `orchestration` is excluded: its own tools are already directly callable
# on this same server. `provenance` is cross-cutting infrastructure (see
# `knowledge_bases/registry.py`) that owns no dispatchable `Tool`.
# `development` includes the reads over Noesis's own source (retired from the
# separate `self_catalog` knowledge base in migration 23), so the expert that
# answers "how is Noesis built" dispatches through here like any other.
# Shared with `knowledge_bases.tools.catalog
# ._call_noesis_tool_schema` so the tool's declared `knowledge_base` enum
# and this handler's own validation can never drift apart.
# Derived from each knowledge base's `call_noesis_tool` declaration in its
# `package.py` (see `knowledge_bases/base.py`). Membership is what matters;
# the declaration order in `registry.DEFAULT_PACKAGES` defines the tuple.
CALL_NOESIS_TOOL_KNOWLEDGE_BASES = call_noesis_tool_knowledge_bases()


def search_noesis_tools_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Search the graph-registered `Tool` catalog by plain-language need,
    across every knowledge base (or one, via `knowledge_base`) -- e.g. "I
    need to check project blockers" surfaces `project_management.list_project_blockers`
    even if the caller doesn't know that tool's exact name yet.

    Every result is already directly callable: Noesis's own tools are graph
    metadata pointing at code already deployed wherever a
    `noesis-orchestrator` process runs (see
    `knowledge_bases/tools/resolver.py`), so there is no separate install
    step -- call it via `call_noesis_tool` (or directly, if the caller has
    its own MCP connection to that `knowledge_base`) as soon as it's found.
    """
    return _search_noesis_tools(
        arguments["need"], arguments.get("limit", 10), graph,
        knowledge_base=arguments.get("knowledge_base"),
    )


def hybrid_search_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """The one cross-domain search primitive a client-side agent is expected
    to call directly and repeatedly while it drives its own reasoning/tool
    loop (see `docs/architecture/graph_driven_mcp_connectors.md`). This is
    dispatched like every other `orchestration` skill, through
    `knowledge_bases.mcp.dispatch_tool`'s generic `SpanEvent` wrapping --
    which reads the caller's `trace_id`/`session_id` straight out of
    `arguments` (see `knowledge_bases.mcp.trace_context`) when the caller
    (not an internal orchestrator dispatch) supplied them -- so a
    client-driven sequence of `hybrid_search` calls for one user turn lands
    in one trace even outside `delegate_noesis_question`.
    """
    return _hybrid_search(
        arguments["query"], arguments["entity_types"],
        arguments.get("filters"), arguments.get("limit", 10), graph,
    )


def catalog_noesis_feedback_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Record user-supplied facts as claims.

    Every filed claim is substantiated directly by the conversation turn it
    came from: `(Claim)-[:FOUND_IN {role: "origin"}]->(:RoutingTrace)`, whose recorded
    `question` carries the verbatim words. The trace resolves from trusted
    runtime context (`trace_id` argument, else this connection's active
    trace), never a model-supplied identity, and no copy of the conversation
    is made. When there is no trace to ground into, that is reported on the
    result -- filing is never silently ungrounded.

    Resolved as a capability rather than imported: the tool writes `Claim` nodes, so
    it only works when a package owning that label is installed, and the engine must
    still import this module in a product that ships without one.

    Raises when the capability is absent instead of degrading quietly, which is the
    opposite of `OrchestrationService.delegate`. The difference is what the caller
    asked for: delegate was asked to answer a question and cataloging is a side
    effect, but this tool was called for the express purpose of recording something,
    so silently recording nothing would tell the user their fact was stored.
    """
    from knowledge_bases.registry import capability

    recorder = capability("feedback.record")
    claim_model = capability("feedback.claim_model")
    if recorder is None or claim_model is None:
        raise RuntimeError(
            "cataloging feedback needs an installed package providing the "
            "'feedback.record' capability (the claims ontology); this product "
            "ships without one, so there is nowhere to record a claim"
        )
    recorded = recorder(graph).record(
        [claim_model.model_validate(claim) for claim in arguments["claims"]],
        user_id="unknown",
    )

    # Direct trace grounding: the claim's substantiation is the turn, not a
    # copy of it. A missing trace endpoint is reported per claim (grounded:
    # false/null with the reason), never swallowed -- the claim is filed
    # either way, the user learns its provenance is incomplete.
    trace_id, _ = resolve_trace_context(arguments.get("trace_id"), None)
    claim_ids = list(getattr(recorded, "claim_ids", []))
    grounding = []
    for claim_id in claim_ids:
        if not trace_id:
            grounding.append({
                "claim_id": claim_id, "trace_id": None, "grounded": None,
                "reason": "no trace context on this call: the claim is filed "
                          "ungrounded (route/delegate first, or pass trace_id)",
            })
            continue
        try:
            conversation.ground_claim_in_trace(graph, trace_id, claim_id)
            grounding.append({"claim_id": claim_id, "trace_id": trace_id, "grounded": True})
        except ValueError as exc:
            grounding.append({
                "claim_id": claim_id, "trace_id": trace_id, "grounded": False,
                "reason": str(exc),
            })
    payload = (recorded.model_dump(mode="json")
               if hasattr(recorded, "model_dump") else dict(recorded))
    payload["conversation_grounding"] = grounding
    return payload


def list_noesis_write_commands_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return KnowledgeWriteService(graph).list_commands()


def propose_noesis_write_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return KnowledgeWriteService(graph).propose(
        arguments["command"], arguments["arguments"], arguments["summary"]
    )


def execute_noesis_write_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return KnowledgeWriteService(graph).execute(
        arguments["proposal_id"], arguments["confirmed"]
    )


def reject_noesis_write_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return KnowledgeWriteService(graph).reject(
        arguments["proposal_id"], arguments.get("reason")
    )


def get_noesis_context_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """The whole context plane in one call: which experts exist and what they
    handle, the callable tool surface, the orchestrator's durable instructions,
    and what prior sessions established.

    Call this once at session start. Everything it returns is graph data, so it
    reflects experts created, skills retired, and prompts refined since the
    harness was launched -- no restart, no re-export.
    """
    return get_noesis_context(arguments, graph)


def resume_noesis_session_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Recover what a previous conversation established, to continue it.

    Returns a *derived* digest -- recomputed from the session's recorded
    traces, delegation results, and outcomes -- not a saved transcript or a
    serialized process state. Omit `session_id` to resume this user's most
    recent session. Adopting the session also re-points subsequent spans at
    it, so the resumed turns extend the same conversation instead of starting
    a parallel one.
    """
    repository = SessionRepository(graph)
    identity = resolve_identity(arguments)
    session_id = arguments.get("session_id") or repository.most_recent(
        user_id=identity.get("user_id"),
    )
    if not session_id:
        return {
            "found": False,
            "message": "No previous session recorded for this user.",
            "sessions": [],
        }
    digest = repository.digest(session_id)
    if digest.get("found"):
        adopt_session_id(session_id)
    return digest


def list_noesis_sessions_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Recent conversations, most recently active first."""
    identity = resolve_identity(arguments)
    sessions = SessionRepository(graph).list_sessions(
        user_id=arguments.get("user_id") or identity.get("user_id"),
        limit=arguments.get("limit", 10),
    )
    return {"sessions": [s.model_dump(mode="json") for s in sessions]}


def record_delegation_result_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """An expert records its own judgement for the delegation it just finished.

    The factual record (skills called, entities resolved, errors) is derived
    from spans either way; this adds what spans cannot hold -- confidence,
    what could not be settled, and why the expert stopped.
    """
    return record_delegation_result(arguments, graph)


def get_delegation_results_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Every expert's structured result for one trace.

    Read this before synthesizing rather than re-reading each worker's prose,
    and read it in a dependent delegation to build on what an earlier expert
    already resolved.
    """
    return get_delegation_results(arguments, graph)


def spawn_ephemeral_subagent_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Spawn an ephemeral subagent with its own isolated scope to resolve a task."""
    return spawn_ephemeral_subagent(arguments, graph)


def register_tool_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Register a new tool or capability directly into the graph."""
    from knowledge_bases.tools.tool_admin import register_tool
    return register_tool(arguments, graph)


def create_expert_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return expert_admin.create_expert(arguments, graph)


def revise_expert_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return expert_admin.revise_expert(arguments, graph)


def retire_expert_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return expert_admin.retire_expert(arguments, graph)


def set_expert_prompt_prefix_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return expert_admin.set_expert_prompt_prefix(arguments, graph)


def deprecate_entity_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return deprecate_entity(arguments, graph)


def reactivate_entity_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    return reactivate_entity(arguments, graph)


def route_noesis_question_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Retrieval-based routing: ranks graph-defined experts by keyword +
    semantic evidence and returns a `trace_id`, ranked `candidates` (with
    `reasons` and `suggested_tools`), and per-expert starting
    `delegations` -- which bounded tool (via `call_noesis_tool`) or
    `hybrid_search` arguments to begin with. The calling harness LLM makes
    the final call among close candidates; pass `decide: true` to have
    Noesis's own LLM do so server-side instead.
    Every plan also carries `ingestion`, naming the two calls that record
    whatever the question asserts. It is unconditional on purpose: what can be
    written is decided by whether it fits the ontology, not by whether the
    phrasing sounded like a request to store something, and an assertion usually
    arrives inside a question rather than instead of one.
    Remembered as this process's active trace (see
    `knowledge_bases.tools.active_trace`) so every later call in this turn
    that doesn't carry its own `trace_id` still correlates with this one --
    with no id bookkeeping asked of the calling model.
    """
    plan = ExpertOrchestrator(graph).route(
        arguments["question"], decide=bool(arguments.get("decide", False)),
    )
    set_active_trace(plan.trace_id)
    # Routing is the orchestrator acting, so attribute it. Without this, the
    # orchestrator never appears in `Span.expert_key` and
    # `TraceStore.expert_corrections` can never surface *its* prompt as the
    # thing to improve -- only the experts it delegates to. An expert worker
    # overrides this for its own calls via `get_noesis_expert`.
    set_active_identity(expert_key=ORCHESTRATOR_KEY)
    return plan


def delegate_noesis_question_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    result = ExpertOrchestrator(graph).delegate(arguments["question"], user_id="unknown")
    set_active_trace(result.trace_id)
    return result


def get_noesis_routing_trace_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Merge the routing decision with the skill/tool spans it produced, so
    one call returns the full trace: *which* experts were considered/
    delegated to (`RoutingTrace`) and *what actually ran* for each
    delegation (`Span` nodes), correlated by the same `trace_id` (see
    `knowledge_bases.tools.tracing.record_tool_span`).
    """
    trace_id = arguments["trace_id"]
    routing = RoutingRepository(graph).get_trace(trace_id)
    return {
        "routing": routing,
        "spans": get_trace(trace_id, graph),
        "metrics": get_trace_metrics(trace_id, graph),
    }


def sync_noesis_experts_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    ExpertOrchestrator(graph).sync_experts()
    return {"synced": True}


def _expert_payload(
    repository: RoutingRepository, expert: Any, include_skills: bool,
) -> dict[str, Any]:
    payload = expert.model_dump(mode="json")
    payload["base_system_prompt"] = expert.system_prompt
    payload["base_capabilities"] = list(expert.capabilities)
    payload["system_prompt"] = expert.effective_prompt
    payload["capabilities"] = expert.effective_capabilities
    if include_skills:
        payload["tools"] = repository.expert_tools(expert)
    return payload


def list_noesis_experts_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Every active graph-defined expert -- what a harness enumerates to
    know which subagents exist. Routable experts are the ones
    `route_noesis_question` can delegate to; the `orchestrator` entry is
    the primary agent's own definition."""
    repository = RoutingRepository(graph)
    experts = repository.load_experts(include_unroutable=True)
    if not experts:
        ExpertOrchestrator(graph).sync_experts()
        experts = repository.load_experts(include_unroutable=True)
    return {
        "experts": [
            _expert_payload(repository, expert, arguments.get("include_skills", False))
            for expert in experts
        ]
    }


def get_noesis_expert_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """One expert's complete definition: `system_prompt`, `entry_tools`,
    and the live `skills` (with `input_schema`) it may call -- everything a
    harness needs to instantiate that expert's subagent. Also served as an
    MCP prompt named after the expert's key."""
    repository = RoutingRepository(graph)
    expert = repository.get_expert(arguments["key"])
    if expert is None:
        raise ValueError(f"Unknown expert {arguments['key']!r}")
    payload = _expert_payload(repository, expert, include_skills=True)
    if arguments.get("include_revisions"):
        payload["revisions"] = repository.expert_revisions(expert.key)
    set_active_identity(expert_key=expert.key)
    return payload


def list_noesis_agent_skills_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Every harness instruction bundle the graph currently offers, with the
    one-line description a harness shows in its startup prompt.

    This is the roster half of dynamic skill lookup. Bodies are omitted on
    purpose: a harness only needs descriptions until the model actually opens
    one, which is what `get_noesis_agent_skill` is for."""
    from knowledge_bases.orchestration.agent_skills import AgentSkillRepository
    from knowledge_bases.orchestration.agent_skills_catalog import sync_agent_skills

    repository = AgentSkillRepository(graph)
    skills = repository.load_skills(arguments.get("include_inactive", False))
    if not skills:
        # First call on a fresh graph: derive them from the tool catalog rather
        # than reporting an empty roster a harness would cache as "no skills".
        sync_agent_skills(graph)
        skills = repository.load_skills(arguments.get("include_inactive", False))
    return {
        "source": "noesis-graph",
        "skills": [
            {
                "key": skill.key,
                "name": skill.name,
                "description": skill.description,
                "knowledge_base": skill.knowledge_base,
                "version": skill.version,
                "active": skill.active,
            }
            for skill in skills
        ],
    }


def get_noesis_agent_skill_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """One instruction bundle's full body, read live from the graph.

    The call a generated stub tells the model to make when it opens a Noesis
    skill. Because the body is resolved here and not baked into the file, a
    reworded instruction or a retired tool is reflected on the very next
    lookup -- no re-export, no restart."""
    from knowledge_bases.orchestration.agent_skills import AgentSkillRepository
    from knowledge_bases.orchestration.agent_skills_catalog import (
        render_body,
        sync_agent_skills,
    )

    key = arguments["key"]
    repository = AgentSkillRepository(graph)
    skill = repository.get_skill(key)
    if skill is None:
        sync_agent_skills(graph)
        skill = repository.get_skill(key)
    if skill is None:
        known = [entry.key for entry in repository.load_skills()]
        raise ValueError(f"Unknown agent skill {key!r}. Available: {known}")

    payload = skill.model_dump(mode="json")
    if arguments.get("refresh", True) and skill.knowledge_base:
        # Re-render the tool list from the current catalog, so the body reflects
        # the graph as of *this* call rather than as of the last sync.
        from knowledge_bases.orchestration.agent_skills_catalog import _tool_rows
        from knowledge_bases.registry import KNOWLEDGE_BASES

        kb = KNOWLEDGE_BASES.get(skill.knowledge_base)
        tools = [
            row for row in _tool_rows(graph)
            if row["knowledge_base"] == skill.knowledge_base
        ]
        if kb is not None and tools:
            payload["body"] = render_body(skill.knowledge_base, kb.name, tools)
    return payload


def sync_noesis_agent_skills_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Regenerate every instruction bundle from the current tool catalog."""
    from knowledge_bases.orchestration.agent_skills_catalog import sync_agent_skills

    skills = sync_agent_skills(graph)
    return {"synced": [skill.key for skill in skills]}


def _enforce_expert_scope(
    arguments: dict[str, Any], knowledge_base: str, graph: Graph,
) -> None:
    """Hold execution to the same boundary as discovery.

    `search_noesis_tools` already narrows results to the skills an expert's
    `allowed_tools` permits, but `call_noesis_tool` only ever validated the
    knowledge base -- so an expert could call anything in any reachable
    knowledge base, and "never bypass a knowledge base's bounded tools" was
    prompt text rather than an enforced rule. Since every worker dispatches
    through this one proxy, this is the natural chokepoint to enforce it.

    Deliberately permissive in two cases: a call with no `expert_key` (the
    orchestrator's own direct calls, and any harness not acting as an expert)
    and an `expert_key` the graph does not know. Neither is an escalation --
    the knowledge-base allowlist still applies -- and failing closed here
    would break every unattributed caller.
    """
    expert_key = arguments.get("expert_key")
    if not isinstance(expert_key, str) or not expert_key:
        return
    expert = RoutingRepository(graph).get_expert(expert_key)
    if expert is None:
        return
    skill_key = f"{knowledge_base}.{arguments['operation']}"
    if not _skill_allowed(skill_key, expert.allowed_tools):
        raise ValueError(
            f"Expert {expert_key!r} may not call {skill_key!r}: its allowed skills "
            f"are {', '.join(expert.allowed_tools)}. Use search_noesis_tools with "
            "your expert_key to find a tool you are permitted, or report that the "
            "delegation needs a different expert."
        )


def call_noesis_tool_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Call one bounded tool owned by any knowledge base, through the
    single `noesis-orchestrator` MCP connection.

    Every knowledge base (`business_dictionary`, `semantic_layer`, `methodology`,
    `project_management`, `architecture`, `strategy`, `claims`, `skills`) is its own
    MCP server process (`knowledge_bases/*/mcp_server.py`) -- a harness that
    connects only to `noesis-orchestrator` (the common, recommended setup;
    see `.opencode/agents/noesis-orchestrator.md`) has no direct connection
    to any of those and so cannot call e.g. `resolve_business_language` or
    `consult_workflow` by name, even though `route_noesis_question`'s
    `delegations` name exactly those operations. This proxies to
    `knowledge_bases.mcp.dispatch_tool` -- the same deterministic,
    graph-registered dispatch every domain server already uses internally,
    no server-side LLM involved -- so one MCP connection can still reach
    every bounded tool. `hybrid_search` remains directly callable on this
    same server for cross-domain/exploratory lookups; this tool is for
    everything else `route_noesis_question` might recommend.
    """
    knowledge_base = arguments["knowledge_base"]
    if knowledge_base not in CALL_NOESIS_TOOL_KNOWLEDGE_BASES:
        raise ValueError(
            f"Unknown knowledge base {knowledge_base!r}; "
            f"expected one of {sorted(CALL_NOESIS_TOOL_KNOWLEDGE_BASES)}"
        )
    _enforce_expert_scope(arguments, knowledge_base, graph)
    # `trace_id`/`session_id` are correlation hints on *this* call
    # (`call_noesis_tool`'s own top-level arguments); the inner
    # `arguments` payload is the delegated tool's own arguments and
    # shouldn't need to duplicate them, so they're passed through
    # explicitly rather than relying on `dispatch_tool`'s
    # arguments-dict fallback to see them.
    inner = dict(arguments.get("arguments", {}))
    for key in ("user_id", "client", "expert_key"):
        if arguments.get(key) and key not in inner:
            inner[key] = arguments[key]
    return dispatch_tool(
        graph, knowledge_base, arguments["operation"], inner,
        trace_id=arguments.get("trace_id"), session_id=arguments.get("session_id"),
    )


def record_noesis_outcome_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Record whether a trace's result held up, for a `trace_id` from
    *any* prior call -- a routed `delegate_noesis_question`/
    `route_noesis_question` trace, or a trace_id a client-side agent minted
    itself for its own `hybrid_search`/bounded-tool loop (see
    `knowledge_bases.mcp.trace_context`). No `RoutingTrace` is required:
    this only ever writes a `TraceOutcome` node that `skill_health()` mines
    (see `knowledge_bases.tools.trace_store`), keyed purely by `trace_id`.
    Never raises on a write failure -- an unavailable graph degrades to
    `recorded: false`.
    """
    outcome = TraceOutcome(
        trace_id=arguments["trace_id"],
        success=arguments["success"],
        reason=arguments.get("reason"),
        source=OutcomeSource(arguments.get("source", OutcomeSource.user_correction.value)),
    )
    recorded = record_trace_outcome(outcome)
    return {"recorded": recorded, "outcome_id": outcome.outcome_id}

def record_noesis_session_memory_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Store one durable lesson, optionally linked to entities that exist.

    Linking is resolve-only and ontology-checked, so a rejected link comes back
    in `rejected` rather than silently succeeding or quietly widening the
    schema.
    """
    from knowledge_bases.orchestration.memory import SessionMemoryRepository
    from knowledge_bases.orchestration.memory_models import SessionMemory

    memory = SessionMemory.model_validate({
        key: value for key, value in arguments.items()
        if key != "supersedes_key" and value is not None
    })
    repository = SessionMemoryRepository(graph)

    old_key = arguments.get("supersedes_key")
    if old_key:
        return repository.supersede(old_key, memory)
    return repository.record(memory)


def search_noesis_session_memories_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    from knowledge_bases.orchestration.memory import SessionMemoryRepository

    return SessionMemoryRepository(graph).search(
        repo=arguments.get("repo"),
        expert_key=arguments.get("expert_key"),
        query=arguments.get("query"),
        limit=int(arguments.get("limit", 20)),
        include_superseded=bool(arguments.get("include_superseded", False)),
    )


def get_noesis_session_memory_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    from knowledge_bases.orchestration.memory import SessionMemoryRepository

    result = SessionMemoryRepository(graph).get(arguments["key"])
    if result is None:
        return {"error": f"No session memory with key {arguments['key']!r}."}
    return result


def entity_neighborhood_tool(arguments: dict[str, Any], graph: Graph) -> Any:
    """Walk out from an entity to what the ontology says is related to it."""
    from knowledge_bases.orchestration.neighborhood import (
        NeighborhoodError,
        entity_neighborhood,
    )

    try:
        return entity_neighborhood(
            graph,
            arguments["label"],
            arguments["identity"],
            identity_property=arguments.get("identity_property", "id"),
            relationship_types=arguments.get("relationship_types"),
            depth=int(arguments.get("depth", 1)),
            limit=int(arguments.get("limit", 50)),
            include_deprecated=bool(arguments.get("include_deprecated", False)),
        )
    except NeighborhoodError as exc:
        return {"error": str(exc)}
