"""Durable sessions, delegation results, and the agent mailbox.

The claims under test are the ones that justify doing this in the graph rather
than with a serialized kernel snapshot: the resume digest is *derived*, and the
delegation record is *guaranteed* because it comes from spans rather than from
an expert choosing to report.
"""
from __future__ import annotations

import json

import pytest

from knowledge_bases.orchestration import mcp_tools
from knowledge_bases.orchestration.subagents import (
    SubagentError,
    spawn_ephemeral_subagent,
)
from knowledge_bases.orchestration.delegation_results import (
    _entities_from,
    derive_delegation_results,
    get_delegation_results,
    record_delegation_result,
)
from knowledge_bases.orchestration.sessions import SessionRepository
from knowledge_bases.tools import active_trace
from knowledge_bases.tools.resolver import resolve_handler
from knowledge_bases.tools.tool_admin import SkillAdminError, register_tool


def setup_function():
    active_trace.reset_for_testing()


def teardown_function():
    active_trace.reset_for_testing()


class RecordingGraph:
    """Routes each query shape to a canned result and records every write."""

    def __init__(self, results=None):
        self.results = results or {}
        self.writes = []

    def run(self, query, **params):
        self.writes.append((query, params))
        for marker, rows in self.results.items():
            if marker in query:
                return rows(params) if callable(rows) else rows
        return []


# -- sessions -------------------------------------------------------------------

def test_recording_a_session_is_idempotent_and_moves_the_activity_clock():
    graph = RecordingGraph()
    SessionRepository(graph).record_session("s1", user_id="u", client="opencode")

    query, params = graph.writes[0]
    assert "MERGE (s:Session {id: $id})" in query
    assert "ON CREATE SET s.started_at" in query
    assert "SET s.last_active_at" in query
    assert params == {"id": "s1", "user_id": "u", "client": "opencode"}


def test_digest_of_an_unknown_session_reports_not_found_rather_than_raising():
    assert SessionRepository(RecordingGraph()).digest("ghost") == {
        "session_id": "ghost", "found": False,
    }


def test_digest_is_derived_from_traces_and_says_so():
    graph = RecordingGraph({
        "RETURN s.id AS id, s.started_at AS started_at": [
            {"id": "s1", "started_at": "t0", "last_active_at": "t1",
             "user_id": "u", "client": "opencode"},
        ],
        "t.question_summary AS question": [
            {"trace_id": "tr1", "question": "What is ARPU?",
             "primary_expert": "business_dictionary", "confidence": 0.9,
             "status": "complete", "success": True, "outcome_reason": None},
        ],
        "RETURN DISTINCT r.entity_labels[i] AS label": [
            {"label": "BusinessTerm", "key": "arpu", "expert_key": "business_dictionary"},
        ],
        "r.unresolved AS unresolved": [
            {"expert_key": "semantic_layer", "unresolved": ["Q3 grain"], "trace_id": "tr1"},
        ],
    })
    digest = SessionRepository(graph).digest("s1")

    assert digest["found"]
    assert digest["turns"][0]["question"] == "What is ARPU?"
    assert digest["established_entities"][0]["key"] == "arpu"
    assert digest["unresolved"][0]["unresolved"] == ["Q3 grain"]
    assert "not a saved transcript" in digest["note"]
    # Nothing about the digest is a stored blob: every part came from a read.
    assert all("CREATE" not in q and "MERGE" not in q for q, _ in graph.writes)


def test_resume_adopts_the_session_so_later_spans_extend_it():
    """Without this a resumed conversation would start a parallel session."""
    graph = RecordingGraph({
        "RETURN s.id AS id, s.started_at AS started_at": [
            {"id": "s-old", "started_at": "t0", "last_active_at": "t1",
             "user_id": None, "client": None},
        ],
    })
    minted = active_trace.current_session_id()
    result = mcp_tools.resume_noesis_session_tool({"session_id": "s-old"}, graph)

    assert result["found"]
    assert active_trace.current_session_id() == "s-old" != minted


def test_resume_without_a_session_id_and_no_history_is_a_clean_miss():
    result = mcp_tools.resume_noesis_session_tool({}, RecordingGraph())

    assert result["found"] is False
    assert result["sessions"] == []


def test_a_routed_plan_links_its_trace_to_the_session():
    from knowledge_bases.orchestration.models import RoutingPlan
    from knowledge_bases.orchestration.repository import RoutingRepository

    writes = []

    class Tx:
        def run(self, query, **params):
            writes.append((query, params))

            class Result:
                def consume(self_inner):
                    return None

            return Result()

    class Graph:
        def execute_write(self, work):
            return work(Tx())

    plan = RoutingPlan(trace_id="tr1", question_summary="q", primary_expert="project_management",
                       confidence=0.8, candidates=[], delegations=[])
    RoutingRepository(Graph()).record_plan(plan, "q")

    linked = next(q for q, _ in writes if "MERGE (s)-[ht:HAS_TRACE]->(t)" in q)
    assert "MERGE (s:Session {id: $session_id})" in linked
    assert "ON CREATE SET ht.created_at = datetime()" in linked
    created = next(p for q, p in writes if "CREATE (t:RoutingTrace" in q)
    assert created["session_id"] == active_trace.current_session_id()


# -- delegation results: derived (tier 1) ---------------------------------------

def _span(expert_key, skill, output=None, status="ok", error=None):
    return {"expert_key": expert_key, "skill": skill, "status": status,
            "error": error, "raw_output": output}


class SpanGraph:
    """A graph whose only content is one trace's spans."""

    def __init__(self, spans, declared=None):
        self.spans = spans
        self.declared = declared or []
        self.writes = []

    def run(self, query, **params):
        self.writes.append((query, params))
        if "MATCH (s:Span {trace_id: $trace_id})" in query:
            return [{"span": dict(s)} for s in self.spans]
        if "MATCH (r:DelegationResult {trace_id: $trace_id})" in query:
            return self.declared
        return []


def test_results_are_derived_from_spans_with_no_expert_participation():
    """The core tier-1 guarantee: no expert had to call anything for this."""
    graph = SpanGraph([
        _span("business_dictionary", "business_dictionary.resolve_business_language",
              {"canonical_name": "ARPU", "label": "BusinessTerm"}),
        _span("business_dictionary", "business_dictionary.metrics_for_business_term",
              {"metric": "arpu_monthly"}),
        _span("semantic_layer", "semantic_layer.find_metrics", None, "error", "boom"),
    ])
    results = derive_delegation_results("tr1", graph)

    by_expert = {r["expert_key"]: r for r in results}
    assert by_expert["business_dictionary"]["tools_called"] == [
        "business_dictionary.resolve_business_language",
        "business_dictionary.metrics_for_business_term",
    ]
    assert {"label": "BusinessTerm", "key": "ARPU"} in by_expert["business_dictionary"]["entities"]
    assert by_expert["semantic_layer"]["errors"] == [
        {"skill": "semantic_layer.find_metrics", "error": "boom"},
    ]
    assert all(r["declared"] is False for r in results)


def test_spans_without_an_expert_key_are_still_reported():
    """Unattributed work must not vanish from the record."""
    results = derive_delegation_results("tr1", SpanGraph([_span(None, "orchestration.hybrid_search")]))

    assert results[0]["expert_key"] == "unattributed"


def test_missing_declared_results_are_reported_as_a_compliance_signal():
    graph = SpanGraph([_span("project_management", "project_management.list_project_blockers")])
    payload = get_delegation_results({"trace_id": "tr1"}, graph)

    assert payload["experts_without_declared_results"] == ["project_management"]


def test_declared_judgement_merges_onto_the_derived_record():
    graph = SpanGraph(
        [_span("project_management", "project_management.list_project_blockers", {"id": "epic-1"})],
        declared=[{"expert_key": "project_management", "summary": "Two blockers.",
                   "confidence": 0.8, "unresolved": ["owner of epic-2"],
                   "stopped_because": "resolved", "recorded_at": "t"}],
    )
    result = derive_delegation_results("tr1", graph)[0]

    assert result["declared"] is True
    assert result["summary"] == "Two blockers."
    assert result["unresolved"] == ["owner of epic-2"]
    # Derived facts survive the merge.
    assert result["tools_called"] == ["project_management.list_project_blockers"]


def test_a_declared_result_with_no_spans_is_still_returned():
    """Found live: derivation iterated only over experts that had spans, so a
    declared result from an expert whose spans were never written (tracing is
    best-effort and swallows failures) vanished entirely."""
    graph = SpanGraph(
        [_span("project_management", "project_management.list_project_blockers")],
        declared=[{"expert_key": "epistemology", "summary": "Nothing found.",
                   "confidence": 0.3, "unresolved": [], "stopped_because": "dead end",
                   "recorded_at": "t"}],
    )
    results = {r["expert_key"]: r for r in derive_delegation_results("tr1", graph)}

    assert set(results) == {"project_management", "epistemology"}
    assert results["epistemology"]["declared"] is True
    assert results["epistemology"]["span_count"] == 0
    assert results["epistemology"]["summary"] == "Nothing found."
    assert results["project_management"]["tools_called"] == ["project_management.list_project_blockers"]


@pytest.mark.parametrize("payload,expected", [
    ({"key": "a", "label": "Metric"}, ("Metric", "a")),
    ({"id": "b", "type": "Table"}, ("Table", "b")),
    ({"name": "c"}, ("unknown", "c")),
    ({"nested": [{"key": "d", "label": "Column"}]}, ("Column", "d")),
])
def test_entity_extraction_handles_the_shapes_payloads_actually_use(payload, expected):
    assert expected in _entities_from(payload)


def test_entity_extraction_never_raises_on_odd_payloads():
    for payload in (None, "text", 42, [], {}, {"a": {"b": {"c": {"d": {"e": 1}}}}}):
        assert isinstance(_entities_from(payload), list)


def test_recording_a_result_stores_derived_facts_alongside_the_judgement():
    graph = SpanGraph([_span("project_management", "project_management.list_project_blockers", {"id": "epic-1"})])
    result = record_delegation_result({
        "trace_id": "tr1", "expert_key": "project_management", "summary": "s",
        "confidence": 0.7, "unresolved": ["x"], "stopped_because": "resolved",
    }, graph)

    assert result["recorded"] == "tr1:project_management"
    write = next(p for q, p in graph.writes if "MERGE (r:DelegationResult" in q)
    assert write["tools_called"] == ["project_management.list_project_blockers"]
    assert write["entity_keys"] == ["epic-1"]
    assert write["unresolved"] == ["x"]


# -- expert scope enforcement ---------------------------------------------------

class ExpertScopeGraph:
    def __init__(self, allowed_tools):
        self.allowed_tools = allowed_tools

    def run(self, query, **params):
        if "MATCH (e:KnowledgeBaseExpert {key: $key})" in query:
            return [{"key": params["key"], "name": "N", "description": "d",
                     "allowed_tools": self.allowed_tools, "capabilities": [],
                     "system_prompt": "p", "entry_tools": [], "routable": True,
                     "priority": 10, "version": 1, "active": True}]
        return []


def test_an_expert_cannot_call_a_skill_outside_its_allowed_skills():
    """Discovery was already scoped; execution now matches it."""
    graph = ExpertScopeGraph(["project_management.*"])
    with pytest.raises(ValueError, match="may not call"):
        mcp_tools._enforce_expert_scope(
            {"expert_key": "project_management", "operation": "consult_workflow"}, "methodology", graph,
        )


def test_an_expert_may_call_its_declared_cross_knowledge_base_skills():
    graph = ExpertScopeGraph(["semantic_layer.*", "business_dictionary.resolve_business_language"])
    mcp_tools._enforce_expert_scope(
        {"expert_key": "semantic_layer", "operation": "resolve_business_language"},
        "business_dictionary", graph,
    )


def test_calls_without_an_expert_key_are_not_blocked():
    """The orchestrator's own calls, and any harness not acting as an expert."""
    mcp_tools._enforce_expert_scope({"operation": "find_metrics"}, "semantic_layer",
                                    ExpertScopeGraph([]))
    mcp_tools._enforce_expert_scope({"expert_key": "", "operation": "find_metrics"},
                                    "semantic_layer", ExpertScopeGraph([]))


def test_every_seeded_expert_may_make_the_calls_its_loop_contract_mandates():
    from knowledge_bases.orchestration.experts_catalog import (
        SHARED_EXPERT_TOOLS,
        bootstrap_experts,
    )
    from knowledge_bases.orchestration.repository import _skill_allowed

    for expert in bootstrap_experts():
        for skill in SHARED_EXPERT_TOOLS:
            assert _skill_allowed(skill, expert.allowed_tools), (
                f"{expert.key} could not call {skill}, which its prompt requires"
            )
        for skill in expert.entry_tools:
            assert _skill_allowed(skill, expert.allowed_tools), expert.key


# -- ephemeral subagents --------------------------------------------------------

class SubagentGraph:
    def __init__(self, known_expert=True, allowed_tools=None, active=True, tool_rows=None, spans=None):
        self.known_expert = known_expert
        self.allowed_tools = allowed_tools or ["*"]
        self.active = active
        self.tool_rows = tool_rows or []
        self.spans = spans or []
        self.writes = []

    def run(self, query, **params):
        self.writes.append((query, params))
        if "MATCH (e:KnowledgeBaseExpert {key: $key})" in query:
            if not self.known_expert:
                return []
            return [{
                "key": params["key"], "name": params["key"].title(), "description": "desc",
                "priority": 10, "system_prompt": "You are the expert.", "entry_tools": [],
                "allowed_tools": self.allowed_tools, "model_hint": None, "version": 1,
                "active": self.active, "routable": True, "capabilities": [],
                "capability_overlays": None, "prompt_overlays": None, "prompt_prefix_overlays": None,
            }]
        if "MATCH (s:Tool {active: true})" in query:
            return self.tool_rows
        if "MATCH (s:Span {trace_id: $trace_id})" in query:
            return self.spans
        return []

    def execute_write(self, fn):
        class FakeTx:
            def __init__(self, parent):
                self.parent = parent

            def run(self, query, **params):
                self.parent.writes.append((query, params))
                class FakeResult:
                    def consume(self):
                        pass
                return FakeResult()

        return fn(FakeTx(self))



def test_spawn_subagent_refuses_empty_task():
    graph = SubagentGraph()
    with pytest.raises(SubagentError, match="cannot be empty"):
        spawn_ephemeral_subagent({"expert_key": "semantic_layer", "task": "   "}, graph)


def test_spawn_subagent_refuses_unknown_expert():
    graph = SubagentGraph(known_expert=False)
    with pytest.raises(SubagentError, match="Unknown expert"):
        spawn_ephemeral_subagent({"expert_key": "ghost", "task": "Investigate something."}, graph)


def test_spawn_subagent_refuses_self_delegation():
    active_trace.set_active_identity(expert_key="semantic_layer")
    graph = SubagentGraph()
    with pytest.raises(SubagentError, match="cannot spawn a subagent of itself"):
        spawn_ephemeral_subagent({"expert_key": "semantic_layer", "task": "Investigate something."}, graph)


def test_spawn_subagent_refuses_disallowed_operation():
    graph = SubagentGraph(allowed_tools=["semantic_layer.get_metric"])
    with pytest.raises(SubagentError, match="not allowed to run"):
        spawn_ephemeral_subagent(
            {
                "expert_key": "semantic_layer",
                "task": "Check blockers",
                "operation": "project_management.list_project_blockers",
            },
            graph,
        )


def dummy_tool_handler(arguments: dict[str, Any], graph: Any) -> Any:
    return {"metric": arguments.get("metric_name", "unknown"), "status": "active", "grain": "monthly"}


def test_spawn_subagent_executes_isolated_investigation():
    """Subagent executes in an isolated context and preserves caller's identity."""
    active_trace.set_active_identity(expert_key="epistemology")
    graph = SubagentGraph(
        allowed_tools=["semantic_layer.*"],
        tool_rows=[{
            "key": "semantic_layer.get_metric",
            "name": "Get Metric",
            "version": "1.0.0",
            "knowledge_base": "semantic_layer",
            "active": True,
            "description": "Get metric definition",
            "input_schema": json.dumps({
                "type": "object",
                "properties": {"metric_name": {"type": "string"}},
                "required": [],
            }),
            "handler": "tests.test_sessions_and_delegation:dummy_tool_handler",
        }],
    )

    result = spawn_ephemeral_subagent(
        {
            "expert_key": "semantic_layer",
            "task": "Find ARPU definition",
            "operation": "get_metric",
            "starting_arguments": {"metric_name": "ARPU"},
        },
        graph,
    )

    assert result["status"] == "completed"
    assert result["expert_key"] == "semantic_layer"
    assert result["subagent_id"].startswith("subagent-")
    assert result["tools_called"] == ["semantic_layer.get_metric"]
    assert "ARPU" in result["summary"] or "completed" in result["summary"]
    # Caller identity preserved (not overwritten by subagent)
    assert active_trace.resolve_identity({})["expert_key"] == "epistemology"
    # Recorded a DelegationResult write
    assert any("MERGE (r:DelegationResult" in q for q, _ in graph.writes)


def test_spawn_subagent_supports_agent_and_prompt_aliases():
    """Harnesses like OpenCode pass agent/prompt or subagent_type/instruction."""
    active_trace.set_active_identity(expert_key="orchestrator")
    graph = SubagentGraph(
        allowed_tools=["semantic_layer.*"],
        tool_rows=[{
            "key": "semantic_layer.get_metric",
            "name": "Get Metric",
            "version": "1.0.0",
            "knowledge_base": "semantic_layer",
            "active": True,
            "description": "Get metric definition",
            "input_schema": json.dumps({
                "type": "object",
                "properties": {"metric_name": {"type": "string"}},
                "required": [],
            }),
            "handler": "tests.test_sessions_and_delegation:dummy_tool_handler",
        }],
    )

    result = spawn_ephemeral_subagent(
        {
            "agent": "semantic_layer",
            "prompt": "Find MRR calculation",
            "operation": "get_metric",
            "starting_arguments": {"metric_name": "MRR"},
        },
        graph,
    )
    assert result["status"] == "completed"
    assert result["expert_key"] == "semantic_layer"
    assert result["task"] == "Find MRR calculation"


def test_spawn_subagent_requires_expert_key():
    graph = SubagentGraph()
    with pytest.raises(SubagentError, match="expert_key"):
        spawn_ephemeral_subagent({"task": "Some task without agent"}, graph)


def test_register_tool_success_with_auto_generated_key():
    graph = SubagentGraph()
    result = register_tool(
        {
            "name": "Query Snowflake",
            "description": "Execute read-only queries against Snowflake data warehouse.",
            "knowledge_base": "analytics",
        },
        graph,
    )

    assert result["status"] == "registered"
    assert result["key"] == "analytics.query_snowflake"
    assert result["name"] == "Query Snowflake"
    assert result["knowledge_base"] == "analytics"
    assert result["handler"] == "harness:analytics"
    assert result["active"] is True
    assert any("MERGE (s:Tool {key: $key})" in q for q, _ in graph.writes)


def test_register_tool_with_custom_schema_and_handler():
    graph = SubagentGraph()
    result = register_tool(
        {
            "name": "Search Slack",
            "description": "Search company Slack messages.",
            "key": "custom.slack_search",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "handler": "my_module:search_slack",
        },
        graph,
    )

    assert result["key"] == "custom.slack_search"
    assert result["handler"] == "my_module:search_slack"
    assert result["input_schema"]["required"] == ["query"]


def test_register_tool_validates_required_fields():
    graph = SubagentGraph()
    with pytest.raises(SkillAdminError, match="name cannot be empty"):
        register_tool({"name": "", "description": "Some tool"}, graph)

    with pytest.raises(SkillAdminError, match="description cannot be empty"):
        register_tool({"name": "Some Tool", "description": "   "}, graph)


def test_external_harness_handler_dispatch():
    """Tools registered with harness:* handlers return cleanly when invoked."""
    handler = resolve_handler("harness:opencode")
    out = handler({"query": "SELECT 1"}, SubagentGraph())
    assert out["status"] == "external_tool"
    assert "harness:opencode" in out["message"]
    assert out["arguments"] == {"query": "SELECT 1"}


def test_call_noesis_tool_dispatches_orchestration_tools():
    """When tools are gated behind search_noesis_tools, orchestration tools
    (like register_tool) can be called through call_noesis_tool."""
    from knowledge_bases.orchestration.mcp_tools import call_noesis_tool_tool

    tool_row = {
        "key": "orchestration.register_tool",
        "name": "Register Tool",
        "version": "1",
        "knowledge_base": "orchestration",
        "description": "Register a new tool directly into the graph",
        "active": True,
        "input_schema": '{"type": "object"}',
        "handler": "knowledge_bases.tools.tool_admin:register_tool",
    }
    graph = SubagentGraph(tool_rows=[tool_row])
    out = call_noesis_tool_tool(
        {
            "knowledge_base": "orchestration",
            "operation": "register_tool",
            "arguments": {
                "name": "ClickHouse Query",
                "description": "Run clickstream queries",
                "knowledge_base": "events",
            },
        },
        graph,
    )
    assert out["status"] == "registered"
    assert out["key"] == "events.clickhouse_query"


