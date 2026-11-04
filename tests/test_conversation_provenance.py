"""Conversation provenance: claims substantiated directly by the trace.

The turn is the record: `RoutingTrace.question` carries the verbatim words
(the trace store writes it every turn; the routing repository writes it at
plan time), and a filed claim is substantiated by one edge -- the provenance
spine's own `FOUND_IN`, aimed at the turn:
`(Claim)-[:FOUND_IN {role: "origin"}]->(RoutingTrace)`, the same edge and
role `extract.py` gives an ingested claim pointing at its verbatim `Chunk`.
These tests pin that shape, including
the negative half: no conversation `SourceNode`, no `Chunk` copies, no
per-proposal confirmation nodes (the `confirmed: true` already lives on the
execute span). Exercised against recorder fakes, no live graph (mirrors
`tests/test_provenance.py`).
"""
from types import SimpleNamespace

import pytest

from knowledge_bases.provenance import conversation
from knowledge_bases.tools.trace_store import TraceStore


class _Result(list):
    def consume(self):
        return self


class RecordingGraph:
    def __init__(self, run_rows=None, rel_rows=None):
        self.upserted_nodes = []
        self.upserted_rels = []
        self.queries = []
        self._run_rows = list(run_rows or [])
        self._rel_rows = list(rel_rows or [])

    def upsert_node(self, label, key, props=None):
        self.upserted_nodes.append((label, key, props))
        return []

    def upsert_rel(self, start_label, start_key, rel_type, end_label, end_key, props=None):
        self.upserted_rels.append((start_label, start_key, rel_type, end_label, end_key, props))
        return self._rel_rows.pop(0) if self._rel_rows else []

    def run(self, query, **params):
        self.queries.append((query, params))
        return _Result(self._run_rows.pop(0) if self._run_rows else [])


class TxRecordingGraph(RecordingGraph):
    """Captures the Cypher a `execute_write(work)` unit runs."""

    def execute_write(self, work):
        tx = RecordingGraph()
        work(tx)
        self.queries.extend(tx.queries)
        return []


# ------------------------------------------------- the trace keeps the verbatim text

def test_record_turn_start_writes_verbatim_question_outside_on_create():
    graph = RecordingGraph()

    TraceStore(graph).record_turn_start(
        "trace-1", "ses_1", "the full user prompt, verbatim", client="opencode",
    )

    query, params = graph.queries[0]
    # The full text is a plain SET: a tool-call span's MERGE often creates the
    # trace node first (detached tracer spawn races the MCP self-trace), and
    # under ON CREATE the prompt was then dropped -- 46 of 57 traces had none.
    assert "SET t.question = $question" in query
    assert params["question"] == "the full user prompt, verbatim"


def test_record_turn_start_links_session_and_trace():
    graph = RecordingGraph()

    TraceStore(graph).record_turn_start("trace-1", "ses_1", "q")

    query = graph.queries[0][0]
    assert "MERGE (s:Session {id: $session_id})" in query
    assert "MERGE (s)-[ht:HAS_TRACE]->(t)" in query
    assert "ON CREATE SET ht.created_at = datetime()" in query


def test_routing_plan_trace_carries_the_full_question_not_just_the_summary():
    """The MCP-side grounding resolves to the routing trace, so the verbatim
    words must be on it too -- the 240-char summary is not the evidence."""
    from knowledge_bases.orchestration.models import RoutingPlan
    from knowledge_bases.orchestration.repository import RoutingRepository

    full = ("For Enterprise plans, revenue is recognized monthly over the "
            "subscription term, not annually upfront -- decided 2026-10-29. " + "x" * 300)
    graph = TxRecordingGraph()
    plan = RoutingPlan(
        trace_id="trace-1", question_summary=full[:240], primary_expert=None,
        confidence=0.0, candidates=[], delegations=[], clarification_needed=True,
    )

    RoutingRepository(graph).record_plan(plan, full)

    create = next((q, p) for q, p in graph.queries if "CREATE (t:RoutingTrace" in q)
    assert "question: $question" in create[0]
    assert create[1]["question"] == full
    assert create[1]["summary"] == full[:240]


# ------------------------------------------------ the grounding edge itself

def test_ground_claim_merges_the_spine_edge_with_origin_role():
    """The grounding edge goes through `graph.upsert_rel`, which is the boundary
    that validates the type and endpoint labels against the registry and stamps
    `created_at`. So `FOUND_IN` stays the `{role}` edge the ontology declares --
    no bespoke timestamp property invented at the call site."""
    graph = RecordingGraph(rel_rows=[[{"claim_id": "claim-1", "trace_id": "trace-1"}]])

    result = conversation.ground_claim_in_trace(graph, "trace-1", "claim-1")

    assert result == {"claim_id": "claim-1", "trace_id": "trace-1", "grounded": True}
    start_label, start_key, rel_type, end_label, end_key, props = graph.upserted_rels[0]
    assert (start_label, rel_type, end_label) == ("Claim", "FOUND_IN", "RoutingTrace")
    assert start_key == {"id": "claim-1"}, "the claim is the subject: provenance points at its record"
    assert end_key == {"id": "trace-1"}
    assert props == {"role": "origin"}, "FOUND_IN declares only role; created_at comes from the boundary"
    assert graph.queries == [], "no hand-rolled Cypher for a write the boundary already owns"


def test_ground_claim_raises_when_there_is_nothing_to_substantiate_it():
    """A grounding edge to a trace that does not exist would be a provenance
    lie; the caller must hear about it, not receive a silent no-op."""
    with pytest.raises(ValueError, match="no Claim or RoutingTrace"):
        conversation.ground_claim_in_trace(RecordingGraph(), "trace-1", "claim-1")
    with pytest.raises(ValueError, match="trace_id is required"):
        conversation.ground_claim_in_trace(RecordingGraph(), "   ", "claim-1")


def test_the_trace_target_of_the_spine_is_a_declared_contract():
    """One spine edge for all verbatim provenance: FOUND_IN targets the
    SourceNode (origin metadata), the Chunk (document text, `extract.py`),
    and the RoutingTrace (the conversation turn substantiating user claims).
    The trace is the conversation's chunk -- no bespoke edge per source."""
    from knowledge_bases.registry import edge_contracts

    contract = next(c for c in edge_contracts() if c.relationship == "FOUND_IN")
    assert contract.to_labels == ("SourceNode", "Chunk", "RoutingTrace")

    relationships = {rel for contract in edge_contracts() if contract.relationship == "GROUNDED_IN"}
    assert relationships == set(), "the bespoke conversation edge was unified into FOUND_IN"


# ------------------------------------------------------ the removed layers stay removed

def test_the_conversation_indirection_layers_stay_removed():
    """Every one of these duplicated data the traces already keep: chunk
    copies, a per-session conversation SourceNode, per-proposal confirmation
    nodes. Grounding is one edge straight to the turn."""
    for gone in ("record_user_utterance", "attribute_utterance", "ground_claim",
                 "assert_from_conversation", "record_conversation_source",
                 "conversation_source_id", "ground_claim_in_conversation",
                 "record_confirmation_provenance"):
        assert not hasattr(conversation, gone), f"{gone} was indirection, not provenance"


def test_write_service_records_no_confirmation_node():
    import inspect

    from knowledge_bases.orchestration import write_service

    source = inspect.getsource(write_service)
    assert "record_confirmation_provenance" not in source
    assert "conversation" not in source


# ------------------------------------------------------------ tool-surface policy

def test_no_utterance_tools_and_feedback_identity_stays_runtime_supplied():
    """No record/attribute-utterance tools (the trace keeps the conversation),
    no model-supplied `user_id`/`session_id`/`chunk_id` (see
    docs/architecture/user_feedback_loop.md); `trace_id` is correlation, not
    identity, and stays optional."""
    from knowledge_bases.tools.catalog import bootstrap_catalog

    tools = {skill.key.removeprefix("orchestration."): skill
             for skill in bootstrap_catalog() if skill.knowledge_base == "orchestration"}

    for gone in ("record_user_utterance", "attribute_user_utterance"):
        assert gone not in tools
    feedback = tools["catalog_noesis_feedback"].input_schema
    props = feedback["properties"]
    assert "chunk_id" not in props
    assert "user_id" not in props
    assert "trace_id" in props
    assert "trace_id" not in feedback["required"]


def _fake_feedback(monkeypatch, grounding_results=None):
    """Patch the recorder capability + grounding so the tool's policy (resolve
    trace, ground every claim, report honestly) is what runs."""
    from types import SimpleNamespace

    import knowledge_bases.orchestration.mcp_tools as mt
    import knowledge_bases.registry as registry

    calls = []

    def fake_ground(graph, trace_id, claim_id):
        calls.append((trace_id, claim_id))
        if grounding_results is not None:
            raise ValueError(grounding_results)

    monkeypatch.setattr(mt.conversation, "ground_claim_in_trace", fake_ground)
    recorded = SimpleNamespace(claim_ids=["c1", "c2"],
                               model_dump=lambda mode=None: {"claim_ids": ["c1", "c2"]})
    recorder = SimpleNamespace(record=lambda claims, user_id: recorded)
    monkeypatch.setattr(registry, "capability",
                        lambda name: (lambda graph: recorder) if name == "feedback.record"
                        else SimpleNamespace(model_validate=lambda claim: claim))
    return mt, calls


def test_feedback_grounds_every_claim_in_the_active_trace(monkeypatch):
    from knowledge_bases.tools import active_trace

    mt, calls = _fake_feedback(monkeypatch)
    monkeypatch.setattr(active_trace, "current_scope", lambda: SimpleNamespace(
        active_trace_id="trace-active",
        current_session_id=lambda: "ses_x",
        identity={},
    ))

    payload = mt.catalog_noesis_feedback_tool(
        {"claims": [{"statement": "fact one"}, {"statement": "fact two"}]},
        RecordingGraph(),
    )

    assert calls == [("trace-active", "c1"), ("trace-active", "c2")]
    grounding = payload["conversation_grounding"]
    assert [row["trace_id"] for row in grounding] == ["trace-active", "trace-active"]
    assert all(row["grounded"] is True for row in grounding)


def test_explicit_trace_id_wins_over_the_active_one(monkeypatch):
    mt, calls = _fake_feedback(monkeypatch)

    payload = mt.catalog_noesis_feedback_tool(
        {"claims": [{"statement": "fact one"}], "trace_id": "trace-explicit"},
        RecordingGraph(),
    )

    assert calls == [("trace-explicit", "c1"), ("trace-explicit", "c2")]
    assert payload["conversation_grounding"][0]["grounded"] is True


def test_claims_filed_without_trace_context_are_reported_ungrounded(monkeypatch):
    """Filing succeeds (the user's fact is stored), but the missing
    substantiation is echoed, never swallowed."""
    mt, calls = _fake_feedback(monkeypatch)

    payload = mt.catalog_noesis_feedback_tool(
        {"claims": [{"statement": "fact one"}]},
        RecordingGraph(),
    )

    assert calls == []
    row = payload["conversation_grounding"][0]
    assert row["trace_id"] is None
    assert row["grounded"] is None
    assert "ungrounded" in row["reason"]


def test_missing_trace_endpoint_is_reported_per_claim(monkeypatch):
    mt, calls = _fake_feedback(monkeypatch, grounding_results="no Claim or RoutingTrace")

    payload = mt.catalog_noesis_feedback_tool(
        {"claims": [{"statement": "fact one"}], "trace_id": "trace-gone"},
        RecordingGraph(),
    )

    row = payload["conversation_grounding"][0]
    assert row["grounded"] is False
    assert "RoutingTrace" in row["reason"]
