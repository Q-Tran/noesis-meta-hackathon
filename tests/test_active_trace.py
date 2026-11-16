"""`knowledge_bases.tools.active_trace` is the harness-agnostic correlation
mechanism: a client-side agent must never mint or remember a `trace_id`/
`session_id` itself for this to work (see `.opencode/agents/noesis-orchestrator.md`
and `knowledge_bases.mcp.trace_context`)."""
from __future__ import annotations

import knowledge_bases.tools.active_trace as active_trace


def setup_function():
    active_trace.reset_for_testing()


def teardown_function():
    active_trace.reset_for_testing()


def test_resolve_trace_context_has_no_active_trace_before_anything_is_set():
    trace_id, session_id = active_trace.resolve_trace_context(None, None)

    assert trace_id is None
    assert session_id  # a session id is still auto-minted


def test_set_active_trace_is_then_used_as_the_fallback():
    active_trace.set_active_trace("trace-42")

    trace_id, _ = active_trace.resolve_trace_context(None, None)

    assert trace_id == "trace-42"


def test_explicit_trace_id_wins_over_the_active_trace():
    active_trace.set_active_trace("trace-42")

    trace_id, _ = active_trace.resolve_trace_context("explicit-trace", None)

    assert trace_id == "explicit-trace"


def test_current_session_id_is_minted_once_and_reused():
    first = active_trace.current_session_id()
    second = active_trace.current_session_id()

    assert first == second
    assert first  # non-empty


def test_resolve_trace_context_reuses_the_same_session_id_across_calls():
    _, session_a = active_trace.resolve_trace_context(None, None)
    _, session_b = active_trace.resolve_trace_context(None, None)

    assert session_a == session_b


def test_explicit_session_id_wins_over_the_process_session_id():
    active_trace.current_session_id()  # mint the process session id first

    _, session_id = active_trace.resolve_trace_context(None, "explicit-session")

    assert session_id == "explicit-session"


def test_set_active_trace_updates_the_fallback_for_a_new_turn():
    active_trace.set_active_trace("turn-1")
    assert active_trace.resolve_trace_context(None, None)[0] == "turn-1"

    active_trace.set_active_trace("turn-2")
    assert active_trace.resolve_trace_context(None, None)[0] == "turn-2"


# -- per-connection isolation (B0) ------------------------------------------------

def test_two_connections_do_not_share_a_trace_or_session():
    """The assumption process-global state used to rely on -- one process, one
    conversation -- stops holding once sessions can be resumed. Two bound
    connections must not attribute each other's spans."""
    active_trace.bind_connection("conn-a")
    active_trace.set_active_trace("trace-a")
    session_a = active_trace.current_session_id()

    active_trace.bind_connection("conn-b")
    active_trace.set_active_trace("trace-b")
    session_b = active_trace.current_session_id()

    assert session_a != session_b
    assert active_trace.resolve_trace_context(None, None) == ("trace-b", session_b)

    active_trace.bind_connection("conn-a")
    assert active_trace.resolve_trace_context(None, None) == ("trace-a", session_a)


def test_identity_is_per_connection():
    active_trace.bind_connection("conn-a")
    active_trace.set_active_identity(expert_key="project_management", client="opencode")
    active_trace.bind_connection("conn-b")
    active_trace.set_active_identity(expert_key="epistemology")

    assert active_trace.resolve_identity()["expert_key"] == "epistemology"
    active_trace.bind_connection("conn-a")
    assert active_trace.resolve_identity()["expert_key"] == "project_management"


def test_rebinding_an_existing_connection_resumes_its_scope():
    active_trace.bind_connection("conn-a")
    first = active_trace.current_session_id()
    active_trace.bind_connection("conn-b")
    active_trace.bind_connection("conn-a")

    assert active_trace.current_session_id() == first


def test_releasing_a_connection_forgets_it_and_falls_back_to_default():
    active_trace.bind_connection("conn-a")
    active_trace.set_active_trace("trace-a")
    active_trace.release_connection("conn-a")

    assert active_trace.resolve_trace_context(None, None)[0] is None
    active_trace.bind_connection("conn-a")
    assert active_trace.active_trace_id() is None  # a fresh scope, not the old one


def test_unbound_callers_keep_the_single_scope_behaviour():
    """A stdio server serving one conversation needs no changes at all."""
    active_trace.set_active_trace("trace-1")
    session = active_trace.current_session_id()

    assert active_trace.resolve_trace_context(None, None) == ("trace-1", session)
    assert active_trace.current_scope().connection == active_trace.DEFAULT_CONNECTION


def test_adopt_session_id_continues_an_existing_session():
    """How a resumed conversation lands in the same :Session node instead of
    minting a new id."""
    minted = active_trace.current_session_id()
    adopted = active_trace.adopt_session_id("session-from-last-week")

    assert adopted == "session-from-last-week"
    assert active_trace.current_session_id() == "session-from-last-week" != minted
