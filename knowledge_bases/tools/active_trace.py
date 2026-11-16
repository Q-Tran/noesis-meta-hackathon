"""Per-connection "active trace" state for harness-agnostic correlation.

A client-side agent (any MCP harness -- OpenCode, Claude Desktop, a bare
script) should never have to mint or remember a `trace_id`/`session_id`
itself: that would put ID bookkeeping in the model's hands, which is exactly
what this module removes. Instead:

- `route_noesis_question`/`delegate_noesis_question` mint a `trace_id`
  (`knowledge_bases.orchestration.routing.plan_route`, plain `uuid4()`, no
  LLM). This module remembers that as the connection's *active* trace.
- Every other call this connection dispatches (`hybrid_search`,
  `call_noesis_tool`, bounded tools, generic browsing) that doesn't explicitly
  pass its own `trace_id`/`session_id` (see `knowledge_bases.mcp.trace_context`)
  falls back to whatever is active -- so correlation happens automatically,
  with zero participation from the calling model and zero per-harness code.

## Why this is a scope object and not module globals

It used to be module globals, which was safe only under an assumption stated
in this docstring: *one server process serves one conversation at a time*.
Durable sessions (`:Session` nodes, `resume_noesis_session`) break that
assumption -- the moment a session can be resumed, and the moment a server
might field two of them, process-global state silently attributes one
conversation's spans to another's session.

So state now lives in a `TraceScope`, and the process keeps a *registry* of
scopes keyed by connection. The default key preserves the old behaviour
exactly (one implicit connection = one scope), so a stdio server that really
does serve one conversation needs no changes; a server that fields several
calls `bind_connection(key)` per connection and gets isolation.

`contextvars` is deliberately still not the mechanism: the MCP SDK dispatches
each incoming request as its own `asyncio` task (`Server.run`'s
`tg.start_soon`), and a `contextvar.set()` inside one request's handler is not
visible to a later, separate request's handler -- verified directly;
`contextvars` do not propagate across sibling tasks. An explicit registry
does, and unlike module globals it can hold more than one conversation.

Lives under `knowledge_bases.tools` (general tracing infrastructure, see
`knowledge_bases.tools.tracing`), not `knowledge_bases.orchestration`, so
`knowledge_bases.mcp` -- used by every domain's `mcp_server.py`, including
orchestration's own -- can import it without a circular dependency back
through `knowledge_bases.orchestration.__init__`'s `ExpertOrchestrator`
import.

A `session_id` is minted once per scope, automatically, the first time that
scope is asked for one.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

# The connection key used when a caller never binds one: a stdio server
# spawned per harness connection (see `opencode.json`'s `"type": "local"`),
# which is the common case.
DEFAULT_CONNECTION = "__default__"


@dataclass
class TraceScope:
    """One conversation's correlation state.

    `session_id` is lazily minted so merely creating a scope does not invent a
    session that never gets used.
    """

    connection: str = DEFAULT_CONNECTION
    active_trace_id: str | None = None
    session_id: str | None = None
    identity: dict[str, str | None] = field(
        default_factory=lambda: {"user_id": None, "client": None, "expert_key": None}
    )

    def current_session_id(self) -> str:
        if self.session_id is None:
            self.session_id = str(uuid4())
        return self.session_id


# Guarded because a threaded harness may bind connections from several threads;
# the per-scope fields themselves are only touched by their own connection.
_lock = threading.Lock()
_scopes: dict[str, TraceScope] = {}
_current_connection = DEFAULT_CONNECTION


def bind_connection(connection: str | None) -> TraceScope:
    """Make `connection` the active scope, creating it if needed.

    Called once per MCP connection by a server that serves more than one (see
    `knowledge_bases.mcp.create_domain_server`). `None` restores the default
    scope, which is what a single-conversation stdio server always uses.
    """
    global _current_connection
    key = connection or DEFAULT_CONNECTION
    with _lock:
        _current_connection = key
        return _scopes.setdefault(key, TraceScope(connection=key))


def current_scope() -> TraceScope:
    """The active connection's scope, created on first use."""
    with _lock:
        return _scopes.setdefault(
            _current_connection, TraceScope(connection=_current_connection)
        )


def release_connection(connection: str) -> None:
    """Forget one connection's scope when it closes, so a long-lived server
    does not accumulate state for conversations that are over."""
    global _current_connection
    with _lock:
        _scopes.pop(connection, None)
        if _current_connection == connection:
            _current_connection = DEFAULT_CONNECTION


def set_active_trace(trace_id: str) -> None:
    """Remember `trace_id` as this connection's active trace.

    Called by `route_noesis_question_tool`/`delegate_noesis_question_tool`
    right after `plan_route()` mints it -- never by the calling model.
    """
    current_scope().active_trace_id = trace_id


def active_trace_id() -> str | None:
    return current_scope().active_trace_id


def current_session_id() -> str:
    """Return this connection's session id, minting one on first use.

    Deterministic Python, not a model decision: the id is generated once per
    scope (`uuid4()`) and reused for every call thereafter, giving every
    harness a session correlation id with zero effort on its part.
    """
    return current_scope().current_session_id()


def adopt_session_id(session_id: str) -> str:
    """Continue an existing session in this scope.

    How `resume_noesis_session` makes a resumed conversation land in the same
    `:Session` as the original instead of minting a fresh id: subsequent spans
    correlate to the session the user actually resumed.
    """
    current_scope().session_id = session_id
    return session_id


def resolve_trace_context(
    trace_id: str | None, session_id: str | None,
) -> tuple[str | None, str]:
    """Fill in whatever the caller omitted from this connection's state.

    `trace_id` stays `None` when there is no active trace yet (e.g. a
    `hybrid_search` call made before any `route_noesis_question` on this
    connection) -- callers already fall back further to a fresh id at that
    point (see `knowledge_bases.tools.tracing.record_tool_span`).
    `session_id` never stays `None`: it always resolves to this connection's
    session id unless the caller explicitly supplied a different one.
    """
    scope = current_scope()
    return (
        trace_id if trace_id is not None else scope.active_trace_id,
        session_id if session_id is not None else scope.current_session_id(),
    )


def set_active_identity(
    user_id: str | None = None, client: str | None = None, expert_key: str | None = None,
) -> None:
    """Remember who this connection is acting for: the `user_id`, the harness
    (`client`, e.g. "opencode"), and the graph-defined expert currently
    acting (`expert_key`). Each is only overwritten when given, so a harness
    can set `user_id`/`client` once at session start (via
    `NOESIS_USER_ID`/`NOESIS_CLIENT` or an explicit argument) and an expert
    worker can set just `expert_key` when it adopts an expert."""
    identity = current_scope().identity
    for key, value in (("user_id", user_id), ("client", client), ("expert_key", expert_key)):
        if value:
            identity[key] = value


def resolve_identity(arguments: dict[str, Any] | None = None) -> dict[str, str | None]:
    """Identity fields for a span: explicit `user_id`/`client`/`expert_key`
    in `arguments` win, then this connection's active identity, then the
    `NOESIS_USER_ID`/`NOESIS_CLIENT` environment (how a harness config
    like `opencode.json` tags every server it spawns)."""
    import os

    arguments = arguments or {}
    identity = current_scope().identity
    env_defaults = {"user_id": os.getenv("NOESIS_USER_ID"), "client": os.getenv("NOESIS_CLIENT")}
    resolved = {}
    for key in ("user_id", "client", "expert_key"):
        explicit = arguments.get(key)
        resolved[key] = (
            explicit if isinstance(explicit, str) and explicit
            else identity[key] or env_defaults.get(key)
        )
    return resolved


def reset_for_testing() -> None:
    """Test-only: drop every scope and return to the default connection."""
    global _current_connection
    with _lock:
        _scopes.clear()
        _current_connection = DEFAULT_CONNECTION
