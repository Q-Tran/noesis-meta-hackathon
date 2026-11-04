"""Tell the connected harness when the live tool surface changed.

Every part of Noesis's tool surface is already read live from the graph: a
harness calling `tools/list` gets whatever `Tool` nodes are currently active
(`knowledge_bases.mcp.load_tool_definitions`), and dispatch resolves the handler at
call time. So retiring a skill, correcting an `input_schema`, or creating an
expert takes effect server-side immediately.

That is only half of "no restart", because **a harness caches the tool list**.
MCP's answer is the `notifications/tools/list_changed` server notification, and
a client that supports it re-fetches. Without sending it, a mid-session catalog
change is invisible to the running client until it reconnects -- the change is
live in the graph, and the model never learns about it.

This module is the missing edge. It is deliberately:

- **best-effort** -- a notification failure must never fail the write that
  triggered it. The graph is the source of truth; a missed notification costs a
  stale tool list until the next reconnect, not correctness.
- **fire-and-forget across the async boundary** -- the writes that trigger it
  are synchronous skill handlers (`(arguments, graph) -> Any`), called from
  inside an already-running event loop by the MCP server. Scheduling onto that
  loop keeps handlers synchronous rather than making every caller async.
- **connection-scoped** -- only sessions registered by a live server are
  notified, so a script or test that changes the catalog with no server
  attached is a silent no-op.

A client that does not advertise the capability is skipped by the SDK, so
sending unconditionally is safe.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Live MCP sessions to notify, keyed by id() of the session object. Populated
# by `knowledge_bases.mcp`'s request handlers, which are the only place a
# session is actually in scope.
_sessions: dict[int, Any] = {}
_loop: asyncio.AbstractEventLoop | None = None


def register_session(session: Any) -> None:
    """Remember a live session (and the loop it runs on) as notifiable.

    Called on every request rather than at connect time: the low-level MCP
    server exposes the session through its request context, so this is the
    only place it is reachable, and re-registering the same session is a
    cheap dict write.
    """
    global _loop
    if session is None:
        return
    _sessions[id(session)] = session
    if _loop is None:
        try:
            _loop = asyncio.get_running_loop()
        except RuntimeError:
            _loop = None


def unregister_session(session: Any) -> None:
    _sessions.pop(id(session), None)


def reset_for_testing() -> None:
    global _loop
    _sessions.clear()
    _loop = None


def notify_tool_list_changed() -> int:
    """Ask every live session to re-fetch its tool list.

    Returns how many sessions were notified (0 when nothing is connected --
    the normal case for a CLI script). Never raises.
    """
    if not _sessions:
        return 0
    notified = 0
    for session in list(_sessions.values()):
        if _schedule(_send(session)):
            notified += 1
    return notified


async def _send(session: Any) -> None:
    try:
        await session.send_tool_list_changed()
    except Exception:  # noqa: BLE001 -- a dead session must not break a write
        logger.debug("Could not send tools/list_changed", exc_info=True)
        unregister_session(session)


def _schedule(coro: Any) -> bool:
    """Run `coro` on the server's event loop from synchronous code.

    Returns whether it was scheduled. Closes the coroutine when it cannot be,
    so a skipped notification never leaves an un-awaited coroutine warning.
    """
    loop = _loop
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    target = running or loop
    if target is None or target.is_closed():
        coro.close()
        return False
    try:
        if running is target:
            target.create_task(coro)
        else:
            asyncio.run_coroutine_threadsafe(coro, target)
        return True
    except Exception:  # noqa: BLE001
        coro.close()
        logger.debug("Could not schedule tools/list_changed", exc_info=True)
        return False
