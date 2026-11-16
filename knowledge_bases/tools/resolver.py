"""Resolve a graph-registered `Tool.handler` path to a live Python callable.

Tool nodes only ever store a dotted `"module:function"` reference, not
code. This is the one place that turns that string into something callable,
so every MCP server can dispatch dynamically instead of hardcoding a
`{name: handler}` map.
"""
from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any, Callable

from graph import Graph

Handler = Callable[[dict[str, Any], Graph], Any]

# Keyed by module name (not the full "module:function" path -- several
# skills can share one module and should share one reload), holding the
# live module object and the source file's mtime at last import/reload.
# Manual caching instead of `functools.lru_cache` specifically so an edited
# handler file takes effect on the very next dispatch with no process
# restart: a graph-registered skill's Python *code*, unlike its
# description/schema (already always read live from Neo4j, see
# `knowledge_bases.tools.repository.ToolRepository.load_tools`), would
# otherwise be frozen at whatever it was the first time this process
# resolved it. `None` mtime means "no file to watch" (e.g. a built-in/
# namespace module) -- cached indefinitely in that case, same as the old
# `lru_cache` behavior, since there's nothing to detect a change against.
_module_cache: dict[str, tuple[ModuleType, float | None]] = {}


class HandlerResolutionError(RuntimeError):
    pass


def _source_mtime(module: ModuleType) -> float | None:
    source_file = getattr(module, "__file__", None)
    if not source_file:
        return None
    try:
        return os.stat(source_file).st_mtime
    except OSError:
        return None


def _load_module(module_name: str) -> ModuleType:
    """Import `module_name`, or reload it if its source file has changed
    since the last time this process imported/reloaded it. Failures are
    never cached (this function isn't decorated with any cache), so a
    fixed import can be retried on the very next call, matching the
    previous `lru_cache`-based implementation's stated guarantee.
    """
    cached = _module_cache.get(module_name)
    if cached is None:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise HandlerResolutionError(
                f"Cannot import handler module {module_name!r}: {exc}"
            ) from exc
        _module_cache[module_name] = (module, _source_mtime(module))
        return module

    module, cached_mtime = cached
    if cached_mtime is None:
        # No file to watch (e.g. a built-in module) -- nothing to compare,
        # keep using the already-imported module object.
        return module
    current_mtime = _source_mtime(module)
    if current_mtime is not None and current_mtime != cached_mtime:
        try:
            module = importlib.reload(module)
        except Exception as exc:
            raise HandlerResolutionError(
                f"Cannot reload changed handler module {module_name!r}: {exc}"
            ) from exc
        _module_cache[module_name] = (module, _source_mtime(module))
    return module


def resolve_handler(path: str) -> Handler:
    """Import `module:function` and return the callable.

    Re-imports the module the first time it's needed, and transparently
    `importlib.reload()`s it on any later call if its source file's mtime
    has changed since -- so editing a skill's handler implementation on
    disk takes effect on the very next dispatch, with no MCP server
    restart, exactly like the graph-registered `Tool.description`/
    `input_schema` this same handler is attached to (already always read
    live, never cached at process start -- see
    `knowledge_bases.tools.repository.ToolRepository.load_tools`).
    Failures are never cached, so a fixed import/reload can be retried
    immediately without a process restart either.
    """
    if not path or path.startswith("harness:") or path.startswith("external:"):
        return lambda arguments, graph: {
            "status": "external_tool",
            "handler": path or "external",
            "message": f"This tool is managed by an external harness ({path or 'external'}). Execute it within your harness environment.",
            "arguments": arguments,
        }
    if ":" not in path:
        raise HandlerResolutionError(
            f"Invalid handler path {path!r}; expected 'module.submodule:function'"
        )
    module_name, _, function_name = path.partition(":")
    module = _load_module(module_name)
    try:
        handler = getattr(module, function_name)
    except AttributeError as exc:
        raise HandlerResolutionError(
            f"Module {module_name!r} has no attribute {function_name!r}"
        ) from exc
    if not callable(handler):
        raise HandlerResolutionError(f"{path} is not callable")
    return handler


# Backward-compat shim: `resolve_handler` used to be `@lru_cache`-memoized by
# the full "module:function" path, and several tests call
# `resolve_handler.cache_clear()` between cases to force a fresh import.
# Caching moved to the module level (`_module_cache`, see above), so expose
# an equivalent `cache_clear` here rather than updating every call site.
resolve_handler.cache_clear = _module_cache.clear


def invoke_tool(path: str, arguments: dict[str, Any], graph: Graph) -> Any:
    return resolve_handler(path)(arguments, graph)
