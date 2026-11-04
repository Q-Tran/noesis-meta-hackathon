"""Runtime skill lifecycle: fix a bad argument schema, retire, reactivate.

A skill's `description` was already changeable in the graph
(`Tool.description_override`, how self-improvement rewords a tool), but two
things still needed a code edit and a re-sync:

- **A wrong `input_schema`.** The schema is what a harness shows the model, so
  a missing property or a wrong enum makes a working handler uncallable. That
  is a data problem with a code-shaped fix, which is the wrong shape.
- **Retiring a skill.** `catalog.py` had to set `active=false` and be re-synced,
  so removing a misfiring tool from every user's surface meant a deploy.

Both are now graph writes, through the ordinary propose/confirm/execute flow.
Neither can widen what a skill *does*: the `handler` is untouched, so this
cannot point a skill at different code -- that stays a reviewed source edit
(`source_editor.py`).

The schema override is validated against the handler's real contract rather
than merely being well-formed JSON Schema: a schema that stops requiring an
argument the handler indexes would turn every call into a `KeyError` at
dispatch, which is worse than the wrong schema it replaced.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, model_validator

from graph import Graph
from knowledge_bases.tools.catalog_notify import notify_tool_list_changed
from knowledge_bases.tools.repository import ToolRepository

# Arguments every skill accepts for correlation, injected into every seeded
# schema by `catalog._skill`. An override doesn't have to repeat them, and
# dropping them is never treated as removing a required argument.
_TRACE_PROPERTIES = frozenset({"trace_id", "session_id", "user_id", "client", "expert_key"})

# `arguments["name"]` in a handler's source -- a read with no default, so the
# schema must declare it. `arguments.get("name")` is deliberately NOT matched:
# it has a default and is therefore optional by construction.
_REQUIRED_ARGUMENT_READ = re.compile(
    r"""arguments\[\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]\s*\]"""
)


class SkillAdminError(ValueError):
    """A refused skill change: unknown skill, malformed schema, or a schema
    that would break the handler it describes."""


class OverrideInputSchemaRequest(BaseModel):
    key: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class SetSkillActiveRequest(BaseModel):
    key: str
    reason: str = ""


class RegisterToolRequest(BaseModel):
    name: str = ""
    description: str = ""
    key: str = ""
    knowledge_base: str = "custom"
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    handler: str = ""
    version: str = "1.0.0"

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if not data.get("name"):
                for alias in ("tool_name", "title"):
                    if data.get(alias):
                        data["name"] = data[alias]
                        break
            if not data.get("description"):
                for alias in ("doc", "help", "summary", "about", "instructions"):
                    if data.get(alias):
                        data["description"] = data[alias]
                        break
            if not data.get("input_schema"):
                for alias in ("schema", "parameters", "arguments_schema"):
                    if data.get(alias):
                        data["input_schema"] = data[alias]
                        break
            if not data.get("knowledge_base"):
                for alias in ("namespace", "kb", "domain"):
                    if data.get(alias):
                        data["knowledge_base"] = data[alias]
                        break
        return data


def _required_names(schema: dict[str, Any]) -> set[str]:
    required = schema.get("required") or []
    if not isinstance(required, list):
        raise SkillAdminError("'required' must be a list of property names.")
    return {name for name in required if isinstance(name, str)}


def validate_input_schema_override(
    schema: dict[str, Any], base_schema: dict[str, Any], handler: str,
) -> dict[str, Any]:
    """Check an override is a usable JSON Schema *for this handler*.

    Three checks, in increasing specificity:
      1. It is an object schema with an object `properties` map.
      2. Every property the base schema marked `required` is still declared --
         an override may relax `required`, but not hide the argument entirely.
      3. Every argument the handler's source actually reads without a default
         is still declared, so a model following this schema can supply it.

    The third check is best-effort by design: if the handler cannot be read
    (a dotted path that no longer imports), the first two still apply rather
    than the whole validation being skipped.
    """
    if not isinstance(schema, dict) or schema.get("type") not in (None, "object"):
        raise SkillAdminError("An input schema override must be an object schema.")
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise SkillAdminError("An input schema override needs a 'properties' object.")
    _required_names(schema)  # shape check

    declared = set(properties) | _TRACE_PROPERTIES
    missing_base = _required_names(base_schema) - declared
    if missing_base:
        raise SkillAdminError(
            f"Override drops required argument(s) {', '.join(sorted(missing_base))} "
            "that the seeded schema declares; the handler still expects them."
        )

    used = _handler_arguments(handler) - _TRACE_PROPERTIES
    missing_used = used - declared
    if missing_used:
        raise SkillAdminError(
            f"Override does not declare {', '.join(sorted(missing_used))}, which the "
            "handler reads without a default; a model following this schema could "
            "not call it."
        )
    return schema


def _handler_arguments(handler: str) -> set[str]:
    """Argument keys a handler reads via subscripting (no default).

    `arguments.get("x")` is excluded: a `.get` has a default and is optional
    by construction, so requiring it in the schema would be wrong.
    """
    import inspect

    if not handler or ":" not in handler:
        return set()
    module_path, _, function_name = handler.partition(":")
    try:
        module = __import__(module_path, fromlist=[function_name])
        source = inspect.getsource(getattr(module, function_name))
    except Exception:  # noqa: BLE001 -- fall back to schema-only validation
        return set()
    return {match.group(1) for match in _REQUIRED_ARGUMENT_READ.finditer(source)}


def _load_skill(graph: Graph, key: str):
    for skill in ToolRepository(graph).load_tools():
        if skill.key == key:
            return skill
    return None


def override_skill_input_schema(arguments: dict[str, Any], graph: Graph) -> dict[str, Any]:
    """Replace one skill's argument schema in the graph. An empty
    `input_schema` clears the override, restoring the seeded base."""
    request = OverrideInputSchemaRequest.model_validate(arguments)
    repository = ToolRepository(graph)
    if request.input_schema:
        rows = graph.run(
            "MATCH (s:Tool {key: $key}) RETURN s.input_schema AS base, s.handler AS handler",
            key=request.key,
        )
        if not rows:
            raise SkillAdminError(f"Unknown skill {request.key!r}")
        import json

        raw_base = rows[0].get("base")
        base = json.loads(raw_base) if raw_base else {}
        validate_input_schema_override(
            request.input_schema, base, rows[0].get("handler") or "",
        )
    try:
        record = repository.override_input_schema(
            request.key, request.input_schema or None,
        )
    except ValueError as exc:
        raise SkillAdminError(str(exc)) from exc
    _reindex(graph)
    return {
        "key": request.key,
        "cleared": not request.input_schema,
        "previous_override": record["previous_override"],
    }


def retire_skill(arguments: dict[str, Any], graph: Graph) -> dict[str, Any]:
    """Stop listing and dispatching one skill, without deleting it."""
    request = SetSkillActiveRequest.model_validate(arguments)
    return _set_active(graph, request.key, False, request.reason)


def reactivate_skill(arguments: dict[str, Any], graph: Graph) -> dict[str, Any]:
    request = SetSkillActiveRequest.model_validate(arguments)
    return _set_active(graph, request.key, True, request.reason)


def _set_active(graph: Graph, key: str, active: bool, reason: str) -> dict[str, Any]:
    try:
        record = ToolRepository(graph).set_active(key, active)
    except ValueError as exc:
        raise SkillAdminError(str(exc)) from exc
    _reindex(graph)
    return {
        **record, "reason": reason,
        "note": (
            "Retired: hidden from discovery and dispatch; its node and telemetry "
            "are intact and it can be reactivated."
            if not active else "Reactivated."
        ),
    }


def _reindex(graph: Graph) -> None:
    """Refresh the semantic skill index and tell the harness to re-fetch.

    Two different consumers of the same change: the index is what
    `search_noesis_tools` ranks against, and the notification is what makes a
    *connected* client's cached tool list catch up mid-session. Both are
    best-effort accelerators, never gates -- the graph is already authoritative.
    """
    try:
        from knowledge_bases.tools.catalog_index import CatalogIndex

        CatalogIndex().index_skills(ToolRepository(graph).load_tools())
    except Exception:  # noqa: BLE001
        pass
    notify_tool_list_changed()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", text.lower().strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unnamed_tool"


def register_tool(arguments: dict[str, Any], graph: Graph) -> dict[str, Any]:
    """Register a new tool directly into the graph so any harness can discover and call it."""
    from knowledge_bases.tools.models import Tool

    request = RegisterToolRequest.model_validate(arguments)
    name = request.name.strip()
    if not name:
        raise SkillAdminError("Tool name cannot be empty.")
    description = request.description.strip()
    if not description:
        raise SkillAdminError("Tool description cannot be empty (needed for search and discovery).")

    kb = request.knowledge_base.strip() or "custom"
    key = request.key.strip()
    if not key:
        key = f"{kb}.{_slugify(name)}"
    elif "." not in key:
        key = f"{kb}.{key}"

    schema = request.input_schema if isinstance(request.input_schema, dict) else {"type": "object", "properties": {}}
    handler = request.handler.strip() or f"harness:{kb}"

    tool = Tool(
        key=key,
        name=name,
        version=request.version or "1.0.0",
        knowledge_base=kb,
        description=description,
        input_schema=schema,
        handler=handler,
        active=True,
    )

    ToolRepository(graph).sync_tool(tool)
    _reindex(graph)

    return {
        "status": "registered",
        "key": tool.key,
        "name": tool.name,
        "knowledge_base": tool.knowledge_base,
        "version": tool.version,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "handler": tool.handler,
        "active": tool.active,
    }

