"""Shared builders for a package's bounded write commands.

Extracted from `orchestration/commands.py` so an ontology package can declare its
own writes (`write_commands_seed`) without importing that module's private
helpers. Importing `_required`/`_domain_dispatch` across a future distribution
boundary would be a package depending on an orchestrator internal -- and a
circular one, since `commands.py` composes the packages' contributions.

This module deliberately holds no command definitions. It is machinery, so it
stays in the control plane while the commands themselves live with the ontology
they mutate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from graph import Graph

Validator = Callable[[dict[str, Any]], dict[str, Any]]
Dispatcher = Callable[[dict[str, Any], "Graph"], Any]
Previewer = Callable[[dict[str, Any], "Graph"], dict[str, Any]]


@dataclass(frozen=True)
class WriteCommand:
    """One bounded, confirmable graph mutation.

    Lives here rather than in `commands.py` so a package can declare its own
    commands without importing the module that composes them -- which would be a
    cycle, and across a future distribution boundary an unresolvable one.
    """

    knowledge_base: str
    operation: str
    description: str
    validate: Validator
    dispatch: Dispatcher
    preview: Previewer | None = None

    @property
    def key(self) -> str:
        return f"{self.knowledge_base}.{self.operation}"


def required(*names: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Validator rejecting a command whose required arguments are blank.

    Blank, not merely absent: a write dispatched with `{"alias": ""}` would
    otherwise create an unnamed entity that no query finds.
    """

    def validate(arguments: dict[str, Any]) -> dict[str, Any]:
        missing = [name for name in names if not str(arguments.get(name, "")).strip()]
        if missing:
            raise ValueError(
                f"Missing required command arguments: {', '.join(missing)}"
            )
        return arguments

    return validate


def domain_dispatch(
    knowledge_base: str, operation: str
) -> Callable[[dict[str, Any], Graph], Any]:
    """Route a confirmed write to a knowledge base's own bounded tool.

    The write path never reimplements a domain operation; it calls the same tool
    the read path would, so there is one handler per operation rather than one per
    entry point.
    """

    def dispatch(arguments: dict[str, Any], graph: Graph) -> Any:
        from knowledge_bases.mcp import dispatch_tool

        return dispatch_tool(graph, knowledge_base, operation, arguments)

    return dispatch

def pydantic(model: Any) -> Validator:
    """Validate arguments against a Pydantic model the OWNING package supplies.

    Lives here, with the other builders, because validating against a model is
    machinery; WHICH model is knowledge the owning package has. Keeping the helper in
    core and the model in the package is what lets a package validate its own writes
    without the engine importing its models.

    `actor` is always overwritten rather than trusted from the caller, so a write
    cannot claim to have been made by somebody else.
    """

    def validate(arguments: dict[str, Any]) -> dict[str, Any]:
        trusted_arguments = dict(arguments)
        if "actor" in model.model_fields:
            trusted_arguments["actor"] = "unknown"
        return model.model_validate(trusted_arguments).model_dump(mode="json")

    return validate
