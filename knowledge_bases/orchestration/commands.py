"""Bounded write commands exposed through Noesis orchestration.

Dispatch for every command below resolves to the *same* graph-registered
`Tool` handler its read-side MCP tool uses (see
`knowledge_bases.mcp.dispatch_tool` and `knowledge_bases/tools/catalog.py`);
this module only adds the propose/confirm mutation-governance wrapper
(validation + the allowlist itself), not a second copy of dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from graph import Graph
from knowledge_bases.mcp import dispatch_tool
from knowledge_bases.registry import composed_write_commands
from knowledge_bases.orchestration.expert_admin import (
    CreateExpertRequest,
    RetireExpertRequest,
    ReviseExpertRequest,
    SetExpertPromptPrefixRequest,
)
from knowledge_bases.orchestration.lifecycle import (
    DeprecateEntityRequest,
    ReactivateEntityRequest,
    preview_deprecation,
)
from knowledge_bases.tools.tool_admin import (
    OverrideInputSchemaRequest,
    RegisterToolRequest,
    SetSkillActiveRequest,
)


from knowledge_bases.orchestration.write_dispatch import (  # noqa: E402
    Dispatcher,
    Previewer,
    Validator,
    WriteCommand,
    domain_dispatch as _domain_dispatch,
    required as _required,
)


def _pydantic(model: Any) -> Validator:
    def validate(arguments: dict[str, Any]) -> dict[str, Any]:
        trusted_arguments = dict(arguments)
        if "actor" in model.model_fields:
            trusted_arguments["actor"] = "unknown"
        return model.model_validate(trusted_arguments).model_dump(mode="json")

    return validate


def _preview_deprecation(arguments: dict[str, Any], graph: Graph) -> dict[str, Any]:
    return preview_deprecation(graph, DeprecateEntityRequest.model_validate(arguments))


WRITE_COMMANDS = {
    command.key: command
    for command in (
        WriteCommand(
            "orchestration",
            "deprecate_entity",
            "Tag one node, or a cascaded block of nodes, as :Deprecated "
            "(see docs/architecture/lifecycle.md); nodes are never deleted.",
            _pydantic(DeprecateEntityRequest),
            _domain_dispatch("orchestration", "deprecate_entity"),
            preview=_preview_deprecation,
        ),
        WriteCommand(
            "orchestration",
            "reactivate_entity",
            "Remove the :Deprecated tag from one node or a cascaded block of nodes.",
            _pydantic(ReactivateEntityRequest),
            _domain_dispatch("orchestration", "reactivate_entity"),
        ),
        # -- expert roster (see orchestration/expert_admin.py). Data, not
        # architecture: an expert is composed of already-registered skills, so
        # these add no label, relationship, or migration.
        WriteCommand(
            "orchestration",
            "create_expert",
            "Define a new graph-defined expert from existing knowledge bases' "
            "tools. Gated by the capability-pollution guard and the golden "
            "routing eval; delegable immediately, with no harness restart.",
            _pydantic(CreateExpertRequest),
            _domain_dispatch("orchestration", "create_expert"),
        ),
        WriteCommand(
            "orchestration",
            "revise_expert",
            "Change an existing expert's prompt, capabilities, priority, or "
            "allowed skills, as a new versioned ExpertRevision.",
            _pydantic(ReviseExpertRequest),
            _domain_dispatch("orchestration", "revise_expert"),
        ),
        WriteCommand(
            "orchestration",
            "retire_expert",
            "Deactivate an expert so routing stops selecting it. Never deletes: "
            "its revisions and telemetry stay intact and it can be revised back.",
            _pydantic(RetireExpertRequest),
            _domain_dispatch("orchestration", "retire_expert"),
        ),
        # -- skill lifecycle (see skills/tool_admin.py). Graph overrides on
        # top of the seeded catalog; the `handler` is never touched, so these
        # cannot repoint a skill at different code.
        WriteCommand(
            "tools",
            "override_skill_input_schema",
            "Correct one skill's argument schema in the graph, leaving the seeded "
            "base intact. Validated against the handler's real required arguments; "
            "pass an empty schema to clear the override.",
            _pydantic(OverrideInputSchemaRequest),
            _domain_dispatch("tools", "override_skill_input_schema"),
        ),
        WriteCommand(
            "tools",
            "retire_skill",
            "Stop listing and dispatching one skill without deleting it; its node, "
            "history, and telemetry stay intact.",
            _pydantic(SetSkillActiveRequest),
            _domain_dispatch("tools", "retire_skill"),
        ),
        WriteCommand(
            "tools",
            "reactivate_skill",
            "Return a retired skill to discovery and dispatch.",
            _pydantic(SetSkillActiveRequest),
            _domain_dispatch("tools", "reactivate_skill"),
        ),
        WriteCommand(
            "orchestration",
            "set_expert_prompt_prefix",
            "Set the bounded supplemental context rendered ahead of an expert's "
            "immutable base prompt (max 4 entries / 2000 characters). Pass an "
            "empty list to clear it.",
            _pydantic(SetExpertPromptPrefixRequest),
            _domain_dispatch("orchestration", "set_expert_prompt_prefix"),
        ),
        WriteCommand(
            "orchestration",
            "register_tool",
            "Register a new tool or capability directly into the graph.",
            _pydantic(RegisterToolRequest),
            _domain_dispatch("orchestration", "register_tool"),
        ),
        # Packages contribute their own bounded writes (`write_commands_seed`),
        # so uninstalling a package removes its writes with it rather than
        # leaving an allowlisted command whose handler no longer ships.
        *composed_write_commands(),
    )
}


def list_write_commands() -> list[dict[str, str]]:
    return [
        {
            "command": command.key,
            "knowledge_base": command.knowledge_base,
            "operation": command.operation,
            "description": command.description,
        }
        for command in WRITE_COMMANDS.values()
    ]


def command_for(key: str) -> WriteCommand:
    try:
        return WRITE_COMMANDS[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported Noesis write command: {key}") from exc
