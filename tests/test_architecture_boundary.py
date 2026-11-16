"""The architecture boundary is enforced, not just documented.

`knowledge_bases/boundary.py` declares which aspects of Noesis may change at
runtime. These tests check the *actual code paths* against that declaration, so
the rule cannot rot into prose: a new self-improvement kind or a new write
command that could reach a structural surface fails here.
"""
from __future__ import annotations

import inspect
import pathlib
import re

import pytest

from knowledge_bases.boundary import (
    IMMUTABLE_SURFACES,
    MUTABLE_SURFACES,
    POLICIES,
    Surface,
    is_mutable_at_runtime,
    require_runtime_mutable,
)
from knowledge_bases.orchestration.commands import WRITE_COMMANDS
from knowledge_bases.tools.graph_edits import GRAPH_EDIT_KINDS
from knowledge_bases.tools.models import SelfImprovementKind
from knowledge_bases.tools.self_improvement_apply import (
    AUTO_APPLIABLE_KINDS,
    TEST_GATED_KINDS,
)


# -- the declaration itself ------------------------------------------------------

def test_every_surface_has_exactly_one_policy():
    assert set(POLICIES) == set(Surface)
    assert IMMUTABLE_SURFACES | MUTABLE_SURFACES == set(Surface)
    assert not (IMMUTABLE_SURFACES & MUTABLE_SURFACES)


def test_require_runtime_mutable_blocks_architecture_and_allows_data():
    with pytest.raises(PermissionError, match="graph_ontology"):
        require_runtime_mutable(Surface.graph_ontology, actor="self_improvement")
    require_runtime_mutable(Surface.tool_description, actor="self_improvement")


@pytest.mark.parametrize("surface", sorted(IMMUTABLE_SURFACES, key=lambda s: s.value))
def test_immutable_surfaces_state_a_rationale_and_an_owner(surface):
    policy = POLICIES[surface]
    assert policy.rationale.strip() and policy.owner.strip()


# -- self-improvement cannot reach architecture ----------------------------------

def test_architecture_kind_is_never_auto_appliable():
    """The single most important invariant of the self-improvement loop."""
    assert SelfImprovementKind.architecture not in AUTO_APPLIABLE_KINDS
    assert SelfImprovementKind.architecture not in GRAPH_EDIT_KINDS
    assert SelfImprovementKind.architecture not in TEST_GATED_KINDS


def test_every_self_improvement_kind_is_classified_against_the_boundary():
    """A newly added kind must be deliberately mapped to a surface here, and
    that surface must be runtime-mutable if the kind auto-applies."""
    kind_surfaces = {
        SelfImprovementKind.tool_wording: Surface.tool_description,
        SelfImprovementKind.routing_capability_wording: Surface.expert_routing_capability,
        SelfImprovementKind.expert_prompt_wording: Surface.expert_prompt,
        SelfImprovementKind.expert_prompt_prefix: Surface.expert_prompt,
        SelfImprovementKind.new_tool: Surface.new_bounded_tool,
        SelfImprovementKind.architecture: Surface.knowledge_base_packages,
    }
    assert set(kind_surfaces) == set(SelfImprovementKind), (
        "a new SelfImprovementKind must be mapped to a boundary Surface"
    )
    for kind in AUTO_APPLIABLE_KINDS:
        assert is_mutable_at_runtime(kind_surfaces[kind]), (
            f"{kind.value} auto-applies but targets a non-mutable surface"
        )


def test_graph_edit_kinds_only_touch_graph_data_surfaces():
    for kind in GRAPH_EDIT_KINDS:
        assert kind in AUTO_APPLIABLE_KINDS
        assert kind != SelfImprovementKind.new_tool, (
            "new_tool is a source edit and must not be applied as a graph edit"
        )


def test_apply_finding_rejects_architecture_before_doing_any_work():
    """Rejection must happen up front, not as a downstream side effect."""
    from knowledge_bases.tools import self_improvement_apply

    source = inspect.getsource(self_improvement_apply.apply_finding)
    guard = source.index("AUTO_APPLIABLE_KINDS")
    for later in ("apply_graph_edit(", "apply_suggestion(", "commit_files("):
        assert guard < source.index(later), (
            f"apply_finding must check AUTO_APPLIABLE_KINDS before {later}"
        )


# -- write commands cannot reach architecture -----------------------------------

def test_write_commands_only_target_registered_knowledge_bases():
    from knowledge_bases.registry import KNOWLEDGE_BASES

    for key, command in WRITE_COMMANDS.items():
        assert command.knowledge_base in KNOWLEDGE_BASES, key


def test_no_write_command_offers_generic_mutation_or_raw_cypher():
    """No write command may *be* an unbounded graph-access primitive.

    Checked against each command's own `operation` identifier, whole-word, so
    a bounded domain operation is not flagged for containing a substring
    (`evaluate_decision_health` is not `eval`) and a description mentioning
    the routing eval gate is not either.
    """
    forbidden = {
        "cypher", "eval", "exec", "sql", "python", "script", "node", "label",
        "relationship", "property", "raw", "arbitrary", "generic",
    }
    for key, command in WRITE_COMMANDS.items():
        words = set(re.split(r"[^a-z0-9]+", command.operation.lower()))
        overlap = words & forbidden
        assert not overlap, (
            f"write command {key!r} names a structural/unbounded primitive: {overlap}"
        )


def test_no_write_command_mutates_the_allowlist_or_the_registry():
    """Structural surfaces have no write-command surface at all."""
    structural = ("write_command", "knowledge_base", "migration", "schema",
                  "registry", "search_type", "permission", "label")
    for key, command in WRITE_COMMANDS.items():
        for token in structural:
            assert not command.operation.startswith(f"create_{token}"), key
            assert not command.operation.startswith(f"add_{token}"), key


# -- the guardrails the boundary claims exist, actually exist -------------------

def test_generated_handler_guardrails_are_present_and_broad():
    from knowledge_bases.tools.source_editor import UNSAFE_NAMES

    for name in ("os", "sys", "subprocess", "eval", "exec", "compile",
                 "__import__", "open", "getattr", "setattr"):
        assert name in UNSAFE_NAMES, f"{name} must stay in UNSAFE_NAMES"


def test_the_export_writes_no_executable_code():
    """Noesis contributes context, never a code path.

    Under Prime, tool access is not gated by per-agent frontmatter -- its kernel
    is a control environment, not a sandbox -- so the boundary that matters is
    that Noesis's own export introduces no capability of its own. It writes
    markdown and settings; reaching the graph is Prime's own MCP client calling
    bounded tools.
    """
    import json as _json

    from knowledge_bases.orchestration.experts_catalog import (
        ORCHESTRATOR_KEY,
        bootstrap_experts,
    )
    from knowledge_bases.orchestration.prime_export import (
        mcp_setup_command,
        render_agents_md,
        render_settings,
        render_skill,
    )

    orchestrator = next(e for e in bootstrap_experts() if e.key == ORCHESTRATOR_KEY)
    written = "\n".join([
        render_agents_md(orchestrator),
        render_skill("project_management", "Execution", "d",
                     [{"operation": "x", "description": "y"}]),
        _json.dumps(render_settings()),
    ])

    # No executable artifact is emitted at all.
    assert "export default" not in written
    for forbidden in (
        "child_process", "execSync", "spawnSync", "node:fs", "writeFileSync",
        "eval(", "createLocalBashOperations", "subprocess",
    ):
        assert forbidden not in written, f"export introduced {forbidden}"

    # The one command a user runs is an ordinary MCP registration, and the
    # export returns it rather than writing a user-level config itself.
    command = mcp_setup_command(pathlib.Path("/repo"))
    assert command[:3] == ["prime-agent", "mcp", "add"]


def test_the_generated_context_export_is_an_immutable_surface():
    """It is code-shaped and review-gated: a runtime path that could rewrite
    what a workspace loads at startup would bypass every other gate."""
    assert Surface.generated_context_export in IMMUTABLE_SURFACES


def test_orchestrator_prompt_states_its_prohibitions_as_non_overridable():
    """A1 lets supplemental text be prepended to this prompt, so the
    prohibitions must explicitly refuse to be relaxed by it."""
    from knowledge_bases.orchestration.experts_catalog import (
        ORCHESTRATOR_KEY,
        bootstrap_experts,
    )

    prompt = next(
        e for e in bootstrap_experts() if e.key == ORCHESTRATOR_KEY
    ).system_prompt.lower()

    assert "cannot be relaxed" in prompt
    assert "raw cypher" in prompt
    for phrase in ("supplemental guidance", "session context", "user instruction"):
        assert phrase in prompt


# -- composition and the methodology profile are architecture -------------------

def test_the_composition_surfaces_are_immutable():
    """Composition is the one surface that contains all the others: installing
    a package contributes labels, relationships, search types, tools, experts,
    and versionable kinds at once."""
    for surface in (
        Surface.product_profile,
        Surface.package_composition,
        Surface.versioned_kind_registry,
        Surface.workflow_profile,
    ):
        assert surface in IMMUTABLE_SURFACES
        with pytest.raises(PermissionError):
            require_runtime_mutable(surface, actor="self_improvement")


def test_no_write_command_installs_a_package_or_contributes_a_kind():
    """The bounded write surface must not reach composition at all."""
    forbidden = (
        "package", "ontology_package", "kind", "phase", "gate", "profile",
        "expert_prompt_base",
    )
    for key, command in WRITE_COMMANDS.items():
        words = set(re.split(r"[^a-z0-9]+", command.operation.lower()))
        assert not words & set(forbidden), (
            f"write command {key!r} reaches a composition surface"
        )


def test_package_declarations_are_frozen_dataclasses():
    """A mutable declaration could be edited in-process, which would make the
    "composition is code" guarantee unenforceable at runtime."""
    import dataclasses

    from knowledge_bases.base import (
        KindRequirement,
        KnowledgeBase,
        OntologyPackage,
        PhaseGate,
        SearchType,
        VersionedKind,
        WorkflowPhase,
    )

    for contract in (
        OntologyPackage, KnowledgeBase, SearchType, VersionedKind,
        KindRequirement, WorkflowPhase, PhaseGate,
    ):
        assert dataclasses.is_dataclass(contract)
        assert contract.__dataclass_params__.frozen, f"{contract.__name__} is mutable"


def test_the_contract_module_stays_a_leaf():
    """Every package declaration imports `base`, so a dependency from `base`
    back into the codebase would make composition circular -- which is what
    forced `registry` to exec package files standalone in the first place."""
    from knowledge_bases import base

    source = inspect.getsource(base)
    for line in source.splitlines():
        if line.startswith(("import ", "from ")):
            assert "knowledge_bases" not in line and "graph" not in line, (
                f"knowledge_bases/base.py must import nothing from Noesis: {line}"
            )


def test_generated_tools_cannot_be_written_outside_the_product_owned_package():
    """`new_bounded_tool` is runtime-mutable, but only inside the one package
    the product owns in its own source tree. Every other installed package is a
    separate distribution in `site-packages`: an appended handler there is
    unwritable at worst, and destroyed by the next upgrade at best -- leaving a
    registered `Tool` pointing at a handler that no longer exists."""
    from knowledge_bases.product import PROFILE
    from knowledge_bases.tools.source_editor import (
        SourceEditError,
        _require_generated_tool_target,
    )

    _require_generated_tool_target(PROFILE.generated_tool_target)

    for key in PROFILE.packages:
        if key == PROFILE.generated_tool_target:
            continue
        with pytest.raises(SourceEditError, match="product-owned"):
            _require_generated_tool_target(key)


def test_the_product_profile_is_declared_in_one_place():
    """If the engine kept its own package list beside the profile's, the two
    would drift and the ontology would change without a profile edit."""
    from knowledge_bases.product import PROFILE
    from knowledge_bases.registry import DEFAULT_PACKAGES

    assert DEFAULT_PACKAGES is PROFILE.packages
