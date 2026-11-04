"""What Noesis may change about itself at runtime, and what it may not.

Noesis edits its own behaviour in two places -- the self-improvement loop
(`knowledge_bases/tools/self_improvement_apply.py`) and the bounded write
commands a user drives through propose/confirm/execute
(`knowledge_bases/orchestration/commands.py`). Both need one shared,
*enforceable* answer to "is this change allowed to happen without a human?",
because the interesting failure is not a bad wording change -- it is a change
to the shape of the system that no gate was designed to catch.

The rule, stated once:

    Anything that changes the *architecture* requires a human writing code and
    a reviewed commit. Everything else is graph data and may change at
    runtime, gated and reversible.

"Architecture" is not a vibe; it is this list (`IMMUTABLE_SURFACES`):

- **Graph ontology** -- the labels and relationships each knowledge base owns
  (`knowledge_bases/registry.py`), and the constraints/indexes over them
  (`migrations.py`). A new label or relationship is a schema change.
- **Knowledge base packages** -- adding `knowledge_bases/<name>/` with its
  models, tools, and registry entry.
- **The package composition API** -- the `OntologyPackage` contract
  (`knowledge_bases/base.py`) and the installed package list
  (`registry.DEFAULT_PACKAGES`). A product *is* its set of packages, so a
  runtime path that could install one could add labels, relationships, search
  types, tools, experts, and versionable kinds in a single step -- every other
  immutable surface at once.
- **The versioned-kind registry** -- which domain nouns the methodology engine
  may version and what a changeset must satisfy to create one
  (each package's `kinds_seed`, composed in
  `knowledge_bases/methodology/kinds.py`). These are the creation-traceability
  invariants; a runtime path that could relax one could baseline untraceable
  work.
- **The workflow profile** -- the phase sequence, personas, and phase gates
  (`knowledge_bases/methodology/phases.py`). The gates are what
  `approve_baseline` refuses to pass; weakening one at runtime would approve
  work no invariant held for.
- **Routing mathematics** -- the fused scoring weights and thresholds in
  `knowledge_bases/orchestration/routing.py`. Wording that routing *matches
  against* is graph data; how the match is computed is not.
- **The write-command allowlist itself** -- `WRITE_COMMANDS`. A runtime path
  that could add a write command could add one that bypasses this boundary,
  so the set of legal writes is code.
- **Search type registry** -- `hybrid_search`'s `SEARCH_TYPES`, which decides
  what may be searched and in which family.
- **Memory ontology mapping** -- which labels and relationship types an
  extracted triplet may use (`knowledge_bases/ontology.py`). It is derived
  from the registry, so declaring ontology there is the only way to widen it;
  a runtime path that could widen it could turn extracted prose into schema.
- **Generated-code guardrails** -- `source_editor.UNSAFE_NAMES` and the
  one-function-with-a-fixed-signature validation around LLM-authored handlers.
- **The generated Prime context export** -- what
  `knowledge_bases/orchestration/prime_export.py` writes into a workspace.
  Under Prime, tool *access* is no longer enforced by per-agent frontmatter:
  Prime's kernel is a control environment, not a sandbox, so the enforced
  boundary is the bounded-tool surface itself (`WRITE_COMMANDS`, the
  `call_noesis_tool` expert-scope check, and the absence of any raw-Cypher
  path), not a permission line in a file.

And what *is* mutable at runtime (`MUTABLE_SURFACES`), each with its own gate
and its own revert: skill descriptions and input schemas, skill activation,
expert prompts (append and prepend overlays), expert routing capabilities,
expert creation/revision/retirement, brand-new bounded skills (statically
validated, test-gated, single git commit), durable session memories (prose
plus edges to entities that already exist, superseded rather than
overwritten), and all domain knowledge.

This module is deliberately declarative -- it holds no logic beyond the two
frozensets and a lookup, so it reads as documentation. It is enforced by
`tests/test_architecture_boundary.py`, which asserts that no
`SelfImprovementKind` and no `WriteCommand` can reach an immutable surface.
Before this existed the same rule lived only in prose
(`docs/architecture/self_improvement.md`), where nothing could check it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Surface(str, Enum):
    """One changeable aspect of Noesis itself."""

    # --- immutable: human writes code, commit is reviewed -----------------
    graph_ontology = "graph_ontology"
    product_profile = "product_profile"
    package_composition = "package_composition"
    versioned_kind_registry = "versioned_kind_registry"
    workflow_profile = "workflow_profile"
    schema_migrations = "schema_migrations"
    knowledge_base_packages = "knowledge_base_packages"
    routing_mathematics = "routing_mathematics"
    write_command_allowlist = "write_command_allowlist"
    search_type_registry = "search_type_registry"
    memory_ontology_mapping = "memory_ontology_mapping"
    generated_code_guardrails = "generated_code_guardrails"
    generated_context_export = "generated_context_export"

    # --- mutable at runtime, gated and reversible -------------------------
    tool_description = "tool_description"
    tool_input_schema = "tool_input_schema"
    tool_activation = "tool_activation"
    agent_skill_content = "agent_skill_content"
    new_bounded_tool = "new_bounded_tool"
    expert_prompt = "expert_prompt"
    expert_routing_capability = "expert_routing_capability"
    expert_lifecycle = "expert_lifecycle"
    domain_knowledge = "domain_knowledge"
    session_memory = "session_memory"


@dataclass(frozen=True)
class SurfacePolicy:
    surface: Surface
    mutable_at_runtime: bool
    rationale: str
    # Where the authoritative definition lives, for a human following up.
    owner: str


_POLICIES: tuple[SurfacePolicy, ...] = (
    SurfacePolicy(
        Surface.graph_ontology, False,
        "Labels/relationships are the contract every knowledge base and every "
        "bounded tool is written against; adding one is a schema change.",
        "knowledge_bases/registry.py",
    ),
    SurfacePolicy(
        Surface.package_composition, False,
        "A product is its set of installed ontology packages: installing one "
        "contributes labels, relationships, search types, tools, experts, and "
        "versionable kinds at once, so composition is every other immutable "
        "surface in a single step.",
        "knowledge_bases/base.py + registry.DEFAULT_PACKAGES",
    ),
    SurfacePolicy(
        Surface.product_profile, False,
        "The profile chooses which ontologies are installed, what the product "
        "registers as, and where generated tools may be written -- a runtime "
        "path that could edit it could install an ontology or redirect "
        "generated code into a package the product does not own.",
        "knowledge_bases/product.py",
    ),
    SurfacePolicy(
        Surface.versioned_kind_registry, False,
        "Which nouns may be versioned, and the creation-traceability rules for "
        "each, are the methodology's invariants; relaxing one at runtime would "
        "let untraceable work be baselined.",
        "knowledge_bases/<owner>/kinds.py via methodology/kinds.py",
    ),
    SurfacePolicy(
        Surface.workflow_profile, False,
        "Phase gates are what approve_baseline refuses to pass. Personas and "
        "questions are wording, but they ship with the gates in one "
        "declaration, so the whole profile is code.",
        "knowledge_bases/methodology/phases.py",
    ),
    SurfacePolicy(
        Surface.schema_migrations, False,
        "Constraints and indexes are ordered, tracked, and irreversible in "
        "effect; they are applied by a human-run migration.",
        "migrations.py",
    ),
    SurfacePolicy(
        Surface.knowledge_base_packages, False,
        "A new knowledge base is new models, tools, labels, and a registry "
        "entry at once -- the 'paradigm' case self-improvement routes to a human.",
        "knowledge_bases/<name>/",
    ),
    SurfacePolicy(
        Surface.routing_mathematics, False,
        "Wording that routing matches against is graph data; how the match is "
        "scored is code, so a bad self-edit cannot silently reshape routing.",
        "knowledge_bases/orchestration/routing.py",
    ),
    SurfacePolicy(
        Surface.write_command_allowlist, False,
        "A runtime path that could add a write command could add one that "
        "bypasses this boundary; the set of legal writes must be code.",
        "knowledge_bases/orchestration/commands.py",
    ),
    SurfacePolicy(
        Surface.memory_ontology_mapping, False,
        "Which labels and relationship types a memory may write is derived from "
        "the registry; a runtime path that could widen it could mint ontology "
        "from extracted prose, which is the hole this check exists to close.",
        "knowledge_bases/ontology.py",
    ),
    SurfacePolicy(
        Surface.search_type_registry, False,
        "Decides what is searchable and in which family; widening it widens "
        "what any expert can read.",
        "knowledge_bases/orchestration/hybrid_search.py",
    ),
    SurfacePolicy(
        Surface.generated_code_guardrails, False,
        "These are what make LLM-authored handlers safe to commit at all; "
        "they cannot be relaxed by the loop they constrain.",
        "knowledge_bases/tools/source_editor.py",
    ),
    SurfacePolicy(
        Surface.generated_context_export, False,
        "What Noesis writes into a Prime workspace. Under Prime the enforced "
        "boundary is the bounded-tool surface itself, not a per-agent "
        "permission line -- its kernel is a control environment, not a sandbox.",
        "knowledge_bases/orchestration/prime_export.py",
    ),
    SurfacePolicy(
        Surface.tool_description, True,
        "Description drives tool discovery; overridden in the graph, base "
        "seed untouched, revert clears the override.",
        "Tool.description_override",
    ),
    SurfacePolicy(
        Surface.tool_input_schema, True,
        "Overridden in the graph and validated as a JSON Schema that still "
        "satisfies the handler's required arguments.",
        "Tool.input_schema_override",
    ),
    SurfacePolicy(
        Surface.tool_activation, True,
        "Retiring a skill hides it from discovery and dispatch without "
        "deleting it or its telemetry.",
        "Tool.active",
    ),
    SurfacePolicy(
        Surface.agent_skill_content, True,
        "An AgentSkill is instructions a harness discloses to a model, not a "
        "callable operation: it has no handler and is never dispatched, so "
        "rewording one cannot widen what any expert can do. The body is "
        "rendered from the live Tool catalog on each lookup, and the harness "
        "constraints (name shape, description limit) are validated on write.",
        "AgentSkill.body / knowledge_bases/orchestration/agent_skills.py",
    ),
    SurfacePolicy(
        Surface.new_bounded_tool, True,
        "One statically validated handler with a fixed signature, appended to "
        "an existing module, test-gated, in an isolated revertible commit.",
        "knowledge_bases/tools/source_editor.py",
    ),
    SurfacePolicy(
        Surface.expert_prompt, True,
        "Appended and prepended overlays on top of an immutable base prompt, "
        "each a versioned ExpertRevision; revert removes one item.",
        "KnowledgeBaseExpert.prompt_overlays / prompt_prefix_overlays",
    ),
    SurfacePolicy(
        Surface.expert_routing_capability, True,
        "Capability phrases are graph data, gated by the pollution guard and "
        "the golden routing eval before they are written.",
        "KnowledgeBaseExpert.capability_overlays",
    ),
    SurfacePolicy(
        Surface.expert_lifecycle, True,
        "An expert composed of existing knowledge bases' skills adds no label "
        "and no relationship, so it is data, not architecture.",
        "KnowledgeBaseExpert",
    ),
    SurfacePolicy(
        Surface.domain_knowledge, True,
        "The point of the system: business knowledge changes through bounded "
        "write commands under propose/confirm/execute.",
        "knowledge_bases/orchestration/commands.py WRITE_COMMANDS",
    ),
    SurfacePolicy(
        Surface.session_memory, True,
        "Prose plus edges to entities that already exist, using only declared "
        "ontology -- it adds no label and no relationship type, so it is data. "
        "Never destructive: a correction supersedes rather than overwrites.",
        "SessionMemory / knowledge_bases/orchestration/memory.py",
    ),
)

POLICIES: dict[Surface, SurfacePolicy] = {policy.surface: policy for policy in _POLICIES}

IMMUTABLE_SURFACES = frozenset(
    policy.surface for policy in _POLICIES if not policy.mutable_at_runtime
)
MUTABLE_SURFACES = frozenset(
    policy.surface for policy in _POLICIES if policy.mutable_at_runtime
)


def is_mutable_at_runtime(surface: Surface) -> bool:
    return POLICIES[surface].mutable_at_runtime


def require_runtime_mutable(surface: Surface, actor: str) -> None:
    """Raise for an attempt to change an architecture surface at runtime.

    A guard for any future runtime path that touches one of these surfaces:
    failing loudly here is the difference between "this change needs review"
    and a silently applied structural edit.
    """
    if not is_mutable_at_runtime(surface):
        policy = POLICIES[surface]
        raise PermissionError(
            f"{actor} may not change {surface.value!r} at runtime: "
            f"{policy.rationale} Authoritative definition: {policy.owner}."
        )
