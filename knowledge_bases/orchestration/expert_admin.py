"""User-driven expert administration: create, revise, retire, and frame.

Experts are graph data (`KnowledgeBaseExpert`, see
`knowledge_bases/orchestration/experts_catalog.py` for the seed), so changing
the roster does not require a code change -- provided the change stays on the
data side of the architecture boundary (`knowledge_bases/boundary.py`). It does
when an expert is composed of *existing* knowledge bases' already-registered
skills: no new label, no new relationship, no new package, no migration. It
does not the moment an expert would need a capability nothing implements yet,
which is why `allowed_tools` is resolved against the live catalog and an
unresolvable pattern is refused rather than created as an expert that can do
nothing.

Every operation here runs through the ordinary `WriteProposal`
propose -> confirm -> execute flow (`commands.py`, `write_service.py`), so a
human sees the exact change first -- unlike self-improvement's wording edits,
which auto-apply because they are measured and auto-reverted.

Two gates protect the roster, reused verbatim from the self-improvement path
(`knowledge_bases/tools/graph_edits.py`) so there is one definition of
"would this wreck routing?":

1. **Pollution guard** -- a new or revised expert's capability vocabulary must
   read more like itself than like any existing expert, or it will pull their
   questions away.
2. **Golden routing eval** -- top-1 accuracy on
   `knowledge_bases/orchestration/routing_eval.jsonl` must not drop. Adding an
   expert perturbs routing for *every* question, not just its own, so this is
   the gate that matters most here.

Nothing in this module deletes: retiring an expert sets `active = false`
(`graph.py`'s never-delete policy), and every write is snapshotted as an
`ExpertRevision`, so the roster's history is complete and a retirement is
reversible.

Because harness workers are generic and fetch their expert's definition at
delegation time (see `prime_export.py`), an expert created here is
delegable immediately -- no file export and no harness restart.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from graph import Graph
from knowledge_bases.orchestration.experts_catalog import (
    EXPERT_LOOP_CONTRACT,
    ORCHESTRATOR_KEY,
)
from knowledge_bases.orchestration.models import (
    ExpertDefinition,
    PromptPrefixError,
    validate_prompt_prefix_overlays,
)
from knowledge_bases.orchestration.repository import RoutingRepository, _skill_allowed
from knowledge_bases.tools.catalog_notify import notify_tool_list_changed
from knowledge_bases.tools.repository import ToolRepository


class ExpertAdminError(ValueError):
    """A roster change that is refused: unresolvable skills, a protected
    expert, or a change that would damage routing."""


# The orchestrator is the primary agent's own definition, not a routable
# expert; letting a runtime write reshape it would let a runtime write reshape
# the control loop itself.
PROTECTED_EXPERT_KEYS = frozenset({ORCHESTRATOR_KEY})


class CreateExpertRequest(BaseModel):
    key: str
    name: str
    description: str
    system_prompt: str
    capabilities: list[str] = Field(default_factory=list)
    entry_tools: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    priority: int = 100
    routable: bool = True


class ReviseExpertRequest(BaseModel):
    key: str
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    capabilities: list[str] | None = None
    entry_tools: list[str] | None = None
    allowed_tools: list[str] | None = None
    priority: int | None = None


class RetireExpertRequest(BaseModel):
    key: str
    reason: str = ""


class SetExpertPromptPrefixRequest(BaseModel):
    """Replace an expert's supplemental prompt prefix wholesale.

    Set-not-append on purpose: the slot is small and budget-bounded, so a user
    managing it needs to see and state the whole thing rather than discovering
    they have hit a limit built from forgotten increments. Pass an empty list
    to clear it.
    """

    key: str
    entries: list[str] = Field(default_factory=list)


def _require_not_protected(key: str, action: str) -> None:
    if key in PROTECTED_EXPERT_KEYS:
        raise ExpertAdminError(
            f"{key!r} is the primary agent's own definition and cannot be {action}. "
            "Its base prompt lives in experts_catalog.py; refine it with a prompt "
            "overlay or prefix instead."
        )


def _resolve_skills(graph: Graph, patterns: list[str], entry_tools: list[str]) -> list[str]:
    """Check every pattern resolves to at least one active skill, and that
    entry skills are within the allowed set. Returns the resolved keys."""
    if not patterns:
        raise ExpertAdminError(
            "An expert with no allowed_tools could not do anything; name at "
            "least one skill key or '<knowledge_base>.*' glob."
        )
    active = [skill.key for skill in ToolRepository(graph).load_tools()]
    unresolved = [
        pattern for pattern in patterns
        if not any(_skill_allowed(key, [pattern]) for key in active)
    ]
    if unresolved:
        raise ExpertAdminError(
            f"No active skill matches {', '.join(unresolved)}. An expert can only "
            "be composed of capabilities that already exist -- adding a new one is "
            "a code change (see knowledge_bases/boundary.py)."
        )
    resolved = [key for key in active if _skill_allowed(key, patterns)]
    outside = [key for key in entry_tools if key not in resolved]
    if outside:
        raise ExpertAdminError(
            f"entry_tools {', '.join(outside)} are not covered by allowed_tools; "
            "an expert cannot start with a skill it may not call."
        )
    return resolved


def _check_routing_impact(
    graph: Graph, candidate: ExpertDefinition, existing: list[ExpertDefinition],
) -> dict[str, Any]:
    """Both routing gates, for a candidate roster containing `candidate`.

    Imported lazily so this module does not pull the self-improvement package
    (and its LLM/index dependencies) into every orchestration import.
    """
    from knowledge_bases.tools.graph_edits import (
        GraphEditError,
        check_capability_pollution,
        check_routing_eval,
    )

    others = [e for e in existing if e.key != candidate.key]
    report: dict[str, Any] = {}
    if candidate.routable and candidate.capabilities:
        # Pollution is judged against the *candidate's* own vocabulary, so the
        # candidate must be in the roster being compared.
        try:
            report["pollution_check"] = check_capability_pollution(
                " ".join(candidate.capabilities), candidate.key, [candidate, *others],
            )
        except GraphEditError as exc:
            raise ExpertAdminError(str(exc)) from exc
    try:
        report.update(check_routing_eval(others, [candidate, *others]))
    except GraphEditError as exc:
        raise ExpertAdminError(str(exc)) from exc
    return report


def create_expert(arguments: dict[str, Any], graph: Graph) -> dict[str, Any]:
    request = CreateExpertRequest.model_validate(arguments)
    _require_not_protected(request.key, "created")
    repository = RoutingRepository(graph)
    if repository.get_expert(request.key) is not None:
        raise ExpertAdminError(
            f"Expert {request.key!r} already exists; use revise_expert to change it."
        )
    resolved = _resolve_skills(graph, request.allowed_tools, request.entry_tools)
    if request.routable and not request.capabilities:
        raise ExpertAdminError(
            "A routable expert needs at least one capability phrase; that is what "
            "routing matches a question against."
        )
    candidate = ExpertDefinition(
        key=request.key,
        name=request.name,
        description=request.description,
        capabilities=request.capabilities,
        priority=request.priority,
        # Same shape the seed catalog builds, so a graph-created expert and a
        # seeded one are indistinguishable to a worker.
        system_prompt=(
            f"You are the Noesis {request.name} expert.\n\n"
            f"{request.system_prompt.strip()}\n\n{EXPERT_LOOP_CONTRACT}"
        ),
        entry_tools=request.entry_tools,
        allowed_tools=request.allowed_tools,
        routable=request.routable,
    )
    existing = repository.load_experts(include_unroutable=True)
    report = _check_routing_impact(graph, candidate, existing)
    version = repository.revise_expert(candidate, reason="user:create_expert")
    _reindex_experts(graph)
    return {
        "created": request.key, "version": version,
        "resolved_skills": resolved, "routing": report,
        "delegable": "immediately -- workers fetch expert definitions at delegation time",
    }


def revise_expert(arguments: dict[str, Any], graph: Graph) -> dict[str, Any]:
    request = ReviseExpertRequest.model_validate(arguments)
    _require_not_protected(request.key, "revised this way")
    repository = RoutingRepository(graph)
    expert = repository.get_expert(request.key)
    if expert is None:
        raise ExpertAdminError(f"Unknown expert {request.key!r}")

    updates = {
        field: value
        for field, value in request.model_dump(exclude={"key"}).items()
        if value is not None
    }
    if not updates:
        raise ExpertAdminError("Nothing to revise.")
    candidate = expert.model_copy(update=updates)
    if "system_prompt" in updates:
        # Keep the loop contract attached; it is the shared discipline every
        # expert is held to, not per-expert prose.
        prompt = candidate.system_prompt.strip()
        if EXPERT_LOOP_CONTRACT not in prompt:
            prompt = f"{prompt}\n\n{EXPERT_LOOP_CONTRACT}"
        candidate = candidate.model_copy(update={"system_prompt": prompt})

    resolved = _resolve_skills(graph, candidate.allowed_tools, candidate.entry_tools)
    existing = repository.load_experts(include_unroutable=True)
    report = _check_routing_impact(graph, candidate, existing)
    version = repository.revise_expert(candidate, reason="user:revise_expert")
    _reindex_experts(graph)
    return {
        "revised": request.key, "version": version, "changed": sorted(updates),
        "resolved_skills": resolved, "routing": report,
    }


def retire_expert(arguments: dict[str, Any], graph: Graph) -> dict[str, Any]:
    request = RetireExpertRequest.model_validate(arguments)
    _require_not_protected(request.key, "retired")
    repository = RoutingRepository(graph)
    expert = repository.get_expert(request.key)
    if expert is None:
        raise ExpertAdminError(f"Unknown expert {request.key!r}")
    if not expert.active:
        return {"retired": request.key, "already_inactive": True}
    graph.run(
        "MATCH (e:KnowledgeBaseExpert {key: $key}) "
        "SET e.active = false, e.retired_at = datetime(), e.retired_reason = $reason",
        key=request.key, reason=request.reason,
    )
    _reindex_experts(graph)
    return {
        "retired": request.key,
        "note": "Deactivated, not deleted; its revisions and telemetry are intact.",
    }


def set_expert_prompt_prefix(arguments: dict[str, Any], graph: Graph) -> dict[str, Any]:
    """Set the bounded supplemental context rendered ahead of an expert's
    base prompt. Allowed for the orchestrator too -- framing the primary agent
    is the main reason this exists."""
    request = SetExpertPromptPrefixRequest.model_validate(arguments)
    repository = RoutingRepository(graph)
    expert = repository.get_expert(request.key)
    if expert is None:
        raise ExpertAdminError(f"Unknown expert {request.key!r}")
    try:
        entries = validate_prompt_prefix_overlays(request.entries)
    except PromptPrefixError as exc:
        raise ExpertAdminError(str(exc)) from exc
    version = repository.revise_expert(
        expert.model_copy(update={"prompt_prefix_overlays": entries}),
        reason="user:set_expert_prompt_prefix",
    )
    return {
        "key": request.key, "version": version, "entries": entries,
        "characters": sum(len(entry) for entry in entries),
    }


def _reindex_experts(graph: Graph) -> None:
    """Refresh the semantic expert index and tell the harness to re-fetch.

    Routing scores against the index, and a connected client caches the prompt
    list an expert is served through -- so a roster change has to reach both.
    Both are accelerators, never gates, so failure is silent.
    """
    try:
        from knowledge_bases.tools.catalog_index import CatalogIndex

        CatalogIndex().index_experts(
            RoutingRepository(graph).load_experts(include_unroutable=True)
        )
    except Exception:  # noqa: BLE001
        pass
    notify_tool_list_changed()
