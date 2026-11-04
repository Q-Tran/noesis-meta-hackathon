"""Bootstrap catalog of every graph-defined expert.

Like `knowledge_bases/tools/catalog.py` for skills, this is *seed data*: run
`python scripts/sync_tools_catalog.py` (which also syncs experts) to
`MERGE` it into the graph. At runtime nothing reads this module -- the
orchestrator MCP server serves experts straight from `KnowledgeBaseExpert`
nodes (`list_noesis_experts`/`get_noesis_expert`), harness agent files are
generated from the graph (`scripts/export_prime_context.py`), and
self-improvement revises the graph nodes directly, so an edit here only
matters once it is synced.

The `orchestrator` entry is the primary agent's own definition: it is not
routable, and its prompt describes how to drive the loop -- route, delegate
each delegation to that expert's subagent, synthesize, record the outcome.
"""
from __future__ import annotations

from knowledge_bases.orchestration.models import ExpertDefinition
from knowledge_bases.registry import KNOWLEDGE_BASES


ORCHESTRATOR_KEY = "orchestrator"

# Control-plane skills every routable expert may call, on top of its own
# knowledge base's. These are exactly what `EXPERT_LOOP_CONTRACT` instructs an
# expert to use, so they must be within its `allowed_tools` for
# `call_noesis_tool`'s expert-scope enforcement to permit them. They carry no
# domain authority: discovery, read-only cross-domain search, and recording
# the expert's own result.
SHARED_EXPERT_TOOLS = (
    "orchestration.search_noesis_tools",
    "orchestration.hybrid_search",
    "orchestration.get_delegation_results",
    "orchestration.get_noesis_expert",
    "orchestration.spawn_ephemeral_subagent",
    "orchestration.register_tool",
)

# Shared tail of every routable expert's prompt: how to loop over the
# graph-registered tools. Kept in one place so the loop discipline is
# identical across experts and a change here reaches all of them on sync.
EXPERT_LOOP_CONTRACT = """
How you work:
1. You have been handed one delegation from the orchestrator: the user's
   question, a `trace_id`, and usually a starting `knowledge_base`/`operation`/
   `arguments`. Start there; pass the `trace_id` and your `expert_key` on every
   `call_noesis_tool` call so your work stays correlated with the routing
   decision.
2. Keep going until the delegation is actually resolved or you hit a real dead
   end. If a result is ambiguous, empty, or partial, make the next call
   yourself. When the right next tool isn't obvious, call `search_noesis_tools`
   with a short plain-language description of what you need (optionally scoped
   with your `expert_key`) -- it returns real tool names, knowledge bases, and
   argument schemas ready for `call_noesis_tool`. Never guess a tool, table, or
   column name when a bounded tool can resolve it.
3. Use `hybrid_search` for exploratory lookups across entity types in your
   knowledge base family. If you need facts or operations from another expert's
   domain, spawn an isolated ephemeral subagent via `spawn_ephemeral_subagent`
   (with the target `expert_key` and focused `task`).
4. Never run raw Cypher, mutate graph entities generically, read arbitrary
   files, or bypass a knowledge base's bounded tools. Writes go through the
   orchestrator's propose/confirm/execute flow, never directly from you.
5. Report back concisely: what you found (or did not), every tool you called
   and why, the `trace_id`, and any remaining ambiguity or missing data. Do not
   dump raw intermediate output.
6. The tools you called, entities you resolved, and execution results are
   captured and derived automatically from your trace spans by the harness —
   manual result recording is not required.
""".strip()


def _expert(
    key: str,
    *,
    priority: int,
    capabilities: list[str],
    system_prompt: str,
    entry_tools: list[str],
    allowed_tools: list[str] | None = None,
    routable: bool = True,
    name: str | None = None,
    description: str | None = None,
) -> ExpertDefinition:
    kb = KNOWLEDGE_BASES.get(key)
    return ExpertDefinition(
        key=key,
        name=name or (kb.name if kb else key),
        description=description or (kb.description if kb else ""),
        capabilities=capabilities,
        priority=priority,
        system_prompt=(
            f"You are the Noesis {name or (kb.name if kb else key)} expert.\n\n"
            f"{system_prompt.strip()}\n\n{EXPERT_LOOP_CONTRACT}"
            if routable else system_prompt.strip()
        ),
        entry_tools=entry_tools,
        # Every expert may also reach the control-plane skills the loop
        # contract tells it to use: tool discovery, cross-domain search, and
        # recording its own delegation result. Without these in
        # `allowed_tools`, `call_noesis_tool`'s expert-scope check would
        # refuse the very calls the contract mandates.
        allowed_tools=(
            allowed_tools if allowed_tools is not None else [f"{key}.*"]
        ) + list(SHARED_EXPERT_TOOLS),
        routable=routable,
    )

def bootstrap_experts() -> list[ExpertDefinition]:
    """Compose every installed ontology package's seeded experts.

    Each knowledge base owns its slice in `knowledge_bases/<key>/experts.py`;
    `experts_seed` references are resolved lazily in `_EXPERT_PACKAGE_ORDER`
    (the historical seed order -- routing can tie-break on candidate order,
    so composition order is deliberately stable, not derived from
    `registry.DEFAULT_PACKAGES`). Packages not listed are appended in
    registration order.
    """
    from knowledge_bases.registry import installed_packages, resolve_seed

    packages = {package.key: package for package in installed_packages()}
    experts: list[ExpertDefinition] = []
    for key in _EXPERT_PACKAGE_ORDER:
        package = packages.pop(key, None)
        if package and package.experts_seed:
            experts.extend(resolve_seed(package.experts_seed))
    for package in packages.values():
        if package.experts_seed:
            experts.extend(resolve_seed(package.experts_seed))
    return experts


# Historical seed order for expert composition (see bootstrap_experts).
_EXPERT_PACKAGE_ORDER = (
    "business_dictionary",
    "semantic_layer",
    "epistemology",
    "strategy",
    "methodology",
    "architecture",
    "project_management",
    "development",
    "orchestration",
)


def bootstrap_profiles() -> list[ExpertDefinition]:
    """Routable experts only -- what routing scores a question against."""
    return [expert for expert in bootstrap_experts() if expert.routable]
