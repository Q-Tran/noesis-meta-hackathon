"""Mixture-of-experts orchestration over bounded graph knowledge bases."""
from __future__ import annotations

from typing import Any, Callable

from graph import Graph, json_safe
from knowledge_bases.mcp import dispatch_tool
from knowledge_bases.orchestration.models import (
    ExpertCandidate,
    ExpertResult,
    OrchestrationResult,
    RoutingDecision,
    RoutingPlan,
)
from knowledge_bases.orchestration.experts_catalog import bootstrap_experts
from knowledge_bases.registry import orchestrator_dispatch_knowledge_bases
from knowledge_bases.orchestration.repository import RoutingRepository
from knowledge_bases.orchestration.routing import plan_route
from knowledge_bases.orchestration.hybrid_search import hybrid_search
from knowledge_bases.tools.models import SpanKind
from knowledge_bases.tools.tracing import record_tool_span
from llm import BaseLLM, get_llm


Dispatcher = Callable[..., Any]

# Every routable expert's read-side operations resolve through the same
# graph-registered `Tool` handler its own MCP server dispatches to (see
# `knowledge_bases.mcp.dispatch_tool`); there is no separate copy of
# domain dispatch logic here. `trace_id` is threaded through so every
# delegated skill call is tagged with the RoutingTrace that produced it.
# Derived from each knowledge base's `orchestrator_dispatch` declaration in
# its `package.py` (see `knowledge_bases/base.py`).
_ROUTABLE_KNOWLEDGE_BASES = orchestrator_dispatch_knowledge_bases()


def domain_dispatchers() -> dict[str, Dispatcher]:
    return {
        knowledge_base: (
            lambda operation, arguments, graph, trace_id=None, _kb=knowledge_base: dispatch_tool(
                graph, _kb, operation, arguments, trace_id=trace_id,
            )
        )
        for knowledge_base in _ROUTABLE_KNOWLEDGE_BASES
    }


class ExpertOrchestrator:
    def __init__(
        self,
        graph: Graph,
        dispatchers: dict[str, Dispatcher] | None = None,
        repository: RoutingRepository | None = None,
        llm: BaseLLM | None = None,
        index: Any | None = None,
    ):
        self.graph = graph
        self.repository = repository or RoutingRepository(graph)
        self.dispatchers = dispatchers or domain_dispatchers()
        self.llm = llm or get_llm()
        self._index = index

    @property
    def index(self) -> Any:
        if self._index is None:
            from knowledge_bases.tools.catalog_index import CatalogIndex

            self._index = CatalogIndex()
        return self._index

    def sync_experts(self, index: Any | None = None) -> None:
        from knowledge_bases.tools.catalog_index import CatalogIndex

        experts = bootstrap_experts()
        self.repository.sync_experts(experts)
        (index or CatalogIndex()).index_experts(self.repository.load_experts(include_unroutable=True))

    def _llm_select(self, question: str) -> Callable[[list[ExpertCandidate]], list[str]]:
        """Server-side final choice among the retrieved candidates. Falls
        back to threshold selection (empty list) on unparseable output."""
        def select(candidates: list[ExpertCandidate]) -> list[str]:
            ranked = "\n".join(
                f"- {c.knowledge_base} (score {c.score:.2f}; {'; '.join(c.reasons) or 'no evidence'})"
                for c in candidates
            )
            decision = self.llm.extract(
                f"Question: {question}\n\nRanked expert candidates:\n{ranked}\n\n"
                "Select every expert that must contribute to answer the question, "
                "in the order they should run. Prefer fewer experts; include an "
                "expert only if its evidence is relevant.",
                RoutingDecision,
                system="You route questions to bounded knowledge-graph experts.",
            )
            return decision.selected
        return select

    def route(self, question: str, decide: bool = False) -> RoutingPlan:
        """Retrieve and rank experts for `question` and build starting
        delegations. Semantic evidence (expert capabilities and skill
        descriptions from the catalog index) is fused with keyword overlap;
        both degrade gracefully to keyword-only. With `decide=True` Noesis's
        own LLM makes the final selection over the ranked candidates;
        otherwise the threshold rule does, and the calling harness LLM is
        expected to weigh close candidates itself.
        """
        profiles = self.repository.load_experts()
        if not profiles:
            self.sync_experts()
            profiles = self.repository.load_experts()
        expert_hits = skill_hits = None
        try:
            expert_hits = self.index.search_experts(question)
            skill_hits = self.index.search_skills(question)
        except Exception:
            expert_hits = skill_hits = None
        skills_by_key = None
        try:
            from knowledge_bases.tools.repository import ToolRepository

            skills_by_key = {s.key: s for s in ToolRepository(self.graph).load_tools()}
        except Exception:
            skills_by_key = None
        plan = plan_route(
            question, profiles,
            expert_hits=expert_hits, skill_hits=skill_hits, skills_by_key=skills_by_key,
            select=self._llm_select(question) if decide else None,
        )
        self.repository.record_plan(plan, question)
        return plan

    def execute(self, plan: RoutingPlan) -> OrchestrationResult:
        results = []
        for delegation in plan.delegations:
            try:
                if delegation.operation == "hybrid_search":
                    with record_tool_span(
                        delegation.knowledge_base, "hybrid_search",
                        delegation.arguments, kind=SpanKind.tool_call,
                        trace_id=plan.trace_id,
                    ):
                        data = hybrid_search(graph=self.graph, **delegation.arguments)
                else:
                    data = self.dispatchers[delegation.knowledge_base](
                        delegation.operation, delegation.arguments, self.graph,
                        trace_id=plan.trace_id,
                    )
                if hasattr(data, "model_dump"):
                    data = data.model_dump(mode="python")
                data = json_safe(data)
                results.append(
                    ExpertResult(
                        knowledge_base=delegation.knowledge_base,
                        operation=delegation.operation,
                        status="completed",
                        data=data,
                    )
                )
            except Exception as exc:
                results.append(
                    ExpertResult(
                        knowledge_base=delegation.knowledge_base,
                        operation=delegation.operation,
                        status="failed",
                        error=str(exc),
                    )
                )
        if all(result.status == "completed" for result in results):
            status = "completed"
        elif any(result.status == "completed" for result in results):
            status = "partial"
        else:
            status = "failed"
        self.repository.complete_trace(plan.trace_id, results, status)
        return OrchestrationResult(
            trace_id=plan.trace_id,
            plan=plan,
            expert_results=results,
            status=status,
        )

    def delegate(self, question: str, user_id: str = "unknown") -> OrchestrationResult:
        """Route, execute, and catalog whatever the user asserted along the way.

        The cataloging step is a CAPABILITY, not an import. Extracting claims from an
        answer requires a package that owns `Claim`, so a product shipping without
        one must still be able to route and answer -- and previously could not even
        import this module. When no installed package provides the hook, the answer
        is returned uncatalogued rather than the request failing, because fewer
        installed ontologies is a supported product, not an error.
        """
        from knowledge_bases.registry import capability

        result = self.execute(self.route(question))

        analyze = capability("feedback.analyze")
        recorder = capability("feedback.record")
        if analyze is None or recorder is None:
            return result

        analysis = analyze(
            question,
            [expert.model_dump(mode="json") for expert in result.expert_results],
            self.llm,
            result.plan.clarification_needed,
        )
        result.feedback = recorder(self.graph).record(
            analysis.claims, user_id=user_id
        )
        return result
