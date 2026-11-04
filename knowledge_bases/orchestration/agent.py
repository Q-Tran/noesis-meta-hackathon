"""Legacy one-shot CLI over the deterministic delegate path.

The real orchestrator is a harness agent instantiated from the graph's
`orchestrator` expert (see `experts_catalog.py`, `scripts/export_prime_context.py`);
this module only runs each routed delegation's entry skill once and narrates
the raw results, for smoke tests and `app.py`. It does not loop.
"""
from __future__ import annotations

import json

from graph import Graph
from knowledge_bases.orchestration.experts_catalog import ORCHESTRATOR_KEY
from knowledge_bases.orchestration.repository import RoutingRepository
from knowledge_bases.orchestration.service import ExpertOrchestrator
from llm import get_llm


def answer(question: str, user_id: str = "unknown") -> str:
    with Graph() as graph:
        result = ExpertOrchestrator(graph).delegate(question, user_id=user_id)
        orchestrator = RoutingRepository(graph).get_expert(ORCHESTRATOR_KEY)
    payload = result.model_dump(mode="json")
    narration = get_llm().complete(
        f"Question: {question}\nRouting and expert output:\n{json.dumps(payload, default=str)}\n"
        "Synthesize the expert output above into an answer. Use only these results as "
        "graph facts, attribute statements to their expert, report partial failures.",
        system=orchestrator.effective_prompt if orchestrator else None,
    ).strip()
    if narration:
        questions = ""
        if result.feedback.open_questions:
            questions = "\n\nQuestions:\n" + "\n".join(
                f"- {question}" for question in result.feedback.open_questions
            )
        return f"{narration}{questions}\n\nRouting trace: `{result.trace_id}`"
    sections = [
        f"Routing trace: `{result.trace_id}`",
        f"Status: {result.status}",
        "Experts: " + ", ".join(
            expert.knowledge_base for expert in result.expert_results
        ),
    ]
    for expert in result.expert_results:
        sections.append(
            f"\n[{expert.knowledge_base}:{expert.operation}]\n"
            f"{expert.data if expert.status == 'completed' else expert.error}"
        )
    if result.feedback.open_questions:
        sections.append("\nQuestions:\n" + "\n".join(
            f"- {question}" for question in result.feedback.open_questions
        ))
    return "\n".join(sections)


def main() -> None:
    import sys
    print(answer(" ".join(sys.argv[1:]) or "Which decisions need review?"))


if __name__ == "__main__":
    main()
