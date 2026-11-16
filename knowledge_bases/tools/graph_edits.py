"""Apply and revert wording-kind self-improvements *in the graph*.

Tool descriptions and expert definitions are graph data (see
`knowledge_bases/tools/catalog.py`, `knowledge_bases/orchestration/
experts_catalog.py` for the seeds), so a wording improvement is a graph
write, not a source edit:

- `tool_wording` sets `Tool.description_override` (the seed's base
  `description` is untouched, so `sync_catalog()` never clobbers it);
  revert clears the override back to what it was.
- `routing_capability_wording` appends one phrase to the expert's
  `capability_overlays`; `expert_prompt_wording` appends one paragraph to
  its `prompt_overlays`. Each is a new `ExpertRevision`; revert removes the
  one overlay item as another revision, so the audit trail is complete.

Two gates protect routing wording before it is written:

1. **Pollution guard** -- the phrase must be closer to the target expert's
   own current capabilities than to any other expert's. This is exactly the
   failure that motivated it: an earlier pass appended methodology/startup
   vocabulary to `business_dictionary`, making methodology questions co-route
   there.
2. **Eval gate** -- top-1 accuracy on the golden routing set
   (`knowledge_bases/orchestration/routing_eval.jsonl`) must not drop.

`new_tool` is the one kind that remains a real source edit
(`source_editor.py` + git + test gate); it does not pass through here.
"""
from __future__ import annotations

import math
from typing import Any

from graph import Graph
from knowledge_bases.orchestration.models import (
    ExpertDefinition,
    PromptPrefixError,
    validate_prompt_prefix_overlays,
)
from knowledge_bases.orchestration.repository import RoutingRepository
from knowledge_bases.orchestration.routing import tokens
from knowledge_bases.orchestration.routing_eval import answerable_cases, evaluate_routing
from knowledge_bases.tools.catalog_notify import notify_tool_list_changed
from knowledge_bases.tools.models import SelfImprovementKind, SelfImprovementSuggestion
from knowledge_bases.tools.repository import ToolRepository

GRAPH_EDIT_KINDS = frozenset({
    SelfImprovementKind.tool_wording,
    SelfImprovementKind.routing_capability_wording,
    SelfImprovementKind.expert_prompt_wording,
    SelfImprovementKind.expert_prompt_prefix,
})
_REASON_APPLY = "self_improvement"
_REASON_REVERT = "self_improvement_revert"


class GraphEditError(RuntimeError):
    pass


# --- pollution guard -------------------------------------------------------------

def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def check_capability_pollution(
    phrase: str, target_key: str, experts: list[ExpertDefinition], llm: Any | None = None,
) -> dict[str, Any]:
    """Raise `GraphEditError` if `phrase` resembles another routable
    expert's vocabulary more than the target's own. Token Jaccard is always
    used; when an `llm` with `embed()` is given, embedding cosine is
    averaged in so paraphrases are caught too."""
    phrase_tokens = tokens(phrase)
    embed = getattr(llm, "embed", None) if llm is not None else None
    phrase_vec = embed(phrase) if embed else None

    def similarity(expert: ExpertDefinition) -> float:
        text = " ".join(expert.effective_capabilities + [expert.description])
        score = _jaccard(phrase_tokens, tokens(text))
        if phrase_vec is not None:
            score = (score + max(0.0, _cosine(phrase_vec, embed(text)))) / 2
        return score

    target = next((e for e in experts if e.key == target_key), None)
    if target is None:
        raise GraphEditError(f"Unknown expert {target_key!r}")
    own = similarity(target)
    others = {e.key: similarity(e) for e in experts if e.key != target_key and e.routable}
    closest = max(others, key=others.get) if others else None
    report = {"own": round(own, 4), "closest_other": closest, "closest_other_score": round(others.get(closest, 0.0), 4)}
    if closest is not None and others[closest] > 0 and others[closest] >= own:
        raise GraphEditError(
            f"Rejected capability phrase for {target_key!r}: it reads more like "
            f"{closest!r} ({others[closest]:.2f}) than like {target_key!r} ({own:.2f}); "
            "applying it would pull that expert's questions here."
        )
    return report


def check_routing_eval(before: list[ExpertDefinition], after: list[ExpertDefinition]) -> dict[str, Any]:
    # Scored on the cases this product can possibly answer. The eval set names
    # experts an installed profile may not have, and an unwinnable case is a
    # constant in both numbers -- it cannot change the verdict, but it dilutes
    # the comparison, so a real regression moves the accuracy less than it
    # should. `before`/`after` differ only by the edit, so the same filter
    # applies to both.
    cases = answerable_cases(before)
    pre, post = evaluate_routing(before, cases), evaluate_routing(after, cases)
    if post["accuracy"] < pre["accuracy"]:
        raise GraphEditError(
            f"Rejected: routing eval accuracy would drop {pre['accuracy']:.2f} -> "
            f"{post['accuracy']:.2f}; new misses: {post['misses'][:3]}"
        )
    return {"eval_accuracy_before": pre["accuracy"], "eval_accuracy_after": post["accuracy"]}


# --- apply / revert ---------------------------------------------------------------

def _reindex(graph: Graph, experts: bool = False, skills: bool = False) -> None:
    try:
        from knowledge_bases.tools.catalog_index import CatalogIndex

        index = CatalogIndex()
        if experts:
            index.index_experts(RoutingRepository(graph).load_experts(include_unroutable=True))
        if skills:
            index.index_skills(ToolRepository(graph).load_tools())
    except Exception:  # noqa: BLE001 -- the index is an accelerator, never a gate
        pass
    # A reworded skill description changes what a connected harness should be
    # showing the model, so ask it to re-fetch (see
    # `knowledge_bases.tools.catalog_notify`). Expert prompt/capability
    # overlays are fetched per delegation and need no notification.
    if skills:
        notify_tool_list_changed()


def apply_graph_edit(
    suggestion: SelfImprovementSuggestion, graph: Graph, llm: Any | None = None,
) -> dict[str, Any]:
    """Write one wording suggestion to the graph. Returns the record a
    caller stores on the `ReviewFinding` so `revert_graph_edit` can undo
    exactly this change later."""
    change = (suggestion.suggested_change or "").strip()
    if not change:
        raise GraphEditError("Suggestion has no suggested_change to apply")
    kind, target = suggestion.kind, suggestion.target

    if kind == SelfImprovementKind.tool_wording:
        previous = ToolRepository(graph).override_description(target, change)
        _reindex(graph, skills=True)
        return {"kind": kind.value, "target": target, "value": change,
                "previous_override": previous["previous_override"]}

    repository = RoutingRepository(graph)
    experts = repository.load_experts(include_unroutable=True)
    expert = next((e for e in experts if e.key == target), None)
    if expert is None:
        raise GraphEditError(f"Unknown expert {target!r}")
    record: dict[str, Any] = {"kind": kind.value, "target": target, "value": change}

    if kind == SelfImprovementKind.routing_capability_wording:
        if change in expert.effective_capabilities:
            raise GraphEditError(f"{target!r} already has capability phrase {change!r}")
        record["pollution_check"] = check_capability_pollution(change, target, experts, llm)
        revised = expert.model_copy(update={"capability_overlays": expert.capability_overlays + [change]})
        record.update(check_routing_eval(experts, [revised if e.key == target else e for e in experts]))
    elif kind == SelfImprovementKind.expert_prompt_wording:
        if change in expert.prompt_overlays:
            raise GraphEditError(f"{target!r} already carries that prompt guidance")
        revised = expert.model_copy(update={"prompt_overlays": expert.prompt_overlays + [change]})
    elif kind == SelfImprovementKind.expert_prompt_prefix:
        if change in expert.prompt_prefix_overlays:
            raise GraphEditError(f"{target!r} already carries that supplemental context")
        try:
            overlays = validate_prompt_prefix_overlays(
                expert.prompt_prefix_overlays + [change]
            )
        except PromptPrefixError as exc:
            # Budget refusals are a normal outcome, not a bug: the slot is
            # deliberately small, so say which limit was hit.
            raise GraphEditError(f"Rejected supplemental context for {target!r}: {exc}") from exc
        revised = expert.model_copy(update={"prompt_prefix_overlays": overlays})
    else:
        raise GraphEditError(f"{kind.value!r} is not a graph-editable kind")

    record["version"] = repository.revise_expert(revised, reason=f"{_REASON_APPLY}:{kind.value}")
    _reindex(graph, experts=kind == SelfImprovementKind.routing_capability_wording)
    return record


_REVERT_OVERLAY_KINDS = {
    SelfImprovementKind.routing_capability_wording: "capability",
    SelfImprovementKind.expert_prompt_wording: "prompt",
    SelfImprovementKind.expert_prompt_prefix: "prompt_prefix",
}


def revert_graph_edit(record: dict[str, Any], graph: Graph) -> dict[str, Any]:
    """Undo exactly the change `apply_graph_edit` recorded."""
    kind = SelfImprovementKind(record["kind"])
    target, value = record["target"], record["value"]
    if kind == SelfImprovementKind.tool_wording:
        ToolRepository(graph).override_description(target, record.get("previous_override"))
        _reindex(graph, skills=True)
        return {"reverted": kind.value, "target": target}
    try:
        overlay = _REVERT_OVERLAY_KINDS[kind]
    except KeyError as exc:
        raise GraphEditError(f"{kind.value!r} is not a revertible graph edit") from exc
    version = RoutingRepository(graph).revert_expert_overlay(
        target, overlay, value, reason=f"{_REASON_REVERT}:{kind.value}",
    )
    _reindex(graph, experts=overlay == "capability")
    return {"reverted": kind.value, "target": target, "version": version}
