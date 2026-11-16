"""The decision-health engine.

evaluate_decisions(): for every non-deprecated Decision (see `Graph.mark_deprecated`
in `graph.py`), run the 4 signals and set status = ReviewNeeded (with reasons)
if any fire.

Signals:
  - Premise:        a SUPPORTS edge's `valid_if` no longer holds vs its Claim.
  - Outcome:        the MEASURED_BY Claim breaches the INTENDED_FOR Goal target.
  - Alternative:    a CHOSEN_OVER alternative now out-scores the decision.
  - Counter-evidence: a high-weight CHALLENGES edge, or net opposing > erosion.
"""
from __future__ import annotations

import operator
import re
from datetime import datetime, timezone

from graph import Graph

OPS = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt,
    ">=": operator.ge, "==": operator.eq, "!=": operator.ne,
}

HARD_DEFEATER = 0.7
EROSION_LIMIT = 0.5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _eval_cond(value: float, cond: str | None) -> bool:
    """Evaluate a premise condition like '< 3' or 'value <= 1.5'."""
    if not cond:
        return True
    cond = cond.replace("value", "").strip()
    m = re.match(r"(<=|>=|==|!=|<|>)\s*(-?\d+(?:\.\d+)?)", cond)
    if not m:
        return True
    return OPS[m.group(1)](float(value), float(m.group(2)))


# --- decision signals ---------------------------------------------------

def _check_premise(graph: Graph, did: str) -> list[str]:
    rows = graph.run(
        "MATCH (c:Claim)-[r:SUPPORTS]->(d:Decision {id: $did}) "
        "WHERE r.valid_if IS NOT NULL AND coalesce(r.status, 'active') = 'active' "
        "RETURN c.statement AS stmt, c.value AS value, r.valid_if AS cond",
        did=did,
    )
    out = []
    for r in rows:
        if r["value"] is not None and not _eval_cond(r["value"], r["cond"]):
            out.append(f"Premise broken: '{r['stmt']}' no longer satisfies '{r['cond']}'")
    return out


def _check_outcome(graph: Graph, did: str) -> list[str]:
    """Flag a decision whose goal is no longer being met.

    The path the ontology supports is `Decision -[:INTENDED_FOR]-> Goal`, and the
    goal names the metric it is measured by (`Goal.metric_id`) and the bar
    (`comparator`, `target_value`). The measured number is not on the Metric --
    a Metric is the executable definition, not a reading -- and there is no edge
    from a Decision to a Metric. The reading lives on a `Claim` that carries
    `metric_id` and `value`, which is how a claim Noesis derived from data is
    recorded. So the traversal goes through the goal, not from the decision.

    The most recent reading is what counts, so a stale measurement does not
    outvote a fresh one.
    """
    rows = graph.run(
        "MATCH (d:Decision {id: $did})-[:INTENDED_FOR]->(g:Goal) "
        "WHERE g.metric_id IS NOT NULL AND g.comparator IS NOT NULL "
        "  AND g.target_value IS NOT NULL "
        "MATCH (c:Claim) "
        "WHERE c.metric_id = g.metric_id AND c.value IS NOT NULL "
        "  AND coalesce(c.status, 'active') <> 'refuted' "
        "WITH g, c ORDER BY c.computed_at DESC "
        "RETURN g.target AS gtext, g.comparator AS comp, g.target_value AS target, "
        "  c.statement AS stmt, c.value AS value, c.computed_at AS at "
        "LIMIT 1",
        did=did,
    )
    out = []
    for r in rows:
        if r["value"] is None:
            continue
        ok = OPS.get(r["comp"], lambda a, b: True)(float(r["value"]), float(r["target"]))
        if not ok:
            out.append(
                f"Outcome breach: goal '{r['gtext']}' needs {r['comp']} {r['target']}, "
                f"measured {r['value']} ({r['at']})"
            )
    return out


def _check_alternative(graph: Graph, did: str) -> list[str]:
    rows = graph.run(
        "MATCH (d:Decision {id: $did})-[r:CHOSEN_OVER]->(a:Alternative) "
        "RETURN a.name AS name, a.score AS alt, d.score AS cur, r.better AS better",
        did=did,
    )
    out = []
    for r in rows:
        if r["alt"] is None or r["cur"] is None:
            continue
        better = r.get("better") or "higher"
        if better == "higher" and r["alt"] > r["cur"]:
            out.append(f"Better alternative: '{r['name']}' scores {r['alt']} > current {r['cur']}")
        elif better == "lower" and r["alt"] < r["cur"]:
            out.append(f"Better alternative: '{r['name']}' scores {r['alt']} < current {r['cur']}")
    return out


def _check_counter_evidence(graph: Graph, did: str) -> list[str]:
    challenges = graph.run(
        "MATCH (c:Claim)-[r:CHALLENGES]->(d:Decision {id: $did}) "
        "WHERE coalesce(r.status, 'active') = 'active' "
        "RETURN c.statement AS stmt, coalesce(r.weight, 0) AS w",
        did=did,
    )
    supports = graph.run(
        "MATCH (c:Claim)-[r:SUPPORTS]->(d:Decision {id: $did}) "
        "WHERE coalesce(r.status, 'active') = 'active' "
        "RETURN coalesce(r.weight, 0) AS w",
        did=did,
    )
    out = []
    sum_ch = sum((c["w"] or 0) for c in challenges)
    sum_sp = sum((s["w"] or 0) for s in supports)
    hard = False
    for c in challenges:
        if (c["w"] or 0) >= HARD_DEFEATER:
            out.append(f"Hard counter-evidence: '{c['stmt']}' (weight {c['w']})")
            hard = True
    if not hard and (sum_ch - sum_sp) > EROSION_LIMIT:
        out.append(
            f"Eroding counter-evidence: net opposing {round(sum_ch - sum_sp, 2)} "
            f"exceeds {EROSION_LIMIT}"
        )
    return out


def evaluate_decisions(graph: Graph | None = None) -> list[dict]:
    graph = graph or Graph()
    decisions = graph.run(
        "MATCH (d:Decision) WHERE NOT d:Deprecated "
        "RETURN d.id AS id, d.description AS description"
    )
    results = []
    for d in decisions:
        did = d["id"]
        reasons = (
            _check_premise(graph, did)
            + _check_outcome(graph, did)
            + _check_alternative(graph, did)
            + _check_counter_evidence(graph, did)
        )
        status = "ReviewNeeded" if reasons else "Active"
        graph.upsert_node("Decision", {"id": did}, {"status": status, "review_reasons": reasons})
        results.append({"id": did, "description": d["description"],
                        "status": status, "reasons": reasons})
    return results


def run_all(graph: Graph | None = None) -> list[dict]:
    graph = graph or Graph()
    results = evaluate_decisions(graph)
    flagged = [r for r in results if r["status"] == "ReviewNeeded"]
    print(f"[recompute] evaluated {len(results)} decisions, {len(flagged)} need review")
    return results


if __name__ == "__main__":
    run_all()
