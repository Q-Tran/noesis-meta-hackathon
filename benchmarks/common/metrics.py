"""Metrics helpers, adapted from mem0ai/memory-benchmarks (benchmarks/common/metrics.py)."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


def compute_locomo_metrics(evaluations: list[dict[str, Any]], cutoffs: list[int]) -> dict:
    """Per-category and overall metrics at each top-k cutoff."""
    from .utils import cutoff_label

    metrics_by_cutoff: dict[str, Any] = {}
    for c in cutoffs:
        label = cutoff_label(c)
        total = len(evaluations)
        scores = [e.get("cutoff_results", {}).get(label, {}).get("score", 0.0) for e in evaluations]
        correct = sum(1 for s in scores if s >= 0.5)

        by_category: dict[str, list] = defaultdict(list)
        for e in evaluations:
            cat_name = e.get("category_name", "unknown")
            by_category[cat_name].append(e.get("cutoff_results", {}).get(label, {}).get("score", 0.0))

        cat_metrics = {}
        for cat_name in sorted(by_category):
            cat_scores = by_category[cat_name]
            cat_correct = sum(1 for s in cat_scores if s >= 0.5)
            cat_metrics[cat_name] = {
                "total": len(cat_scores),
                "correct": cat_correct,
                "accuracy": cat_correct / len(cat_scores) * 100 if cat_scores else 0.0,
                "avg_score": statistics.mean(cat_scores) * 100 if cat_scores else 0.0,
            }

        metrics_by_cutoff[label] = {
            "overall": {
                "total": total,
                "correct": correct,
                "accuracy": correct / total * 100 if total else 0.0,
                "avg_score": statistics.mean(scores) * 100 if scores else 0.0,
            },
            "by_category": cat_metrics,
        }
    return metrics_by_cutoff


def display_results(metrics_by_cutoff: dict, cutoffs: list[int]) -> None:
    from .utils import cutoff_label

    for c in cutoffs:
        label = cutoff_label(c)
        m = metrics_by_cutoff.get(label, {})
        overall = m.get("overall", {})
        print(f"\n--- {label} ---")
        print(
            f"  Overall: {overall.get('correct', 0)}/{overall.get('total', 0)} "
            f"({overall.get('accuracy', 0):.1f}%) avg={overall.get('avg_score', 0):.1f}%"
        )
        for cat_name, cm in sorted(m.get("by_category", {}).items()):
            print(f"  {cat_name}: {cm['correct']}/{cm['total']} ({cm['accuracy']:.1f}%)")
