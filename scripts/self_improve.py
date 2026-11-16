"""Run the full trace-driven self-improvement pass once: propose, apply,
and reevaluate.

Mirrors `self_recompute.py`'s shape (same `noesis-self` project) but
analyzes execution traces (`SpanEvent`/`TraceOutcome`/`RoutingTrace`) for
skill/routing quality instead of code-vs-rationale architecture drift.
Unlike `self_recompute.py`'s propose-only `ReviewFinding`s, every new
finding here is applied immediately as a real, git-committed source edit
-- by explicit design, the one Noesis write that skips the usual
propose-confirm-execute human gate (see
`knowledge_bases/tools/self_improvement_apply.py` and
`docs/architecture/self_improvement.md` for why that's considered safe
here). Findings applied on an earlier run are re-measured and kept or
reverted once their cooldown has elapsed, and `open` findings whose signal
no longer holds are retired -- a finding is a claim about the present, so
one nothing ever re-checks becomes a permanent accusation.

Run on a schedule (e.g. a cron/CI job) -- each phase is also individually
callable on demand via the `propose_self_improvements`/
`apply_self_improvement`/`reevaluate_self_improvements` MCP tools
(`knowledge_bases.tools.mcp_server`), e.g. for an agent to trigger
mid-conversation per `.opencode/agents/noesis-orchestrator.md`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_bases.tools.self_improvement import finding_key, run_self_improvement
from knowledge_bases.tools.self_improvement_apply import (
    ApplyError,
    apply_finding,
    reevaluate_applied_findings,
    retire_stale_findings,
)

if __name__ == "__main__":
    filed = run_self_improvement()
    print(f"[self_improve] filed {len(filed)} new review finding(s)")

    for suggestion in filed:
        key = finding_key(suggestion)
        try:
            result = apply_finding(suggestion)
            how = result.get("commit_sha") or f"graph edit {result.get('graph_edit', {}).get('kind')}"
            print(f"[self_improve] applied {key}: {how}")
        except ApplyError as exc:
            print(f"[self_improve] could not apply {key}: {exc}")

    verdicts = reevaluate_applied_findings()
    for verdict in verdicts:
        print(f"[self_improve] reevaluated {verdict['finding_key']}: {verdict['verdict']}")

    # Last, so a finding filed moments ago is judged against the same window
    # everything else just used, and a finding applied above is owned by
    # `reevaluate_applied_findings` rather than retired out from under it.
    for retired in retire_stale_findings():
        if retired["verdict"] == "retired":
            print(f"[self_improve] retired {retired['finding_key']}: {retired['reason']}")
