"""Fire self-improvement from failure, without firing it mid-task.

Self-improvement makes git-committed source edits without the usual
propose-confirm-execute human gate, so *when* it runs matters as much as
whether it runs. Two constraints shape this module:

- It must not run inside the turn that failed. The edit would land while the
  user is still working, and the trace that triggered it would not yet have its
  outcome recorded.
- It must not run on every failure. A failing tool called twenty times in an
  afternoon is one signal, not twenty proposals.

So triggering is *debounced and deferred*: a failure marks the graph as
"worth re-examining", and the pass runs at most once per interval. The mark is
a property on the reflexive `noesis-self` project, not a local file, so the
debounce is shared across every process and harness using the graph -- the same
reason everything else lives there.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone

from graph import Graph

logger = logging.getLogger(__name__)

SELF_PROJECT_ID = "noesis-self"

# Minimum wall-clock gap between automatic passes. A day is deliberately long:
# the signals themselves use a 7-day lookback and a 3-day re-evaluation
# cooldown, so running more often than this cannot surface anything new -- it
# would only re-draft the same findings against slightly more data.
DEFAULT_MIN_INTERVAL = timedelta(days=1)

# Opt-out, because an automatic source edit is exactly the thing an operator
# may want to disable without editing code.
ENV_FLAG = "NOESIS_AUTO_SELF_IMPROVE"


def is_enabled() -> bool:
    """Whether automatic passes may run at all.

    Off under pytest. A test that records a failed `TraceOutcome` is asserting
    something about tracing, not asking to spend an LLM budget rewriting tool
    descriptions against the developer's live graph -- which is exactly what
    happened the first time this was wired up, and it tripled the suite's
    runtime before anyone looked at why.
    """
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return os.getenv(ENV_FLAG, "1").lower() not in {"0", "false", "no", "off"}


def _last_run(graph: Graph) -> datetime | None:
    rows = graph.run(
        "MATCH (p:Project {id: $project_id}) "
        "RETURN p.last_auto_self_improvement AS last",
        project_id=SELF_PROJECT_ID,
    )
    if not rows or not rows[0].get("last"):
        return None
    value = rows[0]["last"]
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _mark_run(graph: Graph, when: datetime) -> None:
    graph.run(
        "MATCH (p:Project {id: $project_id}) "
        "SET p.last_auto_self_improvement = $when",
        project_id=SELF_PROJECT_ID, when=when.isoformat(),
    )


def should_run(graph: Graph, min_interval: timedelta = DEFAULT_MIN_INTERVAL) -> bool:
    """Whether enough time has passed since the last automatic pass."""
    if not is_enabled():
        return False
    last = _last_run(graph)
    if last is None:
        return True
    return datetime.now(timezone.utc) - last >= min_interval


def trigger_in_background(min_interval: timedelta | None = None) -> None:
    """Run a due pass on a daemon thread, returning immediately.

    A pass is several LLM round-trips. Run inline from the failure hook it
    would sit inside the caller's failing call, so a tool that errored in 30ms
    would take tens of seconds to return the error -- turning a fast failure
    into a hang, which is a worse bug than the one being diagnosed.

    Daemon, so a pending pass never holds up interpreter exit: the findings are
    a nice-to-have, and the next failure triggers another attempt anyway.
    """
    if not is_enabled():
        return
    kwargs = {} if min_interval is None else {"min_interval": min_interval}
    threading.Thread(
        target=maybe_run_self_improvement,
        kwargs=kwargs,
        name="noesis-self-improvement",
        daemon=True,
    ).start()


def maybe_run_self_improvement(
    graph: Graph | None = None,
    *,
    min_interval: timedelta = DEFAULT_MIN_INTERVAL,
    propose_only: bool = True,
) -> dict:
    """Run a self-improvement pass if one is due, otherwise do nothing.

    `propose_only` defaults to True: an automatic trigger *files findings* but
    does not apply them. Applying edits code and commits to git, and doing that
    as a side effect of a user's task failing is a surprise that no threshold
    tuning makes acceptable. The findings are in the graph either way, and
    `apply_self_improvement` is one call.

    Never raises. This runs off the back of someone else's failure; turning
    their failure into a second, louder failure would be strictly worse.
    """
    owns_graph = graph is None
    graph = graph or Graph()
    try:
        if not should_run(graph, min_interval):
            return {"ran": False, "reason": "not due"}

        # Mark before running, not after: a pass that crashes halfway should
        # still consume its slot rather than retrying on every subsequent
        # failure.
        _mark_run(graph, datetime.now(timezone.utc))

        from knowledge_bases.tools.self_improvement import run_self_improvement

        filed = run_self_improvement(graph=graph)
        return {
            "ran": True,
            "filed": len(filed),
            "findings": [s.target for s in filed],
            "applied": False if propose_only else None,
        }
    except Exception:
        logger.warning("Automatic self-improvement pass failed", exc_info=True)
        return {"ran": False, "reason": "error"}
    finally:
        if owns_graph:
            try:
                graph.close()
            except Exception:
                pass
