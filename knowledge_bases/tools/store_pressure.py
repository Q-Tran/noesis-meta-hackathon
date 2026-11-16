"""Store size against page cache -- the number that decides hot/cold tiering.

Tiering would be written around evicting deprecated and superseded data.
Measured, there is none: 0 `:Deprecated` nodes and 0 superseded edges, so the
eviction trigger never fires and the whole store still fits in the page cache.

What actually grows without bound is telemetry. `Span` and friends are
append-only and nothing deletes them. This module reports the one ratio that
turns "should we build tiering?" from a hunch into a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from graph import Graph

# Above this store-to-page-cache ratio, the graph no longer fits in RAM and
# queries start hitting disk. 0.8 leaves headroom to act before it bites.
PRESSURE_THRESHOLD = 0.8

# Labels that grow with use rather than with imported source data. Catalog
# labels are excluded on purpose: they are bounded by the upstream source and
# replaced wholesale per `snapshot_id`, so they are not a retention problem.
TELEMETRY_LABELS = (
    "Span",
    "RoutingTrace",
    "TraceOutcome",
    "SessionMemory",
)


@dataclass
class StorePressure:
    """Store size versus the cache that is supposed to hold it."""

    nodes: int
    relationships: int
    page_cache_bytes: int | None
    telemetry_counts: dict[str, int] = field(default_factory=dict)
    deprecated_nodes: int = 0

    @property
    def telemetry_nodes(self) -> int:
        return sum(self.telemetry_counts.values())

    @property
    def evictable_nodes(self) -> int:
        """What a cold tier could actually move out of the hot graph today.

        This is the honest version of the tiering premise. If it is 0, tiering
        has nothing to do, however large the graph is.
        """
        return self.deprecated_nodes

    def ratio(self, store_bytes: int) -> float | None:
        """Store size as a fraction of page cache, or None if cache is unknown."""
        if not self.page_cache_bytes:
            return None
        return store_bytes / self.page_cache_bytes

    def under_pressure(self, store_bytes: int) -> bool:
        ratio = self.ratio(store_bytes)
        if ratio is None:
            return False
        return ratio >= PRESSURE_THRESHOLD


def _page_cache_bytes(graph: Graph) -> int | None:
    """Read the configured page cache, or None when it cannot be determined.

    Returning None rather than guessing matters: a fabricated denominator would
    make the pressure ratio look authoritative while being meaningless.
    """
    try:
        rows = graph.run(
            "CALL dbms.listConfig('server.memory.pagecache.size') "
            "YIELD value RETURN value AS value"
        )
    except Exception:  # noqa: BLE001 - config procedure is not always permitted
        return None
    if not rows:
        return None
    return _parse_size(rows[0].get("value"))


_UNITS = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}


def _parse_size(value: object) -> int | None:
    """Parse Neo4j byte sizes such as `512M`, `1g`, or a bare byte count."""
    if value is None:
        return None
    # Neo4j reports this as `512.00MiB`, but a hand-edited neo4j.conf may say
    # `512M` or a bare byte count. Strip the byte suffix, then the IEC `i`.
    text = str(value).strip().lower().removesuffix("b").removesuffix("i")
    if not text:
        return None
    multiplier = _UNITS.get(text[-1:], 1)
    digits = text[:-1] if multiplier != 1 else text
    try:
        return int(float(digits) * multiplier)
    except ValueError:
        return None


def collect_store_pressure(graph: Graph) -> StorePressure:
    """Measure the graph, without assuming anything about its contents."""
    nodes = graph.run("MATCH (n) RETURN count(n) AS count")[0]["count"]
    relationships = graph.run(
        "MATCH ()-[r]->() RETURN count(r) AS count"
    )[0]["count"]
    deprecated = graph.run(
        "MATCH (n:Deprecated) RETURN count(n) AS count"
    )[0]["count"]

    telemetry: dict[str, int] = {}
    for label in TELEMETRY_LABELS:
        rows = graph.run(f"MATCH (n:`{label}`) RETURN count(n) AS count")
        telemetry[label] = rows[0]["count"] if rows else 0

    return StorePressure(
        nodes=nodes,
        relationships=relationships,
        page_cache_bytes=_page_cache_bytes(graph),
        telemetry_counts=telemetry,
        deprecated_nodes=deprecated,
    )
