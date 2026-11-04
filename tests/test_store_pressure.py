"""Store pressure: the measurement that decides whether tiering is warranted.

These tests pin the reasoning, not just the arithmetic. The important case is
the one the live graph is actually in: over the pressure threshold *and* with
nothing evictable, which means the answer is retention and cache sizing rather
than a second database.
"""
from __future__ import annotations

import pytest

from knowledge_bases.tools.store_pressure import (
    PRESSURE_THRESHOLD,
    StorePressure,
    _parse_size,
    collect_store_pressure,
)

MB = 1024**2


@pytest.mark.parametrize(
    "value,expected",
    [
        ("512.00MiB", 512 * MB),  # what Neo4j actually reports
        ("512M", 512 * MB),       # what a hand-edited neo4j.conf says
        ("1g", 1024 * MB),
        ("1GiB", 1024 * MB),
        ("1024", 1024),           # bare byte count
        ("0", 0),
    ],
)
def test_neo4j_size_formats_all_parse(value, expected):
    """The reported format (`512.00MiB`) is not the configured one (`512M`).
    Parsing only one of them silently yields an unknown page cache, which makes
    the pressure ratio unavailable exactly when it is needed."""
    assert _parse_size(value) == expected


@pytest.mark.parametrize("value", ["", None, "garbage", "MiB"])
def test_unparseable_sizes_are_unknown_not_guessed(value):
    """A fabricated denominator would make the ratio look authoritative while
    meaning nothing."""
    assert _parse_size(value) is None


def test_ratio_is_unknown_when_the_page_cache_cannot_be_read():
    pressure = StorePressure(nodes=1, relationships=0, page_cache_bytes=None)

    assert pressure.ratio(466 * MB) is None
    assert pressure.under_pressure(466 * MB) is False, (
        "an unknown cache must not be reported as pressure; that would be an "
        "alert with no evidence behind it"
    )


def test_the_live_situation_is_pressure_with_nothing_to_evict():
    """The measured state of this project, as a test.

    466 MB of store against a 512 MB page cache is 0.91 -- over the threshold.
    But `:Deprecated` is 0, so an eviction-based tier has nothing to move.
    Both halves have to be reported, because either one alone gives the wrong
    instruction: the ratio alone says "build tiering", and the evictable count
    alone says "nothing to do".
    """
    pressure = StorePressure(
        nodes=166_089,
        relationships=578_020,
        page_cache_bytes=512 * MB,
        telemetry_counts={"Span": 286, "RoutingTrace": 99},
        deprecated_nodes=0,
    )
    store_bytes = 466 * MB

    assert pressure.under_pressure(store_bytes) is True
    assert pressure.evictable_nodes == 0
    assert round(pressure.ratio(store_bytes), 2) == 0.91


def test_telemetry_nodes_exclude_catalog_labels():
    """Catalog data is bounded by the upstream source and replaced per
    `snapshot_id`; counting it as growth would point retention at the wrong
    91% of the graph."""
    from knowledge_bases.tools.store_pressure import TELEMETRY_LABELS

    for catalog_label in ("Metric", "Column", "Table", "BusinessTerm"):
        assert catalog_label not in TELEMETRY_LABELS


def test_a_graph_below_the_threshold_is_not_under_pressure():
    pressure = StorePressure(nodes=10, relationships=5, page_cache_bytes=1024 * MB)

    assert pressure.under_pressure(100 * MB) is False
    assert pressure.ratio(100 * MB) < PRESSURE_THRESHOLD


# -- live graph ------------------------------------------------------------------


def test_collect_store_pressure_against_a_real_graph():
    """`collect_store_pressure` runs real Cypher, including a `dbms.listConfig`
    call that a fake cannot validate -- the single-argument form of it is what
    returns the page cache here."""
    from graph import Graph

    try:
        graph = Graph()
        graph.run("RETURN 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"graph unavailable: {exc}")

    try:
        pressure = collect_store_pressure(graph)
    finally:
        graph.close()

    assert pressure.nodes > 0
    assert pressure.page_cache_bytes, (
        "page cache came back unknown; the listConfig call or the size parser "
        "has regressed, and the pressure ratio is silently unavailable"
    )
    assert set(pressure.telemetry_counts) >= {"Span", "RoutingTrace"}
