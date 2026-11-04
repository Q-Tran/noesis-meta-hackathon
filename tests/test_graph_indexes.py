"""Indexes must follow the installed ontologies, not a list kept inside the engine.

`graph.ensure_indexes` used to name 17 domain labels literally. That was wrong twice
over: the engine owns no ontologies, so a product without `semantic_layer` still
created Table and Metric indexes; and a new ontology package could not index its own
labels without editing the engine. The 17 were also an accidental subset -- 44 other
described labels are MERGEd on identity exactly the same way and had no index at all.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from knowledge_bases.registry import KNOWLEDGE_BASES, entity_types, index_specs

# The set `graph.ensure_indexes` shipped before it was derived. Kept verbatim so the
# replacement can be proven a SUPERSET: deriving must not silently drop an index that
# a live deployment already relies on.
HISTORICAL_INDEXES = {
    ("Table", "name"), ("Column", "id"), ("Value", "id"), ("Metric", "id"),
    ("Claim", "id"), ("Decision", "id"), ("Goal", "id"), ("Alternative", "id"),
    ("Person", "id"), ("Project", "id"), ("Epic", "id"), ("UserStory", "id"),
    ("Task", "id"), ("SourceNode", "id"), ("DocumentNode", "id"),
    ("BusinessTerm", "name"), ("AlternateBusinessTerm", "name"),
}

def _code_only(source: str) -> str:
    """`source` with docstrings blanked out, so prose examples are not read as code.

    Uses the AST rather than a regex, because a regex over triple-quoted blocks
    cannot tell a docstring from a Cypher query built in a multi-line string -- and
    the second one is exactly what this must keep looking at.
    """
    import ast

    tree = ast.parse(source)
    lines = source.split("\n")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc is None or not node.body:
            continue
        first = node.body[0]
        for i in range(first.lineno - 1, first.end_lineno):
            lines[i] = ""
    return "\n".join(lines)


def test_the_derived_index_set_still_covers_every_index_the_engine_used_to_create():
    """Deriving must not lose an index, because losing one is invisible.

    A missing index does not fail a query; it degrades a MERGE to a full label scan.
    Nothing raises, no test goes red, the graph just gets slower as it grows -- so
    this is asserted rather than assumed. A superset is fine and expected; a subset
    is a silent performance regression.
    """
    derived = set(index_specs())

    missing = HISTORICAL_INDEXES - derived
    assert missing == set(), (
        f"deriving indexes from declared identity dropped {sorted(missing)}, which "
        "the engine used to create. A dropped index degrades MERGE to a label scan "
        "without failing anything."
    )


def test_every_described_label_is_indexed_on_the_property_upsert_merges_on():
    """An index on the wrong property is worse than none: it costs writes and is unused.

    `upsert_node` MERGEs on the caller's key props, which for every ontology writer is
    the label's declared `identity`. Indexing anything else means the MERGE still
    scans. This passes only because the index property IS the identity by
    construction -- the test exists so a future hand-maintained exception cannot
    reintroduce the drift.
    """
    described = entity_types()

    for label, identity in index_specs():
        assert label in described, f"index for undescribed label {label!r}"
        assert identity == described[label].identity, (
            f"index on {label}.{identity} but its declared identity is "
            f"{described[label].identity!r}"
        )


def test_the_engine_names_no_domain_label_in_its_index_setup():
    """The actual split condition, checked in the source rather than in behaviour.

    `graph.py` ships in `noesis-core`, which owns no ontologies. A derived index set
    is only a real fix if the literal labels are gone from the module -- an engine
    that still spells `Decision` cannot be packaged without the decisions ontology,
    no matter where it gets the rest of its list from.
    """
    # Docstrings are excluded deliberately: `attach_source` illustrates its
    # `MATCH (p:Project ...)` argument in prose, and an example of what a CALLER
    # passes is not the engine depending on that label. Only executable lines and
    # string literals the module actually runs are checked, which is where a
    # hardcoded label would do harm.
    source = _code_only(pathlib.Path("graph.py").read_text())
    domain_labels = {
        label
        for key, kb in KNOWLEDGE_BASES.items()
        for label in kb.labels
        if key not in ("tools", "orchestration", "provenance")
    }

    hardcoded = sorted(
        label for label in domain_labels
        if re.search(rf"[(:]\s*{label}\b", source)
    )

    assert hardcoded == [], (
        f"graph.py names domain labels {hardcoded}; the engine must reach them "
        "through the registry so a product that omits their package omits them too."
    )


def test_ensure_indexes_asks_the_registry_rather_than_listing_labels():
    """Pins the mechanism, not just the absence of names.

    Without this, someone could satisfy the test above by moving the same literal
    list into a module-level constant in another engine file and importing it.
    """
    source = pathlib.Path("graph.py").read_text()
    body = source[source.index("def ensure_indexes"):]
    body = body[:body.index("pass  # index may already exist")]

    assert "index_specs" in body, "ensure_indexes must derive its labels"
    assert "CREATE INDEX FOR (t:" not in body, "a literal label list came back"


@pytest.mark.parametrize("dropped", ["semantic_layer", "strategy"])
def test_uninstalling_an_ontology_removes_exactly_its_indexes(dropped):
    """The product-shaping guarantee: indexes are a strict subset, never a reshuffle.

    This is the behaviour the hardcoded list could not provide. An enterprise product
    installs `semantic_layer` and indexes Table/Column/Metric; a personal product does
    not install it and must not create them -- while keeping every index belonging to
    the packages it DID install, on the same properties.
    """
    import knowledge_bases.registry as registry

    full = dict(index_specs())
    owned = set(KNOWLEDGE_BASES[dropped].labels)

    original = registry._PACKAGES
    try:
        registry._PACKAGES = tuple(p for p in original if p.key != dropped)
        entity_types.cache_clear()
        reduced = dict(index_specs())
    finally:
        registry._PACKAGES = original
        entity_types.cache_clear()

    gone = set(full) - set(reduced)
    assert gone <= owned, (
        f"dropping {dropped!r} removed indexes it does not own: {sorted(gone - owned)}"
    )
    assert gone, f"dropping {dropped!r} removed no index, so its labels were not indexed"
    for label, identity in reduced.items():
        assert full[label] == identity, (
            f"{label} changed index property from {full[label]!r} to {identity!r} "
            "when an unrelated package was uninstalled"
        )

def test_dropping_the_delivery_ontology_is_refused_when_a_connector_contributes_to_person():
    """The same guarantee, one level up.

    `claims`'s Microsoft Graph connector contributes `email` and `teams_id` to
    `Person`. `Person` is owned by the delivery ontology, so dropping
    `project_management` from a profile that installs `claims` leaves a property
    describing a node no installed package can write. Composition refuses it
    rather than writing the index for a label that has no owner.

    This is the product-shaping rule with a dependency made visible: a connector
    narrows what you may uninstall.
    """
    import knowledge_bases.registry as registry

    original = registry._PACKAGES
    try:
        registry._PACKAGES = tuple(p for p in original if p.key != "project_management")
        registry.entity_types.cache_clear()
        registry.property_contributions.cache_clear()
        with pytest.raises(ValueError, match="which no installed package declares"):
            registry.entity_types()
    finally:
        registry._PACKAGES = original
        registry.entity_types.cache_clear()
        registry.property_contributions.cache_clear()

