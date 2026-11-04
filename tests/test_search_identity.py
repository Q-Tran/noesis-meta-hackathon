"""A searchable label must be keyed on the property its writers actually set.

`hybrid_search` reads each candidate's identity from `SearchType.identity_property`
and drops the row when that property is null -- no error, no log, no empty-result
signal distinguishable from "nothing matched". So a label whose declared search
identity disagrees with what its writers key on is indexed, searchable, and returns
nothing at all.

That was live for `Table`: its `SearchType` kept the default `identity_property="id"`
while `import_powerbi.py` MERGEs `(:Table {name: ...})`, `graph.py` indexes
`(:Table) ON (t.name)`, and `EntityType("Table")` declares `name` as its identity and
does not declare `id` at all. Found by the conformance kit, not by any test here,
because every existing test asserted against the declaration rather than against
agreement between the declaration and the writers.
"""
from __future__ import annotations

import pytest

from knowledge_bases.registry import entity_types, search_types


def test_table_search_is_keyed_on_the_property_powerbi_writes():
    """The specific regression. Pinned by name because `Table` is the label a user
    searches when asking what data exists, so its silence is maximally misleading."""
    assert search_types()["semantic_layer.Table"].identity_property == "name"


@pytest.mark.parametrize(
    "qualified", sorted(search_types())
)
def test_a_searchable_labels_identity_property_is_one_it_declares(qualified):
    """The general rule, so the next label cannot repeat it.

    A search type may read an identity property only if the label declares it.
    Reading an undeclared property is not a type error and not a query error -- it
    yields None, and None means "drop this result", which is why this failed silently
    for a whole package.
    """
    config = search_types()[qualified]
    entity = entity_types().get(config.label)
    if entity is None:
        pytest.skip(f"{config.label} is not a described entity type")

    declared = {
        getattr(prop, "name", prop) for prop in entity.properties
    }
    if not declared:
        pytest.skip(f"{config.label} declares no properties to check against")

    assert config.identity_property in declared, (
        f"{qualified} reads identity from {config.identity_property!r}, which "
        f"{config.label} does not declare. hybrid_search will read None and drop "
        f"every candidate, so this label will be searchable and match nothing. "
        f"Declared: {sorted(declared)}"
    )


def test_the_identity_property_agrees_with_the_labels_declared_identity():
    """Search identity and write identity must be the same property.

    They answer the same question -- "which property names this node?" -- so a
    disagreement means one of them is wrong, and whichever it is, search and
    ingestion are looking at different nodes.
    """
    mismatched = []
    for qualified, config in sorted(search_types().items()):
        entity = entity_types().get(config.label)
        if entity is None:
            continue
        identity = entity.identity
        parts = (identity,) if isinstance(identity, str) else tuple(identity or ())
        if not parts:
            continue
        if config.identity_property not in parts:
            mismatched.append(
                f"{qualified}: search reads {config.identity_property!r}, "
                f"EntityType identity is {parts}"
            )

    assert mismatched == [], (
        "these labels are searched by a different property than they are "
        "identified by:\n  " + "\n  ".join(mismatched)
    )
