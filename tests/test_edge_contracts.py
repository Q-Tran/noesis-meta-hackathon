"""Edge contracts: cross-ontology edges are declared, validated, and pinned.

The bridges README table used to be a documented convention; `EdgeContract`
(`knowledge_bases/base.py`) makes it composition-time declarations resolved by
`registry.edge_contracts()`. These tests pin the three properties that make
the mechanism worth having: the enterprise profile composes every bridge row,
the README table IS the contracts (generated content that cannot drift), and
the two packaging mistakes composition refuses -- two describers for one edge,
and a describer that is not a declarer -- raise.

The narrowed-product case is deliberately NOT a raise: a contract whose
endpoint labels no installed package supplies is inert, and the personal
product's dead edges (`MEASURED_BY`, `REALIZES`) are the precedent for
accepting them.
"""
from __future__ import annotations

import pathlib

import pytest

from knowledge_bases.base import EdgeContract, KnowledgeBase, OntologyPackage
from knowledge_bases.product import PROFILE
from knowledge_bases.registry import (
    compose,
    edge_contracts,
    inert_edge_contracts,
    ontology_packages,
)

BRIDGE_ROWS = {
    "MAPS_TO", "SUPPORTS", "CHALLENGES", "ASSERTS",
    "IMPLEMENTS", "FOUND_IN",
}


def _package(key, labels, relationships, contracts):
    kb = KnowledgeBase(
        key=key, name=key.title(), description=f"{key} test package",
        labels=labels, relationships=relationships,
    )
    return OntologyPackage(key=key, knowledge_bases=(kb,), edge_contracts=contracts)


def test_the_enterprise_profile_composes_every_bridge_row():
    active = {contract.relationship for contract in edge_contracts()}
    assert active == BRIDGE_ROWS
    # The enterprise profile installs every package, so no bridge contract
    # is inert here; a row that goes inert means a package stopped supplying
    # an endpoint the table promises.
    assert inert_edge_contracts() == ()


def test_the_bridges_readme_table_is_the_contracts():
    """The README table is generated content: parse it and compare against
    the declared contracts. A row that drifts fails the suite, the same way
    the README diagram test pins the architecture picture."""
    readme = (
        pathlib.Path("knowledge_bases/bridges/README.md")
        .read_text(encoding="utf-8")
    )
    table = {}
    for line in readme.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        table[cells[0].strip("`")] = (
            sorted(cell.strip() for cell in cells[1].split(",")),
            sorted(cell.strip() for cell in cells[2].split(",")),
            sorted(cell.strip() for cell in cells[3].split(",")),
            cells[4],
        )
    declared = {}
    for package in ontology_packages(PROFILE.packages):
        for contract in package.edge_contracts:
            declared[contract.relationship] = (
                sorted(contract.from_labels),
                sorted(contract.to_labels),
                sorted(contract.writers),
                package.key,
            )
    assert table == declared


def test_two_describers_for_one_edge_is_a_packaging_mistake():
    """The `DEPENDS_ON` defect's root cause, made a raise: two packages
    cannot both describe one edge, exactly as two cannot own one label."""
    a = _package("a", ("Alpha",), ("SHARED",), (
        EdgeContract("SHARED", ("Alpha",), ("Beta",), ("a",), "a describes it"),
    ))
    b = _package("b", ("Beta",), ("SHARED",), (
        EdgeContract("SHARED", ("Beta",), ("Alpha",), ("b",), "b describes it"),
    ))
    with pytest.raises(ValueError, match="two edge contracts"):
        compose((a, b))


def test_the_describer_must_be_a_declarer():
    bad = _package("a", ("Alpha",), ("DECLARED",), (
        EdgeContract("UNDCLARED", ("Alpha",), ("Alpha",), ("a",), "not declared"),
    ))
    with pytest.raises(ValueError, match="does not declare"):
        compose((bad,))


def test_a_narrower_product_makes_contracts_inert_not_illegal():
    """A product that omits `semantic_layer` keeps the SUPPORTS contract
    (the enterprise product needs it) and the edge goes inert: Metric is
    not installed, so `edge_contracts()` excludes it and
        `inert_edge_contracts()` reports it. Composition succeeds."""
    packages = ontology_packages(("epistemology", "strategy", "provenance"))
    composed = compose(packages)
    assert "epistemology" in composed
    active = {contract.relationship for contract in edge_contracts()}
    # The module-level accessors read the installed profile, so check the
    # narrowed composition through the contract data itself.
    narrowed_active = tuple(
        contract
        for package in packages
        for contract in package.edge_contracts
        if (set(contract.from_labels) | set(contract.to_labels)) - {"*"}
        <= {label for kb in composed.values() for label in kb.labels}
    )
    active_keys = {contract.relationship for contract in narrowed_active}
    # ASSERTS (SourceNode->Claim) survives; SUPPORTS needs
    # IMPLEMENTS needs Project/Capability. FOUND_IN now names all three of
    # its real targets (SourceNode, Chunk -- extract.py -- and RoutingTrace,
    # the conversation turn that substantiates user claims); that synthetic
    # composition installs no orchestration, so the whole spine goes inert
    # there. The shipped product composes orchestration and keeps it; a
    # product that ships provenance without the control plane would have to
    # re-take this trade (MEASURED_BY/REALIZES are the inert precedent).
    assert "ASSERTS" in active_keys
    assert "FOUND_IN" not in active_keys
    assert "SUPPORTS" not in active_keys
