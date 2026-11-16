"""The conformance kit run against the REAL installed packages.

Split from `test_conformance_kit.py`, which tests the kit against in-memory fixture
packages -- each deliberately wrong in one way -- and therefore needs no ontology
installed at all. These five need the actual vocabularies, so they travel with them.

Both halves matter and neither substitutes for the other. A rule that fires only on
synthetic packages is probably wrong about the real contract; a rule that fires on a
shipped package has found either a defect or its own bug. And a rule can pass both
kinds of fixture test while silently examining nothing, which is what
`test_the_source_rules_actually_see_a_source_tree_for_every_real_package` exists to
catch.
"""
from __future__ import annotations

import pytest

from knowledge_bases.base import KnowledgeBase, OntologyPackage
from knowledge_bases.conformance import (
    ALL_RULES,
    PackageContext,
    Severity,
    check_package,
    check_packages,
    installed_packages,
)


@pytest.mark.parametrize("key", [package.key for package in installed_packages()])
def test_every_installed_ontology_package_is_conformant(key):
    """The kit run against reality. Two reasons this is not redundant with the
    fixture tests: a rule that fires only on synthetic packages is probably
    wrong about the real contract, and a rule that fires on a shipped package has
    found either a defect or its own bug -- both worth failing over.

    KNOWN REAL FINDINGS, deliberately recorded here rather than by weakening the
    rules that found them (none of these files are this kit's to fix):

    - `business_dictionary` imports `semantic_layer.resolution`, the one
      cross-ontology import `docs/operations/split_state.md` measured.
    - (`orchestration` -> `methodology` was here. Its six methodology write
      commands moved to `methodology/write_commands.py`, which is where a
      package's own writes belong; the engine no longer imports the ontology.)

    They are listed as expectations so the suite is green on a tree whose defects
    are known, and so FIXING one fails this test and forces the list to shrink.
    That ratchet already paid for itself once: `tools` was on this list, the
    core-to-domain sever landed while the kit was being written, and this test
    failed until the entry was removed.
    """
    known: dict[str, set[str]] = {
        # `semantic_layer.Table` was here: its search type read
        # `identity_property="id"` while every writer MERGEs on `name`, so Table
        # was indexed, searchable, and matched nothing. Fixed at the declaration
        # (`identity_property="name"`), which is what made this entry go away --
        # this ratchet is how the fix was noticed.
        #
        # `business_dictionary` was here for importing `semantic_layer.resolution`
        # -- the one cross-ontology import the split audit measured. It is gone:
        # `semantic_layer` declares the `resolution.resolve` capability and
        # `business_dictionary` calls it by name, with a local fallback when the
        # capability is absent. The enterprise tree is now clean of cross-ontology
        # imports, and this ratchet is empty.
    }

    report = check_package(key)
    fired = {finding.rule for finding in report.errors}
    expected = known.get(key, set())

    assert fired - expected == set(), (
        f"{key} has a NEW conformance error:\n\n"
        + report.format(Severity.ERROR)
    )
    assert expected - fired == set(), (
        f"{key} no longer violates {sorted(expected - fired)} -- if that was "
        "fixed, remove it from this test's `known` map so the ratchet tightens"
    )

def test_the_known_violations_are_only_the_ones_we_have_accounted_for():
    """A whole-tree count, so a defect cannot be introduced in one package while
    another is fixed and the per-package tests stay green. ONE error across eleven
    packages, down from three: Table's search identity was fixed, and
    orchestration's import of methodology was severed by moving that package's write
    commands into it. This number may only go DOWN."""
    reports = check_packages(installed_packages())
    total = sum(len(report.errors) for report in reports.values())

    assert total <= 1, (
        "conformance errors across the installed packages went UP to "
        f"{total}:\n\n"
        + "\n\n".join(
            report.format(Severity.ERROR)
            for report in reports.values()
            if report.errors
        )
    )

def test_running_the_kit_over_the_whole_product_checks_every_package():
    """Guards the fan-out itself: `check_packages` must actually check each
    package with the others as peers, not silently check one or none."""
    reports = check_packages(installed_packages())

    assert len(reports) == len(installed_packages()) == 11
    assert all(not report.is_vacuous for report in reports.values())
    assert all(report.rules_run == tuple(r.__name__ for r in ALL_RULES)
               for report in reports.values())

def test_the_source_rules_actually_see_a_source_tree_for_every_real_package():
    """A rule that runs against an empty tree passes while checking nothing.

    `PackageContext.source_root` is optional, and the source rules -- sibling-ontology
    imports, reaching into the orchestrator's private helpers -- SKIP silently when it is
    None. That is right for a synthetic fixture package, which has no source, and
    catastrophic for a real one: the rules run, find nothing, and the report is green.

    This is not hypothetical. `source_root` was guessed as "the directory beside the
    conformance kit", which is correct in a single tree and wrong for every separately
    distributed ontology. Run in the ontologies monorepo, the kit reported that
    `business_dictionary` no longer imports `semantic_layer` -- welcome news, except the
    import is still on line 5 of its `queries.py`. The rule had gone blind.

    `is_vacuous` catches a report with NO RULES. Nothing caught rules that run and see an
    empty tree, which is a strictly harder failure to notice because the rule count looks
    right.
    """
    for package in installed_packages():
        context = PackageContext.load(package.key)

        assert context.source_root is not None, (
            f"the conformance kit found no source tree for {package.key!r}, so every "
            "source-level rule silently checked nothing for it"
        )
        assert (context.source_root / "package.py").exists(), (
            f"{package.key!r}'s source root {context.source_root} holds no package.py, "
            "so it is not that package's directory"
        )

def test_a_sibling_import_is_still_detected_where_one_exists(tmp_path):
    """Proves the source rules are wired to reality, not just to a non-empty directory.

    This was pinned to `business_dictionary/queries.py` importing
    `semantic_layer.resolution` -- the one cross-ontology import the split audit
    measured. That import is gone: `semantic_layer` declares the
    `resolution.resolve` capability and `business_dictionary` calls it by name,
    so the real tree is clean and the ratchet is empty. The rule must still
    FIRE, because a rule that reports clean on a clean tree by looking in the
    wrong place is the failure this suite cares most about. It now fires on a
    synthetic violation: a package whose source imports a sibling ontology.
    """
    # `sibling_package_keys` counts a directory as an ontology package when it
    # holds `package.py`, so the fake sibling needs its marker file.
    (tmp_path / "semantic_layer").mkdir()
    (tmp_path / "semantic_layer" / "package.py").write_text("")
    root = tmp_path / "sibling"
    root.mkdir()
    (root / "queries.py").write_text(
        "from knowledge_bases.semantic_layer.resolution import x\n"
    )
    pkg = OntologyPackage(
        key="sibling",
        knowledge_bases=(
            KnowledgeBase(
                key="sibling", name="Sibling", description="synthetic",
                labels=("SiblingLabel",), relationships=(),
            ),
        ),
    )
    report = check_package(pkg, peers=(), source_root=root)

    fired = {finding.rule for finding in report.errors}
    assert "the_package_does_not_import_a_sibling_ontology_package" in fired, (
        "the sibling-import rule went blind: it no longer fires on a package "
        "that imports a sibling ontology at its own source root"
    )
