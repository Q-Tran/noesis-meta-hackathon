"""A profile that omits a package must survive that package being *absent*.

`registry.compose()` happily builds a product without `methodology` or `claims`,
and `tests/test_product_profile.py` checks that the composed data is a clean subset.
That is the data layer, and it passes. The import layer was never tested, and it
fails: `docs/operations/split_state.md` measured 6 control-plane modules failing to
import without `methodology` and 7 without `claims`, including `schema.py`.

The distinction matters because pip-not-installed is what a real product
distribution looks like. A profile can therefore declare a shippable product today
that dies at import, and nothing in the suite noticed -- Phase 1's completion
criterion only ever exercised composition.

These tests do not assert the split is finished. They MEASURE how far it is from
finished and fail only if it regresses, so the number has to come down rather than
drift up while the data-layer tests stay green.
"""
from __future__ import annotations

import pathlib
import re

import pytest

# Shipped (non-test) modules with a static import of each optional package,
# measured from this tree. This counts IMPORT SITES, which is a different and
# larger number than the 6/7 MODULES THAT FAIL TO IMPORT reported in
# split_state.md -- an import site only breaks the importer, while a failing module
# breaks everything downstream of it. Both are worth knowing; this is the one a
# grep can police.
#
# Lower is better. These are ratchets: drive them DOWN as couplings are severed,
# and never raise one to make a new import pass.
# Every ontology a product may omit. The control plane (`tools`, `orchestration`,
# `provenance`) is absent on purpose: it ships with the engine.
OPTIONAL_ONTOLOGY_KEYS = (
    "semantic_layer", "business_dictionary", "epistemology", "strategy", "architecture",
    "project_management", "methodology", "development", "tasks",
)

KNOWN_IMPORT_COUPLING = {
    # `scripts/seed_self.py` and `scripts/self_recompute.py`. Both are self-modeling
    # scripts that record Noesis's own architecture as versioned methodology
    # entities, so they are meaningless without it -- they ship with the ontology
    # bundle, not with the engine. This is a floor, not a debt.
    "methodology": 2,
    # `schema.py`, the compatibility facade, whose whole purpose is re-exporting the
    # ontologies' models. It ships with the enterprise product, which installs them
    # all (tests/test_core_imports_without_ontologies.py asserts it is not core).
    "epistemology": 1,
}


def _modules_importing(package: str) -> set[str]:
    """Shipped modules with a static import of `knowledge_bases.<package>`."""
    pattern = re.compile(
        rf"^\s*(?:from knowledge_bases\.{package}[\s.]|"
        rf"import knowledge_bases\.{package}\b)",
        re.M,
    )
    found = set()
    for path in pathlib.Path(".").rglob("*.py"):
        text = str(path)
        # `tests/` is excluded: a test importing the package it tests is correct,
        # not coupling. The earlier filter checked for "/tests/", which never
        # matches a relative path like "tests/test_x.py" and so counted every test.
        if ".venv" in text or "__pycache__" in text:
            continue
        if path.parts and path.parts[0] == "tests":
            continue
        if path.as_posix().startswith(f"knowledge_bases/{package}/"):
            continue  # a package importing itself is not coupling
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            found.add(path.as_posix())
    return found


@pytest.mark.parametrize("package", sorted(KNOWN_IMPORT_COUPLING))
def test_import_coupling_to_an_optional_package_does_not_grow(package):
    """The ratchet. A new control-plane import of an optional ontology package is
    a new reason the engine cannot be extracted, and it is invisible to every
    composition test."""
    offenders = _modules_importing(package)

    assert len(offenders) <= KNOWN_IMPORT_COUPLING[package], (
        f"{len(offenders)} modules now import {package!r} statically, up from the "
        f"recorded {KNOWN_IMPORT_COUPLING[package]}. Each one blocks shipping a "
        f"product without {package!r}: {sorted(offenders)}"
    )


def test_no_engine_module_imports_an_optional_ontology_package():
    """The condition for extracting `noesis-core`, and it now holds.

    This test previously asserted the opposite -- that `tools` and `orchestration`
    DID import optional ontologies -- because they held 20 of the 26 measured sites,
    bidirectionally, and naming the problem was more useful than pretending it was
    absent. All 20 are severed: schema builders moved to the package that owns them,
    `MANAGED_VECTOR_TYPES` became a package-declared flag, versioned-kind composition
    moved to the registry, and the rest resolve through `registry.capability()`.

    Inverted rather than deleted, because the useful assertion is no longer "where is
    the problem" but "the problem has not come back". Every remaining site is a
    script or the compatibility facade, none of which ship inside the engine.
    """
    offenders = {
        path
        for package in KNOWN_IMPORT_COUPLING
        for path in _modules_importing(package)
        if path.startswith("knowledge_bases/")
    }

    assert offenders == set(), (
        "engine code under knowledge_bases/ imports an optional ontology package "
        f"again, which blocks extracting noesis-core: {sorted(offenders)}. Declare "
        "the dependency with `capabilities_seed` and resolve it through "
        "`registry.capability()` instead of importing."
    )


def test_no_composition_site_reloads_every_declared_package():
    """The bug this prevents made a narrow product silently behave like a full one.

    `ontology_packages()` LOADS a named list and defaults to every DECLARED package.
    Four shipped composition sites called it with no argument -- migrations, the tool
    catalog, the expert catalog, and versioned kinds -- so each answered "what could
    be installed" rather than "what is". A product profile could drop `development`
    and still get its migrations, its tools, and its expert.

    Found by a migration test that asserted uninstalling a package removed its
    migrations, and could not be made to pass. Checked in the source because the
    behaviour is only observable through a profile switch, which most of these sites
    have no test for.
    """
    import pathlib
    import re

    offenders = []
    for path in pathlib.Path(".").rglob("*.py"):
        parts = path.parts
        # `.venv` holds third-party source, some of it deliberately not UTF-8 (a
        # big5-encoded codec fixture), and none of it ours to police.
        if not parts or parts[0] in (".venv", "venv", ".git", "build"):
            continue
        if "__pycache__" in str(path) or parts[0] == "tests" or path.name == "conftest.py":
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        for number, line in enumerate(source.split("\n"), 1):
            if not re.search(r"\bontology_packages\(\s*\)", line):
                continue
            # `registry.py` is where the installed set is BOUND, so it is the one
            # legitimate caller; prose mentioning the function is not a call site.
            if path.name == "registry.py":
                continue
            if line.lstrip().startswith(("#", "*", '"', "'")) or "`" in line:
                continue
            offenders.append(f"{path}:{number}")

    assert offenders == [], (
        f"these sites compose from every DECLARED package rather than the installed "
        f"set: {offenders}. Call `installed_packages()` instead; passing an explicit "
        "key tuple to `ontology_packages(...)` is fine."
    )


def test_every_engine_script_runs_on_a_product_without_the_versioning_ontology():
    """A script that ships with the engine must not require an optional ontology.

    `scripts/create_indexes.py` is the case that motivated this. It creates graph
    indexes, applies migrations and builds vector collections -- none of which is
    about versioning -- but imported `methodology.search` at module scope, so it
    crashed on IMPORT for a product without that ontology. The graph could not be
    initialised at all.

    Reindexing is now resolved through `registry.capability("versioning.reindex")`.
    Absent means there are no versioned entities to reindex, which is a correct
    outcome, so the script reports and continues.

    The scripts that legitimately belong to the ontology bundle are listed explicitly
    rather than skipped by pattern: `seed_self` and `self_recompute` record Noesis's
    own architecture AS versioned methodology entities, so they are meaningless
    without it and ship alongside it.
    """
    import pathlib
    import re

    ONTOLOGY_BUNDLE_SCRIPTS = {
        "seed_self.py", "self_recompute.py",       # self-modeling via methodology
        "dev_catalog.py",                          # indexes code into `development`
        "ingest_semantic_models.py",               # semantic_layer ingestion
        "retry_metric_naming.py", "retry_metric_naming_parallel.py",
    }
    optional = "|".join(OPTIONAL_ONTOLOGY_KEYS)
    pattern = re.compile(rf"^\s*(?:from|import)\s+knowledge_bases\.({optional})\b", re.M)

    offenders = {}
    for path in sorted(pathlib.Path("scripts").glob("*.py")):
        if path.name in ONTOLOGY_BUNDLE_SCRIPTS:
            continue
        found = sorted(set(pattern.findall(path.read_text())))
        if found:
            offenders[path.name] = found

    assert offenders == {}, (
        f"these engine scripts import an optional ontology at module scope, so they "
        f"cannot run on a product without it: {offenders}. Resolve the dependency "
        "with `registry.capability()` and degrade when it is absent."
    )
