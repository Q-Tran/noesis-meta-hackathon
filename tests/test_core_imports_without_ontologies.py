"""The engine must import with every optional ontology absent.

This is the precondition for extracting `noesis-core`, and it is a different claim
from the composition tests. Those prove a narrower PACKAGE SET composes; this proves
the engine's MODULES load when the ontology code is not installed at all -- which is
what `pip install noesis-core` without any ontology distribution actually looks like.

A composition test cannot catch a module-scope `from knowledge_bases.methodology...`,
because by the time it runs, the enterprise profile has already imported everything.
"""
from __future__ import annotations

import dataclasses
import importlib
import sys

import pytest

# Everything bound for `noesis-core`: the engine's root modules plus the control-plane
# knowledge bases. `schema.py` is deliberately ABSENT -- see the test at the bottom.
CORE_MODULES = (
    "graph", "config", "llm", "vector_store", "query", "agent", "worker",
    "recompute", "review", "migrations", "extract",
    "knowledge_bases.base", "knowledge_bases.registry", "knowledge_bases.mcp",
    "knowledge_bases.common", "knowledge_bases.ontology", "knowledge_bases.boundary",
    "knowledge_bases.product", "knowledge_bases.triplets", "knowledge_bases.cascade",
    "knowledge_bases.ingestion", "knowledge_bases.ingestion_service",
    "knowledge_bases.conformance",
    "knowledge_bases.tools.mcp_tools", "knowledge_bases.orchestration.service",
    "knowledge_bases.orchestration.commands", "knowledge_bases.provenance.package",
)

OPTIONAL_ONTOLOGIES = (
    "semantic_layer", "business_dictionary", "epistemology", "strategy", "architecture",
    "project_management", "methodology", "development", "tasks",
)

CORE_ONLY_PACKAGES = ("tools", "orchestration", "provenance")


class _OntologyBlocker:
    """Raises on any import of an optional ontology, simulating it not being installed.

    A `MetaPathFinder` rather than monkeypatching `sys.modules`, because the failure
    being hunted is a module-scope import executed during module load, and only a
    finder sees that.
    """

    def find_spec(self, name, path=None, target=None):
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "knowledge_bases":
            if parts[1] in OPTIONAL_ONTOLOGIES:
                raise ImportError(f"not installed in noesis-core: {name}")
        return None


def _import_core_in_isolation(modules):
    """Import each module in a FRESH INTERPRETER with the optional ontologies blocked.

    A subprocess per module, which is not paranoia. Doing this in-process leaves
    already-imported ontology modules in `sys.modules`, so a module that imports one
    at module scope finds it cached and succeeds -- the test then reports clean while
    the coupling is still there. Purging `sys.modules` by hand does not fix it either:
    partially-initialised parents and `lru_cache`d registry state survive, and the
    results changed depending on which test ran first.

    The profile is rebound before any Noesis import, because `registry` binds
    `_PACKAGES` at import time. If it loads under the enterprise profile first,
    resolving those seeds legitimately imports the ontologies the profile says are
    installed, and blocking them reports coupling that does not exist.
    """
    import json
    import subprocess

    program = f"""
import dataclasses, importlib, json, sys
import knowledge_bases.product as product
product.PROFILE = dataclasses.replace(
    product.PROFILE, packages={CORE_ONLY_PACKAGES!r}, workflow_profile=None)

OPTIONAL = {OPTIONAL_ONTOLOGIES!r}

class Blocker:
    def find_spec(self, name, path=None, target=None):
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "knowledge_bases" and parts[1] in OPTIONAL:
            raise ImportError("not installed in noesis-core: " + name)
        return None

sys.meta_path.insert(0, Blocker())

failures = {{}}
for name in {list(modules)!r}:
    try:
        importlib.import_module(name)
    except BaseException as exc:
        failures[name] = type(exc).__name__ + ": " + str(exc)
print(json.dumps(failures))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 0, f"probe crashed: {result.stderr[-2000:]}"
    return json.loads(result.stdout.strip().split("\n")[-1])


def test_every_core_module_imports_with_no_ontology_installed():
    """The condition for shipping `noesis-core` as an ontology-free distribution.

    When this last ran as an audit, 6 modules failed without `methodology` and 7
    without `claims`. All are fixed -- by moving schema builders to their owners,
    turning `MANAGED_VECTOR_TYPES` into a package-declared flag, moving versioned-kind
    composition into the registry, and routing the rest through
    `registry.capability()`.
    """
    failures = _import_core_in_isolation(CORE_MODULES)

    assert failures == {}, (
        "these core modules cannot load without an optional ontology installed, so "
        f"noesis-core cannot ship without it: {failures}"
    )


def test_the_blocker_actually_blocks():
    """Guards the guard. A finder that quietly stopped raising would make the test
    above vacuous, and it would still pass -- the worst possible failure mode for a
    test whose entire value is a negative result."""
    failures = _import_core_in_isolation(("knowledge_bases.methodology.kinds",))

    assert "knowledge_bases.methodology.kinds" in failures
    assert "not installed in noesis-core" in failures["knowledge_bases.methodology.kinds"]


def test_the_compatibility_facade_is_not_part_of_core():
    """`schema.py` re-exports five ontologies' models; that IS its job.

    It cannot ship in an ontology-free core, and it should not: it exists so
    pre-split external callers keep working. It therefore belongs to the enterprise
    product, which installs everything it names.

    Asserted rather than left implicit so nobody "fixes" the import failure by
    gutting the facade, and so the split plan records where the module goes.
    """
    failures = _import_core_in_isolation(("schema",))

    assert "schema" in failures, (
        "schema.py now imports without the ontologies, which means the facade no "
        "longer re-exports their models and pre-split callers are broken"
    )
