"""`schema.py` is a compatibility facade, and nothing may depend on it.

The facade re-exports domain models under flat names for callers written before the
knowledge bases were split. It is kept -- deleting it would break unmigrated
external callers for no benefit -- but it is the single worst offender in the
planned engine/ontology split: a root-level engine module that imports five
domain packages purely to alias their names. If `noesis-core` is to own zero
ontologies, no code that ships with the engine can route through here.

`docs/operations/split_state.md` measured 26 such core-to-domain imports, 5 of them
in this file, and named deleting them the cheapest first move.
"""
from __future__ import annotations

import pathlib
import re

from knowledge_bases.business_dictionary.models import BusinessTerm as OwnedBusinessTerm
from knowledge_bases.provenance.models import SourceNode as OwnedSourceNode
from knowledge_bases.semantic_layer.models import Metric as OwnedMetric
from schema import BusinessTerm, Metric, SourceNode

# Imports the facade by design: this file tests it.
TESTS_THE_FACADE = {"tests/test_schema_compatibility.py"}


def test_flat_schema_reexports_owned_models():
    assert Metric is OwnedMetric
    assert BusinessTerm is OwnedBusinessTerm
    assert SourceNode is OwnedSourceNode


def test_no_shipped_module_imports_the_facade():
    """A ratchet, not a cleanup. Every caller has been migrated to the owning
    package, so this holds today; the test exists so a new import does not quietly
    re-add a dependency that has to be severed before the engine can be extracted.
    """
    root = pathlib.Path(".")
    pattern = re.compile(r"^\s*(?:from schema import|import schema\b)", re.M)
    offenders = []

    for path in root.rglob("*.py"):
        text = str(path)
        if ".venv" in text or "__pycache__" in text:
            continue
        if str(path.as_posix()) in TESTS_THE_FACADE:
            continue
        if path.name == "schema.py":
            continue
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(path.as_posix())

    assert offenders == [], (
        "these modules import the compatibility facade instead of the owning "
        f"package, which blocks the engine/ontology split: {offenders}"
    )
