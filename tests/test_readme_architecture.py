"""The README's architecture diagram must stay true.

A diagram is documentation that looks authoritative, which makes a stale one
worse than none: it is confidently wrong. Writing this section surfaced three
references that were already wrong -- `ExpertOrchestrator` is in `service.py`
not `experts.py`, `review.py` is at the repo root not under `orchestration/`,
and `MAX_DEPTH=3` is the neighborhood *traversal* bound in `neighborhood.py`,
not a limit on the expert tool loop.

These tests pin the claims that can rot: file paths, symbol names, and the
counts printed in the diagram.
"""
from __future__ import annotations

import pathlib
import re

import pytest

README = pathlib.Path("README.md")


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text()


# Every (file, symbol) the architecture section names. A symbol of None means
# only the file's existence is claimed.
_REFERENCES = [
    ("knowledge_bases/mcp.py", "dispatch_tool"),
    ("knowledge_bases/mcp.py", "_require_schema_fields"),
    ("knowledge_bases/tools/tracing.py", "record_tool_span"),
    ("knowledge_bases/tools/repository.py", "load_tools"),
    ("knowledge_bases/tools/resolver.py", "invoke_tool"),
    ("knowledge_bases/tools/trace_store.py", "insert_span"),
    ("knowledge_bases/tools/auto_improve.py", "trigger_in_background"),
    ("knowledge_bases/tools/self_improvement.py", None),
    ("knowledge_bases/orchestration/routing.py", "score_experts"),
    ("knowledge_bases/orchestration/service.py", "ExpertOrchestrator"),
    ("knowledge_bases/orchestration/commands.py", "WRITE_COMMANDS"),
    ("knowledge_bases/orchestration/experts_catalog.py", "bootstrap_experts"),
    ("knowledge_bases/orchestration/memory.py", None),
    ("knowledge_bases/orchestration/neighborhood.py", "MAX_DEPTH"),
    ("knowledge_bases/boundary.py", "require_runtime_mutable"),
    ("graph.py", "Graph"),
    ("recompute.py", None),
    ("review.py", None),
]


@pytest.mark.parametrize("path,symbol", _REFERENCES)
def test_architecture_references_resolve(path, symbol):
    file = pathlib.Path(path)

    assert file.exists(), f"README names {path}, which does not exist"
    if symbol is not None:
        assert symbol in file.read_text(), (
            f"README names {symbol} in {path}, which does not define it"
        )


def test_the_diagram_names_files_that_all_exist(readme):
    """Catch a path added to the diagram later without a matching reference
    above -- the diagram is the part most likely to be edited by hand."""
    start = readme.index("## Architecture")
    section = readme[start : readme.index("## Setup", start)]

    # Any `path/to/file.py` mentioned anywhere in the section. Names are often
    # written relative to `knowledge_bases/` (e.g. `tools/tracing.py`) or bare
    # (`commands.py`), so a reference counts as resolved if it exists under any
    # of those roots.
    roots = [
        pathlib.Path("."),
        pathlib.Path("knowledge_bases"),
        pathlib.Path("scripts"),
        *sorted(p for p in pathlib.Path("knowledge_bases").iterdir() if p.is_dir()),
        # Post-split, domain package sources live in the sibling `noesis-ontologies`
        # checkout; the import path `knowledge_bases.<key>` is the contract, and the
        # diagram names files by that contract. Resolve against the checkout.
        pathlib.Path("../noesis-ontologies"),
        pathlib.Path("../noesis-ontologies/packages"),
        *[pathlib.Path("../noesis-ontologies/packages") / key
          for key in ("semantic_layer", "business_dictionary", "epistemology",
                      "strategy", "architecture", "project_management",
                      "methodology", "development", "tasks")],
    ]
    names = set(re.findall(r"[\w/]+\.py", section))
    missing = sorted(
        name for name in names
        if not any((root / name).exists() for root in roots)
    )

    assert not missing, f"architecture section names missing files: {missing}"


def test_counts_in_the_diagram_match_the_catalog(readme):
    """The diagram prints 11 knowledge bases, 81 tools, 8 experts, 17 write
    commands, 10 agent skills. Enumerated facts going stale is this project's
    most repeated bug class."""
    from knowledge_bases.orchestration.commands import WRITE_COMMANDS
    from knowledge_bases.orchestration.experts_catalog import bootstrap_experts
    from knowledge_bases.registry import KNOWLEDGE_BASES
    from knowledge_bases.tools.catalog import bootstrap_catalog

    assert f"{len(KNOWLEDGE_BASES)} KNOWLEDGE BASES" in readme
    assert f"{len(bootstrap_catalog())} tools" in readme
    assert f"{len(WRITE_COMMANDS)} WRITE_COMMANDS" in readme
    assert f"{len(bootstrap_experts())} experts" in readme


def test_per_knowledge_base_tool_counts_in_the_diagram_are_right():
    """The diagram breaks 81 tools down per knowledge base. A single tool moved
    between packages makes that table wrong without any test noticing."""
    from collections import Counter

    from knowledge_bases.tools.catalog import bootstrap_catalog

    counts = Counter(tool.key.split(".")[0] for tool in bootstrap_catalog())
    readme = README.read_text()

    for kb, count in counts.items():
        assert f"{kb} {count}" in readme, (
            f"diagram is missing or wrong for {kb}: expected {count}"
        )


def test_boundary_surface_lists_match_the_code(readme):
    """The README lists every mutable and immutable surface by name. Adding a
    surface in code without updating the README leaves a reader believing the
    boundary is narrower than it is."""
    from knowledge_bases.boundary import IMMUTABLE_SURFACES, MUTABLE_SURFACES

    for surface in MUTABLE_SURFACES | IMMUTABLE_SURFACES:
        assert f"`{surface.value}`" in readme, (
            f"boundary surface {surface.value} is not documented in the README"
        )


def test_the_topology_diagram_is_column_aligned(readme):
    """A box-drawing diagram with a ragged right edge looks broken on GitHub,
    and hand-editing one line is the easy way to cause that. Every framed line
    in the topology block must be the same width."""
    start = readme.index("### System topology")
    block = readme[start : readme.index("### What a request", start)]

    framed = [
        line for line in block.split("\n")
        if line.startswith(("│", "┌", "└", "├"))
    ]
    widths = {len(line) for line in framed}

    assert framed, "no diagram found; this test would be vacuous"
    assert len(widths) == 1, (
        f"diagram lines have ragged widths {sorted(widths)}; every framed line "
        "must be padded to the same column"
    )


def test_the_diagram_fits_a_standard_terminal(readme):
    """80 columns renders without wrapping in a terminal and in GitHub's code
    block. Wider silently wraps, which destroys the boxes."""
    start = readme.index("### System topology")
    block = readme[start : readme.index("### What a request", start)]

    # Markdown tables are excluded: they render as tables, not preformatted
    # text, so their source width does not affect how they display.
    too_wide = [
        line for line in block.split("\n")
        if len(line) > 80 and not line.lstrip().startswith("|")
    ]

    assert not too_wide, (
        f"{len(too_wide)} diagram line(s) exceed 80 columns: "
        f"{[line[:50] for line in too_wide[:3]]}"
    )


# -- completeness: every component must appear somewhere ------------------------


def _architecture_section(readme: str) -> str:
    start = readme.index("## Architecture")
    return readme[start : readme.index("## Knowledge Bases and Experts", start)]


def test_every_root_module_is_documented(readme):
    """The rule: every component is either in the topology diagram or in an
    expansion of one of its boxes. A module nobody drew is a module nobody
    knows to reason about."""
    section = _architecture_section(readme)
    modules = sorted(p.name for p in pathlib.Path(".").glob("*.py"))
    missing = [m for m in modules if m not in section]

    assert modules, "no root modules found; this test would be vacuous"
    assert not missing, f"root modules absent from the architecture section: {missing}"


def test_every_knowledge_base_package_is_documented(readme):
    section = _architecture_section(readme)
    packages = sorted(
        p.name for p in pathlib.Path("knowledge_bases").iterdir()
        if p.is_dir() and p.name != "__pycache__"
    )
    missing = [p for p in packages if p not in section]

    assert not missing, f"knowledge-base packages absent from diagrams: {missing}"


def test_every_module_of_the_two_largest_packages_is_documented(readme):
    """`orchestration` (26 modules) and `tools` (21) each get their own
    expansion, so every one of their modules should be named in it."""
    section = _architecture_section(readme)
    missing = []
    for package in ("orchestration", "tools"):
        for module in pathlib.Path("knowledge_bases", package).glob("*.py"):
            if module.stem == "__init__":
                continue
            if f"{module.stem}.py" not in section:
                missing.append(f"{package}/{module.name}")

    assert not missing, f"modules absent from their expansion diagram: {missing}"


def test_every_script_is_documented(readme):
    section = _architecture_section(readme)
    scripts = sorted(
        p.name for p in pathlib.Path("scripts").iterdir()
        if p.suffix in {".py", ".sh"}
    )
    missing = [s for s in scripts if s not in section]

    assert scripts, "no scripts found; this test would be vacuous"
    assert not missing, f"scripts absent from the architecture section: {missing}"


def test_every_declared_label_is_documented(readme):
    """All 56 labels appear in the ontology expansion. A label that exists but
    is undrawn is a data shape with no documented owner."""
    from knowledge_bases.registry import KNOWLEDGE_BASES

    section = _architecture_section(readme)
    labels = set()
    for spec in KNOWLEDGE_BASES.values():
        labels |= set(
            spec.get("labels", []) if isinstance(spec, dict)
            else getattr(spec, "labels", None) or []
        )
    missing = sorted(label for label in labels if label not in section)

    assert labels, "no labels declared; this test would be vacuous"
    assert not missing, f"declared labels absent from the ontology diagram: {missing}"


def test_every_container_is_documented(readme):
    """A service in docker-compose.yml that no diagram mentions is an operational
    surprise."""
    import re as _re

    compose = pathlib.Path("docker-compose.yml").read_text()
    # Only the `services:` block -- a `volumes:` entry is not a container.
    block = compose[compose.index("services:") :]
    for following in ("\nvolumes:", "\nnetworks:"):
        if following in block:
            block = block[: block.index(following)]
    services = _re.findall(r"^  ([a-z][\w-]*):$", block, _re.M)
    section = _architecture_section(readme)
    missing = [s for s in services if s.replace("noesis-", "") not in section]

    assert services, "no services parsed; this test would be vacuous"
    assert not missing, f"containers absent from the architecture section: {missing}"

def test_the_readme_ontology_header_matches_the_live_ontology():
    """The header said 64 labels while the graph declared 61.

    A stale count in a diagram is not cosmetic here: the block is presented as the
    authoritative list of what can be written, so a reader sizing a product or
    checking whether a label exists is misled. The counts move whenever a package
    is added, so they must be asserted rather than maintained by hand.
    """
    from knowledge_bases.registry import entity_types, relationship_types

    readme = README.read_text()

    expected = (
        f"ONTOLOGY — {len(entity_types())} labels, "
        f"{len(relationship_types())} relationship types"
    )
    assert expected in readme, (
        f"README ontology header is stale; expected to find {expected!r}"
    )


def test_the_readme_extractable_counts_match_the_flags():
    """The other half of the same block. These counts drive scoping decisions, so a
    wrong number here leads someone to size a prompt against fiction."""
    from knowledge_bases.registry import entity_types, relationship_types

    readme = README.read_text()
    labels = sum(1 for e in entity_types().values() if e.extractable)
    rels = sum(1 for r in relationship_types().values() if r.extractable)

    assert f"{labels} labels / {rels} rels are" in readme, (
        f"README claims different extractable counts than {labels}/{rels}"
    )

@pytest.mark.parametrize("package,count", [
    ("methodology", 17),
    ("project_management", 21),
])
def test_abbreviated_relationship_counts_are_accurate(package, count):
    """Two ontology blocks abbreviate as "N rels incl. ..." rather than listing
    every type, which is a reasonable choice for a 20-item structural family. But an
    abbreviated count is a claim, and an unasserted claim goes stale silently -- the
    reader cannot spot the error the way a missing name in a list would show.
    """
    from knowledge_bases.registry import KNOWLEDGE_BASES

    live = len(KNOWLEDGE_BASES[package].relationships)
    assert live == count, (
        f"{package} now declares {live} relationships; update the README block "
        f"that says '{count} rels incl.' and this test together"
    )
    assert f"{count} rels incl." in README.read_text() or (
        f"{count} structural rels incl." in README.read_text()
    ), f"README no longer states {package}'s relationship count"
