"""The product profile (`knowledge_bases/product.py`) is the one object a
product distribution replaces: it declares which ontology packages are
installed, what the product registers as, and where self-improvement may write
generated logic. These tests pin the properties that make it safe to swap --
that every declaration resolves, that derived surfaces actually follow it, and
that it cannot redirect generated code into a package the product does not own.
"""
from __future__ import annotations

import pytest

from knowledge_bases.base import ProductProfile
from knowledge_bases.product import PROFILE
from knowledge_bases.registry import (
    DEFAULT_PACKAGES,
    KNOWLEDGE_BASES,
    ontology_packages,
    product_profile,
)


def test_the_registry_composes_exactly_the_profiles_packages():
    """`DEFAULT_PACKAGES` must be the profile's list, not a second copy of it:
    two lists would drift, and the drift would silently change the ontology."""
    assert DEFAULT_PACKAGES == PROFILE.packages
    assert product_profile() is PROFILE


def test_every_declared_package_loads():
    """A profile naming a package with no `package.py` would fail at import of
    the registry, i.e. at startup of every entry point."""
    packages = ontology_packages(PROFILE.packages)

    assert len(packages) == len(PROFILE.packages)
    assert [p.key for p in packages] == list(PROFILE.packages)


def test_the_composed_registry_covers_every_installed_package():
    for key in PROFILE.packages:
        assert key in KNOWLEDGE_BASES, f"{key} is installed but composes no KB"


def test_the_workflow_profile_reference_resolves_to_phases():
    """The product's phase ladder is a seed reference so a product can ship a
    shorter one; an unresolvable reference would break baselining only when a
    user tried to approve a phase."""
    from knowledge_bases.base import WorkflowPhase
    from knowledge_bases.registry import resolve_seed

    assert PROFILE.workflow_profile
    phases = resolve_seed(PROFILE.workflow_profile)

    assert phases
    assert all(isinstance(phase, WorkflowPhase) for phase in phases)


def test_the_generated_tool_target_is_a_package_this_product_installs():
    """Generated handlers are appended to this package's own source. If it were
    not installed, the append would target a directory that does not exist; if
    it were an ontology distribution, it would target `site-packages`."""
    assert PROFILE.generated_tool_target in PROFILE.packages


def test_the_generated_tool_target_owns_the_modules_that_get_appended_to():
    """`new_tool` writes a handler to `mcp_tools.py` and a seed to `catalog.py`;
    both must already exist in the target package."""
    from knowledge_bases.tools.source_editor import REPO_ROOT

    target = REPO_ROOT / "knowledge_bases" / PROFILE.generated_tool_target
    assert (target / "mcp_tools.py").exists()
    assert (target / "catalog.py").exists()


def test_the_mcp_server_name_drives_the_export_surface():
    """Exported skill directories and the harness registration are named from
    the profile, so two products can be installed side by side."""
    from knowledge_bases.orchestration import prime_export

    assert prime_export.MCP_SERVER_NAME == PROFILE.mcp_server_name
    assert prime_export._skill_dir_prefix() == f"{PROFILE.mcp_server_name}-"


def test_a_profile_is_frozen():
    """The profile is architecture, not runtime state (`product_profile` in
    `knowledge_bases/boundary.py`): no code path may retarget the product it
    is running as by assigning to it."""
    with pytest.raises(Exception):
        PROFILE.packages = ()


def test_a_narrower_product_is_declarable_without_touching_the_engine():
    """The point of the object: a second product is a declaration, not a fork.
    A profile installing fewer packages must compose a strictly smaller
    ontology, with no engine change."""
    from knowledge_bases.registry import compose

    individual = ProductProfile(
        key="noesis-individual", name="Noesis",
        packages=tuple(k for k in PROFILE.packages if k != "project_management"),
        generated_tool_target="tools",
    )

    full = compose(ontology_packages(PROFILE.packages))
    narrow = compose(ontology_packages(individual.packages))

    assert set(narrow) < set(full)
    full_labels = {l for kb in full.values() for l in kb.labels}
    narrow_labels = {l for kb in narrow.values() for l in kb.labels}
    assert narrow_labels < full_labels
