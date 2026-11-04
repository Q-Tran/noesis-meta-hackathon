"""Packaging metadata must agree with the ontology declarations.

`pyproject.toml`'s `noesis.ontologies` group is how a SEPARATELY DISTRIBUTED ontology
becomes discoverable once the packages leave this tree. While everything still lives
here the group is unused -- the registry finds packages beside itself first -- which is
exactly why it needs a test: an entry point that is wrong today fails only after the
split, in a different repository, with no obvious cause.
"""
from __future__ import annotations

import pathlib

import pytest

from importlib.metadata import entry_points as _installed_entry_points


@pytest.fixture(scope="module")
def entry_points():
    """The `noesis.ontologies` group as installed distributions declare it.

    Post-split, the domain packages ship in `noesis-ontologies` and declare their
    own entry points; this distribution declares none. Reading the group from
    installed metadata -- not from this repo's `pyproject.toml` -- is the contract
    the registry actually resolves against.
    """
    return {e.name: e.value for e in _installed_entry_points(group="noesis.ontologies")}


def test_every_declared_ontology_package_has_an_entry_point(entry_points):
    """A package with no entry point is invisible once it ships separately.

    Covers every ontology directory on disk, not just the installed ones: `tasks` is
    declared here and installed by the personal product, so omitting its entry point
    would break that product rather than this one -- the hardest kind of packaging bug
    to attribute.
    """
    control_plane = {"tools", "orchestration", "provenance"}
    non_packages = {"bridges", "conformance", "__pycache__"}

    on_disk = {
        path.name
        for path in pathlib.Path("knowledge_bases").iterdir()
        if path.is_dir()
        and (path / "package.py").exists()
        and path.name not in control_plane | non_packages
    }

    missing = sorted(on_disk - set(entry_points))
    assert missing == [], (
        f"these ontology packages have no `noesis.ontologies` entry point, so a "
        f"separate distribution of them could not be found: {missing}"
    )


def test_no_entry_point_names_a_package_that_does_not_exist(entry_points):
    """A stale entry point fails at composition with a confusing import error."""
    import importlib

    for key, module in entry_points.items():
        found = importlib.import_module(module)
        candidate = pathlib.Path(found.__file__).parent / "package.py"
        assert candidate.exists(), (
            f"entry point {key!r} points at {module!r}, whose distribution has no "
            f"package.py beside it ({candidate})"
        )


def test_the_control_plane_is_not_offered_as_an_installable_ontology(entry_points):
    """`tools`, `orchestration` and `provenance` ship WITH the engine.

    They are knowledge bases, so they look like ontologies, but they are the control
    plane -- routing, telemetry and provenance are how the engine works rather than
    what a product chooses to know about. Offering them as separately installable
    would advertise a configuration where the engine has no way to route or record.
    """
    for key in ("tools", "orchestration", "provenance"):
        assert key not in entry_points, (
            f"{key!r} is control-plane infrastructure and ships with noesis-core, not "
            "as an installable ontology"
        )


def test_the_enterprise_profile_installs_only_packages_it_declares(entry_points):
    """Catches a profile naming an ontology this distribution cannot provide."""
    from knowledge_bases.product import PROFILE

    control_plane = {"tools", "orchestration", "provenance"}
    undeclared = sorted(
        set(PROFILE.packages) - set(entry_points) - control_plane
    )

    assert undeclared == [], (
        f"the product profile installs {undeclared}, which this distribution does not "
        "declare as entry points"
    )
