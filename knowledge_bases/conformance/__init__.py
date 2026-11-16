"""Conformance test kit for ontology packages.

An ontology package is a separately installable distribution: it declares an
`OntologyPackage` (`knowledge_bases/base.py`) and a product installs a set of
them. This kit is how a package proves it is a LEGAL package before any product
installs it, and it ships with the engine so a package living in another
repository can run it against itself:

    from knowledge_bases.conformance import check_package

    def test_my_ontology_package_is_conformant():
        check_package(PACKAGE, peers=(), strict=True)

or, as a whole parametrised suite in the author's own test file:

    from knowledge_bases.conformance import conformance_suite

    test_conformance = conformance_suite(PACKAGE)   # returns a test class

Why this exists as shipped code rather than as tests in this repository: every
one of these invariants is already checked here, but at WHOLE-PRODUCT
composition time and against a hardcoded list of the packages this repo happens
to ship. A third-party package is in neither of those. `registry.py` raises on
real violations only once everything composes, which is too late for a package
author and impossible for one who is not installed yet. So each rule is
re-expressed per package, over the package's own declarations plus whatever
peers the caller chooses to supply.

Findings are structured (`Finding`: package, rule, severity, item, detail,
remedy) because the audience is a stranger to this codebase. A bare
`assert False` gives them nothing to act on.

`docs/` note: the rules and their provenance are documented in the docstring of
each rule in `knowledge_bases/conformance/rules.py`; every rule names the
`raise`, test, or audit finding it was harvested from.
"""
from __future__ import annotations

from typing import Iterable

from knowledge_bases.base import OntologyPackage
from knowledge_bases.conformance.context import (
    CONTROL_PLANE_PACKAGES,
    PackageContext,
)
from knowledge_bases.conformance.findings import (
    ConformanceError,
    ConformanceReport,
    Finding,
    Severity,
)
from knowledge_bases.conformance.rules import ALL_RULES, RULES_BY_NAME, Rule

__all__ = [
    "check_package",
    "check_packages",
    "conformance_suite",
    "installed_packages",
    "ALL_RULES",
    "RULES_BY_NAME",
    "PackageContext",
    "ConformanceError",
    "ConformanceReport",
    "Finding",
    "Severity",
    "Rule",
    "CONTROL_PLANE_PACKAGES",
]


def check_package(
    package: OntologyPackage | str,
    *,
    peers: Iterable[OntologyPackage | str] | None = None,
    rules: Iterable[Rule | str] | None = None,
    strict: bool = False,
    threshold: Severity = Severity.ERROR,
    source_root=None,
    control_plane: Iterable[str] | None = None,
) -> ConformanceReport:
    """Check one package and return a `ConformanceReport`.

    `package` may be the `OntologyPackage` object, a `"module.path:ATTR"`
    reference to it, or a package key resolvable through the installed registry.

    `peers` is what the rules compare against for everything that is only
    decidable in composition -- who describes a shared edge, whose label an
    endpoint names. `None` (the default) means "the other packages the local
    product installs", which is the useful default inside a repo that ships
    several. Pass `peers=()` to check a package in true isolation; rules that
    then cannot decide report INFO rather than passing silently.

    `strict=True` raises `ConformanceError` for anything at or above
    `threshold`, which is what a one-line test wants.
    """
    context = PackageContext.load(
        package,
        peers=peers,
        control_plane=control_plane,
        source_root=source_root,
    )
    selected = _select(rules)
    findings: list[Finding] = []
    ran: list[str] = []
    for rule in selected:
        ran.append(rule.__name__)
        findings.extend(rule(context))
    report = ConformanceReport(
        package=context.key,
        findings=tuple(findings),
        rules_run=tuple(ran),
    )
    if strict:
        report.raise_for(threshold)
    return report


def check_packages(
    packages: Iterable[OntologyPackage | str],
    *,
    rules: Iterable[Rule | str] | None = None,
    threshold: Severity = Severity.ERROR,
) -> dict[str, ConformanceReport]:
    """Check several packages against each other as peers.

    Each package is checked with all the OTHERS supplied as its peers, which is
    what a monorepo of ontology packages wants: the shared-ownership rules are
    only meaningful when the set is known.
    """
    resolved = tuple(
        PackageContext.load(package, peers=()).package for package in packages
    )
    reports: dict[str, ConformanceReport] = {}
    for package in resolved:
        peers = tuple(other for other in resolved if other.key != package.key)
        reports[package.key] = check_package(
            package, peers=peers, rules=rules, threshold=threshold
        )
    return reports


def installed_packages() -> tuple[OntologyPackage, ...]:
    """Every ontology package the local product installs, or `()` if none.

    Convenience for a product repository that wants to run the kit over its
    whole profile. Returns empty rather than raising in a repository that
    declares no product, which is the normal state of a standalone ontology
    distribution.
    """
    try:
        from knowledge_bases.registry import DEFAULT_PACKAGES, ontology_packages

        return tuple(ontology_packages(DEFAULT_PACKAGES))
    except Exception:  # noqa: BLE001 - absence of a product is legitimate
        return ()


def conformance_suite(
    package: OntologyPackage | str,
    *,
    peers: Iterable[OntologyPackage | str] | None = None,
    rules: Iterable[Rule | str] | None = None,
    threshold: Severity = Severity.ERROR,
    source_root=None,
    name: str | None = None,
):
    """Build a pytest test class: one test method per conformance rule.

    One method per rule rather than one assert over everything, so a failing
    package reports every rule it breaks in one run and each shows up under its
    own rule name in the test report -- which is the name the author needs to
    look up.

    Usage in a package author's own test file, in any repository:

        from knowledge_bases.conformance import conformance_suite
        from my_ontology.package import PACKAGE

        TestConformance = conformance_suite(PACKAGE)

    The returned class is a plain class with `test_*` methods, so pytest,
    unittest, and nose-style collection all pick it up without a plugin.
    """
    report = check_package(
        package,
        peers=peers,
        rules=rules,
        threshold=threshold,
        source_root=source_root,
    )
    selected = _select(rules)

    def _method_for(rule_name: str):
        def method(self):
            failing = tuple(
                finding
                for finding in report.of_rule(rule_name)
                if finding.severity is not Severity.INFO
                and _at_least(finding.severity, threshold)
            )
            assert not failing, "\n\n".join(str(finding) for finding in failing)

        rule = RULES_BY_NAME.get(rule_name)
        method.__name__ = f"test_{rule_name}"
        method.__doc__ = (rule.__doc__ or "").strip() if rule else None
        return method

    namespace = {"conformance_report": report}
    for rule in selected:
        namespace[f"test_{rule.__name__}"] = _method_for(rule.__name__)

    def test_the_suite_checked_something(self):
        """A conformance suite that runs no rule certifies nothing. This test
        exists so an empty or mis-filtered `rules=` argument fails loudly rather
        than reporting a green package it never looked at."""
        assert not report.is_vacuous, (
            f"no conformance rule ran for {report.package!r}; the suite would "
            "certify the package without checking it"
        )

    namespace["test_the_suite_checked_something"] = test_the_suite_checked_something
    class_name = name or f"TestConformance_{report.package}"
    return type(class_name, (), namespace)


def _at_least(severity: Severity, threshold: Severity) -> bool:
    from knowledge_bases.conformance.findings import at_least

    return at_least(severity, threshold)


def _select(rules: Iterable[Rule | str] | None) -> tuple[Rule, ...]:
    if rules is None:
        return ALL_RULES
    selected: list[Rule] = []
    for rule in rules:
        if isinstance(rule, str):
            try:
                selected.append(RULES_BY_NAME[rule])
            except KeyError as exc:
                raise ValueError(
                    f"unknown conformance rule {rule!r}; known rules: "
                    f"{sorted(RULES_BY_NAME)}"
                ) from exc
        else:
            selected.append(rule)
    return tuple(selected)
