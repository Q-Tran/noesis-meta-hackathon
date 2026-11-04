"""What a conformance check reports, and why it is a report rather than an assert.

The audience for a conformance failure is a package author in a DIFFERENT
repository. They have this kit (shipped with the engine) and their own package;
they do not have this codebase, its history, or the reviewer who knows why a
rule exists. A bare `assert False` tells them nothing they can act on, and
neither does a traceback pointing into engine internals.

So every check yields `Finding`s carrying five things: which package, which
rule, how bad it is, the exact offending item, and a remedy written as an
instruction. `ConformanceReport` groups them, and `ConformanceError` is what a
caller gets if it asked for an exception instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """How much a finding blocks installation.

    - `ERROR`: the package is not legal. Something will be silently wrong at
      runtime -- a write rejected, a type unextractable, a search returning
      nothing -- or composition will raise outright.
    - `WARNING`: legal on its own, fragile in composition. Mostly the
      single-ownership seams: the package is only complete because some OTHER
      installed package supplies a description or a tool, so a product that
      drops that package breaks this one.
    - `INFO`: the check could not be decided from what was supplied (usually
      no peer packages were given), recorded rather than dropped so a clean
      report never means "not looked at".
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


_ORDER = {Severity.INFO: 0, Severity.WARNING: 1, Severity.ERROR: 2}


def at_least(severity: Severity, threshold: Severity) -> bool:
    """Whether `severity` is as bad as `threshold` or worse."""
    return _ORDER[severity] >= _ORDER[threshold]


@dataclass(frozen=True)
class Finding:
    """One conformance violation, stated so a stranger can fix it.

    `item` is the offending thing named exactly as the package declared it
    (a label, a relationship, a seed reference, a step key), because "some
    label is undescribed" costs the author a grep that this field saves.
    `remedy` is phrased as an instruction, not a diagnosis.
    """

    package: str
    rule: str
    severity: Severity
    item: str
    detail: str
    remedy: str

    def __str__(self) -> str:
        return (
            f"[{self.severity.value}] {self.package}: {self.rule}\n"
            f"  item:   {self.item}\n"
            f"  what:   {self.detail}\n"
            f"  fix:    {self.remedy}"
        )


class ConformanceError(AssertionError):
    """Raised by the strict entry points. Subclasses `AssertionError` so it
    reads as a test failure in any runner, while still carrying the structured
    findings for a caller that wants to inspect rather than print."""

    def __init__(self, findings: tuple[Finding, ...]) -> None:
        self.findings = findings
        super().__init__(
            f"{len(findings)} conformance finding(s):\n\n"
            + "\n\n".join(str(finding) for finding in findings)
        )


@dataclass(frozen=True)
class ConformanceReport:
    """Every finding for one package, plus which rules were actually run.

    `rules_run` exists because the dangerous failure mode of a conformance kit
    is silence: a report with no findings because no rule applied looks exactly
    like a report with no findings because the package is clean. Recording the
    rules makes the difference visible, and `is_vacuous` names it.
    """

    package: str
    findings: tuple[Finding, ...] = ()
    rules_run: tuple[str, ...] = ()
    skipped: tuple[tuple[str, str], ...] = field(default=())

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    @property
    def infos(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.INFO)

    @property
    def is_conformant(self) -> bool:
        """No errors. Warnings are composition fragility, not illegality."""
        return not self.errors

    @property
    def is_vacuous(self) -> bool:
        """True when nothing was actually checked, which is never a pass."""
        return not self.rules_run

    def of_rule(self, rule: str) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.rule == rule)

    def at(self, threshold: Severity) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if at_least(f.severity, threshold))

    def raise_for(self, threshold: Severity = Severity.ERROR) -> None:
        """Raise `ConformanceError` if anything reached `threshold`."""
        failing = self.at(threshold)
        if failing:
            raise ConformanceError(failing)

    def format(self, threshold: Severity = Severity.INFO) -> str:
        shown = self.at(threshold)
        if not shown:
            return (
                f"{self.package}: conformant "
                f"({len(self.rules_run)} rules, 0 findings)"
            )
        return f"{self.package}: {len(shown)} finding(s)\n\n" + "\n\n".join(
            str(finding) for finding in shown
        )
