"""A package owns its own bounded writes.

The failure this prevents was real and verified: `add_business_term_alias` was
hardcoded in `orchestration/commands.py`, so a product profile that dropped
`business_dictionary` still advertised the command -- an allowlisted mutation whose
handler no longer shipped. The comment in `commands.py` claimed the composition
prevented this while `composed_write_commands()` returned zero commands, so nothing
was actually composed and no test noticed.
"""
from __future__ import annotations

from knowledge_bases.registry import (
    composed_write_commands,
    ontology_packages,
    product_profile,
    resolve_seed,
)


def _commands_for(package_keys):
    found = []
    for package in ontology_packages(package_keys):
        if package.write_commands_seed:
            found.extend(
                command.key for command in resolve_seed(package.write_commands_seed)
            )
    return found


def test_composition_actually_carries_commands():
    """The guard against the original defect, which was an empty composition behind
    a comment asserting it worked."""
    composed = composed_write_commands()

    assert composed, (
        "composed_write_commands() is empty, so every command is still hardcoded "
        "and uninstalling a package cannot remove its writes"
    )


def test_uninstalling_a_package_removes_its_write_commands():
    profile = product_profile()
    without = tuple(key for key in profile.packages if key != "business_dictionary")

    remaining = _commands_for(without)

    assert not [key for key in remaining if key.startswith("business_dictionary.")], (
        "a profile without business_dictionary still exposes its write commands"
    )


def test_a_packages_commands_are_present_when_it_is_installed():
    """The other half: removal must not be achieved by never contributing."""
    installed = _commands_for(product_profile().packages)

    assert "business_dictionary.add_business_term_alias" in installed
    assert "strategy.evaluate_decision_health" in installed


def test_the_allowlist_still_holds_every_command():
    """Moving ownership must not change the surface. 18 commands."""
    from knowledge_bases.orchestration.commands import WRITE_COMMANDS

    assert len(WRITE_COMMANDS) == 18, (
        f"the write surface changed size: {sorted(WRITE_COMMANDS)}"
    )


def test_no_command_is_declared_twice():
    """A package contributing a command the orchestrator also hardcodes would
    silently shadow one dispatcher with another."""
    from knowledge_bases.orchestration.commands import WRITE_COMMANDS

    composed = [command.key for command in composed_write_commands()]

    assert len(composed) == len(set(composed))
    assert len(WRITE_COMMANDS) >= len(composed)


def test_a_package_does_not_import_the_orchestrators_private_helpers():
    """Packages build commands from `write_dispatch`, not from `commands.py`.

    Importing `commands.py` would be circular -- it composes the packages'
    contributions -- and across a future distribution boundary unresolvable.
    """
    # Located through the registry, not by joining `knowledge_bases/<key>/`. The
    # ontologies monorepo keeps these under `packages/<key>/` and installs them as
    # `knowledge_bases.<key>`, so a hardcoded path raises FileNotFoundError there --
    # which at least fails loudly, unlike the same assumption inside the conformance
    # kit, where it silently made the rule pass.
    from knowledge_bases.registry import _declaration_path

    checked = 0
    for package in ("business_dictionary", "strategy"):
        declaration = _declaration_path(package)
        assert declaration is not None, f"{package} is not installed"
        source = (declaration.parent / "write_commands.py").read_text()
        assert "orchestration.commands" not in source, (
            f"{package} imports the module that composes it"
        )
        assert "write_dispatch" in source
        checked += 1

    assert checked == 2, "both packages must actually be inspected"
