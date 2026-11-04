"""Reach the installed versioning engine without importing it.

`tools/` ships with the engine and holds the self-improvement loop, which files each
suggestion as a versioned `ReviewFinding` for a human to accept or reject. That
requires a versioning engine -- the `methodology` ontology -- so these two modules
held 14 import sites into an optional package. A product without methodology could
not import `tools/`, and `tools/` is the package the planned `noesis-core` is built
around.

These accessors resolve the same objects through `registry.capability()`, so the
dependency is declared by methodology's `package.py` rather than taken by the engine.

WHY REFUSAL, NOT DEGRADATION: elsewhere a missing capability means "skip an optional
side effect". Here it means the caller asked to file a suggestion and there is
nowhere to file it. Returning a suggestion the caller believes was recorded defeats
the propose-confirm posture, whose whole purpose is that a suggestion survives until
somebody rules on it.
"""
from __future__ import annotations

from typing import Any


class VersioningUnavailable(RuntimeError):
    """No installed package provides the versioning engine."""


def _require(name: str) -> Any:
    from knowledge_bases.registry import capability

    resolved = capability(name)
    if resolved is None:
        raise VersioningUnavailable(
            f"this operation needs an installed package providing the {name!r} "
            "capability (the methodology ontology); this product ships without one, "
            "so there is nowhere to file a versioned artifact"
        )
    return resolved


def versioning_repository(graph: Any | None = None) -> Any:
    """An instance of the installed versioning repository."""
    repository = _require("versioning.repository")
    if graph is None:
        from graph import Graph

        graph = Graph()
    return repository(graph)


def changeset_request() -> Any:
    return _require("versioning.changeset_request")


def create_entity() -> Any:
    return _require("versioning.create_entity")


def revise_entity() -> Any:
    return _require("versioning.revise_entity")


def versioning_errors() -> tuple[type[Exception], ...]:
    """Exception types a versioned write raises, for `except` clauses.

    Resolved rather than imported because a caller that cannot import the engine
    also cannot name its exception classes -- and `except Exception` around a
    versioned write would swallow the very bugs these types exist to tell apart.
    """
    from knowledge_bases.registry import capability

    errors = capability("versioning.errors")
    return errors if errors is not None else ()
