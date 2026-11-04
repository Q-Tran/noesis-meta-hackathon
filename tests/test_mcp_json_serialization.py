"""Every MCP server's JSON serializer must handle driver temporal values.

`neo4j.time.DateTime` is not JSON-serialisable, and it arrives nested inside model
payloads rather than at the top level, so a serializer that only handles the top level
raises at the point a tool returns its result -- after the work is done.

The servers are DISCOVERED rather than listed. This module used to import
`methodology.mcp_server` at module scope, which made the whole file uncollectable for a
product that does not install methodology -- taking the shared-serializer test down
with it. Discovery also means a new domain MCP server is covered the day it appears
instead of the day somebody remembers to add it here.
"""
import importlib
import json
import pathlib

import pytest
from neo4j.time import DateTime
from pydantic import BaseModel

from knowledge_bases.mcp import text_json


class DriverValueModel(BaseModel):
    payload: object


def nested_datetime_model():
    return DriverValueModel(payload={
        "nested": [{"created_at": DateTime(2026, 8, 5, 12, 30)}]
    })


def assert_datetime_serialized(serializer):
    content = serializer(nested_datetime_model())

    payload = json.loads(content[0].text)
    assert payload["payload"]["nested"][0]["created_at"] == (
        "2026-08-05T12:30:00.000000000"
    )


def _installed_mcp_servers():
    """Every installed knowledge base that ships an `mcp_server` with a `_json`.

    Globbed across `knowledge_bases.__path__`, not just this repo: post-split the
    domain packages are separate distributions, and `pkgutil.extend_path` spans
    them. This is the same span the registry resolves packages through.
    """
    from knowledge_bases.product import PROFILE
    from knowledge_bases.registry import _declaration_path

    # Keys come from the product profile, package directories from
    # `registry._declaration_path` -- the same resolution the engine uses, so this
    # covers the in-tree control plane AND entry-point-distributed domain
    # packages. A disk glob of `knowledge_bases/` alone sees only in-tree code.
    found = []
    for key in PROFILE.packages:
        decl = _declaration_path(key)
        if decl is None:
            continue
        path = decl.parent / "mcp_server.py"
        if not path.exists():
            continue
        module = importlib.import_module(f"knowledge_bases.{key}.mcp_server")
        if hasattr(module, "_json"):
            found.append((key, module._json))
    return found


def test_shared_text_json_serializes_nested_neo4j_datetime_in_model():
    assert_datetime_serialized(text_json)


@pytest.mark.parametrize(
    "serializer",
    [pytest.param(fn, id=key) for key, fn in _installed_mcp_servers()],
)
def test_every_installed_mcp_server_serializes_nested_driver_datetimes(serializer):
    """Parametrized over what is installed, so this cannot silently cover nothing."""
    assert_datetime_serialized(serializer)


def test_the_discovery_finds_the_servers_that_exist():
    """Guards the guard: an empty parametrization would pass while testing nothing."""
    discovered = {key for key, _ in _installed_mcp_servers()}

    assert "orchestration" in discovered, (
        "the control plane's MCP server was not discovered, so the parametrized test "
        "above is not covering what it claims"
    )
    assert len(discovered) >= 2
