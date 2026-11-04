"""The bootstrap catalog is the one place tool surface is still declared in
Python (as seed data); every entry must resolve to a real, importable
handler and match the owning knowledge base's key prefix."""
from knowledge_bases.registry import KNOWLEDGE_BASES
from knowledge_bases.tools.catalog import TRACE_CONTEXT_PROPERTIES, bootstrap_catalog
from knowledge_bases.tools.resolver import resolve_handler


def test_catalog_has_no_duplicate_keys():
    skills = bootstrap_catalog()
    keys = [skill.key for skill in skills]

    assert len(keys) == len(set(keys))


def test_every_catalog_entry_key_is_prefixed_by_its_knowledge_base():
    for skill in bootstrap_catalog():
        assert skill.key.startswith(f"{skill.knowledge_base}."), skill.key


def test_every_catalog_entry_belongs_to_a_registered_knowledge_base():
    for skill in bootstrap_catalog():
        assert skill.knowledge_base in KNOWLEDGE_BASES, skill.knowledge_base


def test_every_catalog_entry_handler_resolves_to_a_callable():
    resolve_handler.cache_clear()
    for skill in bootstrap_catalog():
        handler = resolve_handler(skill.handler)
        assert callable(handler), skill.key
    resolve_handler.cache_clear()


def test_every_catalog_entry_has_a_non_empty_description_and_object_schema():
    for skill in bootstrap_catalog():
        assert skill.description.strip(), skill.key
        assert skill.input_schema.get("type") == "object", skill.key


def test_every_catalog_entry_accepts_trace_id_and_session_id_correlation_hints():
    """A client-side agent driving its own tool-calling loop must be able to
    pass the same `trace_id`/`session_id` on every call it makes -- across
    every skill, not just `hybrid_search` -- for the self-improvement loop
    to correlate a full turn/session (see `knowledge_bases.mcp.trace_context`
    and `knowledge_bases.mcp.dispatch_tool`)."""
    for skill in bootstrap_catalog():
        properties = skill.input_schema.get("properties", {})
        for prop_name in TRACE_CONTEXT_PROPERTIES:
            assert prop_name in properties, f"{skill.key} missing {prop_name!r}"


def test_call_noesis_tool_is_registered_and_matches_the_handlers_own_allowlist():
    """`orchestration.call_noesis_tool` is what makes every bounded tool
    reachable through the single `noesis-orchestrator` MCP connection (see
    `knowledge_bases.orchestration.mcp_tools.call_noesis_tool_tool`). Its
    declared `knowledge_base` enum and the handler's own validation share
    one constant so they can never drift apart -- this pins that."""
    from knowledge_bases.orchestration.mcp_tools import CALL_NOESIS_TOOL_KNOWLEDGE_BASES

    skills = {skill.key: skill for skill in bootstrap_catalog()}
    schema = skills["orchestration.call_noesis_tool"].input_schema

    assert set(schema["properties"]["knowledge_base"]["enum"]) == set(
        CALL_NOESIS_TOOL_KNOWLEDGE_BASES
    )
    assert "orchestration" in CALL_NOESIS_TOOL_KNOWLEDGE_BASES


def test_search_noesis_tools_is_registered_and_can_search_orchestrations_own_tools():
    """Unlike `call_noesis_tool` (a dispatch boundary that deliberately
    excludes `orchestration`), `search_noesis_tools` is a pure discovery
    surface over the *entire* catalog and should be able to find
    `orchestration`'s own tools too."""
    skills = {skill.key: skill for skill in bootstrap_catalog()}
    schema = skills["orchestration.search_noesis_tools"].input_schema

    assert "orchestration" in schema["properties"]["knowledge_base"]["enum"]
    assert schema["required"] == ["need"]
