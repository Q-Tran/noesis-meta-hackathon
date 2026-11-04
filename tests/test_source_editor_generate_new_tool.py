"""`knowledge_bases.tools.source_editor.generate_new_skill` is the one
`new_tool`-kind apply path that writes LLM-authored *logic* (not a single
string replacement) -- so it's held to a stricter validation bar before
anything is ever written to disk. These tests write to a throwaway fixture
tree under `tmp_path`, never the real `knowledge_bases/*/mcp_tools.py`/
`catalog.py`."""
from __future__ import annotations

import textwrap

import pytest

from knowledge_bases.tools.models import GeneratedToolDraft
from knowledge_bases.tools.source_editor import (
    REPO_ROOT,
    SourceEditError,
    _ToolsListAppender,
    _validate_handler_source,
    append_handler,
    generate_new_skill,
)

MCP_TOOLS_FIXTURE = textwrap.dedent('''\
    """MCP-callable handlers for the tools skill catalog."""
    from __future__ import annotations

    from typing import Any

    from graph import Graph


    def list_project_blockers_tool(arguments: dict[str, Any], graph: Graph) -> Any:
        return graph.run("MATCH (t:Task) WHERE t.blocked = true RETURN t")
    ''')

CATALOG_FIXTURE = textwrap.dedent('''\
    """Seed `Tool` definitions for the tools knowledge base."""
    from knowledge_bases.tools.catalog import _skill


    TOOLS = [
        _skill(
            "tools.list_project_blockers",
            "tools",
            "List Project Blockers",
            "List blocked work.",
            {"type": "object", "properties": {}},
            "knowledge_bases.tools.mcp_tools:list_project_blockers_tool",
        ),
    ]
    ''')

VALID_HANDLER_SOURCE = (
    'def summarize_blocked_epics_tool(arguments, graph):\n'
    '    return graph.run("MATCH (e:Epic) WHERE e.blocked = true RETURN e")\n'
)


@pytest.fixture
def repo_tree(tmp_path):
    """A throwaway package tree shaped like the real one: the knowledge base
    owns both its handlers and its `TOOLS` seed list."""
    tools_dir = tmp_path / "knowledge_bases" / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "mcp_tools.py").write_text(MCP_TOOLS_FIXTURE)
    (tools_dir / "catalog.py").write_text(CATALOG_FIXTURE)
    return tmp_path


def _draft(**overrides) -> GeneratedToolDraft:
    defaults = dict(
        key="tools.summarize_blocked_epics",
        name="Summarize Blocked Epics",
        description="Summarize blocked epics.",
        input_schema={"type": "object", "properties": {}},
        handler_function_name="summarize_blocked_epics_tool",
        handler_source=VALID_HANDLER_SOURCE,
        confidence=0.7,
    )
    defaults.update(overrides)
    return GeneratedToolDraft(**defaults)


# -- _validate_handler_source ---------------------------------------------------

def test_validate_handler_source_accepts_a_well_formed_handler():
    func = _validate_handler_source(VALID_HANDLER_SOURCE, "summarize_blocked_epics_tool")

    assert func.name.value == "summarize_blocked_epics_tool"


def test_validate_handler_source_rejects_invalid_python_syntax():
    with pytest.raises(SourceEditError, match="not valid Python"):
        _validate_handler_source("def broken(:\n    pass", "broken")


def test_validate_handler_source_rejects_more_than_one_top_level_statement():
    source = (
        "def helper():\n    pass\n\n\n"
        "def summarize_blocked_epics_tool(arguments, graph):\n    return []\n"
    )
    with pytest.raises(SourceEditError, match="exactly one top-level"):
        _validate_handler_source(source, "summarize_blocked_epics_tool")


def test_validate_handler_source_rejects_a_non_function_statement():
    source = "x = 1\ndef summarize_blocked_epics_tool(arguments, graph):\n    return []\n"
    with pytest.raises(SourceEditError, match="exactly one top-level"):
        _validate_handler_source(source, "summarize_blocked_epics_tool")


def test_validate_handler_source_rejects_a_mismatched_function_name():
    with pytest.raises(SourceEditError, match="expected 'a_different_name'"):
        _validate_handler_source(VALID_HANDLER_SOURCE, "a_different_name")


def test_validate_handler_source_rejects_a_wrong_signature():
    source = "def summarize_blocked_epics_tool(arguments):\n    return []\n"
    with pytest.raises(SourceEditError, match="signature"):
        _validate_handler_source(source, "summarize_blocked_epics_tool")


def test_validate_handler_source_rejects_an_async_function():
    source = "async def summarize_blocked_epics_tool(arguments, graph):\n    return []\n"
    with pytest.raises(SourceEditError, match="must not be async"):
        _validate_handler_source(source, "summarize_blocked_epics_tool")


@pytest.mark.parametrize("unsafe_call", [
    "import os\n",
    "os.system('rm -rf /')\n",
    "eval('1+1')\n",
    "exec('x=1')\n",
    "open('/etc/passwd')\n",
    "subprocess.run(['ls'])\n",
    "__import__('os')\n",
])
def test_validate_handler_source_rejects_unsafe_constructs(unsafe_call):
    source = f"def summarize_blocked_epics_tool(arguments, graph):\n    {unsafe_call}    return []\n"
    with pytest.raises(SourceEditError, match="disallowed"):
        _validate_handler_source(source, "summarize_blocked_epics_tool")


# -- append_handler ---------------------------------------------------------

def test_append_handler_adds_the_function_to_the_end_of_the_file(repo_tree):
    path = repo_tree / "knowledge_bases" / "tools" / "mcp_tools.py"

    append_handler(VALID_HANDLER_SOURCE, "summarize_blocked_epics_tool", path)

    updated = path.read_text()
    assert "def summarize_blocked_epics_tool(arguments, graph):" in updated
    assert "def list_project_blockers_tool(arguments: dict[str, Any], graph: Graph) -> Any:" in updated


def test_append_handler_preserves_valid_python_syntax(repo_tree):
    path = repo_tree / "knowledge_bases" / "tools" / "mcp_tools.py"

    append_handler(VALID_HANDLER_SOURCE, "summarize_blocked_epics_tool", path)

    compile(path.read_text(), str(path), "exec")


def test_append_handler_refuses_to_shadow_an_existing_function(repo_tree):
    path = repo_tree / "knowledge_bases" / "tools" / "mcp_tools.py"

    with pytest.raises(SourceEditError, match="already exists"):
        append_handler(
            "def list_project_blockers_tool(arguments, graph):\n    return []\n",
            "list_project_blockers_tool", path,
        )

    # Original file untouched.
    assert path.read_text() == MCP_TOOLS_FIXTURE


def test_append_handler_raises_without_writing_for_invalid_source(repo_tree):
    path = repo_tree / "knowledge_bases" / "tools" / "mcp_tools.py"

    with pytest.raises(SourceEditError):
        append_handler("def broken(:\n    pass", "broken", path)

    assert path.read_text() == MCP_TOOLS_FIXTURE


# -- generate_new_skill ------------------------------------------------------

def test_generate_new_skill_writes_the_handler_and_registers_it(repo_tree):
    catalog_path = repo_tree / "knowledge_bases" / "tools" / "catalog.py"

    paths = generate_new_skill(_draft(), repo_root=repo_tree)

    handler_path = repo_tree / "knowledge_bases" / "tools" / "mcp_tools.py"
    assert paths == [handler_path, catalog_path]
    assert "def summarize_blocked_epics_tool(arguments, graph):" in handler_path.read_text()
    assert '"tools.summarize_blocked_epics"' in catalog_path.read_text()


def test_generate_new_skill_registration_is_valid_python(repo_tree):
    """The appended registration has to be executable Python that yields a real
    `Tool`, not just text that parses -- this is the seed the graph is synced
    from."""
    catalog_path = repo_tree / "knowledge_bases" / "tools" / "catalog.py"

    generate_new_skill(_draft(), repo_root=repo_tree)

    namespace: dict = {}
    # Catalog fixture imports the real `_skill` from knowledge_bases.tools.catalog,
    # so this executes against the real dependency, not a further stub.
    exec(compile(catalog_path.read_text(), str(catalog_path), "exec"), namespace)
    skills = {s.key: s for s in namespace["TOOLS"]}
    assert "tools.summarize_blocked_epics" in skills
    assert skills["tools.summarize_blocked_epics"].handler == (
        "knowledge_bases.tools.mcp_tools:summarize_blocked_epics_tool"
    )
    # Original entry survives untouched.
    assert "tools.list_project_blockers" in skills


def test_generate_new_skill_does_not_register_when_the_handler_is_invalid(repo_tree):
    catalog_path = repo_tree / "knowledge_bases" / "tools" / "catalog.py"
    original_catalog = catalog_path.read_text()

    with pytest.raises(SourceEditError):
        generate_new_skill(
            _draft(handler_source="def broken(:\n    pass"),
            repo_root=repo_tree,
        )

    assert catalog_path.read_text() == original_catalog


def test_generate_new_skill_requires_a_key_prefixed_by_an_existing_knowledge_base():
    with pytest.raises(ValueError, match="must be"):
        generate_new_skill(_draft(key="no_dot_here"))


def test_generate_new_skill_raises_when_the_target_has_no_existing_mcp_tools(tmp_path):
    """Even the product-owned package must already exist: `new_tool` extends a
    real handler module, it never creates a knowledge base."""
    (tmp_path / "knowledge_bases" / "tools").mkdir(parents=True)

    with pytest.raises(SourceEditError, match="No existing"):
        generate_new_skill(_draft(), repo_root=tmp_path)


def test_generate_new_skill_refuses_a_knowledge_base_the_product_does_not_own(repo_tree):
    """The product profile names the one package whose source this product may
    append to. Writing generated logic into an installed ontology package would
    target `site-packages` -- unwritable on a read-only install, and silently
    discarded by the next upgrade where it did succeed, leaving a registered
    `Tool` pointing at a vanished handler."""
    with pytest.raises(SourceEditError, match="product-owned"):
        generate_new_skill(
            _draft(key="semantic_layer.summarize_blocked_epics"), repo_root=repo_tree,
        )


def test_generate_new_skill_refuses_when_the_product_allows_no_generated_tools(repo_tree):
    """A product may opt out of tool generation entirely (a consumer build with
    no self-improvement source path); then no target is writable at all."""
    import knowledge_bases.tools.source_editor as source_editor
    from knowledge_bases.base import ProductProfile

    opted_out = ProductProfile(
        key="noesis-individual", name="Noesis", packages=("tools",),
        generated_tool_target=None,
    )
    real = source_editor.product_profile
    source_editor.product_profile = lambda: opted_out
    try:
        with pytest.raises(SourceEditError, match="does not allow generated tools"):
            generate_new_skill(_draft(), repo_root=repo_tree)
    finally:
        source_editor.product_profile = real


def test_generate_new_skill_rejects_a_duplicate_key(repo_tree):
    catalog_path = repo_tree / "knowledge_bases" / "tools" / "catalog.py"

    with pytest.raises(SourceEditError, match="already registered"):
        generate_new_skill(
            _draft(key="tools.list_project_blockers", handler_function_name="list_project_blockers_tool_v2",
                    handler_source="def list_project_blockers_tool_v2(arguments, graph):\n    return []\n"),
            repo_root=repo_tree,
        )


def test_the_catalog_fixture_matches_the_real_per_package_catalog_shape():
    """The break this guards against actually happened: when seeds moved from a
    single `bootstrap_catalog()` list into per-package `TOOLS` lists, these
    tests kept passing against a fixture still shaped like the old catalog,
    while `new_tool` could no longer register against any real one. Assert the
    fixture's structure against a real package catalog instead of trusting it.
    """
    import libcst as cst

    from knowledge_bases.registry import ontology_packages

    real_paths = [
        REPO_ROOT / "knowledge_bases" / package.key / "catalog.py"
        for package in ontology_packages()
        if package.tools_seed
    ]
    real_paths = [path for path in real_paths if path.exists()]
    assert real_paths, "no package owns a catalog.py -- has the seed layout moved?"

    def appendable(source: str) -> bool:
        """True if `_ToolsListAppender` can find the seed list it edits."""
        tree = cst.parse_module(source)
        transformer = _ToolsListAppender(
            cst.parse_module("_skill()").body[0].body[0].value
        )
        tree.visit(transformer)
        return transformer.found

    assert appendable(CATALOG_FIXTURE), "fixture no longer exposes a TOOLS seed list"
    for path in real_paths:
        assert appendable(path.read_text()), (
            f"{path} has no appendable TOOLS list, so new_tool cannot register there"
        )
