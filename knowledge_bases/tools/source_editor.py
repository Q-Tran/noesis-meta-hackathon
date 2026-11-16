"""Apply a `new_tool` `SelfImprovementSuggestion` as a real, minimal
source-code edit.

Wording-kind improvements (skill descriptions, expert capabilities and
prompts) are graph writes -- see `knowledge_bases.tools.graph_edits` --
because the things they change are graph data. A *new skill* is the one
kind whose change is LLM-authored logic, and logic has to live in code:
this module appends one new, statically-validated handler function to an
*existing* `knowledge_bases/<kb>/mcp_tools.py` and registers it in
`catalog.py`'s `bootstrap_catalog()`, using `libcst` (concrete syntax tree,
format-preserving) for a targeted, minimal-diff edit rather than asking an
LLM to regenerate a whole file.

It is held to a strict bar: `_validate_handler_source` rejects anything that
isn't exactly one top-level function with the right signature and no import
statements or unsafe builtin references, *and* (see
`knowledge_bases.tools.self_improvement_apply`) the caller runs the full
test suite before committing it -- a `new_tool` apply that fails either
gate is discarded, never partially committed. Nothing here ever rewrites
unrelated existing code. Callers own the test-gate/commit/revert lifecycle.
"""
from __future__ import annotations

from pathlib import Path

import libcst as cst

from knowledge_bases.registry import product_profile
from knowledge_bases.tools.models import (
    GeneratedToolDraft,
    SelfImprovementKind,
    SelfImprovementSuggestion,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# There is deliberately no single catalog path: seeds live per ontology
# package (`knowledge_bases/<kb>/catalog.py`), resolved by `_catalog_path`.

# Names a generated handler must never reference -- process/filesystem/
# interpreter escape hatches that have no legitimate use in a bounded,
# read-only `(arguments: dict, graph: Graph) -> Any` MCP handler. Checked
# by `_reject_unsafe_constructs` on every `new_tool` draft before it's
# ever written to disk (see `_validate_handler_source`).
UNSAFE_NAMES = frozenset({
    "os", "sys", "subprocess", "shutil", "socket", "importlib", "ctypes",
    "eval", "exec", "compile", "__import__", "open", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
})


class SourceEditError(RuntimeError):
    pass



def _quoted(text: str) -> str:
    """A Python string literal for `text` (double-quoted, escaped)."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


class _CollectNames(cst.CSTVisitor):
    """Collect every `Name`/`Attribute` base referenced anywhere in the
    function -- used to reject a handler that touches anything in
    `UNSAFE_NAMES` (see `_reject_unsafe_constructs`)."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: cst.Name) -> None:
        self.names.add(node.value)


def _validate_handler_source(
    handler_source: str, expected_function_name: str,
) -> cst.FunctionDef:
    """Parse and validate a `new_tool` draft's `handler_source` before it
    is ever written to disk. Raises `SourceEditError` for anything that
    isn't exactly one safe, correctly-shaped top-level function -- never
    attempts to "fix" or partially accept a bad draft."""
    try:
        module = cst.parse_module(handler_source)
    except cst.ParserSyntaxError as exc:
        raise SourceEditError(f"Generated handler source is not valid Python: {exc}") from exc

    top_level_defs = [
        node for node in module.body
        if isinstance(node, cst.FunctionDef)
    ]
    non_def_statements = [node for node in module.body if not isinstance(node, cst.FunctionDef)]
    if len(top_level_defs) != 1 or non_def_statements:
        raise SourceEditError(
            "Generated handler source must contain exactly one top-level "
            "function definition and nothing else (no imports, no other "
            "statements)"
        )
    func = top_level_defs[0]
    if func.name.value != expected_function_name:
        raise SourceEditError(
            f"Generated handler function is named {func.name.value!r}, "
            f"expected {expected_function_name!r}"
        )
    param_names = [param.name.value for param in func.params.params]
    if param_names != ["arguments", "graph"]:
        raise SourceEditError(
            f"Generated handler must have signature (arguments, graph), "
            f"got ({', '.join(param_names)})"
        )
    if func.asynchronous is not None:
        raise SourceEditError("Generated handler must not be async")

    collector = _CollectNames()
    func.visit(collector)
    unsafe = collector.names & UNSAFE_NAMES
    if unsafe:
        raise SourceEditError(
            f"Generated handler references disallowed name(s): {sorted(unsafe)}"
        )
    return func


class _AppendFunctionToModule(cst.CSTTransformer):
    """Append one function definition to the end of a module, with a
    blank-line separator matching the file's existing style."""

    def __init__(self, func: cst.FunctionDef) -> None:
        self.func = func

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        blank_line = cst.EmptyLine(newline=cst.Newline())
        spaced_func = self.func.with_changes(
            leading_lines=[blank_line, blank_line],
        )
        return updated_node.with_changes(body=[*updated_node.body, spaced_func])


def append_handler(handler_source: str, function_name: str, path: Path) -> None:
    """Append one validated handler function to the end of `path` (an
    existing `knowledge_bases/<kb>/mcp_tools.py`). Raises `SourceEditError`
    without writing anything if `handler_source` fails validation or
    `function_name` already exists in the file (never silently shadows an
    existing handler)."""
    func = _validate_handler_source(handler_source, function_name)

    source = path.read_text()
    existing = cst.parse_module(source)
    existing_names = {
        node.name.value for node in existing.body if isinstance(node, cst.FunctionDef)
    }
    if function_name in existing_names:
        raise SourceEditError(
            f"{function_name!r} already exists in {path}; refusing to shadow it"
        )

    updated = existing.visit(_AppendFunctionToModule(func))
    path.write_text(updated.code)


def _mcp_tools_path(knowledge_base: str, repo_root: Path) -> Path:
    return repo_root / "knowledge_bases" / knowledge_base / "mcp_tools.py"


def _require_generated_tool_target(knowledge_base: str) -> None:
    """Refuse to write generated logic anywhere but the product's own package.

    The product profile names exactly one knowledge base whose source this
    product owns and may append to (`generated_tool_target`). Every other
    installed ontology package is a separate distribution living in
    `site-packages`: appending an LLM-authored handler there would fail on a
    read-only install, and where it did succeed the handler would be silently
    destroyed by the next package upgrade while its `Tool` row survived in the
    graph -- a registered tool pointing at a handler that no longer exists.
    """
    profile = product_profile()
    target = profile.generated_tool_target
    if target is None:
        raise SourceEditError(
            f"Product {profile.key!r} does not allow generated tools "
            f"(generated_tool_target is unset)"
        )
    if knowledge_base != target:
        raise SourceEditError(
            f"Generated tools may only be written to the product-owned "
            f"{target!r} package, not {knowledge_base!r}: an installed ontology "
            f"package's source is not this product's to edit"
        )


def generate_new_skill(
    draft: GeneratedToolDraft, repo_root: Path = REPO_ROOT,
) -> list[Path]:
    """Apply a `GeneratedToolDraft`: append its validated handler to the
    target knowledge base's `mcp_tools.py`, then register it in that same
    knowledge base's `catalog.py` `TOOLS` seed list, via the same `_skill(...)`
    call shape every other entry uses (`libcst`, appended as a new element,
    not a full-file rewrite).

    Both edits land inside the knowledge base that owns the tool, because a
    tool is part of its ontology package: `bootstrap_catalog()` composes the
    per-package `TOOLS` lists, so a product that does not install the package
    does not get the generated tool either. Registering into a shared central
    list instead would leak one package's tool into every product.

    `repo_root` is read from the module-level global at call time (never bound
    as the caller's own default), the same convention
    `apply_skill_wording`/`apply_routing_capability_wording` use, so a test
    (or a future override) can point this at a throwaway fixture tree.

    Returns both edited paths, in the order they were written (handler
    file first) -- so a caller that needs to discard a failed attempt
    (e.g. `new_tool`'s test-gate failing, see
    `knowledge_bases.tools.self_improvement_apply`) knows exactly which
    files to revert. Raises `SourceEditError`/`ValueError` without writing
    the catalog registration if the handler itself fails validation; never
    leaves a registered `Tool` pointing at a handler that was rejected.
    """
    if "." not in draft.key.strip():
        raise ValueError(f"Generated skill key {draft.key!r} must be '<knowledge_base>.<name>'")
    knowledge_base = draft.key.split(".", 1)[0]
    _require_generated_tool_target(knowledge_base)
    mcp_tools_path = _mcp_tools_path(knowledge_base, repo_root)
    if not mcp_tools_path.exists():
        raise SourceEditError(
            f"No existing knowledge_bases/{knowledge_base}/mcp_tools.py to append to "
            f"(new_tool can only extend an existing knowledge base)"
        )
    catalog_path = _catalog_path(knowledge_base, repo_root)
    if not catalog_path.exists():
        raise SourceEditError(
            f"No existing knowledge_bases/{knowledge_base}/catalog.py to register in "
            f"(new_tool can only extend an installed ontology package)"
        )

    append_handler(draft.handler_source, draft.handler_function_name, mcp_tools_path)
    _register_skill_in_catalog(draft, knowledge_base, path=catalog_path)
    return [mcp_tools_path, catalog_path]


def _append_list_call_element(lst: cst.List, call: cst.Call) -> cst.List:
    """Append `call` to `lst` with comma formatting matching the last element,
    a `Call` expression (a new `_skill(...)` entry) instead of a string
    literal -- see that function's docstring for why the separator/
    trailing comma distinction matters."""
    new_element = cst.Element(value=call)
    if not lst.elements:
        return lst.with_changes(elements=[new_element])

    old_last = lst.elements[-1]
    trailing_comma = old_last.comma
    if len(lst.elements) >= 2:
        separator_comma = lst.elements[-2].comma
    else:
        indent = (
            lst.lbracket.whitespace_after.last_line
            if isinstance(lst.lbracket.whitespace_after, cst.ParenthesizedWhitespace)
            else cst.SimpleWhitespace("")
        )
        separator_comma = cst.Comma(
            whitespace_after=cst.ParenthesizedWhitespace(
                first_line=cst.TrailingWhitespace(newline=cst.Newline()),
                indent=True,
                last_line=indent,
            )
        )
    updated_old_last = old_last.with_changes(comma=separator_comma)
    new_element = new_element.with_changes(comma=trailing_comma)
    return lst.with_changes(elements=[*lst.elements[:-1], updated_old_last, new_element])


def _skill_call_node(draft: GeneratedToolDraft, knowledge_base: str) -> cst.Call:
    """Build the `_skill(key, knowledge_base, name, description,
    input_schema, handler)` call expression for a generated skill,
    matching the shape every hand-written catalog entry already uses (see
    `_skill()` in `knowledge_bases/tools/catalog.py`)."""
    import json

    handler = (
        f"knowledge_bases.{knowledge_base}.mcp_tools:{draft.handler_function_name}"
    )
    schema_source = json.dumps(draft.input_schema)
    args_source = (
        f"_skill(\n"
        f"    {_quoted(draft.key)},\n"
        f"    {_quoted(knowledge_base)},\n"
        f"    {_quoted(draft.name)},\n"
        f"    {_quoted(draft.description)},\n"
        f"    {schema_source},\n"
        f"    {_quoted(handler)},\n"
        f")"
    )
    # Parse the call as a standalone expression rather than hand-building
    # every Arg/Comma node -- far less error-prone for a six-argument call,
    # and the whole call becomes one new list element regardless of its
    # own internal formatting (see `_append_list_call_element`).
    parsed = cst.parse_module(args_source).body[0]
    return parsed.body[0].value


class _ToolsListAppender(cst.CSTTransformer):
    """Append one `_skill(...)` call to a package's module-level `TOOLS` list.

    Each ontology package owns its seed slice as `TOOLS` in
    `knowledge_bases/<kb>/catalog.py` (composed by `bootstrap_catalog()`), so a
    generated tool is registered with its own knowledge base rather than in a
    single shared list.
    """

    def __init__(self, call: cst.Call) -> None:
        self.call = call
        self.found = False

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module,
    ) -> cst.Module:
        body = list(updated_node.body)
        for index, statement in enumerate(body):
            if not isinstance(statement, cst.SimpleStatementLine):
                continue
            new_subs = []
            changed = False
            for sub in statement.body:
                if (
                    isinstance(sub, cst.Assign)
                    and len(sub.targets) == 1
                    and isinstance(sub.targets[0].target, cst.Name)
                    and sub.targets[0].target.value == "TOOLS"
                    and isinstance(sub.value, cst.List)
                ):
                    self.found = True
                    changed = True
                    new_subs.append(
                        sub.with_changes(
                            value=_append_list_call_element(sub.value, self.call)
                        )
                    )
                else:
                    new_subs.append(sub)
            if changed:
                body[index] = statement.with_changes(body=new_subs)
        return updated_node.with_changes(body=body)


def _catalog_path(knowledge_base: str, repo_root: Path) -> Path:
    return repo_root / "knowledge_bases" / knowledge_base / "catalog.py"


def _register_skill_in_catalog(
    draft: GeneratedToolDraft, knowledge_base: str, path: Path,
) -> None:
    """Add the generated tool to its own knowledge base's `TOOLS` seed list.

    `path` is that package's `catalog.py` -- the registration lands with the
    package that owns the handler, so a product that does not install the
    package does not get the tool either.
    """
    if not draft.key.strip():
        raise ValueError("Generated skill draft has no key")
    source = path.read_text()
    tree = cst.parse_module(source)
    existing_keys = {
        node.args[0].value.evaluated_value
        for node in _find_skill_calls(tree)
        if node.args and isinstance(node.args[0].value, cst.SimpleString)
    }
    if draft.key in existing_keys:
        raise SourceEditError(f"Tool key {draft.key!r} is already registered in {path}")

    call = _skill_call_node(draft, knowledge_base)
    transformer = _ToolsListAppender(call)
    updated = tree.visit(transformer)
    if not transformer.found:
        raise SourceEditError(f"Could not find the TOOLS seed list in {path}")
    path.write_text(updated.code)


def _find_skill_calls(module: cst.Module) -> list[cst.Call]:
    calls = []

    class _Collector(cst.CSTVisitor):
        def visit_Call(self, node: cst.Call) -> None:
            if isinstance(node.func, cst.Name) and node.func.value == "_skill":
                calls.append(node)

    module.visit(_Collector())
    return calls


def apply_suggestion(suggestion: SelfImprovementSuggestion) -> list[Path]:
    """Apply one `new_tool` suggestion as a source edit.

    Returns every file path that was edited (for the caller to stage/
    commit/discard -- see `knowledge_bases.tools.self_improvement_apply`),
    in the order they were written. Raises `SourceEditError`/`ValueError`
    without writing anything if the suggestion has no generated handler or
    it fails validation; never applies a partial edit -- the handler file is
    only ever written *after* it passes `_validate_handler_source`, and the
    catalog registration only after the handler file itself.

    Every other kind is rejected here: wording kinds are graph edits
    (`knowledge_bases.tools.graph_edits`) and `architecture` is always
    routed to human review (see `docs/architecture/self_improvement.md`).
    """
    if suggestion.kind == SelfImprovementKind.new_tool:
        if suggestion.generated_skill is None or not suggestion.generated_skill.handler_source.strip():
            raise ValueError(
                f"Suggestion for {suggestion.target!r} has no generated_skill handler to apply"
            )
        return generate_new_skill(suggestion.generated_skill)
    raise ValueError(f"No source-edit strategy for suggestion kind {suggestion.kind!r}")
