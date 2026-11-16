"""The facts a conformance rule is allowed to look at.

A rule may only read the package under test and whatever peer packages the
caller chose to supply. It must not reach for `registry.KNOWLEDGE_BASES` or any
other composed global, because the whole point of the kit is to run in a
repository where the composition is somebody else's and mostly absent. Every
composed-registry fact a rule needs is therefore derived here from the declared
objects, and "I could not decide this without peers" is a reportable outcome
rather than a crash.

`PackageContext.load` accepts the three things a caller plausibly has: the
`OntologyPackage` object itself, a `"module.path:ATTR"` reference to it, or a
package key resolvable through the installed registry.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import pathlib
from dataclasses import dataclass, field
from typing import Any, Iterable

from knowledge_bases.base import (
    CascadeStep,
    EntityType,
    OntologyPackage,
    PropertyType,
    RelationshipType,
    VersionedKind,
    property_names,
)

# Packages that ship with the engine rather than as domain ontologies, so
# importing one is a dependency on core and not on a sibling ontology. Named in
# `knowledge_bases/registry.py`'s module docstring: "Like `tools`,
# `provenance` is cross-cutting infrastructure rather than a user-routable
# expert". A product distribution overrides this if it draws the line
# elsewhere.
CONTROL_PLANE_PACKAGES: frozenset[str] = frozenset(
    {"tools", "orchestration", "provenance"}
)

# `PropertyType.value_type` values the engine knows how to write. Taken from
# the field comment in `knowledge_bases/base.py`: "`str`, `int`, `float`,
# `bool`, `list`, or an ISO-8601 `datetime`".
LEGAL_VALUE_TYPES: frozenset[str] = frozenset(
    {"str", "int", "float", "bool", "list", "datetime"}
)


class SeedResolutionError(Exception):
    """A `"module.path:ATTR"` reference that does not resolve.

    Carried rather than raised out of the kit: an unresolvable seed is a
    finding about the package, not a bug in the checker.
    """


def resolve_seed(reference: str) -> Any:
    """Resolve a `"module.path:ATTR"` seed reference the way the engine does.

    Deliberately re-implemented instead of imported from `registry`, so the kit
    keeps working in a repository whose engine version moved that helper.
    """
    module_path, separator, attr = reference.partition(":")
    if not separator or not module_path or not attr:
        raise SeedResolutionError(
            f"{reference!r} is not a 'module.path:ATTR' reference"
        )
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001 - reported, not handled
        raise SeedResolutionError(f"importing {module_path!r} failed: {exc}") from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise SeedResolutionError(
            f"{module_path!r} has no attribute {attr!r}"
        ) from exc


def _declaration_from_key(key: str) -> OntologyPackage:
    """Load `<key>`'s declaration without importing the knowledge base.

    Mirrors `registry._load_declaration`: a package declaration must compose
    even when the knowledge base's `__init__` would pull in half a codebase.
    Falls back to the installed registry when the file layout differs, which is
    what an installed distribution looks like.
    """
    # Asks the registry WHERE the package is rather than guessing one directory.
    # `knowledge_bases` spans several distributions after the split (`pkgutil.extend_path`),
    # so `parent.parent` only ever sees the engine's own directory and found nothing for
    # every separately distributed ontology -- while working fine in a single tree.
    from knowledge_bases.registry import _declaration_path

    path = _declaration_path(key)
    if path is not None and path.exists():
        spec = importlib.util.spec_from_file_location(f"_conformance_pkg_{key}", path)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.PACKAGE
    from knowledge_bases.registry import _load_declaration

    return _load_declaration(key)


def _as_package(candidate: OntologyPackage | str) -> OntologyPackage:
    if isinstance(candidate, OntologyPackage):
        return candidate
    if ":" in candidate:
        resolved = resolve_seed(candidate)
        if not isinstance(resolved, OntologyPackage):
            raise TypeError(
                f"{candidate!r} resolved to {type(resolved).__name__}, "
                "not an OntologyPackage"
            )
        return resolved
    return _declaration_from_key(candidate)


@dataclass(frozen=True)
class PackageContext:
    """One package under test, its peers, and the derived views rules read."""

    package: OntologyPackage
    peers: tuple[OntologyPackage, ...] = ()
    control_plane: frozenset[str] = CONTROL_PLANE_PACKAGES
    source_root: pathlib.Path | None = None
    # Seed references that failed to resolve, so every later rule can skip the
    # slice it cannot see instead of each one re-raising the same error.
    seed_errors: dict[str, str] = field(default_factory=dict)

    # -- construction ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        package: OntologyPackage | str,
        *,
        peers: Iterable[OntologyPackage | str] | None = None,
        control_plane: Iterable[str] | None = None,
        source_root: pathlib.Path | str | None = None,
    ) -> "PackageContext":
        """Build a context, resolving keys and references to declarations.

        `peers=None` means "ask the installed registry for its other packages",
        which is what makes the shared-ownership rules meaningful inside a repo
        that ships several. A foreign package with no peers to compare against
        passes `peers=()` and gets those rules reported as undecidable rather
        than as passes.
        """
        under_test = _as_package(package)
        if peers is None:
            resolved_peers = _installed_peers(under_test.key)
        else:
            resolved_peers = tuple(_as_package(peer) for peer in peers)
        resolved_peers = tuple(
            peer for peer in resolved_peers if peer.key != under_test.key
        )
        root = pathlib.Path(source_root) if source_root is not None else None
        if root is None:
            # The package's own directory, wherever it is installed. Derived from
            # the declaration's location for the same reason as above: the source
            # rules (sibling imports, private-helper reach) silently check NOTHING
            # when this is None, so a wrong guess turns real rules vacuous instead
            # of failing.
            from knowledge_bases.registry import _declaration_path

            located = _declaration_path(under_test.key)
            root = located.parent if located is not None and located.parent.is_dir() else None
        return cls(
            package=under_test,
            peers=resolved_peers,
            control_plane=frozenset(
                control_plane if control_plane is not None else CONTROL_PLANE_PACKAGES
            ),
            source_root=root,
        )

    # -- what this package declares -------------------------------------------

    @property
    def key(self) -> str:
        return self.package.key

    @property
    def has_peers(self) -> bool:
        return bool(self.peers)

    @property
    def declared_labels(self) -> frozenset[str]:
        return frozenset(
            label
            for kb in self.package.knowledge_bases
            for label in kb.labels
        )

    @property
    def declared_relationships(self) -> frozenset[str]:
        return frozenset(
            relationship
            for kb in self.package.knowledge_bases
            for relationship in kb.relationships
        )

    # -- what this package describes ------------------------------------------

    def _seeded(self, field_name: str) -> tuple[Any, ...]:
        reference = getattr(self.package, field_name)
        if not reference:
            return ()
        if field_name in self.seed_errors:
            return ()
        try:
            resolved = resolve_seed(reference)
        except SeedResolutionError as exc:
            self.seed_errors[field_name] = str(exc)
            return ()
        try:
            return tuple(resolved)
        except TypeError:
            self.seed_errors[field_name] = (
                f"{reference!r} resolved to {type(resolved).__name__}, "
                "which is not iterable"
            )
            return ()

    @property
    def entity_types(self) -> tuple[EntityType, ...]:
        return self._seeded("entity_types_seed")

    @property
    def relationship_types(self) -> tuple[RelationshipType, ...]:
        return self._seeded("relationship_types_seed")

    @property
    def cascade_steps(self) -> tuple[CascadeStep, ...]:
        return self._seeded("cascade_steps_seed")

    @property
    def kinds(self) -> tuple[VersionedKind, ...]:
        return self._seeded("kinds_seed")

    @property
    def tools(self) -> tuple[Any, ...]:
        return self._seeded("tools_seed")

    @property
    def experts(self) -> tuple[Any, ...]:
        return self._seeded("experts_seed")

    @property
    def write_commands(self) -> tuple[Any, ...]:
        return self._seeded("write_commands_seed")

    @property
    def migrations(self) -> tuple[Any, ...]:
        return self._seeded("migration_stream")

    @property
    def described_labels(self) -> dict[str, EntityType]:
        return {entity.label: entity for entity in self.entity_types}

    @property
    def described_relationships(self) -> dict[str, RelationshipType]:
        return {rel.relationship: rel for rel in self.relationship_types}

    # -- what the peers declare and describe ----------------------------------

    def _peer_view(self, attribute: str, name_of) -> dict[str, list[str]]:
        view: dict[str, list[str]] = {}
        for peer in self.peers:
            for item in _peer_items(peer, attribute):
                view.setdefault(name_of(item), []).append(peer.key)
        return view

    @property
    def peer_declared_labels(self) -> dict[str, list[str]]:
        view: dict[str, list[str]] = {}
        for peer in self.peers:
            for kb in peer.knowledge_bases:
                for label in kb.labels:
                    view.setdefault(label, []).append(peer.key)
        return view

    @property
    def peer_declared_relationships(self) -> dict[str, list[str]]:
        view: dict[str, list[str]] = {}
        for peer in self.peers:
            for kb in peer.knowledge_bases:
                for relationship in kb.relationships:
                    view.setdefault(relationship, []).append(peer.key)
        return view

    @property
    def peer_described_labels(self) -> dict[str, list[str]]:
        return self._peer_view("entity_types_seed", lambda e: e.label)

    @property
    def peer_described_relationships(self) -> dict[str, list[str]]:
        return self._peer_view(
            "relationship_types_seed", lambda r: r.relationship
        )

    @property
    def peer_tool_keys(self) -> dict[str, str]:
        keys: dict[str, str] = {}
        for peer in self.peers:
            for tool in _peer_items(peer, "tools_seed"):
                key = getattr(tool, "key", None)
                if key:
                    keys[key] = peer.key
        return keys

    def known_label(self, label: str) -> bool:
        return label in self.declared_labels or label in self.peer_declared_labels

    def known_relationship(self, relationship: str) -> bool:
        return (
            relationship in self.declared_relationships
            or relationship in self.peer_declared_relationships
        )

    # -- source-level facts ---------------------------------------------------

    def sibling_package_keys(self) -> frozenset[str]:
        """Which `knowledge_bases.<X>` names are ONTOLOGY PACKAGES, not engine.

        The discriminator is the declaration file: a package is a directory
        holding `package.py`, while `registry.py`, `mcp.py`, `triplets.py`,
        `ontology.py` and `ingestion*.py` are flat engine modules in the same
        namespace. Without this distinction every package looks coupled to every
        sibling, because all of them legitimately import the engine.
        """
        keys = {peer.key for peer in self.peers} | set(self.control_plane)
        if self.source_root is not None:
            parent = self.source_root.parent
            if parent.is_dir():
                keys |= {
                    child.name
                    for child in parent.iterdir()
                    if child.is_dir() and (child / "package.py").exists()
                }
        return frozenset(keys - {self.key})

    def imported_packages(self) -> dict[str, list[str]]:
        """Absolute imports of a SIBLING ONTOLOGY PACKAGE from this package.

        AST-parsed rather than grepped so a mention inside a docstring or a
        string literal does not count as a dependency. Imports of engine modules
        (`knowledge_bases.registry`, `.mcp`, `.triplets`, ...) are not
        dependencies on another ontology and are excluded -- see
        `sibling_package_keys`. Returns `{other_package: ["path:line", ...]}`.
        """
        found: dict[str, list[str]] = {}
        if self.source_root is None or not self.source_root.is_dir():
            return found
        # The namespace an import STATEMENT uses, which is not the directory name.
        # These diverge exactly where it matters: the ontologies monorepo organises
        # packages under `packages/<key>` but they install and import as
        # `knowledge_bases.<key>`, so deriving the namespace from the parent directory
        # looked for `packages.semantic_layer` and found no sibling imports anywhere.
        # The rule ran, reported clean, and the import it exists to catch was untouched
        # three lines into the file.
        #
        # `knowledge_bases` is therefore taken as the namespace, with the directory name
        # kept as an alternative so a package laid out some third way still resolves.
        namespaces = {"knowledge_bases", self.source_root.parent.name}
        siblings = self.sibling_package_keys()
        for path in sorted(self.source_root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text())
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module or ""]
                for module in modules:
                    parts = module.split(".")
                    if len(parts) < 2 or parts[0] not in namespaces:
                        continue
                    other = parts[1]
                    if other == self.key or other not in siblings:
                        continue
                    found.setdefault(other, []).append(
                        f"{path.as_posix()}:{node.lineno}"
                    )
        return found


def _peer_items(peer: OntologyPackage, attribute: str) -> tuple[Any, ...]:
    reference = getattr(peer, attribute, None)
    if not reference:
        return ()
    try:
        return tuple(resolve_seed(reference))
    except (SeedResolutionError, TypeError):
        # A broken peer is the peer's conformance problem, not this package's.
        return ()


def _installed_peers(exclude: str) -> tuple[OntologyPackage, ...]:
    """The other packages the local product installs, if there is one.

    Wrapped in a try/except because a standalone ontology distribution may have
    no product profile at all -- that is the normal state of the repository this
    kit is designed to be USED from, and it must degrade to "no peers" rather
    than fail to import.
    """
    try:
        from knowledge_bases.registry import DEFAULT_PACKAGES, ontology_packages

        return tuple(
            package
            for package in ontology_packages(DEFAULT_PACKAGES)
            if package.key != exclude
        )
    except Exception:  # noqa: BLE001 - absence of a product is legitimate
        return ()


def described_property_types(
    properties: tuple[str | PropertyType, ...],
) -> tuple[PropertyType, ...]:
    return tuple(prop for prop in properties if isinstance(prop, PropertyType))


def declared_property_names(
    properties: tuple[str | PropertyType, ...],
) -> frozenset[str]:
    return frozenset(property_names(properties))
