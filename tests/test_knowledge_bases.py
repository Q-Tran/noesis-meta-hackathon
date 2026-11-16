"""Registry invariants that hold for the engine alone.

No optional ontology is imported here on purpose: these are statements about
composition itself -- label exclusivity, shared-relationship declaration, package
discovery -- so they must be runnable by a distribution that installs no ontology.
The versioned-kind invariants that DO need `methodology` live in
tests/test_versioned_kind_ownership.py.
"""
import pytest

from knowledge_bases.registry import KNOWLEDGE_BASES


def test_semantic_layer_and_dictionary_are_distinct_boundaries():
    semantic = KNOWLEDGE_BASES["semantic_layer"]
    dictionary = KNOWLEDGE_BASES["business_dictionary"]

    assert semantic.labels == ("Table", "Column", "Value", "Metric")
    assert dictionary.labels == ("BusinessTerm", "AlternateBusinessTerm")
    assert not set(semantic.labels) & set(dictionary.labels)
    assert "MAPS_TO" not in semantic.relationships
    assert "MAPS_TO" in dictionary.relationships


def test_every_knowledge_base_has_unique_owned_labels():
    owners = {}
    for key, kb in KNOWLEDGE_BASES.items():
        for label in kb.labels:
            assert label not in owners, f"{label} owned by {owners[label]} and {key}"
            owners[label] = key


def test_two_packages_may_not_declare_the_same_label():
    """Silent co-ownership is worse than a refused install.

    `compose` already refused a duplicate knowledge-base KEY, but not a duplicate
    LABEL, so two packages could both declare `Task` and composition succeeded. The
    result is a label with two owners: two descriptions competing for the extraction
    prompt, and two tool surfaces writing the same nodes under different rules.

    This is the real case that exposed it. A consumer-scale `tasks` ontology is a
    lighter ALTERNATIVE to `project_management`, and both want the name `Task`. Being
    alternatives, they must be mutually exclusive -- and the collision is what makes
    them so, but only if it raises. Otherwise the product looks composable and
    writes contradictory data.
    """
    from knowledge_bases.base import KnowledgeBase, OntologyPackage
    from knowledge_bases.registry import compose, ontology_packages

    lighter_tasks = OntologyPackage(
        key="tasks",
        knowledge_bases=(
            KnowledgeBase(
                key="tasks",
                name="Tasks",
                description="A personal task list.",
                labels=("Task",),
                relationships=(),
            ),
        ),
    )

    with pytest.raises(ValueError, match="declared by both"):
        compose(ontology_packages() + (lighter_tasks,))


def test_a_shared_relationship_may_be_declared_by_several_packages():
    """The deliberate asymmetry: labels are exclusive, relationship types are not.

    `DEPENDS_ON` is declared by three packages on purpose -- a Task depends on a
    Task, a Capability on a Capability -- and the edge means the same thing in each.
    Only its DESCRIPTION has a single owner, which `relationship_types()` enforces
    separately. Treating a shared edge like a colliding label would make the
    packages that share it un-installable together.
    """
    from knowledge_bases.registry import KNOWLEDGE_BASES, compose, ontology_packages

    sharers = [
        key for key, kb in KNOWLEDGE_BASES.items() if "DEPENDS_ON" in kb.relationships
    ]

    assert len(sharers) > 1, "expected DEPENDS_ON to be shared; pick another edge"
    composed = compose(ontology_packages())
    assert all("DEPENDS_ON" in composed[key].relationships for key in sharers)


def test_a_separately_installed_ontology_is_found_through_its_entry_point(tmp_path):
    """The mechanism that makes `noesis-core` shippable without any ontology.

    `_load_declaration` used to resolve `knowledge_bases/<key>/package.py` relative to
    the registry's own directory, so a package could only ever be found if it sat
    physically beside the engine. That is precisely what a separate distribution does
    NOT do: an installed `noesis-ontology-tasks` lands in site-packages, nowhere near
    `noesis_core/knowledge_bases/`, and would have been invisible.

    Simulated with a real out-of-tree directory and a fake entry point rather than by
    monkeypatching `_declaration_path`, because patching the lookup would test nothing
    -- the lookup IS the thing under test.
    """
    import importlib.metadata
    import sys
    import types

    from knowledge_bases import registry

    external = tmp_path / "noesis_ontology_probe"
    external.mkdir()
    (external / "__init__.py").write_text("")
    (external / "package.py").write_text(
        "from knowledge_bases.base import KnowledgeBase, OntologyPackage\n"
        "PACKAGE = OntologyPackage(key='probe', knowledge_bases=(KnowledgeBase(\n"
        "    key='probe', name='Probe', description='An out-of-tree ontology.',\n"
        "    labels=('ProbeThing',), relationships=()),))\n"
    )
    sys.path.insert(0, str(tmp_path))

    class _Entry:
        name = "probe"
        value = "noesis_ontology_probe"

    real_entry_points = importlib.metadata.entry_points

    def fake_entry_points(*args, **kwargs):
        if kwargs.get("group") == "noesis.ontologies":
            return [_Entry()]
        return real_entry_points(*args, **kwargs)

    importlib.metadata.entry_points = fake_entry_points
    registry._load_declaration.cache_clear()
    try:
        declaration = registry._load_declaration("probe")
    finally:
        importlib.metadata.entry_points = real_entry_points
        registry._load_declaration.cache_clear()
        sys.path.remove(str(tmp_path))
        sys.modules.pop("noesis_ontology_probe", None)

    assert declaration.key == "probe"
    assert declaration.knowledge_bases[0].labels == ("ProbeThing",)


def test_an_uninstalled_package_names_itself_and_where_it_was_sought():
    """A product profile listing a package nobody installed must say so usefully.

    The old message was "Ontology package 'x' has no package.py at <path>", which
    reads as a corrupt install rather than a missing dependency and names only one of
    the two places now searched.
    """
    from knowledge_bases import registry

    with pytest.raises(ValueError) as failure:
        registry._load_declaration("no_such_ontology")

    message = str(failure.value)
    assert "not installed" in message
    assert "noesis.ontologies" in message, "the message must name the entry-point group"
    assert "product.py" in message, "the message must say how to fix it"
