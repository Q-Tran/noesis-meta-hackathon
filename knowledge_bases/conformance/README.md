# Ontology package conformance kit

`knowledge_bases/conformance/` is the suite an ontology package runs **against
itself** to prove it is a legal package before a product installs it.

It exists because of the planned split into `noesis-core` (the engine),
`noesis-ontologies` (the packages that ship here today), and product repos that
install some subset of them. After that split, the author of a package lives in
a different repository from this one. They cannot run this repo's tests, read its
history, or ask why a rule exists. So the kit is designed for that reader:

* every finding names the **package**, the **rule**, the **offending item**, a
  **detail** that explains what is wrong, and a **remedy** that says what to do;
* every rule's docstring names its **provenance** -- the `raise` in the engine,
  the test, or the recorded defect the invariant comes from;
* nothing is an `assert False`.

This directory is **engine code, not an ontology package**. It declares no
`package.py` and no `PACKAGE`, so it never enters `registry.DEFAULT_PACKAGES`
and adds nothing to any composed surface.

## Use

```python
from knowledge_bases.conformance import check_package

report = check_package(PACKAGE)              # peers default to the installed set
print(report.format())
assert report.is_conformant
```

One line of pytest integration, which gives one test per rule so a broken
package reports everything it breaks in a single run:

```python
from knowledge_bases.conformance import conformance_suite
from my_ontology.package import PACKAGE

TestConformance = conformance_suite(PACKAGE)
```

Check a package that is **not installed here** -- the case the kit is for -- by
passing its peers explicitly, or none at all:

```python
report = check_package(FOREIGN_PACKAGE, peers=())
```

Some rules cannot be decided without peers (an endpoint may point at a label
another package declares). Those report `INFO` rather than passing quietly, so a
clean report is never mistaken for a complete one.

## Severity

| Severity | Meaning |
| --- | --- |
| `ERROR` | Not a legal package. Something will fail, or fail silently. |
| `WARNING` | Legal, but incomplete on its own: it needs a specific peer. |
| `INFO` | Undecidable with the peers supplied, or a dependency worth seeing. |

The split is the point. If everything were an `ERROR`, the shared-edge findings
would make 8 of the 11 packages here illegal and the kit would be switched off.
If everything were a `WARNING`, nothing would ever fail.

## Layout

| Module | Contents |
| --- | --- |
| `findings.py` | `Severity`, `Finding`, `ConformanceReport`, `ConformanceError` |
| `context.py` | `PackageContext`: resolves seeds, declared/described views, peers, imports |
| `rules.py` | one function per rule; `ALL_RULES`, `RULES_BY_NAME` |
| `__init__.py` | `check_package`, `check_packages`, `installed_packages`, `conformance_suite` |

A rule is a plain function `(PackageContext) -> Iterable[Finding]`. Adding one
means adding a function to `rules.py` and appending it to `ALL_RULES`; the
generated pytest suite and the rule filter pick it up with no further wiring.

## Rules

Grouped as they are in `rules.py`. Each rule's own docstring carries the
provenance; the sources are the engine's `raise` sites, `__post_init__`
validation in `knowledge_bases/base.py`, existing tests under `tests/`, and
`docs/operations/split_state.md`.

**Declaration well-formedness** -- `package_declares_its_own_knowledge_base`,
`every_seed_reference_resolves`, `declared_types_are_well_formed_names`,
`the_declaration_module_imports_only_the_contract`,
`declared_migrations_are_in_this_packages_stream`,
`the_package_composes_standalone`.

**Describedness** -- `every_declared_label_is_described_by_this_package`,
`every_declared_relationship_is_described_by_this_package`,
`no_description_targets_a_type_this_package_does_not_declare`,
`descriptions_are_substantive`, `confusable_types_name_types_that_exist`.

**Endpoints and properties** --
`relationship_endpoints_reference_labels_that_exist`,
`an_open_endpoint_is_deliberate_and_explained`,
`an_identity_property_is_declared_as_a_property`,
`a_property_is_declared_once_per_type`, `declared_property_enums_are_usable`.

**Extractability and cascade** --
`an_extractable_relationship_has_an_extractable_endpoint`,
`cascade_steps_are_owned_and_produce_only_extractable_owned_labels`,
`a_cascade_step_writes_a_relationship_that_is_described`.

**Dispatch surfaces** -- `write_commands_are_namespaced_to_this_package`,
`contributed_tools_belong_to_this_package`,
`a_contributed_experts_entry_tools_are_reachable`,
`versioned_kinds_are_owned_and_declared`,
`changeset_relationships_are_declared_by_this_package`,
`search_types_target_this_packages_own_labels`,
`a_search_type_reads_the_labels_identity_property`.

**Independence** -- `the_package_does_not_import_a_sibling_ontology_package`.

## Findings on the packages that ship here

The kit was run over all 11 installed packages. Real failures were **reported,
not fixed and not designed around** -- none of the files involved belong to this
kit. `tests/test_conformance_kit.py` records them as an explicit expected set,
so fixing one makes that test fail and forces the list to shrink.

Three errors, all pre-existing:

1. **`semantic_layer.Table` is searchable and returns nothing.** Its
   `SearchType` keeps the default `identity_property="id"`, while
   `EntityType("Table")` converges on `name` and does not declare `id` at all.
   `hybrid_search` reads `row["properties"][identity_property]` and drops any
   candidate whose value is null, so every Table hit is discarded with no error
   logged. Found by the kit, not harvested from an existing test.
2. **`business_dictionary` imports `semantic_layer`** (`queries.py:5`) -- the one
   cross-ontology domain import `docs/operations/split_state.md` records.
3. **`orchestration` imports `methodology`** -- a core-to-domain import, in the
   class of couplings that blocks extracting the engine.

That ratchet has already caught a change: `tools` was a fourth entry, the
core-to-domain sever landed while this kit was being written, and the test failed
until the entry was removed.

The warnings are dominated by two shapes that are legal by design and reported so
they stay visible: shared edges under single ownership (`DEPENDS_ON`,
`CONFLICTS_WITH`, `FOUND_IN`, `SUPPORTS`, `CHALLENGES`, `ADDRESSES`, `ADVANCES`,
`IMPLEMENTS`), and `development`'s extractable edges between non-extractable
labels.
