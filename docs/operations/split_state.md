# Noesis core/enterprise/consumer split — completed state

Updated 2026-09-30 after Phase A (one-off migration code removed) and Phase B
(domain packages moved out of the enterprise tree). The 2026-09-19 audit this
replaced described a single monorepo; that layout no longer exists.

## Four repositories

| Repo | Role | Contents |
|---|---|---|
| `noesis-core` | the engine, ontology-free | root modules (`graph.py`, `extract.py`, `worker.py`, ...), control-plane knowledge bases (`tools`, `orchestration`, `provenance`, `conformance`, `bridges`), the conformance kit. No `noesis.ontologies` entry points. |
| `noesis-ontologies` | the domain vocabularies | nine packages under `packages/<key>`, installed as `knowledge_bases.<key>`; each owns its labels, tools, experts, kinds, write commands, migrations, cascade steps, capabilities. Declares the `noesis.ontologies` entry-point group. |
| `noesis` | the enterprise product | the engine modules, the control plane, `knowledge_bases/product.py` (the enterprise profile), the scripts, and the README. No domain package sources; depends on `noesis-ontologies`. |
| `noesis-individual` | the personal product | a profile (control plane + `development`, `tasks`, `strategy`, `methodology`, `architecture`, `claims`), its worker, and its own graph. Owns no vocabulary. |

## The seam

`knowledge_bases` is a `pkgutil.extend_path` package. The import path
`knowledge_bases.<key>` is the contract; the installation directory is not.
`registry._declaration_path` resolves a package in-tree first, then through the
`noesis.ontologies` entry-point group. A bare checkout of `noesis-individual`
(or `noesis-ontologies`) appends the sibling `packages/` directory to
`knowledge_bases.__path__` in `conftest.py`; installed products get the same
span from site-packages.

The enterprise `pyproject.toml` declares no entry points: `noesis-ontologies`
declares all nine, including `tasks`, which the enterprise profile does not
install (`project_management` owns `Task` too, and `compose()` refuses two
owners).

## What moved

- The nine domain package directories were deleted from `noesis/knowledge_bases/`
  (they were byte-identical copies of `noesis-ontologies/packages/`).
- 30 domain test files that existed in both repos (byte-identical) were deleted
  from `noesis/tests/`; 5 domain-only tests (`test_live_graph_writes`,
  `test_open_codebase`, `test_strategy_decisions`, `test_task_runner`,
  `test_tools_repository`) moved to `noesis-ontologies/tests/`.
- `networkx` moved to `noesis-ontologies/requirements.txt` (the development
  package declares its own soft dependency).
- One-off migration scripts were removed: `retry_metric_naming*.py`,
  `backfill_source_relationships.py`, `merge_legacy_env.sh`, and the archived
  `business-brain-6f0d30.md`. The `ExternalInfo` fold-in they served is part of
  the `core` migration stream; every deployment that needed the backfill has
  run it.

## Historical measurements (2026-09-19 audit, preserved as provenance)

The ratchet baselines in `test_package_import_independence.py`,
`test_schema_compatibility.py`, `test_conformance_kit.py`, and
`test_conformance_against_installed_packages.py` cite this file. The audit measured:

- 26 core-to-domain imports, 5 of them in `schema.py` (the worst offender).
- 26 control-plane-to-domain import couplings -- the hardest remaining coupling,
  the direction that blocks extracting the engine.
- 6 control-plane modules failed to import without `methodology`, 7 without
  `claims`, including `schema.py`; 7 modules failed the capability seam
  (`orchestration` importing `claims`).
- Exactly one cross-ontology import: `business_dictionary` importing
  `semantic_layer.resolution`. It remains the single known real finding in
  `test_conformance_against_installed_packages.py`; the ratchet total is 1.
- The `DEPENDS_ON` defect: an edge described by a package a narrower product
  does not install leaves the edge undescribed for everyone who declares it.

Since the audit: `orchestration`'s six methodology write commands moved to
`methodology/write_commands.py`, the capability seam (`registry.capability`)
inverted the `orchestration`→`claims` dependency, and the domain packages left
the enterprise tree entirely, so the engine/ontology import direction is now
enforced by `test_core_is_ontology_free` and the entry-point seam rather than
by measurement.

## Edge contracts (2026-09-30)

The bridges README table is no longer a convention; it is generated content
pinned by `tests/test_edge_contracts.py`. Each cross-ontology edge is an
`EdgeContract` (`knowledge_bases/base.py`) declared by the package that
DESCRIBES it: `business_dictionary` (MAPS_TO), `claims` (SUPPORTS,
CHALLENGES, ASSERTS), `architecture` (IMPLEMENTS), `provenance`
(FOUND_IN, the `*` spine). `registry.compose()` raises on two describers
for one edge and on a describer that is not a declarer; contracts whose
endpoint labels no installed package supplies are inert, not errors.
`registry.edge_contracts()` / `inert_edge_contracts()` read the live profile.

The one tolerated cross-ontology import is gone: `semantic_layer` declares
the `resolution.resolve` capability and `business_dictionary` calls it by
name, with a local fallback when the capability is absent. The ratchet in
`test_conformance_against_installed_packages.py` is empty, and the
sibling-import rule now fires on a synthetic violation so it cannot go
blind on a clean tree.

## Test counts

    noesis            918 passed
    noesis-core       402 passed
    noesis-individual 445 passed, 3 skipped
    noesis-ontologies 741 passed

Layout constraints the suites enforce: `test_packaging_metadata` reads the entry-point
group from installed distributions; `test_readme_architecture` resolves diagram paths
against the sibling `noesis-ontologies` checkout; `test_mcp_json_serialization`
discovers servers through `registry._declaration_path`, not a disk glob;
`test_core_is_ontology_free` keeps `noesis-core` free of domain dirs and entry points.
