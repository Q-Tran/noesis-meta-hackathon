# Knowledge Base Bridges

Cross-domain edges are declared, not convention. Each row below is an
`EdgeContract` (`knowledge_bases/base.py`) declared by the package that
DESCRIBES the edge, validated at composition (`registry.compose`), and read
through `registry.edge_contracts()`. This table is pinned by
`tests/test_edge_contracts.py`: it is generated content, and a row that
drifts from the contracts fails the suite.

| Relationship | From | To | Writers | Describer |
|---|---|---|---|---|
| `MAPS_TO` | BusinessTerm, AlternateBusinessTerm | Metric, Table, Column | business_dictionary | business_dictionary |
| `SUPPORTS` | Metric, SourceNode | Claim, Decision | epistemology, semantic_layer, provenance | epistemology |
| `CHALLENGES` | Claim, SourceNode | Decision, Claim | epistemology, provenance | epistemology |
| `ASSERTS` | SourceNode | Claim | provenance | epistemology |
| `IMPLEMENTS` | Architecture, Project | Capability, Decision | architecture, project_management | architecture |
| `FOUND_IN` | * | Chunk, RoutingTrace, SourceNode | * | provenance |

A contract whose endpoint labels no installed package supplies is INERT: the
declaration stays (the enterprise product needs it), the edge is unwritable
in this product, and `registry.inert_edge_contracts()` reports it. A
narrower product accepts inert edges -- `strategy.MEASURED_BY` and
`architecture.REALIZES` are the personal product's dead edges, pruned from
the extraction prompt by `drop_unsatisfiable`. Two describers for one edge,
or a describer that does not declare the edge on its own knowledge base, is
a packaging mistake, and composition raises.

Domain MCP servers may read these edges. Cross-domain writes go through the
typed services of the writing package (cascade steps, write commands,
changeset relationships), never through generic Cypher or graph upserts.
