"""What the Provenance package's label and relationship mean.

Provenance is the cross-cutting epistemology layer: exactly one node type
(`SourceNode`) and one edge (`FOUND_IN`), which every other knowledge base
points outward at to record where its data came from, instead of keeping a flat
`source_type`/`url` string property.

Grounded in `models.py` (metadata only -- never the source's content or an
excerpt), `queries.py::register_source`/`trace_sources`,
`graph.py::attach_source` (upserts the SourceNode and MERGEs
`(n)-[:FOUND_IN {role}]->(:SourceNode)`), and its real writers:
`extract.py` (`role: "extracted"` from Person/BusinessTerm/Table/Column/Value/
Claim/Decision/Task/Project/Epic), `knowledge_bases/ingestion.py
(`role: "origin"` from whatever label is being ingested),
`knowledge_bases/semantic_layer/{metric_registry,import_powerbi}.py`
(`role: "description"`, `"hosted_by"`), `knowledge_bases/development/persist.py`
(CodeRepository), and `knowledge_bases/orchestration/memory.py`.

`FOUND_IN` is declared by `semantic_layer` and `development` as well, because a
relationship is declared by every package whose entities can produce it; it is
described here because `provenance` owns it (see
`registry.relationship_types()`, which permits exactly one describing package).
"""
from knowledge_bases.base import EntityType, RelationshipType


ENTITY_TYPES = (
    EntityType(
        label="SourceNode",
        description=(
            "The origin any fact in the graph can be traced back to: a connector, "
            "system, document store, LLM run, extraction job, or the user who "
            "said it, with a reliability rating and when it was observed or "
            "published. It holds metadata about the origin only -- never the "
            "source's text -- and it is the origin itself, not the individual "
            "document read from it (DocumentNode) nor the run that processed one "
            "delivery (IngestionEvent)."
        ),
        identity="id",
        properties=(
            "id", "name", "type", "reliability", "url", "observed_at",
            "published_at", "identity_status", "created_at", "updated_at",
        ),
        not_to_be_confused_with=("DocumentNode", "IngestionEvent", "Person"),
    ),
    EntityType(
        label="Chunk",
        description=(
            "A discrete excerpt or segment of unstructured text from a source "
            "document, message, or file. Vectorized into Milvus via DiskANN to "
            "provide a searchable provenance layer grounding extracted entities "
            "and relationships back to the verbatim source material."
        ),
        identity="id",
        extractable=False,
        properties=(
            "id", "text", "source_id", "document_id", "index", "created_at",
            "identity_status", "updated_at",
        ),
        not_to_be_confused_with=("SourceNode", "DocumentNode"),
    ),
)


RELATIONSHIP_TYPES = (
    RelationshipType(
        relationship="FOUND_IN",
        description=(
            "Provenance: this entity is known because it was found in that "
            "source, with a `role` property naming which aspect the source backs "
            "(origin, extracted, description, definition, hosted_by), so one node "
            "may cite several sources for different aspects. Targets: a SourceNode "
            "(the origin metadata), a Chunk (verbatim document text), or a "
            "RoutingTrace (one conversation turn whose recorded question is the "
            "verbatim words a user-sourced claim was made from). It records where "
            "knowledge came from, not that the source vouches for a statement "
            "(ASSERTS, SUPPORTS) and not physical or hierarchical containment "
            "(CONTAINS, PART_OF, BELONGS_TO, COLUMN_OF)."
        ),
        # Deliberately open: `graph.attach_source` accepts any label, and the
        # real writers span semantic_layer, claims, decisions,
        # project_management, business_dictionary, development and
        # orchestration -- a narrower list would make an existing write illegal.
        subject=(),
        object=("SourceNode", "Chunk", "RoutingTrace",),
        not_to_be_confused_with=(
            "ASSERTS", "SUPPORTS", "AUTHORED", "CONTAINS", "PART_OF",
            "BELONGS_TO", "DERIVED_FROM",
        ),
    ),
    RelationshipType(
        relationship="EXTRACTED_FROM",
        description=(
            "Provenance: this entity was extracted from a specific text chunk, "
            "grounding the entity in verbatim source content."
        ),
        subject=(),
        object=("Chunk",),
        extractable=False,
        not_to_be_confused_with=("FOUND_IN", "PART_OF", "ASSERTS"),
    ),
    RelationshipType(
        relationship="MENTIONS",
        description=(
            "Provenance: this text chunk mentions or discusses the target entity."
        ),
        subject=("Chunk",),
        object=(),
        extractable=False,
        not_to_be_confused_with=("ASSERTS", "SUPPORTS"),
    ),
)
