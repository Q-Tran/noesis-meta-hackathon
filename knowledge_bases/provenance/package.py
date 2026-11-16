"""Ontology package declaration for the Provenance knowledge base."""
from knowledge_bases.base import EdgeContract, KnowledgeBase, OntologyPackage, SearchType

PACKAGE = OntologyPackage(
    key="provenance",
    knowledge_bases=(
        KnowledgeBase(
            key="provenance",
            name="Provenance",
            description=(
                "Cross-cutting epistemology layer: the SourceNode any entity in any "
                "knowledge base points to via FOUND_IN to record where it came from "
                "(connector, LLM run, human edit, extraction, etc.)."
            ),
            labels=("SourceNode", "Chunk"),
            relationships=("FOUND_IN", "EXTRACTED_FROM", "MENTIONS"),
            # Cross-cutting infrastructure: no dispatchable tools of its own.
            call_noesis_tool=False,
            orchestrator_dispatch=False,
        ),
    ),
    search_types=(
        SearchType("provenance", "SourceNode", "evidence_metadata_text"),
        SearchType("provenance", "Chunk", None, "chunks"),
    ),
    entity_types_seed="knowledge_bases.provenance.ontology_types:ENTITY_TYPES",
    relationship_types_seed=(
        "knowledge_bases.provenance.ontology_types:RELATIONSHIP_TYPES"
    ),
    edge_contracts=(
        EdgeContract(
            relationship="FOUND_IN",
            from_labels=("*",),
            to_labels=("SourceNode", "Chunk", "RoutingTrace"),
            writers=("*",),
            description="Any entity in any knowledge base points to the "
                        "SourceNode it came from, the Chunk of verbatim text "
                        "it was extracted from, or the RoutingTrace (one "
                        "conversation turn) whose recorded question is its "
                        "verbatim evidence; the provenance spine.",
        ),
    ),
)
