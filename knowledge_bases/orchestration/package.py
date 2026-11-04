"""Ontology package declaration for the Orchestration knowledge base."""
from knowledge_bases.base import KnowledgeBase, OntologyPackage

PACKAGE = OntologyPackage(
    key="orchestration",
    knowledge_bases=(
        KnowledgeBase(
            key="orchestration",
            name="Orchestration",
            description=(
                "Mixture-of-experts control plane: routing, feedback cataloging, "
                "and bounded write propose/execute, itself exposed as a "
                "graph-registered Tool catalog like every delegated expert."
            ),
            labels=(
                "KnowledgeBaseExpert", "ExpertRevision", "ExpertCapability", "RoutingTrace",
                "WriteProposal", "Session", "DelegationResult",
                # The harness-facing instruction bundle. Owned here rather than by
                # `tools` because it is context an agent is *given*, not an
                # operation it can call.
                "AgentSkill",
                # A durable lesson from a session. Owned here because it is written
                # by the control plane and read back as context, like AgentSkill.
                "SessionMemory",
            ),
            relationships=(
                "HAS_CAPABILITY", "CAN_USE", "CURRENT", "SUPERSEDES", "CONSIDERED",
                "DELEGATED_TO", "HAS_TRACE", "RESULTED_IN",
                # AgentSkill -[:DOCUMENTS]-> Tool: the instruction bundle literally
                # contains the tools it teaches -- "tools live inside skills" made
                # graph-true at sync time (`agent_skills_catalog.sync_agent_skills`).
                "DOCUMENTS",
            ),
            # Orchestration tools are reachable via call_noesis_tool when gated behind search_noesis_tools.
            call_noesis_tool=True,
            orchestrator_dispatch=False,
        ),
    ),
    tools_seed="knowledge_bases.orchestration.catalog:TOOLS",
    experts_seed="knowledge_bases.orchestration.experts:EXPERTS",
    entity_types_seed="knowledge_bases.orchestration.ontology_types:ENTITY_TYPES",
    relationship_types_seed=(
        "knowledge_bases.orchestration.ontology_types:RELATIONSHIP_TYPES"
    ),
)
