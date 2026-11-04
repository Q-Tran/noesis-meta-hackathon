"""Ontology package declaration for the Tools & Telemetry knowledge base."""
from knowledge_bases.base import KnowledgeBase, OntologyPackage

PACKAGE = OntologyPackage(
    key="tools",
    knowledge_bases=(
        KnowledgeBase(
            key="tools",
            name="Tools & Telemetry",
            description=(
                "Bounded tool definitions plus the execution telemetry recorded "
                "against them: every dispatch is a Span node linked to the Tool it "
                "called, and TraceOutcome nodes record whether a trace's answer "
                "held up. Not to be confused with a harness Agent Skill: a Tool "
                "here is one callable operation with a handler and an input "
                "schema, whereas an AgentSkill (owned by `orchestration`) is the "
                "instruction bundle a harness discloses to a model."
            ),
            labels=("Tool", "Span", "TraceOutcome"),
            relationships=("CALLED", "PART_OF", "JUDGES"),
            call_noesis_tool=True,
            orchestrator_dispatch=False,
        ),
    ),
    tools_seed="knowledge_bases.tools.catalog:TOOLS",
    entity_types_seed="knowledge_bases.tools.ontology_types:ENTITY_TYPES",
    relationship_types_seed="knowledge_bases.tools.ontology_types:RELATIONSHIP_TYPES",
)
