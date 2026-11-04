"""What Tools & Telemetry's labels and relationships mean.

Two things live here that are easy to conflate: the *definition* of a callable
operation, and the *record* of one execution of it. A Tool is the definition -- one
handler plus an input schema, low-cardinality and versioned. A Span is one
dispatch that happened, high-cardinality and immutable. Counting Tools tells you
what Noesis can do; counting Spans tells you what it did.

A second confusion is worth naming explicitly because the repo does it in prose
too: a Tool here is NOT an AgentSkill. An AgentSkill (owned by `orchestration`) is
an instruction bundle a harness discloses to a model; a Tool is one callable
operation with a handler. The `DOCUMENTS` edge between them is orchestration's.

Grounded in `trace_store.py`, which writes every edge below:
`(:Span)-[:CALLED]->(:Tool)` when the dispatched skill is registered,
`(:Span)-[:PART_OF]->(:RoutingTrace)`, and `(:TraceOutcome)-[:JUDGES]->(:RoutingTrace)`.
Spans are MERGEd on `span_id`, so a retried write is a no-op rather than a duplicate.
"""
from knowledge_bases.base import EntityType, PropertyType, RelationshipType

ENTITY_TYPES = (
    EntityType(
        label="Tool",
        extractable=False,
        description=(
            "One callable bounded operation: a handler plus the input schema it "
            "accepts, identified by a stable key. It is the definition of a "
            "capability, not a single use of it (Span) and not the instruction "
            "bundle a harness shows a model (AgentSkill)."
        ),
        identity="key",
        properties=("key", "knowledge_base", "operation", "description",
                    "input_schema", "handler", "active", "version"),
        not_to_be_confused_with=("AgentSkill", "Span", "Interface",
                                 "ExpertCapability"),
    ),
    EntityType(
        label="Span",
        extractable=False,
        description=(
            "One recorded dispatch of one tool: its arguments, duration, outcome "
            "and error if it failed, merged on `span_id` so retries never "
            "duplicate it. A Span is history and is never revised -- it is an "
            "event, not the Tool it called and not a judgement of whether the "
            "answer was right (TraceOutcome)."
        ),
        identity="span_id",
        properties=("span_id", "trace_id", "skill", "knowledge_base", "operation",
                    PropertyType(
                "status",
                "Whether the tool call itself succeeded. `error` means the handler raised, not that the answer was wrong -- a correct refusal is `ok`. Judgement about the answer lives on TraceOutcome.",
                enum=("ok", "error",),
            ), "error", "duration_ms", "started_at", "finished_at",
                    "expert_key"),
        not_to_be_confused_with=("Tool", "TraceOutcome", "RoutingTrace"),
    ),
    EntityType(
        label="TraceOutcome",
        extractable=False,
        description=(
            "The verdict on whether one routing trace's answer actually held up, "
            "recorded after the fact from user confirmation or correction. It "
            "judges a whole trace, not one call: a Span says an operation ran, "
            "while a TraceOutcome says the resulting answer was right or wrong."
        ),
        properties=("id", "trace_id", "success", "detail", "recorded_at"),
        not_to_be_confused_with=("Span", "RoutingTrace", "ReviewFinding"),
    ),
)

RELATIONSHIP_TYPES = (
    RelationshipType(
        relationship="CALLED",
        extractable=False,
        description=(
            "Execution link from an event to the definition it invoked: this Span "
            "was a dispatch of that Tool. Written only when the dispatched skill "
            "is a registered Tool, so an unregistered call leaves a Span with no "
            "CALLED edge rather than inventing a Tool node."
        ),
        subject=("Span",),
        object=("Tool",),
        not_to_be_confused_with=("CAN_USE", "PART_OF", "USES"),
    ),
    RelationshipType(
        relationship="PART_OF",
        extractable=False,
        description=(
            "Groups one execution into the routing decision that caused it: this "
            "Span belongs to that RoutingTrace. It is how a trace's whole call "
            "tree is reassembled, and is containment of an event in a trace, not "
            "structural containment of code (CONTAINS) or delivery hierarchy."
        ),
        subject=("Span",),
        object=("RoutingTrace",),
        not_to_be_confused_with=("CONTAINS", "HAS_TRACE", "BELONGS_TO", "MEMBER_OF"),
    ),
    RelationshipType(
        relationship="JUDGES",
        extractable=False,
        description=(
            "Evaluation link: this TraceOutcome is the verdict on that "
            "RoutingTrace. It always points from the judgement to the thing "
            "judged, and asserts correctness of an answer rather than membership "
            "of a call in a trace (PART_OF)."
        ),
        subject=("TraceOutcome",),
        object=("RoutingTrace",),
        not_to_be_confused_with=("PART_OF", "VERIFIES", "MEASURES"),
    ),
)
