"""Skills/Tools/Traces model ownership."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Tool(BaseModel):
    """A versioned, graph-registered, MCP-invokable capability.

    `description` and `input_schema` let a harness build a real MCP `Tool`
    definition straight from this node, with no local hardcoding. `handler`
    is a dotted `"module:function"` path to a Python callable with signature
    `(arguments: dict, graph: Graph) -> Any` that performs the work; only the
    *registration* lives in the graph, not the implementation.
    """

    key: str
    name: str
    version: str
    knowledge_base: str
    description: str = ""
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    handler: str = ""
    active: bool = True


class SpanKind(str, Enum):
    skill_call = "skill_call"
    tool_call = "tool_call"
    llm_call = "llm_call"
    retrieval = "retrieval"
    internal = "internal"


class SpanStatus(str, Enum):
    ok = "ok"
    error = "error"


class OutcomeSource(str, Enum):
    self_assessed = "self_assessed"
    user_correction = "user_correction"
    harness_inferred = "harness_inferred"


class SpanEvent(BaseModel):
    """One `Span` node. `span_id` is the node's unique key, so writing the
    same event twice is a safe no-op (`MERGE`).

    `user_id`/`client` identify *who* (which user, through which harness)
    made the call, and `expert_key` which graph-defined expert was acting,
    so telemetry from many users sharing one graph stays attributable."""

    span_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_span_id: Optional[str] = None
    name: str
    kind: SpanKind
    knowledge_base: str
    skill: Optional[str] = None
    skill_version: Optional[str] = None
    tool: Optional[str] = None
    tool_version: Optional[str] = None
    model: Optional[str] = None
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    status: SpanStatus = SpanStatus.ok
    error: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    environment: str = "dev"
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    client: Optional[str] = None
    expert_key: Optional[str] = None
    raw_input: Optional[Any] = None
    raw_output: Optional[Any] = None
    # How many results a retrieval call returned. `None` for tools that do not
    # retrieve. Recorded as a first-class field rather than dug out of
    # `raw_output` because a search returning zero rows is a *failure of
    # expression* -- status is `ok`, nothing errored, and the caller still got
    # nothing -- which is invisible to any signal that only counts errors.
    result_count: Optional[int] = None


class TraceOutcome(BaseModel):
    """One node of the append-only `TraceOutcome` log. `outcome_id` is the
    node's unique key (dedup on retry); the *current* verdict for a trace
    is whichever node has the latest `recorded_at`, so a correction is a
    new node, never an edit of the original."""

    outcome_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str
    success: bool
    reason: Optional[str] = None
    source: OutcomeSource
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolHealthSignal(BaseModel):
    """Aggregated `Span`/`TraceOutcome` stats for one skill over a lookback
    window -- the input to self-improvement, not a stored record itself."""

    skill: str
    knowledge_base: str
    call_count: int
    error_count: int
    error_rate: float
    user_correction_count: int
    avg_duration_ms: Optional[float] = None
    sample_errors: list[str] = Field(default_factory=list)


class SearchRetrySignal(BaseModel):
    """A retrieval tool that callers had to invoke repeatedly in one turn.

    Distinct from `ToolHealthSignal`: nothing errored and the user may never
    have complained, yet the caller could not express what they wanted. The
    `empty_call_share` is what separates the two fixes -- searches returning
    *nothing* mean the tool's description misdescribes what it covers, while
    searches returning the *wrong* rows mean its scope wording is off.
    """

    skill: str
    knowledge_base: str
    retry_traces: int
    total_calls: int
    empty_calls: int
    sample_inputs: list[Any] = Field(default_factory=list)

    @property
    def empty_call_share(self) -> float:
        return self.empty_calls / self.total_calls if self.total_calls else 0.0

    @property
    def failure_level(self) -> str:
        """Which layer to fix, in the language a suggestion is drafted against."""
        if self.empty_call_share >= 0.5:
            return "no_results"
        return "wrong_results"


class RoutingGapSignal(BaseModel):
    """Aggregated `RoutingTrace` stats suggesting a knowledge base's
    `ExpertCapability` wording doesn't match how questions are actually
    phrased -- ambiguous/low-confidence routing, not a skill failure."""

    knowledge_base: str
    low_confidence_count: int
    clarification_needed_count: int
    sample_questions: list[str] = Field(default_factory=list)


class ExpertCorrectionSignal(BaseModel):
    """Traces a graph-defined expert acted in that the user later corrected
    -- a signal that the expert's *prompt* guidance, not one skill's
    wording, is what needs improving."""

    expert_key: str
    trace_count: int
    corrected_trace_count: int
    correction_share: float
    sample_reasons: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class SelfImprovementKind(str, Enum):
    tool_wording = "tool_wording"
    routing_capability_wording = "routing_capability_wording"
    # Appends a guidance paragraph to a graph-defined expert's prompt
    # (`KnowledgeBaseExpert.prompt_overlays`), versioned as an ExpertRevision.
    expert_prompt_wording = "expert_prompt_wording"
    # Prepends one bounded supplemental-context entry *ahead* of the expert's
    # base prompt (`KnowledgeBaseExpert.prompt_prefix_overlays`), for framing
    # that must be read before the domain instructions rather than after.
    # Budget-checked at write time; see `orchestration.models`.
    expert_prompt_prefix = "expert_prompt_prefix"
    new_tool = "new_tool"
    # Covers both a brand-new knowledge base ("paradigm": a new
    # `knowledge_bases/<name>` package, new Neo4j labels/relationships, a
    # new `KnowledgeBase` registry entry) and any change to existing
    # structure spanning multiple files/packages. Collapsed into one kind
    # rather than adding a separate "paradigm" value: both require the same
    # human-review treatment (see `knowledge_bases.tools.self_improvement_apply`),
    # so there is no behavioral difference between them today.
    architecture = "architecture"


class SelfImprovementDraft(BaseModel):
    """The shape an LLM fills in via `BaseLLM.extract()` for a wording-kind
    signal. All fields have safe defaults so a provider that returns
    unparseable/empty output (e.g. `NullLLM`, or a malformed completion)
    degrades to an empty diagnosis rather than raising -- `llm.extract()`'s
    own fallback path constructs this type with no arguments on validation
    failure."""

    diagnosis: str = ""
    suggested_change: str = ""
    confidence: float = Field(default=0.5, ge=0, le=1)


class SelfImprovementClassification(BaseModel):
    """The shape an LLM fills in to classify a health/gap signal into one
    of the three real code-change shapes this loop knows how to handle --
    see `knowledge_bases.tools.self_improvement.classify_signal` and
    `SelfImprovementKind`. Defaults to the safest/most conservative
    classification (`architecture`, i.e. "route to human review") on any
    unparseable/empty LLM output, never to an auto-appliable kind by
    default."""

    kind: SelfImprovementKind = SelfImprovementKind.architecture
    diagnosis: str = ""
    confidence: float = Field(default=0.5, ge=0, le=1)


class GeneratedToolDraft(BaseModel):
    """The shape an LLM fills in via `BaseLLM.extract()` to propose a brand
    new bounded skill/handler -- see
    `knowledge_bases.tools.source_editor.generate_new_skill`. Scoped to
    exactly one new top-level function appended to one existing
    `knowledge_bases/<knowledge_base>/mcp_tools.py`; anything wider
    (a new file/package, new Neo4j labels) is an `architecture`-kind
    suggestion instead, never this."""

    key: str = ""  # e.g. "execution.summarize_blocked_epics"
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    handler_function_name: str = ""
    handler_source: str = ""  # the full `def ...` source of the new handler
    confidence: float = Field(default=0.5, ge=0, le=1)


class SelfImprovementSuggestion(BaseModel):
    """One LLM-drafted suggestion, filed as a `ReviewFinding`. Filing
    itself is read-only review data, not a code change -- but see
    `knowledge_bases.tools.self_improvement_apply`, which (by explicit
    design choice, unlike every other Noesis write) *does* apply a
    `tool_wording`/`routing_capability_wording`/`new_tool` suggestion as a
    real, git-committed source edit automatically -- test-gated for
    `new_tool` (see `docs/architecture/self_improvement.md`) -- then
    auto-reverts it if a post-apply metric snapshot shows the change made
    things worse. `architecture`-kind suggestions are the one exception:
    never auto-applied, always routed to human review."""

    kind: SelfImprovementKind
    target: str  # e.g. skill key or knowledge_base key the finding is about
    diagnosis: str
    suggested_change: str = ""
    confidence: float = Field(default=0.5, ge=0, le=1)
    generated_skill: Optional[GeneratedToolDraft] = None


class SelfImprovementAnalysis(BaseModel):
    suggestions: list[SelfImprovementSuggestion] = Field(default_factory=list)
