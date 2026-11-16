"""Trace-driven self-improvement: signals -> LLM-drafted suggestions ->
reviewable ReviewFinding, never an automatic edit to any catalog."""
from datetime import datetime, timedelta, timezone

import pytest

from knowledge_bases.tools.models import (
    GeneratedToolDraft,
    RoutingGapSignal,
    SelfImprovementClassification,
    SelfImprovementDraft,
    SelfImprovementKind,
    ToolHealthSignal,
)
from knowledge_bases.tools.self_improvement import (
    classify_signal,
    collect_routing_gap_signals,
    collect_skill_health_signals,
    draft_suggestions,
    file_suggestions,
)

from knowledge_bases.tools.trace_store import TraceStore  # noqa: E402
from tests.trace_store_fakes import InMemoryTraceStore  # noqa: E402
from knowledge_bases.tools.models import (  # noqa: E402
    OutcomeSource,
    SpanEvent,
    SpanKind,
    SpanStatus,
    TraceOutcome,
)


def make_store() -> TraceStore:
    return InMemoryTraceStore()


# -- collect_skill_health_signals ---------------------------------------------

def test_collect_skill_health_signals_skips_healthy_skills_below_thresholds():
    store = make_store()
    now = datetime.now(timezone.utc)
    for i in range(5):
        store.insert_span(SpanEvent(
            span_id=f"span-{i}", trace_id=f"trace-{i}", name="lookup",
            kind=SpanKind.skill_call, knowledge_base="business_dictionary",
            skill="business_dictionary.lookup_business_term", started_at=now,
            status=SpanStatus.ok,
        ))

    signals = collect_skill_health_signals(store, since=now - timedelta(hours=1))

    assert signals == []


def test_collect_skill_health_signals_flags_high_error_rate():
    store = make_store()
    now = datetime.now(timezone.utc)
    for i in range(5):
        store.insert_span(SpanEvent(
            span_id=f"span-{i}", trace_id=f"trace-{i}", name="lookup",
            kind=SpanKind.skill_call, knowledge_base="business_dictionary",
            skill="business_dictionary.lookup_business_term", started_at=now,
            status=SpanStatus.error if i < 2 else SpanStatus.ok,
            error="KeyError: term" if i < 2 else None,
        ))

    signals = collect_skill_health_signals(store, since=now - timedelta(hours=1))

    assert len(signals) == 1
    assert signals[0].skill == "business_dictionary.lookup_business_term"
    assert signals[0].error_count == 2
    assert signals[0].sample_errors == ["KeyError: term"]


def test_collect_skill_health_signals_flags_user_corrections_even_with_no_errors():
    store = make_store()
    now = datetime.now(timezone.utc)
    for i in range(5):
        store.insert_span(SpanEvent(
            span_id=f"span-{i}", trace_id=f"trace-{i}", name="lookup",
            kind=SpanKind.skill_call, knowledge_base="epistemology",
            skill="epistemology.get_claim_provenance", started_at=now, status=SpanStatus.ok,
        ))
        store.insert_outcome(TraceOutcome(
            trace_id=f"trace-{i}",
            success=(i >= 2),  # 2 of 5 traces marked unsuccessful
            source=OutcomeSource.user_correction,
        ))

    signals = collect_skill_health_signals(store, since=now - timedelta(hours=1))

    assert len(signals) == 1
    assert signals[0].error_count == 0
    assert signals[0].user_correction_count == 2


def test_collect_skill_health_signals_ignores_skills_below_min_calls():
    store = make_store()
    now = datetime.now(timezone.utc)
    for i in range(2):
        store.insert_span(SpanEvent(
            span_id=f"span-{i}", trace_id=f"trace-{i}", name="lookup",
            kind=SpanKind.skill_call, knowledge_base="epistemology",
            skill="epistemology.get_claim_provenance", started_at=now,
            status=SpanStatus.error, error="boom",
        ))

    signals = collect_skill_health_signals(store, since=now - timedelta(hours=1), min_calls=5)

    assert signals == []


# -- collect_routing_gap_signals ----------------------------------------------

class FakeRoutingRepository:
    def __init__(self, rows):
        self.rows = rows

    def routing_gaps(self, min_traces=5):
        return self.rows


def test_collect_routing_gap_signals_filters_by_share_threshold():
    repository = FakeRoutingRepository([
        {
            "knowledge_base": "methodology", "low_confidence_count": 10,
            "clarification_needed_count": 6, "sample_questions": ["q1"],
        },
        {
            "knowledge_base": "project_management", "low_confidence_count": 1,
            "clarification_needed_count": 0, "sample_questions": ["q2"],
        },
    ])

    signals = collect_routing_gap_signals(repository, min_traces=5)

    assert len(signals) == 1
    assert signals[0].knowledge_base == "methodology"


# -- draft_suggestions ---------------------------------------------------------

class RecordingLLM:
    def __init__(self, draft: SelfImprovementDraft):
        self.draft = draft
        self.prompts = []

    def extract(self, prompt, model, system=None):
        self.prompts.append(prompt)
        return self.draft


class EmptyLLM:
    def extract(self, prompt, model, system=None):
        return SelfImprovementDraft()  # empty diagnosis, as NullLLM effectively produces


def test_draft_suggestions_produces_one_suggestion_per_signal():
    llm = RecordingLLM(SelfImprovementDraft(
        diagnosis="Missing case normalization.", suggested_change="Lowercase the term.",
        confidence=0.7,
    ))
    skill_signal = ToolHealthSignal(
        skill="business_dictionary.lookup_business_term", knowledge_base="business_dictionary",
        call_count=10, error_count=4, error_rate=0.4, user_correction_count=1,
    )
    routing_signal = RoutingGapSignal(
        knowledge_base="methodology", low_confidence_count=10, clarification_needed_count=6,
        sample_questions=["some question"],
    )

    analysis = draft_suggestions([skill_signal], [routing_signal], llm=llm)

    assert len(analysis.suggestions) == 2
    kinds = {s.kind for s in analysis.suggestions}
    assert kinds == {
        SelfImprovementKind.tool_wording, SelfImprovementKind.routing_capability_wording,
    }
    targets = {s.target for s in analysis.suggestions}
    assert targets == {"business_dictionary.lookup_business_term", "methodology"}
    assert len(llm.prompts) == 2


def test_draft_suggestions_drops_empty_diagnoses():
    skill_signal = ToolHealthSignal(
        skill="epistemology.get_claim_provenance", knowledge_base="epistemology",
        call_count=10, error_count=4, error_rate=0.4, user_correction_count=0,
    )

    analysis = draft_suggestions([skill_signal], [], llm=EmptyLLM())

    assert analysis.suggestions == []


# -- classify_signal / non-wording branches ------------------------------------

class SequencedLLM:
    """Returns one canned response per `extract()` call, in order -- for
    exercising the diagnose -> classify -> (maybe) generate call chain."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def extract(self, prompt, model, system=None):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_draft_suggestions_routes_a_narrow_gap_to_new_skill():
    llm = SequencedLLM([
        SelfImprovementDraft(diagnosis="No tool exists for this at all.", suggested_change=""),
        SelfImprovementClassification(kind=SelfImprovementKind.new_tool, confidence=0.8),
        GeneratedToolDraft(
            key="project_management.summarize_blocked_epics",
            name="Summarize Blocked Epics",
            description="Summarize blocked epics.",
            handler_function_name="summarize_blocked_epics_tool",
            handler_source="def summarize_blocked_epics_tool(arguments, graph):\n    return []\n",
            confidence=0.8,
        ),
    ])
    skill_signal = ToolHealthSignal(
        skill="project_management.list_project_blockers", knowledge_base="project_management",
        call_count=10, error_count=4, error_rate=0.4, user_correction_count=1,
    )

    analysis = draft_suggestions([skill_signal], [], llm=llm)

    assert len(analysis.suggestions) == 1
    suggestion = analysis.suggestions[0]
    assert suggestion.kind == SelfImprovementKind.new_tool
    assert suggestion.target == "project_management.summarize_blocked_epics"
    assert suggestion.generated_skill.handler_function_name == "summarize_blocked_epics_tool"


def test_draft_suggestions_routes_a_wide_gap_to_architecture():
    llm = SequencedLLM([
        SelfImprovementDraft(diagnosis="Requires a new knowledge base entirely.", suggested_change=""),
        SelfImprovementClassification(
            kind=SelfImprovementKind.architecture, diagnosis="Needs a new domain package.", confidence=0.9,
        ),
    ])
    skill_signal = ToolHealthSignal(
        skill="project_management.list_project_blockers", knowledge_base="project_management",
        call_count=10, error_count=4, error_rate=0.4, user_correction_count=1,
    )

    analysis = draft_suggestions([skill_signal], [], llm=llm)

    assert len(analysis.suggestions) == 1
    suggestion = analysis.suggestions[0]
    assert suggestion.kind == SelfImprovementKind.architecture
    assert suggestion.target == "project_management.list_project_blockers"
    assert suggestion.generated_skill is None


def test_draft_suggestions_falls_back_to_architecture_when_new_skill_draft_is_unusable():
    """A `new_tool` classification whose generation step never actually
    produced usable handler source must not be filed as `new_tool` --
    it would only ever fail `apply_self_improvement` later with nothing
    for a human to act on. Fall back to `architecture` instead."""
    llm = SequencedLLM([
        SelfImprovementDraft(diagnosis="No tool exists for this.", suggested_change=""),
        SelfImprovementClassification(kind=SelfImprovementKind.new_tool, confidence=0.6),
        GeneratedToolDraft(key="", handler_source="", confidence=0.2),  # empty draft
    ])
    skill_signal = ToolHealthSignal(
        skill="project_management.list_project_blockers", knowledge_base="project_management",
        call_count=10, error_count=4, error_rate=0.4, user_correction_count=1,
    )

    analysis = draft_suggestions([skill_signal], [], llm=llm)

    assert len(analysis.suggestions) == 1
    assert analysis.suggestions[0].kind == SelfImprovementKind.architecture
    assert analysis.suggestions[0].generated_skill is None


def test_classify_signal_defaults_to_architecture_on_unparseable_output():
    class BrokenLLM:
        def extract(self, prompt, model, system=None):
            return model()  # default instance, mirrors llm.extract()'s own fallback

    classification = classify_signal("s", "kb", "some diagnosis", BrokenLLM())

    assert classification.kind == SelfImprovementKind.architecture


# -- file_suggestions -----------------------------------------------------------

class FakeMethodologyRepository:
    def __init__(self, graph):
        self.graph = graph
        self.changesets = []

    def start_workflow(self, project_id, workflow, actor):
        from types import SimpleNamespace
        return SimpleNamespace(run_id="run-1")

    def submit_changeset(self, request):
        self.changesets.append(request)
        return {"graph_version": 1}


class FakeGraph:
    def __init__(self, project_version=5, existing_keys=None):
        self.project_version = project_version
        self.existing_keys = existing_keys or set()

    def run(self, query, **params):
        if "RETURN p.graph_version" in query:
            if self.project_version is None:
                return []
            return [{"v": self.project_version}]
        if "ReviewFinding" in query and "RETURN f.key" in query:
            return [{"key": key} for key in self.existing_keys]
        return []


def _suggestion(target="business_dictionary.lookup_business_term"):
    from knowledge_bases.tools.models import SelfImprovementSuggestion
    return SelfImprovementSuggestion(
        kind=SelfImprovementKind.tool_wording, target=target,
        diagnosis="A diagnosis.", suggested_change="A suggestion.", confidence=0.6,
    )


def test_file_suggestions_returns_empty_when_project_not_seeded():
    graph = FakeGraph(project_version=None)
    repo = FakeMethodologyRepository(graph)

    filed = file_suggestions([_suggestion()], repo)

    assert filed == []
    assert repo.changesets == []


def test_file_suggestions_submits_one_changeset_per_new_suggestion():
    graph = FakeGraph(project_version=5)
    repo = FakeMethodologyRepository(graph)

    filed = file_suggestions([_suggestion()], repo)

    assert len(filed) == 1
    assert len(repo.changesets) == 1
    operation = repo.changesets[0].operations[0]
    assert operation.entity_type == "ReviewFinding"
    assert operation.key == "self_improvement::tool_wording::business_dictionary.lookup_business_term"
    assert operation.properties["message"] == "A diagnosis."
    assert operation.properties["suggested_change"] == "A suggestion."


def test_file_suggestions_skips_already_flagged_targets():
    graph = FakeGraph(
        project_version=5,
        existing_keys={"self_improvement::tool_wording::business_dictionary.lookup_business_term"},
    )
    repo = FakeMethodologyRepository(graph)

    filed = file_suggestions([_suggestion()], repo)

    assert filed == []
    assert repo.changesets == []


def test_file_suggestions_returns_empty_for_no_suggestions_without_touching_graph():
    repo = FakeMethodologyRepository(FakeGraph())

    filed = file_suggestions([], repo)

    assert filed == []
    assert repo.changesets == []


def test_file_suggestions_files_wording_and_new_skill_kinds_as_advisory():
    from knowledge_bases.tools.models import GeneratedToolDraft, SelfImprovementSuggestion

    graph = FakeGraph(project_version=5)
    repo = FakeMethodologyRepository(graph)
    new_skill_suggestion = SelfImprovementSuggestion(
        kind=SelfImprovementKind.new_tool, target="project_management.summarize_blocked_epics",
        diagnosis="Missing capability.", confidence=0.7,
        generated_skill=GeneratedToolDraft(
            key="project_management.summarize_blocked_epics", handler_function_name="x",
            handler_source="def x(arguments, graph):\n    return []\n",
        ),
    )

    filed = file_suggestions([new_skill_suggestion], repo)

    assert len(filed) == 1
    assert repo.changesets[0].operations[0].properties["severity"] == "advisory"


def test_file_suggestions_files_architecture_kind_as_blocking():
    from knowledge_bases.tools.models import SelfImprovementSuggestion

    graph = FakeGraph(project_version=5)
    repo = FakeMethodologyRepository(graph)
    architecture_suggestion = SelfImprovementSuggestion(
        kind=SelfImprovementKind.architecture, target="project_management.list_project_blockers",
        diagnosis="Needs a new domain package.", confidence=0.9,
    )

    filed = file_suggestions([architecture_suggestion], repo)

    assert len(filed) == 1
    assert repo.changesets[0].operations[0].properties["severity"] == "blocking"


def test_file_suggestions_persists_the_generated_skill_payload():
    from knowledge_bases.tools.models import GeneratedToolDraft, SelfImprovementSuggestion

    graph = FakeGraph(project_version=5)
    repo = FakeMethodologyRepository(graph)
    suggestion = SelfImprovementSuggestion(
        kind=SelfImprovementKind.new_tool, target="project_management.summarize_blocked_epics",
        diagnosis="Missing capability.", confidence=0.7,
        generated_skill=GeneratedToolDraft(
            key="project_management.summarize_blocked_epics", handler_function_name="summarize_tool",
            handler_source="def summarize_tool(arguments, graph):\n    return []\n",
        ),
    )

    file_suggestions([suggestion], repo)

    import json
    stored = json.loads(repo.changesets[0].operations[0].properties["generated_skill_json"])
    assert stored["handler_function_name"] == "summarize_tool"
    assert stored["key"] == "project_management.summarize_blocked_epics"


# -- search retry loops -------------------------------------------------------

def test_a_retry_loop_is_a_signal_even_though_nothing_errored():
    """The third kind of failure: the call succeeded, nobody complained, and
    the caller still went round again. Invisible to error rates and user
    corrections, which is why it needed its own signal."""
    from knowledge_bases.tools.models import SearchRetrySignal

    signal = SearchRetrySignal(
        skill="orchestration.hybrid_search", knowledge_base="orchestration",
        retry_traces=3, total_calls=11, empty_calls=0,
    )

    assert signal.failure_level == "wrong_results"


def test_mostly_empty_results_point_at_a_different_fix():
    """Level matters: nothing found means the description misdescribes what the
    tool covers; wrong things found means its scope wording is off."""
    from knowledge_bases.tools.models import SearchRetrySignal

    signal = SearchRetrySignal(
        skill="orchestration.hybrid_search", knowledge_base="orchestration",
        retry_traces=3, total_calls=10, empty_calls=8,
    )

    assert signal.empty_call_share == 0.8
    assert signal.failure_level == "no_results"


def test_the_two_levels_get_different_prompts():
    """Otherwise the LLM is asked to 'improve this description' with no idea
    which way it was wrong."""
    from knowledge_bases.tools.self_improvement import _SEARCH_RETRY_PROMPTS

    assert "returned NOTHING" in _SEARCH_RETRY_PROMPTS["no_results"]
    assert "wrong KIND of answer" in _SEARCH_RETRY_PROMPTS["wrong_results"]


def test_a_retry_signal_drafts_a_tool_wording_suggestion():
    from knowledge_bases.tools.models import SearchRetrySignal, SelfImprovementKind
    from knowledge_bases.tools.self_improvement import _draft_search_retry_suggestion

    class FakeLLM:
        def complete(self, prompt):
            return "Searches only semantic-layer tables and columns."

    suggestion = _draft_search_retry_suggestion(
        SearchRetrySignal(
            skill="semantic_layer.search", knowledge_base="semantic_layer",
            retry_traces=3, total_calls=9, empty_calls=7,
        ),
        FakeLLM(),
    )

    assert suggestion.kind == SelfImprovementKind.tool_wording
    assert suggestion.target == "semantic_layer.search"
    assert "7 returned nothing" in suggestion.diagnosis


def test_a_failed_draft_is_dropped_rather_than_filed_empty():
    from knowledge_bases.tools.models import SearchRetrySignal
    from knowledge_bases.tools.self_improvement import _draft_search_retry_suggestion

    class BrokenLLM:
        def complete(self, prompt):
            raise RuntimeError("llm down")

    assert _draft_search_retry_suggestion(
        SearchRetrySignal(
            skill="a.b", knowledge_base="a",
            retry_traces=2, total_calls=6, empty_calls=0,
        ),
        BrokenLLM(),
    ) is None


# -- LLM gateway resilience ---------------------------------------------------

def test_urls_and_internal_hostnames_are_stripped_from_prompts():
    """Not cosmetic: the corporate gateway rejects prompts containing URLs or
    internal hostnames outright (401/500) before the model sees them. A
    Snowflake privatelink host in a sample error made that signal
    permanently undiagnosable."""
    from knowledge_bases.tools.self_improvement import redact_for_prompt

    redacted = redact_for_prompt(
        "250003: HTTPSConnectionPool("
        "host='attbusinessdatacoe.east-us-2.privatelink.snowflakecomputing.com', "
        "port=443) see https://errors.pydantic.dev/2.11/v/missing"
    )

    assert "snowflakecomputing.com" not in redacted
    assert "https://" not in redacted
    assert "250003" in redacted, "the diagnostic content must survive"


def test_redaction_leaves_ordinary_errors_alone():
    from knowledge_bases.tools.self_improvement import redact_for_prompt

    assert redact_for_prompt("'trace_id'") == "'trace_id'"


def test_one_undraftable_signal_does_not_discard_the_others():
    """A pass analyzing six signals died on its second and filed nothing,
    losing the four that were fine."""
    from knowledge_bases.tools.models import (
        SelfImprovementKind,
        SelfImprovementSuggestion,
        ToolHealthSignal,
    )
    from knowledge_bases.tools.self_improvement import draft_suggestions

    def flaky(signal, llm):
        if signal.skill == "b.bad":
            raise RuntimeError("500 from gateway")
        return SelfImprovementSuggestion(
            kind=SelfImprovementKind.tool_wording,
            target=signal.skill, diagnosis="d", suggested_change="c",
        )

    signals = [
        ToolHealthSignal(
            skill=k, knowledge_base="x", call_count=10, error_count=5,
            error_rate=0.5, user_correction_count=0,
        )
        for k in ("a.good", "b.bad", "c.good")
    ]

    import knowledge_bases.tools.self_improvement as mod
    original = mod._draft_skill_wording_suggestion
    mod._draft_skill_wording_suggestion = flaky
    try:
        analysis = draft_suggestions(signals, [], llm=object())
    finally:
        mod._draft_skill_wording_suggestion = original

    assert [s.target for s in analysis.suggestions] == ["a.good", "c.good"]
