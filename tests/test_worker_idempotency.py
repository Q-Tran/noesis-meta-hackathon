"""The worker consumes the topics its installed packages declare, and is idempotent.

The topic list used to be a constant in `worker.py`, so this test asserted a
vocabulary the engine does not own. It now declares its own source: the engine's
worker is tested against a source the test supplies, which is exactly how a real
product supplies one -- a package declares `ExtractionSource`, the worker
subscribes to its topic and dispatches to its parser.
"""
import json

from knowledge_bases.base import ExtractionSource, NormalizedDocument
import worker


def parse(raw: dict) -> NormalizedDocument:
    """The test source's parser: one payload, one normalized document."""
    return NormalizedDocument(
        text=raw.get("text", ""),
        doc_title=raw.get("doc_title", "Message"),
        source_name=raw.get("source_name", "Test Stream"),
        url=raw.get("url", ""),
        owners={
            "email": raw.get("owner_email", ""),
            "teams_id": raw.get("owner_teams_id", ""),
        },
        source_type=raw.get("source_type", "Test Event"),
        event_id=raw.get("event_id"),
    )


SOURCE = ExtractionSource(
    key="test.stream",
    topic="test-events",
    source_type="Test Event",
    parser="test_worker_idempotency:parse",
)


class FakeConsumer:
    def subscribe(self, topics): self.topics = topics


class FakeGraph:
    def __init__(self, completed=False): self.completed, self.writes = completed, []
    def run(self, query, **params):
        if "RETURN e.id AS id" in query:
            return [{"id": params["id"]}] if self.completed else []
        self.writes.append((query, params)); return []


def build_worker(monkeypatch, graph):
    monkeypatch.setattr(worker, "Consumer", lambda _conf: FakeConsumer())
    monkeypatch.setattr(worker, "extraction_sources", lambda: (SOURCE,))
    return worker.KnowledgeWorker(graph=graph)



def test_worker_subscribes_to_the_topics_its_packages_declare(monkeypatch):
    graph = FakeGraph()
    knowledge_worker = build_worker(monkeypatch, graph)
    assert knowledge_worker.topics == ["test-events"]


def test_worker_skips_completed_event(monkeypatch):
    graph = FakeGraph(completed=True)
    knowledge_worker = build_worker(monkeypatch, graph)
    monkeypatch.setattr(worker, "extract_claims", lambda **kwargs: (_ for _ in ()).throw(AssertionError()))
    knowledge_worker.process_message(json.dumps({"event_id": "event-1", "text": "content", "source_type": "Test Event"}))
    assert graph.writes == []


def test_worker_records_event_after_extraction(monkeypatch):
    graph = FakeGraph()
    knowledge_worker = build_worker(monkeypatch, graph)
    extracted = []
    monkeypatch.setattr(worker, "extract_claims", lambda **kwargs: extracted.append(kwargs))
    knowledge_worker.process_message(json.dumps({
        "event_id": "event-1", "text": "content", "source_type": "Test Event"
    }))
    assert extracted
    assert graph.writes[0][1]["id"] == "event-1"
    assert "test-events" in knowledge_worker.topics


def test_a_message_from_an_uninstalled_source_is_skipped(monkeypatch):
    """A payload naming a source type no installed package declares cannot be
    written: the product has no vocabulary for it. Commit past it, write nothing."""
    graph = FakeGraph()
    knowledge_worker = build_worker(monkeypatch, graph)
    monkeypatch.setattr(worker, "extract_claims", lambda **kwargs: (_ for _ in ()).throw(AssertionError()))
    knowledge_worker.process_message(json.dumps({"text": "content", "source_type": "Teams Chat"}), "teams-messages")
    assert graph.writes == []


def test_a_worker_with_no_sources_refuses_to_start(monkeypatch):
    """Running a consumer that subscribes to nothing is a silent no-op; the
    misconfiguration should say so at startup."""
    graph = FakeGraph()
    monkeypatch.setattr(worker, "Consumer", lambda _conf: FakeConsumer())
    monkeypatch.setattr(worker, "extraction_sources", lambda: ())
    import pytest
    with pytest.raises(RuntimeError, match="No installed package declares an ingestion source"):
        worker.KnowledgeWorker(graph=graph)


def test_a_deterministic_source_writes_triplets_not_text(monkeypatch):
    """The second dispatch shape: a source whose format is exact emits typed
    triplets straight into `ingestion_service`, skipping the model. A Power BI
    table states its columns; asking a model to guess a schema it is handed is a
    way to invent ontology."""
    from knowledge_bases.base import ExtractionSource
    import knowledge_bases.ingestion_service as ingestion_service

    source = ExtractionSource(
        key="test.table",
        topic="test-tables",
        source_type="Test Table",
        triplets="test_worker_idempotency:emit_triplets",
    )
    graph = FakeGraph()
    monkeypatch.setattr(worker, "Consumer", lambda _conf: FakeConsumer())
    monkeypatch.setattr(worker, "extraction_sources", lambda: (source,))
    monkeypatch.setattr(worker, "extract_claims",
                        lambda **kwargs: (_ for _ in ()).throw(AssertionError()))
    written = []
    monkeypatch.setattr(ingestion_service, "ingest_triplets",
                        lambda g, t, **kw: written.append((g, t, kw)) or {"entities": [], "relationships": [], "rejected": []})

    knowledge_worker = worker.KnowledgeWorker(graph=graph)
    knowledge_worker.process_message(json.dumps({"table_name": "Orders", "source_type": "Test Table"}), "test-tables")

    assert written, "the triplet source never reached ingest_triplets"
    triplets = written[0][1]
    assert triplets[0].subject_label == "Column"
    # The ontology check is the point, not the model: the source's caller passes
    # through `partition`, and the extractable filter is lifted only for a source
    # that IS repository code.
    assert written[0][2]["extractable_only"] is True


def emit_triplets(raw: dict) -> list:
    from knowledge_bases.triplets import Triplet
    return [Triplet(
        subject_label="Column",
        subject_identity=f"column::{raw['table_name']}.amount",
        relationship="COLUMN_OF",
        object_label="Table",
        object_identity=f"table::{raw['table_name']}",
    )]


def test_a_pull_source_is_collected_by_the_package_that_declares_it(monkeypatch):
    """A pull source takes no arguments: the worker asks the package for its
    payloads, and the package reads its own configuration."""
    from knowledge_bases.base import ExtractionSource
    import knowledge_bases.ingestion_service as ingestion_service

    source = ExtractionSource(
        key="test.pull",
        topic=None,
        source_type="Test Pull",
        triplets="test_worker_idempotency:emit_triplets",
        collect="test_worker_idempotency:collect_two",
    )
    graph = FakeGraph()
    monkeypatch.setattr(worker, "Consumer", lambda _conf: FakeConsumer())
    monkeypatch.setattr(worker, "extraction_sources", lambda: (source,))
    collected = []
    monkeypatch.setattr(ingestion_service, "ingest_triplets",
                        lambda g, t, **kw: collected.append(t) or {"entities": [], "relationships": [], "rejected": []})

    knowledge_worker = worker.KnowledgeWorker(graph=graph)
    outcomes = knowledge_worker.run_pull_sources()

    assert outcomes == [{"key": "test.pull", "collected": 2}]
    assert len(collected) == 2


def collect_two() -> list:
    return [{"table_name": "Orders"}, {"table_name": "Items"}]


def test_a_pull_only_worker_is_constructible_and_says_run_has_nothing_to_do(monkeypatch):
    """A product with pull sources and no streams -- a code-and-schema product --
    is a supported shape. The worker can be built so `run_pull_sources()` is
    callable; `run()` is the part with nothing to consume, and it says so rather
    than looping on an empty subscription."""
    from knowledge_bases.base import ExtractionSource

    source = ExtractionSource(
        key="test.pull",
        topic=None,
        source_type="Test Pull",
        triplets="test_worker_idempotency:emit_triplets",
        collect="test_worker_idempotency:collect_two",
    )
    monkeypatch.setattr(worker, "Consumer", lambda _conf: FakeConsumer())
    monkeypatch.setattr(worker, "extraction_sources", lambda: (source,))

    knowledge_worker = worker.KnowledgeWorker(graph=FakeGraph())
    assert knowledge_worker.topics == []
    assert knowledge_worker.pull_sources == [source]

    import pytest
    with pytest.raises(RuntimeError, match="Nothing to consume"):
        knowledge_worker.run()
