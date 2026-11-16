"""Kafka worker: subscribe to what the installed packages publish, dispatch to their parsers.

The topic list used to be a constant in the engine
(`['raw-emails', 'teams-messages', 'evidence-documents']`), so a product with no
`claims` package subscribed to three streams it had no parser for, and a product
that added a source had to edit engine code to hear it. Now the topics come from
`registry.extraction_sources()`: the package that owns the vocabulary a stream
produces declares the topic and the parser, and the worker is a dispatcher.

Two dispatch shapes, and the difference is deliberate:
- `parser` sources normalize a raw payload into a `NormalizedDocument` and go
  through the LLM path (`extract.py`), because the payload is prose.
- `triplets` sources emit typed triplets straight into
  `ingestion_service.ingest_triplets`, because their format is deterministic and
  a model has nothing to infer. Both end at the same ontology check.

A single bad message must not take the loop down: log, commit past it, keep
consuming. A source that raises is the source's problem, not the loop's.
"""
import json
import logging
from confluent_kafka import Consumer, KafkaError
from config import settings
from extract import extract_claims
from graph import Graph
from knowledge_bases.registry import extraction_sources, resolve_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _mark_processed(graph: Graph, event_id: str, source_type: str, url: str) -> None:
    graph.run(
        "MERGE (e:IngestionEvent {id: $id}) "
        "SET e.status = 'completed', e.source_type = $source_type, "
        "e.url = $url, e.processed_at = datetime()",
        id=event_id,
        source_type=source_type,
        url=url,
    )


def _already_processed(graph: Graph, event_id: str) -> bool:
    return bool(graph.run(
        "MATCH (e:IngestionEvent {id: $id, status: 'completed'}) RETURN e.id AS id",
        id=event_id,
    ))


def _source_id(raw: dict, source) -> str:
    """The `SourceNode` a payload belongs to, keyed on what the payload names.

    A pull source names the file it read (`source_path`); a stream payload names
    its own `source_name`. Neither is the engine's to decide.
    """
    from extract import _id_from_text
    key = raw.get("source_path") or raw.get("source_name") or source.key
    return _id_from_text("source", key)


class KnowledgeWorker:
    def __init__(self, graph: Graph | None = None):
        conf = {
            'bootstrap.servers': settings.kafka_brokers,
            'group.id': 'knowledge-extraction-group',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False
        }
        self.consumer = Consumer(conf)
        self.graph = graph or Graph()

        # Topics come from the installed profile. A source with a `topic` is a
        # stream; a source with a `collect` is a pull source, run separately.
        self.sources = extraction_sources()
        self.topics = sorted({s.topic for s in self.sources if s.topic})
        self.pull_sources = [s for s in self.sources if s.collect]

        if not self.sources:
            raise RuntimeError(
                "No installed package declares an ingestion source. The worker "
                "has nothing to run: install a package that owns a source (for "
                "example `claims`), or run the pull sources from a scheduler."
            )
        if self.topics:
            self.consumer.subscribe(self.topics)
            logger.info(f"Worker subscribed to topics: {self.topics}")
        else:
            # A product with pull sources and no streams is a supported shape --
            # a code-and-schema product, for example. The worker is constructible
            # so `run_pull_sources()` can be called; `run()` is the part that has
            # nothing to do.
            logger.warning(
                "No installed package declares an ingestion stream; this worker "
                "has %d pull source(s) and no topics to subscribe to",
                len(self.pull_sources),
            )

    def _dispatch(self, raw: dict, source) -> None:
        """One payload, one source, one shared write path."""
        event_id = raw.get("event_id")
        if event_id and _already_processed(self.graph, event_id):
            logger.info("Skipping previously processed event: %s", event_id)
            return

        if source.triplets:
            triplets = resolve_seed(source.triplets)(raw)
            if not triplets:
                logger.info("Source %s produced no triplets; nothing written", source.key)
                return
            from knowledge_bases.ingestion_service import ingest_triplets
            result = ingest_triplets(
                self.graph,
                triplets,
                source_id=_source_id(raw, source),
                source_props={"name": raw.get("source_name", source.source_type),
                              "type": source.source_type},
                actor=source.key,
                # A scan is repository code writing its own graph, so the
                # code-only labels it owns (`CodeRepository`, `CodeFile`) are
                # allowed. The ontology checks still run.
                extractable_only=not source.code_authored,
            )
            logger.info(
                "source %s: %d entities, %d relationships, %d rejected",
                source.key,
                len(result.get("entities", [])),
                len(result.get("relationships", [])),
                len(result.get("rejected", [])),
            )
            if result.get("rejected"):
                logger.warning("rejected: %s", result["rejected"])
        else:
            doc = resolve_seed(source.parser)(raw)
            if not doc.text:
                logger.info("Source %s produced no text; nothing to extract", source.key)
                return
            logger.info(f"Extracting claims from message: {doc.doc_title}")
            extract_claims(
                text=doc.text,
                doc_title=doc.doc_title,
                source_name=doc.source_name,
                url=doc.url,
                owners=doc.owners,
                source_type=doc.source_type,
                graph=self.graph,
            )

        if event_id:
            _mark_processed(self.graph, event_id, raw.get("source_type", source.source_type),
                            raw.get("url", ""))

    def _find_source(self, raw: dict, topic: str | None):
        """The source that owns this payload.

        A Kafka message names its topic, which is the primary key. A payload read
        from anywhere else names its own `source_type`, which the declared sources
        also carry -- so a payload pulled from a file, a queue, or a test fixture
        reaches the same parser it would have reached on the stream.
        """
        if topic:
            return next((s for s in self.sources if s.topic == topic), None)
        return next(
            (s for s in self.sources if s.source_type == raw.get("source_type")), None
        )

    def process_message(self, msg_value: str, topic: str | None = None) -> None:
        raw = json.loads(msg_value)
        source = self._find_source(raw, topic)
        if source is None:
            # A message on a topic, or of a source type, no installed package
            # declares. Commit past it: the product cannot write what it has no
            # vocabulary for.
            logger.warning(
                "No installed source for topic %r / source_type %r; message skipped",
                topic, raw.get("source_type"),
            )
            return
        self._dispatch(raw, source)

    def run_pull_sources(self) -> list[dict]:
        """Run every installed pull source. Not a Kafka path: a scheduler, a cron,
        or a script calls this, and each source reads its own configuration."""
        outcomes = []
        for source in self.pull_sources:
            raws = resolve_seed(source.collect)()
            for raw in raws:
                self._dispatch(raw, source)
            outcomes.append({"key": source.key, "collected": len(raws)})
        return outcomes

    def run(self):
        if not self.topics:
            raise RuntimeError(
                "Nothing to consume: no installed package declares an ingestion "
                "stream. Run the pull sources with `run_pull_sources()`."
            )
        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(msg.error())
                        break

                val = msg.value().decode('utf-8')
                try:
                    self.process_message(val, msg.topic())
                except Exception as e:
                    logger.exception("Error processing message: %s", e)
                self.consumer.commit(msg)
        except KeyboardInterrupt:
            logger.info("Worker stopped by user.")
        finally:
            self.consumer.close()
            self.graph.close()

if __name__ == "__main__":
    worker = KnowledgeWorker()
    worker.run()
