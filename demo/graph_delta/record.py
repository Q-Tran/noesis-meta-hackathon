"""Record one knowledge-graph session as a fine-grained graph delta.

Model being demonstrated: **every knowledge entity exists because an argument
says so**. The argument is a real `Argument` node, its premises are `Claim`
nodes (`USES_PREMISE`), and it concludes the entity (`CONCLUDES`). Each claim is
grounded by exactly one `FOUND_IN` edge to either a `Chunk` (document evidence)
or a `RoutingTrace` (the verbatim conversation turn). Evidence and telemetry --
`SourceNode`, `Chunk`, `RoutingTrace`, `Session`, `Span`, `Tool` -- are the
ground, not the grounded, so they carry no argument.

The demo is not a mock: every step runs through the shipped write paths
(`TraceStore.record_turn_start`/`insert_span`, `graph.upsert_node`,
`graph.upsert_rel`, `ground_claim_in_trace`). Between steps the recorder
snapshots the nodes it tracks, their properties and the edges between them, and
diffs. Creations take their timestamp from the graph (`created_at`, stamped at
the write boundary); mutations, which Neo4j Community cannot date, are stamped
when observed.

Output: `events.json` -- node/edge records plus an ordered event list. The
player (`render.py` -> `index.html`) only replays that file.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from graph import Graph
from knowledge_bases import registry
from knowledge_bases.provenance.conversation import ground_claim_in_trace
from knowledge_bases.tools.models import SpanEvent, SpanKind
from knowledge_bases.tools.trace_store import TraceStore

HERE = Path(__file__).resolve().parent
OUT = HERE / "events.json"

DEMO_PREFIX = "demo-"
SESSION_ID = f"{DEMO_PREFIX}session-q3-review"
TRACE_ID = f"{DEMO_PREFIX}trace-q3-review"
SPAN_ID = f"{DEMO_PREFIX}span-provenance"
TOOL_KEY = "epistemology.get_claim_provenance"

QUESTION = (
    "Which Muse model should I use to build a coding agent, and how do I call "
    "it from OpenCode?"
)
STEP_PAUSE = 0.26

# Evidence and telemetry are the ground of the graph: they are what justifies
# everything else, so they are exempt from the "every entity has an argument"
# rule. Claims and Arguments are the justification itself.
GROUND_LABELS = {"SourceNode", "Chunk", "RoutingTrace", "Session", "Span", "Tool",
                 "TraceOutcome"}
JUSTIFICATION_LABELS = {"Claim", "Argument"}


# --------------------------------------------------------------------------
# ontology ownership: label -> knowledge base, relationship -> knowledge base
# --------------------------------------------------------------------------
def _ownership() -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """Who owns each label and each relationship type.

    Packages declare `edge_contracts` at package level; those declarations are
    authoritative for cross-cutting edges. `FOUND_IN` is declared by
    `provenance`, but `semantic_layer` also lists it among its relationships,
    so the knowledge-base lists alone would colour the provenance spine as a
    semantic-layer edge -- exactly the distinction this demo exists to show.
    """
    labels: dict[str, str] = {}
    edges: dict[str, str] = {}
    owners: dict[str, set[str]] = {}
    for package in registry.installed_packages():
        for contract in getattr(package, "edge_contracts", ()) or ():
            edges.setdefault(contract.relationship, package.key)
            owners.setdefault(contract.relationship, set()).add(package.key)
    for package in registry.installed_packages():
        for kb in package.knowledge_bases:
            for label in kb.labels:
                labels.setdefault(label, kb.key)
            for rel in kb.relationships:
                edges.setdefault(rel, kb.key)
                owners.setdefault(rel, set()).add(kb.key)
    return labels, edges, owners


LABEL_KB, EDGE_KB, EDGE_OWNERS = _ownership()

# The two graphs the demo has to keep distinct. Canonical: what queries are
# allowed to assume. Truth: the evidence and resolution machinery that may or
# may not imply the canonical graph; the mention web lives here.
TRUTH_LABELS = {"SourceNode", "Chunk", "RoutingTrace", "Argument", "Claim",
                "DocumentNode", "IngestionEvent"}
TELEMETRY_LABELS = {"Session", "Span", "Tool", "TraceOutcome"}


def layer_of(label: str) -> str:
    if label in TELEMETRY_LABELS:
        return "telemetry"
    if label in TRUTH_LABELS:
        return "truth"
    return "canonical"


def edge_ontology(rel: str, from_ont: str, to_ont: str) -> str:
    owner = EDGE_KB.get(rel, "development")
    if (len(EDGE_OWNERS.get(rel, ())) > 1 and from_ont == to_ont
            and from_ont in EDGE_OWNERS.get(rel, ())):
        return from_ont
    return owner


def jsonable(value):  # Neo4j temporal / spatial types -> JSON
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:  # neo4j Duration
            return str(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


def epoch_ms(value) -> int | None:
    if value is None:
        return None
    if hasattr(value, "to_native"):
        return int(value.to_native().timestamp() * 1000)
    if isinstance(value, str):
        try:
            return int(datetime.fromisoformat(value).timestamp() * 1000)
        except ValueError:
            return None
    return None


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def name_of(label: str, props: dict) -> str:
    if label == "Argument":
        return "argument"
    for key in ("statement", "name", "target", "description", "text",
                "question_summary", "span_id", "id"):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            text = " ".join(value.split())
            return text if len(text) <= 78 else text[:75] + "..."
    return label


# --------------------------------------------------------------------------
# tracked entities: (slug, label, key property, key value)
# --------------------------------------------------------------------------
class Recorder:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.refs: list[dict] = []
        self.nodes: dict[str, dict] = {}       # eid -> node record
        self.node_ref: dict[str, str] = {}     # eid -> slug
        self.edges: dict[str, dict] = {}       # start|type|end -> edge record
        self.events: list[dict] = []
        self.prev_nodes: dict[str, dict] = {}
        self.prev_edges: dict[str, dict] = {}
        self.batch = 0
        self.batch_title = ""

    # -- registration ------------------------------------------------------
    def track(self, slug: str, label: str, key: str, value: str) -> None:
        self.refs.append({"slug": slug, "label": label, "key": key, "value": value})

    # -- snapshot ----------------------------------------------------------
    def _snapshot(self) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
        rows = self.graph.run(
            "UNWIND $refs AS ref "
            "MATCH (n) WHERE ref.label IN labels(n) AND n[ref.key] = ref.value "
            "RETURN ref.slug AS slug, elementId(n) AS eid, labels(n) AS labels, "
            "       properties(n) AS props",
            refs=self.refs,
        )
        nodes: dict[str, dict] = {}
        for row in rows:
            label = next((l for l in row["labels"] if l in LABEL_KB), row["labels"][0])
            eid = row["eid"]
            nodes[eid] = {
                "ref": row["slug"],
                "eid": eid,
                "label": label,
                "ontology": LABEL_KB.get(label, "development"),
                "layer": layer_of(label),
                "name": name_of(label, row["props"]),
                "props": jsonable(row["props"]),
                "created_at": jsonable(row["props"].get("created_at")),
                "created_ms": epoch_ms(row["props"].get("created_at")),
            }

        node_ref = {eid: n["ref"] for eid, n in nodes.items()}
        context: dict[str, dict] = {}
        edges: dict[str, dict] = {}
        if nodes:
            rels = self.graph.run(
                "MATCH (a)-[r]->(b) "
                "WHERE elementId(a) IN $ids OR elementId(b) IN $ids "
                "RETURN elementId(a) AS a, elementId(b) AS b, type(r) AS type, "
                "       properties(r) AS props, labels(a) AS al, labels(b) AS bl, "
                "       properties(a) AS ap, properties(b) AS bp",
                ids=list(nodes),
            )
            for rel in rels:
                for eid, labels, props in (
                    (rel["a"], rel["al"], rel["ap"]),
                    (rel["b"], rel["bl"], rel["bp"]),
                ):
                    if eid in nodes or eid in context:
                        continue
                    label = next((l for l in labels if l in LABEL_KB), labels[0])
                    context[eid] = {
                        "ref": f"ctx-{len(context)}",
                        "eid": eid,
                        "label": label,
                        "ontology": LABEL_KB.get(label, "development"),
                        "layer": layer_of(label),
                        "name": name_of(label, props),
                        "props": jsonable(props),
                        "created_at": jsonable(props.get("created_at")),
                        "created_ms": epoch_ms(props.get("created_at")),
                        "pre": True,
                    }
                    node_ref[eid] = context[eid]["ref"]
                key = f"{node_ref[rel['a']]}|{rel['type']}|{node_ref[rel['b']]}"
                from_ont = (nodes if rel["a"] in nodes else context)[rel["a"]]["ontology"]
                to_ont = (nodes if rel["b"] in nodes else context)[rel["b"]]["ontology"]
                from_layer = (nodes if rel["a"] in nodes else context)[rel["a"]]["layer"]
                to_layer = (nodes if rel["b"] in nodes else context)[rel["b"]]["layer"]
                edge_layer = ("canonical" if from_layer == to_layer == "canonical"
                              else "telemetry" if "telemetry" in (from_layer, to_layer)
                              else "truth")
                edges[key] = {
                    "key": key,
                    "from": node_ref[rel["a"]],
                    "to": node_ref[rel["b"]],
                    "type": rel["type"],
                    "layer": edge_layer,
                    "ontology": edge_ontology(rel["type"], from_ont, to_ont),
                    "props": jsonable(rel["props"]),
                    "created_at": jsonable(rel["props"].get("created_at")),
                    "created_ms": epoch_ms(rel["props"].get("created_at")),
                }
        return nodes, edges, context

    # -- step --------------------------------------------------------------
    def step(self, title: str, action) -> None:
        self.batch += 1
        self.batch_title = title
        action()
        time.sleep(STEP_PAUSE)
        observed = int(time.time() * 1000)
        nodes, edges, context = self._snapshot()

        created: list[dict] = []
        for eid, node in nodes.items():
            before = self.prev_nodes.get(eid)
            if before is None:
                node["pre"] = False
                created.append(node)
            else:
                changed = {
                    k: [before["props"].get(k), node["props"].get(k)]
                    for k in set(before["props"]) | set(node["props"])
                    if k != "created_at" and before["props"].get(k) != node["props"].get(k)
                }
                if changed:
                    self._emit(observed, "node.update", node, {
                        "title": f"{node['label']} updated",
                        "detail": ", ".join(
                            f"{k} {json.dumps(v[0])} → {json.dumps(v[1])}"
                            for k, v in sorted(changed.items())
                        ),
                        "changed": changed,
                        "source": "observed",
                        "step": title,
                    })
                node["pre"] = before.get("pre", False)

        for key, edge in edges.items():
            before = self.prev_edges.get(key)
            if before is None:
                created.append(edge)
            else:
                changed = {
                    k: [before["props"].get(k), edge["props"].get(k)]
                    for k in set(before["props"]) | set(edge["props"])
                    if k != "created_at" and before["props"].get(k) != edge["props"].get(k)
                }
                if changed:
                    self._emit(observed, "edge.update", edge, {
                        "title": f"{edge['type']} updated",
                        "detail": ", ".join(
                            f"{k} {json.dumps(v[0])} → {json.dumps(v[1])}"
                            for k, v in sorted(changed.items())
                        ),
                        "changed": changed,
                        "source": "observed",
                        "step": title,
                    })

        created.sort(key=lambda r: (r["created_ms"] or observed,
                                    "type" in r, r.get("label", "")))
        for record in created:
            if "type" in record:
                self._emit(record["created_ms"] or observed, "edge.create", record, {
                    "title": f"{record['type']} created",
                    "detail": f"{record['from']} → {record['to']}",
                    "source": "graph" if record["created_ms"] else "observed",
                    "step": title,
                })
            else:
                self._emit(record["created_ms"] or observed, "node.create", record, {
                    "title": f"{record['label']} created",
                    "detail": record["name"],
                    "source": "graph" if record["created_ms"] else "observed",
                    "step": title,
                })

        for key, edge in self.prev_edges.items():
            if key not in edges:
                edge["deleted"] = True
                self._emit(observed, "edge.delete", edge, {
                    "title": f"{edge['type']} removed",
                    "detail": f"{edge['from']} → {edge['to']}",
                    "source": "observed", "step": title,
                })
        for eid, node in self.prev_nodes.items():
            if eid not in nodes and not node.get("pre"):
                node["deleted"] = True
                self._emit(observed, "node.delete", node, {
                    "title": f"{node['label']} removed",
                    "detail": node["name"],
                    "source": "observed", "step": title,
                })

        self.prev_nodes = nodes
        self.prev_edges = edges
        self.nodes.update(nodes)
        self.nodes.update(context)
        self.node_ref.update({eid: n["ref"] for eid, n in context.items()})
        self.edges.update(edges)

    def _emit(self, t_ms: int, kind: str, record: dict, extra: dict) -> None:
        self.events.append({
            "seq": len(self.events) + 1,
            "kind": kind,
            "t": iso(t_ms),
            "t_ms": t_ms,
            "ontology": record["ontology"],
            "label": record.get("label") or record.get("type"),
            "ref": record["ref"] if "ref" in record else record["key"],
            "props": record["props"],
            "batch": self.batch,
            "batch_title": self.batch_title,
            **extra,
        })

    # -- output ------------------------------------------------------------
    def payload(self, meta: dict) -> dict:
        events = sorted(self.events, key=lambda e: (e["batch"], e["t_ms"], e["seq"]))
        for index, event in enumerate(events, 1):
            event["seq"] = index
        if events:
            meta["recorded_span_ms"] = events[-1]["t_ms"] - events[0]["t_ms"]
            meta["started_at"] = events[0]["t"]
            meta["ended_at"] = events[-1]["t"]
        batches = []
        for event in events:
            if not batches or batches[-1]["id"] != event["batch"]:
                batches.append({
                    "id": event["batch"], "title": event["batch_title"],
                    "t_ms": event["t_ms"], "first_seq": event["seq"],
                    "last_seq": event["seq"], "count": 0,
                })
            batches[-1]["count"] += 1
            batches[-1]["last_seq"] = event["seq"]
            batches[-1]["t_ms"] = min(batches[-1]["t_ms"], event["t_ms"])
        meta["bundles"] = len(batches)
        return {
            "meta": meta,
            "batches": batches,
            "ontologies": [
                {
                    "key": kb.key,
                    "name": kb.name,
                    "labels": list(kb.labels),
                    "relationships": list(kb.relationships),
                }
                for package in registry.installed_packages()
                for kb in package.knowledge_bases
            ],
            "nodes": list(self.nodes.values()),
            "edges": list(self.edges.values()),
            "events": events,
        }


def reset(graph: Graph) -> None:
    """Remove a previous recording so reruns replace the same entities.

    Resolution arguments and their derivation claims are keyed by
    `arg::<hash>` / `claim::<hash>` rather than the demo prefix, so they have
    to be found through the chunk they were resolved against; leaving them
    behind would keep their old `created_at` and interleave them into a rerun.
    """
    stale_claims = graph.run(
        "MATCH (a:Argument)-[:CONCLUDES]->(c:Claim) "
        "WHERE a.chunk_id STARTS WITH $p RETURN DISTINCT c.id AS id",
        p=DEMO_PREFIX,
    )
    for row in stale_claims:
        graph.run("MATCH (c:Claim {id: $id}) DETACH DELETE c", id=row["id"])
    stale_args = graph.run(
        "MATCH (a:Argument) WHERE a.chunk_id STARTS WITH $p RETURN a.id AS id",
        p=DEMO_PREFIX,
    )
    for row in stale_args:
        graph.run("MATCH (a:Argument {id: $id}) DETACH DELETE a", id=row["id"])
    graph.run(
        "MATCH (n) WHERE n.id STARTS WITH $p OR n.span_id STARTS WITH $p "
        "   OR n.session_id STARTS WITH $p "
        "DETACH DELETE n",
        p=DEMO_PREFIX,
    )
    graph.run(
        "MATCH (n:BusinessTerm) WHERE n.created_by = 'demo-recorder' DETACH DELETE n"
    )
    graph.run(
        "MATCH (n:AlternateBusinessTerm) WHERE n.name = 'Annual Recurring Revenue' "
        "DETACH DELETE n"
    )


def scenario(rec: Recorder, graph: Graph) -> None:
    store = TraceStore(graph)

    def node(slug, label, key_prop, key_value, props):
        graph.upsert_node(label, {key_prop: key_value}, props)
        rec.track(slug, label, key_prop, key_value)

    def relate(label, key, rel, end_label, end_key, props=None):
        graph.upsert_rel(label, key, rel, end_label, end_key, props)

    def source(slug, name, url):
        node(slug, "SourceNode", "id", f"{DEMO_PREFIX}{slug}", {
            "name": name, "type": "documentation", "url": url,
            "reliability": "high", "observed_at": "2026-11-03T00:00:00+00:00",
        })

    def chunk(slug, source_slug, index, text):
        node(slug, "Chunk", "id", f"{DEMO_PREFIX}{slug}", {
            "text": text, "source_id": f"{DEMO_PREFIX}{source_slug}", "index": index,
        })
        relate("Chunk", {"id": f"{DEMO_PREFIX}{slug}"}, "FOUND_IN",
               "SourceNode", {"id": f"{DEMO_PREFIX}{source_slug}"},
               {"role": "extracted"})
        return f"{DEMO_PREFIX}{slug}"

    def claim(slug, statement, chunk_slug, claim_type="fact", confidence=0.92):
        cid = f"{DEMO_PREFIX}{slug}"
        node(slug, "Claim", "id", cid, {
            "statement": statement, "claim_type": claim_type, "status": "active",
            "confidence": confidence, "created_by": "demo-recorder",
        })
        relate("Claim", {"id": cid}, "FOUND_IN", "Chunk",
               {"id": f"{DEMO_PREFIX}{chunk_slug}"}, {"role": "extracted"})
        return cid

    def trace_claim(slug, statement):
        cid = f"{DEMO_PREFIX}{slug}"
        node(slug, "Claim", "id", cid, {
            "statement": statement, "claim_type": "question", "status": "open",
            "confidence": 0.95, "created_by": "demo-recorder",
        })
        ground_claim_in_trace(graph, TRACE_ID, cid)
        return cid

    def argument(arg_slug, conclusion_label, conclusion_key, premises):
        aid = f"{DEMO_PREFIX}arg-{arg_slug}"
        node(f"arg-{arg_slug}", "Argument", "id", aid, {"status": "in"})
        relate("Argument", {"id": aid}, "CONCLUDES", conclusion_label, conclusion_key)
        for premise in premises:
            relate("Argument", {"id": aid}, "USES_PREMISE", "Claim", {"id": premise})
        return aid

    def justified(slug, label, name, description, arg_slug, premises, edges=()):
        eid = f"{DEMO_PREFIX}{slug}"
        props = {"name": name}
        if description:
            props["description"] = description
        node(slug, label, "id", eid, props)
        argument(arg_slug, label, {"id": eid}, premises)
        for rel, end_label, end_slug, eprops in edges:
            relate(label, {"id": eid}, rel, end_label,
                   {"id": f"{DEMO_PREFIX}{end_slug}"}, eprops or None)
        return eid

    C: dict[str, str] = {}
    CH: dict[str, str] = {}

    # ------------------------------------------------------------------ turn
    rec.track("session", "Session", "id", SESSION_ID)
    rec.track("trace", "RoutingTrace", "id", TRACE_ID)

    def turn():
        store.record_turn_start(TRACE_ID, SESSION_ID, QUESTION, client="noesis-demo")
        C["q_model"] = trace_claim(
            "claim-q-model",
            "The user asked which Muse model to use to build a coding agent.",
        )
        C["q_call"] = trace_claim(
            "claim-q-call",
            "The user asked how to call it from OpenCode.",
        )

    rec.step("Turn · the question", turn)

    rec.track("span", "Span", "span_id", SPAN_ID)
    rec.step("Tool call · get_claim_provenance", lambda: store.insert_span(SpanEvent(
        span_id=SPAN_ID, trace_id=TRACE_ID, session_id=SESSION_ID,
        name="tool.call get_claim_provenance", kind=SpanKind.tool_call,
        knowledge_base="epistemology", skill=TOOL_KEY,
        started_at=datetime.now(timezone.utc),
    )))

    # -------------------------------------------------------------- evidence
    def overview_source():
        source("src-meta-overview", "Meta Model API · Overview",
               "https://dev.meta.ai/docs/overview")
        CH["chunk-overview-api"] = chunk(
            "chunk-overview-api", "src-meta-overview", 0,
            "Meta Model API is an inference API for the latest models from Meta, "
            "including Muse Wonder and Muse Spark, and Muse Code is a coding agent "
            "built for Muse Spark. Call the API to build your own app or agent, or "
            "run Muse Code for a ready-made agent in your terminal — same models, "
            "same auth.",
        )
        CH["chunk-overview-code"] = chunk(
            "chunk-overview-code", "src-meta-overview", 1,
            "Muse Code is a coding agent from Meta for the terminal and CI, built "
            "for Muse Spark and powered by Model API. It plans, edits, and runs "
            "commands to do tasks, with approvals and an OS sandbox on from the "
            "first run.",
        )
        CH["chunk-overview-protocols"] = chunk(
            "chunk-overview-protocols", "src-meta-overview", 2,
            "Responses API: agentic, multi-step workloads with reasoning replay and "
            "server-managed state. Chat Completions API is the OpenAI-compatible "
            "messages-array endpoint. Messages API is the Anthropic "
            "Messages-compatible endpoint.",
        )
        CH["chunk-overview-capabilities"] = chunk(
            "chunk-overview-capabilities", "src-meta-overview", 3,
            "Tool calling connects the model to your APIs with parallel, streamed "
            "tool calls. Reasoning dials reasoning effort up or down per request. "
            "Structured output returns valid JSON that matches your schema every "
            "time.",
        )

    rec.step("Evidence · Meta Model API overview", overview_source)

    def models_source():
        source("src-meta-models", "Meta Model API · Models",
               "https://dev.meta.ai/docs/models")
        CH["chunk-models-wonder"] = chunk(
            "chunk-models-wonder", "src-meta-models", 0,
            "Muse Wonder takes text, image, video, audio, and PDF input, returns "
            "text, has a 1,048,576-token context window, and supports reasoning "
            "effort levels from \"minimal\" to \"max\". Muse Wonder comes in two "
            "variants. Muse Wonder is on Model API and in Muse Code. Muse Wonder "
            "Full, the higher-capability variant, is available only in Muse Code "
            "with a subscription, so setting reasoning effort to \"max\" on Muse "
            "Wonder doesn't select it.",
        )
        CH["chunk-models-spark"] = chunk(
            "chunk-models-spark", "src-meta-models", 1,
            "Muse Spark is Meta's model for agentic and coding work, such as "
            "multi-step tool loops, software engineering assistants, and "
            "long-context reasoning. All Muse Spark versions take text, image, "
            "video, audio, and PDF input, return text, and have a 1,048,576-token "
            "context window. Muse Spark 1.3 is the latest version, tuned for "
            "agentic workflows (multi-step tool, browser, and long-horizon tasks) "
            "with improved coding over 1.2. Muse Wonder and Muse Spark take the "
            "same requests on the same endpoints, so moving a workload between "
            "them is a model ID change.",
        )
        CH["chunk-models-audio"] = chunk(
            "chunk-models-audio", "src-meta-models", 2,
            "Audio understanding in Muse Spark 1.3 is not fully supported, and "
            "response quality for requests that include audio may be degraded. "
            "For audio, use Muse Spark 1.2, or Muse Voice Transcribe for dedicated "
            "speech-to-text.",
        )
        CH["chunk-models-image"] = chunk(
            "chunk-models-image", "src-meta-models", 3,
            "Muse Image is Meta's image generation model line. Reach Muse Image "
            "two ways: the Responses API for conversational, multi-turn editing, "
            "or the single-shot /v1/images/generations and /v1/images/edits "
            "endpoints for one-off calls.",
        )
        CH["chunk-models-voice"] = chunk(
            "chunk-models-voice", "src-meta-models", 4,
            "Muse Voice Transcribe is Meta's speech-to-text model on Meta Model "
            "API. Reach Muse Voice Transcribe two ways: the realtime WebSocket "
            "endpoint (wss://api.meta.ai/v1/asr/realtime) for live audio, or the "
            "file endpoint (POST /v1/asr/transcribe) for a recording you already "
            "have.",
        )
        CH["chunk-models-open"] = chunk(
            "chunk-models-open", "src-meta-models", 5,
            "Muse Spark Open is the open-weight release of Muse Spark: an "
            "816B-parameter mixture-of-experts model with 42B active parameters. "
            "Instead of calling it over Model API, you download the BF16 or NVFP4 "
            "checkpoint and serve it on your own GPUs with a runtime such as "
            "SGLang or vLLM, or call it through a hosted inference provider.",
        )

    rec.step("Evidence · Meta Model API models reference", models_source)

    def quickstart_source():
        source("src-meta-quickstart", "Meta Model API · Quickstart",
               "https://dev.meta.ai/docs/quickstart")
        CH["chunk-quickstart-connect"] = chunk(
            "chunk-quickstart-connect", "src-meta-quickstart", 0,
            "Model API is drop-in compatible with the OpenAI SDK, the Anthropic "
            "SDK, and OpenAI-compatible agent CLIs, so most stacks work with a "
            "base-URL and key change. Most tools ask for three values: the base "
            "URL https://api.meta.ai/v1, your MODEL_API_KEY, and the model ID "
            "muse-spark-1.3.",
        )
        CH["chunk-quickstart-opencode"] = chunk(
            "chunk-quickstart-opencode", "src-meta-quickstart", 1,
            "Manual setup follows the same shape everywhere: register a provider, "
            "point it at the base URL, and select muse-spark-1.3. The include: "
            "[\"reasoning.encrypted_content\"] setting is what carries Muse "
            "Spark's reasoning across turns.",
        )

    rec.step("Evidence · Meta Model API quickstart", quickstart_source)

    # ------------------------------------------------------------ interfaces
    def responses():
        C["responses"] = claim(
            "claim-responses",
            "Responses API: agentic, multi-step workloads with reasoning replay "
            "and server-managed state.",
            "chunk-overview-protocols",
        )
        C["values"] = claim(
            "claim-values",
            "Most tools ask for three values: the base URL "
            "https://api.meta.ai/v1, your MODEL_API_KEY, and the model ID "
            "muse-spark-1.3.",
            "chunk-quickstart-connect",
        )
        justified("iface-responses", "Interface", "Responses API",
                  "Agentic, multi-step workloads with reasoning replay and "
                  "server-managed state.",
                  "responses", [C["responses"], C["values"], C["q_call"]])

    rec.step("Interface · Responses API", responses)

    def chat_completions():
        C["chat"] = claim(
            "claim-chat",
            "Chat Completions API is the OpenAI-compatible messages-array "
            "endpoint.",
            "chunk-overview-protocols",
        )
        C["compat"] = claim(
            "claim-compat",
            "Model API is drop-in compatible with the OpenAI SDK, the Anthropic "
            "SDK, and OpenAI-compatible agent CLIs, so most stacks work with a "
            "base-URL and key change.",
            "chunk-quickstart-connect",
        )
        C["opencode"] = claim(
            "claim-opencode",
            "Manual setup follows the same shape everywhere: register a provider, "
            "point it at the base URL, and select muse-spark-1.3.",
            "chunk-quickstart-opencode",
        )
        justified("iface-chat", "Interface", "Chat Completions API",
                  "The OpenAI-compatible messages-array endpoint.",
                  "chat", [C["chat"], C["compat"], C["opencode"]])

    rec.step("Interface · Chat Completions API", chat_completions)

    def messages():
        C["messages"] = claim(
            "claim-messages",
            "Messages API is the Anthropic Messages-compatible endpoint.",
            "chunk-overview-protocols",
        )
        justified("iface-messages", "Interface", "Messages API",
                  "The Anthropic Messages-compatible endpoint.",
                  "messages", [C["messages"], C["compat"]])

    rec.step("Interface · Messages API", messages)

    def images_api():
        C["image"] = claim(
            "claim-image",
            "Muse Image is Meta's image generation model line. Reach Muse Image "
            "two ways: the Responses API for conversational, multi-turn editing, "
            "or the single-shot /v1/images/generations and /v1/images/edits "
            "endpoints for one-off calls.",
            "chunk-models-image",
        )
        justified("iface-images", "Interface", "Images API",
                  "Single-shot image generation and editing endpoints.",
                  "images", [C["image"]])

    rec.step("Interface · Images API", images_api)

    def asr_realtime():
        C["voice"] = claim(
            "claim-voice",
            "Muse Voice Transcribe is Meta's speech-to-text model on Meta Model "
            "API. Reach Muse Voice Transcribe two ways: the realtime WebSocket "
            "endpoint (wss://api.meta.ai/v1/asr/realtime) for live audio, or the "
            "file endpoint (POST /v1/asr/transcribe) for a recording you already "
            "have.",
            "chunk-models-voice",
        )
        justified("iface-asr", "Interface", "ASR Realtime",
                  "Realtime WebSocket endpoint for live audio transcription.",
                  "asr", [C["voice"]])

    rec.step("Interface · ASR Realtime", asr_realtime)

    # ---------------------------------------------------------- capabilities
    def tool_calling():
        C["tool"] = claim(
            "claim-tool",
            "Tool calling connects the model to your APIs with parallel, "
            "streamed tool calls.",
            "chunk-overview-capabilities",
        )
        justified("cap-tool", "Capability", "Tool calling", None,
                  "tool", [C["tool"]])

    rec.step("Capability · Tool calling", tool_calling)

    def structured_output():
        C["structured"] = claim(
            "claim-structured",
            "Structured output returns valid JSON that matches your schema every "
            "time.",
            "chunk-overview-capabilities",
        )
        justified("cap-structured", "Capability", "Structured output", None,
                  "structured", [C["structured"]])

    rec.step("Capability · Structured output", structured_output)

    def reasoning():
        C["reasoning_cap"] = claim(
            "claim-reasoning-cap",
            "Reasoning dials reasoning effort up or down per request.",
            "chunk-overview-capabilities",
        )
        C["reasoning"] = claim(
            "claim-reasoning",
            "The include: [\"reasoning.encrypted_content\"] setting is what "
            "carries Muse Spark's reasoning across turns.",
            "chunk-quickstart-opencode",
        )
        justified("cap-reasoning", "Capability", "Reasoning", None,
                  "reasoning", [C["reasoning_cap"], C["reasoning"]])

    rec.step("Capability · Reasoning", reasoning)

    # --------------------------------------------------------------- models
    def muse_wonder():
        C["wonder_api"] = claim(
            "claim-wonder-api",
            "Muse Wonder is on Model API and in Muse Code.",
            "chunk-models-wonder",
        )
        C["wonder_spec"] = claim(
            "claim-wonder-spec",
            "Muse Wonder takes text, image, video, audio, and PDF input, returns "
            "text, has a 1,048,576-token context window, and supports reasoning "
            "effort levels from \"minimal\" to \"max\".",
            "chunk-models-wonder",
        )
        C["same"] = claim(
            "claim-same-endpoints",
            "Muse Wonder and Muse Spark take the same requests on the same "
            "endpoints, so moving a workload between them is a model ID change.",
            "chunk-models-spark",
        )
        justified("comp-muse-wonder", "Component", "Muse Wonder",
                  "Meta's most capable model, built for complex software "
                  "engineering and long-horizon agentic work.",
                  "muse-wonder",
                  [C["wonder_api"], C["wonder_spec"], C["same"]],
                  edges=(
                      ("EXPOSES", "Interface", "iface-chat", None),
                      ("EXPOSES", "Interface", "iface-messages", None),
                      ("IMPLEMENTS", "Capability", "cap-tool", None),
                      ("IMPLEMENTS", "Capability", "cap-structured", None),
                      ("IMPLEMENTS", "Capability", "cap-reasoning", None),
                  ))

    rec.step("Component · Muse Wonder", muse_wonder)

    def muse_spark():
        C["spark"] = claim(
            "claim-spark",
            "Muse Spark is Meta's model for agentic and coding work, such as "
            "multi-step tool loops, software engineering assistants, and "
            "long-context reasoning.",
            "chunk-models-spark",
        )
        C["spark_context"] = claim(
            "claim-spark-context",
            "All Muse Spark versions take text, image, video, audio, and PDF "
            "input, return text, and have a 1,048,576-token context window.",
            "chunk-models-spark",
        )
        C["spark_13"] = claim(
            "claim-spark-13",
            "Muse Spark 1.3 is the latest version, tuned for agentic workflows "
            "(multi-step tool, browser, and long-horizon tasks) with improved "
            "coding over 1.2.",
            "chunk-models-spark",
        )
        C["audio"] = claim(
            "claim-spark-audio",
            "Audio understanding in Muse Spark 1.3 is not fully supported, and "
            "response quality for requests that include audio may be degraded.",
            "chunk-models-audio",
        )
        justified("comp-muse-spark", "Component", "Muse Spark 1.3",
                  "Meta's model for agentic and coding work; the latest version, "
                  "tuned for multi-step tool, browser, and long-horizon tasks.",
                  "muse-spark",
                  [C["spark"], C["spark_context"], C["spark_13"], C["same"], C["audio"]],
                  edges=(
                      ("EXPOSES", "Interface", "iface-responses", None),
                      ("EXPOSES", "Interface", "iface-chat", None),
                      ("EXPOSES", "Interface", "iface-messages", None),
                      ("IMPLEMENTS", "Capability", "cap-tool", None),
                      ("IMPLEMENTS", "Capability", "cap-structured", None),
                      ("IMPLEMENTS", "Capability", "cap-reasoning", None),
                  ))

    rec.step("Component · Muse Spark 1.3", muse_spark)

    def muse_code():
        C["code"] = claim(
            "claim-code",
            "Muse Code is a coding agent from Meta for the terminal and CI, "
            "built for Muse Spark and powered by Model API.",
            "chunk-overview-code",
        )
        justified("comp-muse-code", "Component", "Muse Code",
                  "Coding agent from Meta for the terminal and CI, built for Muse "
                  "Spark and powered by Model API.",
                  "muse-code", [C["code"], C["q_model"]],
                  edges=(("DEPENDS_ON", "Component", "comp-muse-spark", None),))

    rec.step("Component · Muse Code", muse_code)

    def muse_wonder_full():
        C["wonder_full"] = claim(
            "claim-wonder-full",
            "Muse Wonder Full, the higher-capability variant, is available only "
            "in Muse Code with a subscription.",
            "chunk-models-wonder",
        )
        justified("comp-muse-wonder-full", "Component", "Muse Wonder Full",
                  "The higher-capability Muse Wonder variant, available only in "
                  "Muse Code with a subscription.",
                  "muse-wonder-full", [C["wonder_full"], C["wonder_api"]],
                  edges=(("DEPENDS_ON", "Component", "comp-muse-code", None),))

    rec.step("Component · Muse Wonder Full", muse_wonder_full)

    def muse_spark_open():
        C["open"] = claim(
            "claim-spark-open",
            "Muse Spark Open is the open-weight release of Muse Spark: an "
            "816B-parameter mixture-of-experts model with 42B active parameters.",
            "chunk-models-open",
        )
        justified("comp-muse-spark-open", "Component", "Muse Spark Open",
                  "Open-weight release of Muse Spark: 816B-parameter "
                  "mixture-of-experts with 42B active parameters.",
                  "muse-spark-open", [C["open"]])

    rec.step("Component · Muse Spark Open", muse_spark_open)

    def muse_image():
        justified("comp-muse-image", "Component", "Muse Image",
                  "Meta's image generation model line.",
                  "muse-image", [C["image"]],
                  edges=(("EXPOSES", "Interface", "iface-images", None),))

    rec.step("Component · Muse Image", muse_image)

    def muse_voice():
        justified("comp-muse-voice", "Component", "Muse Voice Transcribe",
                  "Meta's speech-to-text model on Model API.",
                  "muse-voice", [C["voice"]],
                  edges=(("EXPOSES", "Interface", "iface-asr", None),))

    rec.step("Component · Muse Voice Transcribe", muse_voice)

    # ---------------------------------------------------- reference vs canonical
    CHUNK_WONDER = f"{DEMO_PREFIX}chunk-models-wonder"
    WONDER = f"{DEMO_PREFIX}comp-muse-wonder"
    WONDER_FULL = f"{DEMO_PREFIX}comp-muse-wonder-full"
    RESPONSES = f"{DEMO_PREFIX}iface-responses"

    def register_resolution(chunk_id: str) -> None:
        seen = {ref["slug"] for ref in rec.refs}
        for row in graph.run(
            "MATCH (a:Argument) WHERE a.chunk_id = $c RETURN a.id AS id ORDER BY id",
            c=chunk_id,
        ):
            slug = f"res-arg-{row['id']}"
            if slug not in seen:
                rec.track(slug, "Argument", "id", row["id"])
                seen.add(slug)
        for row in graph.run(
            "MATCH (a:Argument)-[:CONCLUDES]->(c:Claim) WHERE a.chunk_id = $c "
            "RETURN DISTINCT c.id AS id ORDER BY id",
            c=chunk_id,
        ):
            slug = f"res-claim-{row['id']}"
            if slug not in seen:
                rec.track(slug, "Claim", "id", row["id"])
                seen.add(slug)

    def wrong_resolution():
        winner = graph.record_resolution_arguments(
            chunk_id=CHUNK_WONDER, entity_label="Component", surface_text="Muse Wonder",
            candidates=[
                {"id": WONDER_FULL, "score": 0.84, "status": "in"},
                {"id": WONDER, "score": 0.61, "status": "out"},
                {"id": f"{DEMO_PREFIX}comp-muse-new", "score": 0.20, "status": "out"},
            ],
            ref_id="models-wonder",
        )
        # The wrong canonical claim: Full is not on Model API.
        relate("Component", {"id": WONDER_FULL}, "EXPOSES", "Interface", {"id": RESPONSES}, {
            "id": f"{DEMO_PREFIX}rel-wonder-full-responses",
            "chunk_id": CHUNK_WONDER,
            "argument_id": winner,
            "status": "active",
        })
        relate("Chunk", {"id": CHUNK_WONDER}, "MENTIONS", "Component", {"id": WONDER_FULL}, {
            "id": f"{DEMO_PREFIX}mention-wonder-full",
            "chunk_id": CHUNK_WONDER,
            "argument_id": winner,
            "status": "active",
        })
        graph.upsert_node("Component", {"id": WONDER_FULL},
                          {"availability": "Model API"})
        graph.run(
            "MATCH (c:Component {id: $id}) SET c._property_origins = $origins",
            id=WONDER_FULL,
            origins=json.dumps({"availability": {
                "argument_id": winner, "chunk_id": CHUNK_WONDER, "status": "active",
            }}),
        )
        register_resolution(CHUNK_WONDER)

    rec.step("Reference · Muse Wonder resolves to Muse Wonder Full", wrong_resolution)

    def correction():
        graph.repoint_entity_reference(
            chunk_id=CHUNK_WONDER,
            old_entity_id=WONDER_FULL,
            new_entity_id=WONDER,
            new_entity_label="Component",
            reason="user_feedback",
            ref_id="models-wonder",
        )
        register_resolution(CHUNK_WONDER)

    rec.step("Correction · repoint Muse Wonder to the Model API variant",
             correction)

    # ----------------------------------------------------------------- review
    def review():
        graph.upsert_node("Claim", {"id": C["wonder_spec"]}, {
            "statement": "Muse Wonder takes text, image, video, audio, and PDF input, "
                         "returns text, has a 1,048,576-token context window, and "
                         "supports reasoning effort levels from \"minimal\" to \"max\".",
            "claim_type": "fact", "status": "active", "confidence": 0.97,
            "created_by": "demo-recorder",
        })
        graph.upsert_rel("Component", {"id": f"{DEMO_PREFIX}comp-muse-spark"},
                         "IMPLEMENTS", "Capability", {"id": f"{DEMO_PREFIX}cap-tool"},
                         {"weight": 0.35})
        graph.upsert_node("SourceNode", {"id": f"{DEMO_PREFIX}src-meta-overview"},
                          {"reliability": "medium"})

    rec.step("Review · corroborate and reweight", review)


def main() -> None:
    with Graph() as graph:
        reset(graph)
        rec = Recorder(graph)
        scenario(rec, graph)
        version = graph.run("CALL dbms.components() YIELD name, versions "
                            "RETURN name, versions[0] AS version")[0]
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, check=True, cwd=HERE,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            commit = "unknown"
        payload = rec.payload({
            "title": "One question, one justified graph: Meta Model API",
            "question": QUESTION,
            "session_id": SESSION_ID,
            "trace_id": TRACE_ID,
            "neo4j": f"{version['name']} {version['version']}",
            "commit": commit,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    kinds: dict[str, int] = {}
    for event in payload["events"]:
        kinds[event["kind"]] = kinds.get(event["kind"], 0) + 1
    print(f"recorded {len(payload['events'])} events, "
          f"{len(payload['nodes'])} nodes, {len(payload['edges'])} edges")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
