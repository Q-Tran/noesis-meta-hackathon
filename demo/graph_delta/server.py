"""Serve the graph-delta page plus a tool-calling chat endpoint.

The chat model is Muse Wonder 1.0 on Meta Model API (the same `meta` endpoint
OpenCode is configured with; override with CHAT_MODEL). Tools are sent as
native OpenAI `tools`, not pasted into the prompt; results are compact. Reads
go through the corpus/entity search; writes go through bounded graph tools
(no raw Cypher).

    set -a && source .env && set +a
    python3 demo/graph_delta/server.py

Then open http://127.0.0.1:8766/ (the tunnel can point at it).
"""
from __future__ import annotations

import json
import time
import os
import pathlib
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from chat_tools import TOOL_SPECS, ChatTools
import corpus
from graph import Graph
from knowledge_bases.tools.trace_store import TraceStore

HERE = pathlib.Path(__file__).resolve().parent
INDEX = HERE / "index.html"
SHOWCASE = HERE / "showcase.html"
API_URL = "https://api.meta.ai/v1/chat/completions"
API_URL_RESPONSES = "https://api.meta.ai/v1/responses"
MODEL = os.environ.get("CHAT_MODEL", "muse-wonder-1.0")
MAX_ITERATIONS = 11

SYSTEM = """You are Noesis, the graph orchestrator for the Meta Model API documentation.
You answer from the graph, never from memory. The graph contains 204 docs pages,
3,194 chunks, 16k claims, canonical entities, metrics, code entities and their
relationships.

Use tools for every factual claim:
- search_corpus for documentation evidence (page + URL + claims).
- search_entities for canonical entities (components, interfaces, capabilities,
  specifications, terms, metrics, code); hybrid lexical+vector search.
- get_entity for one entity's properties, claims and relationships.
Cite the page URLs returned by search_corpus. If the graph does not contain the
answer, say so instead of inventing it. When a tool needs an entity type, either
the bare label (`Component`) or the qualified name (`architecture.Component`) is
accepted.

Write to the graph only when the user explicitly asks to add or change
knowledge, and keep the tool budget tight: most tasks need at most 4-6 tool
calls -- search once or twice, look up with get_entity, then write. Do not
re-search what you already have.
- add_claim first. A docs-backed claim MUST pass the exact chunk_id returned by
  search_corpus, so it is FOUND_IN that Chunk. Only omit chunk_id when the claim
  is about the conversation itself; then it grounds in the current turn. Keep
  the statement to one assertion: never append a source list or URLs, because
  provenance is already an edge;
- add_argument to attach claims to the entity with an Argument;
- relate_entities to add a relationship, update_entity for allowed properties.
Never write speculatively. After writing, report exactly what changed with the
ids returned by the tools. Keep answers concise.

Corrections are first-class and epistemic -- you never delete and you never
re-ingest a chunk to fix the canonical layer:
- A user statement that the graph is stale or wrong ("the graph says X... that
  is stale: Y") IS a correction request, even without an imperative verb.
  Carry it out end to end without asking for confirmation: identify the
  canonical entity it names (search_entities, then get_entity), find the exact
  stale claim and the argument behind it from get_entity's claims/arguments,
  challenge_argument(claim_id=<the stale claim's id>, reason=<the user's stated
  reason, in their words>), then add_claim the updated fact (no chunk_id -- it
  grounds in this conversation) and add_argument concluding the same entity
  with the new claim. Report the ids of what you challenged and what you added.
  Do not search_corpus for this: the stale claim, its argument and its docs
  grounding are already on the entity -- get_entity is the whole lookup.
- challenge_argument when the user disputes an argument ("this argument doesn't
  make sense from this chunk", "the chunk doesn't say that"). It writes a new
  Argument that ATTACKS the disputed one with target=warrant; the reason becomes
  its premise claim, grounded in this conversation turn. The disputed argument
  keeps its status: an open challenge is visible in the ATTACKS relation, and
  the resolution is the next tool.
- repoint_reference when the user says a reference points at the wrong thing
  ("this entity/property/relationship refers to that other one"). You need the
  chunk_id (from search_corpus or the argument's chunk_id) and the old and new
  entity ids. The old argument is superseded (status 'out', its conclusion
  'superseded'), the new one installed (status 'in'), a feedback argument
  ATTACKS the old conclusion, canonical edges are rewired through SUPERSEDES
  pairs, and chunk-contributed properties transfer. The change originates from
  an argument with its own provenance -- that is the point to make when you
  report it.
When a user asks you to show this flow, do it end to end: find the grounding
chunk and argument first, challenge it, then repoint, and walk through what
each new node and edge means."""


def api_key() -> str:
    key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
    if key:
        return key
    auth = pathlib.Path.home() / ".local/share/opencode/auth.json"
    if auth.exists():
        return json.loads(auth.read_text())["meta"]["key"]
    raise SystemExit("set META_API_KEY or configure the meta provider in OpenCode")


def responses_tools() -> list[dict]:
    """The 9 bounded tools as DEFERRED Responses-API tools plus a tool_search
    marker: the model asks Meta's hosted tool search which of the 9 apply, the
    API loads only those. The agent loop thus uses the exact token-saving
    feature the demo is about."""
    tools = []
    for spec in TOOL_SPECS:
        fn = spec["function"]
        tools.append({
            "type": "function",
            "name": fn["name"],
            "description": fn.get("description", ""),
            "defer_loading": True,
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    tools.append({"type": "tool_search"})
    return tools


def model_call(key: str, input_items: list[dict]) -> dict:
    body = json.dumps({
        "model": MODEL,
        "input": input_items,
        "tools": responses_tools(),
        "tool_choice": "auto",
        "max_output_tokens": 1600,
        "temperature": 0,
        "reasoning": {"effort": "minimal"},
    }).encode()
    request = urllib.request.Request(
        API_URL_RESPONSES, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


def summarize(name: str, result: dict) -> dict:
    if "error" in result:
        return {"name": name, "error": result["error"]}
    if name == "search_corpus":
        return {"name": name, "hits": len(result.get("results", [])),
                "pages": [r.get("page") for r in result.get("results", [])[:3]]}
    if name == "search_entities":
        rows = result.get("results", [])
        return {"name": name, "hits": len(rows),
                "entities": [r.get("name") or r.get("identity") for r in rows[:3]]}
    if name == "add_claim":
        return {"name": name, "claim_id": result.get("claim_id"),
                "grounded_in": result.get("grounded_in")}
    if name == "add_argument":
        return {"name": name, "argument_id": result.get("argument_id"),
                "concludes": result.get("concludes")}
    if name in ("relate_entities", "update_entity"):
        return {"name": name, **{k: v for k, v in result.items() if k != "properties"}}
    return {"name": name, "ok": True}


# ---------------------------------------------------------------------------
# The showcase scenario: the "Tool search" capability and everything that
# justifies or corrects it, as a diffable before/after snapshot.

SHOWCASE_CAPABILITY_ID = corpus.SHOWCASE_CAPABILITY_ID  # "Tool search"


def showcase_subgraph() -> dict:
    """The scenario's affected subgraph; the query and assembly live in
    `corpus.showcase_subgraph` so the explorer app and this demo read the same
    story. The demo passes its own Graph so tests can fake it."""
    return corpus.showcase_subgraph(graph=Graph())


def showcase_reset() -> dict:
    """Delete only what the showcase scenario can create, so it can run again:
    chat arguments concluding the showcase capability, chat arguments
    attacking the scenario's stale argument, chat claims grounded in chat
    traces (the scenario's turns), and the chat traces with their spans.
    Docs chunks, canonical entities, and extraction-era claims are untouched.
    """
    graph = Graph()
    try:
        stale_arg = corpus.SHOWCASE_STALE_ARGUMENT_ID
        args = graph.run(
            "MATCH (n:Argument) WHERE n.id STARTS WITH 'chatarg-' "
            "AND ( EXISTS { (n)-[:ATTACKS]->(:Argument {id: $stale_arg}) } "
            "   OR EXISTS { (n)-[:CONCLUDES]->(:Capability {id: $cap_id}) } ) "
            "DETACH DELETE n RETURN count(*) AS c",
            stale_arg=stale_arg, cap_id=SHOWCASE_CAPABILITY_ID,
        )
        claims = graph.run(
            "MATCH (c:Claim)-[:FOUND_IN]->(t:RoutingTrace) "
            "WHERE c.id STARTS WITH 'chatclaim-' AND t.id STARTS WITH 'chat-' "
            "DETACH DELETE c RETURN count(*) AS c",
        )
        traces = graph.run(
            "MATCH (t:RoutingTrace) WHERE t.id STARTS WITH 'chat-' "
            "DETACH DELETE t RETURN count(*) AS c",
        )
        spans = graph.run(
            "MATCH (s:Span) WHERE s.trace_id STARTS WITH 'chat-' "
            "DETACH DELETE s RETURN count(*) AS c",
        )
        total = sum(r["c"] for rows in (args, claims, traces, spans) for r in rows)
        return {"deleted_nodes": total}
    finally:
        graph.close()


def turn_delta(graph, trace_id, writes) -> dict:
    """The graph delta of one chat turn: every node the turn created, every
    status the correction flipped, and a plain-English explanation keyed by
    where each change came from -- a docs chunk (extracted) or this
    conversation's trace (the user is the source).

    Deterministic: the writes list names the created ids and the trace id
    names the turn, so nothing is inferred from timestamps or guessed.
    """
    created_ids: list[str] = []
    attacks: list[dict] = []
    repoint = None
    for w in writes:
        for key in ("claim_id", "argument_id", "premise_claim_id",
                    "correction_claim_id", "new_arg_id"):
            if w.get(key):
                created_ids.append(w[key])
        if w.get("attack_argument_id"):
            created_ids.append(w["attack_argument_id"])
            attacks.append(w)
        if w.get("new_arg_id"):
            repoint = w
    ids = list(dict.fromkeys(created_ids))

    # The visual diff also draws what the correction defeated: the attacked
    # argument (⚔) and the repoint's retired argument/entity (Δ), next to the
    # created nodes (+). They are fetched with the same id-keyed query.
    marked_extra: list[tuple[str, str]] = []
    for w in attacks:
        if w.get("attacks"):
            marked_extra.append((w["attacks"], "⚔"))
    if repoint:
        if repoint.get("old_arg_id"):
            marked_extra.append((repoint["old_arg_id"], "Δ"))
        if repoint.get("deprecated_old_entity") and repoint.get("old_entity_id"):
            marked_extra.append((repoint["old_entity_id"], "Δ"))
    extra_marks = {i: m for i, m in marked_extra if i not in ids}
    all_ids = ids + [i for i in extra_marks]

    nodes: list[dict] = []
    if all_ids:
        rows = graph.run(
            "MATCH (n) WHERE n.id IN $ids "
            "RETURN elementId(n) AS element_id, labels(n) AS labels, properties(n) AS props",
            ids=all_ids,
        )
        for row in rows:
            props = row["props"] or {}
            title = next((str(props[k]) for k in
                          ("name", "statement", "surface_text", "id") if props.get(k)),
                         "(unnamed)")
            node_id = props.get("id")
            nodes.append({
                "element_id": row["element_id"],
                "id": node_id,
                "label": (row["labels"] or ["Node"])[0],
                "title": title[:160],
                "status": props.get("status"),
                "mark": "+" if node_id in ids else extra_marks.get(node_id),
            })
        # Created nodes keep the query's row order; the defeated extras follow
        # in the order the correction named them.
        extra_order = {node_id: i for i, (node_id, _) in enumerate(marked_extra)}
        nodes = ([n for n in nodes if n["mark"] == "+"]
                 + sorted((n for n in nodes if n["mark"] != "+"),
                          key=lambda n: extra_order.get(n["id"], 0)))

    # Source attribution: which created claims ground in this conversation's
    # trace (the user is the source) versus a docs chunk.
    trace_claims: set[str] = set()
    if ids and trace_id:
        rows = graph.run(
            "MATCH (c:Claim)-[:FOUND_IN]->(t:RoutingTrace {id: $tid}) "
            "WHERE c.id IN $ids RETURN c.id AS id",
            tid=trace_id, ids=ids,
        )
        trace_claims = {row["id"] for row in rows}

    # The turn's edges: everything touching a created node (created nodes have
    # no pre-existing edges, so the scope is exact).
    edges: list[dict] = []
    if ids:
        rows = graph.run(
            "MATCH (a)-[r]-(b) WHERE a.id IN $ids OR b.id IN $ids "
            "RETURN elementId(r) AS element_id, type(r) AS type, "
            "       a.id AS source_id, labels(a)[0] AS source_label, "
            "       coalesce(a.name, a.statement, a.id) AS source_title, "
            "       b.id AS target_id, labels(b)[0] AS target_label, "
            "       coalesce(b.name, b.statement, b.id) AS target_title "
            "LIMIT 80",
            ids=ids,
        )
        # Both endpoints of a correction's edges are usually in scope, and the
        # undirected match then returns each relationship twice (once per
        # binding). The visual draws per element id, so dedupe on it.
        seen_eids: set[str] = set()
        edges = []
        for row in rows:
            if row["element_id"] in seen_eids:
                continue
            seen_eids.add(row["element_id"])
            edges.append({
                "element_id": row["element_id"], "type": row["type"],
                "source": row["source_id"], "source_label": row["source_label"],
                "source_title": str(row.get("source_title") or row["source_id"]),
                "target": row["target_id"], "target_label": row["target_label"],
                "target_title": str(row.get("target_title") or row["target_id"]),
                "mark": ("⚔" if row["type"] == "ATTACKS"
                         else "Δ" if row["type"] in ("SUPERSEDES", "SUPERSEDED_BY")
                         else "+"),
            })
        # The undirected bindings leave an ATTACKS/SUPERSEDES edge pointing
        # either way; the writes name the attacker and the superseding side,
        # so orient the correction's edges deterministically (the visual's
        # arrows and node roles read source → target).
        orientation: dict[frozenset, tuple[str, str]] = {}
        for w in attacks:
            if w.get("attack_argument_id") and w.get("attacks"):
                orientation[frozenset((w["attack_argument_id"], w["attacks"]))] = (
                    w["attack_argument_id"], w["attacks"])
        if repoint and repoint.get("new_arg_id") and repoint.get("old_arg_id"):
            orientation[frozenset((repoint["new_arg_id"], repoint["old_arg_id"]))] = (
                repoint["new_arg_id"], repoint["old_arg_id"])
        for e in edges:
            want = orientation.get(frozenset((e["source"], e["target"])))
            if want and (e["source"], e["target"]) != want:
                e["source"], e["target"] = want
                e["source_label"], e["target_label"] = e["target_label"], e["source_label"]
                e["source_title"], e["target_title"] = e["target_title"], e["source_title"]

    # Status flips from the correction (deterministic from its result).
    flips: list[dict] = []
    if repoint:
        flips.append({"kind": "Argument", "title": repoint["old_arg_id"],
                      "from": "in", "to": "out"})
        if repoint.get("deprecated_old_entity"):
            flips.append({"kind": "Entity", "title": repoint["old_entity_id"],
                          "from": "active", "to": "Deprecated"})

    # The explanation, in audience order: what is new, why we know it (source),
    # what it defeats, what flipped.
    explanation: list[dict] = []
    for node in nodes:
        if node["label"] != "Claim":
            continue
        if node["id"] in trace_claims:
            explanation.append({
                "mark": "+", "text": f"Claim: {node['title']}",
                "why": "you are the source -- grounded in this conversation's trace, not in any docs chunk",
            })
        else:
            explanation.append({
                "mark": "+", "text": f"Claim: {node['title']}",
                "why": "grounded in a docs chunk",
            })
    for node in nodes:
        if node["label"] != "Argument" or node["id"] in (w.get("attack_argument_id") for w in attacks):
            continue
        concludes = next((e for e in edges if e["type"] == "CONCLUDES" and e["source"] == node["id"]), None)
        if concludes:
            explanation.append({
                "mark": "+", "text": f"Argument {node['id']} concludes {concludes['target_label']} {concludes['target']}",
                "why": "installed with status in",
            })
    for w in attacks:
        explanation.append({
            "mark": "⚔", "text": f"Argument {w['attack_argument_id']} attacks {w['attacks']} ({w.get('attack_target', 'warrant')})",
            "why": "the formal record of why the old claim no longer holds -- challenged, not deleted",
        })
    for flip in flips:
        explanation.append({
            "mark": "Δ", "text": f"{flip['kind']} {flip['title']}: {flip['from']} → {flip['to']}",
            "why": "superseded by the correction's new argument",
        })
    if repoint:
        explanation.append({
            "mark": "✓", "text": "the chunk was not re-ingested",
            "why": "the change originates from an argument with its own provenance",
        })

    return {"nodes": nodes, "edges": edges, "flips": flips, "explanation": explanation}


def _record_span(graph, trace_id, session_id, name, arguments, result, duration_ms) -> None:
    """Every tool call in the loop lands as a Span on the turn's trace, so the
    audience can open the turn and see exactly what was called, how long it
    took, and whether it failed. The chat is its own harness."""
    try:
        from datetime import datetime, timezone

        from knowledge_bases.tools.models import SpanEvent, SpanKind, SpanStatus

        now = datetime.now(timezone.utc)
        TraceStore(graph).insert_span(SpanEvent(
            trace_id=trace_id,
            session_id=session_id,
            name=name,
            kind=SpanKind.tool_call,
            knowledge_base="web-chat",
            started_at=now,
            ended_at=now,
            duration_ms=duration_ms,
            status=SpanStatus.error if "error" in result else SpanStatus.ok,
            error=result.get("error") if "error" in result else None,
            raw_input={"arguments": arguments},
            raw_output=summarize(name, result),
            client="web-chat",
        ))
    except Exception as exc:  # noqa: BLE001
        print("span error:", repr(exc), flush=True)


def chat(payload: dict, on_event=None) -> dict:
    """One chat turn. `on_event`, when given, receives progress dicts as the
    turn unfolds (thinking / tool_search / tool_start / tool_done) so a UI can
    stream the trace live; the return value is unchanged either way."""
    messages = payload.get("messages") or []
    if not messages:
        raise ValueError("messages is required")
    session_id = payload.get("session_id") or "web-chat"
    question = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    if not question.strip():
        raise ValueError("a user message is required")

    def emit(event: dict) -> None:
        if on_event is None:
            return
        try:
            on_event(event)
        except Exception:  # noqa: BLE001 -- progress must never break the turn
            pass

    key = api_key()
    graph = Graph()
    try:
        tools = ChatTools(graph, session_id)
        trace_id = tools.begin_turn(question)
        emit({"type": "start", "trace_id": trace_id})
        input_items = [{"role": "system", "content": SYSTEM}]
        for message in messages[-12:]:
            role = message.get("role")
            content = message.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                input_items.append({"role": role, "content": content})

        trace = []
        writes = []
        raw_writes = []
        tool_search_events = []
        for _iteration in range(MAX_ITERATIONS):
            emit({"type": "thinking", "iteration": _iteration + 1})
            response = model_call(key, input_items)
            output = response.get("output") or []
            input_items.extend(output)
            function_calls = []
            final_text = None
            for item in output:
                item_type = item.get("type")
                if item_type == "tool_search_call":
                    tool_search_events.append({"searched": item.get("arguments") or {}})
                elif item_type == "tool_search_output":
                    loaded = [t.get("name") for t in (item.get("tools") or [])]
                    if tool_search_events:
                        tool_search_events[-1]["loaded"] = loaded
                    else:
                        tool_search_events.append({"searched": {}, "loaded": loaded})
                    emit({"type": "tool_search", **tool_search_events[-1]})
                elif item_type == "function_call":
                    function_calls.append(item)
                elif item_type == "message":
                    texts = [c.get("text") for c in (item.get("content") or [])
                             if c.get("type") == "output_text" and c.get("text")]
                    if texts:
                        final_text = "\n".join(texts)
            if not function_calls:
                return {
                    "answer": final_text or "",
                    "tools": trace,
                    "writes": writes,
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "tool_search": tool_search_events,
                    "entities": list(tools.turn_entities.values()),
                    "model": MODEL,
                    "delta": _safe_delta(graph, trace_id, raw_writes),
                }
            for call in function_calls:
                name = call.get("name")
                try:
                    arguments = json.loads(call.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                handler = getattr(tools, name, None)
                emit({"type": "tool_start", "name": name})
                started = time.monotonic()
                if handler is None:
                    result = {"error": f"unknown tool {name}"}
                else:
                    try:
                        result = handler(**arguments)
                    except TypeError as exc:
                        result = {"error": f"bad arguments: {exc}"}
                    except Exception as exc:  # noqa: BLE001
                        result = {"error": str(exc)[:240]}
                duration_ms = int((time.monotonic() - started) * 1000)
                summary = summarize(name, result)
                trace.append(summary)
                _record_span(graph, trace_id, session_id, name, arguments, result, duration_ms)
                is_write = name in ("add_claim", "add_argument", "relate_entities",
                                    "update_entity", "challenge_argument",
                                    "repoint_reference") and "error" not in result
                emit({"type": "tool_done", "name": name, "duration_ms": duration_ms,
                      "summary": summary, "write": is_write})
                if is_write:
                    writes.append(summarize(name, result))
                    raw_writes.append(result)
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call.get("call_id"),
                    "output": json.dumps(result, default=str)[:4000],
                })
        stop_note = (
            "I made the writes you asked for (" + ", ".join(w["name"] for w in writes) + ")"
            if writes else
            "I stopped after several tool calls without reaching an answer."
        )
        return {
            "answer": stop_note,
            "tools": trace, "writes": writes,
            "trace_id": trace_id, "session_id": session_id,
            "tool_search": tool_search_events,
            "entities": list(tools.turn_entities.values()),
            "model": MODEL,
            "delta": _safe_delta(graph, trace_id, raw_writes),
        }
    finally:
        graph.close()


def trace_detail(trace_id: str) -> dict:
    """The recorded spans of one conversation turn; the query and shape live in
    `corpus.trace_detail` so the explorer app and this demo serve the same turn."""
    return corpus.trace_detail(trace_id, store=TraceStore(Graph()))


def _safe_delta(graph, trace_id, writes) -> dict | None:
    """The delta is a courtesy over the chat, never its failure mode."""
    if not writes:
        return None
    try:
        return turn_delta(graph, trace_id, writes)
    except Exception as exc:  # noqa: BLE001
        print("delta error:", repr(exc), flush=True)
        return None


class Handler(BaseHTTPRequestHandler):
    server_version = "noesis-chat"

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            body = INDEX.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/showcase":
            body = SHOWCASE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/health":
            self._json(200, {"ok": True, "model": MODEL})
            return
        if self.path == "/api/showcase/subgraph":
            try:
                self._json(200, showcase_subgraph())
            except Exception as exc:  # noqa: BLE001
                self._json(503, {"error": str(exc)[:240]})
            return
        if self.path.startswith("/api/trace/"):
            trace_id = self.path[len("/api/trace/"):]
            try:
                self._json(200, trace_detail(trace_id))
            except Exception as exc:  # noqa: BLE001
                self._json(503, {"error": str(exc)[:240]})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/showcase/reset":
            try:
                self._json(200, showcase_reset())
            except Exception as exc:  # noqa: BLE001
                self._json(503, {"error": str(exc)[:240]})
            return
        if self.path != "/api/chat":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._json(200, chat(payload))
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            print("chat error:", repr(exc), flush=True)
            self._json(500, {"error": "chat failed; see server log"})

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    host = os.environ.get("CHAT_HOST", "127.0.0.1")
    port = int(os.environ.get("CHAT_PORT", "8766"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"serving {INDEX} + /api/chat on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
