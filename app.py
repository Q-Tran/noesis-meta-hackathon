"""Noesis web UI: a snappy, read-only explorer over the whole graph corpus.

Serves the self-contained single-page app in ``web/index.html``, a small JSON
API backed by ``corpus.py``, and the ask-agent endpoint.

Neo4j must be started separately (``./scripts/start_local.sh``). The page loads
even when Neo4j is down: graph-backed endpoints answer 503 and the UI shows its
offline state instead of refusing to boot.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import corpus
from graph import Graph

# The Ask tab's agent loop is the proven demo loop (bounded tools, Meta tool
# search, turn spans, per-turn delta). It lives in demo/graph_delta/server.py;
# the explorer app serves it rather than re-implementing it.
_DEMO_DIR = Path(__file__).resolve().parent / "demo" / "graph_delta"
import sys  # noqa: E402

if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

import server as demo_server  # noqa: E402

WEB_DIR = Path(__file__).resolve().parent / "web"

# One driver per process, not one per request: Graph() builds a neo4j driver,
# and the handshake cost is exactly what makes a UI feel sluggish. The driver
# is thread-safe; Graph.run() opens a session per call.
_shared_graph: Graph | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if _shared_graph is not None:
        _shared_graph.close()


app = FastAPI(title="Noesis Corpus Explorer", lifespan=lifespan)

# The full-layer payload is multi-megabyte JSON. Level-1 gzip is the right
# trade on a small host: a little larger on the wire, several seconds less
# compression per load. Small responses stay uncompressed.
app.add_middleware(GZipMiddleware, minimum_size=100_000, compresslevel=1)


def get_graph() -> Graph:
    """FastAPI dependency, overridable in tests."""
    global _shared_graph
    if _shared_graph is None:
        _shared_graph = Graph()
    return _shared_graph


def _query(fn, *args, **kwargs):
    """Convert a dead/unreachable Neo4j into a 503 the UI can render."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}") from exc


@app.get("/health")
def health(graph: Graph = Depends(get_graph)):
    """Process health (the page serves even when Neo4j is down), with graph
    reachability as an informational field for the monitor polling this."""
    graph_up = False
    try:
        graph.run("RETURN 1")
        graph_up = True
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "graph": "up" if graph_up else "down"}


class AskRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def landing() -> HTMLResponse:
    return HTMLResponse(
        (WEB_DIR / "landing.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/app", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(
        (WEB_DIR / "index.html").read_text(encoding="utf-8"),
        # The page is small and read per request; no-cache keeps phones from
        # holding a stale copy between deploys.
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/showcase/subgraph")
def api_showcase_subgraph(graph: Graph = Depends(get_graph)):
    return _query(corpus.showcase_subgraph, graph=graph)


@app.get("/api/overview")
def api_overview(refresh: bool = False, graph: Graph = Depends(get_graph)):
    return _query(corpus.overview, graph, refresh=refresh)


@app.get("/api/growth")
def api_growth(days: int = Query(30, ge=1, le=365), graph: Graph = Depends(get_graph)):
    return _query(corpus.growth, days, graph)


@app.get("/api/search")
def api_search(
    q: str = "",
    label: str | None = None,
    limit: int = Query(25, ge=1, le=100),
    graph: Graph = Depends(get_graph),
):
    return _query(corpus.search_nodes, q, label, limit, graph)


@app.get("/api/node/{element_id}")
def api_node(element_id: str, graph: Graph = Depends(get_graph)):
    node = _query(corpus.node_detail, element_id, graph)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@app.get("/api/node/{element_id}/neighbors")
def api_neighbors(
    element_id: str,
    limit: int = Query(40, ge=1, le=200),
    graph: Graph = Depends(get_graph),
):
    neighborhood = _query(corpus.neighbors, element_id, limit, graph)
    if neighborhood is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return neighborhood


@app.get("/api/activity")
def api_activity(limit: int = Query(12, ge=1, le=100), graph: Graph = Depends(get_graph)):
    return _query(corpus.recent_activity, limit, graph)


@app.get("/api/graph/seed")
def api_graph_seed(limit: int = Query(25, ge=1, le=100), graph: Graph = Depends(get_graph)):
    seed = _query(corpus.graph_seed, limit, graph)
    if seed is None:
        raise HTTPException(status_code=404, detail="The corpus is empty")
    return seed


@app.get("/api/graph/subset")
def api_graph_subset(
    limit: int = Query(100, ge=1, le=500),
    refresh: bool = False,
    graph: Graph = Depends(get_graph),
):
    return _query(corpus.subset, limit, graph, refresh=refresh)


class LayerNeighborsRequest(BaseModel):
    ids: list[str]
    layer: str
    per_node: int = 5


class RelatedRequest(BaseModel):
    ids: list[str]
    per_node: int = 4


@app.post("/api/graph/related")
def api_graph_related(request: RelatedRequest, graph: Graph = Depends(get_graph)):
    """The depth-1 neighborhood of the entities a chat turn surfaced -- the
    ask view draws it as the small "what the agent touched" map."""
    if not request.ids:
        raise HTTPException(status_code=422, detail="ids must not be empty")
    if len(request.ids) > 12:
        raise HTTPException(status_code=422, detail="at most 12 ids per call")
    if not 1 <= request.per_node <= 8:
        raise HTTPException(status_code=422, detail="per_node must be between 1 and 8")
    return _query(corpus.entity_neighborhood, request.ids, request.per_node, graph)


@app.post("/api/graph/layer_neighbors")
def api_layer_neighbors(request: LayerNeighborsRequest, graph: Graph = Depends(get_graph)):
    if request.layer not in ("epistemology", "provenance"):
        raise HTTPException(
            status_code=422,
            detail="layer_neighbors expands into epistemology or provenance",
        )
    if not 1 <= request.per_node <= 25:
        raise HTTPException(status_code=422, detail="per_node must be between 1 and 25")
    if not request.ids:
        raise HTTPException(status_code=422, detail="ids must not be empty")
    if len(request.ids) > 500:
        raise HTTPException(status_code=422, detail="at most 500 ids per call")
    return _query(corpus.layer_neighborhood, request.ids, request.layer, request.per_node, graph)


@app.get("/api/graph/full")
def api_graph_full(
    layers: str = Query("canonical", description="comma-separated layer names"),
    refresh: bool = False,
    graph: Graph = Depends(get_graph),
):
    requested = [layer.strip() for layer in layers.split(",") if layer.strip()]
    if not requested:
        raise HTTPException(status_code=422, detail="At least one layer is required")
    unknown = [layer for layer in requested if layer not in corpus.LAYERS]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown layers {unknown}; expected one of {sorted(corpus.LAYERS)}",
        )
    return _query(corpus.full_layer, requested, graph, refresh=refresh)


@app.post("/api/ask")
def api_ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty")
    try:
        return demo_server.chat({
            "messages": [{"role": "user", "content": question}],
            "session_id": f"web-{int(time.time())}",
        })
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Agent failed: {exc}") from exc


@app.post("/api/ask/stream")
def api_ask_stream(request: AskRequest):
    """The ask turn as server-sent events: progress dicts from the agent loop
    (thinking / tool_search / tool_start / tool_done) as they happen, then one
    `final` event whose `result` is byte-for-byte the /api/ask payload."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty")
    events: queue.Queue = queue.Queue()
    holder: dict = {}

    def run() -> None:
        try:
            holder["result"] = demo_server.chat(
                {"messages": [{"role": "user", "content": question}],
                 "session_id": f"web-{int(time.time())}"},
                on_event=events.put,
            )
        except Exception as exc:  # noqa: BLE001 -- surfaced as an SSE error event
            holder["error"] = exc
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()

    def generate():
        while True:
            event = events.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, default=str)}\n\n"
        if "error" in holder:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(holder['error'])})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'final', 'result': holder.get('result')}, default=str)}\n\n"

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/trace/{trace_id}")
def api_trace(trace_id: str, graph: Graph = Depends(get_graph)):
    return _query(corpus.trace_detail, trace_id, graph=graph)


class WorkerRequest(BaseModel):
    key: str
    name: str
    description: str = ""
    mission: str = ""
    capabilities: list[str] = []
    priority: int = 100
    active: bool = True
    kind: str = "custom"
    autonomy: str = "auto"
    system_prompt: str = ""
    entry_tools: list[str] = []
    allowed_tools: list[str] = []
    skills: list[str] = []
    model_hint: str | None = None
    routable: bool = True


class WorkerVersionRequest(BaseModel):
    changes: dict
    reason: str = ""


class ProposalActionRequest(BaseModel):
    confirm: bool = False
    reason: str = ""


class ConnectorInstanceRequest(BaseModel):
    name: str
    config: dict = {}
    schedule: str = ""


class OntologyReflectRequest(BaseModel):
    kind: str = "add_label"
    target: str = ""
    auto_approve: bool = False


def _worker_repo(graph: Graph):
    from knowledge_bases.orchestration.workers import WorkerRepository

    return WorkerRepository(graph)


def _reflection(graph: Graph):
    from knowledge_bases.orchestration.workers import ReflectionService

    return ReflectionService(graph)


@app.get("/api/workers")
def api_workers(include_inactive: bool = False, graph: Graph = Depends(get_graph)):
    return {
        "workers": [
            worker.model_dump(mode="json")
            for worker in _query(lambda: _worker_repo(graph).load_workers(include_inactive))
        ]
    }


@app.post("/api/workers")
def api_create_worker(request: WorkerRequest, graph: Graph = Depends(get_graph)):
    from knowledge_bases.orchestration.models import WorkerDefinition

    try:
        worker = _worker_repo(graph).upsert_worker(WorkerDefinition.model_validate(request.model_dump()), "created via API")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return worker.model_dump(mode="json")


@app.get("/api/workers/{key}")
def api_worker(key: str, graph: Graph = Depends(get_graph)):
    worker = _query(lambda: _worker_repo(graph).load_worker(key))
    if worker is None:
        raise HTTPException(status_code=404, detail="Worker not found")
    return worker.model_dump(mode="json")


@app.post("/api/workers/{key}/run")
def api_run_worker(key: str, request: AskRequest, graph: Graph = Depends(get_graph)):
    from knowledge_bases.orchestration.workers import WorkerRuntime

    try:
        return WorkerRuntime(graph).run(key, request.question.strip())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/workers/{key}/versions")
def api_worker_versions(key: str, graph: Graph = Depends(get_graph)):
    repo = _worker_repo(graph)
    if _query(lambda: repo.load_worker(key)) is None:
        raise HTTPException(status_code=404, detail="Worker not found")
    return {
        "versions": [
            version.model_dump(mode="json")
            for version in _query(lambda: repo.worker_versions(key))
        ]
    }


@app.post("/api/workers/{key}/versions")
def api_create_worker_version(key: str, request: WorkerVersionRequest, graph: Graph = Depends(get_graph)):
    try:
        version = _worker_repo(graph).create_draft_version(key, request.changes, request.reason)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return version.model_dump(mode="json")


@app.post("/api/workers/{key}/versions/{version}/activate")
def api_activate_worker_version(key: str, version: int, graph: Graph = Depends(get_graph)):
    try:
        active = _worker_repo(graph).activate_version(key, version, "explicit activation")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return active.model_dump(mode="json")


@app.post("/api/workers/{key}/reflect")
def api_reflect_worker(key: str, graph: Graph = Depends(get_graph)):
    try:
        result = _reflection(graph).reflect_and_apply(key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "proposal": result.get("proposal").model_dump(mode="json") if result.get("proposal") else None,
        "evaluation": result.get("evaluation").model_dump(mode="json") if result.get("evaluation") else None,
        "action": result.get("action", "reflected"),
        "run": result.get("run").model_dump(mode="json") if result.get("run") else None,
    }


@app.post("/api/workers/{key}/split")
def api_split_worker(key: str, graph: Graph = Depends(get_graph)):
    try:
        result = _reflection(graph).propose_worker_split(key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "children": result.get("children", []),
        "proposal": result.get("proposal").model_dump(mode="json") if result.get("proposal") else None,
    }


@app.get("/api/proposals")
def api_proposals(status: str | None = None, graph: Graph = Depends(get_graph)):
    return {
        "proposals": [
            proposal.model_dump(mode="json")
            for proposal in _query(lambda: _reflection(graph).proposals.list(status))
        ]
    }


@app.post("/api/proposals/{proposal_id}/evaluate")
def api_evaluate_proposal(proposal_id: str, graph: Graph = Depends(get_graph)):
    try:
        report = _reflection(graph).evaluate_proposal(proposal_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return report.model_dump(mode="json")


@app.post("/api/proposals/{proposal_id}/approve")
def api_approve_proposal(proposal_id: str, graph: Graph = Depends(get_graph)):
    try:
        proposal = _reflection(graph).approve_proposal(proposal_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return proposal.model_dump(mode="json")


@app.post("/api/proposals/{proposal_id}/reject")
def api_reject_proposal(proposal_id: str, request: ProposalActionRequest, graph: Graph = Depends(get_graph)):
    try:
        proposal = _reflection(graph).reject_proposal(proposal_id, request.reason)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return proposal.model_dump(mode="json")


@app.post("/api/proposals/{proposal_id}/apply")
def api_apply_proposal(proposal_id: str, request: ProposalActionRequest, graph: Graph = Depends(get_graph)):
    try:
        proposal = _reflection(graph).apply_proposal(proposal_id, confirm=request.confirm)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return proposal.model_dump(mode="json")


@app.post("/api/proposals/{proposal_id}/revert")
def api_revert_proposal(proposal_id: str, graph: Graph = Depends(get_graph)):
    try:
        proposal = _reflection(graph).revert_proposal(proposal_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return proposal.model_dump(mode="json")


@app.post("/api/proposals/{proposal_id}/check")
def api_check_proposal(proposal_id: str, graph: Graph = Depends(get_graph)):
    try:
        return _reflection(graph).post_apply_check(proposal_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/ontologies/proposals")
def api_ontology_proposals(status: str | None = None, graph: Graph = Depends(get_graph)):
    from knowledge_bases.orchestration.ontology_reflection import OntologyReflector

    return {
        "proposals": [
            proposal.model_dump(mode="json")
            for proposal in _query(lambda: OntologyReflector(graph).list(status))
        ]
    }


@app.post("/api/ontologies/{package}/reflect")
def api_reflect_ontology(package: str, request: OntologyReflectRequest, graph: Graph = Depends(get_graph)):
    from knowledge_bases.orchestration.ontology_reflection import OntologyReflector

    try:
        proposal = OntologyReflector(graph).reflect(
            package, kind=request.kind, target=request.target, auto_approve=request.auto_approve,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return proposal.model_dump(mode="json")


@app.post("/api/ontology-proposals/{proposal_id}/approve")
def api_approve_ontology_proposal(proposal_id: str, graph: Graph = Depends(get_graph)):
    from knowledge_bases.orchestration.ontology_reflection import OntologyReflector

    try:
        proposal = OntologyReflector(graph).approve(proposal_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return proposal.model_dump(mode="json")


@app.get("/api/connectors")
def api_connectors(graph: Graph = Depends(get_graph)):
    from knowledge_bases.orchestration.connectors import ConnectorRepository

    return {
        "connectors": [
            connector.model_dump(mode="json")
            for connector in _query(lambda: ConnectorRepository(graph).list_connectors())
        ],
        "instances": [
            instance.model_dump(mode="json")
            for instance in _query(lambda: ConnectorRepository(graph).list_instances())
        ],
    }


@app.post("/api/connectors/{connector_key}/instances")
def api_create_connector_instance(connector_key: str, request: ConnectorInstanceRequest, graph: Graph = Depends(get_graph)):
    from knowledge_bases.orchestration.connectors import ConnectorRepository

    try:
        instance = ConnectorRepository(graph).create_instance(
            connector_key, request.name, request.config, request.schedule,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return instance.model_dump(mode="json")


@app.get("/api/connectors/health")
def api_connector_health(graph: Graph = Depends(get_graph)):
    from knowledge_bases.orchestration.connectors import ConnectorRepository

    return {"health": _query(lambda: ConnectorRepository(graph).health())}



if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
