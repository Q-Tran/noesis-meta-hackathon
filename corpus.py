"""Read-only corpus queries powering the Noesis web UI.

Every function takes an optional ``graph`` so the web layer (app.py) and the
tests can inject a connection. Nothing here writes to the graph: this module is
the "productive view of the entire graph corpus" -- aggregate counts, keyword
search, node inspection, neighborhood expansion, and recent harness activity.
"""
from __future__ import annotations

import time
from typing import Any

from graph import Graph

# Property keys, in priority order, from which a human-readable node title is
# derived. Covers every label in the corpus (Claim.statement, Argument
# .surface_text, Chunk.text, CodeFile.path, everything else has name or id).
TITLE_KEYS = ("name", "statement", "surface_text", "text", "path", "question", "key", "id")

# Long free-text properties (Chunk.text, expert prompts) are trimmed to this
# many characters in API payloads so list views stay snappy.
_SNIPPET_LEN = 400

# The overview aggregate scans every node once; cache it briefly so repeated
# dashboard loads and sidebar refreshes do not rescan.
_overview_cache: dict[str, Any] = {"at": 0.0, "data": None}
_OVERVIEW_TTL_SECONDS = 60.0


def _title(props: dict) -> str:
    """Pick the most human-readable property as a node's title."""
    for key in TITLE_KEYS:
        value = props.get(key)
        if value:
            return str(value)
    return "(unnamed)"


def _trim_strings(props: dict, limit: int = _SNIPPET_LEN) -> dict:
    """Cap long string values, noting the original length for the UI."""
    trimmed = {}
    for key, value in props.items():
        if isinstance(value, str) and len(value) > limit:
            trimmed[key] = value[:limit] + "…"
            trimmed[f"_{key}_length"] = len(value)
        else:
            trimmed[key] = value
    return trimmed


def _node_row(row: dict, trim: bool = True) -> dict:
    props = row.get("props") or {}
    return {
        "element_id": row["element_id"],
        "labels": row.get("labels") or [],
        "title": _title(props),
        "properties": _trim_strings(props) if trim else props,
    }


def overview(graph: Graph | None = None, *, refresh: bool = False) -> dict:
    """Corpus totals: node/edge counts per label and relationship type."""
    now = time.monotonic()
    if not refresh and _overview_cache["data"] is not None and now - _overview_cache["at"] < _OVERVIEW_TTL_SECONDS:
        return _overview_cache["data"]

    graph = graph or Graph()
    label_rows = graph.run(
        "MATCH (n) UNWIND labels(n) AS label "
        "RETURN label, count(*) AS count ORDER BY count DESC"
    )
    rel_rows = graph.run(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
    )
    data = {
        "total_nodes": sum(row["count"] for row in label_rows),
        "total_relationships": sum(row["count"] for row in rel_rows),
        "labels": [{"label": row["label"], "count": row["count"]} for row in label_rows],
        "relationships": [{"type": row["type"], "count": row["count"]} for row in rel_rows],
    }
    _overview_cache["at"] = now
    _overview_cache["data"] = data
    return data


def growth(days: int = 30, graph: Graph | None = None) -> list[dict]:
    """Nodes created per day over the last `days` days (for the sparkline)."""
    graph = graph or Graph()
    rows = graph.run(
        "MATCH (n) WHERE n.created_at IS NOT NULL "
        "RETURN substring(toString(n.created_at), 0, 10) AS day, count(*) AS count "
        "ORDER BY day DESC LIMIT $days",
        days=days,
    )
    return [{"day": row["day"], "count": row["count"]} for row in reversed(rows)]


def search_nodes(
    query_text: str,
    label: str | None = None,
    limit: int = 25,
    graph: Graph | None = None,
) -> list[dict]:
    """Case-insensitive keyword search over every node's title properties.

    Exact and prefix matches rank above mere substring hits, then most recent
    first. `label` (one of the labels from `overview()`) scopes the search to
    one kind of node; an empty query with a label browses that kind, most
    recent first.
    """
    query_text = (query_text or "").strip()
    if not query_text and not label:
        return []
    graph = graph or Graph()
    if not query_text:
        # Browse mode: no text to match, so skip the property scan entirely and
        # list the kind, most recently created first.
        rows = graph.run(
            "MATCH (n) WHERE $label IN labels(n) "
            "RETURN elementId(n) AS element_id, labels(n) AS labels, "
            "       properties(n) AS props, 0 AS score "
            "ORDER BY n.created_at DESC LIMIT $limit",
            label=label,
            limit=limit,
        )
        return [{**_node_row(row), "score": 0} for row in rows]
    rows = graph.run(
        "MATCH (n) "
        "WHERE ($label IS NULL OR $label IN labels(n)) "
        "AND any(key IN $title_keys WHERE n[key] IS NOT NULL "
        "        AND toLower(toString(n[key])) CONTAINS $q) "
        "WITH n, CASE "
        "  WHEN any(key IN $title_keys WHERE n[key] IS NOT NULL "
        "            AND toLower(toString(n[key])) = $q) THEN 2 "
        "  WHEN any(key IN $title_keys WHERE n[key] IS NOT NULL "
        "            AND toLower(toString(n[key])) STARTS WITH $q) THEN 1 "
        "  ELSE 0 END AS score "
        "RETURN elementId(n) AS element_id, labels(n) AS labels, "
        "       properties(n) AS props, score "
        "ORDER BY score DESC, n.created_at DESC LIMIT $limit",
        label=label,
        title_keys=list(TITLE_KEYS),
        q=query_text.lower(),
        limit=limit,
    )
    results = []
    for row in rows:
        node = _node_row(row)
        node["score"] = row.get("score", 0)
        results.append(node)
    return results


def node_detail(element_id: str, graph: Graph | None = None) -> dict | None:
    """One node by element id, with its full (untrimmed) properties."""
    graph = graph or Graph()
    rows = graph.run(
        "MATCH (n) WHERE elementId(n) = $element_id "
        "RETURN elementId(n) AS element_id, labels(n) AS labels, properties(n) AS props",
        element_id=element_id,
    )
    if not rows:
        return None
    return _node_row(rows[0], trim=False)


def neighbors(element_id: str, limit: int = 40, graph: Graph | None = None) -> dict | None:
    """The neighborhood graph around one node, for lists and the canvas view.

    Returns `nodes` (center first, then neighbors) and `edges` with a
    `direction` relative to the center node. `None` if the center is gone.
    """
    graph = graph or Graph()
    center = node_detail(element_id, graph)
    if center is None:
        return None
    rows = graph.run(
        "MATCH (n)-[r]-(m) WHERE elementId(n) = $element_id "
        "RETURN elementId(m) AS element_id, labels(m) AS labels, properties(m) AS props, "
        "       type(r) AS rel_type, elementId(r) AS rel_id, (startNode(r) = n) AS outgoing "
        "ORDER BY rel_type LIMIT $limit",
        element_id=element_id,
        limit=limit,
    )
    center["properties"] = _trim_strings(center["properties"])
    nodes = [center]
    edges = []
    for row in rows:
        node = _node_row(row)
        nodes.append(node)
        edges.append({
            "element_id": row["rel_id"],
            "type": row["rel_type"],
            "source": element_id if row["outgoing"] else node["element_id"],
            "target": node["element_id"] if row["outgoing"] else element_id,
            "outgoing": row["outgoing"],
        })
    return {"nodes": nodes, "edges": edges}


# The corpus in three layers (the same taxonomy the demo player teaches).
# Canonical is what the product knows; epistemology is why it is justified in
# knowing it; provenance is where every piece of it came from. `Tool` is
# canonical: a callable capability of the product is knowledge, not evidence.
_GROUND_LABELS = ("SourceNode", "Chunk", "RoutingTrace", "Session", "Span", "TraceOutcome")

LAYERS = {
    "canonical": None,  # resolved dynamically: every label that is not in the other two
    "epistemology": ("Claim", "Argument"),
    "provenance": _GROUND_LABELS,
}


def layer_labels(layer: str, graph: Graph | None = None) -> list[str]:
    """The concrete node labels of a layer. `canonical` is computed as the
    complement, so a newly installed ontology lands in it automatically."""
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}; expected one of {sorted(LAYERS)}")
    static = LAYERS[layer]
    if static is not None:
        return sorted(static)
    graph = graph or Graph()
    rows = graph.run("MATCH (n) UNWIND labels(n) AS label RETURN DISTINCT label")
    return sorted(_layer_label_set(layer, {row["label"] for row in rows}))


def _layer_label_set(layer: str, db_labels: set[str]) -> set[str]:
    """A layer's labels, resolved against labels already read from the graph
    (what `full_layer` uses to ask once instead of once per layer)."""
    static = LAYERS[layer]
    if static is not None:
        return set(static)
    other = set(LAYERS["epistemology"]) | set(LAYERS["provenance"])
    return db_labels - other


# The full layer is expensive on small hosts (a few seconds per query) and
# changes only when the corpus does; a 5-minute cache keeps every load after
# the first one instant. The hint in the UI says when it refreshes.
_FULL_TTL_SECONDS = 300.0
_full_cache: dict[tuple[str, ...], dict[str, Any]] = {}


def full_layer(layers: list[str], graph: Graph | None = None, *, refresh: bool = False) -> dict:
    """Every node and edge of the given layers, for the explorer's "all" view.

    Properties stay on the server: each node is reduced in Cypher to its id,
    labels and a 200-char title (detail loads on demand via `node_detail`).
    Returning full properties costs several seconds and megabytes on a small
    host; the title projection is what keeps this view viable at 35k nodes.
    """
    graph = graph or Graph()
    key = tuple(sorted(layers))
    now = time.monotonic()
    cached = _full_cache.get(key)
    if not refresh and cached and now - cached["at"] < _FULL_TTL_SECONDS:
        return cached["data"]

    rows = graph.run("MATCH (n) UNWIND labels(n) AS label RETURN DISTINCT label")
    db_labels = {row["label"] for row in rows}
    labels = sorted({label for layer in layers for label in _layer_label_set(layer, db_labels)})
    if not labels:
        return {"labels_dict": [], "rel_types": [], "ids": [],
                "nodes": [], "edges": [], "layers": layers}

    node_rows = graph.run(
        "MATCH (n) WHERE any(label IN labels(n) WHERE label IN $labels) "
        "WITH n, labels(n) AS node_labels, "
        "     [k IN $title_keys WHERE n[k] IS NOT NULL][0] AS title_key "
        "RETURN elementId(n) AS element_id, node_labels AS labels, "
        "       substring(toString(n[title_key]), 0, 200) AS title",
        labels=labels,
        title_keys=list(TITLE_KEYS),
    )
    if set(labels) >= db_labels:
        # All three layers: nothing to filter out, so skip the per-edge label
        # checks (several seconds on a small host) and scan the edges plain.
        edge_rows = graph.run(
            "MATCH ()-[r]->() RETURN elementId(startNode(r)) AS source, "
            "       elementId(endNode(r)) AS target, type(r) AS type"
        )
    else:
        edge_rows = graph.run(
            "MATCH (a)-[r]->(b) "
            "WHERE any(label IN labels(a) WHERE label IN $labels) "
            "  AND any(label IN labels(b) WHERE label IN $labels) "
            "RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS type",
            labels=labels,
        )

    # Indexed wire form: element ids and type/label names are sent once in a
    # lookup table and rows refer to them by position. At 35k nodes / 63k
    # edges this is a third of the naive JSON, which on a small host is the
    # difference between a stall and a load.
    ids = [row["element_id"] for row in node_rows]
    id_pos = {eid: i for i, eid in enumerate(ids)}
    labels_dict = sorted({label for row in node_rows for label in row["labels"] or []})
    label_pos = {label: i for i, label in enumerate(labels_dict)}
    nodes = [
        [label_pos[row["labels"][0]] if row["labels"] else -1, row["title"] or "(unnamed)"]
        for row in node_rows
    ]
    rel_types = sorted({row["type"] for row in edge_rows})
    type_pos = {rel_type: i for i, rel_type in enumerate(rel_types)}
    edges = [
        [type_pos[row["type"]], id_pos[row["source"]], id_pos[row["target"]]]
        for row in edge_rows
        if row["source"] in id_pos and row["target"] in id_pos
    ]
    data = {
        "labels_dict": labels_dict,
        "rel_types": rel_types,
        "ids": ids,
        "nodes": nodes,
        "edges": edges,
        "layers": layers,
    }
    _full_cache[key] = {"at": now, "data": data}
    return data


# The development ontology's codebase indexing (the engine's own source) is
# real knowledge, but it is not part of the product's canonical map: the home
# subset and its demo exclude these labels.
CODEBASE_LABELS = ("CodeFunction", "CodeMethod", "CodeClass", "CodeFile", "CodeRoute", "CodeRepository")


# How many of the corpus's most-connected tools the home subset carries.
TOOL_SUBSET_LIMIT = 10

# The tool catalog's own component (tools and the experts/skills that own
# them). Canonical in the layer taxonomy, but excluded from the subset's
# degree-ranked pool: the catalog is appended explicitly, so its high-degree
# nodes must not displace the domain map.
TOOL_CLUSTER_LABELS = ("Tool", "KnowledgeBaseExpert", "AgentSkill",
                       "ExpertCapability", "ExpertRevision")


def subset(limit: int = 100, graph: Graph | None = None, *, refresh: bool = False) -> dict:
    """A representative home map: ~`limit` canonical nodes, ONE connected
    component, no codebase nodes.

    Selection is a connectivity-first degree walk: start from the
    highest-degree canonical node, then repeatedly add the highest-degree
    node ADJACENT to the selection. Raw degree ranking leaves silos in the
    map (a node whose edges all land outside the chosen set); the walk makes
    silos impossible by construction while keeping degree as the importance
    signal. Cached like `full_layer`; same indexed wire form, plus degree.
    """
    graph = graph or Graph()
    key = ("__subset__", limit)
    now = time.monotonic()
    cached = _full_cache.get(key)
    if not refresh and cached and now - cached["at"] < _FULL_TTL_SECONDS:
        return cached["data"]

    rows = graph.run("MATCH (n) UNWIND labels(n) AS label RETURN DISTINCT label")
    db_labels = {row["label"] for row in rows}
    canonical = sorted(_layer_label_set("canonical", db_labels)
                       - set(CODEBASE_LABELS) - set(TOOL_CLUSTER_LABELS))
    if not canonical:
        data = {"labels_dict": [], "rel_types": [], "ids": [],
                "nodes": [], "edges": [], "layers": ["canonical"]}
        _full_cache[key] = {"at": now, "data": data}
        return data

    # The candidate pool (limit×5, so the walk has room to choose), ranked by
    # degree. COUNT {} uses the degree store, so ranking does not scan edges.
    node_rows = graph.run(
        "MATCH (n) WHERE any(label IN labels(n) WHERE label IN $labels) "
        "WITH n, labels(n) AS node_labels, COUNT { (n)-[]-() } AS degree, "
        "     [k IN $title_keys WHERE n[k] IS NOT NULL][0] AS title_key "
        "ORDER BY degree DESC LIMIT $limit "
        "RETURN elementId(n) AS element_id, node_labels AS labels, degree, "
        "       substring(toString(n[title_key]), 0, 200) AS title",
        labels=canonical,
        title_keys=list(TITLE_KEYS),
        limit=limit * 5,
    )
    pool_ids = [row["element_id"] for row in node_rows]
    if not pool_ids:
        data = {"labels_dict": [], "rel_types": [], "ids": [],
                "nodes": [], "edges": [], "layers": ["canonical"]}
        _full_cache[key] = {"at": now, "data": data}
        return data
    pool_pos = {eid: i for i, eid in enumerate(pool_ids)}

    # Pool-internal edges, deduped on elementId(r) (the undirected match
    # materializes each edge twice, once from each end).
    edge_rows = graph.run(
        "UNWIND $ids AS id MATCH (a)-[r]-(b) WHERE elementId(a) = id "
        "RETURN elementId(r) AS rel_id, elementId(a) AS source, "
        "       elementId(b) AS target, type(r) AS type",
        ids=pool_ids,
    )
    adj: dict[str, list[str]] = {}
    pool_edges = {}
    for row in edge_rows:
        if row["target"] not in pool_pos or row["rel_id"] in pool_edges:
            continue
        pool_edges[row["rel_id"]] = row
        adj.setdefault(row["source"], []).append(row["target"])
        adj.setdefault(row["target"], []).append(row["source"])

    # The walk: a degree-priority frontier over the pool, grown from the top.
    import heapq

    degree = {row["element_id"]: row["degree"] for row in node_rows}
    selected: list[str] = []
    in_selection: set[str] = set()
    frontier = [(-degree[pool_ids[0]], pool_ids[0])]
    heapq.heapify(frontier)
    in_frontier = {pool_ids[0]}
    while frontier and len(selected) < limit:
        _, nid = heapq.heappop(frontier)
        if nid in in_selection:
            continue
        selected.append(nid)
        in_selection.add(nid)
        for neighbor in adj.get(nid, []):
            if neighbor not in in_selection and neighbor not in in_frontier:
                heapq.heappush(frontier, (-degree[neighbor], neighbor))
                in_frontier.add(neighbor)
    # If the component exhausted before `limit`, the corpus genuinely has
    # fewer related nodes; do not pad the map with silos.

    ids = list(selected)
    kept_rows = {row["element_id"]: row for row in node_rows}

    # The tool catalog: the most-connected active tools plus the experts and
    # skills that connect them, so the tools join as a related island rather
    # than ten isolated dots. Tools belong to their own corpus component (the
    # product's callable surface), which is why ranking never pulls them in.
    tool_rows = graph.run(
        "MATCH (t:Tool) WHERE t.active = true "
        "WITH t, COUNT { (t)-[]-() } AS degree, "
        "     [k IN $title_keys WHERE t[k] IS NOT NULL][0] AS title_key "
        "ORDER BY degree DESC LIMIT $n "
        "RETURN elementId(t) AS element_id, labels(t) AS labels, "
        "       substring(toString(t[title_key]), 0, 200) AS title, degree",
        title_keys=list(TITLE_KEYS),
        n=TOOL_SUBSET_LIMIT,
    )
    bridge_rows = []
    if tool_rows:
        bridge_rows = graph.run(
            "MATCH (t:Tool)-[r]-(n) WHERE elementId(t) IN $tool_ids "
            "AND NOT any(label IN labels(n) WHERE label IN $skip) "
            "WITH n, COUNT { (n)-[]-() } AS degree, "
            "     [k IN $title_keys WHERE n[k] IS NOT NULL][0] AS title_key "
            "RETURN DISTINCT elementId(n) AS element_id, labels(n) AS labels, "
            "       substring(toString(n[title_key]), 0, 200) AS title, degree",
            tool_ids=[row["element_id"] for row in tool_rows],
            skip=sorted(set(LAYERS["provenance"])),
            title_keys=list(TITLE_KEYS),
        )
    for row in tool_rows + bridge_rows:
        if row["element_id"] not in kept_rows:
            kept_rows[row["element_id"]] = row
            ids.append(row["element_id"])

    id_pos = {eid: i for i, eid in enumerate(ids)}

    # Final edge set: everything touching the selected nodes, both ends kept.
    # (The pool expansion above only served the connectivity walk.)
    final_edges = graph.run(
        "UNWIND $ids AS id MATCH (a)-[r]-(b) WHERE elementId(a) = id "
        "RETURN elementId(r) AS rel_id, elementId(a) AS source, "
        "       elementId(b) AS target, type(r) AS type",
        ids=ids,
    )
    rel_types = sorted({row["type"] for row in final_edges if row["target"] in id_pos})
    type_pos = {rel_type: i for i, rel_type in enumerate(rel_types)}
    deduped = {}
    for row in final_edges:
        if row["target"] not in id_pos or row["rel_id"] in deduped:
            continue
        deduped[row["rel_id"]] = [type_pos[row["type"]], id_pos[row["source"]], id_pos[row["target"]]]

    labels_dict = sorted({label for eid, row in kept_rows.items() if eid in id_pos
                          for label in row["labels"] or []})
    label_pos = {label: i for i, label in enumerate(labels_dict)}
    nodes = [
        [
            label_pos[kept_rows[eid]["labels"][0]] if kept_rows[eid]["labels"] else -1,
            kept_rows[eid]["title"] or "(unnamed)",
            kept_rows[eid]["degree"],
        ]
        for eid in ids
    ]
    edges = list(deduped.values())
    data = {
        "labels_dict": labels_dict,
        "rel_types": rel_types,
        "ids": ids,
        "nodes": nodes,
        "edges": edges,
        "layers": ["canonical"],
    }
    _full_cache[key] = {"at": now, "data": data}
    return data


# The showcase scenario: the "Tool search" capability and the argument behind
# its stale claim ("Tool search is not available on Chat Completions"). Both
# web servers (the explorer app and the demo chat) read the same affected
# subgraph from here.
SHOWCASE_CAPABILITY_ID = "extent-591dfe996e"
SHOWCASE_STALE_ARGUMENT_ID = "extarg-8dd04b0c8e-fc846d6c"


def showcase_subgraph(
    capability_id: str = SHOWCASE_CAPABILITY_ID,
    stale_argument_id: str = SHOWCASE_STALE_ARGUMENT_ID,
    graph: Graph | None = None,
) -> dict:
    """The scenario's affected subgraph, narrowed to the story's core: the
    capability, the STALE argument with its premise claims and their chunks
    (and source), attack arguments on it (and their premises), and chat-created
    arguments concluding the capability with the update. Snapshot it before
    and after a correction; the page diffs the two. Read-only.
    """
    graph = graph or Graph()
    rows = graph.run(
        """
        MATCH (cap:Capability {id: $cap_id})
        OPTIONAL MATCH (stale:Argument {id: $stale_arg})
        OPTIONAL MATCH (stale)-[ce:CONCLUDES]->(cap)
        OPTIONAL MATCH (stale)-[up:USES_PREMISE]->(c:Claim)
        OPTIONAL MATCH (c)-[fi:FOUND_IN]->(ev)
        WITH cap, stale, ce, up, c, fi, ev
        WHERE ev IS NULL OR ev:Chunk OR ev:RoutingTrace
        OPTIONAL MATCH (ev:Chunk)-[fs:FOUND_IN]->(src:SourceNode)
        OPTIONAL MATCH (att:Argument)-[at:ATTACKS]->(stale)
        OPTIONAL MATCH (att)-[up2:USES_PREMISE]->(c2:Claim)
        OPTIONAL MATCH (c2)-[fi2:FOUND_IN]->(t2:RoutingTrace)
        OPTIONAL MATCH (new:Argument)-[ce2:CONCLUDES]->(cap)
        WHERE new.id STARTS WITH 'chatarg-'
        OPTIONAL MATCH (new)-[up3:USES_PREMISE]->(c3:Claim)
        OPTIONAL MATCH (c3)-[fi3:FOUND_IN]->(ev3)
        WITH cap, stale, ce, up, c, fi, ev, fs, src, att, at, up2, c2, fi2, t2, new, ce2, up3, c3, fi3, ev3
        WHERE ev3 IS NULL OR ev3:Chunk OR ev3:RoutingTrace
        RETURN
          elementId(cap) AS cap_eid, labels(cap) AS cap_labels, properties(cap) AS cap_props,
          elementId(stale) AS arg_eid, labels(stale) AS arg_labels, properties(stale) AS arg_props,
          elementId(ce) AS ce_eid, type(ce) AS ce_type,
          elementId(up) AS up_eid, type(up) AS up_type,
          elementId(c) AS c_eid, labels(c) AS c_labels, properties(c) AS c_props,
          elementId(fi) AS fi_eid, type(fi) AS fi_type,
          elementId(ev) AS ev_eid, labels(ev) AS ev_labels, properties(ev) AS ev_props,
          elementId(fs) AS fs_eid, type(fs) AS fs_type,
          elementId(src) AS src_eid, labels(src) AS src_labels, properties(src) AS src_props,
          elementId(att) AS att_eid, labels(att) AS att_labels, properties(att) AS att_props,
          elementId(at) AS at_eid, type(at) AS at_type,
          elementId(up2) AS up2_eid, type(up2) AS up2_type,
          elementId(c2) AS c2_eid, labels(c2) AS c2_labels, properties(c2) AS c2_props,
          elementId(fi2) AS fi2_eid, type(fi2) AS fi2_type,
          elementId(t2) AS t2_eid, labels(t2) AS t2_labels, properties(t2) AS t2_props,
          elementId(new) AS new_eid, labels(new) AS new_labels, properties(new) AS new_props,
          elementId(ce2) AS ce2_eid, type(ce2) AS ce2_type,
          elementId(up3) AS up3_eid, type(up3) AS up3_type,
          elementId(c3) AS c3_eid, labels(c3) AS c3_labels, properties(c3) AS c3_props,
          elementId(fi3) AS fi3_eid, type(fi3) AS fi3_type,
          elementId(ev3) AS ev3_eid, labels(ev3) AS ev3_labels, properties(ev3) AS ev3_props
        """,
        cap_id=capability_id,
        stale_arg=stale_argument_id,
    )

    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}

    def add_node(eid, labels, props):
        if eid is None:
            return
        props = props or {}
        title = next((str(props[k]) for k in
                      ("name", "statement", "surface_text", "text", "id") if props.get(k)),
                     "(unnamed)")
        if eid not in nodes:
            nodes[eid] = {
                "element_id": eid,
                "labels": labels or [],
                "title": title[:160],
                "status": props.get("status"),
            }

    def add_edge(eid, rel_type, source, target):
        if eid is None or source is None or target is None:
            return
        edges[eid] = {
            "element_id": eid, "type": rel_type,
            "source": source, "target": target,
        }

    for row in rows:
        add_node(row["cap_eid"], row["cap_labels"], row["cap_props"])
        add_node(row["arg_eid"], row["arg_labels"], row["arg_props"])
        add_node(row["c_eid"], row["c_labels"], row["c_props"])
        add_node(row["ev_eid"], row["ev_labels"], row["ev_props"])
        add_node(row["src_eid"], row["src_labels"], row["src_props"])
        add_node(row["att_eid"], row["att_labels"], row["att_props"])
        add_node(row["c2_eid"], row["c2_labels"], row["c2_props"])
        add_node(row["t2_eid"], row["t2_labels"], row["t2_props"])
        add_node(row["new_eid"], row["new_labels"], row["new_props"])
        add_node(row["c3_eid"], row["c3_labels"], row["c3_props"])
        add_node(row["ev3_eid"], row["ev3_labels"], row["ev3_props"])
        add_edge(row["ce_eid"], row["ce_type"], row["arg_eid"], row["cap_eid"])
        add_edge(row["up_eid"], row["up_type"], row["arg_eid"], row["c_eid"])
        add_edge(row["fi_eid"], row["fi_type"], row["c_eid"], row["ev_eid"])
        add_edge(row["fs_eid"], row["fs_type"], row["ev_eid"], row["src_eid"])
        add_edge(row["at_eid"], row["at_type"], row["att_eid"], row["arg_eid"])
        add_edge(row["up2_eid"], row["up2_type"], row["att_eid"], row["c2_eid"])
        add_edge(row["fi2_eid"], row["fi2_type"], row["c2_eid"], row["t2_eid"])
        add_edge(row["ce2_eid"], row["ce2_type"], row["new_eid"], row["cap_eid"])
        add_edge(row["up3_eid"], row["up3_type"], row["new_eid"], row["c3_eid"])
        add_edge(row["fi3_eid"], row["fi3_type"], row["c3_eid"], row["ev3_eid"])

    return {
        "capability_id": capability_id,
        "stale_argument_id": stale_argument_id,
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
    }


def trace_detail(trace_id: str, graph: Graph | None = None, store=None) -> dict:
    """The recorded spans of one conversation turn, oldest first."""
    from knowledge_bases.tools.trace_store import TraceStore

    store = store or TraceStore(graph or Graph())
    spans = store.get_trace_spans(trace_id)
    return {"trace_id": trace_id, "spans": [
        {key: s.get(key) for key in ("span_id", "name", "kind", "status", "error",
                                      "duration_ms", "result_count", "session_id",
                                      "client", "raw_input", "raw_output")}
        for s in spans
    ]}


def layer_neighborhood(
    node_ids: list[str],
    layer: str,
    per_node: int = 5,
    graph: Graph | None = None,
) -> dict:
    """A bounded expansion of the given nodes into one layer: for each of them,
    its top `per_node` neighbors (by node weight) that carry a layer label.

    This is what the explorer's layer pills load when toggled on: the 100-node
    canonical map gains a readable ring of its justification (claims and
    arguments) or evidence (chunks, sources, traces) instead of every
    connected node, which would be a hairball at hub degrees. Same return
    shape as `neighbors()`, minus the center (the caller already has it).
    """
    if layer not in ("epistemology", "provenance"):
        raise ValueError(
            f"layer_neighborhood expands into epistemology or provenance, not {layer!r}"
        )
    graph = graph or Graph()
    rows = graph.run(
        "UNWIND $ids AS id MATCH (c)-[r]-(n) "
        "WHERE elementId(c) = id AND any(l IN labels(n) WHERE l IN $labels) "
        "WITH c, r, n ORDER BY coalesce(n.weight, 0.0) DESC "
        "WITH c, elementId(c) AS cid, collect({rel: r, node: n})[0..$per_node] AS top "
        "UNWIND top AS entry "
        "RETURN cid, elementId(entry.node) AS element_id, labels(entry.node) AS labels, "
        "       properties(entry.node) AS props, type(entry.rel) AS rel_type, "
        "       elementId(entry.rel) AS rel_id, (startNode(entry.rel) = c) AS outgoing",
        ids=node_ids,
        labels=sorted(LAYERS[layer]),
        per_node=per_node,
    )
    nodes = []
    edges = []
    for row in rows:
        node = _node_row(row)
        nodes.append(node)
        edges.append({
            "element_id": row["rel_id"],
            "type": row["rel_type"],
            "source": row["cid"] if row["outgoing"] else node["element_id"],
            "target": node["element_id"] if row["outgoing"] else row["cid"],
            "outgoing": row["outgoing"],
        })
    return {"nodes": nodes, "edges": edges, "layer": layer}


def entity_neighborhood(ids: list[str], per_node: int = 4, graph: Graph | None = None) -> dict:
    """The depth-1 neighborhood of the entities a chat turn surfaced, keyed by
    node `id` (not element id): the centers plus each center's top `per_node`
    neighbors by weight, any label -- the small "what the agent touched" map
    the ask view draws under an answer.
    """
    if not ids:
        raise ValueError("entity_neighborhood needs at least one entity id")
    per_node = max(1, min(int(per_node), 8))
    graph = graph or Graph()
    centers = graph.run(
        "MATCH (n) WHERE n.id IN $ids "
        "RETURN elementId(n) AS element_id, labels(n) AS labels, properties(n) AS props",
        ids=ids,
    )
    center_eids = [row["element_id"] for row in centers]
    nodes = [_node_row(row) for row in centers]
    edges: list[dict] = []
    if center_eids:
        rows = graph.run(
            "UNWIND $eids AS eid MATCH (c)-[r]-(n) "
            "WHERE elementId(c) = eid "
            "WITH c, r, n ORDER BY coalesce(n.weight, 0.0) DESC "
            "WITH c, elementId(c) AS cid, collect({rel: r, node: n})[0..$per_node] AS top "
            "UNWIND top AS entry "
            "RETURN cid, elementId(entry.node) AS element_id, labels(entry.node) AS labels, "
            "       properties(entry.node) AS props, type(entry.rel) AS rel_type, "
            "       elementId(entry.rel) AS rel_id, (startNode(entry.rel) = c) AS outgoing",
            eids=center_eids,
            per_node=per_node,
        )
        seen = {n["element_id"] for n in nodes}
        for row in rows:
            node = _node_row(row)
            if node["element_id"] not in seen:
                seen.add(node["element_id"])
                nodes.append(node)
            edges.append({
                "element_id": row["rel_id"],
                "type": row["rel_type"],
                "source": row["cid"] if row["outgoing"] else node["element_id"],
                "target": node["element_id"] if row["outgoing"] else row["cid"],
                "outgoing": row["outgoing"],
            })
    return {"nodes": nodes, "edges": edges, "centers": center_eids}


def graph_seed(limit: int = 25, graph: Graph | None = None) -> dict | None:
    """A starter subgraph for the explorer: the most-connected node and its
    neighborhood, with the hub's degree attached.

    The whole corpus will never fit readably on one canvas, so the explorer
    starts here and grows by user-driven expansion instead.
    """
    graph = graph or Graph()
    hubs = graph.run(
        "MATCH (n)-[r]-() WITH n, count(r) AS degree ORDER BY degree DESC LIMIT 1 "
        "RETURN elementId(n) AS element_id, labels(n) AS labels, properties(n) AS props, degree"
    )
    if not hubs:
        return None
    hood = neighbors(hubs[0]["element_id"], limit=limit, graph=graph)
    if hood is None:
        return None
    hood["nodes"][0]["degree"] = hubs[0]["degree"]
    return hood


def recent_activity(limit: int = 12, graph: Graph | None = None) -> list[dict]:
    """Most recent harness sessions and routing traces, merged and sorted.

    This is the "what has the system been doing" feed for the dashboard.
    """
    graph = graph or Graph()
    rows = graph.run(
        "MATCH (s:Session) "
        "RETURN 'Session' AS kind, s.id AS id, s.started_at AS started, "
        "       s.last_active_at AS active, null AS status, null AS error "
        "UNION ALL "
        "MATCH (t:RoutingTrace) "
        "RETURN 'RoutingTrace' AS kind, t.id AS id, t.started_at AS started, "
        "       t.last_active_at AS active, t.status AS status, t.error AS error "
        "ORDER BY active DESC LIMIT $limit",
        limit=limit,
    )
    return [
        {
            "kind": row["kind"],
            "id": row["id"],
            "started_at": row.get("started"),
            "last_active_at": row.get("active"),
            "status": row.get("status"),
            "error": row.get("error"),
        }
        for row in rows
    ]
