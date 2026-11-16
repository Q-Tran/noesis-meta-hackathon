"""Extract claims, arguments and canonical entities from the full docs corpus.

Runs Muse (Meta Model API) over every `metadocs-chunk-*` chunk, validates the
JSON against the ontology, and writes into the graph under the extraction
namespace (`extclaim-*`, `extarg-*`, `extent-*`):

    (Claim)-[:FOUND_IN]->(Chunk)-[:FOUND_IN]->(SourceNode)
    (Argument)-[:USES_PREMISE]->(Claim)
    (Argument)-[:CONCLUDES]->(canonical entity)
    (Chunk)-[:MENTIONS]->(canonical entity)

Stable ids make reruns idempotent. A JSONL ledger lets `--resume` skip chunks
already extracted. The API key is read from META_API_KEY, or from the local
OpenCode auth store (`meta` provider) when that env var is absent; it is never
printed.

    set -a && source .env && set +a
    python3 demo/graph_delta/extract_meta_claims.py --limit 5 --workers 2
    python3 demo/graph_delta/extract_meta_claims.py --workers 10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from graph import Graph

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "sources" / "claims-ledger.jsonl"
SUMMARY = HERE / "sources" / "extraction-summary.json"
API_URL = "https://api.meta.ai/v1/chat/completions"
MODEL = "muse-spark-1.3"
ALLOWED_LABELS = {"Component", "Interface", "Capability", "Specification",
                  "BusinessTerm", "Metric", "Table", "Column", "Value"}
ALLOWED_RELS = {"EXPOSES", "IMPLEMENTS", "DEPENDS_ON", "CONSUMES", "REALIZES"}
CLAIM_TYPES = {"fact", "finding", "assumption", "rationale", "question"}

SYSTEM_RETRY = """You extract claims from Meta Model API documentation chunks that may be prose,
schema definitions, configuration, or code examples.
Return ONLY valid JSON, no markdown:
{"claims":[{"statement":"standalone assertion","claim_type":"fact|finding|assumption|rationale|question","confidence":0.0}],
 "entities":[{"name":"canonical API concept","label":"Component|Interface|Capability|Specification|BusinessTerm|Metric|Table","description":"short description","claims":[0]}],
 "relationships":[{"from":"entity name","from_label":"Component|Interface|Capability|Specification|BusinessTerm|Metric|Table","type":"EXPOSES|IMPLEMENTS|DEPENDS_ON|CONSUMES|REALIZES","to":"entity name","to_label":"Component|Interface|Capability|Specification|BusinessTerm|Metric|Table","claims":[0]}]}
Rules:
- Only assertions supported by the chunk; never invent facts, fields, endpoints or numbers.
- Prose/configuration: extract the assertions it makes.
- Schema/table: state fields, types, defaults and constraints as claims; a Specification entity is appropriate.
- Code example: state what the example does (calls, parameters, outputs); do NOT invent repository structure and do NOT emit CodeFile/CodeFunction/CodeClass/CodeMethod -- those require a real repository scan.
- 1-6 claims; 0-4 entities; 0-3 relationships; claims lists are indexes into the claims array.
- If the chunk truly contains no assertion, return {"claims":[],"entities":[],"relationships":[]}."""

SYSTEM = """You extract a knowledge graph from Meta Model API documentation.
Return ONLY valid JSON, no markdown, with this exact shape:
{"claims":[{"statement":"standalone assertion","claim_type":"fact|finding|assumption|rationale|question","confidence":0.0}],
 "entities":[{"name":"canonical API concept","label":"Component|Interface|Capability|Specification|BusinessTerm|Metric|Table","description":"short description","claims":[0]}],
 "relationships":[{"from":"entity name","from_label":"Component|Interface|Capability|Specification|BusinessTerm|Metric|Table","type":"EXPOSES|IMPLEMENTS|DEPENDS_ON|CONSUMES|REALIZES","to":"entity name","to_label":"Component|Interface|Capability|Specification|BusinessTerm|Metric|Table","claims":[0]}]}
Rules:
- Only assertions supported by the chunk; never invent facts, names, endpoints or numbers.
- Claims are standalone and specific; 3-8 per chunk if the chunk has that much substance, otherwise fewer or none.
- Entities are canonical API concepts the chunk is about (models, APIs, capabilities, specifications); 0-5.
- Relationships only when the chunk states the link; 0-5. claims lists are indexes into the claims array.
- Use exact product names (Muse Wonder, Muse Wonder Full, Muse Spark 1.3, Responses API, ...).
- If the chunk contains no extractable knowledge, return {"claims":[],"entities":[],"relationships":[]}."""


def api_key() -> str:
    key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
    if key:
        return key
    auth = pathlib.Path.home() / ".local/share/opencode/auth.json"
    if auth.exists():
        try:
            return json.loads(auth.read_text())["meta"]["key"]
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"cannot read the meta credential: {exc}")
    raise SystemExit("set META_API_KEY or configure the meta provider in OpenCode")


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def digest(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def clean(text: str, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit]


def parse_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start:end + 1])


def call_model(key: str, prompt: str, system: str = SYSTEM) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1400,
        "temperature": 0,
        "reasoning_effort": "minimal",
    }).encode()
    request = urllib.request.Request(
        API_URL, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read())
    return payload


def call_with_retry(key: str, prompt: str, system: str = SYSTEM, attempts: int = 4) -> dict:
    last = None
    for attempt in range(attempts):
        try:
            payload = call_model(key, prompt, system)
            message = payload["choices"][0]["message"].get("content") or ""
            return parse_json(message)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep((2 ** attempt) + random.random())
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1 + random.random())
    raise last if last else RuntimeError("no response")


def known_entities(graph: Graph) -> dict[str, tuple[str, str, str]]:
    """normalized name -> (label, id, display name) for existing canonical nodes."""
    rows = graph.run(
        "MATCH (n) WHERE (n:Component OR n:Interface OR n:Capability "
        "OR n:Specification OR n:BusinessTerm OR n:Metric OR n:Table) "
        "AND n.name IS NOT NULL RETURN labels(n)[0] AS label, n.id AS id, n.name AS name"
    )
    out = {}
    for row in rows:
        if row["label"] in ALLOWED_LABELS and row["id"]:
            out.setdefault(norm(row["name"]), (row["label"], row["id"], row["name"]))
    return out


_tls = threading.local()


def thread_graph() -> Graph:
    graph = getattr(_tls, "graph", None)
    if graph is None:
        graph = Graph()
        _tls.graph = graph
    return graph


def process_chunk(key: str, chunk: dict, known: dict, dry_run: bool,
                  system: str = SYSTEM) -> dict:
    graph = None if dry_run else thread_graph()
    prompt = (
        f"PAGE: {chunk['source']}\nURL: {chunk['url']}\n\n"
        f"CHUNK:\n{chunk['text']}"
    )
    data = call_with_retry(key, prompt, system)
    claims = []
    for item in (data.get("claims") or [])[:8]:
        statement = clean(item.get("statement"))
        if len(statement) < 12:
            continue
        claims.append({
            "statement": statement,
            "claim_type": item.get("claim_type") if item.get("claim_type") in CLAIM_TYPES else "fact",
            "confidence": max(0.0, min(1.0, float(item.get("confidence", 0.85)))),
        })

    entities = []
    seen_entities = set()
    seen_names = set()
    for item in (data.get("entities") or [])[:5]:
        name = clean(item.get("name"), 160)
        label = item.get("label")
        if not name or label not in ALLOWED_LABELS:
            continue
        name_key = norm(name)
        if name_key in seen_names:
            continue
        seen_names.add(name_key)
        hit = known.get(name_key)
        if hit:
            label, entity_id, name = hit
        else:
            entity_id = f"extent-{digest(label + ':' + norm(name))}"
        if (label, entity_id) in seen_entities:
            continue
        seen_entities.add((label, entity_id))
        entities.append({
            "label": label,
            "id": entity_id,
            "name": name,
            "description": clean(item.get("description"), 300),
            "claims": [i for i in (item.get("claims") or []) if isinstance(i, int) and 0 <= i < len(claims)],
        })

    by_name = {norm(e["name"]): e for e in entities}
    relationships = []
    for item in (data.get("relationships") or [])[:5]:
        rel = item.get("type")
        if rel not in ALLOWED_RELS:
            continue
        source = by_name.get(norm(item.get("from") or ""))
        target = by_name.get(norm(item.get("to") or ""))
        if not source or not target or source["id"] == target["id"]:
            continue
        relationships.append({
            "from": source, "to": target, "type": rel,
            "claims": [i for i in (item.get("claims") or []) if isinstance(i, int) and 0 <= i < len(claims)],
        })

    chunk_hash = digest(chunk["id"])
    claim_ids = []
    if graph is not None:
        for index, claim in enumerate(claims):
            claim_id = f"extclaim-{chunk_hash}-{index}"
            claim_ids.append(claim_id)
            graph.upsert_node("Claim", {"id": claim_id}, {
                "statement": claim["statement"],
                "claim_type": claim["claim_type"],
                "status": "active",
                "confidence": claim["confidence"],
                "created_by": "meta-docs-extractor",
            })
            graph.upsert_rel("Claim", {"id": claim_id}, "FOUND_IN", "Chunk",
                             {"id": chunk["id"]}, {"role": "extracted"})
        for entity in entities:
            premise_ids = [claim_ids[i] for i in entity["claims"] if i < len(claim_ids)]
            if not premise_ids:
                premise_ids = claim_ids[:2]
            if not premise_ids:
                continue
            arg_id = f"extarg-{chunk_hash}-{digest(entity['id'], 8)}"
            graph.upsert_node(entity["label"], {"id": entity["id"]}, {
                "name": entity["name"],
                **({"description": entity["description"]} if entity["description"] else {}),
            })
            graph.upsert_node("Argument", {"id": arg_id}, {"status": "in"})
            graph.upsert_rel("Argument", {"id": arg_id}, "CONCLUDES",
                             entity["label"], {"id": entity["id"]})
            for premise in premise_ids:
                graph.upsert_rel("Argument", {"id": arg_id}, "USES_PREMISE",
                                 "Claim", {"id": premise})
            graph.upsert_rel("Chunk", {"id": chunk["id"]}, "MENTIONS",
                             entity["label"], {"id": entity["id"]},
                             {"chunk_id": chunk["id"], "argument_id": arg_id,
                              "status": "active"})
        for rel in relationships:
            try:
                graph.upsert_rel(
                    rel["from"]["label"], {"id": rel["from"]["id"]}, rel["type"],
                    rel["to"]["label"], {"id": rel["to"]["id"]},
                    {"chunk_id": chunk["id"], "status": "active"},
                )
            except Exception:  # noqa: BLE001
                pass

    return {
        "chunk_id": chunk["id"],
        "status": "ok",
        "claims": len(claims),
        "entities": len(entities),
        "relationships": len(relationships),
        "graph_written": not dry_run,
    }


def load_chunks(graph: Graph, unclaimed: bool = False) -> list[dict]:
    condition = ("AND NOT ()-[:FOUND_IN]->(c) " if unclaimed else "")
    return graph.run(
        "MATCH (c:Chunk)-[:FOUND_IN]->(s:SourceNode) "
        "WHERE c.id STARTS WITH 'metadocs-chunk-' " + condition +
        "RETURN c.id AS id, c.text AS text, s.name AS source, s.url AS url "
        "ORDER BY c.id"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--unclaimed", action="store_true",
                        help="retry chunks that produced no claims")
    args = parser.parse_args()

    key = api_key()
    with Graph() as graph:
        known = known_entities(graph)
        chunks = load_chunks(graph, unclaimed=args.unclaimed)
    if args.limit:
        chunks = chunks[: args.limit]
    done = set()
    if args.resume and LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            try:
                row = json.loads(line)
                if row.get("status") == "ok":
                    done.add(row["chunk_id"])
            except Exception:  # noqa: BLE001
                pass
    chunks = [c for c in chunks if c["id"] not in done]
    print(f"chunks={len(chunks)} workers={args.workers} resume_skipped={len(done)}")

    totals = {"ok": 0, "failed": 0, "chunks_ok": 0, "chunks_failed": 0,
              "claims": 0, "entities": 0, "relationships": 0}
    started = datetime.now(timezone.utc)
    ledger = LEDGER.open("a", encoding="utf-8")
    try:
        system = SYSTEM_RETRY if args.unclaimed else SYSTEM
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_chunk, key, chunk, known, args.dry_run, system): chunk
                       for chunk in chunks}
            for index, future in enumerate(as_completed(futures), 1):
                chunk = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {"chunk_id": chunk["id"], "status": "failed",
                              "error": str(exc)[:300], "claims": 0, "entities": 0,
                              "relationships": 0, "graph_written": not args.dry_run}
                    totals["failed"] += 1
                    totals["chunks_failed"] += 1
                else:
                    totals["ok"] += 1
                    totals["chunks_ok"] += 1
                    totals["claims"] += result["claims"]
                    totals["entities"] += result["entities"]
                    totals["relationships"] += result["relationships"]
                ledger.write(json.dumps(result) + "\n")
                ledger.flush()
                if index % 25 == 0 or index == len(chunks):
                    print(f"  [{index}/{len(chunks)}] ok={totals['ok']} failed={totals['failed']} "
                          f"claims={totals['claims']} entities={totals['entities']} "
                          f"rels={totals['relationships']}", flush=True)
    finally:
        ledger.close()

    finished = datetime.now(timezone.utc)
    summary = {
        **totals,
        "chunks_total": len(chunks),
        "model": MODEL,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "seconds": round((finished - started).total_seconds(), 1),
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        target = (HERE / "sources" / "extraction-retry-summary.json"
                  if args.unclaimed else SUMMARY)
        target.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
