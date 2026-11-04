"""Export the ingested Meta docs corpus for the full-corpus view.

Reads the `metadocs-` SourceNode/Chunk corpus from Neo4j and writes
`corpus.json`: one record per page with its section, chunk count, character
count and a short first-chunk snippet. The player uses this to draw the full
corpus map without loading 3,194 individual chunk records.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from graph import Graph

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus.json"

SECTION_ORDER = [
    "Guides", "Protocols", "Capabilities", "API reference", "Cookbook",
    "Muse Code", "Muse Spark Open", "Muse Glimmer", "Llama docs", "Other",
]

CAPABILITIES = {
    "tool-calling", "tool-search", "search-grounding", "image-understanding",
    "image-generation", "speech-to-text", "video-understanding",
    "media-segmentation", "file-handling", "reasoning", "structured-output",
    "prompt-caching", "token-counting",
}
PROTOCOLS = {"responses", "chat-completions", "messages", "choosing-an-api"}


def section_for(url: str) -> str:
    if "/llama/docs" in url:
        return "Llama docs"
    if "/docs/" not in url:
        return "Other"
    path = url.split("/docs/", 1)[1].strip("/")
    if not path:
        return "Guides"
    top = path.split("/")[0]
    if top == "api-reference":
        return "API reference"
    if top == "cookbook":
        return "Cookbook"
    if top == "muse-code":
        return "Muse Code"
    if top == "muse-spark-open":
        return "Muse Spark Open"
    if top == "muse-glimmer":
        return "Muse Glimmer"
    if top in CAPABILITIES:
        return "Capabilities"
    if top in PROTOCOLS:
        return "Protocols"
    return "Guides"


def main() -> None:
    with Graph() as graph:
        rows = graph.run(
            "MATCH (s:SourceNode) WHERE s.id STARTS WITH 'metadocs-src-' "
            "OPTIONAL MATCH (c:Chunk)-[:FOUND_IN]->(s) "
            "WITH s, c ORDER BY c.index "
            "WITH s, collect(c) AS chunks "
            "RETURN s.id AS sid, s.name AS title, s.url AS url, "
            "       size(chunks) AS chunks, "
            "       reduce(total = 0, c IN chunks | total + size(c.text)) AS characters, "
            "       chunks[0].text AS snippet "
            "ORDER BY url"
        )
        claim_rows = graph.run(
            "MATCH (cl:Claim)-[:FOUND_IN]->(c:Chunk)-[:FOUND_IN]->(s:SourceNode) "
            "WHERE c.id STARTS WITH 'metadocs-chunk-' "
            "RETURN s.id AS sid, count(DISTINCT cl) AS claims"
        )
        entity_rows = graph.run(
            "MATCH (c:Chunk)-[:MENTIONS]->(e) WHERE c.id STARTS WITH 'metadocs-chunk-' "
            "MATCH (c)-[:FOUND_IN]->(s:SourceNode) "
            "RETURN s.id AS sid, count(DISTINCT e) AS entities"
        )
        code_rows = graph.run(
            "MATCH (r:CodeRepository {id: 'noesis'}) "
            "OPTIONAL MATCH (r)-[:CONTAINS]->(f:CodeFile) "
            "WITH r, count(DISTINCT f) AS files "
            "OPTIONAL MATCH (fn:CodeFunction) WITH r, files, count(fn) AS functions "
            "OPTIONAL MATCH (cl:CodeClass) WITH r, files, functions, count(cl) AS classes "
            "OPTIONAL MATCH (m:CodeMethod) WITH r, files, functions, classes, count(m) AS methods "
            "OPTIONAL MATCH ()-[call:CALLS]->() "
            "RETURN r.id AS repo, files, functions, classes, methods, count(call) AS calls"
        )
        codebase = code_rows[0] if code_rows else {}

    claims_by_source = {row["sid"]: row["claims"] for row in claim_rows}
    entities_by_source = {row["sid"]: row["entities"] for row in entity_rows}
    pages = []
    for row in rows:
        url = row["url"] or ""
        section = section_for(url)
        snippet = (row["snippet"] or "").replace("\n", " ")
        pages.append({
            "title": row["title"] or url,
            "url": url,
            "section": section,
            "chunks": row["chunks"],
            "characters": row["characters"],
            "claims": claims_by_source.get(row["sid"], 0),
            "entities": entities_by_source.get(row["sid"], 0),
            "snippet": snippet[:240],
        })

    pages.sort(key=lambda p: (SECTION_ORDER.index(p["section"])
                              if p["section"] in SECTION_ORDER else 99,
                              -p["chunks"], p["title"].lower()))
    section_counts = Counter(p["section"] for p in pages)
    sections = [
        {"key": key, "pages": section_counts.get(key, 0),
         "chunks": sum(p["chunks"] for p in pages if p["section"] == key)}
        for key in SECTION_ORDER if section_counts.get(key)
    ]
    payload = {
        "meta": {
            "title": "Meta Model API documentation",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pages": len(pages),
            "chunks": sum(p["chunks"] for p in pages),
            "characters": sum(p["characters"] for p in pages),
            "claims": sum(p.get("claims", 0) for p in pages),
            "entities": sum(p.get("entities", 0) for p in pages),
            "codebase": codebase,
        },
        "sections": sections,
        "pages": pages,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"exported {len(pages)} pages, {payload['meta']['chunks']} chunks, "
          f"{payload['meta']['characters']} chars, {payload['meta']['claims']} claims, "
          f"{payload['meta']['entities']} entities -> {OUT}")
    for section in sections:
        print(f"  {section['key']:<16} {section['pages']:>3} pages  "
              f"{section['chunks']:>4} chunks")


if __name__ == "__main__":
    main()
