"""Ingest the full Meta Model API documentation corpus into the graph.

The animated demo is a curated scene; this is the large truth layer behind it.
It discovers every docs URL from `llms.txt` (Markdown) and `sitemap.xml`
(HTML), snapshots each page as a `SourceNode`, splits the page into chunks, and
writes `(:Chunk)-[:FOUND_IN]->(:SourceNode)` for each one.

Run from the noesis root with the environment loaded:

    set -a && source .env && set +a
    python3 demo/graph_delta/crawl_meta_docs.py

Idempotent: source/chunk ids are stable, so a rerun refreshes text and weight
without duplicating the corpus. The demo recorder ignores the `metadocs-`
namespace, so the animation stays small.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from graph import Graph

BASE = "https://dev.meta.ai"
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "sources" / "metadocs-manifest.json"
PREFIX = "metadocs-"
MAX_CHARS = 900
USER_AGENT = "Mozilla/5.0 (noesis-demo; docs corpus ingestion)"


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", "ignore")


def html_text(raw: str) -> str:
    text = re.sub(r"<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ",
                  raw, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"[ \t]+", " ", text)


def markdown_chunks(markdown: str) -> list[str]:
    """Group Markdown paragraphs/sections into <= 900-char chunks."""
    text = re.sub(r"\n{3,}", "\n\n", markdown)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        # Keep code blocks with their explanatory paragraph, but cap them.
        if len(paragraph) > MAX_CHARS * 2:
            paragraph = paragraph[: MAX_CHARS * 2]
        if current and len(current) + len(paragraph) + 2 > MAX_CHARS:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def text_chunks(text: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks, current = [], ""
    for sentence in sentences:
        if len(sentence) < 20:
            continue
        if current and len(current) + len(sentence) + 1 > MAX_CHARS:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def title_of(markdown: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", markdown, flags=re.M)
    return match.group(1).strip() if match else fallback


def discover() -> list[dict]:
    """Every docs page, preferring Markdown when llms.txt offers it."""
    pages: dict[str, dict] = {}
    llms = fetch(f"{BASE}/llms.txt")
    for title, url in re.findall(
        r"\[([^\]]+)\]\((https://dev\.meta\.ai/docs[^)]+?\.md)\)", llms
    ):
        url = url.split("#")[0]
        key = url.removesuffix(".md").rstrip("/")
        pages[key] = {"url": url, "title": title, "kind": "markdown"}

    sitemap = fetch(f"{BASE}/sitemap.xml")
    for url in re.findall(r"<loc>([^<]+)</loc>", sitemap):
        url = url.rstrip("/")
        if "/docs" not in url:
            continue
        key = url.removesuffix(".md").rstrip("/")
        pages.setdefault(key, {"url": url, "title": "", "kind": "html"})
    return sorted(pages.values(), key=lambda p: p["url"])


def resolve_page(page: dict) -> tuple[str, str, list[str]]:
    """Return (title, canonical_url, chunks) for one discovered page."""
    url = page["url"]
    if page["kind"] == "markdown":
        markdown = fetch(url)
        title = title_of(markdown, page["title"] or url)
        return title, url.removesuffix(".md"), markdown_chunks(markdown)
    raw = fetch(url)
    text = html_text(raw)
    title = page["title"]
    if not title:
        match = re.search(r"<title>([^<]+)</title>", raw, flags=re.I)
        title = html.unescape(match.group(1)).strip() if match else url
    return title, url, text_chunks(text)


def slug_for(url: str) -> str:
    path = url.split("dev.meta.ai", 1)[-1].lstrip("/")
    path = path.removesuffix(".md").removesuffix("/")
    return re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="only ingest the first N discovered pages")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and count, write nothing to the graph")
    args = parser.parse_args()

    pages = discover()
    if args.limit:
        pages = pages[: args.limit]
    print(f"discovered {len(pages)} documentation pages")

    graph = None if args.dry_run else Graph()
    manifest = []
    totals = {"pages": 0, "chunks": 0, "characters": 0, "failed": 0}
    try:
        for index, page in enumerate(pages, 1):
            try:
                title, url, chunks = resolve_page(page)
            except Exception as exc:  # noqa: BLE001
                totals["failed"] += 1
                print(f"  [{index}/{len(pages)}] FAILED {page['url']}: {exc}")
                continue
            slug = slug_for(url)
            source_id = f"{PREFIX}src-{slug}"
            if graph is not None:
                graph.upsert_node("SourceNode", {"id": source_id}, {
                    "name": title,
                    "type": "documentation",
                    "url": url,
                    "reliability": "high",
                    "corpus": "meta-model-api-docs",
                    "observed_at": "2026-11-03T00:00:00+00:00",
                })
                for chunk_index, text in enumerate(chunks):
                    chunk_id = f"{PREFIX}chunk-{slug}-{chunk_index}"
                    graph.upsert_node("Chunk", {"id": chunk_id}, {
                        "text": text,
                        "source_id": source_id,
                        "index": chunk_index,
                    })
                    graph.upsert_rel("Chunk", {"id": chunk_id}, "FOUND_IN",
                                     "SourceNode", {"id": source_id},
                                     {"role": "extracted"})
            manifest.append({
                "title": title, "url": url, "kind": page["kind"],
                "source_id": source_id, "chunks": len(chunks),
                "characters": sum(len(c) for c in chunks),
            })
            totals["pages"] += 1
            totals["chunks"] += len(chunks)
            totals["characters"] += sum(len(c) for c in chunks)
            if index % 10 == 0 or index == len(pages):
                print(f"  [{index}/{len(pages)}] {totals['chunks']} chunks  {title[:60]}")
            time.sleep(0.05)
    finally:
        if graph is not None:
            graph.close()

    if not args.dry_run:
        MANIFEST.write_text(json.dumps({
            "corpus": "meta-model-api-docs",
            "base": BASE,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "totals": totals,
            "pages": manifest,
        }, indent=1) + "\n")
        print(f"manifest -> {MANIFEST}")
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    main()
