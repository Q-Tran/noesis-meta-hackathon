"""Refresh the Meta Model API documentation snapshots used by the demo.

The recorder intentionally reads fixed sentence snapshots so a recording is
reproducible offline. Run this when the docs change:

    python3 demo/graph_delta/source_refresh.py

Each page is saved as `<slug>.txt` with the source URL in the first line and
one normalized sentence per line. Navigation labels and whitespace are cleaned
up; the documentation itself belongs to Meta and stays linked to its source.
"""
from __future__ import annotations

import html
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "sources"
PAGES = {
    "overview": "https://dev.meta.ai/docs/overview",
    "models": "https://dev.meta.ai/docs/models",
    "quickstart": "https://dev.meta.ai/docs/quickstart",
}


def fetch(url: str) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (noesis-demo)"}
    )
    raw = urllib.request.urlopen(request, timeout=40).read().decode("utf-8", "ignore")
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)]
    return "\n".join(s for s in sentences if 40 < len(s) < 700)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for slug, url in PAGES.items():
        body = fetch(url)
        (OUT / f"{slug}.txt").write_text(f"SOURCE: {url}\n\n{body}\n")
        print(f"{slug:12} {len(body):>6} chars  -> {OUT / (slug + '.txt')}")


if __name__ == "__main__":
    main()
