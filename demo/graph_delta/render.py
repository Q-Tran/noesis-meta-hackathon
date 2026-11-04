"""Build the self-contained player: `events.json` + `template.html` -> `index.html`.

`index.html` embeds the recording, so it opens by double-click (file://) with no
server and no network: the player is plain canvas/JS with no dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template.html"
EVENTS = HERE / "events.json"
OUT = HERE / "index.html"


def main() -> None:
    data = json.loads(EVENTS.read_text())
    corpus_path = HERE / "corpus.json"
    corpus = (json.loads(corpus_path.read_text()) if corpus_path.exists()
              else {"meta": {"pages": 0, "chunks": 0, "characters": 0},
                    "sections": [], "pages": []})
    html = (TEMPLATE.read_text()
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__CORPUS__", json.dumps(corpus, ensure_ascii=False)))
    OUT.write_text(html)
    print(f"rendered {OUT}  ({OUT.stat().st_size/1024:.0f} KB, "
          f"{len(data['events'])} events, {len(data['nodes'])} nodes, "
          f"{len(data['edges'])} edges, {len(corpus.get('pages', []))} corpus pages)")


if __name__ == "__main__":
    main()
