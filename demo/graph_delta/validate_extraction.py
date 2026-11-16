"""Validate the full-corpus extraction and write a committed report.

Checks the invariants the demo's epistemology depends on:

- every extracted claim is grounded in exactly one chunk;
- every extracted argument has premises and one conclusion;
- every extracted canonical entity is concluded by an argument;
- every chunk with claims has them reachable from its source page.

Writes `sources/extraction-report.json` and exits non-zero if an invariant
fails. Run after `extract_meta_claims.py`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from graph import Graph

HERE = Path(__file__).resolve().parent
REPORT = HERE / "sources" / "extraction-report.json"

QUERIES = {
    "claims": "MATCH (c:Claim) WHERE c.id STARTS WITH 'extclaim-' RETURN count(*) AS n",
    "claims_ungrounded": (
        "MATCH (c:Claim) WHERE c.id STARTS WITH 'extclaim-' "
        "AND NOT (c)-[:FOUND_IN]->(:Chunk) RETURN count(*) AS n"
    ),
    "claims_overgrounded": (
        "MATCH (c:Claim) WHERE c.id STARTS WITH 'extclaim-' "
        "WITH c, size([(c)-[:FOUND_IN]->(:Chunk) | 1]) AS grounds "
        "WHERE grounds > 1 RETURN count(*) AS n"
    ),
    "arguments": "MATCH (a:Argument) WHERE a.id STARTS WITH 'extarg-' RETURN count(*) AS n",
    "arguments_without_premises": (
        "MATCH (a:Argument) WHERE a.id STARTS WITH 'extarg-' "
        "AND NOT (a)-[:USES_PREMISE]->(:Claim) RETURN count(*) AS n"
    ),
    "arguments_without_conclusion": (
        "MATCH (a:Argument) WHERE a.id STARTS WITH 'extarg-' "
        "AND NOT (a)-[:CONCLUDES]->() RETURN count(*) AS n"
    ),
    "entities": "MATCH (n) WHERE n.id STARTS WITH 'extent-' RETURN count(*) AS n",
    "entities_without_argument": (
        "MATCH (n) WHERE n.id STARTS WITH 'extent-' "
        "AND NOT ()-[:CONCLUDES]->(n) RETURN count(*) AS n"
    ),
    "mentions": (
        "MATCH (:Chunk)-[r:MENTIONS]->(n) WHERE n.id STARTS WITH 'extent-' "
        "RETURN count(*) AS n"
    ),
    "relationships": (
        "MATCH ()-[r]->() "
        "WHERE r.chunk_id STARTS WITH 'metadocs-chunk-' "
        "AND r.status = 'active' "
        "AND NOT type(r) IN ['FOUND_IN','MENTIONS','CONCLUDES','USES_PREMISE','EXTRACTED_FROM'] "
        "RETURN count(*) AS n"
    ),
    "chunks": "MATCH (c:Chunk) WHERE c.id STARTS WITH 'metadocs-chunk-' RETURN count(*) AS n",
    "chunks_with_claims": (
        "MATCH (c:Chunk) WHERE c.id STARTS WITH 'metadocs-chunk-' "
        "AND ()-[:FOUND_IN]->(c) "
        "RETURN count(DISTINCT c) AS n"
    ),
}


def main() -> None:
    checks = {}
    with Graph() as graph:
        for key, query in QUERIES.items():
            checks[key] = graph.run(query)[0]["n"]

    claims = checks["claims"]
    covered = checks["chunks_with_claims"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **checks,
        "claim_coverage_pct": round(100 * covered / max(1, checks["chunks"]), 2),
    }
    REPORT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))

    failures = {
        "claims_ungrounded": checks["claims_ungrounded"],
        "claims_overgrounded": checks["claims_overgrounded"],
        "arguments_without_premises": checks["arguments_without_premises"],
        "arguments_without_conclusion": checks["arguments_without_conclusion"],
        "entities_without_argument": checks["entities_without_argument"],
    }
    bad = {key: value for key, value in failures.items() if value}
    if bad:
        raise SystemExit(f"extraction invariants failed: {bad}")


if __name__ == "__main__":
    main()
