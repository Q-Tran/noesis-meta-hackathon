"""Development catalog sync: index repositories into the code graph.

    python scripts/dev_catalog.py sync                 # all repos
    python scripts/dev_catalog.py sync --repo noesis   # one repo
    python scripts/dev_catalog.py sync --skip-analytics --skip-embed
    python scripts/dev_catalog.py status               # indexed repos + staleness

Repos default to git checkouts under ~/Projects (override with
DEV_CATALOG_REPOS="name=path,name=path"). Each sync runs the full phase
DAG (scan -> structure -> parse -> cross_file -> scope_resolution ->
mro -> communities -> processes) and persists the GraphDelta. Syncs are
full reindexes; absent entities are tagged :Deprecated, never deleted.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph import Graph
from knowledge_bases.development.persist import persist
from knowledge_bases.development.phases import default_phases
from knowledge_bases.development.pipeline import PipelineContext, run_pipeline


# Repo discovery belongs to `development`, which owns `CodeRepository`; the script
# keeps the DB boundary (persist) and the analytics phases, which are its own.
from knowledge_bases.development.extraction import discover_repos


def head_sha(repo_path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def sync_repo(repo: str, path: Path, graph: Graph, *,
              skip_analytics: bool, embed: bool) -> dict:
    ctx = PipelineContext(
        repo_name=repo, repo_path=path,
        options={"commit_sha": head_sha(path)},
    )
    skip = {"communities", "processes"} if skip_analytics else set()
    results = run_pipeline(ctx, default_phases(), skip=skip)
    stats = persist(ctx, graph, embed=embed)
    stats["phases"] = {n: round(r.duration_ms) for n, r in results.items()}
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync", help="Index repositories")
    sync.add_argument("--repo", help="Sync one repo by name")
    sync.add_argument("--skip-analytics", action="store_true",
                      help="Skip communities/processes phases")
    sync.add_argument("--skip-embed", action="store_true",
                      help="Skip Milvus embedding upserts")
    sub.add_parser("status", help="Show indexed repos + staleness")
    args = parser.parse_args()

    if args.command == "status":
        from knowledge_bases.development import queries
        with Graph() as g:
            for repo in queries.list_code_repos(g):
                s = repo["staleness"]
                print(f"{repo['id']:30} {s['status']:10} "
                      f"indexed={repo['commit_sha'] or '-':10} "
                      f"files={repo['stats'].get('nodes.CodeFile', 0)} "
                      f"calls={repo['stats'].get('edges.CALLS', 0)} "
                      f"unresolved={s['unresolved_receivers']}")
        return

    repos = discover_repos()
    if args.repo:
        if args.repo not in repos:
            repos[args.repo] = (Path.home() / "Projects" / args.repo).resolve()
        repos = {args.repo: repos[args.repo]}
    with Graph() as g:
        for name, path in repos.items():
            if not path.exists():
                print(f"[dev_catalog] skipping {name}: {path} missing")
                continue
            print(f"[dev_catalog] indexing {name} ({path}) ...")
            stats = sync_repo(name, path, g,
                              skip_analytics=args.skip_analytics,
                              embed=not args.skip_embed)
            print(f"[dev_catalog] {name}: "
                  f"{stats.get('nodes.CodeFunction', 0) + stats.get('nodes.CodeMethod', 0)} defs, "
                  f"{stats.get('edges.CALLS', 0)} calls, "
                  f"{stats.get('unresolved_total', 0)} unresolved, "
                  f"{stats.get('embedded', 0)} embedded, "
                  f"phases={stats.get('phases')}")


if __name__ == "__main__":
    main()
