"""Minimal, narrowly-scoped git automation for self-improvement commits.

This is the one place self-improvement touches git: committing exactly the
file(s) a `source_editor` edit just changed (never `git add -A`, never
anything path-unrelated to the applied suggestion), discarding an edit that
failed a downstream gate (e.g. `new_tool`'s test-gate -- see
`knowledge_bases.tools.self_improvement_apply`) before it's ever
committed, and reverting exactly one prior self-improvement commit by sha
if a re-evaluation finds the change made things worse. Every command runs
with `cwd` pinned to the repo root and a hard timeout; a failure raises
rather than silently leaving a half-applied edit uncommitted or an
unrevertible mess.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_TIMEOUT_SECONDS = 30


class GitOpsError(RuntimeError):
    pass


def _run(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=_TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitOpsError(f"git {' '.join(args)} timed out") from exc
    if result.returncode != 0:
        raise GitOpsError(
            f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def commit_file(path: Path, message: str) -> str:
    """Stage and commit exactly `path`, returning the new commit's sha.

    Raises `GitOpsError` (rather than committing an empty/unrelated diff)
    if `path` has no staged changes to commit -- e.g. a suggestion whose
    "new" wording happens to match what's already on disk. A thin wrapper
    over `commit_files([path], message)` for the common single-file case.
    """
    return commit_files([path], message)


def commit_files(paths: list[Path], message: str) -> str:
    """Stage and commit exactly `paths` (e.g. a `new_tool` apply's
    both the new handler file and the catalog registration edit, as one
    atomic commit) as a single commit, returning its sha.

    Raises `GitOpsError` (rather than committing an empty/unrelated diff)
    if none of `paths` have staged changes to commit.
    """
    if not paths:
        raise GitOpsError("commit_files called with no paths")
    relatives = [str(path.resolve().relative_to(REPO_ROOT)) for path in paths]
    _run("add", "--", *relatives)
    diff = _run("diff", "--cached", "--name-only", "--", *relatives)
    if not diff.strip():
        raise GitOpsError(f"No staged changes to commit for {relatives}")
    _run("commit", "-m", message, "--", *relatives)
    return _run("rev-parse", "HEAD").strip()


def discard_paths(paths: list[Path]) -> None:
    """Discard any uncommitted working-tree changes to `paths` -- used
    when a downstream gate (e.g. `new_tool`'s test run) rejects a
    `source_editor` edit before it's ever committed. Restores each path to
    its last-committed content (`git checkout --`); a path with no
    tracked history at all (a brand-new file that was never committed) is
    removed outright instead, since `git checkout` cannot restore what was
    never committed.
    """
    if not paths:
        return
    for path in paths:
        relative = str(path.resolve().relative_to(REPO_ROOT))
        try:
            _run("cat-file", "-e", f"HEAD:{relative}")
        except GitOpsError:
            # Never committed under this path -- nothing to check out back
            # to; just remove the untracked file so it doesn't linger.
            path.unlink(missing_ok=True)
            continue
        _run("checkout", "--", relative)


def revert_commit(sha: str) -> str:
    """Revert `sha` with a new commit (never a destructive history rewrite,
    so this is safe even if other commits have landed since). Returns the
    new revert commit's sha."""
    _run("revert", "--no-edit", sha)
    return _run("rev-parse", "HEAD").strip()


def commit_exists(sha: str) -> bool:
    try:
        _run("cat-file", "-e", f"{sha}^{{commit}}")
        return True
    except GitOpsError:
        return False
