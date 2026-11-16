"""`knowledge_bases.tools.git_ops` is the one place self-improvement
touches git: committing exactly one edited file, and reverting exactly one
prior commit by sha. These tests run real `git` commands against a
throwaway repo under `tmp_path`, never the real Noesis repo."""
from __future__ import annotations

import subprocess

import pytest

import knowledge_bases.tools.git_ops as git_ops
from knowledge_bases.tools.git_ops import (
    GitOpsError,
    commit_exists,
    commit_file,
    commit_files,
    discard_paths,
    revert_commit,
)


def _run(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
    ).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    _run("init", "-q", cwd=tmp_path)
    _run("config", "user.email", "test@example.com", cwd=tmp_path)
    _run("config", "user.name", "Test", cwd=tmp_path)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("original content\n")
    untracked = tmp_path / "untouched.py"
    untracked.write_text("should never be committed\n")
    _run("add", "tracked.py", "untouched.py", cwd=tmp_path)
    _run("commit", "-q", "-m", "initial commit", cwd=tmp_path)

    monkeypatch.setattr(git_ops, "REPO_ROOT", tmp_path)
    return tmp_path


def test_commit_file_commits_only_the_given_path(repo):
    (repo / "tracked.py").write_text("edited content\n")
    (repo / "untouched.py").write_text("also edited, but should not be staged\n")

    sha = commit_file(repo / "tracked.py", "self-improve: edit tracked.py")

    log = _run("log", "-1", "--name-only", "--pretty=format:", cwd=repo).strip()
    assert log == "tracked.py"
    status = _run("status", "--porcelain", cwd=repo)
    assert "untouched.py" in status  # still dirty/uncommitted, untouched by the commit


def test_commit_file_returns_the_new_commit_sha(repo):
    (repo / "tracked.py").write_text("edited content\n")

    sha = commit_file(repo / "tracked.py", "self-improve: edit tracked.py")

    head = _run("rev-parse", "HEAD", cwd=repo).strip()
    assert sha == head


def test_commit_file_uses_the_given_commit_message(repo):
    (repo / "tracked.py").write_text("edited content\n")

    commit_file(repo / "tracked.py", "self-improve(tool_wording): my target\n\ndiagnosis text")

    message = _run("log", "-1", "--pretty=format:%B", cwd=repo)
    assert "self-improve(tool_wording): my target" in message
    assert "diagnosis text" in message


def test_commit_file_raises_when_there_is_nothing_to_commit(repo):
    with pytest.raises(GitOpsError, match="No staged changes"):
        commit_file(repo / "tracked.py", "should not commit an empty diff")


def test_revert_commit_undoes_exactly_that_commits_change(repo):
    (repo / "tracked.py").write_text("edited content\n")
    sha = commit_file(repo / "tracked.py", "self-improve: edit tracked.py")

    revert_sha = revert_commit(sha)

    assert (repo / "tracked.py").read_text() == "original content\n"
    assert revert_sha != sha
    head = _run("rev-parse", "HEAD", cwd=repo).strip()
    assert revert_sha == head


def test_revert_commit_is_itself_a_new_commit_not_a_history_rewrite(repo):
    (repo / "tracked.py").write_text("edited content\n")
    sha = commit_file(repo / "tracked.py", "self-improve: edit tracked.py")

    revert_commit(sha)

    log = _run("log", "--oneline", cwd=repo).strip().splitlines()
    assert len(log) == 3  # initial + self-improve commit + revert commit


def test_commit_exists_true_for_a_real_commit(repo):
    (repo / "tracked.py").write_text("edited content\n")
    sha = commit_file(repo / "tracked.py", "self-improve: edit tracked.py")

    assert commit_exists(sha) is True


def test_commit_exists_false_for_a_bogus_sha(repo):
    assert commit_exists("0" * 40) is False


# -- commit_files (multi-path) ---------------------------------------------------

def test_commit_files_commits_all_given_paths_as_one_commit(repo):
    second = repo / "second.py"
    second.write_text("original second\n")
    _run("add", "second.py", cwd=repo)
    _run("commit", "-q", "-m", "add second.py", cwd=repo)

    (repo / "tracked.py").write_text("edited content\n")
    second.write_text("edited second\n")

    sha = commit_files([repo / "tracked.py", second], "self-improve(new_tool): both files")

    log = sorted(_run("log", "-1", "--name-only", "--pretty=format:", cwd=repo).strip().splitlines())
    assert log == ["second.py", "tracked.py"]
    head = _run("rev-parse", "HEAD", cwd=repo).strip()
    assert sha == head


def test_commit_files_raises_for_an_empty_path_list(repo):
    with pytest.raises(GitOpsError, match="no paths"):
        commit_files([], "message")


def test_commit_files_raises_when_none_of_the_paths_changed(repo):
    with pytest.raises(GitOpsError, match="No staged changes"):
        commit_files([repo / "tracked.py"], "should not commit an empty diff")


# -- discard_paths ---------------------------------------------------------------

def test_discard_paths_restores_a_tracked_files_last_committed_content(repo):
    tracked = repo / "tracked.py"
    tracked.write_text("uncommitted edit\n")

    discard_paths([tracked])

    assert tracked.read_text() == "original content\n"


def test_discard_paths_removes_a_never_committed_file(repo):
    brand_new = repo / "brand_new.py"
    brand_new.write_text("never committed\n")

    discard_paths([brand_new])

    assert not brand_new.exists()


def test_discard_paths_handles_a_mix_of_tracked_and_new_files(repo):
    tracked = repo / "tracked.py"
    tracked.write_text("uncommitted edit\n")
    brand_new = repo / "brand_new.py"
    brand_new.write_text("never committed\n")

    discard_paths([tracked, brand_new])

    assert tracked.read_text() == "original content\n"
    assert not brand_new.exists()


def test_discard_paths_is_a_no_op_for_an_empty_list(repo):
    discard_paths([])  # must not raise
