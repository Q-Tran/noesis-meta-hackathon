"""Shared utilities, adapted from mem0ai/memory-benchmarks (benchmarks/common/utils.py)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def cutoff_label(cutoff: int | None) -> str:
    return "all" if cutoff is None else f"top_{cutoff}"


def parse_cutoffs(cutoffs_str: str) -> list[int]:
    return [int(c.strip()) for c in cutoffs_str.split(",") if c.strip()]


def save_result_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_result_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


class Checkpoint:
    """Per-question JSON checkpoint for resumable runs."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, question_id: str) -> Path:
        safe_id = question_id.replace("/", "_").replace("\\", "_")
        return self.checkpoint_dir / f"{safe_id}.json"

    def exists(self, question_id: str) -> bool:
        return self._path(question_id).exists()

    def load(self, question_id: str) -> dict | None:
        path = self._path(question_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, question_id: str, data: dict) -> None:
        self._path(question_id).write_text(json.dumps(data, indent=2, default=str))
