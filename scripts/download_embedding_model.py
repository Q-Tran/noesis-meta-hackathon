"""Prefetch Noesis's configured sentence-transformer model."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--destination", type=Path, default=Path(settings.embedding_model_path))
    parser.add_argument("--endpoint", help="Optional Hugging Face-compatible mirror URL")
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    snapshot_download(
        args.source,
        local_dir=destination,
        endpoint=args.endpoint,
    )
    print(f"Embedding model available at {destination}")


if __name__ == "__main__":
    main()
