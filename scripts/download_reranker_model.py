"""Prefetch Noesis's configured cross-encoder reranker."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tarfile
import zipfile

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings

CANONICAL_WEIGHT_SHA256 = "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=settings.reranker_model,
        help="Hugging Face repository or corporate mirror repository ID",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(settings.reranker_model_path),
        help="Persistent local model directory",
    )
    parser.add_argument(
        "--endpoint",
        help="Optional Hugging Face-compatible corporate mirror URL",
    )
    parser.add_argument(
        "--ca-bundle", type=Path,
        help="Corporate CA bundle used to verify the model source",
    )
    parser.add_argument(
        "--archive", type=Path,
        help="Import a corporate-supplied .zip, .tar.gz, or .tgz model archive",
    )
    parser.add_argument(
        "--directory", type=Path,
        help="Import a checked-out model directory",
    )
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    if args.ca_bundle:
        ca_bundle = str(args.ca_bundle.expanduser().resolve())
        os.environ["SSL_CERT_FILE"] = ca_bundle
        os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
    if args.archive and args.directory:
        raise ValueError("Use either --archive or --directory, not both")
    if args.directory:
        source_directory = args.directory.expanduser().resolve()
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source_directory, destination, ignore=shutil.ignore_patterns(".git"))
    elif args.archive:
        archive = args.archive.expanduser().resolve()
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as package:
                package.extractall(destination)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as package:
                package.extractall(destination, filter="data")
        else:
            raise ValueError("Model archive must be a .zip, .tar.gz, or .tgz file")
        children = [child for child in destination.iterdir() if child.name != "__MACOSX"]
        if len(children) == 1 and children[0].is_dir():
            nested = children[0]
            for child in nested.iterdir():
                shutil.move(str(child), destination / child.name)
            nested.rmdir()
    else:
        snapshot_download(
            args.source,
            local_dir=destination,
            endpoint=args.endpoint,
        )
    required = (destination / "config.json", destination / "tokenizer_config.json")
    missing = [str(path.name) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Invalid reranker model package; missing: {', '.join(missing)}")
    weight = destination / "model.safetensors"
    pytorch_weight = destination / "pytorch_model.bin"
    if not weight.is_file() and not pytorch_weight.is_file():
        raise RuntimeError("Invalid reranker model package; model weights are missing")
    if weight.is_file():
        import hashlib
        digest_hash = hashlib.sha256()
        with weight.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest_hash.update(block)
        digest = digest_hash.hexdigest()
        if digest != CANONICAL_WEIGHT_SHA256:
            raise RuntimeError(
                f"Unexpected model.safetensors SHA-256: {digest}"
            )
    print(f"Reranker model available at {destination}")


if __name__ == "__main__":
    main()
