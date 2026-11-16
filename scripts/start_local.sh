#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ ! -f .env ]]; then
    cp .env.example .env
    chmod 600 .env
    echo "Created .env from .env.example; configure optional integrations before enabling them."
fi

docker compose up -d --build neo4j noesis-etcd noesis-minio milvus zookeeper kafka knowledge-worker

echo "Noesis infrastructure and knowledge worker started."
echo "Neo4j Browser: http://localhost:7474"
echo "Enable Microsoft Graph ingestion with:"
echo "  docker compose --profile microsoft-graph up -d microsoft-graph-ingestion"
