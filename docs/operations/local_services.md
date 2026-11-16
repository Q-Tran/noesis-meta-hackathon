# Local services

    docker compose up -d                    # infrastructure: neo4j, milvus, kafka, ...
    docker compose --profile app up -d      # + the Python services (needs a build)

## Why the app services are behind a profile

`knowledge-worker` and the connector daemons build from `Dockerfile`, and that
build runs `pip install`. Behind a TLS-intercepting corporate proxy it fails:

    SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED]
    certificate verify failed: unable to get local issuer certificate'))
    ERROR: Could not find a version that satisfies the requirement neo4j>=5.20

That is an application build problem. But while those services were in the
default set, it took down `docker compose up -d` *as a whole* -- so Neo4j and
Milvus, which are prebuilt images needing no build at all, never started
either. One failing build meant no vector search, which is why entity
resolution silently ran name-only.

The `app` profile separates them: infrastructure starts regardless of whether
the application image can currently be built.

To build behind the proxy, point pip at an internal mirror:

    PIP_INDEX_URL=https://mirror.example/simple \
    PIP_TRUSTED_HOST=mirror.example docker compose --profile app build

## Checking Milvus

    docker ps --format '{{.Names}}\t{{.Status}}' | grep noesis
    .venv/bin/python -c "from vector_store import VectorStore; VectorStore()"

Milvus is an accelerator, never a dependency: with it down, catalog search
falls back to keyword-only ranking and entity resolution to name matching.
Both print a warning rather than failing, so *check for it explicitly* -- a
degraded system looks like a working one until you compare results.
