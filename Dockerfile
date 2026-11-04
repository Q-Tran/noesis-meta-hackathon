# Shared image for Noesis's long-running Python services (knowledge-worker,
# microsoft-graph-ingestion). Dependencies are installed at build time -- not
# `pip install` on every container start -- so a container never fails to
# come up because PyPI was unreachable from inside Docker (e.g. behind a
# corporate proxy that requires a CA bundle).
#
#   docker compose build
#   # behind a corporate mirror:
#   PIP_INDEX_URL=https://mirror.example/simple PIP_TRUSTED_HOST=mirror.example docker compose build
FROM python:3.12-slim

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TRUSTED_HOST=
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Source is bind-mounted over /app by docker-compose for local development;
# copying it here too makes the image runnable on its own.
COPY . .
