"""Connector catalog and user-owned connector instances.

Connectors are how outside data becomes graph data. The catalog is derived
from the composed product (extraction sources + tool-backed pull sources),
so a narrower product never lists a connector it cannot run. Instances are
user-owned runtime data: scope, schedule, health, and last sync.
"""
from __future__ import annotations

import json
from typing import Any

from graph import Graph
from knowledge_bases.orchestration.models import ConnectorDefinition, ConnectorInstance


def _connector_from_source(source) -> ConnectorDefinition:
    handler = source.parser or source.triplets or source.collect or "worker.KnowledgeWorker"
    return ConnectorDefinition(
        key=source.key,
        name=source.source_type,
        description=f"Ingests {source.source_type} documents into the graph.",
        source_system=source.key.split(".", 1)[0],
        produces_labels=["SourceNode", "Chunk"],
        handler=handler,
        active=True,
        config_schema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "default": source.topic or ""},
                "source_type": {"type": "string", "default": source.source_type},
                "scope": {"type": "string"},
            },
        },
    )


def connector_catalog() -> list[ConnectorDefinition]:
    """The default connector set for the installed product.

    Extraction sources are the graph-backed connectors. Tool-backed pull
    sources (Power BI, Snowflake, code repositories) are included when the
    product actually registers their tools.
    """
    from knowledge_bases.registry import composed_tools, extraction_sources

    catalog = [_connector_from_source(source) for source in extraction_sources()]
    tool_keys = {tool.key for tool in composed_tools()}
    if "semantic_layer.ingest_powerbi_directory" in tool_keys:
        catalog.append(ConnectorDefinition(
            key="semantic_layer.powerbi",
            name="Power BI semantic models",
            description="Imports TMDL semantic models as Tables, Columns, and Metrics.",
            source_system="powerbi",
            produces_labels=["Table", "Column", "Metric"],
            handler="semantic_layer.ingest_powerbi_directory",
            config_schema={"type": "object", "properties": {"directory": {"type": "string"}}},
        ))
    if "semantic_layer.list_snowflake_tables" in tool_keys:
        catalog.append(ConnectorDefinition(
            key="semantic_layer.snowflake",
            name="Snowflake tables",
            description="Reads opaque and governed Snowflake objects through bounded tools.",
            source_system="snowflake",
            produces_labels=["Table", "Column"],
            handler="semantic_layer.list_snowflake_tables",
            config_schema={"type": "object", "properties": {"database": {"type": "string"}, "schema": {"type": "string"}}},
        ))
    deduped: dict[str, ConnectorDefinition] = {}
    for item in catalog:
        deduped.setdefault(item.key, item)
    return list(deduped.values())


class ConnectorRepository:
    def __init__(self, graph: Graph):
        self.graph = graph

    def sync_catalog(self, reason: str = "connector_catalog") -> list[ConnectorDefinition]:
        for connector in connector_catalog():
            self.graph.run(
                "MERGE (c:Connector {key: $key}) "
                "ON CREATE SET c.created_at = datetime() "
                "SET c.name = $name, c.description = $description, "
                "    c.source_system = $source_system, c.handler = $handler, "
                "    c.active = $active, c.produces_labels = $produces, "
                "    c.config_schema_json = $schema, c.updated_at = datetime()",
                key=connector.key, name=connector.name, description=connector.description,
                source_system=connector.source_system, handler=connector.handler,
                active=connector.active, produces=list(connector.produces_labels),
                schema=json.dumps(connector.config_schema, sort_keys=True),
            )
        return connector_catalog()

    def list_connectors(self) -> list[ConnectorDefinition]:
        rows = self.graph.run(
            "MATCH (c:Connector) RETURN properties(c) AS connector ORDER BY c.name"
        )
        if rows:
            return [self._to_definition(row.get("connector") or {}) for row in rows]
        return connector_catalog()

    @staticmethod
    def _to_definition(props: dict[str, Any]) -> ConnectorDefinition:
        schema = props.get("config_schema_json")
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except ValueError:
                schema = {}
        return ConnectorDefinition.model_validate({
            **props,
            "produces_labels": props.get("produces_labels") or [],
            "config_schema": schema or {},
        })

    def create_instance(self, connector_key: str, name: str, config: dict[str, Any] | None = None,
                        schedule: str = "") -> ConnectorInstance:
        if connector_key not in {item.key for item in self.list_connectors()}:
            raise ValueError(f"Unknown connector {connector_key!r}")
        instance = ConnectorInstance(
            id=f"connector-instance-{connector_key}-{abs(hash((name, id(config)))) :x}",
            connector_key=connector_key,
            name=name,
            config=config or {},
            schedule=schedule,
            health="unknown",
        )
        self.graph.run(
            "MERGE (i:ConnectorInstance {id: $id}) "
            "ON CREATE SET i += $props, i.created_at = datetime() "
            "WITH i "
            "MATCH (c:Connector {key: $connector_key}) "
            "MERGE (i)-[:CONFIGURES]->(c) "
            "ON CREATE SET i.created_at = datetime()",
            id=instance.id,
            props={
                "id": instance.id, "connector_key": instance.connector_key,
                "name": instance.name, "config_json": json.dumps(instance.config, sort_keys=True),
                "schedule": instance.schedule, "health": instance.health, "records_added": 0,
                "last_error": "", "created_at": None, "updated_at": None,
            },
            connector_key=connector_key,
        )
        return instance

    def list_instances(self, connector_key: str | None = None) -> list[ConnectorInstance]:
        rows = self.graph.run(
            "MATCH (i:ConnectorInstance) WHERE $key IS NULL OR i.connector_key = $key "
            "RETURN properties(i) AS instance ORDER BY i.name",
            key=connector_key,
        )
        return [self._to_instance(row.get("instance") or {}) for row in rows]

    @staticmethod
    def _to_instance(props: dict[str, Any]) -> ConnectorInstance:
        config = props.get("config_json")
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except ValueError:
                config = {}
        return ConnectorInstance.model_validate({
            **props,
            "config": config or {},
            "records_added": props.get("records_added") or 0,
            "last_error": props.get("last_error") or "",
        })

    def record_sync(self, instance_id: str, *, records_added: int = 0, error: str = "") -> ConnectorInstance:
        rows = self.graph.run(
            "MATCH (i:ConnectorInstance {id: $id}) "
            "SET i.records_added = coalesce(i.records_added, 0) + $added, "
            "    i.last_error = $error, i.last_sync_at = datetime(), "
            "    i.health = CASE WHEN $error = '' THEN 'ok' ELSE 'error' END, "
            "    i.updated_at = datetime() "
            "RETURN properties(i) AS instance",
            id=instance_id, added=records_added, error=error,
        )
        if not rows:
            raise ValueError(f"Unknown connector instance {instance_id!r}")
        return self._to_instance(rows[0].get("instance") or {})

    def health(self) -> list[dict]:
        """Per-instance health, derived from sync outcomes and graph pressure."""
        instances = self.list_instances()
        return [
            {
                "id": instance.id,
                "connector_key": instance.connector_key,
                "name": instance.name,
                "health": instance.health,
                "last_sync_at": instance.last_sync_at,
                "records_added": instance.records_added,
                "last_error": instance.last_error,
            }
            for instance in instances
        ]
