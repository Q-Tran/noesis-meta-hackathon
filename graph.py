"""Neo4j connection + generic MERGE upsert helpers (Bolt via neo4j driver).

Lifecycle policy: nodes are never deleted. Single-entity replacement is
recorded with `SUPERSEDED_BY`/`SUPERSEDES` edges (see the methodology `Revision`
pattern in `knowledge_bases/methodology/repository.py`); bulk/logical-block retirement is recorded
by tagging every affected node with a `:Deprecated` secondary label via
`Graph.mark_deprecated` (see `Graph.reactivate` for the inverse). Generic
traversal/search entry points (`knowledge_bases/orchestration/hybrid_search.py`,
`knowledge_bases/mcp.py`) exclude `:Deprecated` nodes by default and expose an
explicit opt-in for callers that want historical results.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any, TypeVar

from config import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id_from_text(prefix: str, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}::{h}"


T = TypeVar("T")


try:
    from neo4j.time import Date, DateTime, Duration, Time

    _NEO4J_TEMPORAL_TYPES = (Date, DateTime, Duration, Time)
except (ImportError, AttributeError):
    _NEO4J_TEMPORAL_TYPES = ()

try:
    from neo4j.spatial import Point

    _NEO4J_POINT_TYPES = (Point,)
except (ImportError, AttributeError):
    _NEO4J_POINT_TYPES = ()


def json_safe(value: Any) -> Any:
    """Recursively convert graph-driver values to JSON-safe Python values."""
    value_type = type(value)
    module = value_type.__module__
    if isinstance(value, _NEO4J_TEMPORAL_TYPES) or module.startswith("neo4j.time"):
        formatter = getattr(value, "iso_format", None) or getattr(value, "isoformat", None)
        return formatter() if formatter else str(value)
    if isinstance(value, _NEO4J_POINT_TYPES) or module.startswith("neo4j.spatial"):
        return {
            "type": value_type.__name__,
            "srid": getattr(value, "srid", None),
            "coordinates": [json_safe(coordinate) for coordinate in value],
        }
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "type": value_type.__name__,
            "base64": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class Graph:
    def __init__(self, uri: str | None = None, user: str | None = None,
                 password: str | None = None):
        from neo4j import GraphDatabase  # lazy: only needed when connecting

        uri = uri or settings.neo4j_uri
        user = settings.neo4j_user if user is None else user
        password = settings.neo4j_password if password is None else password
        auth = (user, password) if user else None
        self._driver = GraphDatabase.driver(uri, auth=auth)

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Graph":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- raw query ------------------------------------------------------
    def run(self, query: str, **params: Any) -> list[dict]:
        with self._driver.session() as session:
            # neo4j expects LiteralString, ignoring type check for dynamic query string
            return [json_safe(record.data()) for record in session.run(query, **params)] # type: ignore

    def execute_write(self, work: Callable[[Any], T]) -> T:
        """Run a unit of work in one driver-managed write transaction."""
        with self._driver.session() as session:
            return session.execute_write(work)

    # --- upserts --------------------------------------------------------
    def upsert_node(self, label: str, key: dict, props: dict | None = None) -> list[dict]:
        """MERGE a node on `key` props, then SET the remaining `props` and handle weighting.

        `label` is interpolated into Cypher, so it is validated against the
        registry for the same reason `upsert_rel` validates its own: creating a
        *node* is data, but creating a *label* is a schema change, and an
        unchecked caller (LLM extraction, session memory) cannot be allowed to
        do the second while doing the first.

        Creation time is stamped here, on `ON CREATE SET`, alongside `weight`.
        Both are technical properties the ontology deliberately does not declare
        per label: when a node was written is a fact about the write, not about
        the domain, and Neo4j Community has no triggers, so this is the only
        place it can be recorded once for every writer. The `SET` below runs
        *after* `ON CREATE SET`, so a repository that supplies its own
        `created_at` keeps its value; `ON MATCH SET` never touches it, so a
        re-touch records contact (`weight`) without moving the birth time.
        """
        from knowledge_bases.ontology import require_known_label

        require_known_label(label, actor="graph.upsert_node")
        props = {k: v for k, v in (props or {}).items() if v is not None}
        key_match = ", ".join(f"`{k}`: $key_{k}" for k in key)
        params = {f"key_{k}": v for k, v in key.items()}
        
        query = f"MERGE (n:`{label}` {{{key_match}}}) "
        query += "ON CREATE SET n.weight = 1, n.created_at = datetime() "
        query += "ON MATCH SET n.weight = coalesce(n.weight, 0) + 1 "
        
        if props:
            # We want to set properties on both CREATE and MATCH for mutations
            set_props = ", ".join(f"n.`{k}` = $set_{k}" for k in props)
            params.update({f"set_{k}": v for k, v in props.items()})
            query += f"SET {set_props} "
            
        query += "RETURN n"
        return self.run(query, **params)

    def upsert_rel(self, start_label: str, start_key: dict, rel_type: str,
                   end_label: str, end_key: dict, props: dict | None = None) -> list[dict]:
        """MERGE a relationship between two existing nodes and handle weighting.

        `rel_type` and both labels are interpolated into Cypher, so an unchecked
        caller (notably LLM extraction, which asks a model for a free-text
        relationship type) could add a relationship type or label to the graph
        at runtime -- an architecture change, per `knowledge_bases/boundary.py`.
        They are validated against the registry here, at the single write
        boundary every such caller funnels through. Declaring new ontology in
        `knowledge_bases/registry.py` is what widens this.

        Edges get the same `created_at` stamp as `upsert_node` gives nodes, on
        `ON CREATE SET`. That is what makes the provenance spine and the
        structural edges (`HAS_TRACE`, `FOUND_IN`, `PART_OF`, `CALLED`) datable
        at all: they carried no time of any kind before, so "when did this
        become known" was unanswerable for 96% of the graph's edges.
        """
        from knowledge_bases.ontology import require_known_label, require_known_relationship

        actor = "graph.upsert_rel"
        require_known_relationship(rel_type, actor=actor)
        require_known_label(start_label, actor=actor)
        require_known_label(end_label, actor=actor)

        sk = ", ".join(f"`{k}`: $s_{k}" for k in start_key)
        ek = ", ".join(f"`{k}`: $e_{k}" for k in end_key)
        params = {f"s_{k}": v for k, v in start_key.items()}
        params.update({f"e_{k}": v for k, v in end_key.items()})
        
        query = (
            f"MATCH (a:`{start_label}` {{{sk}}}), (b:`{end_label}` {{{ek}}}) "
            f"MERGE (a)-[r:`{rel_type}`]->(b) "
            "ON CREATE SET r.weight = 1, r.created_at = datetime() "
            "ON MATCH SET r.weight = coalesce(r.weight, 0) + 1 "
        )
        
        props = {k: v for k, v in (props or {}).items() if v is not None}
        if props:
            set_props = ", ".join(f"r.`{k}` = $r_{k}" for k in props)
            params.update({f"r_{k}": v for k, v in props.items()})
            query += f"SET {set_props} "
            
        query += "RETURN r"
        return self.run(query, **params)

    # --- provenance -------------------------------------------------------
    def attach_source(self, label: str, key: dict, source_id: str,
                      source_props: dict | None = None, role: str | None = None) -> list[dict]:
        """Upsert a `SourceNode` and MERGE a `(n)-[:FOUND_IN]->(SourceNode)` edge.

        `role` distinguishes which aspect of `n` this source backs (e.g.
        "hosted_by", "description", "definition"), since a single node can be
        linked to more than one `SourceNode` for different aspects.
        """
        self.upsert_node("SourceNode", {"id": source_id}, source_props)
        rel_props = {"role": role} if role else None
        return self.upsert_rel(label, key, "FOUND_IN", "SourceNode", {"id": source_id}, rel_props)

    def trace_sources(self, label: str, key: dict) -> list[dict]:
        """Return every `SourceNode` a node is provenance-linked to, with the edge role."""
        key_match = ", ".join(f"`{k}`: $key_{k}" for k in key)
        params = {f"key_{k}": v for k, v in key.items()}
        query = (
            f"MATCH (n:`{label}` {{{key_match}}})-[r:FOUND_IN]->(s:SourceNode) "
            "RETURN s AS source, r.role AS role"
        )
        return self.run(query, **params)

    def attach_chunk(self, label: str, key: dict, chunk_id: str,
                     chunk_props: dict | None = None, role: str | None = None) -> list[dict]:
        """Upsert a `Chunk` node and MERGE a `(n)-[:EXTRACTED_FROM]->(Chunk)` edge."""
        self.upsert_node("Chunk", {"id": chunk_id}, chunk_props)
        rel_props = {"role": role} if role else None
        res = self.upsert_rel(label, key, "EXTRACTED_FROM", "Chunk", {"id": chunk_id}, rel_props)
        self.upsert_rel("Chunk", {"id": chunk_id}, "MENTIONS", label, key)
        return res

    def trace_chunks(self, label: str, key: dict) -> list[dict]:
        """Return every `Chunk` an entity is provenance-linked to."""
        key_match = ", ".join(f"`{k}`: $key_{k}" for k in key)
        params = {f"key_{k}": v for k, v in key.items()}
        query = (
            f"MATCH (n:`{label}` {{{key_match}}})-[:EXTRACTED_FROM]->(c:Chunk) "
            "RETURN c AS chunk"
        )
        return self.run(query, **params)

    def trace_chunk_entities(self, chunk_id: str) -> list[dict]:
        """Return all entities extracted from or mentioned by a given chunk."""
        query = (
            "MATCH (e)-[:EXTRACTED_FROM]->(c:Chunk {id: $chunk_id}) "
            "RETURN e AS entity, labels(e) AS labels"
        )
        return self.run(query, chunk_id=chunk_id)

    def trace_chunk_relationships(self, chunk_id: str) -> list[dict]:
        """Return all relationships extracted from a given chunk."""
        query = (
            "MATCH (a)-[r]->(b) WHERE r.chunk_id = $chunk_id "
            "RETURN a AS subject, labels(a) AS subject_labels, type(r) AS predicate, "
            "properties(r) AS props, b AS object, labels(b) AS object_labels"
        )
        return self.run(query, chunk_id=chunk_id)

    def trace_relationship_chunk(self, start_label: str, start_key: dict, rel_type: str,
                                 end_label: str, end_key: dict) -> dict | None:
        """Return the `Chunk` that backs a given relationship, if any."""
        sk = ", ".join(f"`{k}`: $s_{k}" for k in start_key)
        ek = ", ".join(f"`{k}`: $e_{k}" for k in end_key)
        params = {f"s_{k}": v for k, v in start_key.items()}
        params.update({f"e_{k}": v for k, v in end_key.items()})
        query = (
            f"MATCH (a:`{start_label}` {{{sk}}})-[r:`{rel_type}`]->(b:`{end_label}` {{{ek}}}) "
            "WHERE r.chunk_id IS NOT NULL "
            "MATCH (c:Chunk {id: r.chunk_id}) "
            "RETURN c AS chunk"
        )
        rows = self.run(query, **params)
        return rows[0]["chunk"] if rows else None

    def trace_chunk_provenance(self, chunk_id: str) -> dict:
        """Full provenance for a chunk: chunk node, source node, entities, and relationships."""
        chunk_res = self.run("MATCH (c:Chunk {id: $id}) RETURN c AS chunk", id=chunk_id)
        chunk = chunk_res[0]["chunk"] if chunk_res else None
        source_res = self.run(
            "MATCH (c:Chunk {id: $id})-[:FOUND_IN]->(s:SourceNode) RETURN s AS source",
            id=chunk_id,
        )
        source = source_res[0]["source"] if source_res else None
        entities = self.trace_chunk_entities(chunk_id)
        relationships = self.trace_chunk_relationships(chunk_id)
        return {
            "chunk_id": chunk_id,
            "chunk": chunk,
            "source": source,
            "entities": entities,
            "relationships": relationships,
        }

    # --- argumentation-driven resolution & repointing --------------------
    _ARGUMENT_LABEL = "Argument"
    _CLAIM_LABEL = "Claim"

    def _active_argument_exclusion(self, status: str) -> str:
        """The EXISTS clause excluding a defeated argument's edge, with the
        domain label reached through the registry like every other repoint
        query (the engine names no domain labels inline)."""
        return f"{{(:`{self._ARGUMENT_LABEL}` {{status: '{status}'}})-[r]-(n) }}"

    def record_resolution_arguments(
        self,
        chunk_id: str,
        entity_label: str,
        surface_text: str,
        candidates: list[dict],
        ref_id: str | None = None,
    ) -> str | None:
        """Record argumentation structure for entity resolution candidates at ingestion.

        The winning candidate is marked status='in'. Runner-up candidates and the null
        hypothesis ('new entity') are marked status='out' and attacked by the winner
        with reason='higher_confidence'.
        Returns the winner's argument_id, or None if Argument is not in the installed ontology.
        """
        from knowledge_bases.ontology import is_known_label

        if not is_known_label(self._ARGUMENT_LABEL) or not is_known_label(self._CLAIM_LABEL):
            return None

        if not candidates:
            return None

        winner_cand = next((c for c in candidates if c.get("status") == "in"), candidates[0])
        winner_arg_id = _id_from_text("arg", f"res::{chunk_id}::{entity_label}::{winner_cand['id']}::{ref_id or ''}")

        now_ts = _now()
        for cand in candidates:
            cand_id = cand["id"]
            is_winner = (cand == winner_cand)
            status = "in" if is_winner else "out"
            arg_id = _id_from_text("arg", f"res::{chunk_id}::{entity_label}::{cand_id}::{ref_id or ''}")
            claim_id = _id_from_text("claim", f"res::{chunk_id}::{entity_label}::{cand_id}::{ref_id or ''}")

            # Upsert Claim derivation
            claim_props = {
                "statement": f"Reference '{surface_text}' resolves to {entity_label} '{cand_id}'",
                "claim_type": "derivation",
                "status": "active" if is_winner else "superseded",
                "confidence": float(cand.get("score", 1.0)),
                "created_at": now_ts,
            }
            self.upsert_node(self._CLAIM_LABEL, {"id": claim_id}, claim_props)
            self.upsert_rel(self._CLAIM_LABEL, {"id": claim_id}, "FOUND_IN", "Chunk", {"id": chunk_id}, {"role": "origin"})

            # Upsert Argument
            arg_props = {
                "status": status,
                "computed_at": now_ts,
                "candidate_id": cand_id,
                "candidate_label": entity_label,
                "chunk_id": chunk_id,
                "score": float(cand.get("score", 1.0)),
                "surface_text": surface_text,
            }
            self.upsert_node(self._ARGUMENT_LABEL, {"id": arg_id}, arg_props)
            self.upsert_rel(self._ARGUMENT_LABEL, {"id": arg_id}, "CONCLUDES", self._CLAIM_LABEL, {"id": claim_id})

            # Winner attacks runner-up candidates
            if not is_winner:
                self.upsert_rel(
                    self._ARGUMENT_LABEL, {"id": winner_arg_id},
                    "ATTACKS",
                    self._ARGUMENT_LABEL, {"id": arg_id},
                    {"target": "conclusion", "reason": "higher_confidence"},
                )

        return winner_arg_id

    def repoint_entity_reference(
        self,
        chunk_id: str,
        old_entity_id: str,
        new_entity_id: str,
        new_entity_label: str | None = None,
        reason: str = "user_feedback",
        ref_id: str | None = None,
    ) -> dict:
        """Repoint an entity reference grounded in a chunk to a different entity.

        Replaces canonical edges directly while preserving complete history via:
        1. Updating Argument attack chain (Arg_New / Arg_Feedback -> Arg_Old).
        2. Bidirectional edge linking (old edge: status='superseded', superseded_by=new_id;
           new edge: status='active', supersedes=old_id).
        3. Property rollback (removes properties contributed solely by chunk from old entity,
           transfers them to new entity).
        4. Rewiring EXTRACTED_FROM and MENTIONS edges to chunk.
        5. Tagging orphan old entity with :Deprecated if no active edges/origins remain.
        """
        from knowledge_bases.ontology import is_known_label

        now_ts = _now()

        # 1. Determine labels
        old_nodes = self.run("MATCH (n {id: $id}) RETURN n, labels(n) AS labels", id=old_entity_id)
        old_label = "Entity"
        old_node_props = {}
        if old_nodes:
            old_node_props = old_nodes[0].get("n") or {}
            labels = [l for l in old_nodes[0].get("labels", []) if l not in ("Deprecated", "SchemaMigration")]
            if labels:
                old_label = labels[0]
        target_label = new_entity_label or old_label

        # 2. Argumentation resolution update
        old_arg_id = None
        new_arg_id = None
        if is_known_label(self._ARGUMENT_LABEL) and is_known_label(self._CLAIM_LABEL):
            old_arg_rows = self.run(
                f"MATCH (arg:`{self._ARGUMENT_LABEL}` {{chunk_id: $chunk_id, candidate_id: $old_id}}) RETURN arg",
                chunk_id=chunk_id, old_id=old_entity_id,
            )
            if old_arg_rows:
                old_arg_id = old_arg_rows[0]["arg"]["id"]
                self.run(
                    f"MATCH (arg:`{self._ARGUMENT_LABEL}` {{id: $id}}) SET arg.status = 'out', arg.updated_at = $ts",
                    id=old_arg_id, ts=now_ts,
                )
            else:
                old_arg_id = _id_from_text("arg", f"res::{chunk_id}::{old_label}::{old_entity_id}::{ref_id or ''}")

            # Mark old claim as superseded
            self.run(
                f"MATCH (arg:`{self._ARGUMENT_LABEL}` {{id: $id}})-[:CONCLUDES]->(c:`{self._CLAIM_LABEL}`) SET c.status = 'superseded'",
                id=old_arg_id,
            )

            # Check if new_entity already had a candidate argument (e.g. runner-up promotion)
            new_arg_rows = self.run(
                f"MATCH (arg:`{self._ARGUMENT_LABEL}` {{chunk_id: $chunk_id, candidate_id: $new_id}}) RETURN arg",
                chunk_id=chunk_id, new_id=new_entity_id,
            )
            if new_arg_rows:
                new_arg_id = new_arg_rows[0]["arg"]["id"]
                self.run(
                    f"MATCH (arg:`{self._ARGUMENT_LABEL}` {{id: $id}}) SET arg.status = 'in', arg.updated_at = $ts",
                    id=new_arg_id, ts=now_ts,
                )
                self.run(
                    f"MATCH (arg:`{self._ARGUMENT_LABEL}` {{id: $id}})-[:CONCLUDES]->(c:`{self._CLAIM_LABEL}`) SET c.status = 'active'",
                    id=new_arg_id,
                )
            else:
                new_arg_id = _id_from_text("arg", f"res::{chunk_id}::{target_label}::{new_entity_id}::{ref_id or ''}")
                new_claim_id = _id_from_text("claim", f"res::{chunk_id}::{target_label}::{new_entity_id}::{ref_id or ''}")
                self.upsert_node(self._CLAIM_LABEL, {"id": new_claim_id}, {
                    "statement": f"Reference repointed to {target_label} '{new_entity_id}'",
                    "claim_type": "derivation",
                    "status": "active",
                    "confidence": 1.0,
                    "created_at": now_ts,
                })
                self.upsert_rel(self._CLAIM_LABEL, {"id": new_claim_id}, "FOUND_IN", "Chunk", {"id": chunk_id}, {"role": "origin"})
                self.upsert_node(self._ARGUMENT_LABEL, {"id": new_arg_id}, {
                    "status": "in",
                    "computed_at": now_ts,
                    "candidate_id": new_entity_id,
                    "candidate_label": target_label,
                    "chunk_id": chunk_id,
                    "score": 1.0,
                    "source": reason,
                })
                self.upsert_rel(self._ARGUMENT_LABEL, {"id": new_arg_id}, "CONCLUDES", self._CLAIM_LABEL, {"id": new_claim_id})

            # Record feedback argument attacking old argument
            feedback_arg_id = _id_from_text("arg", f"feedback::{chunk_id}::{old_entity_id}->{new_entity_id}::{now_ts}")
            self.upsert_node(self._ARGUMENT_LABEL, {"id": feedback_arg_id}, {
                "status": "in",
                "reason": reason,
                "computed_at": now_ts,
                "chunk_id": chunk_id,
            })
            self.upsert_rel(
                self._ARGUMENT_LABEL, {"id": feedback_arg_id},
                "ATTACKS",
                self._ARGUMENT_LABEL, {"id": old_arg_id},
                {"target": "conclusion", "reason": reason},
            )
            # New argument also attacks old argument
            self.upsert_rel(
                self._ARGUMENT_LABEL, {"id": new_arg_id},
                "ATTACKS",
                self._ARGUMENT_LABEL, {"id": old_arg_id},
                {"target": "conclusion", "reason": reason},
            )

        # 3. Rewire active canonical relationships
        out_rels = self.run(
            "MATCH (a {id: $old_id})-[r]->(b) "
            "WHERE r.chunk_id = $chunk_id AND (r.status = 'active' OR r.status IS NULL) "
            "RETURN labels(a) AS s_labels, properties(a) AS s_props, type(r) AS rel_type, "
            "properties(r) AS r_props, labels(b) AS o_labels, properties(b) AS o_props",
            old_id=old_entity_id, chunk_id=chunk_id,
        )
        in_rels = self.run(
            "MATCH (a)-[r]->(b {id: $old_id}) "
            "WHERE r.chunk_id = $chunk_id AND (r.status = 'active' OR r.status IS NULL) "
            "RETURN labels(a) AS s_labels, properties(a) AS s_props, type(r) AS rel_type, "
            "properties(r) AS r_props, labels(b) AS o_labels, properties(b) AS o_props",
            old_id=old_entity_id, chunk_id=chunk_id,
        )

        internal_rels = {"EXTRACTED_FROM", "FOUND_IN", "MENTIONS", "SUPERSEDES", "SUPERSEDED_BY"}
        repointed_rels = []

        for rel in out_rels:
            predicate = rel.get("rel_type")
            if predicate in internal_rels:
                continue
            o_props = rel.get("o_props") or {}
            o_labels = [l for l in rel.get("o_labels", []) if l not in ("Deprecated", "SchemaMigration")]
            b_id = o_props.get("id")
            b_label = o_labels[0] if o_labels else "Entity"
            r_props = rel.get("r_props") or {}

            old_rel_id = r_props.get("id") or _id_from_text("rel", f"{old_entity_id}::{predicate}::{b_id}::{chunk_id}")
            new_rel_id = _id_from_text("rel", f"{new_entity_id}::{predicate}::{b_id}::{chunk_id}")

            # Mark old edge as superseded
            self.run(
                f"MATCH (a {{id: $old_id}})-[r:`{predicate}`]->(b {{id: $b_id}}) "
                "WHERE r.chunk_id = $chunk_id "
                "SET r.status = 'superseded', r.superseded_by = $new_rel_id, r.superseded_at = $ts",
                old_id=old_entity_id, b_id=b_id, chunk_id=chunk_id, new_rel_id=new_rel_id, ts=now_ts,
            )

            # Create new active edge
            new_edge_props = dict(r_props)
            new_edge_props.update({
                "id": new_rel_id,
                "status": "active",
                "supersedes": old_rel_id,
                "superseded_by": None,
                "superseded_at": None,
                "created_at": now_ts,
            })
            if new_arg_id:
                new_edge_props["subject_arg_id"] = new_arg_id
            self.upsert_rel(target_label, {"id": new_entity_id}, predicate, b_label, {"id": b_id}, new_edge_props)
            repointed_rels.append({"old_rel_id": old_rel_id, "new_rel_id": new_rel_id, "predicate": predicate})

        for rel in in_rels:
            predicate = rel.get("rel_type")
            if predicate in internal_rels:
                continue
            s_props = rel.get("s_props") or {}
            s_labels = [l for l in rel.get("s_labels", []) if l not in ("Deprecated", "SchemaMigration")]
            a_id = s_props.get("id")
            a_label = s_labels[0] if s_labels else "Entity"
            r_props = rel.get("r_props") or {}

            old_rel_id = r_props.get("id") or _id_from_text("rel", f"{a_id}::{predicate}::{old_entity_id}::{chunk_id}")
            new_rel_id = _id_from_text("rel", f"{a_id}::{predicate}::{new_entity_id}::{chunk_id}")

            # Mark old edge as superseded
            self.run(
                f"MATCH (a {{id: $a_id}})-[r:`{predicate}`]->(b {{id: $old_id}}) "
                "WHERE r.chunk_id = $chunk_id "
                "SET r.status = 'superseded', r.superseded_by = $new_rel_id, r.superseded_at = $ts",
                a_id=a_id, old_id=old_entity_id, chunk_id=chunk_id, new_rel_id=new_rel_id, ts=now_ts,
            )

            # Create new active edge
            new_edge_props = dict(r_props)
            new_edge_props.update({
                "id": new_rel_id,
                "status": "active",
                "supersedes": old_rel_id,
                "superseded_by": None,
                "superseded_at": None,
                "created_at": now_ts,
            })
            if new_arg_id:
                new_edge_props["object_arg_id"] = new_arg_id
            self.upsert_rel(a_label, {"id": a_id}, predicate, target_label, {"id": new_entity_id}, new_edge_props)
            repointed_rels.append({"old_rel_id": old_rel_id, "new_rel_id": new_rel_id, "predicate": predicate})

        # 4. Property rollback and transfer
        origins_raw = old_node_props.get("_property_origins")
        old_origins = {}
        if origins_raw:
            try:
                old_origins = json.loads(origins_raw)
            except Exception:
                old_origins = {}

        transferred_props = {}
        remaining_origins = dict(old_origins)
        new_origins_to_add = {}

        for prop_name, meta in list(old_origins.items()):
            c_id = meta.get("chunk_id")
            a_id = meta.get("argument_id")
            if c_id == chunk_id or (old_arg_id and a_id == old_arg_id):
                val = old_node_props.get(prop_name)
                if val is not None:
                    transferred_props[prop_name] = val
                    new_origins_to_add[prop_name] = {
                        "argument_id": new_arg_id,
                        "chunk_id": chunk_id,
                        "status": "active",
                    }
                del remaining_origins[prop_name]

        if transferred_props:
            # Transfer to new node
            self.upsert_node(target_label, {"id": new_entity_id}, transferred_props)
            new_node_res = self.run("MATCH (n {id: $id}) RETURN n._property_origins AS origins", id=new_entity_id)
            new_existing_origins = {}
            if new_node_res and new_node_res[0].get("origins"):
                try:
                    new_existing_origins = json.loads(new_node_res[0]["origins"])
                except Exception:
                    new_existing_origins = {}
            new_existing_origins.update(new_origins_to_add)
            self.run("MATCH (n {id: $id}) SET n._property_origins = $origins", id=new_entity_id, origins=json.dumps(new_existing_origins))

            # Roll back on old node
            remove_clauses = [f"n.`{k}` = null" for k in transferred_props.keys()]
            set_rem = ", ".join(remove_clauses) + ", n._property_origins = $rem_origins"
            self.run(f"MATCH (n {{id: $id}}) SET {set_rem}", id=old_entity_id, rem_origins=json.dumps(remaining_origins))

        # 5. Rewire chunk provenance edges. The old MENTIONS edge has to go with
        # the old EXTRACTED_FROM edge: otherwise the chunk keeps pointing at the
        # entity the correction just rejected, and a "which entity does this
        # mention resolve to?" query returns both answers.
        self.run("MATCH (a {id: $old_id})-[r:EXTRACTED_FROM]->(c:Chunk {id: $chunk_id}) DELETE r", old_id=old_entity_id, chunk_id=chunk_id)
        self.run("MATCH (c:Chunk {id: $chunk_id})-[r:MENTIONS]->(a {id: $old_id}) DELETE r", old_id=old_entity_id, chunk_id=chunk_id)
        self.upsert_rel(target_label, {"id": new_entity_id}, "EXTRACTED_FROM", "Chunk", {"id": chunk_id}, {"role": "extracted"})
        self.upsert_rel("Chunk", {"id": chunk_id}, "MENTIONS", target_label, {"id": new_entity_id})

        # 6. Check for orphan old entity and mark deprecated. An entity is
        # orphaned only when nothing ACTIVE vouches for it: a defeated ('out')
        # argument is not an active vouch, so its conclusion edge does not
        # count either.
        active_edges = self.run(
            "MATCH (n {id: $id})-[r]-(other) "
            "WHERE (r.status = 'active' OR (r.status IS NULL AND NOT type(r) IN ['SUPERSEDES', 'SUPERSEDED_BY', 'MENTIONS', 'EXTRACTED_FROM'])) "
            f"  AND NOT EXISTS {self._active_argument_exclusion('out')} "
            "RETURN count(r) AS c",
            id=old_entity_id,
        )
        edge_count = active_edges[0]["c"] if active_edges else 0
        if edge_count == 0 and len(remaining_origins) == 0:
            self.mark_deprecated(
                "MATCH (n) WHERE n.id = $id", {"id": old_entity_id},
                superseded_by_key={"id": new_entity_id},
                superseded_by_label=target_label,
            )

        return {
            "chunk_id": chunk_id,
            "old_entity_id": old_entity_id,
            "new_entity_id": new_entity_id,
            "old_arg_id": old_arg_id,
            "new_arg_id": new_arg_id,
            "transferred_properties": list(transferred_props.keys()),
            "repointed_relationships": repointed_rels,
            "deprecated_old_entity": (edge_count == 0 and len(remaining_origins) == 0),
            "status": "repointed",
        }

    def trace_relationship_history(self, rel_id: str) -> dict:
        """Trace the version lineage and argumentation audit trail for a relationship."""
        rows = self.run("MATCH (a)-[r {id: $id}]->(b) RETURN a, properties(r) AS r, type(r) AS predicate, b", id=rel_id)
        if not rows:
            return {"rel_id": rel_id, "found": False}
        rel_data = rows[0]
        r_props = rel_data.get("r") or {}
        if not isinstance(r_props, dict):
            r_props = {}
        history = [r_props]

        # Trace backwards (supersedes)
        curr_supersedes = r_props.get("supersedes")
        while curr_supersedes:
            prev_rows = self.run("MATCH ()-[r {id: $id}]->() RETURN properties(r) AS r", id=curr_supersedes)
            if prev_rows:
                prev_props = prev_rows[0].get("r") or {}
                if not isinstance(prev_props, dict):
                    prev_props = {}
                history.insert(0, prev_props)
                curr_supersedes = prev_props.get("supersedes")
            else:
                break

        # Trace forwards (superseded_by)
        curr_superseded_by = r_props.get("superseded_by")
        while curr_superseded_by:
            next_rows = self.run("MATCH ()-[r {id: $id}]->() RETURN properties(r) AS r", id=curr_superseded_by)
            if next_rows:
                next_props = next_rows[0].get("r") or {}
                if not isinstance(next_props, dict):
                    next_props = {}
                history.append(next_props)
                curr_superseded_by = next_props.get("superseded_by")
            else:
                break

        arg_id = r_props.get("argument_id")
        argument_trail = self.trace_argument_lineage(arg_id) if arg_id else None

        return {
            "rel_id": rel_id,
            "found": True,
            "predicate": rel_data.get("predicate"),
            "subject": rel_data.get("a"),
            "object": rel_data.get("b"),
            "current": r_props,
            "lineage": history,
            "argument_trail": argument_trail,
        }

    def trace_argument_lineage(self, argument_id: str) -> dict:
        """Trace an argument's status, concluded claims, and incoming/outgoing attacks."""
        arg_rows = self.run(f"MATCH (arg:`{self._ARGUMENT_LABEL}` {{id: $id}}) RETURN arg", id=argument_id)
        if not arg_rows:
            return {"argument_id": argument_id, "found": False}
        arg = arg_rows[0].get("arg") or {}
        claims = self.run(f"MATCH (arg:`{self._ARGUMENT_LABEL}` {{id: $id}})-[:CONCLUDES]->(c:`{self._CLAIM_LABEL}`) RETURN c", id=argument_id)
        attacks_out = self.run(f"MATCH (arg:`{self._ARGUMENT_LABEL}` {{id: $id}})-[r:ATTACKS]->(target:`{self._ARGUMENT_LABEL}`) RETURN r, target", id=argument_id)
        attacks_in = self.run(f"MATCH (attacker:`{self._ARGUMENT_LABEL}`)-[r:ATTACKS]->(arg:`{self._ARGUMENT_LABEL}` {{id: $id}}) RETURN attacker, r", id=argument_id)
        return {
            "argument_id": argument_id,
            "found": True,
            "argument": arg,
            "concludes": [c.get("c") for c in claims],
            "attacks": [{"target": a.get("target"), "props": a.get("r")} for a in attacks_out],
            "attacked_by": [{"attacker": a.get("attacker"), "props": a.get("r")} for a in attacks_in],
        }

    # --- lifecycle (never delete; tag instead) ---------------------------
    def mark_deprecated(self, match_clause: str, params: dict | None = None,
                        superseded_by_key: dict | None = None,
                        superseded_by_label: str | None = None) -> list[dict]:
        """Tag every node bound to `n` in `match_clause` with `:Deprecated`.

        Nodes are never deleted (see `docs/architecture/lifecycle.md`).
        `match_clause` is caller-supplied Cypher (e.g.
        `"MATCH (p:Project {id: $id})-[:HAS_EPIC|HAS_TASK*0..]->(n)"`) so each
        domain can walk whatever containment edges it already has, without a
        generic cross-domain "contains" concept. The clause must bind a
        variable named `n` for the node(s) to tag.

        If `superseded_by_key`/`superseded_by_label` are given (and exactly
        one node is bound to `n`), a `(n)-[:SUPERSEDED_BY]->(replacement)` /
        `(replacement)-[:SUPERSEDES]->(n)` edge pair is also created, mirroring
        the methodology revision convention (see `knowledge_bases/methodology/repository.py`).
        """
        params = dict(params or {})
        query = f"{match_clause} SET n:Deprecated "
        if superseded_by_key and superseded_by_label:
            key_match = ", ".join(f"`{k}`: $sup_{k}" for k in superseded_by_key)
            params.update({f"sup_{k}": v for k, v in superseded_by_key.items()})
            query += (
                f"WITH n MATCH (s:`{superseded_by_label}` {{{key_match}}}) "
                "MERGE (n)-[sb:SUPERSEDED_BY]->(s) "
                "ON CREATE SET sb.created_at = datetime() "
                "MERGE (s)-[sv:SUPERSEDES]->(n) "
                "ON CREATE SET sv.created_at = datetime() "
            )
        query += "RETURN n"
        return self.run(query, **params)

    def reactivate(self, match_clause: str, params: dict | None = None) -> list[dict]:
        """Remove the `:Deprecated` label from every node bound to `n`.

        Inverse of `mark_deprecated`; does not remove any `SUPERSEDED_BY`/
        `SUPERSEDES` edges, since those remain valid lineage even if a node
        is reactivated.
        """
        params = dict(params or {})
        query = f"{match_clause} REMOVE n:Deprecated RETURN n"
        return self.run(query, **params)

    # --- maintenance ----------------------------------------------------
    def wipe(self) -> None:
        self.run("MATCH (n) DETACH DELETE n")

    def ensure_indexes(self) -> None:
        """Create an index on every installed label's identity property.

        The labels come from the registry, not from a list here. This module ships
        with the engine, which owns no ontologies, so naming `Table` or `Decision`
        in it made the engine depend on ontologies a given product may not install
        -- and left a new ontology package unable to index its own labels without
        editing the engine.

        The property is each label's declared `identity`, which is what
        `upsert_node` MERGEs on, so an index can never index the wrong property.

        Failures are swallowed per statement, not per batch: "index already exists"
        is the normal case on every boot after the first, and one unsupported
        statement must not stop the remaining labels from being indexed.
        """
        from knowledge_bases.registry import index_specs

        for label, identity in index_specs():
            try:
                self.run(f"CREATE INDEX FOR (n:`{label}`) ON (n.`{identity}`)")
            except Exception:
                pass  # index may already exist



if __name__ == "__main__":
    with Graph() as g:
        print("Memgraph connection OK:", g.run("RETURN 1 AS ok"))
