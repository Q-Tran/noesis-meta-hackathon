"""Tool layer for the graph chat: read the corpus/entities, write claims and arguments.

The chat model is Muse Wonder 1.0 through Meta Model API (the same `meta`
provider OpenCode uses; override with CHAT_MODEL). Tools are native OpenAI
function calls, not schemas pasted into the prompt, so tool definitions stay
out of the conversation tokens and each result is compact.

Writes are bounded: labels, relationship types and update properties are
validated against the registry; there is no raw Cypher tool. A claim is linked
to the current chat turn (`FOUND_IN` the RoutingTrace) unless the caller passes
a `chunk_id`, in which case it is linked to that verbatim chunk.

Corrections are first-class and epistemic, never deletions or re-ingestions:
`challenge_argument` records a warrant attack (`Argument-[:ATTACKS {target:
"warrant"}]->Argument`) with the reason as its premise claim, and
`repoint_reference` supersedes the old argument with a new one (plus a
feedback argument ATTACKING the old conclusion) while rewiring the canonical
edges through SUPERSEDES pairs. Every change originates from an argument or
claim with its own provenance, so the canonical layer's history stays intact.
"""
from __future__ import annotations

import hashlib
import uuid
from typing import Any

from graph import Graph
from knowledge_bases import registry
from knowledge_bases.orchestration.hybrid_search import SEARCH_TYPES, hybrid_search
from knowledge_bases.provenance.conversation import ground_claim_in_trace
from knowledge_bases.tools.trace_store import TraceStore

DEFAULT_ENTITY_TYPES = [
    "architecture.Component",
    "architecture.Interface",
    "project_management.Capability",
    "project_management.Specification",
    "business_dictionary.BusinessTerm",
    "semantic_layer.Metric",
    "development.CodeFunction",
    "development.CodeMethod",
    "development.CodeClass",
]
WRITE_LABELS = {
    "Component", "Interface", "Capability", "Specification", "BusinessTerm",
    "Metric",
}
UPDATE_PROPS = {
    "description", "status", "confidence", "definition", "calculation_summary",
    "unit", "grain", "availability", "review_status", "target",
}
CLAIM_TYPES = {"fact", "finding", "assumption", "rationale", "question"}

_model = None
_store = None
_llm = None


def _hash(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode()).hexdigest()[:12]


def _models():
    global _model, _store, _llm
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from config import settings

        # The repo root holds `models/`; settings paths are relative to it, and
        # the server may be started from any CWD (starting it inside demo/ is
        # exactly how the weights go "missing" at demo time).
        import pathlib

        repo_root = pathlib.Path(__file__).resolve().parents[2]
        model_path = pathlib.Path(settings.embedding_model_path)
        if not model_path.is_absolute():
            model_path = repo_root / model_path
        _model = SentenceTransformer(str(model_path), local_files_only=True)
    if _store is None:
        from vector_store import VectorStore

        _store = VectorStore(timeout=20)
    if _llm is None:
        from llm import get_llm

        _llm = get_llm()
    return _model, _store, _llm


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _qualified_type(entity_type: str) -> str | None:
    """Accept `Component` or `architecture.Component` and return the qualified one."""
    if entity_type in SEARCH_TYPES:
        return entity_type
    wanted = entity_type.strip().lower()
    matches = [
        name for name, config in SEARCH_TYPES.items()
        if name.lower() == wanted or config.label.lower() == wanted
    ]
    return matches[0] if len(matches) == 1 else None


def _bare_label(entity_type: str) -> str | None:
    """Accept `architecture.Component` or `Component` and return the graph label."""
    label = entity_type.rsplit(".", 1)[-1].strip()
    return label if label in registry.entity_types() else None


class ChatTools:
    """One chat turn: trace + graph tools."""

    def __init__(self, graph: Graph, session_id: str):
        self.graph = graph
        self.session_id = session_id
        self.trace_id: str | None = None
        # The entities the turn surfaced through search/lookup, for the ask
        # view's "what the agent touched" mini map. Capped: a map, not a list.
        self.turn_entities: dict[str, dict] = {}

    def _note_entity(self, entity_id, title, label) -> None:
        if entity_id and len(self.turn_entities) < 8:
            self.turn_entities.setdefault(
                str(entity_id),
                {"id": str(entity_id), "title": str(title or entity_id)[:80],
                 "label": str(label or "Node")},
            )

    def begin_turn(self, question: str) -> str:
        self.trace_id = f"chat-{uuid.uuid4().hex}"
        TraceStore(self.graph).record_turn_start(
            self.trace_id, self.session_id, question, client="web-chat",
        )
        return self.trace_id

    # ---------------------------------------------------------------- reads
    def search_corpus(self, query: str, top_k: int = 5) -> dict:
        model, store, _ = _models()
        hits = store.search_vectors(
            "chunks", model.encode(query, normalize_embeddings=True).tolist(),
            top_k=max(8, top_k * 3),
        )
        hits = [h for h in hits if str(h.get("id", "")).startswith("metadocs-chunk-")]
        results = []
        for hit in hits[:top_k]:
            rows = self.graph.run(
                "MATCH (c:Chunk {id: $id})-[:FOUND_IN]->(s:SourceNode) "
                "OPTIONAL MATCH (cl:Claim)-[:FOUND_IN]->(c) "
                "RETURN s.name AS page, s.url AS url, c.text AS text, "
                "       collect(DISTINCT cl.statement)[0..4] AS claims",
                id=hit["id"],
            )
            if not rows:
                continue
            row = rows[0]
            results.append({
                "chunk_id": hit["id"],
                "similarity": round(float(hit.get("similarity", 0)), 3),
                "page": row["page"],
                "url": row["url"],
                "claims": [c for c in row["claims"] if c],
                "snippet": " ".join((row["text"] or "").split())[:240],
            })
        return {"results": results}

    def search_entities(
        self, query: str, entity_types: list[str] | None = None, limit: int = 5,
    ) -> dict:
        if isinstance(entity_types, str):
            entity_types = [entity_types]
        requested = entity_types or DEFAULT_ENTITY_TYPES
        types = []
        unknown = []
        for name in requested:
            qualified = _qualified_type(name)
            if qualified is None:
                unknown.append(name)
            elif qualified not in types:
                types.append(qualified)
        if unknown:
            return {
                "error": f"unknown entity types: {unknown}",
                "hint": "Use a qualified name such as architecture.Component, "
                        "project_management.Capability, business_dictionary.BusinessTerm, "
                        "semantic_layer.Metric, or development.CodeFunction.",
            }
        _, store, llm = _models()
        groups: dict[str, list[str]] = {}
        for name in types:
            groups.setdefault(SEARCH_TYPES[name].family, []).append(name)
        results = []
        for family, group in groups.items():
            try:
                rows = hybrid_search(
                    query, group, limit=limit, graph=self.graph,
                    vector_store=store, llm=llm,
                )
            except Exception as exc:  # noqa: BLE001
                results.append({"family": family, "error": str(exc)[:160]})
                continue
            for row in rows:
                props = row.get("properties") or {}
                results.append({
                    "type": row["qualified_type"],
                    "identity": row.get("identity"),
                    "name": props.get("name") or props.get("qualname"),
                    "description": props.get("description") or props.get("docstring"),
                    "score": round(row.get("score", 0), 4),
                    "vector_rank": row.get("semantic_rank"),
                    "text_rank": row.get("lexical_rank"),
                })
        results.sort(key=lambda r: r.get("score", 0), reverse=True)
        results = results[: limit * 3]
        for row in results[:5]:
            self._note_entity(row.get("identity"), row.get("name"),
                              str(row.get("type") or "").split(".")[-1])
        return {"results": results}

    def get_entity(self, entity_type: str, identity: str) -> dict:
        label = _bare_label(entity_type)
        if label is None:
            return {"error": f"unknown entity type: {entity_type}"}
        rows = self.graph.run(
            f"MATCH (n:`{label}`) WHERE n.id = $id OR n.name = $id "
            "RETURN properties(n) AS props LIMIT 1",
            id=identity,
        )
        if not rows:
            return {"error": f"{label} {identity!r} not found"}
        props = rows[0]["props"]
        self._note_entity(props.get("id") or identity,
                          props.get("name") or props.get("statement"), label)
        claims = self.graph.run(
            "MATCH (arg:Argument)-[:CONCLUDES]->(n) WHERE n.id = $id OR n.name = $id "
            "MATCH (arg)-[:USES_PREMISE]->(c:Claim) "
            "RETURN DISTINCT c.id AS id, c.statement AS statement, c.confidence AS confidence LIMIT 20",
            id=identity,
        )
        rels = self.graph.run(
            "MATCH (n)-[r]-(m) WHERE n.id = $id OR n.name = $id "
            "RETURN type(r) AS type, labels(m)[0] AS other_label, "
            "       coalesce(m.name, m.id) AS other LIMIT 30",
            id=identity,
        )
        # The arguments concluding this entity, with their ids verbatim:
        # challenge_argument takes an argument id, and guessing one from a
        # claim id's prefix fails exactly the way a demo audience will notice.
        args = self.graph.run(
            "MATCH (arg:Argument)-[:CONCLUDES]->(n) WHERE n.id = $id OR n.name = $id "
            "RETURN arg.id AS id, arg.status AS status, arg.chunk_id AS chunk_id, "
            "       [(arg)-[:USES_PREMISE]->(c:Claim) | c.statement][0..3] AS premises "
            "LIMIT 12",
            id=identity,
        )
        return {
            "type": label,
            "identity": identity,
            "properties": _jsonable(props),
            "claims": _jsonable(claims),
            "relationships": _jsonable(rels),
            "arguments": _jsonable(args),
        }

    # --------------------------------------------------------------- writes
    def add_claim(
        self, statement: str, claim_type: str = "fact",
        confidence: float = 0.85, chunk_id: str | None = None,
    ) -> dict:
        statement = " ".join((statement or "").split())[:600]
        if len(statement) < 12:
            return {"error": "statement is too short"}
        if claim_type not in CLAIM_TYPES:
            return {"error": f"invalid claim_type: {claim_type}"}
        # Validate the grounding target before writing: a bad chunk id must not
        # leave a claim node with no provenance.
        if chunk_id:
            rows = self.graph.run("MATCH (c:Chunk {id: $id}) RETURN c.id AS id", id=chunk_id)
            if not rows:
                return {"error": f"chunk {chunk_id!r} not found"}
        elif not self.trace_id:
            return {"error": "no active chat turn to ground the claim"}
        claim_id = f"chatclaim-{_hash(self.session_id, statement)}"
        self.graph.upsert_node("Claim", {"id": claim_id}, {
            "statement": statement,
            "claim_type": claim_type,
            "status": "active",
            "confidence": max(0.0, min(1.0, float(confidence))),
            "created_by": "web-chat",
        })
        if chunk_id:
            self.graph.upsert_rel("Claim", {"id": claim_id}, "FOUND_IN", "Chunk",
                                  {"id": chunk_id}, {"role": "extracted"})
            grounded = f"chunk:{chunk_id}"
        else:
            ground_claim_in_trace(self.graph, self.trace_id, claim_id)
            grounded = f"trace:{self.trace_id}"
        return {"claim_id": claim_id, "statement": statement, "grounded_in": grounded}

    def _resolve_entity(
        self, entity_type: str, entity_id: str | None, entity_name: str | None,
        create: bool = False,
    ) -> tuple[dict | None, str | None, str | None]:
        """Return (props, identity_property, identity_value)."""
        entity_type = _bare_label(entity_type) or entity_type
        if entity_type not in WRITE_LABELS:
            return None, None, f"writes are not allowed for {entity_type}"
        entity_types = registry.entity_types()
        if entity_type not in entity_types:
            return None, None, f"unknown entity type {entity_type}"
        identity_prop = entity_types[entity_type].identity
        # Callers may put either an id or a name in either field; resolve one
        # identity against both properties (BusinessTerm identifies by name).
        rows = self.graph.run(
            f"MATCH (n:`{entity_type}`) "
            "WHERE n.id = $ident OR n.name = $ident "
            "RETURN properties(n) AS props LIMIT 1",
            ident=entity_id or entity_name,
        )
        if rows:
            props = rows[0]["props"]
            return props, identity_prop, str(props.get(identity_prop))
        if not create:
            return None, None, f"{entity_type} not found"
        name = (entity_name or "").strip()
        if not name:
            return None, None, "entity_name is required to create an entity"
        value = name if identity_prop == "name" else f"chatent-{_hash(entity_type, name)}"
        key = {identity_prop: value}
        self.graph.upsert_node(entity_type, key, {"name": name})
        return {"name": name}, identity_prop, value

    def add_argument(
        self, entity_type: str, claim_ids: list[str],
        entity_id: str | None = None, entity_name: str | None = None,
        description: str | None = None,
    ) -> dict:
        entity_type = _bare_label(entity_type) or entity_type
        if isinstance(claim_ids, str):
            claim_ids = [claim_ids]
        if not claim_ids:
            return {"error": "claim_ids is required"}
        found = self.graph.run(
            "MATCH (c:Claim) WHERE c.id IN $ids RETURN c.id AS id", ids=claim_ids,
        )
        existing = {row["id"] for row in found}
        missing = [c for c in claim_ids if c not in existing]
        if missing:
            return {"error": f"claims not found: {missing}"}
        props, identity_prop, identity_value = self._resolve_entity(
            entity_type, entity_id, entity_name, create=True,
        )
        if props is None:
            return {"error": identity_value}
        arg_id = f"chatarg-{_hash(self.session_id, entity_type, identity_value or '', *sorted(claim_ids), description or '')}"
        arg_props: dict[str, Any] = {
            "status": "in",
            **({"description": description} if description else {}),
        }
        # An argument concluding an entity from claims grounded in ONE chunk is
        # an entity-resolution candidate for (chunk, entity): stamp it so
        # corrections (repoint_reference) can find and supersede it properly.
        chunk_rows = self.graph.run(
            "MATCH (c:Claim)-[:FOUND_IN]->(ch:Chunk) WHERE c.id IN $ids "
            "RETURN DISTINCT ch.id AS chunk_id",
            ids=claim_ids,
        )
        chunks = {row["chunk_id"] for row in chunk_rows}
        if len(chunks) == 1:
            arg_props.update({
                "chunk_id": chunks.pop(),
                "candidate_id": identity_value,
                "candidate_label": entity_type,
            })
        self.graph.upsert_node("Argument", {"id": arg_id}, arg_props)
        self.graph.upsert_rel("Argument", {"id": arg_id}, "CONCLUDES",
                              entity_type, {identity_prop: identity_value})
        for claim_id in claim_ids:
            self.graph.upsert_rel("Argument", {"id": arg_id}, "USES_PREMISE",
                                  "Claim", {"id": claim_id})
        return {
            "argument_id": arg_id,
            "concludes": f"{entity_type}:{identity_value}",
            "premises": claim_ids,
        }

    def relate_entities(
        self, start_type: str, start_id: str, rel_type: str,
        end_type: str, end_id: str, properties: dict | None = None,
    ) -> dict:
        start_type = _bare_label(start_type) or start_type
        end_type = _bare_label(end_type) or end_type
        properties = properties if isinstance(properties, dict) else {}
        # `start_id`/`end_id` are identities, not necessarily the `id` property:
        # BusinessTerm is identified by name. Resolve each endpoint first so the
        # MERGE key uses that label's identity property.
        _, start_prop, start_value = self._resolve_entity(
            start_type, start_id, start_id, create=False,
        )
        if start_prop is None:
            return {"error": f"start: {start_value}"}
        _, end_prop, end_value = self._resolve_entity(
            end_type, end_id, end_id, create=False,
        )
        if end_prop is None:
            return {"error": f"end: {end_value}"}
        try:
            self.graph.upsert_rel(
                start_type, {start_prop: start_value}, rel_type,
                end_type, {end_prop: end_value},
                {k: v for k, v in (properties or {}).items()
                 if isinstance(v, (str, int, float, bool))},
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)[:200]}
        return {"relationship": rel_type,
                "from": f"{start_type}:{start_value}",
                "to": f"{end_type}:{end_value}"}

    def update_entity(
        self, entity_type: str, identity: str, properties: dict,
    ) -> dict:
        entity_type = _bare_label(entity_type) or entity_type
        properties = properties if isinstance(properties, dict) else {}
        props, identity_prop, identity_value = self._resolve_entity(
            entity_type, identity, identity, create=False,
        )
        if props is None:
            return {"error": identity_value}
        clean = {k: v for k, v in properties.items()
                 if k in UPDATE_PROPS and isinstance(v, (str, int, float, bool))}
        if not clean:
            return {"error": f"no allowed properties; allowed: {sorted(UPDATE_PROPS)}"}
        self.graph.upsert_node(entity_type, {identity_prop: identity_value}, clean)
        return {"updated": f"{entity_type}:{identity_value}", "properties": clean}

    # --------------------------------------------------------- corrections
    def challenge_argument(
        self, argument_id: str | None = None, reason: str = "",
        claim_id: str | None = None,
    ) -> dict:
        """Dispute an argument's derivation from its chunk: a warrant attack.

        The ontology's shape for "this doesn't make sense from this chunk": a
        NEW argument ATTACKS the disputed one with target=warrant, taking the
        user's reason as its premise claim (grounded in this chat turn). The
        disputed argument's status is untouched on purpose -- an unresolved
        challenge lives in the ATTACKS relation, and the resolution is
        repoint_reference, which supersedes properly.

        Pass `argument_id` directly, or `claim_id` to attack the argument using
        that exact claim as a premise -- the only reliable way to challenge
        "the argument behind THIS statement"; models that guess argument ids
        from claim-id prefixes attack the wrong argument.
        """
        reason = " ".join((reason or "").split())[:600]
        if len(reason) < 12:
            return {"error": "reason is too short (state why the chunk does not support the argument)"}
        if not argument_id:
            if not claim_id:
                return {"error": "argument_id (or claim_id) is required"}
            rows = self.graph.run(
                "MATCH (arg:Argument)-[:USES_PREMISE]->(c:Claim {id: $id}) "
                "RETURN arg.id AS id, properties(arg) AS props "
                "ORDER BY coalesce(arg.computed_at, '') DESC LIMIT 1",
                id=claim_id,
            )
            if not rows:
                return {"error": f"no argument uses claim {claim_id!r} as a premise"}
            argument_id = rows[0]["id"]
            arg = rows[0]["props"]
        else:
            rows = self.graph.run(
                "MATCH (arg:Argument {id: $id}) RETURN properties(arg) AS props LIMIT 1",
                id=argument_id,
            )
            if not rows:
                return {"error": f"argument {argument_id!r} not found"}
            arg = rows[0]["props"]

        claim = self.add_claim(reason, claim_type="finding")
        if "error" in claim:
            return claim
        attack_id = f"chatarg-attack-{_hash(self.session_id, argument_id, reason)}"
        self.graph.upsert_node("Argument", {"id": attack_id}, {
            "status": "in",
            "reason": reason,
            **({"chunk_id": arg.get("chunk_id")} if arg.get("chunk_id") else {}),
        })
        self.graph.upsert_rel("Argument", {"id": attack_id}, "USES_PREMISE",
                              "Claim", {"id": claim["claim_id"]})
        self.graph.upsert_rel("Argument", {"id": attack_id}, "ATTACKS",
                              "Argument", {"id": argument_id},
                              {"target": "warrant", "reason": reason})
        return {
            "attack_argument_id": attack_id,
            "attacks": argument_id,
            "attack_target": "warrant",
            "premise_claim_id": claim["claim_id"],
            "grounded_in": claim["grounded_in"],
            "note": "the disputed argument keeps its status; resolve with repoint_reference",
        }

    def repoint_reference(
        self, chunk_id: str, old_entity_id: str, new_entity_id: str,
        new_entity_label: str | None = None, reason: str = "user_feedback",
    ) -> dict:
        """Correct which entity a chunk-grounded reference resolves to.

        The no-re-ingestion primitive: `repoint_entity_reference` supersedes
        the old argument (status 'out', conclusion 'superseded'), installs the
        new one (status 'in'), records a feedback argument ATTACKING the old
        conclusion, rewires canonical edges through SUPERSEDES pairs, and
        transfers chunk-contributed properties. A turn-grounded claim records
        that this correction originated in this conversation.
        """
        rows = self.graph.run("MATCH (c:Chunk {id: $id}) RETURN c.id AS id", id=chunk_id)
        if not rows:
            return {"error": f"chunk {chunk_id!r} not found"}
        old_rows = self.graph.run(
            "MATCH (n {id: $id}) RETURN labels(n) AS labels LIMIT 1", id=old_entity_id,
        )
        if not old_rows:
            return {"error": f"entity {old_entity_id!r} not found"}
        old_label = next(
            (l for l in old_rows[0]["labels"] if l not in ("Deprecated", "SchemaMigration")),
            "Entity",
        )
        # Extraction-era concluding arguments predate the candidate model: no
        # chunk_id/candidate_id stamp, so the engine's supersede chain cannot
        # find them. Stamp them first, or a repoint retires the entity's edges
        # while an 'in' argument still vouches for it.
        stale_args = self.graph.run(
            "MATCH (arg:Argument)-[:CONCLUDES]->(n {id: $old_id}) "
            "WHERE arg.candidate_id IS NULL OR arg.chunk_id IS NULL "
            "RETURN arg.id AS id",
            old_id=old_entity_id,
        )
        for row in stale_args:
            self.graph.run(
                "MATCH (arg:Argument {id: $id}) "
                "SET arg.chunk_id = $chunk_id, arg.candidate_id = $old_id, "
                "    arg.candidate_label = $label",
                id=row["id"], chunk_id=chunk_id, old_id=old_entity_id, label=old_label,
            )
        result = self.graph.repoint_entity_reference(
            chunk_id, old_entity_id, new_entity_id,
            new_entity_label=new_entity_label, reason=reason,
        )
        note = self.add_claim(
            f"Correction: the reference grounded in chunk {chunk_id} was repointed "
            f"from {old_label} '{old_entity_id}' to "
            f"{new_entity_label or old_label} '{new_entity_id}'. Reason: {reason}.",
            claim_type="rationale",
        )
        result["correction_claim_id"] = note.get("claim_id")
        result["stamped_old_arguments"] = [row["id"] for row in stale_args]
        result["grounded_in"] = note.get("grounded_in")
        result["reingested"] = False
        return result


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": "Semantic search over the Meta Model API docs chunks. Returns chunk_id, page, URL, claims and snippet. Pass chunk_id to add_claim when a claim should cite that chunk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_entities",
            "description": "Hybrid lexical+vector search over canonical entities (components, interfaces, capabilities, specifications, terms, metrics, code).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "entity_types": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity",
            "description": "Fetch one canonical entity's properties, claims, relationships, and the arguments concluding it (with their exact ids — required by challenge_argument).",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string"},
                    "identity": {"type": "string"},
                },
                "required": ["entity_type", "identity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_claim",
            "description": "Add a claim to the graph. The statement must be one concise assertion (no source list or URLs; provenance is recorded as graph edges). Pass chunk_id from search_corpus to ground a docs-backed claim in that Chunk (FOUND_IN); omit it only when the claim is about the conversation. Only call when the user explicitly asks to add knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "claim_type": {"type": "string", "enum": sorted(CLAIM_TYPES)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "chunk_id": {
                        "type": "string",
                        "description": "Exact Chunk id returned by search_corpus. "
                                       "Required when the claim cites documentation; omit "
                                       "only for a claim about this conversation.",
                    },
                },
                "required": ["statement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_argument",
            "description": "Attach claims to a canonical entity with an Argument (CONCLUDES + USES_PREMISE). Creates the entity if entity_name is supplied and it does not exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string", "enum": sorted(WRITE_LABELS)},
                    "entity_id": {
                        "type": "string",
                        "description": "Existing entity id or name.",
                    },
                    "entity_name": {
                        "type": "string",
                        "description": "Existing name, or the name to create when absent.",
                    },
                    "claim_ids": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                },
                "required": ["entity_type", "claim_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "relate_entities",
            "description": "Create or update a validated relationship between two existing canonical entities. start_id and end_id are each entity's identity: its id, or its name for BusinessTerm and other name-identified labels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_type": {"type": "string"},
                    "start_id": {"type": "string"},
                    "rel_type": {"type": "string"},
                    "end_type": {"type": "string"},
                    "end_id": {"type": "string"},
                    "properties": {"type": "object"},
                },
                "required": ["start_type", "start_id", "rel_type", "end_type", "end_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_entity",
            "description": "Update whitelisted properties of a canonical entity (identity is the id, or the name for BusinessTerm) (description, status, confidence, definition, calculation_summary, unit, grain, availability, review_status, target).",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string", "enum": sorted(WRITE_LABELS)},
                    "identity": {"type": "string"},
                    "properties": {"type": "object"},
                },
                "required": ["entity_type", "identity", "properties"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "challenge_argument",
            "description": (
                "Challenge an argument: record that its derivation from its chunk "
                "does not hold. Writes a new Argument that ATTACKS it with "
                "target=warrant, with your reason as the premise claim (grounded "
                "in this conversation). The disputed argument keeps its status; "
                "resolve the challenge with repoint_reference. Use when the user "
                "says an argument does not make sense or is not supported by its chunk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "argument_id": {
                        "type": "string",
                        "description": "Exact id of the argument to attack (from get_entity's arguments list).",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the chunk does not support this argument (one assertion).",
                    },
                    "claim_id": {
                        "type": "string",
                        "description": "Alternative: attack the argument using this exact claim as a premise (the reliable way to challenge the argument behind a specific statement).",
                    },
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repoint_reference",
            "description": (
                "Repoint a chunk-grounded reference to a different entity, WITHOUT "
                "re-ingesting the chunk. The old argument is superseded (status "
                "'out', its conclusion 'superseded'), the new one is installed "
                "(status 'in'), a feedback argument ATTACKS the old conclusion, "
                "canonical edges are rewired through SUPERSEDES pairs, and "
                "chunk-contributed properties transfer. Use when the user says an "
                "entity, property or relationship refers to a different one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk the reference is grounded in (from search_corpus or the argument's chunk_id).",
                    },
                    "old_entity_id": {"type": "string"},
                    "new_entity_id": {"type": "string"},
                    "new_entity_label": {
                        "type": "string",
                        "description": "Label of the new entity when it must be created or relabeled; defaults to the old entity's label.",
                    },
                    "reason": {"type": "string"},
                },
                "required": ["chunk_id", "old_entity_id", "new_entity_id"],
            },
        },
    },
]
