"""Neo4j persistence for `SessionMemory`.

Append-only by construction. `record` MERGEs on a stable key so re-recording
the same lesson updates one node; `supersede` writes a *new* node and links it
`-[:SUPERSEDES]->` the old, tagging the old `:Deprecated` so it drops out of
retrieval the same way every other retired node does
(`docs/architecture/lifecycle.md`) while remaining readable as history.

Memory never expires, so nothing here deletes.
"""
from __future__ import annotations

from typing import Any

from graph import Graph
from knowledge_bases.ontology import Triplet, is_known_label, partition_triplets
from knowledge_bases.orchestration.resolution import (
    SEMANTIC_COLLECTIONS,
    resolve_entity,
)
from knowledge_bases.orchestration.memory_models import (
    MemoryScope,
    SessionMemory,
)


class SessionMemoryRepository:
    def __init__(self, graph: Graph):
        self.graph = graph

    # -- writing -------------------------------------------------------------

    def record(self, memory: SessionMemory) -> dict[str, Any]:
        """Store one memory and link it to entities that already exist.

        Returns the stored key plus any triplets the ontology refused, so the
        caller can surface them instead of the write looking wholly successful.
        """
        memory.validate_for_storage()

        self.graph.run(
            "MERGE (m:SessionMemory {key: $key}) "
            "ON CREATE SET m.created_at = datetime(), m.version = 1 "
            "SET m.version = CASE WHEN m.body <> $body THEN coalesce(m.version, 1) + 1 "
            "                ELSE coalesce(m.version, 1) END, "
            "m.title = $title, m.body = $body, m.kind = $kind, m.scope = $scope, "
            "m.repo = $repo, m.expert_key = $expert_key, m.session_id = $session_id, "
            "m.trace_id = $trace_id, m.updated_at = datetime()",
            key=memory.key, title=memory.title, body=memory.body,
            kind=memory.kind.value, scope=memory.scope.value, repo=memory.repo,
            expert_key=memory.expert_key, session_id=memory.session_id,
            trace_id=memory.trace_id,
        )

        # A memory exists as a claim: the lesson it records is an assertion
        # attributable to the session, so it is stored as a Claim the memory
        # node ASSERTS. The entity links then hang off the claim, which is what
        # makes them ontology-checked stances rather than topic pointers.
        # Written after linking so a created entity stays the first node the
        # graph double sees in tests.
        written, created, rejected, methods = self._link_entities(memory)
        self._record_claim(memory)
        # Scope is a property, not an edge: the lesson was learned against these
        # entities. It is written after linking because resolution is what
        # decides the ids, and a created entity has an id only then.
        self.graph.run(
            "MATCH (m:SessionMemory {key: $key}) SET m.applies_to = $scopes",
            key=memory.key, scopes=[
                {"label": s_["label"], "id": s_["id"], "name": s_["name"]}
                for s_ in memory.applies_to
            ],
        )
        if rejected:
            # Kept on the node so a recurring gap is visible without needing
            # the originating session's logs.
            self.graph.run(
                "MATCH (m:SessionMemory {key: $key}) SET m.unmapped_claims = $claims",
                key=memory.key,
                claims=[
                    f"{c['subject_id']} -{c['predicate']}-> {c['object_id']} ({c['reason']})"
                    for c in rejected
                ],
            )
        return {
            "key": memory.key,
            "links_written": written,
            "entities_created": created,
            "resolved_by": methods,
            "applies_to": memory.applies_to,
            "rejected": rejected,
        }

    def _record_claim(self, memory: SessionMemory) -> None:
        """Store the lesson as a Claim the memory node asserts.

        `body` is prose, and prose is what gets read back, so the claim's
        statement is the body verbatim -- not a shredding of it. `claim_type`
        is `finding`: a memory is what a session concluded, and it is
        defeasible like every other claim, so a corrected memory supersedes
        this one rather than editing it.
        """
        self.graph.run(
            "MERGE (c:Claim {id: $id}) "
            "ON CREATE SET c.created_at = datetime() "
            "SET c.statement = $statement, c.claim_type = 'finding', "
            "c.status = 'cataloged', c.source = 'session_memory', "
            "c.updated_at = datetime()",
            id=memory.key, statement=memory.body,
        )
        self.graph.run(
            "MATCH (m:SessionMemory {key: $key}) "
            "MATCH (c:Claim {id: $id}) "
            "MERGE (m)-[a:ASSERTS]->(c) ON CREATE SET a.created_at = datetime()",
            key=memory.key, id=memory.key,
        )

    def _resolve_or_create(
        self, triplet: Triplet, memory: SessionMemory,
    ) -> tuple[str | None, bool, str | None, str]:
        """Find the entity this link points at, creating it if it is genuinely new.

        Returns `(entity_id, created, rejection_reason, method)`.

        Creating a *node* is data; creating a *label* is a schema change, which
        is why the label is checked before we get here and never derived from
        the model's wording. Resolution runs id -> name -> embedding
        (`knowledge_bases/orchestration/resolution.py`) before creating, because
        the failure mode that would make memory-driven creation worse than
        useless is silently minting a second `Project` that differs only in
        wording. The embedding step is what catches paraphrase; it degrades to
        name matching when Milvus is down.
        """
        label, entity_id = triplet.object_label, triplet.object_id
        name = (triplet.object_name or "").strip()

        resolution = resolve_entity(
            self.graph, label, entity_id, name,
            semantic=memory.semantic_resolution,
        )
        if resolution.found:
            return resolution.entity_id, False, None, resolution.method

        if not memory.create_missing_entities:
            return None, False, (
                f"no existing {label} with id {entity_id!r}, and "
                "create_missing_entities is off"
            ), "none"
        if not name:
            # Without a name there is nothing to dedupe on later, so this would
            # be an unmergeable orphan rather than a useful entity.
            return None, False, (
                f"cannot create {label} {entity_id!r} without `target_name` to "
                "identify and later deduplicate it"
            ), "none"

        self.graph.upsert_node(
            label, {"id": entity_id},
            {
                "name": name,
                # Marks this as asserted by a memory rather than curated or
                # ingested, so a reader can weigh it accordingly.
                "provisional": True,
                "created_by": "session_memory",
            },
        )
        # Provenance, so "where did this entity come from?" is answerable.
        self.graph.attach_source(
            label, {"id": entity_id},
            source_id=f"session-memory::{memory.key}",
            source_props={
                "kind": "session_memory",
                "title": memory.title,
                "session_id": memory.session_id,
            },
            role="asserted_by_memory",
        )
        self._index_for_semantic_resolution(label, entity_id, name)
        return entity_id, True, None, "created"

    def _index_for_semantic_resolution(
        self, label: str, entity_id: str, name: str,
    ) -> None:
        """Embed a newly created entity so the *next* memory can find it.

        Without this, memory-created entities are invisible to embedding search
        and the second paraphrase of the same thing would create a duplicate --
        which is the exact failure this resolution layer exists to prevent.
        Best-effort: Milvus is an accelerator, so failing to index must not
        fail the write.
        """
        if label not in SEMANTIC_COLLECTIONS or not name:
            return
        collection, _ = SEMANTIC_COLLECTIONS[label]
        try:
            from llm import get_llm
            from vector_store import VectorStore

            VectorStore().upsert_vector(collection, entity_id, get_llm().embed(name))
        except Exception:
            pass

    def _link_entities(
        self, memory: SessionMemory,
    ) -> tuple[int, int, list[dict[str, str]], dict[str, int]]:
        """Link the memory to entities, creating any that do not exist yet.

        A memory may create entities but never ontology: the label must already
        be declared, so `upsert_rel` refuses anything else.

        Two tiers, and they are not the same tier. The **edge** is a stance the
        memory's claim takes toward a proposition -- SUPPORTS, CHALLENGES,
        ANSWERS, CONFLICTS_WITH. A claim is a proposition, so it can only argue
        with a proposition; a CodeRepository is not one, and no edge from a
        claim to it exists. The **scope** is which entity the lesson was
        learned against, and it is recorded as the `applies_to` property on the
        memory node, not as an edge. Naming a topic is not an epistemic
        position, which is why `ABOUT` was removed; a scope edge would be
        `ABOUT` renamed.

        Resolution and creation run for every link, so a lesson about a
        repository the graph lacks still brings that repository in, even when
        the lesson takes no stance toward it.
        """
        # A link that states no predicate is scope only: it names the entity the
        # lesson was learned against. It is not a triplet, so it is not checked
        # as one and not reported as a rejection -- it resolves and creates, and
        # lands in `applies_to`.
        scope_only = [link for link in memory.entity_links if not str(link.get("predicate", "")).strip()]
        candidates = [
            Triplet(
                # The stance hangs off the memory's claim, not the memory node:
                # the claim is the epistemic object, and the memory node is the
                # record that the session asserted it.
                subject_id=memory.key,
                subject_label="Claim",
                predicate=str(link.get("predicate", "")).upper(),
                object_id=str(link.get("target_id", "")),
                object_label=str(link.get("target_label", "")),
                object_name=str(link.get("target_name", "") or ""),
                source_text=str(link.get("source_text", "")),
            )
            for link in memory.entity_links
            if str(link.get("predicate", "")).strip()
        ]

        writable, rejected = partition_triplets(candidates)
        writable_ids = {id(t) for t in writable}

        written = created = 0
        methods: dict[str, int] = {}
        scopes: list[dict[str, str]] = []
        for triplet in candidates + [
            Triplet(
                subject_id=memory.key, subject_label="Claim", predicate="",
                object_id=str(link.get("target_id", "")),
                object_label=str(link.get("target_label", "")),
                object_name=str(link.get("target_name", "") or ""),
                source_text=str(link.get("source_text", "")),
            )
            for link in scope_only
        ]:
            # A link whose label the registry has not declared is refused before
            # any write is attempted: creating a node is data, creating a label
            # is schema, and `upsert_node` would raise rather than report.
            if not is_known_label(triplet.object_label):
                rejected.append({
                    "subject_id": triplet.subject_id,
                    "predicate": triplet.predicate,
                    "object_id": triplet.object_id,
                    "source_text": triplet.source_text,
                    "reason": f"undeclared object label {triplet.object_label!r}",
                })
                continue
            entity_id, was_created, reason, method = self._resolve_or_create(
                triplet, memory,
            )
            if entity_id:
                scopes.append({
                    "label": triplet.object_label,
                    "id": entity_id,
                    "name": (triplet.object_name or "").strip(),
                })
            if reason is not None:
                rejected.append({
                    "subject_id": triplet.subject_id,
                    "predicate": triplet.predicate,
                    "object_id": triplet.object_id,
                    "source_text": triplet.source_text,
                    "reason": reason,
                })
                continue
            created += int(was_created)
            methods[method] = methods.get(method, 0) + 1
            if id(triplet) in writable_ids:
                self.graph.upsert_rel(
                    "Claim", {"key": triplet.subject_id},
                    triplet.predicate,
                    triplet.object_label, {"id": entity_id},
                    # How the entity was matched, so a merge made by an embedding
                    # (which can be wrong) stays distinguishable from an exact id.
                    {"resolved_by": method},
                )
                written += 1
        memory.applies_to = scopes
        return written, created, rejected, methods


    def supersede(self, old_key: str, replacement: SessionMemory) -> dict[str, Any]:
        """Replace a memory without destroying it.

        Memory never expires, so a correction is a new node plus a
        `SUPERSEDES` edge; the old one is tagged `:Deprecated` so retrieval
        skips it while history stays queryable.
        """
        existing = self.graph.run(
            "MATCH (m:SessionMemory {key: $key}) RETURN m.key AS key", key=old_key,
        )
        if not existing:
            raise ValueError(f"No session memory with key {old_key!r} to supersede.")

        replacement.supersedes = old_key
        result = self.record(replacement)

        self.graph.run(
            "MATCH (old:SessionMemory {key: $old}), (new:SessionMemory {key: $new}) "
            "MERGE (new)-[sv:SUPERSEDES]->(old) "
            "ON CREATE SET sv.created_at = datetime() "
            "SET old:Deprecated, old.superseded_by = $new, "
            "old.deprecated_at = datetime()",
            old=old_key, new=replacement.key,
        )
        result["superseded"] = old_key
        return result

    # -- reading -------------------------------------------------------------

    def search(
        self,
        *,
        repo: str | None = None,
        expert_key: str | None = None,
        query: str | None = None,
        limit: int = 20,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Return applicable memories, most specific scope first.

        Scope is unioned rather than filtered to one axis: a session in a repo
        working through an expert should see repo memories, that expert's
        memories, and global ones together.
        """
        clauses = ["NOT m:Deprecated"] if not include_superseded else []
        params: dict[str, Any] = {"limit": limit}

        scope_or = ["m.scope = 'global'"]
        if repo:
            scope_or.append("(m.scope = 'repo' AND m.repo = $repo)")
            params["repo"] = repo
        if expert_key:
            scope_or.append("(m.scope = 'expert' AND m.expert_key = $expert_key)")
            params["expert_key"] = expert_key
        clauses.append("(" + " OR ".join(scope_or) + ")")

        if query:
            # Not `$query`: `Graph.run(query, **params)` takes `query` as its
            # own first argument, so a parameter of that name collides.
            clauses.append(
                "(toLower(m.title) CONTAINS toLower($text) "
                "OR toLower(m.body) CONTAINS toLower($text))"
            )
            params["text"] = query

        where = " AND ".join(clauses)
        rows = self.graph.run(
            f"MATCH (m:SessionMemory) WHERE {where} "
            "RETURN m.key AS key, m.title AS title, m.body AS body, m.kind AS kind, "
            "m.scope AS scope, m.repo AS repo, m.expert_key AS expert_key, "
            "m.version AS version, m.unmapped_claims AS unmapped_claims, "
            "m.created_at AS created_at "
            # Most specific scope wins: expert, then repo, then global.
            "ORDER BY CASE m.scope WHEN 'expert' THEN 0 WHEN 'repo' THEN 1 "
            "         WHEN 'session' THEN 2 ELSE 3 END, m.created_at DESC "
            "LIMIT $limit",
            **params,
        )
        return [{k: v for k, v in row.items() if v is not None} for row in rows]

    def get(self, key: str) -> dict[str, Any] | None:
        rows = self.graph.run(
            "MATCH (m:SessionMemory {key: $key}) "
            "OPTIONAL MATCH (m)-[r]->(e) WHERE NOT e:SessionMemory "
            "WITH m, collect({predicate: type(r), target_id: e.id, "
            "                 target_label: head(labels(e))}) AS links "
            "OPTIONAL MATCH (m)-[:SUPERSEDES]->(old:SessionMemory) "
            "RETURN m.key AS key, m.title AS title, m.body AS body, m.kind AS kind, "
            "m.scope AS scope, m.repo AS repo, m.expert_key AS expert_key, "
            "m.version AS version, m.unmapped_claims AS unmapped_claims, "
            "m.superseded_by AS superseded_by, links, "
            "collect(old.key) AS supersedes",
            key=key,
        )
        if not rows:
            return None
        row = {k: v for k, v in rows[0].items() if v is not None}
        row["links"] = [
            link for link in row.get("links", []) if link.get("target_id")
        ]
        return row
