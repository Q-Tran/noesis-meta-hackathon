"""LLM extraction against the composed product profile.

The schema is composed, not hardcoded. `extract.py` used to keep ten Pydantic
classes in the engine, so the extraction vocabulary was a constant: a product that
installed no `project_management` still had `Epic` offered to the model, a product
that installed `semantic_layer` got no shape for `Metric` because nobody wrote a
class for it in the engine, and the identity rules (how a Person merges, how a
Column is keyed) lived far from the packages that own those labels.

Now every shape is declared by the package that owns its label
(`extraction_shapes_seed`), and this module is a driver: it builds the schema from
`registry.extraction_shapes()`, asks each shape's package to turn the extracted
item into graph properties, asks that package for the graph id, and writes.
Installing a package extends extraction; uninstalling one narrows it. Nothing
here knows what a `BusinessTerm` is.

Two write policies are preserved, and both are the point of this path:
- A label the installed profile does not declare is skipped and reported, never
  written. An LLM that reaches for a type the product lacks must not add ontology
  at runtime, and must not abort a document that was mostly fine.
- Relationships go through `partition_triplets`, so an edge between the wrong
  kinds of thing is rejected by the endpoint typing the ontology describes.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional, List
import re

from pydantic import BaseModel, Field, create_model

from graph import Graph
from vector_store import VectorStore
from llm import get_llm
from knowledge_bases.ontology import is_known_label
from knowledge_bases.triplets import filter_properties
from knowledge_bases.provenance.models import SourceNode, SourceReliability


# The relationship vocabulary the extractor is allowed to use, named for the
# LLM explicitly instead of being hinted at in prose. The previous free-text
# hint suggested `WORKS_FOR`, which no knowledge base declares, and
# `graph.upsert_rel` would have written it -- adding ontology at runtime. The
# list is derived from the registry, so it cannot drift from what is declared.
def _extractable_relationships() -> list[str]:
    """The relationship vocabulary an extractor reading prose may use.

    Derived from each package's `RelationshipType.extractable` flag rather than
    from a list kept here. This used to be `allowed_relationships()` minus a
    hardcoded set of 24 names, which had two problems: a package adding a
    versioning or telemetry edge would have it silently offered to the LLM until
    someone remembered to update the constant, and the exclusion reason lived far
    away from the declaration it applied to.
    """
    from knowledge_bases.registry import relationship_types

    return sorted(
        name for name, rel in relationship_types().items() if rel.extractable
    )


def _typed_relationship_lines(scope: tuple[str, ...] | None = None) -> str:
    """The extractable vocabulary rendered WITH its endpoint typing.

    A flat name list is what let the engine write edges its own ontology forbids:
    `SUPPORTS` declares `object=('Decision', 'Claim', ...)`, and a model given
    only the name had nothing telling it that
    `(:Claim)-[:SUPPORTS]->(:Project)` was illegal. Each line states the legal
    ends, so the model can only name a pair it has been told is legal. `ANY`
    means the declaration leaves that end open.
    """
    from knowledge_bases.registry import relationship_types

    rels = relationship_types()
    # Narrow to the labels the scope's shapes produce. A paper scope declares
    # claims, documents, and arguments, so the vocabulary is the pairs those
    # labels can form -- not all 61 pairs the full profile allows. Fewer lines
    # means a smaller schema, and schema size is what pushed calls past the
    # point where the local server stops emitting.
    labels = None
    if scope:
        from knowledge_bases.registry import extraction_shapes

        labels = {s.label for s in extraction_shapes() if s.field in scope}
    lines = []
    for name in _extractable_relationships():
        rel = rels[name]
        if labels and rel.subject and not (set(rel.subject) & labels):
            continue
        if labels and rel.object and not (set(rel.object) & labels):
            continue
        subj = "ANY" if not rel.subject else " | ".join(rel.subject)
        obj = "ANY" if not rel.object else " | ".join(rel.object)
        lines.append(f"  ({subj}) -[:{name}]-> ({obj})")
    return "\n".join(lines)


class ExRelationship(BaseModel):
    source_ref_id: str
    target_ref_id: str
    relationship_type: str = Field(
        ...,
        description=(
            "UPPER_SNAKE_CASE. MUST be one of the typed pairs below, where the "
            "relationship may only join the labels listed for its ends. Do not "
            "invent a new relationship type; choose the closest one, or omit the "
            "relationship.\n" + _typed_relationship_lines()
        ),
    )


def _extraction_model(scope: tuple[str, ...] | None = None) -> type[BaseModel]:
    """The extraction schema, composed from the installed product profile.

    Built from `registry.extraction_shapes()`, so a label the profile does not
    declare never reaches the prompt and the model cannot extract it. The
    relationship field is the engine's own: relationships are the graph, not a
    package's noun, so its vocabulary comes from `_extractable_relationships()`.

    `scope` narrows the schema to the shape fields a given source can produce.
    A scientific paper has no epics, tasks, tables, or columns; asking for them
    makes the model shove "Transformer" into `Project` and "BLEU" into `Column`.
    Passing `("claims",)` for a document keeps the extraction honest. `None`
    keeps the full profile, which is right for a delivery tracker.
    """
    from knowledge_bases.registry import extraction_shapes, resolve_seed

    fields = {}
    for shape in extraction_shapes():
        if scope is not None and shape.field not in scope:
            continue
        model = resolve_seed(shape.model)
        fields[shape.field] = (list[model], Field(default_factory=list))
    # The relationship field is rebuilt per scope so its vocabulary carries only
    # the pairs the scoped labels can form. `ExRelationship` is the unscoped
    # default; a scoped call gets a copy whose `relationship_type` description
    # lists the narrowed typed pairs.
    rel_field = ExRelationship
    if scope is not None:
        rel_field = create_model(
            "ScopedRelationship",
            source_ref_id=(str, ...),
            target_ref_id=(str, ...),
            relationship_type=(str, Field(
                ...,
                description=(
                    "UPPER_SNAKE_CASE. MUST be one of the typed pairs below, where "
                    "the relationship may only join the labels listed for its ends. "
                    "Do not invent a new relationship type; choose the closest one, "
                    "or omit the relationship.\n" + _typed_relationship_lines(scope)
                ),
            )),
        )
    fields["relationships"] = (list[rel_field], Field(default_factory=list))
    return create_model("InstalledExtraction", **fields)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _slugify(text: str) -> str:
    from knowledge_bases.base import slugify
    return slugify(text)

def _id_from_text(prefix: str, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}::{h}"

def _resolve_identity_candidates(
    graph: Graph,
    vector_store: VectorStore,
    llm,
    label: str,
    collection: str,
    text: str,
    id_prefix: str,
    text_field_name: str,
    top_k: int | None = None,
) -> tuple[str, bool, list[float], list[dict]]:
    """Resolve fuzzy identity returning (resolved_id, is_existing, embedding, candidates).

    `candidates` contains the winner (status='in') and top-k runner-ups (status='out'),
    including the null hypothesis ('new entity') if an existing entity won, or top existing
    candidates if the null hypothesis won.
    """
    from config import settings

    limit_k = top_k or getattr(settings, "resolution_top_k", 3)
    embedding = llm.embed(text)
    candidate_k = llm.rerank_candidate_count(1, dedupe=True)
    nearest = vector_store.search_vectors(collection, embedding, top_k=candidate_k)

    candidates = []
    for hit in nearest:
        if not hit.get("similarity"):
            continue
        match_id = hit["id"]
        node_res = graph.run(f"MATCH (n:`{label}` {{id: $id}}) RETURN n.`{text_field_name}` AS text", id=match_id)
        if node_res:
            candidate = dict(hit)
            candidate["text"] = node_res[0]["text"]
            candidates.append(candidate)

    reranked = llm.rerank(text, candidates, text_key="text") if candidates else []

    winner_match = None
    winner_score = 0.0

    for candidate in reranked:
        sim = candidate["similarity"]
        match_id = candidate["id"]
        match_text = candidate["text"]

        if sim >= 0.94:
            winner_match = candidate
            winner_score = sim
            break
        elif sim >= 0.85:
            arbiter_prompt = (
                f"Entity A: '{match_text}'. Entity B: '{text}'. "
                "Do these refer to the exact same entity? Answer ONLY 'Yes' or 'No'."
            )
            answer = llm.complete(arbiter_prompt).strip().lower()
            if "yes" in answer:
                winner_match = candidate
                winner_score = sim
                break

    new_id = _id_from_text(id_prefix, text)

    if winner_match:
        # Existing entity won
        winner_id = winner_match["id"]
        winner_cand = {
            "id": winner_id,
            "label": label,
            "text": winner_match.get("text", text),
            "score": winner_score,
            "is_new": False,
            "status": "in",
        }
        # Runner-ups: other candidates from reranked + null candidate
        runner_ups = []
        for c in reranked:
            if c["id"] != winner_id and len(runner_ups) < (limit_k - 1):
                runner_ups.append({
                    "id": c["id"],
                    "label": label,
                    "text": c.get("text", ""),
                    "score": float(c.get("similarity", 0.0)),
                    "is_new": False,
                    "status": "out",
                })
        # Add null candidate as runner-up
        runner_ups.append({
            "id": new_id,
            "label": label,
            "text": text,
            "score": 0.5,
            "is_new": True,
            "status": "out",
        })
        return winner_id, True, embedding, [winner_cand] + runner_ups[:limit_k]
    else:
        # Null hypothesis won (new entity)
        winner_cand = {
            "id": new_id,
            "label": label,
            "text": text,
            "score": 1.0,
            "is_new": True,
            "status": "in",
        }
        # Runner-ups: top existing candidates from reranked
        runner_ups = []
        for c in reranked[:limit_k]:
            runner_ups.append({
                "id": c["id"],
                "label": label,
                "text": c.get("text", ""),
                "score": float(c.get("similarity", 0.0)),
                "is_new": False,
                "status": "out",
            })
        return new_id, False, embedding, [winner_cand] + runner_ups


def _resolve_fuzzy_identity(
    graph: Graph,
    vector_store: VectorStore,
    llm,
    label: str,
    collection: str,
    text: str,
    id_prefix: str,
    text_field_name: str,
    top_k: int | None = None,
    return_candidates: bool = False,
) -> tuple[str, bool, list[float]] | tuple[str, bool, list[float], list[dict]]:
    """Returns (resolved_id, is_existing, embedding) by default, or with candidates if return_candidates=True."""
    res_id, is_existing, emb, candidates = _resolve_identity_candidates(
        graph, vector_store, llm, label, collection, text, id_prefix, text_field_name, top_k=top_k
    )
    if return_candidates:
        return res_id, is_existing, emb, candidates
    return res_id, is_existing, emb


def extract_claims(
    text: str,
    doc_title: str,
    source_name: str,
    url: str,
    owners: dict[str, str] | None = None,
    source_type: str = "Document",
    graph: Graph | None = None,
    vector_store: VectorStore | None = None,
    scope: tuple[str, ...] | None = None,
    chunk_id: str | None = None,
    chunk_index: int | None = None,
    document_id: str | None = None,
) -> str:
    """Extract full knowledge graph (entities and relationships) from unstructured text.

    `scope` restricts which shape fields the model is asked to fill, by field name
    (`"claims"`, `"tasks"`, ...). `None` asks for every shape the profile declares.
    Use a narrow scope for a source whose subject matter has no home in most of the
    profile: a paper yields claims, not epics.
    """
    from knowledge_bases.registry import extraction_shapes, resolve_seed

    graph = graph or Graph()
    vector_store = vector_store or VectorStore()
    llm = get_llm()
    shapes = [s for s in extraction_shapes() if scope is None or s.field in scope]

    # Ensure source node
    source_id = _id_from_text("source", source_name)
    graph.upsert_node("SourceNode", {"id": source_id}, {
        "name": source_name,
        "type": source_type,
        "reliability": SourceReliability.medium.value,
    })

    # Provenance: vectorize chunk and insert into DiskANN collection in Milvus
    if not chunk_id:
        chunk_id = _id_from_text("chunk", f"{source_id}::{text}")
    doc_id = document_id or _id_from_text("doc", doc_title)

    try:
        chunk_emb = llm.embed(text)
        chunk_meta = {
            "text": text,
            "source_id": source_id,
            "document_id": doc_id,
        }
        if chunk_index is not None:
            chunk_meta["index"] = chunk_index
        vector_store.upsert_chunk(chunk_id, chunk_emb, metadata=chunk_meta)
    except Exception as e:
        print(f"[extract] Failed to vectorize chunk {chunk_id}: {e}")

    # Upsert Chunk node in graph and link to SourceNode
    chunk_props = {
        "text": text,
        "source_id": source_id,
        "document_id": doc_id,
        "index": chunk_index,
        "created_at": _now(),
    }
    clean_chunk_props = {k: v for k, v in chunk_props.items() if v is not None}
    graph.upsert_node("Chunk", {"id": chunk_id}, clean_chunk_props)
    graph.upsert_rel("Chunk", {"id": chunk_id}, "FOUND_IN", "SourceNode", {"id": source_id},
                     {"role": "origin"})

    # Assign ownership of SourceNode. The id rule belongs to the package that
    # owns `Person`, so the engine asks it: a product with no Person shape
    # records the source and skips the ownership edge, rather than inventing a
    # `person_email::` key no package in this product merges on.
    person_shape = next((s for s in shapes if s.label == "Person"), None)
    if person_shape and owners:
        # Only the identifiers the installed profile declares are written. The
        # keys arrive as a dict keyed by the names the contributing connector
        # declared, so the engine never names a Microsoft Graph concept; a
        # product with no Graph connector receives no `teams_id` and writes none.
        owner_props = filter_properties("Person", owners)
        if owner_props:
            person_id = resolve_seed(person_shape.identity)
            owner_node_id = person_id(owner_props)
            graph.upsert_node("Person", {"id": owner_node_id}, owner_props)
            graph.upsert_rel("Person", {"id": owner_node_id}, "OWNS", "SourceNode", {"id": source_id})

    # THE PROMPT IS DELIBERATELY SHORT, AND THAT IS A MEASURED CONSTRAINT, NOT A
    # STYLE CHOICE. On the local Strata server, the same 400-char text extracted in
    # 62.3 s with 16 claims, 1 document, and 41 relationships. Adding
    # ONE instruction line -- "Extract all distinct entities and assign each a
    # unique ref_id" -- took the same call to 4585.9 s and returned nothing. The
    # instruction block made `swift-1.5` reason endlessly without emitting; the
    # schema already declares ref_id, the fields, and the relationship ends, so the
    # numbered instructions asked for what the schema states and cost 4500 seconds.
    # Keep this prompt at the schema plus the text. Do not add instructions.
    prompt = (
        f"Document Title: {doc_title}\n\n"
        f"Text:\n{text}\n\n"
        "Extract entities and relationships from the text, using only the schema."
    )

    try:
        result = llm.extract(prompt, _extraction_model(scope))
    except Exception as e:
        print(f"[extract] Failed to extract knowledge: {e}")
        raise e

    # Map to hold resolved Graph IDs and Labels for relationships
    resolved = {} # ref_id -> {"id": graph_id, "label": Label, "argument_id": str | None}

    # Helper to track mutations
    def upsert_entity(label, ref_id, graph_id, props, is_existing=False, win_arg=None):
        if not is_known_label(label):
            # Same policy as relationship partitioning: skip and report, never
            # abort the ingest and never write undeclared ontology.
            print(f"[extract] skipped entity {label!r} ({ref_id}): label not in installed ontology")
            return
        resolved[ref_id] = {"id": graph_id, "label": label, "argument_id": win_arg}

        # Build set properties dynamically (ignore Nones)
        clean_props = {k: v for k, v in props.items() if v is not None}

        # Track property attribution ledger in _property_origins
        try:
            node_res = graph.run(f"MATCH (n:`{label}` {{id: $id}}) RETURN n._property_origins AS origins", id=graph_id)
            origins = {}
            if node_res and node_res[0].get("origins"):
                try:
                    origins = json.loads(node_res[0]["origins"])
                except Exception:
                    origins = {}
            for k in clean_props.keys():
                if k not in ("id", "weight", "_property_origins"):
                    origins[k] = {
                        "argument_id": win_arg,
                        "chunk_id": chunk_id,
                        "status": "active",
                    }
            clean_props["_property_origins"] = json.dumps(origins)
        except Exception:
            pass

        # We always call upsert_node because it's now an atomic ON CREATE / ON MATCH MERGE
        # However, for fuzzy matches, we only want to update the properties explicitly provided by the LLM
        # graph.upsert_node handles this by SETting only the passed properties
        graph.upsert_node(label, {"id": graph_id}, clean_props)

        # Draw edge to Source
        graph.upsert_rel(label, {"id": graph_id}, "FOUND_IN", "SourceNode", {"id": source_id},
                         {"role": "extracted"})
        # Draw edges to Chunk (searchable provenance layer)
        graph.upsert_rel(label, {"id": graph_id}, "EXTRACTED_FROM", "Chunk", {"id": chunk_id},
                         {"role": "extracted"})
        graph.upsert_rel("Chunk", {"id": chunk_id}, "MENTIONS", label, {"id": graph_id})

    # One loop for every shape. A shape with `fuzzy_text` merges by embedding
    # similarity in the vector collection named after its field; a shape with an
    # `identity` function merges on the key its owning package declares.
    for shape in shapes:
        items = getattr(result, shape.field, [])
        if not items:
            continue
        props_of = resolve_seed(shape.props)
        for item in items:
            props = props_of(item, url)
            if shape.fuzzy_text:
                surface_text = props[shape.fuzzy_text]
                gid, is_existing, emb, candidates = _resolve_fuzzy_identity(
                    graph, vector_store, llm,
                    shape.label, shape.field,
                    surface_text, shape.label.lower(), shape.fuzzy_text,
                    return_candidates=True,
                )
                win_arg = graph.record_resolution_arguments(
                    chunk_id=chunk_id,
                    entity_label=shape.label,
                    surface_text=surface_text,
                    candidates=candidates,
                    ref_id=item.ref_id,
                )
                upsert_entity(shape.label, item.ref_id, gid, props, is_existing, win_arg=win_arg)
                if not is_existing:
                    vector_store.upsert_vector(shape.field, gid, emb)
            else:
                gid = resolve_seed(shape.identity)(props)
                surface_text = getattr(item, "name", None) or getattr(item, "title", None) or gid
                winner_candidate = [{
                    "id": gid,
                    "label": shape.label,
                    "text": str(surface_text),
                    "score": 1.0,
                    "is_new": True,
                    "status": "in",
                }]
                win_arg = graph.record_resolution_arguments(
                    chunk_id=chunk_id,
                    entity_label=shape.label,
                    surface_text=str(surface_text),
                    candidates=winner_candidate,
                    ref_id=item.ref_id,
                )
                upsert_entity(shape.label, item.ref_id, gid, props, win_arg=win_arg)

    # THE DOCUMENT IS WHAT IS ON RECORD AS STATING THINGS.
    # Every extracted Claim is asserted by the DocumentNode the text
    # came from. Authorship is not a property of the statement: it travels over
    # ASSERTS, so a paper's stated claim, a person's stated claim, and the
    # engine's own reading are all the same kind of node, and the graph can
    # always answer who said it. The engine reads the document, so the document
    # is the subject; a Person is not a legal ASSERTS subject and never was.
    doc_id = _id_from_text("doc", doc_title)
    graph.upsert_node("DocumentNode", {"id": doc_id}, filter_properties("DocumentNode", {
        "title": doc_title,
        "name": doc_title,
        "url": url,
    }))
    graph.upsert_rel("DocumentNode", {"id": doc_id}, "FOUND_IN", "SourceNode",
                     {"id": source_id}, {"role": "origin"})
    for ref, ent in resolved.items():
        if ent["label"] == "Claim":
            graph.upsert_rel("DocumentNode", {"id": doc_id}, "ASSERTS",
                             ent["label"], {"id": ent["id"]})

    # DRAW RELATIONSHIPS
    from knowledge_bases.ontology import Triplet, partition_triplets

    rel_candidates = []
    for rel in getattr(result, "relationships", []):
        if rel.source_ref_id in resolved and rel.target_ref_id in resolved:
            src = resolved[rel.source_ref_id]
            tgt = resolved[rel.target_ref_id]
            triplet = Triplet(
                subject_id=src["id"], subject_label=src["label"],
                predicate=rel.relationship_type.upper(),
                object_id=tgt["id"], object_label=tgt["label"],
                source_text=f"{rel.source_ref_id} -> {rel.target_ref_id}",
            )
            rel_candidates.append((rel, triplet))

    # An LLM that reaches for a relationship the ontology lacks must not abort
    # the whole ingest, and must not silently write new ontology either. Skip
    # and report: a recurring rejection is a signal the ontology needs a
    # reviewed change (see `docs/architecture/ontology_audit.md`).
    writable, rejected = partition_triplets([t for _, t in rel_candidates])
    writable_triplets = {id(t) for t in writable}
    for rel, triplet in rel_candidates:
        if id(triplet) not in writable_triplets:
            continue
        rel_id = _id_from_text("rel", f"{triplet.subject_id}::{triplet.predicate}::{triplet.object_id}::{chunk_id}")
        src = resolved.get(rel.source_ref_id, {})
        tgt = resolved.get(rel.target_ref_id, {})
        src_arg = src.get("argument_id")
        tgt_arg = tgt.get("argument_id")

        rel_arg_id = None
        if is_known_label("Argument") and is_known_label("Claim"):
            rel_arg_id = _id_from_text("arg", f"rel::{rel_id}")
            rel_claim_id = _id_from_text("claim", f"rel::{rel_id}")
            # Record Claim for relationship
            graph.upsert_node("Claim", {"id": rel_claim_id}, {
                "statement": f"{triplet.subject_id} {triplet.predicate} {triplet.object_id}",
                "claim_type": "derivation",
                "status": "active",
                "confidence": 1.0,
                "created_at": _now(),
            })
            graph.upsert_rel("Claim", {"id": rel_claim_id}, "FOUND_IN", "Chunk", {"id": chunk_id}, {"role": "origin"})
            graph.upsert_node("Argument", {"id": rel_arg_id}, {
                "status": "in",
                "computed_at": _now(),
                "chunk_id": chunk_id,
            })
            graph.upsert_rel("Argument", {"id": rel_arg_id}, "CONCLUDES", "Claim", {"id": rel_claim_id})

        rel_props = {
            "id": rel_id,
            "status": "active",
            "chunk_id": chunk_id,
            "created_at": _now(),
        }
        if rel_arg_id:
            rel_props["argument_id"] = rel_arg_id
        if src_arg:
            rel_props["subject_arg_id"] = src_arg
        if tgt_arg:
            rel_props["object_arg_id"] = tgt_arg

        graph.upsert_rel(
            triplet.subject_label, {"id": triplet.subject_id},
            triplet.predicate,
            triplet.object_label, {"id": triplet.object_id},
            props=rel_props,
        )

    print(f"Extracted entities and relationships from '{doc_title}'.")
    if rejected:
        print(
            f"[extract] skipped {len(rejected)} relationship(s) outside the "
            f"declared ontology: "
            + ", ".join(sorted({r["reason"] for r in rejected}))
        )
    return chunk_id


def extract_chunks(
    chunks: list[dict | str],
    doc_title: str,
    source_name: str,
    url: str,
    owners: dict[str, str] | None = None,
    source_type: str = "Document",
    graph: Graph | None = None,
    vector_store: VectorStore | None = None,
    scope: tuple[str, ...] | None = None,
    **kwargs,
) -> list[str]:
    """Process multiple text chunks into the ontology with vectorized provenance."""
    chunk_ids = []
    doc_id = _id_from_text("doc", doc_title)
    for idx, item in enumerate(chunks):
        if isinstance(item, str):
            c_text = item
            c_id = None
            c_index = idx
        elif isinstance(item, dict):
            c_text = item.get("text", "")
            c_id = item.get("id")
            c_index = item.get("index", idx)
        else:
            continue

        cid = extract_claims(
            text=c_text,
            doc_title=doc_title,
            source_name=source_name,
            url=url,
            owners=owners,
            source_type=source_type,
            graph=graph,
            vector_store=vector_store,
            scope=scope,
            chunk_id=c_id,
            chunk_index=c_index,
            document_id=doc_id,
            **kwargs,
        )
        chunk_ids.append(cid)
    return chunk_ids

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Extract knowledge from text")
    parser.add_argument("--text", type=str, help="Text to process")
    parser.add_argument("--title", type=str, help="Title of document")
    parser.add_argument("--source", type=str, help="Source of document")
    parser.add_argument("--url", type=str, help="URL or file path of the document")

    args = parser.parse_args()

    if args.text and args.title and args.source and args.url:
        extract_claims(args.text, args.title, args.source, args.url)
    else:
        sample_text = "Alice noted that Project Phoenix is delayed due to a strike. Bob decided to reroute via South to avoid it. Task: John will update routing tables for Phoenix."
        extract_claims(sample_text, "Q3 Logistics Meeting", "Internal Meetings", "file://sample.txt")
