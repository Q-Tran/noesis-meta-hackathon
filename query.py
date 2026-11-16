"""Answer functions exposed to the agent.

Graph functions traverse the decision layer; raw-row queries go through the
bounded warehouse tools (semantic_layer.run_snowflake_query).
"""
from __future__ import annotations

from graph import Graph


def search_claims(query_text: str, top_k: int = 5, graph: Graph | None = None, vector_store=None) -> list[dict]:
    """Semantic search for claims, returning macro first, then micro."""
    from llm import get_llm
    
    graph = graph or Graph()
    if vector_store is None:
        from vector_store import VectorStore
        vector_store = VectorStore()
    llm = get_llm()
    query_emb = llm.embed(query_text)
    candidate_k = llm.rerank_candidate_count(top_k)
    
    # 1. Vector Search in Milvus
    milvus_results = vector_store.search_claims(query_emb, top_k=candidate_k)
    if not milvus_results:
        return []
        
    # 2. Graph Traversal in Neo4j
    claim_ids = [r["id"] for r in milvus_results]
    
    cypher = """
    UNWIND $claim_ids AS c_id
    MATCH (c:Claim {id: c_id})
    OPTIONAL MATCH (i)-[:FOUND_IN]->(src:SourceNode)
    RETURN i.id AS id, i.statement AS statement, i.weight AS weight, src.name AS source_doc
    """
    
    graph_data = graph.run(cypher, claim_ids=claim_ids)
    graph_map = {row["id"]: row for row in graph_data}
    
    # Combine results, then let the cross-encoder rerank text against the query.
    candidates = []
    for hit in milvus_results:
        g_data = graph_map.get(hit["id"])
        if g_data:
            candidates.append({
                "id": hit["id"],
                "statement": g_data["statement"],
                "text": g_data["statement"],
                "weight": g_data["weight"],
                "sim": hit["similarity"],
                "source_doc": g_data["source_doc"]
            })

    final_results = llm.rerank(query_text, candidates, text_key="text")[:top_k]
    for result in final_results:
        result.pop("text", None)
    return final_results


def search_text(query_text: str, graph: Graph | None = None) -> list[dict]:
    """Keyword-search lexical knowledge; Claims are matched by text."""
    graph = graph or Graph()
    cypher = """
    UNWIND [
      "business_dictionary_text", "semantic_layer_text", "decision_text",
      "delivery_text", "evidence_metadata_text", "bmad_entity_text"
    ] AS index_name
    CALL db.index.fulltext.queryNodes(index_name, $q) YIELD node, score
    RETURN coalesce(node.id, node.name) AS id, labels(node) AS labels,
           properties(node) AS properties, score
    ORDER BY score DESC
    """
    try:
        return graph.run(cypher, q=query_text)
    except Exception as e:
        print(f"FTS Search failed: {e}")
        return []


def lookup_business_term(term_name: str, graph: Graph | None = None) -> list[dict]:
    """Find a business term, resolving aliases if necessary."""
    graph = graph or Graph()
    cypher = """
    MATCH (term)
    WHERE (term:BusinessTerm OR term:AlternateBusinessTerm) 
      AND toLower(term.name) = toLower($search_string)
    OPTIONAL MATCH (term)-[:ALIAS_OF]->(canonical:BusinessTerm)
    WITH coalesce(canonical, term) as true_term
    OPTIONAL MATCH (alt:AlternateBusinessTerm)-[:ALIAS_OF]->(true_term)
    RETURN true_term.name AS recommended_term,
           true_term.concept_family AS concept_family,
           true_term.definition AS definition,
           true_term.context AS context,
           collect(alt.name) AS aliases
    """
    return graph.run(cypher, search_string=term_name)


def map_business_term_to_technical(term_name: str, graph: Graph | None = None) -> list[dict]:
    """Find exactly which tables, columns, and values govern a given business term (resolves aliases)."""
    graph = graph or Graph()
    cypher = """
    MATCH (term)
    WHERE (term:BusinessTerm OR term:AlternateBusinessTerm) 
      AND toLower(term.name) = toLower($search_string)
    OPTIONAL MATCH (term)-[:ALIAS_OF]->(canonical:BusinessTerm)
    WITH coalesce(canonical, term) as true_term
    OPTIONAL MATCH (true_term)-[:MAPS_TO]->(target)
    RETURN true_term.name AS term,
           labels(target)[0] AS mapped_type,
           coalesce(target.id, target.name) AS mapped_id
    """
    return graph.run(cypher, search_string=term_name)


def compromised_decisions(graph: Graph | None = None) -> list[dict]:
    graph = graph or Graph()
    return graph.run(
        "MATCH (d:Decision {status: 'ReviewNeeded'}) "
        "RETURN d.id AS id, d.description AS description, d.review_reasons AS reasons"
    )


def metric_duplicates(graph: Graph | None = None) -> list[dict]:
    graph = graph or Graph()
    return graph.run(
        """
        MATCH (a:Metric)-[r:DUPLICATE_OF|POSSIBLE_DUPLICATE_OF]->(b:Metric)
        OPTIONAL MATCH (term:BusinessTerm)-[:MAPS_TO]->(a)
        RETURN term.name AS business_term,
               a.id AS metric_id,
               a.name AS metric_name,
               type(r) AS relationship,
               r.reason AS reason,
               r.score AS score,
               b.id AS related_metric_id,
               b.name AS related_metric_name
        ORDER BY business_term, metric_name
        """
    )


def metric_conflicts(graph: Graph | None = None) -> list[dict]:
    graph = graph or Graph()
    return graph.run(
        """
        MATCH (a:Metric)-[r:CONFLICTS_WITH]->(b:Metric)
        OPTIONAL MATCH (term:BusinessTerm)-[:MAPS_TO]->(a)
        RETURN term.name AS business_term,
               a.id AS metric_id,
               a.name AS metric_name,
               r.reason AS reason,
               b.id AS related_metric_id,
               b.name AS related_metric_name
        ORDER BY business_term, metric_name
        """
    )


def provisional_business_terms(graph: Graph | None = None) -> list[dict]:
    graph = graph or Graph()
    return graph.run(
        """
        MATCH (t:BusinessTerm)
        WHERE t.review_status = 'needs_review' OR t.created_by = 'metric_registry'
        OPTIONAL MATCH (t)-[:MAPS_TO]->(m:Metric)
        RETURN t.name AS name,
               t.definition AS definition,
               t.context AS context,
               t.created_from_metric_id AS created_from_metric_id,
               collect(m.id) AS metric_ids
        ORDER BY t.name
        """
    )


def metrics_for_business_term(term_name: str, graph: Graph | None = None) -> list[dict]:
    graph = graph or Graph()
    return graph.run(
        """
        MATCH (term)
        WHERE (term:BusinessTerm OR term:AlternateBusinessTerm)
          AND toLower(term.name) = toLower($term_name)
        OPTIONAL MATCH (term)-[:ALIAS_OF]->(canonical:BusinessTerm)
        WITH coalesce(canonical, term) AS true_term
        MATCH (true_term)-[:MAPS_TO]->(m:Metric)
        RETURN true_term.name AS business_term,
               m.id AS metric_id,
               m.name AS metric_name,
               m.definition AS definition,
               m.unit AS unit,
               m.source AS source,
               m.definition_language AS definition_language,
               m.review_status AS review_status
        ORDER BY metric_name
        """,
        term_name=term_name,
    )


def metric_audit_summary(graph: Graph | None = None) -> list[dict]:
    graph = graph or Graph()
    return graph.run(
        """
        MATCH (m:Metric)
        OPTIONAL MATCH (:BusinessTerm)-[:MAPS_TO]->(m)
        WITH count(m) AS total_metrics,
             count(CASE WHEN m.review_status = 'conflict' THEN 1 END) AS conflicts,
             count(CASE WHEN m.review_status = 'duplicate_candidate' THEN 1 END) AS duplicate_candidates,
             count(CASE WHEN m.review_status = 'needs_review' THEN 1 END) AS needs_review,
             count(CASE WHEN m.business_term IS NULL THEN 1 END) AS unmapped
        RETURN total_metrics,
               conflicts,
               duplicate_candidates,
               needs_review,
               unmapped
        """
    )


def project_blockers(graph: Graph | None = None) -> list[dict]:
    """Find blocked projects/tasks, who is impacted, and what decisions are delayed."""
    graph = graph or Graph()
    return graph.run(
        "MATCH (t:Task {status: 'blocked'}) "
        "OPTIONAL MATCH (p:Person)-[:ASSIGNED_TO]->(t) "
        "OPTIONAL MATCH (proj:Project)-[:HAS_EPIC]->(:Epic)-[:HAS_STORY]->(:UserStory)-[:HAS_TASK]->(t) "
        "OPTIONAL MATCH (proj)-[:IMPLEMENTS]->(d:Decision) "
        "RETURN t.id AS task_id, t.name AS task_name, "
        "collect(p.name) AS assignees, proj.name AS project, "
        "d.description AS delayed_decision"
    )


def decision_detail(decision_id: str, graph: Graph | None = None) -> dict:
    """Full evidence picture for one decision: ownership, evidence, goal, alternatives."""
    graph = graph or Graph()
    base = graph.run(
        "MATCH (d:Decision {id: $id}) "
        "RETURN d.id AS id, d.description AS description, d.status AS status, "
        "d.score AS score, d.review_reasons AS reasons, d:Deprecated AS deprecated",
        id=decision_id,
    )
    if not base:
        return {}
    detail = base[0]
    detail["made_by"] = graph.run(
        "MATCH (d:Decision {id: $id})-[:MADE_BY]->(p:Person) "
        "RETURN p.id AS id, p.name AS name, p.title AS title",
        id=decision_id,
    )
    detail["goals"] = graph.run(
        "MATCH (d:Decision {id: $id})-[:INTENDED_FOR]->(g:Goal) "
        "OPTIONAL MATCH (g)-[:HELD_BY]->(p:Person) "
        "RETURN g.id AS id, g.target AS target, g.created_at AS created_at, "
        "collect(p.name) AS held_by",
        id=decision_id,
    )
    detail["supports"] = graph.run(
        "MATCH (c:Claim)-[r:SUPPORTS]->(d:Decision {id: $id}) "
        "RETURN i.statement AS statement, r.weight AS weight, r.valid_if AS valid_if",
        id=decision_id,
    )
    detail["challenges"] = graph.run(
        "MATCH (c:Claim)-[r:CHALLENGES]->(d:Decision {id: $id}) "
        "RETURN i.statement AS statement, r.weight AS weight, r.kind AS kind",
        id=decision_id,
    )
    detail["supporting_metrics"] = graph.run(
        "MATCH (m:Metric)-[:SUPPORTS]->(c:Claim)-[:SUPPORTS|CHALLENGES]->(d:Decision {id: $id}) "
        "RETURN DISTINCT m.id AS id, m.name AS name",
        id=decision_id,
    )
    detail["supporting_external_info"] = graph.run(
        "MATCH (c:Claim)-[:SUPPORTS|CHALLENGES]->(:Decision {id: $id}) "
        "MATCH (i)-[:FOUND_IN {role: 'origin'}]->(s:SourceNode {type: 'external'}) "
        "RETURN DISTINCT s.id AS id, s.name AS name, s.type AS source, s.observed_at AS observed_at",
        id=decision_id,
    )
    return detail


if __name__ == "__main__":
    print("Compromised decisions:", compromised_decisions())
