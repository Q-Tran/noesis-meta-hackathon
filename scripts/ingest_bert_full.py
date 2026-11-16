import os
import time
from pathlib import Path
from extract import extract_claims
from graph import Graph
from vector_store import VectorStore

def main():
    p = Path(os.path.expanduser("~/arxiv/bert.txt"))
    if not p.exists():
        p = Path("/home/k1ng/arxiv/bert.txt")
    text = p.read_text()[:1200]
    doc_title = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    source_name = "arxiv"
    url = "https://arxiv.org/abs/1810.04805"
    
    print(f"Ingesting text of length {len(text)} into domain ontologies...")
    t0 = time.time()
    
    g = Graph()
    v = VectorStore()
    v.ensure_collections()
    
    scope = (
        "concepts", "methods", "models", "datasets", "benchmarks",
        "scores", "hyperparameters", "contributions", "claims",
        "documents", "people"
    )
    
    chunk_id = extract_claims(
        text=text,
        doc_title=doc_title,
        source_name=source_name,
        url=url,
        graph=g,
        vector_store=v,
        scope=scope,
    )
    duration = round(time.time() - t0, 2)
    print(f"\nIngestion completed in {duration}s! Chunk ID: {chunk_id}")
    
    # Check nodes in Neo4j
    print("\n--- Neo4j Node Counts ---")
    nodes = g.run("""
        MATCH (n)
        WHERE n:Model OR n:Method OR n:Dataset OR n:Benchmark OR n:Score OR 
              n:Contribution OR n:Concept OR n:Person OR n:DocumentNode OR 
              n:Claim OR n:Argument OR n:Chunk OR n:SourceNode
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
    """)
    for row in nodes:
        print(f"  {row['label']}: {row['count']}")
        
    # Check relationships
    print("\n--- Neo4j Relationship Types ---")
    rels = g.run("""
        MATCH (a)-[r]->(b)
        WHERE (a:Model OR a:Method OR a:Dataset OR a:Benchmark OR a:Score OR 
               a:Contribution OR a:Concept OR a:Person OR a:DocumentNode OR 
               a:Claim OR a:Argument OR a:Chunk OR a:SourceNode)
        RETURN type(r) AS rel_type, count(r) AS count
        ORDER BY count DESC
    """)
    for row in rels:
        print(f"  {row['rel_type']}: {row['count']}")
        
    g.close()

if __name__ == "__main__":
    main()
